from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import positions as positions_module
from .nav import compute_daily_nav
from .positions import build_corporate_action_cash_flows, build_positions
from .reconciliation import run_reconciliation
from .report import export_output_pack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "nav_recon.db"
DDL_PATH = PROJECT_ROOT / "sql" / "ddl.sql"
POLICY_PATH = PROJECT_ROOT / "policy.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOGGER = logging.getLogger(__name__)


def _load_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_csv(path: Path, parse_dates: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


def _read_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "trade_blotter": _read_csv(data_dir / "trade_blotter.csv", ["date"]),
        "prices": _read_csv(data_dir / "prices.csv", ["date"]),
        "security_master": pd.read_csv(data_dir / "security_master.csv"),
        "custodian_nav": _read_csv(data_dir / "custodian_nav.csv", ["date"]),
        "fx_rates": _read_csv(data_dir / "fx_rates.csv", ["date"]),
        "corporate_actions": _read_csv(data_dir / "corporate_actions.csv", ["ex_date"]),
    }


def _ensure_schema(connection: sqlite3.Connection, ddl_path: Path) -> None:
    ddl_sql = ddl_path.read_text(encoding="utf-8")
    connection.executescript(ddl_sql)
    connection.commit()


def _replace_existing_rows(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return

    unique_keys = frame[["date", "fund_id"]].drop_duplicates()
    for row in unique_keys.itertuples(index=False):
        connection.execute(
            f"DELETE FROM {table_name} WHERE date = ? AND fund_id = ?",
            (row.date, row.fund_id),
        )


def _insert_run_log(connection: sqlite3.Connection, run_log: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO nav_run_log (
            run_ts,
            asof_date,
            fund_id,
            total_nav,
            nav_per_share,
            break_count,
            high_severity_breaks,
            data_dir,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_log["run_ts"],
            run_log["asof_date"],
            run_log["fund_id"],
            run_log["total_nav"],
            run_log["nav_per_share"],
            run_log["break_count"],
            run_log["high_severity_breaks"],
            run_log["data_dir"],
            run_log["status"],
        ),
    )


def _write_to_sqlite(
    connection: sqlite3.Connection,
    positions: pd.DataFrame,
    nav: pd.DataFrame,
    reconciliation: pd.DataFrame,
    run_logs: list[dict[str, Any]],
) -> None:
    with connection:
        # NAV keys define the complete rerun scope even when positions or breaks
        # are now empty; otherwise resolved exceptions survive in the database.
        _replace_existing_rows(connection, "nav_daily_positions", nav)
        _replace_existing_rows(connection, "nav_daily_nav", nav)
        _replace_existing_rows(connection, "nav_reconciliation", nav)
        _replace_existing_rows(connection, "nav_breaks", nav)

        if not positions.empty:
            positions.to_sql("nav_daily_positions", connection, if_exists="append", index=False)
        if not nav.empty:
            nav.to_sql("nav_daily_nav", connection, if_exists="append", index=False)
        if not reconciliation.empty:
            reconciliation.to_sql("nav_reconciliation", connection, if_exists="append", index=False)
            reconciliation.to_sql("nav_breaks", connection, if_exists="append", index=False)

        for run_log in run_logs:
            _insert_run_log(connection, run_log)


def _log_failed_run(connection: sqlite3.Connection, run_log: dict[str, Any]) -> None:
    with connection:
        _insert_run_log(connection, run_log)


def _data_source_label(data_dir: Path) -> str:
    return "real" if "real" in str(data_dir).lower() else "dummy"


def _augment_blotter_for_corporate_actions(
    blotter: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    fund_id: str,
    asof_date: pd.Timestamp,
    corporate_actions_enabled: bool,
) -> pd.DataFrame:
    if not corporate_actions_enabled:
        return blotter.copy()

    synthetic_flows = build_corporate_action_cash_flows(
        blotter=blotter,
        corporate_actions=corporate_actions,
        fund_id=fund_id,
        asof_date=asof_date,
    )
    if synthetic_flows.empty:
        return blotter.copy()

    return (
        pd.concat([blotter, synthetic_flows], ignore_index=True)
        .sort_values(["date", "txn_type", "security_id"], na_position="last")
        .reset_index(drop=True)
    )


def _build_run_log(
    *,
    run_ts: str,
    asof_date: str,
    fund_id: str,
    total_nav: float | None,
    nav_per_share: float | None,
    break_count: int,
    high_severity_breaks: int,
    data_dir: Path,
    status: str,
) -> dict[str, Any]:
    return {
        "run_ts": run_ts,
        "asof_date": asof_date,
        "fund_id": fund_id,
        "total_nav": total_nav,
        "nav_per_share": nav_per_share,
        "break_count": break_count,
        "high_severity_breaks": high_severity_breaks,
        "data_dir": str(data_dir),
        "status": status,
    }


def _selected_fund_ids(
    blotter: pd.DataFrame,
    explicit_fund_id: str | None,
    policy: dict[str, Any],
) -> list[str]:
    available_funds = sorted(blotter["fund_id"].dropna().astype(str).unique().tolist())
    if explicit_fund_id:
        if explicit_fund_id not in available_funds:
            raise ValueError(f"No blotter rows found for fund_id {explicit_fund_id}.")
        return [explicit_fund_id]
    if available_funds:
        return available_funds
    return [str(policy.get("fund_id_default", "FUND1"))]


def _run_single_fund(
    *,
    fund_id: str,
    asof: pd.Timestamp,
    asof_str: str,
    inputs: dict[str, pd.DataFrame],
    policy: dict[str, Any],
    tolerance_bps: float,
    corporate_actions_enabled: bool,
    run_ts: str,
    data_dir: Path,
    data_source: str,
    sofr_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    blotter = inputs["trade_blotter"].loc[inputs["trade_blotter"]["fund_id"].eq(fund_id)].copy()
    if blotter.empty:
        raise ValueError(f"No trade blotter rows found for fund_id {fund_id}.")

    blotter = _augment_blotter_for_corporate_actions(
        blotter=blotter,
        corporate_actions=inputs["corporate_actions"] if corporate_actions_enabled else inputs["corporate_actions"].iloc[:0],
        fund_id=fund_id,
        asof_date=asof,
        corporate_actions_enabled=corporate_actions_enabled,
    )

    positions = build_positions(
        blotter=blotter,
        prices=inputs["prices"],
        fx_rates=inputs["fx_rates"],
        security_master=inputs["security_master"],
        corporate_actions=inputs["corporate_actions"] if corporate_actions_enabled else inputs["corporate_actions"].iloc[:0],
        base_currency=str(policy.get("base_currency", "USD")),
        asof_date=asof,
    )
    positions = positions.loc[positions["fund_id"].eq(fund_id)].reset_index(drop=True)

    nav = compute_daily_nav(
        positions=positions,
        blotter=blotter,
        policy=policy,
        asof_date=asof,
    )
    nav = nav.loc[nav["fund_id"].eq(fund_id)].reset_index(drop=True)

    custodian_slice = inputs["custodian_nav"].loc[
        inputs["custodian_nav"]["date"].eq(asof_str) & inputs["custodian_nav"]["fund_id"].eq(fund_id)
    ].reset_index(drop=True)

    reconciliation = run_reconciliation(
        internal_nav=nav,
        custodian_nav=custodian_slice,
        positions=positions,
        tolerance_bps=tolerance_bps,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = export_output_pack(
        positions=positions,
        nav=nav,
        reconciliation=reconciliation,
        output_dir=OUTPUT_DIR,
        asof_date=asof,
        fund_id=fund_id,
    )

    nav_row = nav.iloc[0].to_dict() if not nav.empty else {}
    break_count = int(len(reconciliation))
    high_breaks = int((reconciliation["severity"] == "HIGH").sum()) if not reconciliation.empty else 0
    status = "REVIEW_REQUIRED" if break_count else "SUCCESS"

    run_log = _build_run_log(
        run_ts=run_ts,
        asof_date=asof_str,
        fund_id=fund_id,
        total_nav=float(nav_row.get("total_nav", 0.0)),
        nav_per_share=float(nav_row.get("nav_per_share", 0.0)),
        break_count=break_count,
        high_severity_breaks=high_breaks,
        data_dir=data_dir,
        status=status,
    )
    result = {
        "run_ts": run_ts,
        "asof_date": asof_str,
        "fund_id": fund_id,
        "total_nav": round(float(nav_row.get("total_nav", 0.0)), 6),
        "nav_per_share": round(float(nav_row.get("nav_per_share", 0.0)), 6),
        "break_count": break_count,
        "high_severity_breaks": high_breaks,
        "status": status,
        "exports_path": str(export_path),
        "data_source": data_source,
        "sofr_rate": sofr_rate,
    }
    return positions, nav, reconciliation, run_log, result


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()
    return pd.concat(non_empty_frames, ignore_index=True)


def run_daily(asof_date: pd.Timestamp, data_dir: Path, fund_id: str | None = None) -> dict[str, Any]:
    policy = _load_policy(POLICY_PATH)
    asof = pd.Timestamp(asof_date).normalize()
    asof_str = asof.strftime("%Y-%m-%d")
    data_source = _data_source_label(data_dir)
    sofr_rate = float(policy.get("cash_interest_rate", 0.0))
    run_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    positions_module.MAX_PRICE_STALENESS_DAYS = int(policy.get("max_price_staleness_days", 3))
    positions_module.PRICE_SOURCE_PRIORITY = tuple(
        policy.get("price_source_priority", ["internal", "vendor", "fallback"])
    )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    current_fund_id = fund_id or "ALL_FUNDS"
    try:
        _ensure_schema(connection, DDL_PATH)
        try:
            LOGGER.info("Running NAV-Recon for %s (%s data)", asof_str, data_source)
            inputs = _read_inputs(data_dir)
            inputs["custodian_nav"]["date"] = inputs["custodian_nav"]["date"].dt.strftime("%Y-%m-%d")
            fund_ids = _selected_fund_ids(inputs["trade_blotter"], fund_id, policy)

            positions_frames: list[pd.DataFrame] = []
            nav_frames: list[pd.DataFrame] = []
            reconciliation_frames: list[pd.DataFrame] = []
            run_logs: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []

            for current_fund_id in fund_ids:
                LOGGER.info("Processing fund %s", current_fund_id)
                positions, nav, reconciliation, run_log, result = _run_single_fund(
                    fund_id=current_fund_id,
                    asof=asof,
                    asof_str=asof_str,
                    inputs=inputs,
                    policy=policy,
                    tolerance_bps=float(policy.get("nav_tolerance_bps", 10.0)),
                    corporate_actions_enabled=bool(policy.get("corporate_actions_enabled", True)),
                    run_ts=run_ts,
                    data_dir=data_dir,
                    data_source=data_source,
                    sofr_rate=sofr_rate,
                )
                positions_frames.append(positions)
                nav_frames.append(nav)
                reconciliation_frames.append(reconciliation)
                run_logs.append(run_log)
                results.append(result)

            combined_positions = _combine_frames(positions_frames)
            combined_nav = _combine_frames(nav_frames)
            combined_reconciliation = _combine_frames(reconciliation_frames)
            _write_to_sqlite(connection, combined_positions, combined_nav, combined_reconciliation, run_logs)

            if len(results) == 1:
                LOGGER.info(
                    "Completed NAV-Recon run for %s with %s breaks",
                    asof_str,
                    results[0]["break_count"],
                )
                return results[0]

            summary = {
                "run_ts": run_ts,
                "asof_date": asof_str,
                "fund_count": len(results),
                "break_count": int(sum(item["break_count"] for item in results)),
                "high_severity_breaks": int(sum(item["high_severity_breaks"] for item in results)),
                "status": "REVIEW_REQUIRED" if any(item["break_count"] for item in results) else "SUCCESS",
                "data_source": data_source,
                "sofr_rate": sofr_rate,
                "runs": results,
            }
            LOGGER.info(
                "Completed NAV-Recon multi-fund run for %s across %s funds",
                asof_str,
                len(results),
            )
            return summary
        except Exception as exc:
            failure_log = _build_run_log(
                run_ts=run_ts,
                asof_date=asof_str,
                fund_id=current_fund_id,
                total_nav=None,
                nav_per_share=None,
                break_count=0,
                high_severity_breaks=0,
                data_dir=data_dir,
                status=f"FAILED: {exc}",
            )
            _log_failed_run(connection, failure_log)
            LOGGER.exception("NAV-Recon run failed for %s", asof_str)
            raise
    finally:
        connection.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the daily NAV-Recon workflow.")
    parser.add_argument("--asof", required=True, help="As-of date in YYYY-MM-DD format.")
    parser.add_argument("--data-dir", required=True, help="Directory containing input CSV files.")
    parser.add_argument("--fund-id", default=None, help="Optional fund override.")
    args = parser.parse_args()

    result = run_daily(
        asof_date=pd.Timestamp(args.asof),
        data_dir=Path(args.data_dir),
        fund_id=args.fund_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
