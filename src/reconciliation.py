from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd


BREAK_COLUMNS = [
    "date",
    "fund_id",
    "break_type",
    "severity",
    "internal_value",
    "custodian_value",
    "diff_bps",
    "details",
    "resolution",
]

RESOLUTION_TEXT = {
    "NAV_BREAK": "Investigate price and cash discrepancies before NAV publication.",
    "POSITION_BREAK": "Reconcile trade blotter against custodian position report.",
    "PRICE_BREAK": "Verify price source and override if custodian price is confirmed.",
    "CASH_BREAK": "Trace cash movements against custodian cash statement.",
    "MISSING_POSITION": "Confirm trade booking - position exists in one source only.",
    "STALE_PRICE": "Request fresh price from vendor or apply override.",
    "MERGER": "Review merger terms and update security mapping manually.",
}


@dataclass
class BreakRecord:
    date: str
    fund_id: str
    break_type: str
    severity: str
    internal_value: float | None
    custodian_value: float | None
    diff_bps: float | None
    details: str
    resolution: str


def _empty_reconciliation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAK_COLUMNS)


def _calc_bps(internal_value: float, custodian_value: float) -> float:
    denominator = abs(internal_value) if internal_value else 1.0
    return abs(internal_value - custodian_value) / denominator * 10000.0


def _resolve_severity(break_type: str, diff_bps: float | None) -> str:
    if break_type == "NAV_BREAK":
        if diff_bps is not None and diff_bps >= 50.0:
            return "HIGH"
        if diff_bps is not None and diff_bps > 10.0:
            return "MEDIUM"
        return "LOW"
    if break_type in {"POSITION_BREAK", "PRICE_BREAK", "CASH_BREAK", "MERGER"}:
        return "MEDIUM"
    if break_type == "MISSING_POSITION":
        return "HIGH"
    if break_type == "STALE_PRICE":
        return "LOW"
    return "LOW"


def _make_break_record(
    *,
    date: str,
    fund_id: str,
    break_type: str,
    internal_value: float | None,
    custodian_value: float | None,
    diff_bps: float | None,
    details: str,
) -> BreakRecord:
    return BreakRecord(
        date=date,
        fund_id=fund_id,
        break_type=break_type,
        severity=_resolve_severity(break_type, diff_bps),
        internal_value=internal_value,
        custodian_value=custodian_value,
        diff_bps=diff_bps,
        details=details,
        resolution=RESOLUTION_TEXT[break_type],
    )


def _parse_custodian_positions(value: str) -> pd.DataFrame:
    if not value or pd.isna(value):
        return pd.DataFrame(columns=["security_id", "custodian_quantity", "custodian_price"])

    records = json.loads(value)
    if not records:
        return pd.DataFrame(columns=["security_id", "custodian_quantity", "custodian_price"])

    frame = pd.DataFrame.from_records(records)
    return frame.rename(
        columns={
            "quantity": "custodian_quantity",
            "price": "custodian_price",
        }
    )[["security_id", "custodian_quantity", "custodian_price"]]


def run_reconciliation(
    internal_nav: pd.DataFrame,
    custodian_nav: pd.DataFrame,
    positions: pd.DataFrame,
    tolerance_bps: float,
) -> pd.DataFrame:
    if internal_nav.empty or custodian_nav.empty:
        return _empty_reconciliation_frame()

    internal_nav_frame = internal_nav.copy()
    custodian_nav_frame = custodian_nav.copy()
    positions_frame = positions.copy()

    merged_nav = internal_nav_frame.merge(
        custodian_nav_frame,
        on=["date", "fund_id"],
        how="inner",
    )
    if merged_nav.empty:
        return _empty_reconciliation_frame()

    records: list[BreakRecord] = []
    for row in merged_nav.itertuples(index=False):
        nav_diff_bps = _calc_bps(float(row.total_nav), float(row.custodian_nav))
        if nav_diff_bps > tolerance_bps:
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="NAV_BREAK",
                    internal_value=float(row.total_nav),
                    custodian_value=float(row.custodian_nav),
                    diff_bps=nav_diff_bps,
                    details="Internal NAV differs from custodian NAV beyond policy tolerance.",
                )
            )

        if abs(float(row.cash_balance) - float(row.custodian_cash)) > 0.01:
            cash_diff_bps = _calc_bps(float(row.cash_balance), float(row.custodian_cash))
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="CASH_BREAK",
                    internal_value=float(row.cash_balance),
                    custodian_value=float(row.custodian_cash),
                    diff_bps=cash_diff_bps,
                    details="Cash balance mismatch between internal books and custodian balance.",
                )
            )

        internal_positions = positions_frame.loc[
            (positions_frame["date"] == row.date) & (positions_frame["fund_id"] == row.fund_id)
        ].copy()
        custodian_positions = _parse_custodian_positions(row.custodian_positions)

        comparison = internal_positions[
            ["security_id", "quantity_eod", "price_local", "corporate_action_flag"]
        ].rename(
            columns={
                "quantity_eod": "internal_quantity",
                "price_local": "internal_price",
            }
        )
        comparison = comparison.merge(custodian_positions, on="security_id", how="outer")

        missing_positions = comparison.loc[
            comparison["internal_quantity"].isna() | comparison["custodian_quantity"].isna()
        ]
        for missing_row in missing_positions.itertuples(index=False):
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="MISSING_POSITION",
                    internal_value=None if pd.isna(missing_row.internal_quantity) else float(missing_row.internal_quantity),
                    custodian_value=None if pd.isna(missing_row.custodian_quantity) else float(missing_row.custodian_quantity),
                    diff_bps=None,
                    details=f"{missing_row.security_id} is missing from one side of the reconciliation.",
                )
            )

        overlapping = comparison.loc[
            comparison["internal_quantity"].notna() & comparison["custodian_quantity"].notna()
        ].copy()

        position_breaks = overlapping.loc[
            (overlapping["internal_quantity"] - overlapping["custodian_quantity"]).abs() > 1e-9
        ]
        for position_row in position_breaks.itertuples(index=False):
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="POSITION_BREAK",
                    internal_value=float(position_row.internal_quantity),
                    custodian_value=float(position_row.custodian_quantity),
                    diff_bps=_calc_bps(float(position_row.internal_quantity), float(position_row.custodian_quantity)),
                    details=f"Quantity mismatch for {position_row.security_id}.",
                )
            )

        price_breaks = overlapping.loc[
            (overlapping["internal_price"] - overlapping["custodian_price"]).abs() > 0.01
        ]
        for price_row in price_breaks.itertuples(index=False):
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="PRICE_BREAK",
                    internal_value=float(price_row.internal_price),
                    custodian_value=float(price_row.custodian_price),
                    diff_bps=_calc_bps(float(price_row.internal_price), float(price_row.custodian_price)),
                    details=f"Price discrepancy greater than $0.01 for {price_row.security_id}.",
                )
            )

        stale_rows = internal_positions.loc[
            internal_positions["corporate_action_flag"].fillna("").str.contains("STALE_PRICE")
        ]
        for stale_row in stale_rows.itertuples(index=False):
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="STALE_PRICE",
                    internal_value=float(stale_row.price_local),
                    custodian_value=float(stale_row.price_local),
                    diff_bps=0.0,
                    details=f"{stale_row.security_id} is priced with stale or invalid market data.",
                )
            )

        merger_rows = internal_positions.loc[
            internal_positions["corporate_action_flag"].fillna("").str.contains("MERGER")
        ]
        for merger_row in merger_rows.itertuples(index=False):
            records.append(
                _make_break_record(
                    date=row.date,
                    fund_id=row.fund_id,
                    break_type="MERGER",
                    internal_value=None,
                    custodian_value=None,
                    diff_bps=None,
                    details=f"MERGER action detected for {merger_row.security_id} on {row.date} - manual review required",
                )
            )

    if not records:
        return _empty_reconciliation_frame()

    output = pd.DataFrame([asdict(record) for record in records], columns=BREAK_COLUMNS)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    output["severity_order"] = output["severity"].map(severity_order).fillna(9)
    output = (
        output.sort_values(["date", "fund_id", "severity_order", "break_type"])
        .drop(columns=["severity_order"])
        .reset_index(drop=True)
    )
    return output[BREAK_COLUMNS]
