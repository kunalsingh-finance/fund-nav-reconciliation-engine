from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from src import run_daily as runner


@pytest.mark.parametrize("fund_id", ["FUND1", None])
def test_failed_rerun_invalidates_previous_success_pack(tmp_path, monkeypatch, fund_id):
    monkeypatch.setattr(runner, "DB_PATH", tmp_path / "nav.db")
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "outputs")
    for fund in ("FUND1", "FUND2"):
        pack = runner.OUTPUT_DIR / f"2024-01-05_{fund}"
        pack.mkdir(parents=True)
        (pack / "summary.json").write_text(json.dumps({"status": "SUCCESS", "total_nav": 100}), encoding="utf-8")
        (pack / "nav.csv").write_text("total_nav\n100\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        runner.run_daily(pd.Timestamp("2024-01-05"), tmp_path / "missing-inputs", fund_id)

    expected = ("FUND1", "FUND2") if fund_id is None else ("FUND1",)
    for fund in expected:
        pack = runner.OUTPUT_DIR / f"2024-01-05_{fund}"
        summary = json.loads((pack / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "FAILED"
        assert summary["total_nav"] is None
        assert summary["exports"] == {}
        assert "trade_blotter.csv" in summary["error"]
        assert (pack / "nav.csv").read_text(encoding="utf-8") == "total_nav\n100\n"
    if fund_id is not None:
        other = runner.OUTPUT_DIR / "2024-01-05_FUND2" / "summary.json"
        assert json.loads(other.read_text(encoding="utf-8"))["status"] == "SUCCESS"


def cash_only_inputs():
    return {
        "trade_blotter": pd.DataFrame([
            dict(date="2024-01-01", fund_id=fund, security_id=None, txn_type="CONTRIB",
                 quantity=0, price=0, cash_amount=0, fees=0) for fund in ("FUND1", "FUND2")
        ]),
        "prices": pd.DataFrame(columns=["date", "security_id", "price", "source"]),
        "fx_rates": pd.DataFrame(columns=["date", "ccy_pair", "rate", "source"]),
        "security_master": pd.DataFrame(columns=["security_id", "currency", "asset_class", "multiplier"]),
        "corporate_actions": pd.DataFrame(columns=["ex_date", "security_id", "action_type", "ratio", "cash_amount"]),
        "custodian_nav": pd.DataFrame([
            dict(date=pd.Timestamp("2024-01-05"), fund_id=fund, custodian_nav=1_000_000,
                 custodian_cash=1_000_000, custodian_positions="[]") for fund in ("FUND1", "FUND2")
        ]),
    }


@pytest.mark.parametrize("failure_stage", ["later_fund", "database"])
def test_partial_all_fund_failure_blocks_every_pack_and_recovers(tmp_path, monkeypatch, failure_stage):
    monkeypatch.setattr(runner, "DB_PATH", tmp_path / "nav.db")
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(runner, "_load_policy", lambda path: {})
    monkeypatch.setattr(runner, "_read_inputs", lambda path: cash_only_inputs())
    original_fund_run = runner._run_single_fund
    original_write = runner._write_to_sqlite

    def fund_run(**kwargs):
        if kwargs["fund_id"] == "FUND2":
            raise ValueError("Invalid FUND2 inputs")
        return original_fund_run(**kwargs)

    def database_write(*args):
        raise sqlite3.OperationalError("Database write rejected")

    if failure_stage == "later_fund":
        monkeypatch.setattr(runner, "_run_single_fund", fund_run)
    else:
        monkeypatch.setattr(runner, "_write_to_sqlite", database_write)
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        runner.run_daily(pd.Timestamp("2024-01-05"), tmp_path)
    for fund in ("FUND1", "FUND2"):
        summary = runner.OUTPUT_DIR / f"2024-01-05_{fund}" / "summary.json"
        assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "FAILED"
    with sqlite3.connect(runner.DB_PATH) as connection:
        assert connection.execute("SELECT COUNT(*) FROM nav_daily_nav").fetchone()[0] == 0
        assert connection.execute("SELECT status FROM nav_run_log").fetchone()[0].startswith("FAILED")

    monkeypatch.setattr(runner, "_run_single_fund", original_fund_run)
    monkeypatch.setattr(runner, "_write_to_sqlite", original_write)
    result = runner.run_daily(pd.Timestamp("2024-01-05"), tmp_path)
    assert result["status"] == "SUCCESS"
    for fund in ("FUND1", "FUND2"):
        summary = runner.OUTPUT_DIR / f"2024-01-05_{fund}" / "summary.json"
        assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "SUCCESS"


def test_database_failure_after_position_insert_restores_prior_rows():
    positions = pd.DataFrame([dict(
        date="2024-01-05", fund_id="FUND1", security_id="AAPL", quantity_eod=1.0,
        price_local=50.0, security_currency="USD", fx_to_base=1.0,
        market_value_base=50.0, corporate_action_flag="",
    )])
    nav = pd.DataFrame([dict(
        date="2024-01-05", fund_id="FUND1", securities_mv=50.0, cash_balance=50.0,
        accrued_income=0.0, total_nav=100.0, shares_outstanding=100.0, nav_per_share=1.0,
    )])
    breaks = pd.DataFrame([dict(
        date="2024-01-05", fund_id="FUND1", break_type="NAV_DIFF", severity="HIGH",
        internal_value=100.0, custodian_value=90.0, diff_bps=1000.0,
        details="Review", resolution="Investigate",
    )])
    with sqlite3.connect(":memory:") as connection:
        runner._ensure_schema(connection, runner.DDL_PATH)
        runner._write_to_sqlite(connection, positions, nav, breaks, [])
        tables = ("nav_daily_positions", "nav_daily_nav", "nav_reconciliation", "nav_breaks")
        before = {table: connection.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
        connection.execute("""CREATE TRIGGER reject_nav BEFORE INSERT ON nav_daily_nav
                              WHEN NEW.total_nav = 200 BEGIN SELECT RAISE(ABORT, 'Rejected NAV'); END""")
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="Rejected NAV"):
            runner._write_to_sqlite(connection, positions.assign(quantity_eod=2.0), nav.assign(total_nav=200.0), breaks.iloc[:0], [])
        after = {table: connection.execute(f"SELECT * FROM {table}").fetchall() for table in tables}
        assert after == before
