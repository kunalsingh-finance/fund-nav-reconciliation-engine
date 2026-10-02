from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from src.positions import build_corporate_action_cash_flows, build_positions
from src.reconciliation import BREAK_COLUMNS, run_reconciliation
from src import run_daily as runner


def trade(date, quantity, txn_type="BUY", fund_id="FUND1"):
    return dict(date=date, fund_id=fund_id, security_id="AAPL", txn_type=txn_type,
                quantity=quantity, price=50.0, cash_amount=0.0, fees=0.0)


def actions(ratio=2.0):
    return pd.DataFrame([dict(ex_date="2024-01-02", security_id="AAPL",
                             action_type="SPLIT", ratio=ratio, cash_amount=0.0)])


def ledger(trades, corporate_actions):
    return build_positions(
        pd.DataFrame(trades),
        pd.DataFrame([dict(date="2024-01-05", security_id="AAPL", price=50.0, source="internal")]),
        pd.DataFrame([dict(date="2024-01-05", ccy_pair="USDUSD", rate=1.0, source="internal")]),
        pd.DataFrame([dict(security_id="AAPL", currency="USD", asset_class="Equity", multiplier=1.0)]),
        corporate_actions, "USD", pd.Timestamp("2024-01-05"),
    )


@pytest.mark.parametrize("later_date,later_type,later_quantity,expected", [
    ("2024-01-03", "BUY", 10, 210),
    ("2024-01-02", "BUY", 10, 210),
    ("2024-01-03", "SELL", 10, 190),
    ("2024-01-03", "SELL", 200, 0),
])
def test_split_only_adjusts_pre_effective_trades(later_date, later_type, later_quantity, expected):
    result = ledger([trade("2024-01-01", 100), trade(later_date, later_quantity, later_type)], actions())
    assert (0 if result.empty else result.iloc[0].quantity_eod) == expected


def test_reverse_and_successive_splits_keep_trade_units():
    corporate_actions = pd.concat([actions(0.5), pd.DataFrame([
        dict(ex_date="2024-01-04", security_id="AAPL", action_type="SPLIT", ratio=3.0, cash_amount=0.0)
    ])], ignore_index=True)
    result = ledger([trade("2024-01-01", 100), trade("2024-01-03", 10), trade("2024-01-04", 5)], corporate_actions)
    assert result.iloc[0].quantity_eod == 185


def test_fractional_split_preserves_fractional_shares():
    result = ledger([trade("2024-01-01", 3)], actions(0.5))
    assert result.iloc[0].quantity_eod == 1.5


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_split_ratio_rejected(ratio):
    with pytest.raises(ValueError, match="Split ratios"):
        ledger([trade("2024-01-01", 100)], actions(ratio))


def test_dividends_use_split_adjusted_prior_holdings_and_requested_fund():
    blotter = pd.DataFrame([
        trade("2024-01-01", 100), trade("2024-01-03", 10),
        trade("2024-01-04", 50), trade("2024-01-01", 1000, fund_id="FUND2"),
    ])
    corporate_actions = pd.concat([actions(), pd.DataFrame([
        dict(ex_date="2024-01-04", security_id="AAPL", action_type="DIV", ratio=1.0, cash_amount=0.5)
    ])], ignore_index=True)
    result = build_corporate_action_cash_flows(blotter, corporate_actions, "FUND1", pd.Timestamp("2024-01-05"))
    assert len(result) == 1
    assert result.iloc[0].quantity == 210
    assert result.iloc[0].cash_amount == 105
    existing = pd.concat([blotter, result], ignore_index=True)
    assert build_corporate_action_cash_flows(existing, corporate_actions, "FUND1", pd.Timestamp("2024-01-05")).empty


def navs():
    internal = pd.DataFrame([dict(date="2024-01-05", fund_id="FUND1", total_nav=100.0, cash_balance=100.0)])
    custodian = pd.DataFrame([dict(date="2024-01-05", fund_id="FUND1", custodian_nav=100.0,
                                  custodian_cash=100.0, custodian_positions=json.dumps([]))])
    return internal, custodian


def empty_positions():
    return pd.DataFrame(columns=["date", "fund_id", "security_id", "quantity_eod", "price_local", "corporate_action_flag"])


@pytest.mark.parametrize("missing_source,break_type", [
    ("custodian", "MISSING_CUSTODIAN"), ("internal", "MISSING_INTERNAL_NAV"),
])
def test_missing_source_is_high_severity(missing_source, break_type):
    internal, custodian = navs()
    result = run_reconciliation(
        internal if missing_source != "internal" else pd.DataFrame(),
        custodian if missing_source != "custodian" else pd.DataFrame(), empty_positions(), 10,
    )
    assert result.break_type.tolist() == [break_type]
    assert result.severity.tolist() == ["HIGH"]


@pytest.mark.parametrize("key,value", [("date", "2024-01-04"), ("fund_id", "FUND2")])
def test_unmatched_fund_or_date_cannot_disappear(key, value):
    internal, custodian = navs()
    custodian[key] = value
    result = run_reconciliation(internal, custodian, empty_positions(), 10)
    assert set(result.break_type) == {"MISSING_CUSTODIAN", "MISSING_INTERNAL_NAV"}


def test_datetime_keys_match_and_cash_only_portfolio_passes():
    internal, custodian = navs()
    internal.date = pd.to_datetime(internal.date)
    assert run_reconciliation(internal, custodian, empty_positions(), 10).empty


@pytest.mark.parametrize("source", ["internal", "custodian"])
def test_duplicate_nav_keys_rejected(source):
    internal, custodian = navs()
    if source == "internal":
        internal = pd.concat([internal, internal])
    else:
        custodian = pd.concat([custodian, custodian])
    with pytest.raises(ValueError, match="one record"):
        run_reconciliation(internal, custodian, empty_positions(), 10)


def test_nonfinite_nav_cannot_pass_reconciliation():
    internal, custodian = navs()
    custodian.loc[0, "custodian_nav"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_reconciliation(internal, custodian, empty_positions(), 10)


def test_clean_rerun_removes_old_breaks_and_closed_positions():
    internal, _ = navs()
    stale_break = pd.DataFrame([dict(date="2024-01-05", fund_id="FUND1", break_type="MISSING_CUSTODIAN",
                                    severity="HIGH", internal_value=100.0, custodian_value=None, diff_bps=None,
                                    details="Missing", resolution="Obtain records")], columns=BREAK_COLUMNS)
    with sqlite3.connect(":memory:") as connection:
        runner._ensure_schema(connection, runner.DDL_PATH)
        stale_positions = ledger([trade("2024-01-01", 100)], actions())
        full_nav = internal.assign(securities_mv=0.0, accrued_income=0.0, shares_outstanding=100.0, nav_per_share=1.0)
        runner._write_to_sqlite(connection, stale_positions, full_nav, stale_break, [])
        runner._write_to_sqlite(connection, stale_positions.iloc[:0], full_nav, stale_break.iloc[:0], [])
        for table in ("nav_breaks", "nav_reconciliation", "nav_daily_positions"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("corporate_actions_enabled", [True, False])
def test_daily_run_and_export_require_review_for_missing_custodian(tmp_path, monkeypatch, corporate_actions_enabled):
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    inputs = dict(
        trade_blotter=pd.DataFrame([trade("2024-01-01", 100)]),
        prices=pd.DataFrame([dict(date="2024-01-05", security_id="AAPL", price=50.0, source="internal")]),
        fx_rates=pd.DataFrame([dict(date="2024-01-05", ccy_pair="USDUSD", rate=1.0, source="internal")]),
        security_master=pd.DataFrame([dict(security_id="AAPL", currency="USD", asset_class="Equity", multiplier=1.0)]),
        corporate_actions=actions(),
        custodian_nav=navs()[1].iloc[:0],
    )
    positions, _, reconciliation, run_log, result = runner._run_single_fund(
        fund_id="FUND1", asof=pd.Timestamp("2024-01-05"), asof_str="2024-01-05",
        inputs=inputs, policy={"base_currency": "USD"}, tolerance_bps=10.0,
        corporate_actions_enabled=corporate_actions_enabled, run_ts="2024-01-05T00:00:00Z", data_dir=tmp_path,
        data_source="dummy", sofr_rate=0.0,
    )
    assert reconciliation.break_type.tolist() == ["MISSING_CUSTODIAN"]
    assert positions.iloc[0].quantity_eod == (200 if corporate_actions_enabled else 100)
    assert run_log["status"] == result["status"] == "REVIEW_REQUIRED"
    summary = json.loads((tmp_path / "2024-01-05_FUND1" / "summary.json").read_text())
    assert summary["status"] == "REVIEW_REQUIRED"
    assert summary["exports"]["positions"] == "positions.csv"
