from __future__ import annotations

import json

import pandas as pd

from src.reconciliation import run_reconciliation


def _internal_nav(total_nav: float, cash_balance: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "securities_mv": 80.0,
                "cash_balance": cash_balance,
                "accrued_income": 0.0,
                "total_nav": total_nav,
                "shares_outstanding": 100.0,
                "nav_per_share": total_nav / 100.0,
            }
        ]
    )


def _custodian_nav(
    custodian_nav_value: float,
    custodian_cash: float,
    positions: list[dict[str, float | str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "custodian_nav": custodian_nav_value,
                "custodian_cash": custodian_cash,
                "custodian_positions": json.dumps(positions),
            }
        ]
    )


def test_nav_break_severity_high_when_diff_exceeds_fifty_bps() -> None:
    positions = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "quantity_eod": 100.0,
                "price_local": 1.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 100.0,
                "corporate_action_flag": "",
            }
        ]
    )
    reconciliation = run_reconciliation(
        internal_nav=_internal_nav(total_nav=100.60, cash_balance=10.0),
        custodian_nav=_custodian_nav(100.00, 10.0, [{"security_id": "AAPL", "quantity": 100.0, "price": 1.0}]),
        positions=positions,
        tolerance_bps=10.0,
    )

    nav_break = reconciliation.loc[reconciliation["break_type"] == "NAV_BREAK"].iloc[0]
    assert nav_break["severity"] == "HIGH"


def test_nav_break_severity_medium_when_diff_is_twenty_five_bps() -> None:
    positions = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "quantity_eod": 100.0,
                "price_local": 1.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 100.0,
                "corporate_action_flag": "",
            }
        ]
    )
    reconciliation = run_reconciliation(
        internal_nav=_internal_nav(total_nav=100.25, cash_balance=10.0),
        custodian_nav=_custodian_nav(100.00, 10.0, [{"security_id": "AAPL", "quantity": 100.0, "price": 1.0}]),
        positions=positions,
        tolerance_bps=10.0,
    )

    nav_break = reconciliation.loc[reconciliation["break_type"] == "NAV_BREAK"].iloc[0]
    assert nav_break["severity"] == "MEDIUM"


def test_missing_position_is_always_high() -> None:
    positions = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "quantity_eod": 100.0,
                "price_local": 1.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 100.0,
                "corporate_action_flag": "",
            }
        ]
    )
    reconciliation = run_reconciliation(
        internal_nav=_internal_nav(total_nav=100.00, cash_balance=10.0),
        custodian_nav=_custodian_nav(100.00, 10.0, []),
        positions=positions,
        tolerance_bps=10.0,
    )

    missing_position = reconciliation.loc[reconciliation["break_type"] == "MISSING_POSITION"].iloc[0]
    assert missing_position["severity"] == "HIGH"


def test_resolution_text_matches_expected_string_per_break_type() -> None:
    positions = pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "quantity_eod": 100.0,
                "price_local": 10.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 1000.0,
                "corporate_action_flag": "STALE_PRICE",
            },
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "MSFT",
                "quantity_eod": 50.0,
                "price_local": 20.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 1000.0,
                "corporate_action_flag": "",
            },
        ]
    )
    reconciliation = run_reconciliation(
        internal_nav=_internal_nav(total_nav=100.25, cash_balance=11.0),
        custodian_nav=_custodian_nav(
            100.00,
            10.0,
            [{"security_id": "AAPL", "quantity": 100.0, "price": 10.5}],
        ),
        positions=positions,
        tolerance_bps=10.0,
    )

    resolutions = reconciliation.set_index("break_type")["resolution"].to_dict()
    assert resolutions["NAV_BREAK"] == "Investigate price and cash discrepancies before NAV publication."
    assert resolutions["CASH_BREAK"] == "Trace cash movements against custodian cash statement."
    assert resolutions["PRICE_BREAK"] == "Verify price source and override if custodian price is confirmed."
    assert resolutions["MISSING_POSITION"] == "Confirm trade booking - position exists in one source only."
    assert resolutions["STALE_PRICE"] == "Request fresh price from vendor or apply override."
