from __future__ import annotations

import math

import pandas as pd

from src.nav import compute_daily_nav


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "quantity_eod": 10.0,
                "price_local": 100.0,
                "security_currency": "USD",
                "fx_to_base": 1.0,
                "market_value_base": 1000.0,
                "corporate_action_flag": "",
            }
        ]
    )


def _blotter() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-03-31",
                "fund_id": "FUND1",
                "security_id": "",
                "txn_type": "CONTRIB",
                "quantity": 0.0,
                "price": 1.0,
                "cash_amount": 0.0,
                "fees": 0.0,
            }
        ]
    )


def test_total_nav_equals_securities_plus_cash_plus_accrued_income() -> None:
    nav = compute_daily_nav(
        positions=_positions(),
        blotter=_blotter(),
        policy={"cash_interest_rate": 0.05, "fund_id_default": "FUND1"},
        asof_date=pd.Timestamp("2026-03-31"),
    )

    row = nav.iloc[0]
    assert math.isclose(
        row["total_nav"],
        row["securities_mv"] + row["cash_balance"] + row["accrued_income"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_nav_per_share_equals_total_nav_divided_by_shares_outstanding() -> None:
    nav = compute_daily_nav(
        positions=_positions(),
        blotter=_blotter(),
        policy={"cash_interest_rate": 0.05, "fund_id_default": "FUND1"},
        asof_date=pd.Timestamp("2026-03-31"),
    )

    row = nav.iloc[0]
    assert math.isclose(
        row["nav_per_share"],
        row["total_nav"] / row["shares_outstanding"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_accrued_income_is_positive_when_cash_interest_rate_is_positive() -> None:
    nav = compute_daily_nav(
        positions=_positions(),
        blotter=_blotter(),
        policy={"cash_interest_rate": 0.05, "fund_id_default": "FUND1"},
        asof_date=pd.Timestamp("2026-03-31"),
    )

    assert nav.iloc[0]["accrued_income"] > 0
