from __future__ import annotations

import pandas as pd

from src import positions as positions_module
from src.positions import build_positions


def _security_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "security_id": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "currency": "USD",
                "asset_class": "Equity",
                "multiplier": 1.0,
            }
        ]
    )


def _fx_rates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "ccy_pair": "USDUSD", "rate": 1.0, "source": "internal"},
            {"date": "2024-01-02", "ccy_pair": "USDUSD", "rate": 1.0, "source": "internal"},
            {"date": "2024-01-05", "ccy_pair": "USDUSD", "rate": 1.0, "source": "internal"},
        ]
    )


def test_split_adjusts_quantity_correctly() -> None:
    blotter = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "txn_type": "BUY",
                "quantity": 100.0,
                "price": 100.0,
                "cash_amount": -10000.0,
                "fees": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "security_id": "AAPL", "price": 100.0, "source": "internal"},
            {"date": "2024-01-02", "security_id": "AAPL", "price": 120.0, "source": "internal"},
        ]
    )
    corporate_actions = pd.DataFrame(
        [
            {
                "ex_date": "2024-01-02",
                "security_id": "AAPL",
                "action_type": "SPLIT",
                "ratio": 2.0,
                "cash_amount": 0.0,
            }
        ]
    )

    positions = build_positions(
        blotter=blotter,
        prices=prices,
        fx_rates=_fx_rates(),
        security_master=_security_master(),
        corporate_actions=corporate_actions,
        base_currency="USD",
        asof_date=pd.Timestamp("2024-01-02"),
    )

    row = positions.iloc[0]
    assert row["quantity_eod"] == 200.0
    assert row["price_local"] == 60.0
    assert "SPLIT" in row["corporate_action_flag"]


def test_stale_price_flag_triggers_when_price_is_older_than_three_days() -> None:
    positions_module.MAX_PRICE_STALENESS_DAYS = 3

    blotter = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "txn_type": "BUY",
                "quantity": 100.0,
                "price": 100.0,
                "cash_amount": -10000.0,
                "fees": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "security_id": "AAPL", "price": 100.0, "source": "internal"}
        ]
    )

    positions = build_positions(
        blotter=blotter,
        prices=prices,
        fx_rates=_fx_rates(),
        security_master=_security_master(),
        corporate_actions=pd.DataFrame(columns=["ex_date", "security_id", "action_type", "ratio", "cash_amount"]),
        base_currency="USD",
        asof_date=pd.Timestamp("2024-01-05"),
    )

    assert "STALE_PRICE" in positions.iloc[0]["corporate_action_flag"]


def test_zero_price_triggers_stale_flag() -> None:
    blotter = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "fund_id": "FUND1",
                "security_id": "AAPL",
                "txn_type": "BUY",
                "quantity": 100.0,
                "price": 100.0,
                "cash_amount": -10000.0,
                "fees": 0.0,
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2024-01-02", "security_id": "AAPL", "price": 0.0, "source": "internal"}
        ]
    )

    positions = build_positions(
        blotter=blotter,
        prices=prices,
        fx_rates=_fx_rates(),
        security_master=_security_master(),
        corporate_actions=pd.DataFrame(columns=["ex_date", "security_id", "action_type", "ratio", "cash_amount"]),
        base_currency="USD",
        asof_date=pd.Timestamp("2024-01-02"),
    )

    assert positions.iloc[0]["price_local"] == 0.0
    assert "STALE_PRICE" in positions.iloc[0]["corporate_action_flag"]
