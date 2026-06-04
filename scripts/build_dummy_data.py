from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import positions as positions_module
from src.nav import compute_daily_nav
from src.positions import build_positions


SECTOR_MAP = {
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corporation", "Technology"),
    "GOOGL": ("Alphabet Inc.", "Communication Services"),
    "AMZN": ("Amazon.com, Inc.", "Consumer Discretionary"),
}

BASE_PRICE_MAP = {
    "AAPL": 185.0,
    "MSFT": 390.0,
    "GOOGL": 145.0,
    "AMZN": 175.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock NAV-Recon input files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated CSV files.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--fund-id", required=True, help="Fund identifier.")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of securities.")
    return parser.parse_args()


def _month_start_business_days(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return (
        dates.to_series()
        .groupby(dates.to_period("M"))
        .min()
        .sort_values()
        .tolist()
    )


def _quarter_end_business_days(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return (
        dates.to_series()
        .groupby(dates.to_period("Q"))
        .max()
        .sort_values()
        .tolist()
    )


def _build_security_master(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        name, sector = SECTOR_MAP.get(ticker, (f"{ticker} Holdings", "Diversified"))
        rows.append(
            {
                "security_id": ticker,
                "name": name,
                "sector": sector,
                "currency": "USD",
                "asset_class": "Equity",
                "multiplier": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _build_price_history(
    business_dates: pd.DatetimeIndex,
    tickers: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        base_price = BASE_PRICE_MAP.get(ticker, float(rng.uniform(50.0, 300.0)))
        daily_returns = rng.normal(loc=0.00025, scale=0.018, size=len(business_dates))
        price_path = base_price * np.exp(np.cumsum(daily_returns))
        price_frame = pd.DataFrame(
            {
                "date": business_dates,
                "security_id": ticker,
                "price": np.round(price_path, 2),
                "source": "internal",
            }
        )
        frames.append(price_frame)
    return pd.concat(frames, ignore_index=True)


def _resolve_split_date(business_dates: pd.DatetimeIndex) -> pd.Timestamp:
    target = pd.Timestamp("2025-06-16")
    eligible = business_dates[business_dates >= target]
    if not eligible.empty:
        return eligible[0]
    return business_dates[len(business_dates) // 2]


def _quantity_on_date(security_id: str, date_value: pd.Timestamp, split_date: pd.Timestamp) -> float:
    base_quantity = 100.0
    if security_id == "AAPL" and date_value >= split_date:
        return base_quantity * 2.0
    return base_quantity


def _build_corporate_actions_and_dividends(
    prices: pd.DataFrame,
    business_dates: pd.DatetimeIndex,
    fund_id: str,
    tickers: list[str],
    split_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarter_ends = _quarter_end_business_days(business_dates)

    corporate_action_rows: list[dict[str, object]] = [
        {
            "ex_date": split_date.strftime("%Y-%m-%d"),
            "security_id": "AAPL",
            "action_type": "SPLIT",
            "ratio": 2.0,
            "cash_amount": 0.0,
        }
    ]
    dividend_rows: list[dict[str, object]] = []

    price_lookup = prices.set_index(["date", "security_id"])["price"]
    for ex_date in quarter_ends:
        for ticker in tickers:
            quantity = _quantity_on_date(ticker, ex_date, split_date)
            price = float(price_lookup.loc[(ex_date, ticker)])
            dividend_per_share = round(price * 0.01, 4)
            total_dividend = round(dividend_per_share * quantity, 2)
            corporate_action_rows.append(
                {
                    "ex_date": ex_date.strftime("%Y-%m-%d"),
                    "security_id": ticker,
                    "action_type": "DIV",
                    "ratio": 1.0,
                    "cash_amount": dividend_per_share,
                }
            )
            dividend_rows.append(
                {
                    "date": ex_date.strftime("%Y-%m-%d"),
                    "fund_id": fund_id,
                    "security_id": ticker,
                    "txn_type": "DIV",
                    "quantity": 0.0,
                    "price": dividend_per_share,
                    "cash_amount": total_dividend,
                    "fees": 0.0,
                }
            )

    corporate_actions = pd.DataFrame(corporate_action_rows)
    dividends = pd.DataFrame(dividend_rows)
    return corporate_actions, dividends


def _build_trade_blotter(
    prices: pd.DataFrame,
    business_dates: pd.DatetimeIndex,
    fund_id: str,
    tickers: list[str],
) -> pd.DataFrame:
    start_date = business_dates[0]
    month_starts = [date_value for date_value in _month_start_business_days(business_dates) if date_value > start_date]
    opening_prices = prices.loc[prices["date"].eq(start_date), ["security_id", "price"]].set_index("security_id")["price"]

    trade_rows: list[dict[str, object]] = []
    for ticker in tickers:
        price = float(opening_prices.loc[ticker])
        quantity = 100.0
        trade_rows.append(
            {
                "date": start_date.strftime("%Y-%m-%d"),
                "fund_id": fund_id,
                "security_id": ticker,
                "txn_type": "BUY",
                "quantity": quantity,
                "price": price,
                "cash_amount": round(-(quantity * price), 2),
                "fees": 4.95,
            }
        )

    for date_value in month_starts:
        trade_rows.append(
            {
                "date": date_value.strftime("%Y-%m-%d"),
                "fund_id": fund_id,
                "security_id": "CASH",
                "txn_type": "CONTRIB",
                "quantity": 0.0,
                "price": 1.0,
                "cash_amount": 10000.0,
                "fees": 0.0,
            }
        )

    return pd.DataFrame(trade_rows)


def _build_fx_rates(business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": business_dates.strftime("%Y-%m-%d"),
            "ccy_pair": "USDUSD",
            "rate": 1.0,
            "source": "internal",
        }
    )


def _load_policy() -> dict:
    with (PROJECT_ROOT / "policy.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_custodian_nav(
    business_dates: pd.DatetimeIndex,
    fund_id: str,
    blotter: pd.DataFrame,
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    security_master: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    policy: dict,
    tickers: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    positions_module.MAX_PRICE_STALENESS_DAYS = int(policy.get("max_price_staleness_days", 3))
    positions_module.PRICE_SOURCE_PRIORITY = tuple(
        policy.get("price_source_priority", ["internal", "vendor", "fallback"])
    )

    last_date = business_dates[-1]
    records: list[dict[str, object]] = []

    for asof_date in business_dates:
        positions = build_positions(
            blotter=blotter,
            prices=prices,
            fx_rates=fx_rates,
            security_master=security_master,
            corporate_actions=corporate_actions,
            base_currency=str(policy.get("base_currency", "USD")),
            asof_date=asof_date,
        )
        nav = compute_daily_nav(
            positions=positions,
            blotter=blotter,
            policy=policy,
            asof_date=asof_date,
        )
        nav_row = nav.loc[nav["fund_id"].eq(fund_id)].iloc[0]

        position_rows = positions.loc[
            positions["fund_id"].eq(fund_id),
            ["security_id", "quantity_eod", "price_local"],
        ].copy()
        custodian_positions = [
            {
                "security_id": row.security_id,
                "quantity": round(float(row.quantity_eod), 6),
                "price": round(float(row.price_local), 4),
            }
            for row in position_rows.itertuples(index=False)
        ]

        custodian_cash = round(float(nav_row["cash_balance"]), 2)
        custodian_nav_value = round(float(nav_row["total_nav"]) * (1 + rng.uniform(-8.0, 8.0) / 10000.0), 2)

        if asof_date == last_date:
            for item in custodian_positions:
                if item["security_id"] == tickers[0]:
                    item["price"] = round(item["price"] + 0.25, 4)
            custodian_positions = [
                item for item in custodian_positions if item["security_id"] != tickers[2]
            ]
            custodian_cash = round(custodian_cash - 250.0, 2)
            custodian_nav_value = round(
                sum(item["quantity"] * item["price"] for item in custodian_positions)
                + custodian_cash
                + float(nav_row["accrued_income"]),
                2,
            )

        records.append(
            {
                "date": asof_date.strftime("%Y-%m-%d"),
                "fund_id": fund_id,
                "custodian_nav": custodian_nav_value,
                "custodian_cash": custodian_cash,
                "custodian_positions": json.dumps(custodian_positions),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    fund_id = args.fund_id
    tickers = args.tickers
    business_dates = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(42)

    security_master = _build_security_master(tickers)
    prices = _build_price_history(business_dates, tickers, rng)
    trade_blotter = _build_trade_blotter(prices, business_dates, fund_id, tickers)
    split_date = _resolve_split_date(business_dates)
    corporate_actions, dividends = _build_corporate_actions_and_dividends(
        prices=prices,
        business_dates=business_dates,
        fund_id=fund_id,
        tickers=tickers,
        split_date=split_date,
    )
    trade_blotter = (
        pd.concat([trade_blotter, dividends], ignore_index=True)
        .sort_values(["date", "txn_type", "security_id"])
        .reset_index(drop=True)
    )
    fx_rates = _build_fx_rates(business_dates)
    policy = _load_policy()
    custodian_nav = _build_custodian_nav(
        business_dates=business_dates,
        fund_id=fund_id,
        blotter=trade_blotter,
        prices=prices,
        fx_rates=fx_rates,
        security_master=security_master,
        corporate_actions=corporate_actions,
        policy=policy,
        tickers=tickers,
        rng=rng,
    )

    trade_blotter.to_csv(out_dir / "trade_blotter.csv", index=False)
    prices.to_csv(out_dir / "prices.csv", index=False)
    security_master.to_csv(out_dir / "security_master.csv", index=False)
    custodian_nav.to_csv(out_dir / "custodian_nav.csv", index=False)
    fx_rates.to_csv(out_dir / "fx_rates.csv", index=False)
    corporate_actions.to_csv(out_dir / "corporate_actions.csv", index=False)

    print(f"Generated dummy NAV-Recon inputs in {out_dir}")


if __name__ == "__main__":
    main()
