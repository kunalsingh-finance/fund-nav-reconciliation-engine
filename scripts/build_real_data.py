from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YF_CACHE_DIR = PROJECT_ROOT / ".cache" / "yfinance"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))

from src import positions as positions_module
from src.nav import compute_daily_nav
from src.positions import build_positions


FRED_SOFR_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_NAV_BREAK_BPS = 60.0
DEFAULT_CASH_BREAK_AMOUNT = 5000.0


@dataclass(frozen=True)
class FundConfig:
    fund_id: str
    tickers: tuple[str, ...]
    benchmark_ticker: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real Level 2 and Level 3 NAV-Recon input files.")
    parser.add_argument("--tickers", nargs="+", help="Single-fund security identifiers to load from yfinance.")
    parser.add_argument("--fund-id", help="Single-fund identifier.")
    parser.add_argument("--funds", help="JSON array of fund configs for multi-fund runs.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--benchmark-ticker", default=None, help="Optional benchmark ticker override.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated CSV files.")
    parser.add_argument("--fred-api-key", default=None, help="Optional FRED API key for live SOFR.")
    parser.add_argument("--dry-run", action="store_true", help="Build frames and print the summary without writing files.")
    args = parser.parse_args()

    if args.funds:
        if args.tickers or args.fund_id:
            parser.error("--funds cannot be combined with --tickers or --fund-id.")
    elif not args.tickers or not args.fund_id:
        parser.error("--tickers and --fund-id are required when --funds is not provided.")

    return args


def _load_policy() -> dict[str, Any]:
    with (PROJECT_ROOT / "policy.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_policy(policy: dict[str, Any]) -> None:
    with (PROJECT_ROOT / "policy.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(policy, handle, sort_keys=False)


def _to_naive_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index))
    if dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    return dt_index


def _frame_from_download(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if downloaded.empty:
        raise ValueError("yfinance returned no price data for the requested date range.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        first_level = downloaded.columns.get_level_values(0)
        price_key = "Adj Close" if "Adj Close" in first_level else "Close"
        price_frame = downloaded[price_key].copy()
    else:
        price_key = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
        price_frame = downloaded[[price_key]].copy()
        price_frame.columns = [tickers[0]]

    if isinstance(price_frame, pd.Series):
        price_frame = price_frame.to_frame(name=tickers[0])

    price_frame.index = _to_naive_index(price_frame.index)
    return price_frame


def _download_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_staleness_days: int,
) -> pd.DataFrame:
    downloaded = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    price_frame = _frame_from_download(downloaded, tickers)
    business_dates = pd.bdate_range(start=start, end=end)

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if ticker not in price_frame.columns:
            raise ValueError(f"No price series was returned for ticker {ticker}.")
        series = price_frame[ticker].reindex(business_dates).ffill(limit=max_staleness_days).dropna()
        frame = series.reset_index()
        frame.columns = ["date", "price"]
        frame["security_id"] = ticker
        frame["source"] = "yfinance"
        frames.append(frame[["date", "security_id", "price", "source"]])

    output = pd.concat(frames, ignore_index=True)
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output["price"] = output["price"].astype(float)
    return output


def _first_trading_day_by_month(dates: pd.Series) -> list[pd.Timestamp]:
    dt_values = pd.to_datetime(dates).sort_values().drop_duplicates()
    return (
        dt_values.to_frame(name="date")
        .assign(period=lambda frame: frame["date"].dt.to_period("M"))
        .groupby("period", as_index=False)["date"]
        .min()["date"]
        .tolist()
    )


def _fetch_ticker_info(ticker: str) -> dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        warnings.warn(f"Ticker info lookup failed for {ticker}: {exc}")
        return {}


def _build_security_master(tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        info = _fetch_ticker_info(ticker)
        rows.append(
            {
                "security_id": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector") or "Equity",
                "currency": str(info.get("currency") or "USD").upper(),
                "asset_class": "Equity",
                "multiplier": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _get_price_lookup(prices: pd.DataFrame) -> pd.Series:
    lookup = prices.copy()
    lookup["date"] = pd.to_datetime(lookup["date"])
    return lookup.set_index(["date", "security_id"])["price"]


def _build_initial_and_contribution_trades(
    prices: pd.DataFrame,
    fund: FundConfig,
) -> pd.DataFrame:
    price_lookup = _get_price_lookup(prices)
    trading_dates = pd.to_datetime(prices["date"]).sort_values().drop_duplicates()
    first_trade_date = trading_dates.iloc[0]
    month_starts = [
        date_value
        for date_value in _first_trading_day_by_month(trading_dates)
        if date_value > first_trade_date
    ]

    rows: list[dict[str, object]] = []
    for ticker in fund.tickers:
        price = float(price_lookup.loc[(first_trade_date, ticker)])
        quantity = 100.0
        rows.append(
            {
                "date": first_trade_date.strftime("%Y-%m-%d"),
                "fund_id": fund.fund_id,
                "security_id": ticker,
                "txn_type": "BUY",
                "quantity": quantity,
                "price": price,
                "cash_amount": round(-(quantity * price), 6),
                "fees": 0.0,
            }
        )

    for date_value in month_starts:
        rows.append(
            {
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "fund_id": fund.fund_id,
                "security_id": "",
                "txn_type": "CONTRIB",
                "quantity": 0.0,
                "price": 1.0,
                "cash_amount": 10000.0,
                "fees": 0.0,
            }
        )

    return pd.DataFrame(rows)


def _series_in_range(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if series.empty:
        return series
    series = series.copy()
    series.index = _to_naive_index(series.index)
    return series.loc[(series.index >= start) & (series.index <= end)]


def _build_dividend_actions(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            dividend_series = _series_in_range(yf.Ticker(ticker).dividends, start, end)
        except Exception as exc:
            warnings.warn(f"Dividend lookup failed for {ticker}: {exc}")
            dividend_series = pd.Series(dtype=float)

        for ex_date, value in dividend_series.items():
            rows.append(
                {
                    "ex_date": pd.Timestamp(ex_date).strftime("%Y-%m-%d"),
                    "security_id": ticker,
                    "action_type": "DIV",
                    "ratio": 1.0,
                    "cash_amount": float(value),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["ex_date", "security_id", "action_type", "ratio", "cash_amount"])
    return pd.DataFrame(rows)


def _build_dividend_blotter(
    dividend_actions: pd.DataFrame,
    fund: FundConfig,
) -> pd.DataFrame:
    if dividend_actions.empty:
        return pd.DataFrame(
            columns=["date", "fund_id", "security_id", "txn_type", "quantity", "price", "cash_amount", "fees"]
        )

    fund_dividends = dividend_actions.loc[dividend_actions["security_id"].isin(fund.tickers)].copy()
    if fund_dividends.empty:
        return pd.DataFrame(
            columns=["date", "fund_id", "security_id", "txn_type", "quantity", "price", "cash_amount", "fees"]
        )

    fund_dividends["date"] = pd.to_datetime(fund_dividends["ex_date"]).dt.strftime("%Y-%m-%d")
    fund_dividends["fund_id"] = fund.fund_id
    fund_dividends["txn_type"] = "DIV"
    fund_dividends["quantity"] = 100.0
    fund_dividends["price"] = fund_dividends["cash_amount"].astype(float)
    fund_dividends["cash_amount"] = fund_dividends["quantity"] * fund_dividends["price"]
    fund_dividends["fees"] = 0.0
    return fund_dividends[
        ["date", "fund_id", "security_id", "txn_type", "quantity", "price", "cash_amount", "fees"]
    ].copy()


def _build_split_rows(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            split_series = _series_in_range(yf.Ticker(ticker).splits, start, end)
        except Exception as exc:
            warnings.warn(f"Split lookup failed for {ticker}: {exc}")
            split_series = pd.Series(dtype=float)

        for ex_date, value in split_series.items():
            rows.append(
                {
                    "ex_date": pd.Timestamp(ex_date).strftime("%Y-%m-%d"),
                    "security_id": ticker,
                    "action_type": "SPLIT",
                    "ratio": float(value),
                    "cash_amount": 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["ex_date", "security_id", "action_type", "ratio", "cash_amount"])
    return pd.DataFrame(rows)


def _build_fx_rates(
    security_master: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_staleness_days: int,
    base_currency: str,
) -> pd.DataFrame:
    business_dates = pd.bdate_range(start=start, end=end)
    currencies = sorted(set(security_master["currency"].dropna().astype(str).str.upper()))
    rows: list[dict[str, object]] = []

    for currency in currencies:
        ccy_pair = f"{currency}{base_currency}"
        if currency == base_currency:
            rows.extend(
                {
                    "date": date_value.strftime("%Y-%m-%d"),
                    "ccy_pair": ccy_pair,
                    "rate": 1.0,
                    "source": "internal",
                }
                for date_value in business_dates
            )
            continue

        fx_ticker = f"{currency}{base_currency}=X"
        downloaded = yf.download(
            tickers=[fx_ticker],
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        fx_frame = _frame_from_download(downloaded, [fx_ticker])
        fx_series = fx_frame[fx_ticker].reindex(business_dates).ffill(limit=max_staleness_days).dropna()
        if fx_series.empty:
            raise ValueError(f"No FX data was available for {currency}/{base_currency}.")

        rows.extend(
            {
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "ccy_pair": ccy_pair,
                "rate": float(rate_value),
                "source": "yfinance",
            }
            for date_value, rate_value in fx_series.items()
        )

    return pd.DataFrame(rows)


def _fetch_sofr_rates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_api_key: str | None,
) -> tuple[pd.DataFrame, float | None]:
    if not fred_api_key:
        warnings.warn("No FRED API key was provided; SOFR update skipped.")
        return pd.DataFrame(columns=["date", "rate_annual"]), None

    try:
        response = requests.get(
            FRED_SOFR_URL,
            params={
                "series_id": "SOFR",
                "api_key": fred_api_key,
                "file_type": "json",
                "observation_start": start.strftime("%Y-%m-%d"),
                "observation_end": end.strftime("%Y-%m-%d"),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        warnings.warn(f"SOFR pull failed; policy cash rate left unchanged: {exc}")
        return pd.DataFrame(columns=["date", "rate_annual"]), None

    observations = payload.get("observations", [])
    rows = [
        {
            "date": observation["date"],
            "rate_annual": float(observation["value"]) / 100.0,
        }
        for observation in observations
        if observation.get("value") not in {".", None, ""}
    ]
    sofr_rates = pd.DataFrame(rows)
    live_sofr = None if sofr_rates.empty else float(sofr_rates.iloc[-1]["rate_annual"])
    return sofr_rates, live_sofr


def _seed_for_fund(fund_id: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(fund_id))


def _simulate_custodian_nav_for_fund(
    valuation_dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    blotter: pd.DataFrame,
    security_master: pd.DataFrame,
    fx_rates: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    policy: dict[str, Any],
    fund: FundConfig,
) -> pd.DataFrame:
    prices_frame = prices.copy()
    prices_frame["date"] = pd.to_datetime(prices_frame["date"])
    fund_blotter = blotter.loc[blotter["fund_id"].eq(fund.fund_id)].copy()
    trading_dates = [pd.Timestamp(date_value).normalize() for date_value in valuation_dates]

    positions_module.MAX_PRICE_STALENESS_DAYS = int(policy.get("max_price_staleness_days", 3))
    positions_module.PRICE_SOURCE_PRIORITY = tuple(
        policy.get("price_source_priority", ["internal", "vendor", "fallback"])
    )

    rng = np.random.default_rng(_seed_for_fund(fund.fund_id))
    nav_break_date = trading_dates[-1]
    cash_break_date = trading_dates[-2] if len(trading_dates) > 1 else trading_dates[-1]
    missing_position_date = trading_dates[-1]

    records: list[dict[str, object]] = []
    for asof_date in trading_dates:
        positions = build_positions(
            blotter=fund_blotter,
            prices=prices_frame,
            fx_rates=fx_rates,
            security_master=security_master,
            corporate_actions=corporate_actions,
            base_currency=str(policy.get("base_currency", "USD")),
            asof_date=asof_date,
        )
        nav = compute_daily_nav(
            positions=positions,
            blotter=fund_blotter,
            policy=policy,
            asof_date=asof_date,
        )
        nav_row = nav.loc[nav["fund_id"].eq(fund.fund_id)].iloc[0]

        noise_bps = rng.uniform(-8.0, 8.0)
        custodian_nav_value = float(nav_row["total_nav"]) * (1 + noise_bps / 10000.0)
        custodian_cash = float(nav_row["cash_balance"])

        if asof_date == nav_break_date:
            custodian_nav_value = float(nav_row["total_nav"]) * (1 + DEFAULT_NAV_BREAK_BPS / 10000.0)
        if asof_date == cash_break_date:
            custodian_cash = custodian_cash + DEFAULT_CASH_BREAK_AMOUNT

        position_rows = positions.loc[
            positions["fund_id"].eq(fund.fund_id),
            ["security_id", "quantity_eod", "price_local"],
        ].sort_values("security_id")
        custodian_positions = [
            {
                "security_id": row.security_id,
                "quantity": round(float(row.quantity_eod), 6),
                "price": round(float(row.price_local), 6),
            }
            for row in position_rows.itertuples(index=False)
        ]
        if asof_date == missing_position_date and custodian_positions:
            custodian_positions = custodian_positions[1:]

        records.append(
            {
                "date": pd.Timestamp(asof_date).strftime("%Y-%m-%d"),
                "fund_id": fund.fund_id,
                "custodian_nav": round(custodian_nav_value, 6),
                "custodian_cash": round(custodian_cash, 6),
                "custodian_positions": json.dumps(custodian_positions),
            }
        )

    return pd.DataFrame(records)


def _stringify_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    return output


def _resolve_fund_configs(args: argparse.Namespace, policy: dict[str, Any]) -> list[FundConfig]:
    benchmark_map = policy.get("fund_benchmark_map", {})
    if args.funds:
        raw_funds = json.loads(args.funds)
        if not isinstance(raw_funds, list) or not raw_funds:
            raise ValueError("--funds must be a non-empty JSON array.")

        fund_configs: list[FundConfig] = []
        for item in raw_funds:
            if not isinstance(item, dict):
                raise ValueError("Each --funds entry must be a JSON object.")
            fund_id = str(item.get("fund_id", "")).strip()
            tickers = [str(ticker).upper() for ticker in item.get("tickers", [])]
            benchmark = str(
                item.get("benchmark_ticker")
                or benchmark_map.get(fund_id)
                or "SPY"
            ).upper()
            if not fund_id or not tickers:
                raise ValueError("Each --funds entry must include fund_id and tickers.")
            fund_configs.append(
                FundConfig(
                    fund_id=fund_id,
                    tickers=tuple(dict.fromkeys(tickers)),
                    benchmark_ticker=benchmark,
                )
            )
        return fund_configs

    benchmark = str(args.benchmark_ticker or benchmark_map.get(args.fund_id) or "SPY").upper()
    return [
        FundConfig(
            fund_id=str(args.fund_id).strip(),
            tickers=tuple(dict.fromkeys(str(ticker).upper() for ticker in args.tickers)),
            benchmark_ticker=benchmark,
        )
    ]


def _print_summary(
    frames: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    sofr_rate: float | None,
) -> None:
    summary_rows = [
        ("prices.csv", len(frames["prices.csv"])),
        ("trade_blotter.csv", len(frames["trade_blotter.csv"])),
        ("security_master.csv", len(frames["security_master.csv"])),
        ("custodian_nav.csv", len(frames["custodian_nav.csv"])),
        ("fx_rates.csv", len(frames["fx_rates.csv"])),
        ("corporate_actions.csv", len(frames["corporate_actions.csv"])),
        ("sofr_rates.csv", len(frames["sofr_rates.csv"])),
        ("Date range", f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"),
        ("SOFR rate used", "unchanged" if sofr_rate is None else f"{sofr_rate:.4f}"),
        ("Intentional breaks", 2),
    ]

    print(f"{'File':<22}Rows")
    for label, value in summary_rows:
        print(f"{label:<22}{value}")


def main() -> None:
    args = _parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    out_dir = Path(args.out_dir)

    policy = _load_policy()
    fund_configs = _resolve_fund_configs(args, policy)
    valuation_dates = pd.bdate_range(start=start, end=end)
    max_staleness_days = int(policy.get("max_price_staleness_days", 3))
    base_currency = str(policy.get("base_currency", "USD"))

    all_tickers = sorted({ticker for fund in fund_configs for ticker in fund.tickers})
    prices = _download_prices(all_tickers, start, end, max_staleness_days)
    security_master = _build_security_master(all_tickers)
    dividend_actions = _build_dividend_actions(all_tickers, start, end)
    split_actions = _build_split_rows(all_tickers, start, end)
    corporate_actions = (
        pd.concat([split_actions, dividend_actions], ignore_index=True)
        .sort_values(["ex_date", "security_id", "action_type"])
        .reset_index(drop=True)
    )

    trade_frames: list[pd.DataFrame] = []
    for fund in fund_configs:
        initial_and_contrib = _build_initial_and_contribution_trades(prices, fund)
        dividend_blotter = _build_dividend_blotter(dividend_actions, fund)
        fund_blotter = (
            pd.concat([initial_and_contrib, dividend_blotter], ignore_index=True)
            .sort_values(["date", "txn_type", "security_id"], na_position="last")
            .reset_index(drop=True)
        )
        trade_frames.append(fund_blotter)

    trade_blotter = pd.concat(trade_frames, ignore_index=True)
    fx_rates = _build_fx_rates(security_master, start, end, max_staleness_days, base_currency)
    sofr_rates, live_sofr = _fetch_sofr_rates(start, end, args.fred_api_key)

    working_policy = dict(policy)
    working_policy.setdefault("fund_benchmark_map", {})
    working_policy.setdefault("sofr_rate_source", "FRED")
    for fund in fund_configs:
        working_policy["fund_benchmark_map"][fund.fund_id] = fund.benchmark_ticker
    if live_sofr is not None:
        working_policy["cash_interest_rate"] = round(live_sofr, 6)

    custodian_frames = [
        _simulate_custodian_nav_for_fund(
            valuation_dates=valuation_dates,
            prices=prices,
            blotter=trade_blotter,
            security_master=security_master,
            fx_rates=fx_rates,
            corporate_actions=corporate_actions,
            policy=working_policy,
            fund=fund,
        )
        for fund in fund_configs
    ]
    custodian_nav = pd.concat(custodian_frames, ignore_index=True).sort_values(["date", "fund_id"]).reset_index(drop=True)

    export_frames = {
        "prices.csv": _stringify_dates(prices, ["date"]),
        "trade_blotter.csv": _stringify_dates(trade_blotter, ["date"]),
        "security_master.csv": security_master.copy(),
        "custodian_nav.csv": _stringify_dates(custodian_nav, ["date"]),
        "fx_rates.csv": _stringify_dates(fx_rates, ["date"]),
        "corporate_actions.csv": _stringify_dates(corporate_actions, ["ex_date"]),
        "sofr_rates.csv": _stringify_dates(sofr_rates, ["date"]),
    }

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for file_name, frame in export_frames.items():
            frame.to_csv(out_dir / file_name, index=False)
        if live_sofr is not None or working_policy.get("fund_benchmark_map") != policy.get("fund_benchmark_map"):
            _write_policy(working_policy)

    _print_summary(export_frames, start, end, live_sofr)
    if args.dry_run:
        print("Dry run only - no files written.")


if __name__ == "__main__":
    main()
