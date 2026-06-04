from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


PRICE_SOURCE_PRIORITY: tuple[str, ...] = ("internal", "vendor", "fallback")
MAX_PRICE_STALENESS_DAYS = 3


def _empty_positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "fund_id",
            "security_id",
            "quantity_eod",
            "price_local",
            "security_currency",
            "fx_to_base",
            "market_value_base",
            "corporate_action_flag",
        ]
    )


def _empty_blotter_like(blotter: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=list(blotter.columns))


def _source_rank(values: pd.Series) -> pd.Series:
    rank_map = {name: idx for idx, name in enumerate(PRICE_SOURCE_PRIORITY)}
    return values.map(rank_map).fillna(len(rank_map)).astype(int)


def _append_flag(current_flag: str, flag: str) -> str:
    if not current_flag:
        return flag
    existing = {item.strip() for item in current_flag.split("|") if item.strip()}
    if flag in existing:
        return current_flag
    return f"{current_flag}|{flag}"


def _normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


def _latest_price_snapshot(prices: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    price_frame = prices.loc[prices["date"] <= asof_date].copy()
    if price_frame.empty:
        return pd.DataFrame(columns=["security_id", "price_local", "price_age_days"])

    price_frame["source_rank"] = _source_rank(price_frame["source"])
    price_frame = price_frame.sort_values(
        ["security_id", "date", "source_rank"],
        ascending=[True, False, True],
    )
    price_frame = price_frame.drop_duplicates(subset=["security_id"], keep="first")
    price_frame["price_age_days"] = (asof_date - price_frame["date"]).dt.days
    return price_frame.rename(columns={"price": "price_local"})[
        ["security_id", "price_local", "price_age_days"]
    ]


def _latest_fx_snapshot(
    fx_rates: pd.DataFrame,
    base_currency: str,
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    fx_frame = fx_rates.loc[fx_rates["date"] <= asof_date].copy()
    if fx_frame.empty:
        return pd.DataFrame(columns=["security_currency", "fx_to_base"])

    fx_frame = fx_frame.loc[fx_frame["ccy_pair"].str.endswith(base_currency)].copy()
    if fx_frame.empty:
        return pd.DataFrame(columns=["security_currency", "fx_to_base"])

    fx_frame["security_currency"] = fx_frame["ccy_pair"].str.replace(
        base_currency,
        "",
        regex=False,
    )
    fx_frame["source_rank"] = _source_rank(fx_frame["source"])
    fx_frame = fx_frame.sort_values(
        ["security_currency", "date", "source_rank"],
        ascending=[True, False, True],
    )
    fx_frame = fx_frame.drop_duplicates(subset=["security_currency"], keep="first")
    return fx_frame.rename(columns={"rate": "fx_to_base"})[
        ["security_currency", "fx_to_base"]
    ]


def _build_action_flag_frame(corporate_actions: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    if corporate_actions.empty:
        return pd.DataFrame(columns=["security_id", "corporate_action_flag"])

    same_day_actions = corporate_actions.loc[
        corporate_actions["ex_date"].eq(asof_date)
        & corporate_actions["action_type"].isin(["SPLIT", "DIV", "MERGER"])
    ].copy()
    if same_day_actions.empty:
        return pd.DataFrame(columns=["security_id", "corporate_action_flag"])

    return (
        same_day_actions.groupby("security_id", as_index=False)["action_type"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename(columns={"action_type": "corporate_action_flag"})
    )


def _warn_on_mergers(corporate_actions: pd.DataFrame, asof_date: pd.Timestamp) -> None:
    merger_actions = corporate_actions.loc[
        corporate_actions["ex_date"].eq(asof_date)
        & corporate_actions["action_type"].eq("MERGER")
    ]
    for row in merger_actions.itertuples(index=False):
        warnings.warn(
            f"MERGER action detected for {row.security_id} on {row.ex_date.strftime('%Y-%m-%d')} - manual review required",
            stacklevel=2,
        )


def build_corporate_action_cash_flows(
    blotter: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    fund_id: str,
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    blotter_frame = blotter.copy()
    corporate_actions_frame = corporate_actions.copy()
    asof = pd.Timestamp(asof_date).normalize()

    if blotter_frame.empty or corporate_actions_frame.empty:
        return _empty_blotter_like(blotter)

    blotter_frame["date"] = _normalize_datetime(blotter_frame["date"])
    corporate_actions_frame["ex_date"] = _normalize_datetime(corporate_actions_frame["ex_date"])

    dividend_actions = corporate_actions_frame.loc[
        (corporate_actions_frame["action_type"] == "DIV")
        & (corporate_actions_frame["ex_date"] <= asof)
    ].copy()
    if dividend_actions.empty:
        return _empty_blotter_like(blotter)

    trades = blotter_frame.loc[
        blotter_frame["txn_type"].isin(["BUY", "SELL"]),
        ["date", "security_id", "txn_type", "quantity"],
    ].copy()
    if trades.empty:
        return _empty_blotter_like(blotter)

    trades["signed_quantity"] = np.where(trades["txn_type"].eq("BUY"), trades["quantity"], -trades["quantity"])
    quantity_history = (
        trades.groupby(["security_id", "date"], as_index=False)["signed_quantity"]
        .sum()
        .sort_values(["security_id", "date"])
    )
    quantity_history["cum_quantity"] = quantity_history.groupby("security_id")["signed_quantity"].cumsum()

    split_actions = corporate_actions_frame.loc[
        (corporate_actions_frame["action_type"] == "SPLIT")
        & (corporate_actions_frame["ex_date"] <= asof)
    ].copy()
    if split_actions.empty:
        split_history = pd.DataFrame(columns=["security_id", "ex_date", "cum_split"])
    else:
        split_history = (
            split_actions.groupby(["security_id", "ex_date"], as_index=False)["ratio"]
            .prod()
            .sort_values(["security_id", "ex_date"])
        )
        split_history["cum_split"] = split_history.groupby("security_id")["ratio"].cumprod()

    dividend_actions = dividend_actions.sort_values(["security_id", "ex_date"])
    quantity_history = quantity_history.rename(columns={"date": "event_date"})
    split_history = split_history.rename(columns={"ex_date": "split_date"})

    dividend_actions = pd.merge_asof(
        dividend_actions.sort_values(["ex_date", "security_id"]),
        quantity_history.sort_values(["event_date", "security_id"]),
        left_on="ex_date",
        right_on="event_date",
        by="security_id",
        direction="backward",
    )
    if split_history.empty:
        dividend_actions["cum_split"] = 1.0
    else:
        dividend_actions = pd.merge_asof(
            dividend_actions.sort_values(["ex_date", "security_id"]),
            split_history.sort_values(["split_date", "security_id"]),
            left_on="ex_date",
            right_on="split_date",
            by="security_id",
            direction="backward",
        )

    dividend_actions["cum_quantity"] = dividend_actions["cum_quantity"].fillna(0.0)
    dividend_actions["cum_split"] = dividend_actions["cum_split"].fillna(1.0)
    dividend_actions["held_quantity"] = dividend_actions["cum_quantity"] * dividend_actions["cum_split"]

    synthetic_flows = dividend_actions.loc[dividend_actions["held_quantity"] > 0].copy()
    if synthetic_flows.empty:
        return _empty_blotter_like(blotter)

    synthetic_flows["date"] = synthetic_flows["ex_date"].dt.strftime("%Y-%m-%d")
    synthetic_flows["fund_id"] = fund_id
    synthetic_flows["txn_type"] = "DIV"
    synthetic_flows["quantity"] = synthetic_flows["held_quantity"]
    synthetic_flows["price"] = synthetic_flows["cash_amount"]
    synthetic_flows["cash_amount"] = synthetic_flows["held_quantity"] * synthetic_flows["cash_amount"]
    synthetic_flows["fees"] = 0.0

    synthetic_flows = synthetic_flows[
        ["date", "fund_id", "security_id", "txn_type", "quantity", "price", "cash_amount", "fees"]
    ].copy()

    existing_dividends = blotter_frame.loc[
        blotter_frame["txn_type"].eq("DIV"),
        ["date", "fund_id", "security_id"],
    ].drop_duplicates()
    existing_dividends["existing"] = True
    synthetic_flows["date"] = pd.to_datetime(synthetic_flows["date"]).dt.normalize()
    synthetic_flows = synthetic_flows.merge(
        existing_dividends,
        on=["date", "fund_id", "security_id"],
        how="left",
    )
    synthetic_flows = synthetic_flows.loc[synthetic_flows["existing"].isna()].drop(columns=["existing"])
    synthetic_flows["date"] = pd.to_datetime(synthetic_flows["date"]).dt.strftime("%Y-%m-%d")
    return synthetic_flows.reset_index(drop=True)


def build_positions(
    blotter: pd.DataFrame,
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    security_master: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    base_currency: str,
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    asof = pd.Timestamp(asof_date).normalize()

    blotter_frame = blotter.copy()
    prices_frame = prices.copy()
    fx_frame = fx_rates.copy()
    master_frame = security_master.copy()
    corporate_actions_frame = corporate_actions.copy()

    if blotter_frame.empty:
        return _empty_positions_frame()

    blotter_frame["date"] = _normalize_datetime(blotter_frame["date"])
    prices_frame["date"] = _normalize_datetime(prices_frame["date"])
    fx_frame["date"] = _normalize_datetime(fx_frame["date"])
    if not corporate_actions_frame.empty:
        corporate_actions_frame["ex_date"] = _normalize_datetime(corporate_actions_frame["ex_date"])
    else:
        corporate_actions_frame["ex_date"] = pd.to_datetime([])

    trades = blotter_frame.loc[
        (blotter_frame["date"] <= asof) & blotter_frame["txn_type"].isin(["BUY", "SELL"])
    ].copy()
    if trades.empty:
        return _empty_positions_frame()

    trades["signed_quantity"] = np.where(
        trades["txn_type"].eq("BUY"),
        trades["quantity"],
        -trades["quantity"],
    )

    positions = (
        trades.groupby(["fund_id", "security_id"], as_index=False)["signed_quantity"]
        .sum()
        .rename(columns={"signed_quantity": "quantity_eod"})
    )
    positions = positions.loc[positions["quantity_eod"].abs() > 1e-9].copy()
    if positions.empty:
        return _empty_positions_frame()

    split_actions = corporate_actions_frame.loc[
        (corporate_actions_frame["action_type"] == "SPLIT")
        & (corporate_actions_frame["ex_date"] <= asof)
    ].copy()
    split_ratios = (
        split_actions.groupby("security_id", as_index=False)["ratio"]
        .prod()
        .rename(columns={"ratio": "split_ratio"})
        if not split_actions.empty
        else pd.DataFrame(columns=["security_id", "split_ratio"])
    )
    positions = positions.merge(split_ratios, on="security_id", how="left")
    positions["split_ratio"] = pd.to_numeric(positions["split_ratio"], errors="coerce").fillna(1.0)
    positions["quantity_eod"] = positions["quantity_eod"] * positions["split_ratio"]

    positions = positions.merge(
        master_frame[
            ["security_id", "currency", "asset_class", "multiplier"]
        ].rename(columns={"currency": "security_currency"}),
        on="security_id",
        how="left",
    )
    positions["security_currency"] = positions["security_currency"].fillna(base_currency)
    positions["multiplier"] = positions["multiplier"].fillna(1.0)

    latest_prices = _latest_price_snapshot(prices_frame, asof)
    positions = positions.merge(latest_prices, on="security_id", how="left")
    positions["price_local"] = positions["price_local"].fillna(0.0)

    same_day_splits = (
        split_actions.loc[split_actions["ex_date"].eq(asof), ["security_id", "ratio"]]
        .groupby("security_id", as_index=False)["ratio"]
        .prod()
        .rename(columns={"ratio": "same_day_split_ratio"})
        if not split_actions.empty
        else pd.DataFrame(columns=["security_id", "same_day_split_ratio"])
    )
    positions = positions.merge(same_day_splits, on="security_id", how="left")
    positions["same_day_split_ratio"] = pd.to_numeric(
        positions["same_day_split_ratio"],
        errors="coerce",
    ).fillna(1.0)
    split_price_mask = positions["same_day_split_ratio"] != 1.0
    positions.loc[split_price_mask, "price_local"] = (
        positions.loc[split_price_mask, "price_local"]
        / positions.loc[split_price_mask, "same_day_split_ratio"]
    )

    latest_fx = _latest_fx_snapshot(fx_frame, base_currency, asof)
    positions = positions.merge(latest_fx, on="security_currency", how="left")
    positions["fx_to_base"] = np.where(
        positions["security_currency"].eq(base_currency),
        1.0,
        positions["fx_to_base"],
    )
    positions["fx_to_base"] = positions["fx_to_base"].fillna(1.0)

    positions["market_value_base"] = (
        positions["quantity_eod"]
        * positions["price_local"]
        * positions["multiplier"]
        * positions["fx_to_base"]
    )

    action_flags = _build_action_flag_frame(corporate_actions_frame, asof)
    positions = positions.merge(action_flags, on="security_id", how="left")
    positions["corporate_action_flag"] = positions["corporate_action_flag"].fillna("")

    stale_mask = (
        positions["price_age_days"].fillna(MAX_PRICE_STALENESS_DAYS + 1) > MAX_PRICE_STALENESS_DAYS
    ) | (positions["price_local"] <= 0)
    positions.loc[stale_mask, "corporate_action_flag"] = positions.loc[
        stale_mask,
        "corporate_action_flag",
    ].map(lambda value: _append_flag(value, "STALE_PRICE"))

    _warn_on_mergers(corporate_actions_frame, asof)

    positions["date"] = asof.strftime("%Y-%m-%d")
    output = positions[
        [
            "date",
            "fund_id",
            "security_id",
            "quantity_eod",
            "price_local",
            "security_currency",
            "fx_to_base",
            "market_value_base",
            "corporate_action_flag",
        ]
    ].copy()
    output = output.sort_values(["fund_id", "security_id"]).reset_index(drop=True)
    return output
