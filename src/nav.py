from __future__ import annotations

import pandas as pd


INITIAL_SHARES_OUTSTANDING = 1_000_000.0
INITIAL_NAV_PER_SHARE = 1.0


def _empty_nav_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "fund_id",
            "securities_mv",
            "cash_balance",
            "accrued_income",
            "total_nav",
            "shares_outstanding",
            "nav_per_share",
        ]
    )


def _resolve_shares_outstanding(total_nav: float, net_subscriptions: float) -> tuple[float, float]:
    shares_outstanding = INITIAL_SHARES_OUTSTANDING
    for _ in range(8):
        nav_per_share = total_nav / shares_outstanding if shares_outstanding else INITIAL_NAV_PER_SHARE
        nav_per_share = max(nav_per_share, 0.01)
        updated_shares = INITIAL_SHARES_OUTSTANDING + (net_subscriptions / nav_per_share)
        if abs(updated_shares - shares_outstanding) < 1e-6:
            shares_outstanding = updated_shares
            break
        shares_outstanding = updated_shares

    nav_per_share = total_nav / shares_outstanding if shares_outstanding else 0.0
    return shares_outstanding, nav_per_share


def _is_business_day(asof_date: pd.Timestamp) -> bool:
    return pd.Timestamp(asof_date).weekday() < 5


def _compute_accrued_income(
    cash_balance: float,
    annual_rate: float,
    asof_date: pd.Timestamp,
) -> float:
    if annual_rate <= 0.0 or not _is_business_day(asof_date):
        return 0.0
    return cash_balance * annual_rate / 365.0


def compute_daily_nav(
    positions: pd.DataFrame,
    blotter: pd.DataFrame,
    policy: dict,
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    asof = pd.Timestamp(asof_date).normalize()

    positions_frame = positions.copy()
    blotter_frame = blotter.copy()

    if positions_frame.empty and blotter_frame.empty:
        return _empty_nav_frame()

    if not positions_frame.empty:
        positions_frame["market_value_base"] = positions_frame["market_value_base"].astype(float)

    if not blotter_frame.empty:
        blotter_frame["date"] = pd.to_datetime(blotter_frame["date"])
        blotter_frame = blotter_frame.loc[blotter_frame["date"] <= asof].copy()
        blotter_frame["cash_amount"] = blotter_frame["cash_amount"].fillna(0.0).astype(float)
        blotter_frame["fees"] = blotter_frame["fees"].fillna(0.0).astype(float)
        blotter_frame["net_cash"] = blotter_frame["cash_amount"] - blotter_frame["fees"]

    funds = set()
    if not positions_frame.empty:
        funds.update(positions_frame["fund_id"].dropna().tolist())
    if not blotter_frame.empty:
        funds.update(blotter_frame["fund_id"].dropna().tolist())
    if not funds:
        funds.add(policy.get("fund_id_default", "FUND1"))

    securities_mv = (
        positions_frame.groupby("fund_id")["market_value_base"].sum()
        if not positions_frame.empty
        else pd.Series(dtype=float)
    )
    cash_flows = (
        blotter_frame.groupby("fund_id")["net_cash"].sum()
        if not blotter_frame.empty
        else pd.Series(dtype=float)
    )
    subscription_flows = (
        blotter_frame.loc[blotter_frame["txn_type"].isin(["CONTRIB", "WITHDRAW"])]
        .groupby("fund_id")["cash_amount"]
        .sum()
        if not blotter_frame.empty
        else pd.Series(dtype=float)
    )

    rate = float(policy.get("cash_interest_rate", 0.0))
    records: list[dict[str, float | str]] = []
    for fund_id in sorted(funds):
        securities_value = float(securities_mv.get(fund_id, 0.0))
        cash_balance = (INITIAL_SHARES_OUTSTANDING * INITIAL_NAV_PER_SHARE) + float(
            cash_flows.get(fund_id, 0.0)
        )
        accrued_income = _compute_accrued_income(cash_balance, rate, asof)
        total_nav = securities_value + cash_balance + accrued_income
        net_subscriptions = float(subscription_flows.get(fund_id, 0.0))
        shares_outstanding, nav_per_share = _resolve_shares_outstanding(total_nav, net_subscriptions)

        records.append(
            {
                "date": asof.strftime("%Y-%m-%d"),
                "fund_id": fund_id,
                "securities_mv": securities_value,
                "cash_balance": cash_balance,
                "accrued_income": accrued_income,
                "total_nav": total_nav,
                "shares_outstanding": shares_outstanding,
                "nav_per_share": nav_per_share,
            }
        )

    return pd.DataFrame.from_records(records)
