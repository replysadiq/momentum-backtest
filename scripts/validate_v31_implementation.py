#!/usr/bin/env python3
"""
V3.1 Implementation Validation Script

Performs three critical checks:
1. Confirm defensive CASH uses stock portfolio returns, not benchmark
2. Confirm defensive portfolio is not accidentally Mom50
3. Attribution split by state (CAGR contribution, turnover)

Usage:
    python scripts/validate_v31_implementation.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random


def load_data(output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load all required data from output directory."""
    output_path = Path(output_dir)

    # Load equity curve
    equity_df = pd.read_csv(output_path / "equity_curve.csv", parse_dates=["date"])
    equity_df.set_index("date", inplace=True)

    # Load state log
    state_log = pd.read_csv(output_path / "state_log.csv", parse_dates=["date"])

    # Load rebalance log
    rebalance_log = pd.read_csv(output_path / "rebalance_log.csv", parse_dates=["date"])

    # Load metrics
    with open(output_path / "metrics.json") as f:
        metrics = json.load(f)

    return equity_df, state_log, rebalance_log, metrics


def load_price_data(parquet_path: str) -> pd.DataFrame:
    """Load price data from parquet and pivot to wide format."""
    df = pd.read_parquet(parquet_path)
    # Data is in long format: date, symbol, open, high, low, close, adj_close, volume
    # Pivot to wide format with symbols as columns
    price_wide = df.pivot(index="date", columns="symbol", values="adj_close")
    price_wide.index = pd.to_datetime(price_wide.index)
    return price_wide


def load_mom50_benchmark(csv_path: str) -> pd.Series:
    """Load Mom50 benchmark data."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df["Close"]


def check_1_portfolio_returns(
    output_dir: str,
    parquet_path: str,
    n_samples: int = 10
) -> None:
    """
    Check 1: Confirm defensive CASH uses stock portfolio returns.

    Pick random days inside CASH invested periods and verify
    the equity curve return matches computed portfolio return.
    """
    print("\n" + "="*70)
    print("CHECK 1: Defensive CASH uses stock portfolio returns")
    print("="*70)

    equity_df, state_log, rebalance_log, _ = load_data(output_dir)
    price_data = load_price_data(parquet_path)

    # Find CASH periods with invested_flag=True
    cash_invested = state_log[
        (state_log["state"] == "CASH") &
        (state_log["invested_flag"] == True)
    ]

    if len(cash_invested) == 0:
        print("ERROR: No CASH invested periods found!")
        return

    print(f"Found {len(cash_invested)} CASH invested rebalance periods")

    # Get rebalance dates with holdings
    rebalance_dates = rebalance_log[rebalance_log["state"] == "CASH"]["date"].tolist()

    # Build a map of rebalance_date -> (tickers, weights)
    holdings_map = {}
    for _, row in rebalance_log[rebalance_log["state"] == "CASH"].iterrows():
        if row["n_stocks"] > 0 and pd.notna(row["tickers"]) and row["tickers"]:
            tickers = row["tickers"].split(",")
            weights = [float(w) for w in row["weights"].split(",")]
            holdings_map[row["date"]] = dict(zip(tickers, weights))

    if not holdings_map:
        print("ERROR: No CASH periods have holdings!")
        return

    print(f"Found {len(holdings_map)} CASH periods with defensive holdings")

    # Sample random days within CASH invested periods
    # Find date ranges where we're in CASH with investments
    cash_date_ranges = []
    sorted_rebal_dates = sorted(holdings_map.keys())

    for i, rebal_date in enumerate(sorted_rebal_dates):
        # Find next rebalance date (could be any state)
        next_rebal = None
        all_rebal_dates = sorted(rebalance_log["date"].tolist())
        for d in all_rebal_dates:
            if d > rebal_date:
                next_rebal = d
                break

        if next_rebal:
            cash_date_ranges.append((rebal_date, next_rebal, holdings_map[rebal_date]))

    # Pick random days from within these ranges
    all_sample_days = []
    for start_date, end_date, weights in cash_date_ranges:
        # Get trading days in this range from equity curve
        mask = (equity_df.index > start_date) & (equity_df.index < end_date)
        days_in_range = equity_df.index[mask].tolist()
        if len(days_in_range) > 1:
            for d in days_in_range[:3]:  # Take a few from each period
                all_sample_days.append((d, weights))

    if len(all_sample_days) < n_samples:
        n_samples = len(all_sample_days)

    sample_days = random.sample(all_sample_days, min(n_samples, len(all_sample_days)))

    print(f"\nValidating {len(sample_days)} random days:")
    print("-"*70)

    mismatches = 0
    for day, weights in sample_days:
        # Get equity curve return for this day
        day_idx = equity_df.index.get_loc(day)
        if day_idx == 0:
            continue

        prev_day = equity_df.index[day_idx - 1]
        equity_return = (equity_df.loc[day, "portfolio_equity"] /
                        equity_df.loc[prev_day, "portfolio_equity"]) - 1

        # Compute portfolio return from stock prices
        computed_return = 0.0
        for ticker, weight in weights.items():
            # Strip .NS suffix if present (rebalance_log has .NS, price_data doesn't)
            clean_ticker = ticker.replace('.NS', '')
            if clean_ticker in price_data.columns:
                try:
                    price_today = price_data.loc[day, clean_ticker]
                    price_prev = price_data.loc[prev_day, clean_ticker]
                    if pd.notna(price_today) and pd.notna(price_prev) and price_prev > 0:
                        stock_return = (price_today / price_prev) - 1
                        computed_return += weight * stock_return
                except KeyError:
                    pass  # Day not in price data

        diff = abs(equity_return - computed_return)
        status = "OK" if diff < 0.001 else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1

        print(f"  {day.strftime('%Y-%m-%d')}: equity={equity_return:+.4%}, "
              f"computed={computed_return:+.4%}, diff={diff:.6f} [{status}]")

    print("-"*70)
    if mismatches == 0:
        print("CHECK 1 PASSED: Portfolio returns match equity curve")
    else:
        print(f"CHECK 1 WARNING: {mismatches}/{len(sample_days)} days have mismatches")
        print("  (Small differences may be due to transaction costs or timing)")


def check_2_not_mom50(output_dir: str, mom50_path: str) -> None:
    """
    Check 2: Confirm defensive portfolio is not accidentally Mom50.

    Compute correlation between strategy and Mom50 daily returns.
    If correlation ≈ 1.0, something might be wrong.
    """
    print("\n" + "="*70)
    print("CHECK 2: Defensive portfolio is NOT accidentally Mom50")
    print("="*70)

    equity_df, state_log, _, _ = load_data(output_dir)
    mom50 = load_mom50_benchmark(mom50_path)

    # Compute daily returns
    strategy_returns = equity_df["portfolio_equity"].pct_change().dropna()
    mom50_returns = mom50.pct_change().dropna()

    # Align dates
    common_dates = strategy_returns.index.intersection(mom50_returns.index)
    strat_aligned = strategy_returns.reindex(common_dates)
    mom50_aligned = mom50_returns.reindex(common_dates)

    # Overall correlation
    overall_corr = strat_aligned.corr(mom50_aligned)

    # Tracking error (annualized)
    diff = strat_aligned - mom50_aligned
    tracking_error = diff.std() * np.sqrt(252)

    print(f"\nOverall Statistics (all days):")
    print(f"  Correlation with Mom50:    {overall_corr:.4f}")
    print(f"  Tracking Error (annual):   {tracking_error:.2%}")

    # Now compute separately for CASH-invested periods
    cash_invested_dates = state_log[
        (state_log["state"] == "CASH") &
        (state_log["invested_flag"] == True)
    ]["date"].tolist()

    # Expand to all days in CASH periods
    all_rebal_dates = sorted(state_log["date"].tolist())
    cash_invested_day_ranges = []

    for i, rebal_date in enumerate(all_rebal_dates):
        if rebal_date in cash_invested_dates:
            # Find next rebalance
            next_rebal = None
            for d in all_rebal_dates:
                if d > rebal_date:
                    next_rebal = d
                    break
            if next_rebal:
                cash_invested_day_ranges.append((rebal_date, next_rebal))

    # Get all trading days in CASH invested periods
    cash_days = []
    for start, end in cash_invested_day_ranges:
        mask = (strat_aligned.index > start) & (strat_aligned.index <= end)
        cash_days.extend(strat_aligned.index[mask].tolist())

    if cash_days:
        cash_strat = strat_aligned.loc[strat_aligned.index.isin(cash_days)]
        cash_mom50 = mom50_aligned.loc[mom50_aligned.index.isin(cash_days)]

        # Align again
        common = cash_strat.index.intersection(cash_mom50.index)
        cash_strat = cash_strat.reindex(common)
        cash_mom50 = cash_mom50.reindex(common)

        if len(common) > 10:
            cash_corr = cash_strat.corr(cash_mom50)
            cash_te = (cash_strat - cash_mom50).std() * np.sqrt(252)

            print(f"\nCASH-Invested Periods Only ({len(common)} days):")
            print(f"  Correlation with Mom50:    {cash_corr:.4f}")
            print(f"  Tracking Error (annual):   {cash_te:.2%}")

    print("\n" + "-"*70)
    if overall_corr > 0.99 and tracking_error < 0.01:
        print("CHECK 2 WARNING: Correlation ≈ 1.0 - may be replicating Mom50!")
    elif overall_corr > 0.95:
        print("CHECK 2 NOTE: High correlation with Mom50 (expected for momentum strategy)")
    else:
        print("CHECK 2 PASSED: Strategy is distinct from Mom50")


def check_3_attribution_by_state(
    output_dir_baseline: str,
    output_dir_defensive: str
) -> None:
    """
    Check 3: Attribution split by state.

    Compute CAGR contribution and turnover by state.
    """
    print("\n" + "="*70)
    print("CHECK 3: Attribution Split by State")
    print("="*70)

    # Load defensive mode data
    equity_df, state_log, rebalance_log, metrics = load_data(output_dir_defensive)

    # Get state for each day by forward-filling from rebalance dates
    state_log_sorted = state_log.sort_values("date")

    # Create a daily state series
    daily_states = pd.DataFrame(index=equity_df.index)
    daily_states["state"] = None
    daily_states["invested_flag"] = None

    for i, row in state_log_sorted.iterrows():
        # Apply state to all days from this rebalance until next
        mask = daily_states.index >= row["date"]
        daily_states.loc[mask, "state"] = row["state"]
        daily_states.loc[mask, "invested_flag"] = row["invested_flag"]

    # Compute daily returns
    daily_returns = equity_df["portfolio_equity"].pct_change()
    daily_states["return"] = daily_returns

    # Group by state
    state_groups = daily_states.groupby("state")

    print("\nDaily Return Statistics by State:")
    print("-"*70)

    for state_name, group in state_groups:
        if len(group) == 0:
            continue

        n_days = len(group)
        mean_daily_ret = group["return"].mean()
        std_daily_ret = group["return"].std()

        # Annualized CAGR contribution estimate
        # Using compound return for the period
        total_return = (1 + group["return"].fillna(0)).prod() - 1
        years = n_days / 252
        if years > 0 and total_return > -1:
            period_cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        else:
            period_cagr = 0

        pct_time = n_days / len(daily_states) * 100

        print(f"  {state_name:20s}: {n_days:4d} days ({pct_time:5.1f}%), "
              f"mean={mean_daily_ret*100:+.3f}%/day, "
              f"period_return={total_return*100:+.1f}%")

    # Turnover by state from rebalance log
    print("\nTurnover by State:")
    print("-"*70)

    turnover_by_state = rebalance_log.groupby("state")["turnover"].sum()
    total_turnover = turnover_by_state.sum()

    for state_name, turnover in turnover_by_state.items():
        pct_turnover = turnover / total_turnover * 100 if total_turnover > 0 else 0
        print(f"  {state_name:20s}: {turnover:6.2f}x ({pct_turnover:5.1f}% of total)")

    print(f"  {'TOTAL':20s}: {total_turnover:6.2f}x")

    # Compare with metrics
    print("\nMetrics Comparison (from metrics.json):")
    print("-"*70)
    print(f"  time_in_risk_on:         {metrics.get('time_in_risk_on', 0)*100:.1f}%")
    print(f"  time_in_defensive_mom:   {metrics.get('time_in_defensive_momentum', 0)*100:.1f}%")
    print(f"  time_in_cash:            {metrics.get('time_in_cash', 0)*100:.1f}%")
    print(f"  time_in_cash_invested:   {metrics.get('time_in_cash_invested', 0)*100:.1f}%")
    print(f"  turnover_risk_on:        {metrics.get('turnover_risk_on', 0):.2f}x")
    print(f"  turnover_defensive:      {metrics.get('turnover_defensive', 0):.2f}x")
    print(f"  turnover_cash_panic:     {metrics.get('turnover_cash_panic', 0):.2f}x")

    # Compute CAGR attribution
    print("\n" + "-"*70)
    print("CAGR Attribution Analysis:")
    print("-"*70)

    # RISK_ON contribution
    risk_on_days = daily_states[daily_states["state"] == "RISK_ON"]
    risk_on_return = (1 + risk_on_days["return"].fillna(0)).prod() - 1

    # DEFENSIVE_MOMENTUM contribution
    def_mom_days = daily_states[daily_states["state"] == "DEFENSIVE_MOMENTUM"]
    def_mom_return = (1 + def_mom_days["return"].fillna(0)).prod() - 1 if len(def_mom_days) > 0 else 0

    # CASH (invested) contribution
    cash_days = daily_states[daily_states["state"] == "CASH"]
    cash_return = (1 + cash_days["return"].fillna(0)).prod() - 1 if len(cash_days) > 0 else 0

    # Total
    total_return = (1 + daily_returns.fillna(0)).prod() - 1

    print(f"  RISK_ON cumulative return:         {risk_on_return*100:+.1f}%")
    print(f"  DEFENSIVE_MOMENTUM cum return:     {def_mom_return*100:+.1f}%")
    print(f"  CASH (invested) cumulative return: {cash_return*100:+.1f}%")
    print(f"  ---")
    print(f"  Total cumulative return:           {total_return*100:+.1f}%")

    # Note: Returns don't simply add - they compound
    # But this gives a rough sense of contribution

    print("\n" + "-"*70)
    if cash_return > 0:
        print("CHECK 3 INSIGHT: CASH-invested periods contributed positive returns")
    else:
        print("CHECK 3 INSIGHT: CASH-invested periods had negative returns")


def main():
    """Run all validation checks."""
    print("="*70)
    print("V3.1 IMPLEMENTATION VALIDATION")
    print("="*70)

    output_baseline = "output/mcap5k_v3_baseline"
    output_defensive = "output/mcap5k_v3_cash_defensive"
    parquet_path = "data/ohlcv_mcap5k.parquet"
    mom50_path = "data/nifty500_momentum50_benchmark.csv"

    # Check that output directories exist
    if not Path(output_defensive).exists():
        print(f"ERROR: {output_defensive} not found. Run ablation first.")
        return 1

    # Run checks
    check_1_portfolio_returns(output_defensive, parquet_path, n_samples=10)
    check_2_not_mom50(output_defensive, mom50_path)
    check_3_attribution_by_state(output_baseline, output_defensive)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
