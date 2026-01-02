#!/usr/bin/env python3
"""
Capture Ratio Diagnostic vs NIFTY500 Momentum 50

Computes Up-Capture and Down-Capture ratios to diagnose whether relative
performance vs Mom50 is driven by:
(A) Reduced upside capture, or
(B) Intentional downside protection

Uses monthly observations aligned to rebalance dates.

Usage:
    python scripts/compute_capture_ratios.py [output_dir]
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_data(output_dir: str, mom50_path: str):
    """Load strategy equity curve and Mom50 benchmark."""
    output_path = Path(output_dir)

    # Load equity curve
    equity_df = pd.read_csv(output_path / "equity_curve.csv", parse_dates=["date"])
    equity_df.set_index("date", inplace=True)

    # Load rebalance log for rebalance dates
    rebalance_log = pd.read_csv(output_path / "rebalance_log.csv", parse_dates=["date"])
    rebalance_dates = rebalance_log["date"].tolist()

    # Load eligibility coverage (rebalance-level)
    eligibility_path = output_path / "eligibility_coverage.csv"
    eligibility_df = None
    if eligibility_path.exists():
        eligibility_df = pd.read_csv(eligibility_path, parse_dates=["date"])

    # Load Mom50 benchmark
    mom50_df = pd.read_csv(mom50_path, parse_dates=["Date"])
    mom50_df.set_index("Date", inplace=True)
    mom50_series = mom50_df["Close"]

    return equity_df, rebalance_dates, mom50_series, eligibility_df


def compute_period_returns(equity_df, mom50_series, rebalance_dates):
    """
    Compute returns for each rebalance period.

    Returns a DataFrame with:
    - period_end: rebalance date (end of period)
    - strategy_return: strategy return over the period
    - mom50_return: Mom50 return over the period
    """
    records = []

    for i in range(1, len(rebalance_dates)):
        period_start = rebalance_dates[i - 1]
        period_end = rebalance_dates[i]

        # Get strategy equity at period boundaries
        try:
            start_equity = equity_df.loc[period_start, "portfolio_equity"]
            end_equity = equity_df.loc[period_end, "portfolio_equity"]
            strategy_return = (end_equity / start_equity) - 1
        except KeyError:
            continue

        # Get Mom50 at period boundaries
        try:
            # Find closest available dates
            mom50_start = mom50_series.asof(period_start)
            mom50_end = mom50_series.asof(period_end)
            if pd.isna(mom50_start) or pd.isna(mom50_end):
                continue
            mom50_return = (mom50_end / mom50_start) - 1
        except (KeyError, TypeError):
            continue

        records.append({
            "date": period_end,
            "strategy_return": strategy_return,
            "mom50_return": mom50_return,
        })

    return pd.DataFrame(records)


def classify_and_compute_capture(df):
    """
    Classify periods as up/down and compute capture ratios.

    Up-Capture = Strategy return / Mom50 return (Mom50 up months)
    Down-Capture = Strategy return / Mom50 return (Mom50 down months)
    """
    # Classify based on Mom50 return sign
    df["capture_type"] = df["mom50_return"].apply(lambda x: "up" if x > 0 else "down")

    # Compute capture ratio for each period
    # Avoid division by zero
    df["capture_ratio"] = df.apply(
        lambda row: row["strategy_return"] / row["mom50_return"]
        if abs(row["mom50_return"]) > 1e-6 else np.nan,
        axis=1
    )

    return df


def compute_summary_metrics(df):
    """Compute summary capture metrics."""
    up_months = df[df["capture_type"] == "up"]
    down_months = df[df["capture_type"] == "down"]

    # Up-Capture: average of (strategy_return / mom50_return) for up months
    # This tells us what fraction of Mom50's upside we captured
    if len(up_months) > 0 and up_months["mom50_return"].sum() != 0:
        up_capture = up_months["strategy_return"].sum() / up_months["mom50_return"].sum()
    else:
        up_capture = np.nan

    # Down-Capture: average of (strategy_return / mom50_return) for down months
    # Lower is better - means we lost less when Mom50 was down
    if len(down_months) > 0 and down_months["mom50_return"].sum() != 0:
        down_capture = down_months["strategy_return"].sum() / down_months["mom50_return"].sum()
    else:
        down_capture = np.nan

    # Alternative: simple mean of ratios
    up_capture_mean = up_months["capture_ratio"].mean() if len(up_months) > 0 else np.nan
    down_capture_mean = down_months["capture_ratio"].mean() if len(down_months) > 0 else np.nan

    # Count months
    n_up = len(up_months)
    n_down = len(down_months)
    n_total = len(df)

    return {
        "up_capture_ratio": round(up_capture, 4) if not np.isnan(up_capture) else None,
        "down_capture_ratio": round(down_capture, 4) if not np.isnan(down_capture) else None,
        "up_capture_mean": round(up_capture_mean, 4) if not np.isnan(up_capture_mean) else None,
        "down_capture_mean": round(down_capture_mean, 4) if not np.isnan(down_capture_mean) else None,
        "pct_up_months": round(n_up / n_total * 100, 1) if n_total > 0 else 0,
        "pct_down_months": round(n_down / n_total * 100, 1) if n_total > 0 else 0,
        "n_up_months": n_up,
        "n_down_months": n_down,
        "n_total_periods": n_total,
    }

def compute_coverage_ranges(dates):
    """Compute contiguous date ranges for coverage/exclusion reporting."""
    if dates.empty:
        return []
    dates = dates.sort_values().reset_index(drop=True)
    ranges = []
    start = dates.iloc[0]
    prev = dates.iloc[0]
    for current in dates.iloc[1:]:
        if (current - prev).days > 5:
            ranges.append((start.date().isoformat(), prev.date().isoformat()))
            start = current
        prev = current
    ranges.append((start.date().isoformat(), prev.date().isoformat()))
    return ranges


def interpret_results(metrics):
    """Generate plain-language interpretation."""
    up_cap = metrics["up_capture_ratio"]
    down_cap = metrics["down_capture_ratio"]

    print("\n" + "="*70)
    print("PLAIN-LANGUAGE INTERPRETATION")
    print("="*70)

    if up_cap is None or down_cap is None:
        print("ERROR: Could not compute capture ratios.")
        return "inconclusive"

    print(f"\nUp-Capture Ratio:   {up_cap:.1%}")
    print(f"Down-Capture Ratio: {down_cap:.1%}")
    print()

    # Interpretation logic
    # If up_capture < 100% and down_capture < 100%: we're capturing less upside but also less downside
    # If up_capture ≈ 100% and down_capture < 100%: pure downside protection
    # If up_capture < 100% and down_capture ≈ 100%: reduced upside with no protection

    up_deficit = 1.0 - up_cap
    down_benefit = 1.0 - down_cap

    print("Analysis:")
    print(f"  - When Mom50 goes UP, strategy captures {up_cap:.1%} of the move")
    print(f"  - When Mom50 goes DOWN, strategy captures {down_cap:.1%} of the loss")
    print()

    # Determine primary driver
    if down_benefit > up_deficit * 1.5:
        # Down-capture benefit significantly exceeds up-capture deficit
        primary_driver = "B"
        conclusion = (
            "Relative performance vs Momentum-50 is driven primarily by "
            "(B) INTENTIONAL DOWNSIDE PROTECTION.\n\n"
            f"The strategy avoids {(1-down_cap)*100:.1f}% of Mom50's drawdowns "
            f"while only sacrificing {(1-up_cap)*100:.1f}% of upside."
        )
    elif up_deficit > down_benefit * 1.5:
        # Up-capture deficit significantly exceeds down-capture benefit
        primary_driver = "A"
        conclusion = (
            "Relative performance vs Momentum-50 is driven primarily by "
            "(A) REDUCED UPSIDE CAPTURE.\n\n"
            f"The strategy misses {(1-up_cap)*100:.1f}% of Mom50's upside "
            f"but only avoids {(1-down_cap)*100:.1f}% of drawdowns."
        )
    else:
        # Balanced trade-off
        if up_cap >= 0.95 and down_cap < 0.85:
            primary_driver = "B"
            conclusion = (
                "Relative performance vs Momentum-50 is driven primarily by "
                "(B) INTENTIONAL DOWNSIDE PROTECTION.\n\n"
                f"The strategy maintains {up_cap:.1%} upside capture while "
                f"reducing downside exposure to {down_cap:.1%}."
            )
        elif down_cap >= 0.95 and up_cap < 0.85:
            primary_driver = "A"
            conclusion = (
                "Relative performance vs Momentum-50 is driven primarily by "
                "(A) REDUCED UPSIDE CAPTURE.\n\n"
                f"The strategy only captures {up_cap:.1%} of upside while "
                f"experiencing {down_cap:.1%} of drawdowns."
            )
        else:
            primary_driver = "MIXED"
            conclusion = (
                "Relative performance vs Momentum-50 reflects a BALANCED trade-off "
                "between (A) reduced upside capture and (B) downside protection.\n\n"
                f"Up-Capture: {up_cap:.1%}  |  Down-Capture: {down_cap:.1%}"
            )

    print("-"*70)
    print(conclusion)
    print("-"*70)

    return primary_driver


def main():
    """Run capture ratio analysis."""
    # Default paths
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/mcap5k_v3_cash_defensive"
    mom50_path = "data/nifty500_momentum50_benchmark.csv"

    print("="*70)
    print("CAPTURE RATIO ANALYSIS vs NIFTY500 Momentum 50")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    equity_df, rebalance_dates, mom50_series, eligibility_df = load_data(output_dir, mom50_path)
    print(f"  Strategy: {len(equity_df)} trading days")
    print(f"  Rebalance periods: {len(rebalance_dates)}")
    print(f"  Mom50: {len(mom50_series)} data points")
    if eligibility_df is None:
        print("  Eligibility coverage: MISSING (eligibility_coverage.csv not found)")
    else:
        print(f"  Eligibility coverage: {len(eligibility_df)} rows")

    # Compute period returns
    print("\nComputing period returns aligned to rebalance dates...")
    returns_df = compute_period_returns(equity_df, mom50_series, rebalance_dates)
    print(f"  Valid periods: {len(returns_df)}")

    # Classify and compute capture ratios
    print("\nClassifying up/down periods and computing capture ratios...")
    capture_df = classify_and_compute_capture(returns_df)

    # Compute summary metrics
    metrics = compute_summary_metrics(capture_df)

    print("\n" + "="*70)
    print("SUMMARY METRICS")
    print("="*70)
    print(f"\n  Up-Capture Ratio:    {metrics['up_capture_ratio']:.1%}" if metrics['up_capture_ratio'] else "  Up-Capture: N/A")
    print(f"  Down-Capture Ratio:  {metrics['down_capture_ratio']:.1%}" if metrics['down_capture_ratio'] else "  Down-Capture: N/A")
    print(f"\n  Up months:   {metrics['n_up_months']} ({metrics['pct_up_months']:.1f}%)")
    print(f"  Down months: {metrics['n_down_months']} ({metrics['pct_down_months']:.1f}%)")

    # Save CSV time series
    output_path = Path(output_dir)
    csv_path = output_path / "capture_ratios.csv"
    capture_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  CSV saved: {csv_path}")

    # Eligibility-filtered capture ratios (eligible_stocks >= 10)
    filtered_metrics = None
    coverage_stats = None
    filtered_csv_path = output_path / "capture_ratios_eligibility_filtered.csv"
    if eligibility_df is not None:
        elig = eligibility_df[["date", "data_eligible_count"]].copy()
        elig["date"] = pd.to_datetime(elig["date"])
        capture_with_elig = capture_df.merge(elig, on="date", how="left")
        capture_with_elig["eligible_ok"] = capture_with_elig["data_eligible_count"] >= 10

        filtered_df = capture_with_elig[capture_with_elig["eligible_ok"]].copy()
        filtered_df.to_csv(filtered_csv_path, index=False, float_format="%.6f")

        filtered_metrics = compute_summary_metrics(filtered_df)

        included = capture_with_elig[capture_with_elig["eligible_ok"]]["date"]
        excluded = capture_with_elig[~capture_with_elig["eligible_ok"]]["date"]
        n_total = len(capture_with_elig)
        coverage_stats = {
            "n_total_periods": n_total,
            "n_included": int(len(included)),
            "n_excluded": int(len(excluded)),
            "pct_included": round(len(included) / n_total * 100, 1) if n_total else 0.0,
            "pct_excluded": round(len(excluded) / n_total * 100, 1) if n_total else 0.0,
            "included_ranges": compute_coverage_ranges(included),
            "excluded_ranges": compute_coverage_ranges(excluded),
        }

        print(f"\n  CSV saved: {filtered_csv_path}")
    else:
        print("\n  Skipping eligibility-filtered capture ratios (no eligibility_coverage.csv)")

    # Generate interpretation
    primary_driver = interpret_results(metrics)

    # Append summary metrics
    json_path = output_path / "metrics_vs_comparison.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            json_blob = json.load(f)
    else:
        json_blob = {}

    json_blob["capture_ratios"] = metrics
    json_blob["capture_ratios"]["primary_driver"] = primary_driver

    if filtered_metrics is not None:
        json_blob["capture_ratios_filtered"] = filtered_metrics
        json_blob["capture_ratios_filtered"]["eligibility_threshold"] = 10
        json_blob["capture_ratios_filtered"]["coverage"] = coverage_stats

        # Determine structural vs sparsity-driven
        up_raw = metrics["up_capture_ratio"]
        down_raw = metrics["down_capture_ratio"]
        up_f = filtered_metrics["up_capture_ratio"]
        down_f = filtered_metrics["down_capture_ratio"]
        if None not in (up_raw, down_raw, up_f, down_f):
            delta_up = abs(up_raw - up_f)
            delta_down = abs(down_raw - down_f)
            structural = delta_up <= 0.05 and delta_down <= 0.05
        else:
            structural = False

        json_blob["capture_ratios_filtered"]["conclusion"] = (
            "structural" if structural else "driven_by_early_period_sparsity"
        )

        print("\nConclusion:")
        if structural:
            print("Capture behavior appears STRUCTURAL (filtered ratios align with raw).")
        else:
            print("Capture behavior appears DRIVEN BY EARLY-PERIOD UNIVERSE SPARSITY.")

    with open(json_path, "w") as f:
        json.dump(json_blob, f, indent=2)
    print(f"  JSON updated: {json_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
