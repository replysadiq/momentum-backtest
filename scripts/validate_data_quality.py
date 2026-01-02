#!/usr/bin/env python
"""
Data quality validation for OHLCV parquet inputs.

Checks schema, nulls, duplicates, price integrity, trading-day coverage, and
reports outlier returns. Designed for backtest readiness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["adj_close"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OHLCV parquet data quality.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/ohlcv_mcap5k_full_coverage_2011_2025.parquet"),
        help="Path to OHLCV parquet file.",
    )
    parser.add_argument(
        "--calendar-parquet",
        type=Path,
        default=Path("data/nifty500_index_yahoo.parquet"),
        help="Path to index parquet used as trading-day calendar.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2011-01-01",
        help="Start date (YYYY-MM-DD) for coverage checks.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-26",
        help="End date (YYYY-MM-DD) for coverage checks.",
    )
    parser.add_argument(
        "--max-outliers",
        type=int,
        default=20,
        help="Max outlier rows to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.parquet.exists():
        print(f"ERROR: Missing parquet file: {args.parquet}")
        return 1
    if not args.calendar_parquet.exists():
        print(f"ERROR: Missing calendar parquet file: {args.calendar_parquet}")
        return 1

    df = pd.read_parquet(args.parquet)
    df["date"] = pd.to_datetime(df["date"])

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        print(f"ERROR: Missing required columns: {missing_required}")
        return 1

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            print(f"WARNING: Optional column missing: {col}")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    errors = 0

    # Nulls in required fields
    null_counts = df[REQUIRED_COLUMNS].isna().sum()
    null_issues = null_counts[null_counts > 0]
    if not null_issues.empty:
        print("ERROR: Nulls in required columns:")
        print(null_issues.to_string())
        errors += 1

    # Date bounds
    out_of_range = df[(df["date"] < start) | (df["date"] > end)]
    if not out_of_range.empty:
        print(f"ERROR: Found {len(out_of_range)} rows outside [{start.date()}, {end.date()}].")
        errors += 1

    # Duplicates
    dupes = df.duplicated(subset=["symbol", "date"])
    dup_count = int(dupes.sum())
    if dup_count > 0:
        print(f"ERROR: Found {dup_count} duplicate (symbol, date) rows.")
        errors += 1

    # Price integrity
    price_cols = ["open", "high", "low", "close"]
    if "adj_close" in df.columns:
        price_cols.append("adj_close")
    non_positive = (df[price_cols] <= 0).any(axis=1)
    non_positive_count = int(non_positive.sum())
    if non_positive_count > 0:
        print(f"ERROR: Found {non_positive_count} rows with non-positive prices.")
        errors += 1

    # OHLC constraints
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]
    ohlc_bad = (high < low) | (open_ < low) | (open_ > high) | (close < low) | (close > high)
    ohlc_bad_count = int(ohlc_bad.sum())
    if ohlc_bad_count > 0:
        print(f"ERROR: Found {ohlc_bad_count} rows with invalid OHLC ranges.")
        errors += 1

    # Volume integrity
    neg_vol = df["volume"] < 0
    neg_vol_count = int(neg_vol.sum())
    if neg_vol_count > 0:
        print(f"ERROR: Found {neg_vol_count} rows with negative volume.")
        errors += 1

    # Coverage vs trading calendar
    calendar_df = pd.read_parquet(args.calendar_parquet, columns=["date", "adj_close"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    calendar = (
        calendar_df.dropna(subset=["adj_close"])
        .loc[(calendar_df["date"] >= start) & (calendar_df["date"] <= end), "date"]
        .drop_duplicates()
        .sort_values()
    )
    calendar_len = len(calendar)
    counts = df[(df["date"] >= start) & (df["date"] <= end)].groupby("symbol")["date"].nunique()
    missing_counts = calendar_len - counts
    missing_only = missing_counts[missing_counts > 0]
    extra_only = (-missing_counts[missing_counts < 0]).astype(int)
    if not missing_only.empty:
        print("ERROR: Symbols missing trading days:")
        print(missing_only.sort_values().head(args.max_outliers).to_string())
        errors += 1
    if not extra_only.empty:
        print("ERROR: Symbols with extra trading days not in index calendar:")
        print(extra_only.sort_values().head(args.max_outliers).to_string())
        errors += 1

    # Outlier returns (warning only)
    df_sorted = df.sort_values(["symbol", "date"]).copy()
    df_sorted["return"] = df_sorted.groupby("symbol")["close"].pct_change(fill_method=None)
    outliers = df_sorted.loc[df_sorted["return"].abs() > 0.5, ["symbol", "date", "close", "return"]]
    if not outliers.empty:
        print(f"WARNING: Found {len(outliers)} daily returns with abs(return) > 50%.")
        print(outliers.sort_values("return", key=lambda s: s.abs(), ascending=False).head(args.max_outliers).to_string(index=False))

    print("OK: Data quality checks complete." if errors == 0 else "FAILED: Data quality checks found issues.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
