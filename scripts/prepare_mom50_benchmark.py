#!/usr/bin/env python3
"""
Prepare NIFTY500 Momentum 50 benchmark data from multiple CSV files.

Merges yearly CSV files into a single time series suitable for backtesting.
"""

import pandas as pd
from pathlib import Path
import sys


def load_and_merge_mom50_data(input_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Load all NIFTY500 Momentum 50 CSV files and merge into a single DataFrame.

    Args:
        input_dir: Directory containing the yearly CSV files
        output_file: Path to save the merged data

    Returns:
        Merged DataFrame with Date index and Close prices
    """
    # Find all CSV files (exclude Zone.Identifier files)
    csv_files = sorted([
        f for f in input_dir.glob("*.csv")
        if "Zone.Identifier" not in str(f)
    ])

    print(f"Found {len(csv_files)} CSV files to merge")

    all_data = []

    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")

        # Read CSV
        df = pd.read_csv(csv_file)

        # Parse date - format is "29 Dec 2017"
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")

        # Keep only Date and Close
        df = df[["Date", "Close"]].copy()

        # Convert Close to numeric (handle any non-numeric values)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        # Drop any rows with NaN
        df = df.dropna()

        all_data.append(df)
        print(f"    Loaded {len(df)} rows")

    # Concatenate all data
    merged = pd.concat(all_data, ignore_index=True)

    # Sort by date (ascending)
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Remove duplicates (keep first occurrence)
    merged = merged.drop_duplicates(subset=["Date"], keep="first")

    # Set Date as index
    merged = merged.set_index("Date")

    print(f"\nMerged data:")
    print(f"  Total rows: {len(merged)}")
    print(f"  Date range: {merged.index[0].date()} to {merged.index[-1].date()}")
    print(f"  Price range: {merged['Close'].min():.2f} to {merged['Close'].max():.2f}")

    # Save to CSV
    merged.to_csv(output_file)
    print(f"\nSaved to: {output_file}")

    return merged


def main():
    input_dir = Path("data/nifty500Mom50")
    output_file = Path("data/nifty500_momentum50_benchmark.csv")

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    df = load_and_merge_mom50_data(input_dir, output_file)

    # Print some statistics
    print("\nYear-by-year coverage:")
    yearly = df.groupby(df.index.year).size()
    for year, count in yearly.items():
        print(f"  {year}: {count} trading days")

    return df


if __name__ == "__main__":
    main()
