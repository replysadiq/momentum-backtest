#!/usr/bin/env python3
"""
V3.1 Cash Replace Mode Ablation Study

Runs 2 backtest configurations with different --cash-replace-mode values
and produces a summary table comparing:
- CAGR
- MaxDD
- Volatility
- Turnover
- time_in_cash_state
- time_in_cash_invested
- Mom50 CAGR
- Excess vs Mom50

Usage:
    python scripts/run_cash_replace_ablation.py
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationResult:
    """Results from a single ablation run."""
    mode: str
    cagr: float
    maxdd: float
    volatility: float
    turnover: float
    time_in_cash_state: float
    time_in_cash_invested: float
    mom50_cagr: float
    excess_vs_mom50: float


# Base configuration (from user specification)
BASE_CONFIG = {
    "tickers_csv": "data/nse_mcap5k_universe.csv",
    "parquet_file": "data/ohlcv_mcap5k.parquet",
    "start": "2011-01-01",
    "end": "2025-12-18",
    "panic_defensive_mode": True,
    "panic_vol_ratio": 1.5,
    "defensive_basket_size": 30,
    "tc_bps": 10,
    "rebalance_months": 2,
    "top_n": 30,
    "comparison_benchmark_csv": "data/nifty500_momentum50_benchmark.csv",
}

# Cash replace modes to test
CASH_REPLACE_MODES = ["none", "defensive"]


def build_command(mode: str, output_dir: str) -> List[str]:
    """Build the backtest command for a given mode."""
    cmd = [
        sys.executable, "-m", "momentum_backtest",
        "--tickers-csv", BASE_CONFIG["tickers_csv"],
        "--parquet-file", BASE_CONFIG["parquet_file"],
        "--start", BASE_CONFIG["start"],
        "--end", BASE_CONFIG["end"],
        "--panic-defensive-mode",
        "--panic-vol-ratio", str(BASE_CONFIG["panic_vol_ratio"]),
        "--defensive-basket-size", str(BASE_CONFIG["defensive_basket_size"]),
        "--tc-bps", str(BASE_CONFIG["tc_bps"]),
        "--rebalance-months", str(BASE_CONFIG["rebalance_months"]),
        "--top-n", str(BASE_CONFIG["top_n"]),
        "--comparison-benchmark-csv", BASE_CONFIG["comparison_benchmark_csv"],
        "--cash-replace-mode", mode,
        "--output-dir", output_dir,
    ]
    return cmd


def run_backtest(mode: str, output_dir: str) -> bool:
    """Run a single backtest and return success status."""
    cmd = build_command(mode, output_dir)
    print(f"\n{'='*60}")
    print(f"Running: --cash-replace-mode {mode}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def extract_metrics(output_dir: str) -> Optional[Dict]:
    """Extract relevant metrics from output files."""
    output_path = Path(output_dir)

    metrics_file = output_path / "metrics.json"
    comparison_file = output_path / "metrics_vs_comparison.json"

    if not metrics_file.exists():
        print(f"ERROR: {metrics_file} not found")
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Get comparison metrics (Mom50 CAGR) if available
    mom50_cagr = None
    if comparison_file.exists():
        with open(comparison_file) as f:
            comp_metrics = json.load(f)
            mom50_cagr = comp_metrics.get("benchmark_cagr")

    return {
        "cagr": metrics.get("cagr", 0),
        "maxdd": metrics.get("max_drawdown", 0),
        "volatility": metrics.get("volatility", 0),
        "turnover": metrics.get("total_turnover", 0),
        "time_in_cash_state": metrics.get("time_in_cash", 0),
        "time_in_cash_invested": metrics.get("time_in_cash_invested", 0),
        "mom50_cagr": mom50_cagr if mom50_cagr else 0,
        "cash_replace_mode": metrics.get("cash_replace_mode", "unknown"),
    }


def print_summary_table(results: List[AblationResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "="*120)
    print("V3.1 CASH REPLACE MODE ABLATION SUMMARY")
    print("="*120)

    # Header
    header = (
        f"{'mode':<12} {'cagr':>10} {'maxdd':>10} {'volatility':>12} "
        f"{'turnover':>10} {'cash_state':>12} {'cash_invested':>14} "
        f"{'mom50_cagr':>12} {'excess':>12}"
    )
    print(header)
    print("-"*120)

    # Rows
    for r in results:
        row = (
            f"{r.mode:<12} "
            f"{r.cagr*100:>9.2f}% "
            f"{r.maxdd*100:>9.2f}% "
            f"{r.volatility*100:>11.2f}% "
            f"{r.turnover:>9.2f}x "
            f"{r.time_in_cash_state*100:>11.1f}% "
            f"{r.time_in_cash_invested*100:>13.1f}% "
            f"{r.mom50_cagr*100:>11.2f}% "
            f"{r.excess_vs_mom50*100:>11.2f}%"
        )
        print(row)

    print("="*120)

    # Compute deltas vs baseline
    if len(results) > 1:
        baseline = results[0]
        print("\nDELTAS VS BASELINE (none):")
        print("-"*60)
        for r in results[1:]:
            delta_cagr = (r.cagr - baseline.cagr) * 100
            delta_maxdd = (r.maxdd - baseline.maxdd) * 100
            delta_turnover = r.turnover - baseline.turnover
            delta_excess = (r.excess_vs_mom50 - baseline.excess_vs_mom50) * 100

            print(f"{r.mode:<12}: cagr={delta_cagr:+.2f}%, maxdd={delta_maxdd:+.2f}%, "
                  f"turnover={delta_turnover:+.2f}x, excess={delta_excess:+.2f}%")

    print()


def save_summary_csv(results: List[AblationResult], output_path: str) -> None:
    """Save summary to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'replace_mode', 'cagr', 'maxdd', 'volatility', 'turnover',
            'time_in_cash_state', 'time_in_cash_invested', 'mom50_cagr', 'excess_vs_mom50'
        ])
        for r in results:
            writer.writerow([
                r.mode,
                f"{r.cagr:.4f}",
                f"{r.maxdd:.4f}",
                f"{r.volatility:.4f}",
                f"{r.turnover:.4f}",
                f"{r.time_in_cash_state:.4f}",
                f"{r.time_in_cash_invested:.4f}",
                f"{r.mom50_cagr:.4f}",
                f"{r.excess_vs_mom50:.4f}",
            ])

    print(f"Summary saved to: {output_path}")


def main():
    """Run the ablation study."""
    print("="*60)
    print("V3.1 CASH REPLACE MODE ABLATION STUDY")
    print("="*60)
    print(f"\nModes to test: {CASH_REPLACE_MODES}")
    print(f"Base config: rebalance_months={BASE_CONFIG['rebalance_months']}, "
          f"top_n={BASE_CONFIG['top_n']}, defensive_basket_size={BASE_CONFIG['defensive_basket_size']}")

    results: List[AblationResult] = []

    for mode in CASH_REPLACE_MODES:
        output_dir = f"output/mcap5k_v3_{'baseline' if mode == 'none' else 'cash_defensive'}"

        # Run backtest
        success = run_backtest(mode, output_dir)

        if not success:
            print(f"ERROR: Backtest failed for mode={mode}")
            continue

        # Extract metrics
        metrics = extract_metrics(output_dir)
        if metrics is None:
            print(f"ERROR: Could not extract metrics for mode={mode}")
            continue

        # Create result
        result = AblationResult(
            mode=mode,
            cagr=metrics["cagr"],
            maxdd=metrics["maxdd"],
            volatility=metrics["volatility"],
            turnover=metrics["turnover"],
            time_in_cash_state=metrics["time_in_cash_state"],
            time_in_cash_invested=metrics["time_in_cash_invested"],
            mom50_cagr=metrics["mom50_cagr"],
            excess_vs_mom50=metrics["cagr"] - metrics["mom50_cagr"],
        )
        results.append(result)

    if not results:
        print("ERROR: No successful runs")
        return 1

    # Print summary table
    print_summary_table(results)

    # Save CSV
    save_summary_csv(results, "output/cash_replace_ablation_summary.csv")

    # Validate acceptance criteria
    print("\nACCEPTANCE CRITERIA CHECK:")
    print("-"*60)

    if len(results) > 1:
        baseline = results[0]
        for r in results[1:]:
            checks = []

            # Check: Excess vs Mom50 improves (less negative)
            delta_excess = r.excess_vs_mom50 - baseline.excess_vs_mom50
            if delta_excess > 0:
                checks.append(f"excess_vs_mom50: PASS (improved by {delta_excess*100:.2f}%)")
            else:
                checks.append(f"excess_vs_mom50: FAIL (worsened by {abs(delta_excess)*100:.2f}%)")

            # Check: MaxDD increase <= +1.5% absolute vs baseline
            delta_maxdd = r.maxdd - baseline.maxdd
            if delta_maxdd <= 0.015:
                checks.append(f"maxdd: PASS (delta={delta_maxdd*100:.2f}%)")
            else:
                checks.append(f"maxdd: FAIL (worsened by {delta_maxdd*100:.2f}%)")

            # Check: Turnover increase <= +5% vs baseline
            delta_turnover_pct = (r.turnover - baseline.turnover) / baseline.turnover * 100 if baseline.turnover > 0 else 0
            if delta_turnover_pct <= 5:
                checks.append(f"turnover: PASS (delta={delta_turnover_pct:+.1f}%)")
            else:
                checks.append(f"turnover: FAIL (delta={delta_turnover_pct:+.1f}%)")

            print(f"\n{r.mode}:")
            for check in checks:
                print(f"  {check}")

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
