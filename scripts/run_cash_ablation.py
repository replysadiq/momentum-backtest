#!/usr/bin/env python3
"""
V3 Cash Entry Mode Ablation Study

Runs 3 backtest configurations with different --cash-entry-mode values
and produces a summary table comparing:
- time_in_cash
- strategy CAGR
- MaxDD (strategy)
- benchmark MaxDD
- Momentum 50 CAGR
- excess vs Momentum 50

Usage:
    python scripts/run_cash_ablation.py
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
    time_in_cash: float
    cagr: float
    maxdd: float
    bench_maxdd: float
    mom50_cagr: float
    excess_vs_mom50: float
    turnover: float


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

# Cash entry modes to test
CASH_ENTRY_MODES = ["baseline", "strict_dual", "strict_persist"]


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
        "--cash-entry-mode", mode,
        "--output-dir", output_dir,
    ]
    return cmd


def run_backtest(mode: str, output_dir: str) -> bool:
    """Run a single backtest and return success status."""
    cmd = build_command(mode, output_dir)
    print(f"\n{'='*60}")
    print(f"Running: --cash-entry-mode {mode}")
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
        "time_in_cash": metrics.get("time_in_cash", 0),
        "cagr": metrics.get("cagr", 0),
        "maxdd": metrics.get("max_drawdown", 0),
        "bench_maxdd": metrics.get("benchmark_max_drawdown", 0),
        "mom50_cagr": mom50_cagr if mom50_cagr else 0,
        "turnover": metrics.get("total_turnover", 0),
        "cash_entry_mode": metrics.get("cash_entry_mode", "unknown"),
    }


def print_summary_table(results: List[AblationResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("CASH ENTRY MODE ABLATION SUMMARY")
    print("="*100)

    # Header
    header = f"{'mode':<15} {'time_in_cash':>12} {'cagr':>10} {'maxdd':>10} {'bench_maxdd':>12} {'mom50_cagr':>12} {'excess_vs_mom50':>16} {'turnover':>10}"
    print(header)
    print("-"*100)

    # Rows
    for r in results:
        row = (
            f"{r.mode:<15} "
            f"{r.time_in_cash*100:>11.1f}% "
            f"{r.cagr*100:>9.2f}% "
            f"{r.maxdd*100:>9.2f}% "
            f"{r.bench_maxdd*100:>11.2f}% "
            f"{r.mom50_cagr*100:>11.2f}% "
            f"{r.excess_vs_mom50*100:>15.2f}% "
            f"{r.turnover:>9.2f}x"
        )
        print(row)

    print("="*100)

    # Compute deltas vs baseline
    if len(results) > 1:
        baseline = results[0]
        print("\nDELTAS VS BASELINE:")
        print("-"*60)
        for r in results[1:]:
            delta_cash = (r.time_in_cash - baseline.time_in_cash) * 100
            delta_cagr = (r.cagr - baseline.cagr) * 100
            delta_maxdd = (r.maxdd - baseline.maxdd) * 100
            delta_excess = (r.excess_vs_mom50 - baseline.excess_vs_mom50) * 100
            delta_turnover = r.turnover - baseline.turnover

            print(f"{r.mode:<15}: cash={delta_cash:+.1f}%, cagr={delta_cagr:+.2f}%, "
                  f"maxdd={delta_maxdd:+.2f}%, excess={delta_excess:+.2f}%, turnover={delta_turnover:+.2f}x")

    print()


def save_summary_csv(results: List[AblationResult], output_path: str) -> None:
    """Save summary to CSV."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode', 'time_in_cash', 'cagr', 'maxdd', 'bench_maxdd',
            'mom50_cagr', 'excess_vs_mom50', 'turnover'
        ])
        for r in results:
            writer.writerow([
                r.mode,
                f"{r.time_in_cash:.4f}",
                f"{r.cagr:.4f}",
                f"{r.maxdd:.4f}",
                f"{r.bench_maxdd:.4f}",
                f"{r.mom50_cagr:.4f}",
                f"{r.excess_vs_mom50:.4f}",
                f"{r.turnover:.4f}",
            ])

    print(f"Summary saved to: {output_path}")


def main():
    """Run the ablation study."""
    print("="*60)
    print("V3 CASH ENTRY MODE ABLATION STUDY")
    print("="*60)
    print(f"\nModes to test: {CASH_ENTRY_MODES}")
    print(f"Base config: rebalance_months={BASE_CONFIG['rebalance_months']}, "
          f"top_n={BASE_CONFIG['top_n']}, defensive_basket_size={BASE_CONFIG['defensive_basket_size']}")

    results: List[AblationResult] = []

    for mode in CASH_ENTRY_MODES:
        output_dir = f"output/mcap5k_v3_cash_{mode}"

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
            time_in_cash=metrics["time_in_cash"],
            cagr=metrics["cagr"],
            maxdd=metrics["maxdd"],
            bench_maxdd=metrics["bench_maxdd"],
            mom50_cagr=metrics["mom50_cagr"],
            excess_vs_mom50=metrics["cagr"] - metrics["mom50_cagr"],
            turnover=metrics["turnover"],
        )
        results.append(result)

    if not results:
        print("ERROR: No successful runs")
        return 1

    # Print summary table
    print_summary_table(results)

    # Save CSV
    save_summary_csv(results, "output/cash_entry_ablation_summary.csv")

    # Validate acceptance criteria
    print("\nACCEPTANCE CRITERIA CHECK:")
    print("-"*60)

    if len(results) > 1:
        baseline = results[0]
        for r in results[1:]:
            checks = []

            # Check: Reduce time_in_cash by 5-15% absolute
            delta_cash = baseline.time_in_cash - r.time_in_cash
            if delta_cash >= 0.05:
                checks.append(f"time_in_cash: PASS (reduced by {delta_cash*100:.1f}%)")
            else:
                checks.append(f"time_in_cash: {'MARGINAL' if delta_cash > 0 else 'FAIL'} (reduced by {delta_cash*100:.1f}%)")

            # Check: Improve excess vs Mom50 (less negative)
            delta_excess = r.excess_vs_mom50 - baseline.excess_vs_mom50
            if delta_excess > 0:
                checks.append(f"excess_vs_mom50: PASS (improved by {delta_excess*100:.2f}%)")
            else:
                checks.append(f"excess_vs_mom50: FAIL (worsened by {abs(delta_excess)*100:.2f}%)")

            # Check: MaxDD not worse by more than 2%
            delta_maxdd = r.maxdd - baseline.maxdd
            if delta_maxdd <= 0.02:
                checks.append(f"maxdd: PASS (delta={delta_maxdd*100:.2f}%)")
            else:
                checks.append(f"maxdd: FAIL (worsened by {delta_maxdd*100:.2f}%)")

            # Check: Turnover within ±5%
            delta_turnover_pct = (r.turnover - baseline.turnover) / baseline.turnover * 100 if baseline.turnover > 0 else 0
            if abs(delta_turnover_pct) <= 5:
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
