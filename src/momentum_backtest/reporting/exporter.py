"""
Export backtest results to files.

Produces:
- equity_curve.csv: Daily portfolio and benchmark equity
- rebalance_log.csv: Rebalance events with tickers, weights, turnover
- state_log.csv: State machine features and transitions
- metrics.json: Performance metrics summary
- benchmark_info.json: Benchmark identity and coverage validation
- drawdown_attribution.csv: Worst drawdown analysis
- holdings_snapshot.csv: Complete holdings at each rebalance
- eligibility_coverage.csv: Stock filtering breakdown
"""

from dataclasses import asdict
from datetime import date
import math
import json
from pathlib import Path
from typing import List, Optional
import logging

import pandas as pd

from ..engine.backtest import BacktestResult, RebalanceRecord
from .metrics import PerformanceMetrics
from .audit import (
    export_benchmark_info,
    export_drawdown_attribution,
    export_holdings_snapshot,
    export_eligibility_coverage,
)


logger = logging.getLogger(__name__)


def export_results(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    output_dir: Path,
    benchmark_ticker: Optional[str] = None,
    benchmark_is_proxy: bool = False,
    benchmark_coverage: float = 1.0,
    benchmark_data: Optional[pd.Series] = None,
    fallback_chain: Optional[List[str]] = None,
) -> None:
    """
    Export all backtest results to files.

    Args:
        result: BacktestResult from backtest run
        metrics: Computed performance metrics
        output_dir: Directory to write files to
        benchmark_ticker: The benchmark ticker used
        benchmark_is_proxy: Whether using a proxy benchmark
        benchmark_coverage: Coverage percentage
        benchmark_data: Benchmark price series
        fallback_chain: List of tickers tried in fallback
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting results to: {output_dir}")

    # Export core results
    _export_equity_curve(result, output_dir)
    _export_rebalance_log(result.rebalance_records, output_dir)
    _export_state_log(result.rebalance_records, output_dir)
    _export_metrics(metrics, output_dir)

    # Export run manifest for reproducibility
    _export_run_manifest(
        result.config,
        len(result.rebalance_records),
        benchmark_ticker,
        benchmark_coverage,
        output_dir,
    )

    # Export audit reports
    logger.info("Exporting audit reports...")

    # Benchmark validation
    if benchmark_ticker and benchmark_data is not None:
        backtest_start = result.equity_curve.index[0].date()
        backtest_end = result.equity_curve.index[-1].date()
        export_benchmark_info(
            benchmark_ticker=benchmark_ticker,
            is_proxy=benchmark_is_proxy,
            coverage=benchmark_coverage,
            benchmark_data=benchmark_data,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            fallback_chain=fallback_chain or [],
            output_dir=output_dir,
        )

    # Drawdown attribution
    export_drawdown_attribution(
        result.equity_curve,
        result.rebalance_records,
        output_dir,
    )

    # Holdings snapshot
    export_holdings_snapshot(result.rebalance_records, output_dir)

    # Eligibility coverage
    if result.eligibility_records:
        export_eligibility_coverage(result.eligibility_records, output_dir)

    logger.info("Export complete")


def _export_equity_curve(
    result: BacktestResult,
    output_dir: Path,
) -> None:
    """Export daily equity curve."""
    # Align curves to common dates
    common_dates = result.equity_curve.index.intersection(
        result.benchmark_curve.index
    )

    df = pd.DataFrame({
        "date": common_dates,
        "portfolio_equity": result.equity_curve.reindex(common_dates).values,
        "benchmark_equity": result.benchmark_curve.reindex(common_dates).values,
    })

    # Add cumulative returns
    df["portfolio_return"] = df["portfolio_equity"] / df["portfolio_equity"].iloc[0] - 1
    df["benchmark_return"] = df["benchmark_equity"] / df["benchmark_equity"].iloc[0] - 1

    # Format date
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    filepath = output_dir / "equity_curve.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} rows)")


def _export_rebalance_log(
    records: List[RebalanceRecord],
    output_dir: Path,
) -> None:
    """Export rebalance log with tickers, weights, turnover."""
    rows = []

    for record in records:
        # Format tickers and weights
        if record.tickers:
            tickers_str = ",".join(record.tickers)
            weights_str = ",".join(
                f"{record.weights.get(t, 0):.4f}" for t in record.tickers
            )
        else:
            tickers_str = ""
            weights_str = ""

        rows.append({
            "date": record.date.strftime("%Y-%m-%d"),
            "state": record.state.name,
            "n_stocks": len(record.tickers),
            "tickers": tickers_str,
            "weights": weights_str,
            "turnover": record.turnover,
            "transaction_cost": record.transaction_cost,
        })

    df = pd.DataFrame(rows)

    filepath = output_dir / "rebalance_log.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} rows)")


def _export_state_log(
    records: List[RebalanceRecord],
    output_dir: Path,
) -> None:
    """Export state machine features and transitions."""
    rows = []

    prev_state = None
    for record in records:
        transition = ""
        if prev_state is not None and prev_state != record.state:
            transition = f"{prev_state.name}->{record.state.name}"

        rows.append({
            "date": record.date.strftime("%Y-%m-%d"),
            "prev_state": record.prev_state,
            "state": record.state.name,
            "transition": transition,
            "transition_reason": record.transition_reason,
            "transition_detail": record.transition_detail,
            "cash_entry_reason": record.cash_entry_reason or "",  # V3: Why CASH was entered
            "invested_flag": record.invested_flag,  # V3.1: True if holding equities
            "invested_weight": record.invested_fraction,
            "cash_weight": 1.0 - record.invested_fraction,
            "selected_count": record.selected_count,
            "defensive_selected_count": record.defensive_selected_count,
            "max_weight": record.max_weight,
            "min_required": record.min_required,
            "invested_fraction": record.invested_fraction,
            "concentration_gate_triggered": record.concentration_gate_triggered,
            "concentration_gate_reason": record.concentration_gate_reason,
            "benchmark_return_6m": record.features.benchmark_return_6m,
            "benchmark_return_3m": record.features.benchmark_return_3m,
            "benchmark_vol_1m": record.features.benchmark_vol_1m,
            "benchmark_vol_6m": record.features.benchmark_vol_6m,
            "vol_ratio": record.features.vol_ratio,
            "portfolio_drawdown_3m": record.features.portfolio_drawdown_3m,
        })

        prev_state = record.state

    df = pd.DataFrame(rows)

    filepath = output_dir / "state_log.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} rows)")


def _export_metrics(
    metrics: PerformanceMetrics,
    output_dir: Path,
) -> None:
    """Export performance metrics to JSON."""
    # Convert to dict
    metrics_dict = asdict(metrics)

    # Format floats for readability
    formatted = {}
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            # Format percentages and ratios appropriately
            if "return" in key or "volatility" in key or "drawdown" in key or "time_in" in key:
                formatted[key] = f"{value:.4f}"  # Keep as decimal for JSON
            else:
                formatted[key] = round(value, 4)
        else:
            formatted[key] = value

    filepath = output_dir / "metrics.json"
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"  Wrote {filepath}")


def _export_run_manifest(
    config: "BacktestConfig",
    n_rebalances: int,
    benchmark_ticker: Optional[str],
    benchmark_coverage: float,
    output_dir: Path,
) -> None:
    """
    Export run manifest for reproducibility and comparison.

    This captures all configuration parameters so runs can be compared.
    """
    from ..config import BacktestConfig

    min_required = None
    if config.max_weight is not None and config.max_weight > 0:
        min_required = int(math.ceil(1.0 / config.max_weight))

    manifest = {
        # Date range
        "start_date": str(config.start_date),
        "end_date": str(config.end_date),

        # Benchmark
        "benchmark_ticker": benchmark_ticker,
        "benchmark_coverage": round(benchmark_coverage, 4),
        "benchmark_role": "mom50",
        "benchmark_used": "mom50",

        # Rebalance calendar
        "rebalance_rule": "first_trading_day_of_month",
        "rebalance_months": config.rebalance_months,
        "n_rebalances": n_rebalances,

        # Scoring
        "scoring_mode": "nse_style" if config.nse_style_scoring else "standard",

        # Eligibility filters
        "filters": {
            "12m_return_positive": True,  # Always on
            "6m_return_positive": not config.disable_6m_filter,
            "positive_months_threshold": 0.60,  # 60% of months positive
            "dd_filter_enabled": config.use_dd_filter,
            "dd_threshold": config.dd_threshold if config.use_dd_filter else None,
        },

        # Portfolio construction
        "top_n": config.top_n_stocks,
        "max_weight": config.max_weight,
        "min_required_holdings": min_required,
        "weighting": "inverse_volatility_6m",

        # Transaction costs
        "tc_bps": config.tc_bps,

        # State machine thresholds
        "state_machine": {
            "panic_dd_threshold": config.panic_dd_threshold,
            "panic_vol_ratio": config.panic_vol_ratio,
            "panic_exit_vol_ratio": config.panic_exit_vol_ratio,
            "panic_exit_bench_ret": config.panic_exit_bench_ret,
        },

        # V2 Strategy levers
        "strategy_version": config.strategy_version,
        "enabled_levers": config.enabled_levers,
        "notes": "CASH split into CASH_INVESTED and CASH_TRUE; no behavioral change",
        "v2_params": {
            "panic_defensive_mode": config.panic_defensive_mode,
            "defensive_basket_size": config.defensive_basket_size,
            "rank_buffer": config.rank_buffer,
            "min_hold_months": config.min_hold_months,
            "disable_6m_filter": config.disable_6m_filter,
        },

        # Data computation notes
        "returns_computed_from": "daily_prices",
        "volatility_computed_from": "daily_returns_annualized",
    }

    filepath = output_dir / "run_manifest.json"
    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Wrote {filepath}")


def print_summary(metrics: PerformanceMetrics) -> None:
    """Print formatted summary to console."""
    from .metrics import format_metrics_table
    print(format_metrics_table(metrics))


def export_rolling_excess_return(
    rolling_df: pd.DataFrame,
    rolling_stats: "RollingExcessStats",
    output_dir: Path,
) -> None:
    """
    Export rolling 3-year excess return CSV and add stats to metrics.

    Args:
        rolling_df: DataFrame with date and rolling_3y_excess_cagr columns
        rolling_stats: Summary statistics
        output_dir: Directory to write files to
    """
    from .metrics import RollingExcessStats

    if len(rolling_df) == 0:
        logger.warning("No rolling excess return data to export (insufficient history)")
        return

    # Export CSV time series
    csv_path = output_dir / "rolling_excess_3y_vs_benchmark.csv"

    # Format for export: date and key columns only
    export_df = rolling_df[['date', 'rolling_3y_excess_cagr', 'strategy_cagr', 'benchmark_cagr']].copy()
    export_df['date'] = pd.to_datetime(export_df['date']).dt.strftime('%Y-%m-%d')

    # Round to 4 decimal places for readability
    for col in ['rolling_3y_excess_cagr', 'strategy_cagr', 'benchmark_cagr']:
        export_df[col] = export_df[col].round(4)

    export_df.to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path} ({len(export_df)} rows)")

    # Update metrics.json with rolling stats
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Add rolling excess stats
        metrics_data['rolling_3y_excess'] = {
            'window_months': rolling_stats.window_months,
            'frequency': rolling_stats.frequency,
            'benchmark_used': rolling_stats.benchmark_used,
            'n_observations': rolling_stats.n_observations,
            'pct_positive': round(rolling_stats.pct_positive, 4),
            'mean_excess': round(rolling_stats.mean_excess, 4),
            'median_excess': round(rolling_stats.median_excess, 4),
            'min_excess': round(rolling_stats.min_excess, 4),
            'max_excess': round(rolling_stats.max_excess, 4),
            'longest_negative_streak_months': rolling_stats.longest_negative_streak_months,
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"  Updated {metrics_path} with rolling_3y_excess stats")


def export_rolling_returns(
    rolling_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export rolling 3-year strategy vs benchmark returns.
    """
    if len(rolling_df) == 0:
        logger.warning("No rolling returns data to export (insufficient history)")
        return

    csv_path = output_dir / "rolling_returns.csv"
    export_df = rolling_df.copy()
    export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
    if "rolling_window_start_date" in export_df.columns:
        export_df["rolling_window_start_date"] = pd.to_datetime(
            export_df["rolling_window_start_date"]
        ).dt.strftime("%Y-%m-%d")

    for col in [
        "strategy_rolling_3y_cagr",
        "mom50_rolling_3y_cagr",
        "rolling_3y_excess_cagr",
        "strategy_rolling_3y_total_return",
        "mom50_rolling_3y_total_return",
        "rolling_3y_excess_total_return",
    ]:
        if col in export_df.columns:
            export_df[col] = export_df[col].round(6)

    export_df.to_csv(csv_path, index=False)
    logger.info(f"  Wrote {csv_path} ({len(export_df)} rows)")
