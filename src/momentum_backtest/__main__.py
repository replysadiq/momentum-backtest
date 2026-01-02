"""
Main entry point for the momentum backtest system.

Orchestrates the full backtest workflow:
1. Parse arguments and build config
2. Stage external data files into project ./data folder
3. Load ticker universe
4. Download benchmark data and determine trading calendar
5. Build rebalance calendar from benchmark trading dates
6. Download/load stock price data
7. Run backtest
8. Compute metrics and export results
"""

import datetime
import logging
import json
from pathlib import Path
import subprocess
from collections import defaultdict
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from .cli import parse_args, setup_logging, build_config
from .config import CashReplaceMode
from .data.staging import stage_parquet_file, stage_ticker_csv, StagingError
from .data.universe import load_universe
from .data.benchmark import load_benchmark_from_csv
from .data.downloader import download_price_data, align_to_calendar, load_price_data_from_parquet
from .data.sanitizer import sanitize_stock_data, required_history_months
from .engine.backtest import run_backtest
from .engine.signals import REJECTION_REASON_CODES
from .engine.state_machine import MarketState
from .reporting.metrics import (
    compute_metrics,
    compute_rolling_excess_return,
    compute_rolling_returns,
    _compute_cagr,
    _compute_max_drawdown,
    _compute_volatility,
)
from .reporting.exporter import export_results, print_summary, export_rolling_excess_return, export_rolling_returns
from .validation import run_all_validations, ValidationError


logger = logging.getLogger(__name__)

# Project data directory for staged files
PROJECT_DATA_DIR = Path("./data")


def _validate_rebalance_invariants(
    rebalance_calendar: "pd.DatetimeIndex",
    trading_dates: "pd.DatetimeIndex",
) -> None:
    """
    Validate critical invariants for the rebalance calendar.

    Invariants:
    1. Every rebalance date exists in the trading calendar
    2. Rebalance dates are strictly increasing
    3. Exactly one rebalance per calendar month

    Args:
        rebalance_calendar: The frozen rebalance calendar
        trading_dates: Trading calendar dates

    Raises:
        ValidationError: If any invariant is violated (fatal bug)
    """
    import pandas as pd

    trading_set = set(trading_dates)

    # Invariant 1: Every rebalance date is a trading day
    for rebal_date in rebalance_calendar:
        if rebal_date not in trading_set:
            raise ValidationError(
                f"FATAL BUG: Rebalance date {rebal_date.date()} not in trading calendar. "
                "This indicates a bug in calendar construction - rebalance dates must be "
                "derived from the trading calendar, never from stock data."
            )

    # Invariant 2: Strictly increasing
    for i in range(1, len(rebalance_calendar)):
        if rebalance_calendar[i] <= rebalance_calendar[i-1]:
            raise ValidationError(
                f"FATAL BUG: Rebalance dates not strictly increasing: "
                f"{rebalance_calendar[i-1].date()} >= {rebalance_calendar[i].date()}"
            )

    # Invariant 3: At most one rebalance per calendar month
    month_counts = pd.Series(rebalance_calendar).dt.to_period("M").value_counts()
    duplicates = month_counts[month_counts > 1]
    if not duplicates.empty:
        raise ValidationError(
            f"FATAL BUG: Multiple rebalance dates in same month: {duplicates.to_dict()}"
        )

    logger.debug(f"All rebalance invariants validated for {len(rebalance_calendar)} dates")


def infer_trading_dates_from_universe(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    """
    Infer ALL trading days from universe data using coverage rule:
    A day is a trading day iff >=20% of universe tickers have valid adj_close.
    """

    d = df.loc[:, ["date", "symbol", "adj_close"]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    d = d[(d["date"] >= start) & (d["date"] <= end)]
    d = d[d["adj_close"].notna()]

    # Universe size (robust for any data)
    universe_size = d["symbol"].nunique()
    min_required = int(np.ceil(0.20 * universe_size))

    counts = d.groupby("date")["symbol"].nunique()
    trading_dates = counts[counts >= min_required].index

    trading_dates = pd.DatetimeIndex(sorted(trading_dates.unique()))

    if trading_dates.empty:
        raise RuntimeError(
            f"No trading days inferred for range {start.date()}..{end.date()} "
            f"(universe_size={universe_size}, min_required={min_required})"
        )

    return trading_dates


def first_weekday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    while d.weekday() >= 5:  # Saturday / Sunday
        d += pd.Timedelta(days=1)
    return d


def build_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rebalance_months: int,
) -> pd.DatetimeIndex:
    months = pd.date_range(
        start=pd.Timestamp(start.year, start.month, 1),
        end=pd.Timestamp(end.year, end.month, 1),
        freq=pd.DateOffset(months=rebalance_months),
    )

    rebalance_dates = []

    for m in months:
        anchor = first_weekday(m.year, m.month)
        idx = trading_dates.searchsorted(anchor, side="left")
        if idx >= len(trading_dates):
            break

        d = trading_dates[idx]
        if d > end:
            break

        rebalance_dates.append(d)

    if not rebalance_dates:
        raise RuntimeError("Failed to construct any rebalance dates.")

    return pd.DatetimeIndex(rebalance_dates)


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command-line arguments (default: sys.argv)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Parse arguments
    args = parse_args(argv)

    # Setup logging
    setup_logging(verbose=args.verbose, debug=args.debug)

    try:
        # Build configuration
        logger.info("Building configuration...")
        config = build_config(args)

        logger.info("=" * 60)
        logger.info("MOMENTUM BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Strategy Version: {config.strategy_version}")
        if config.enabled_levers:
            logger.info(f"Enabled Levers: {', '.join(config.enabled_levers)}")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Top N stocks: {config.top_n_stocks}")
        logger.info(f"Max weight: {config.max_weight}")
        logger.info(f"Transaction costs: {config.tc_bps} bps")
        logger.info(f"Drawdown filter: {config.use_dd_filter} (threshold: {config.dd_threshold})")
        scoring_method = "NSE Momentum 50 (Z-score)" if config.nse_style_scoring else "Standard (Sharpe-weighted)"
        logger.info(f"Scoring method: {scoring_method}")
        logger.info("=" * 60)

        # Require at least one data source for universe
        if args.parquet_file is None and config.tickers_csv is None:
            raise ValueError("Provide --parquet-file or --tickers-csv to define the universe")

        # Step 1: Stage external data files into project ./data folder
        logger.info("Step 1: Staging data files...")
        staged_tickers_csv = None
        if config.tickers_csv is not None:
            staged_tickers_csv = stage_ticker_csv(config.tickers_csv, PROJECT_DATA_DIR)
            logger.info(f"  Staged ticker CSV: {staged_tickers_csv}")
        else:
            logger.info("  No ticker CSV provided; using parquet symbols for universe")

        staged_parquet = None
        if args.parquet_file is not None:
            staged_parquet = stage_parquet_file(args.parquet_file, PROJECT_DATA_DIR)
            logger.info(f"  Staged parquet: {staged_parquet}")

        # Step 2: Load ticker universe from staged file
        logger.info("Step 2: Loading ticker universe...")
        if staged_tickers_csv is None:
            tickers = []
            logger.info("  Skipping ticker CSV load (parquet-only run)")
        else:
            tickers = load_universe(staged_tickers_csv)
            logger.info(f"  Loaded {len(tickers)} tickers")

        # Step 3: Get benchmark data
        logger.info("Step 3: Getting benchmark data...")
        if args.benchmark_csv is not None and args.benchmark_parquet is not None:
            raise ValueError("Provide only one of --benchmark-csv or --benchmark-parquet")
        benchmark_path = args.benchmark_parquet or args.benchmark_csv
        if benchmark_path is not None:
            benchmark_result = load_benchmark_from_csv(
                benchmark_path, config.start_date, config.end_date, ticker_name="MOM50"
            )
        else:
            benchmark_result = None

        if benchmark_result is not None:
            logger.info(f"  Using benchmark: {benchmark_result.ticker} "
                        f"(coverage: {benchmark_result.coverage:.1%})")

        if benchmark_result is None:
            raise ValueError("Benchmark data is required. Provide --benchmark-csv or --benchmark-parquet.")
        # Step 4b: Load comparison benchmark if provided (for metrics only)
        comparison_benchmark = None
        if args.comparison_benchmark_csv is not None and args.comparison_benchmark_parquet is not None:
            raise ValueError("Provide only one of --comparison-benchmark-csv or --comparison-benchmark-parquet")
        comparison_path = args.comparison_benchmark_parquet or args.comparison_benchmark_csv
        if comparison_path is not None:
            logger.info("Step 4b: Loading comparison benchmark...")
            comparison_benchmark = load_benchmark_from_csv(
                comparison_path, config.start_date, config.end_date, ticker_name="MOM50",
            )
            logger.info(f"  Comparison benchmark: {comparison_benchmark.ticker} "
                        f"(coverage: {comparison_benchmark.coverage:.1%})")
        else:
            comparison_benchmark = benchmark_result

        # Step 5: Load or download stock data
        if staged_parquet is not None:
            logger.info("Step 5: Loading stock price data from staged parquet...")
            # When using parquet data, derive the active universe from available symbols
            # to avoid warnings for post-listing/pre-delisting tickers.
            parquet_symbols = pd.read_parquet(staged_parquet, columns=["symbol"])
            parquet_symbols = parquet_symbols["symbol"].astype(str).str.strip()
            parquet_symbols = parquet_symbols.str.replace(".NS", "", regex=False)
            parquet_symbols = parquet_symbols.str.replace(".BO", "", regex=False)
            tickers = sorted({f"{sym}.NS" for sym in parquet_symbols.unique() if sym})
            logger.info(f"  Using {len(tickers)} tickers derived from parquet symbols")
            universe_ohlcv_df = pd.read_parquet(
                staged_parquet, columns=["date", "symbol", "adj_close"]
            )
            raw_price_data = load_price_data_from_parquet(
                staged_parquet,
                tickers,
                config.start_date,
                config.end_date,
                price_column=config.price_column,
                warn_missing=False,
            )
            logger.info(f"  Loaded data for {len(raw_price_data)} stocks from parquet")
        else:
            logger.info("Step 5: Downloading stock price data...")
            raw_price_data = download_price_data(
                tickers,
                config.start_date,
                config.end_date,
                price_column=config.price_column,
                show_progress=True,
            )
            logger.info(f"  Downloaded data for {len(raw_price_data)} stocks")
            universe_rows = []
            for symbol, series in raw_price_data.items():
                if series is None or series.empty:
                    continue
                for idx, value in series.items():
                    universe_rows.append(
                        {"date": idx, "symbol": symbol, "adj_close": value}
                    )
            universe_ohlcv_df = pd.DataFrame(universe_rows)

        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)

        if args.trading_calendar_csv is not None and args.trading_calendar_parquet is not None:
            raise ValueError("Provide only one of --trading-calendar-csv or --trading-calendar-parquet")

        trading_calendar_path = args.trading_calendar_parquet or args.trading_calendar_csv
        calendar_result = None
        if trading_calendar_path is not None:
            calendar_result = load_benchmark_from_csv(
                trading_calendar_path,
                config.start_date,
                config.end_date,
                ticker_name="TRADING_CALENDAR",
            )
            trading_dates = calendar_result.data.index
            trading_dates = trading_dates[(trading_dates >= start) & (trading_dates <= end)]
            trading_dates = pd.DatetimeIndex(sorted(trading_dates.unique()))
            if trading_dates.empty:
                raise ValueError("Trading calendar data is empty for the requested date range.")
            logger.info(
                f"Trading calendar: {len(trading_dates)} days "
                f"({trading_dates[0].date()} → {trading_dates[-1].date()})"
            )
        else:
            trading_dates = infer_trading_dates_from_universe(
                universe_ohlcv_df,
                start,
                end,
            )
            logger.info(
                f"Inferred trading calendar: {len(trading_dates)} days "
                f"({trading_dates[0].date()} → {trading_dates[-1].date()})"
            )

        rebalance_calendar = build_rebalance_dates(
            trading_dates,
            start,
            end,
            rebalance_months=args.rebalance_months,
        )

        logger.info(
            f"Rebalance dates: {len(rebalance_calendar)} "
            f"(first={rebalance_calendar[0].date()}, last={rebalance_calendar[-1].date()})"
        )

        assert trading_dates.is_monotonic_increasing
        assert set(rebalance_calendar).issubset(set(trading_dates))
        _validate_rebalance_invariants(rebalance_calendar, trading_dates)

        # History requirement provenance (diagnostics only)
        required_months = required_history_months(
            config.min_history_months,
            config.momentum_lookback_months,
            config.volatility_lookback_months,
        )
        history_start = rebalance_calendar[0] - pd.DateOffset(months=required_months)
        expected_daily = trading_dates[
            (trading_dates >= history_start) & (trading_dates < rebalance_calendar[0])
        ]
        expected_monthly = pd.date_range(
            start=history_start,
            end=rebalance_calendar[0] - pd.Timedelta(days=1),
            freq="ME",
        )
        daily_required = len(expected_daily)
        monthly_required = len(expected_monthly)
        if config.history_gate_mode == "lenient":
            daily_required = int(np.ceil(daily_required * (1.0 - config.history_leniency)))
            monthly_required = int(np.ceil(monthly_required * (1.0 - config.history_leniency)))
        logger.info(
            "History gate provenance: min_history_months=%d, momentum_lookback_months=%d, "
            "volatility_lookback_months=%d, required_months=%d, daily_required=%d, monthly_required=%d, "
            "return_min_samples=2, vol_min_samples=20, pos_month_min_prices=20, pos_month_min_months=2",
            config.min_history_months,
            config.momentum_lookback_months,
            config.volatility_lookback_months,
            required_months,
            daily_required,
            monthly_required,
        )

        benchmark_result = benchmark_result._replace(
            data=benchmark_result.data.reindex(trading_dates).ffill()
        )
        if comparison_benchmark is not None:
            comparison_benchmark = comparison_benchmark._replace(
                data=comparison_benchmark.data.reindex(trading_dates).ffill()
            )

        # Align to trading calendar
        price_data = align_to_calendar(raw_price_data, trading_dates)

        # Sanitize (no forward-fill, just alignment)
        price_data = sanitize_stock_data(price_data, trading_dates)

        # Step 6: Run backtest
        logger.info("Step 6: Running backtest...")
        result = run_backtest(
            config=config,
            price_data=price_data,
            benchmark_prices=benchmark_result.data,
            trading_calendar_prices=calendar_result.data if calendar_result is not None else None,
            trading_dates=trading_dates,
            rebalance_calendar=rebalance_calendar,
        )

        # Step 7: Run validations
        logger.info("Step 7: Running validations...")
        run_all_validations(
            rebalance_calendar=rebalance_calendar,
            trading_dates=trading_dates,
            rebalance_records=result.rebalance_records,
            equity_curve=result.equity_curve,
            price_data=price_data,
        )

        # Step 8: Compute metrics
        logger.info("Step 8: Computing metrics...")
        metrics = compute_metrics(
            equity_curve=result.equity_curve,
            benchmark_curve=result.benchmark_curve,
            time_in_risk_on=result.time_in_risk_on,
            time_in_panic=result.time_in_panic,
            pct_time_cash_invested=result.time_in_cash_invested,
            pct_time_true_cash=result.time_in_true_cash,
            pct_time_cash_regime_daily=result.pct_time_cash_regime_daily,
            pct_time_cash_invested_daily=result.pct_time_cash_invested_daily,
            pct_time_cash_true_daily=result.pct_time_cash_true_daily,
            total_turnover=result.total_turnover,
            total_transaction_costs=result.total_transaction_costs,
            time_in_defensive_momentum=result.time_in_defensive_momentum,
            strategy_version=config.strategy_version,
            enabled_levers=config.enabled_levers,
            # Panic event tracking
            panic_events_count=result.panic_events_count,
            panic_vol_ratio_triggers=result.panic_vol_ratio_triggers,
            panic_dd_triggers=result.panic_dd_triggers,
            panic_response_mode=result.panic_response_mode,
            # Turnover split
            turnover_risk_on=result.turnover_risk_on,
            turnover_defensive=result.turnover_defensive,
            turnover_cash_panic=result.turnover_cash_panic,
            # V3: Cash entry mode
            cash_entry_mode=config.cash_entry_mode.value,
            # V3.1: Cash replacement mode
            cash_replace_mode=config.cash_replace_mode.value,
            time_in_cash_invested=result.time_in_cash_invested,
            pct_time_concentration_gated=result.pct_time_concentration_gated,
            avg_invested_fraction_by_state=result.avg_invested_fraction_by_state,
            avg_holdings_by_state=result.avg_holdings_by_state,
            avg_cash_weight_overall_rebalance=result.avg_cash_weight_overall,
            avg_cash_weight_overall_daily=result.avg_cash_weight_overall_daily,
            avg_cash_weight_by_state=result.avg_cash_weight_by_state,
            avg_cash_weight_by_state_daily=result.avg_cash_weight_by_state_daily,
            avg_invested_weight_by_state_daily=result.avg_invested_weight_by_state_daily,
        )

        # Compute comparison benchmark metrics if provided
        comparison_metrics = None
        strategy_vs_comparison_metrics = None
        comparison_benchmark_stats = None
        if comparison_benchmark is not None:
            logger.info("  Computing comparison benchmark metrics...")
            # Align comparison benchmark to equity curve dates
            comp_data = comparison_benchmark.data.reindex(result.equity_curve.index).dropna()
            if len(comp_data) > 0:
                # Normalize to start at 1.0
                comp_curve = comp_data / comp_data.iloc[0]
                # Standalone comparison-benchmark stats for metrics_vs_comparison.json.
                comparison_metrics = compute_metrics(
                    equity_curve=comp_curve,
                    benchmark_curve=comp_curve,
                    time_in_risk_on=0.0,
                    time_in_panic=0.0,
                    pct_time_cash_invested=0.0,
                    pct_time_true_cash=0.0,
                    pct_time_cash_regime_daily=0.0,
                    pct_time_cash_invested_daily=0.0,
                    pct_time_cash_true_daily=0.0,
                    total_turnover=0.0,
                    total_transaction_costs=0.0,
                    time_in_defensive_momentum=0.0,
                    strategy_version=config.strategy_version,
                    enabled_levers=config.enabled_levers,
                    panic_events_count=result.panic_events_count,
                    panic_vol_ratio_triggers=result.panic_vol_ratio_triggers,
                    panic_dd_triggers=result.panic_dd_triggers,
                    panic_response_mode=result.panic_response_mode,
                    turnover_risk_on=0.0,
                    turnover_defensive=0.0,
                    turnover_cash_panic=0.0,
                    cash_entry_mode=config.cash_entry_mode.value,
                    cash_replace_mode=config.cash_replace_mode.value,
                    time_in_cash_invested=0.0,
                    pct_time_concentration_gated=0.0,
                    avg_invested_fraction_by_state={},
                    avg_holdings_by_state={},
                    avg_cash_weight_overall_rebalance=0.0,
                    avg_cash_weight_overall_daily=0.0,
                    avg_cash_weight_by_state={},
                    avg_cash_weight_by_state_daily={},
                    avg_invested_weight_by_state_daily={},
                )
                tol = 1e-4
                comp_cagr = _compute_cagr(comp_curve)
                comp_vol = _compute_volatility(comp_curve)
                comp_max_dd, _ = _compute_max_drawdown(comp_curve)
                assert abs(comparison_metrics.cagr - comp_cagr) <= tol, (
                    "Comparison CAGR should be computed from comp_curve."
                )
                assert abs(comparison_metrics.volatility - comp_vol) <= tol, (
                    "Comparison volatility should be computed from comp_curve."
                )
                assert abs(comparison_metrics.max_drawdown - comp_max_dd) <= tol, (
                    "Comparison max drawdown should be computed from comp_curve."
                )
                strategy_vs_comparison_metrics = compute_metrics(
                    equity_curve=result.equity_curve,
                    benchmark_curve=comp_curve,
                    time_in_risk_on=result.time_in_risk_on,
                    time_in_panic=result.time_in_panic,
                    pct_time_cash_invested=result.time_in_cash_invested,
                    pct_time_true_cash=result.time_in_true_cash,
                    pct_time_cash_regime_daily=result.pct_time_cash_regime_daily,
                    pct_time_cash_invested_daily=result.pct_time_cash_invested_daily,
                    pct_time_cash_true_daily=result.pct_time_cash_true_daily,
                    total_turnover=result.total_turnover,
                    total_transaction_costs=result.total_transaction_costs,
                    time_in_defensive_momentum=result.time_in_defensive_momentum,
                    strategy_version=config.strategy_version,
                    enabled_levers=config.enabled_levers,
                    panic_events_count=result.panic_events_count,
                    panic_vol_ratio_triggers=result.panic_vol_ratio_triggers,
                    panic_dd_triggers=result.panic_dd_triggers,
                    panic_response_mode=result.panic_response_mode,
                    turnover_risk_on=result.turnover_risk_on,
                    turnover_defensive=result.turnover_defensive,
                    turnover_cash_panic=result.turnover_cash_panic,
                    cash_entry_mode=config.cash_entry_mode.value,
                    cash_replace_mode=config.cash_replace_mode.value,
                    time_in_cash_invested=result.time_in_cash_invested,
                    pct_time_concentration_gated=result.pct_time_concentration_gated,
                    avg_invested_fraction_by_state=result.avg_invested_fraction_by_state,
                    avg_holdings_by_state=result.avg_holdings_by_state,
                    avg_cash_weight_overall_rebalance=result.avg_cash_weight_overall,
                    avg_cash_weight_overall_daily=result.avg_cash_weight_overall_daily,
                    avg_cash_weight_by_state=result.avg_cash_weight_by_state,
                    avg_cash_weight_by_state_daily=result.avg_cash_weight_by_state_daily,
                    avg_invested_weight_by_state_daily=result.avg_invested_weight_by_state_daily,
                )
                comparison_benchmark_stats = compute_metrics(
                    equity_curve=comp_curve,
                    benchmark_curve=comp_curve,
                    time_in_risk_on=0.0,
                    time_in_panic=0.0,
                    pct_time_cash_invested=0.0,
                    pct_time_true_cash=0.0,
                    pct_time_cash_regime_daily=0.0,
                    pct_time_cash_invested_daily=0.0,
                    pct_time_cash_true_daily=0.0,
                    total_turnover=0.0,
                    total_transaction_costs=0.0,
                    time_in_defensive_momentum=0.0,
                    strategy_version=config.strategy_version,
                    enabled_levers=config.enabled_levers,
                    panic_events_count=result.panic_events_count,
                    panic_vol_ratio_triggers=result.panic_vol_ratio_triggers,
                    panic_dd_triggers=result.panic_dd_triggers,
                    panic_response_mode=result.panic_response_mode,
                    turnover_risk_on=0.0,
                    turnover_defensive=0.0,
                    turnover_cash_panic=0.0,
                    cash_entry_mode=config.cash_entry_mode.value,
                    cash_replace_mode=config.cash_replace_mode.value,
                    time_in_cash_invested=0.0,
                    pct_time_concentration_gated=0.0,
                    avg_invested_fraction_by_state={},
                    avg_holdings_by_state={},
                    avg_cash_weight_overall_rebalance=0.0,
                    avg_cash_weight_overall_daily=0.0,
                    avg_cash_weight_by_state={},
                    avg_cash_weight_by_state_daily={},
                    avg_invested_weight_by_state_daily={},
                )

        # Step 9: Export results (including audit reports)
        logger.info("Step 9: Exporting results...")
        export_results(
            result=result,
            metrics=metrics,
            output_dir=config.output_dir,
            benchmark_ticker=benchmark_result.ticker,
            benchmark_is_proxy=benchmark_result.is_proxy,
            benchmark_coverage=benchmark_result.coverage,
            benchmark_data=benchmark_result.data,
            fallback_chain=benchmark_result.fallback_chain,
        )

        # Export comparison metrics if available
        if comparison_metrics is not None:
            import json
            from dataclasses import asdict
            comp_metrics_path = config.output_dir / "metrics_vs_comparison.json"
            git_commit = None
            try:
                repo_root = Path(__file__).resolve().parents[2]
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                git_commit = None
            generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload = asdict(comparison_metrics)
            payload["metadata"] = {
                "series_name": comparison_benchmark.ticker,
                "meaning": "standalone_benchmark_stats",
                "generated_at": generated_at,
                "git_commit": git_commit,
            }
            with open(comp_metrics_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"  Wrote {comp_metrics_path}")

        if strategy_vs_comparison_metrics is not None:
            import json
            from dataclasses import asdict
            strategy_comp_path = config.output_dir / "metrics_strategy_vs_comparison.json"
            with open(strategy_comp_path, "w") as f:
                json.dump(asdict(strategy_vs_comparison_metrics), f, indent=2, default=str)
            logger.info(f"  Wrote {strategy_comp_path}")

        if comparison_benchmark_stats is not None:
            import json
            from dataclasses import asdict
            comparison_stats_path = config.output_dir / "comparison_benchmark_stats.json"
            with open(comparison_stats_path, "w") as f:
                json.dump(asdict(comparison_benchmark_stats), f, indent=2, default=str)
            logger.info(f"  Wrote {comparison_stats_path}")

        # Step 9b: Compute and export rolling 3-year excess return and returns vs Mom50
        rolling_returns_df = None
        if comparison_benchmark is not None:
            logger.info("Step 9b: Computing rolling 3-year excess return...")
            # Normalize benchmark to same scale as equity curve
            bench_aligned = comparison_benchmark.data.reindex(result.equity_curve.index).dropna()
            if len(bench_aligned) > 0:
                bench_normalized = bench_aligned / bench_aligned.iloc[0]

                rolling_df, rolling_stats = compute_rolling_excess_return(
                    equity_curve=result.equity_curve,
                    benchmark_curve=bench_normalized,
                    rebalance_dates=rebalance_calendar,
                    benchmark_name=comparison_benchmark.ticker,
                    window_months=36,
                )

                export_rolling_excess_return(
                    rolling_df=rolling_df,
                    rolling_stats=rolling_stats,
                    output_dir=config.output_dir,
                )

                logger.info(
                    "Strategy equity curve includes cash yield on idle cash when enabled."
                )
                rolling_returns_df = compute_rolling_returns(
                    equity_curve=result.equity_curve,
                    benchmark_curve=bench_normalized,
                    rebalance_dates=rebalance_calendar,
                    window_months=36,
                )
                if not rolling_returns_df.empty:
                    rolling_returns_view = rolling_returns_df.set_index("date")
                    rolling_3y_strategy = rolling_returns_view["strategy_rolling_3y_cagr"]
                    rolling_3y_benchmark = rolling_returns_view["mom50_rolling_3y_cagr"]
                    if np.allclose(rolling_3y_strategy.values, rolling_3y_benchmark.values):
                        raise RuntimeError(
                            "Rolling 3Y strategy and Mom50 series are identical; "
                            "expected distinct rolling computations."
                        )

                    start_dates = pd.to_datetime(rolling_returns_df["rolling_window_start_date"])
                    rolling_window_start_state = result.daily_state.reindex(start_dates).reset_index(drop=True)
                    rolling_window_start_state.index = rolling_returns_view.index
                    rolling_window_start_cash_weight = result.daily_cash_weight.reindex(start_dates).reset_index(drop=True)
                    rolling_window_start_cash_weight.index = rolling_returns_view.index

                    assert rolling_3y_strategy.index.equals(rolling_3y_benchmark.index)
                    assert rolling_returns_view.index.equals(rolling_window_start_state.index)
                    assert rolling_returns_view.index.equals(rolling_window_start_cash_weight.index)
                    if rolling_window_start_state.isna().any():
                        raise RuntimeError("Rolling window start state contains NaN values.")
                    if rolling_window_start_cash_weight.isna().any():
                        raise RuntimeError("Rolling window start cash weight contains NaN values.")

                    rolling_returns_df["rolling_window_start_state"] = rolling_window_start_state.values
                    rolling_returns_df["rolling_window_start_cash_weight"] = rolling_window_start_cash_weight.values

                    window_pct_cash_true = []
                    window_pct_cash_invested = []
                    window_pct_risk_on = []
                    window_pct_defensive = []

                    window_start_dates = pd.to_datetime(rolling_returns_df["rolling_window_start_date"])
                    window_end_dates = pd.to_datetime(rolling_returns_df["date"])

                    for start_date, end_date in zip(window_start_dates, window_end_dates):
                        window_states = result.daily_state.loc[start_date:end_date]
                        total_days = len(window_states)
                        if total_days == 0:
                            window_pct_cash_true.append(np.nan)
                            window_pct_cash_invested.append(np.nan)
                            window_pct_risk_on.append(np.nan)
                            window_pct_defensive.append(np.nan)
                            continue
                        window_pct_cash_true.append((window_states == "CASH_TRUE").mean())
                        window_pct_cash_invested.append((window_states == "CASH_INVESTED").mean())
                        window_pct_risk_on.append((window_states == "RISK_ON").mean())
                        window_pct_defensive.append((window_states == "DEFENSIVE_MOMENTUM").mean())

                    rolling_returns_df["rolling_window_pct_days_cash_true"] = window_pct_cash_true
                    rolling_returns_df["rolling_window_pct_days_cash_invested"] = window_pct_cash_invested
                    rolling_returns_df["rolling_window_pct_days_risk_on"] = window_pct_risk_on
                    rolling_returns_df["rolling_window_pct_days_defensive_momentum"] = window_pct_defensive

                    if rolling_returns_df["rolling_window_pct_days_cash_true"].isna().any():
                        raise RuntimeError("Rolling window cash-true percentages contain NaN values.")

                    expected_len = len(rolling_returns_view)
                    if len(rolling_returns_df) != expected_len:
                        raise RuntimeError(
                            "Rolling returns row count does not match rolling series length."
                        )
                    required_cols = [
                        "rolling_window_start_date",
                        "rolling_window_start_state",
                        "rolling_window_start_cash_weight",
                        "rolling_window_pct_days_cash_true",
                        "rolling_window_pct_days_cash_invested",
                        "rolling_window_pct_days_risk_on",
                        "rolling_window_pct_days_defensive_momentum",
                    ]
                    if rolling_returns_df[required_cols].isna().any().any():
                        raise RuntimeError(
                            "Rolling returns contains missing values in required window diagnostics."
                        )

                    sample = rolling_returns_df[[
                        "date",
                        "rolling_window_start_date",
                        "rolling_window_start_state",
                        "rolling_window_start_cash_weight",
                    ]]
                    sample_rows = pd.concat([sample.head(3), sample.tail(3)])
                    for row in sample_rows.itertuples(index=False):
                        logger.info(
                            "Rolling window mapping: end=%s start=%s state=%s cash_weight=%.2f%%",
                            pd.to_datetime(row.date).date(),
                            pd.to_datetime(row.rolling_window_start_date).date(),
                            row.rolling_window_start_state,
                            row.rolling_window_start_cash_weight * 100.0,
                        )

                export_rolling_returns(
                    rolling_df=rolling_returns_df,
                    output_dir=config.output_dir,
                )
        else:
            logger.warning("No Mom50 benchmark available for rolling return calculations")

        # Print summary to console
        print()
        print_summary(metrics)

        print()
        print("CASH EXPOSURE (Daily, Weight-Based)")
        print("-" * 30)
        print(f"  Avg Cash Weight:      {metrics.avg_cash_weight_overall_daily*100:>7.2f}%")
        if metrics.avg_cash_weight_by_state_daily:
            for state, value in sorted(metrics.avg_cash_weight_by_state_daily.items()):
                print(f"  Cash Weight {state:>12}: {value*100:>7.2f}%")
        print(f"  Cash Regime (daily):  {metrics.pct_time_cash_regime_daily*100:>7.2f}%")
        print(f"  CASH_INVESTED (daily):{metrics.pct_time_cash_invested_daily*100:>7.2f}%")
        print(f"  CASH_TRUE (daily):    {metrics.pct_time_cash_true_daily*100:>7.2f}%")

        # Print comparison summary if available
        if comparison_metrics is not None:
            print()
            print(f"{comparison_benchmark.ticker.replace('_', ' ')}")
            print("-" * 30)
            print(f"  Comparison CAGR:   {comparison_metrics.benchmark_cagr*100:>7.2f}%")
            print(f"  Comparison Vol:    {comparison_metrics.benchmark_volatility*100:>7.2f}%")
            print(f"  Comparison MaxDD:  {comparison_metrics.benchmark_max_drawdown*100:>7.2f}%")
            print(f"  Comparison Sharpe: {comparison_metrics.sharpe_ratio:>7.2f}")
            print(f"  Comparison Sortino:{comparison_metrics.sortino_ratio:>7.2f}")
            print(f"  Comparison Calmar: {comparison_metrics.calmar_ratio:>7.2f}")

            # Strategy-only rolling 3Y CAGR (console output only)
            rolling_df, _ = compute_rolling_excess_return(
                equity_curve=result.equity_curve,
                benchmark_curve=result.equity_curve,
                rebalance_dates=rebalance_calendar,
                benchmark_name="STRATEGY",
                window_months=36,
            )
            if rolling_df.empty:
                print("  Rolling 3Y CAGR:   N/A (insufficient history)")
            else:
                strat_cagr = rolling_df["strategy_cagr"]
                latest_rolling = strat_cagr.iloc[-1]
                print(f"  Rolling 3Y CAGR:   {latest_rolling*100:>7.2f}%")
                print(f"  Rolling 3Y Min:    {strat_cagr.min()*100:>7.2f}%")
                print(f"  Rolling 3Y Median: {strat_cagr.median()*100:>7.2f}%")
                print(f"  Rolling 3Y Max:    {strat_cagr.max()*100:>7.2f}%")

                buckets = {
                    "<0%": (strat_cagr < 0.0),
                    "0-8%": (strat_cagr >= 0.0) & (strat_cagr <= 0.08),
                    "9-12%": (strat_cagr > 0.08) & (strat_cagr <= 0.12),
                    "13-20%": (strat_cagr > 0.12) & (strat_cagr <= 0.20),
                    ">20%": (strat_cagr > 0.20),
                }
                total = len(strat_cagr)
                for label, mask in buckets.items():
                    count = int(mask.sum())
                    pct = (count / total) * 100 if total else 0.0
                    print(f"  Rolling 3Y {label:>5}: {count:>4} ({pct:>5.1f}%)")

            # Rolling 3Y summary for strategy only (console)
            print()
            print("ROLLING 3Y RETURNS (Strategy Only)")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                strat_cagr = rolling_returns_df["strategy_rolling_3y_cagr"]
                latest_rolling = strat_cagr.iloc[-1]
                print(f"  Rolling 3Y CAGR:   {latest_rolling*100:>7.2f}%")
                print(f"  Rolling 3Y Min:    {strat_cagr.min()*100:>7.2f}%")
                print(f"  Rolling 3Y Median: {strat_cagr.median()*100:>7.2f}%")
                print(f"  Rolling 3Y Max:    {strat_cagr.max()*100:>7.2f}%")

                buckets = {
                    "<0%": (strat_cagr < 0.0),
                    "0-8%": (strat_cagr >= 0.0) & (strat_cagr <= 0.08),
                    "9-12%": (strat_cagr > 0.08) & (strat_cagr <= 0.12),
                    "13-20%": (strat_cagr > 0.12) & (strat_cagr <= 0.20),
                    ">20%": (strat_cagr > 0.20),
                }
                total = len(strat_cagr)
                for label, mask in buckets.items():
                    count = int(mask.sum())
                    pct = (count / total) * 100 if total else 0.0
                    print(f"  Rolling 3Y {label:>5}: {count:>4} ({pct:>5.1f}%)")

            # Rolling 3Y summary for Mom50 only (console)
            print()
            print("ROLLING 3Y RETURNS (Mom50 Only)")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                mom_cagr = rolling_returns_df["mom50_rolling_3y_cagr"]
                latest_rolling = mom_cagr.iloc[-1]
                print(f"  Rolling 3Y CAGR:   {latest_rolling*100:>7.2f}%")
                print(f"  Rolling 3Y Min:    {mom_cagr.min()*100:>7.2f}%")
                print(f"  Rolling 3Y Median: {mom_cagr.median()*100:>7.2f}%")
                print(f"  Rolling 3Y Max:    {mom_cagr.max()*100:>7.2f}%")

                buckets = {
                    "<0%": (mom_cagr < 0.0),
                    "0-8%": (mom_cagr >= 0.0) & (mom_cagr <= 0.08),
                    "9-12%": (mom_cagr > 0.08) & (mom_cagr <= 0.12),
                    "13-20%": (mom_cagr > 0.12) & (mom_cagr <= 0.20),
                    ">20%": (mom_cagr > 0.20),
                }
                total = len(mom_cagr)
                for label, mask in buckets.items():
                    count = int(mask.sum())
                    pct = (count / total) * 100 if total else 0.0
                    print(f"  Rolling 3Y {label:>5}: {count:>4} ({pct:>5.1f}%)")

            # Rolling 3Y excess return distribution (console)
            print()
            print("ROLLING 3Y EXCESS RETURNS (Strategy - Mom50)")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                strat_cagr = rolling_returns_df["strategy_rolling_3y_cagr"]
                mom_cagr = rolling_returns_df["mom50_rolling_3y_cagr"]

                common_index = strat_cagr.index.intersection(mom_cagr.index)
                strat_cagr = strat_cagr.reindex(common_index)
                mom_cagr = mom_cagr.reindex(common_index)

                assert strat_cagr.index.equals(mom_cagr.index)
                rolling_3y_excess_cagr = strat_cagr - mom_cagr
                assert rolling_3y_excess_cagr.notna().all()

                print(f"  Mean Excess CAGR: {rolling_3y_excess_cagr.mean()*100:>7.2f}%")
                print(f"  Min Excess CAGR:  {rolling_3y_excess_cagr.min()*100:>7.2f}%")
                print(f"  Median Excess:    {rolling_3y_excess_cagr.median()*100:>7.2f}%")
                print(f"  Max Excess CAGR:  {rolling_3y_excess_cagr.max()*100:>7.2f}%")

                buckets = {
                    "<-10%": (rolling_3y_excess_cagr < -0.10),
                    "-10--5%": (rolling_3y_excess_cagr >= -0.10) & (rolling_3y_excess_cagr < -0.05),
                    "-5-0%": (rolling_3y_excess_cagr >= -0.05) & (rolling_3y_excess_cagr < 0.0),
                    "0-5%": (rolling_3y_excess_cagr >= 0.0) & (rolling_3y_excess_cagr < 0.05),
                    "5-10%": (rolling_3y_excess_cagr >= 0.05) & (rolling_3y_excess_cagr < 0.10),
                    ">10%": (rolling_3y_excess_cagr >= 0.10),
                }
                total = len(rolling_3y_excess_cagr)
                for label, mask in buckets.items():
                    count = int(mask.sum())
                    pct = (count / total) * 100 if total else 0.0
                    print(f"  Rolling 3Y {label:>6}: {count:>4} ({pct:>5.1f}%)")

            print()
            print("ROLLING 3Y EXCESS RETURNS BY START STATE")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                rolling_returns_view = rolling_returns_df.set_index("date")
                excess_cagr = rolling_returns_view["rolling_3y_excess_cagr"]
                start_state = rolling_returns_view["rolling_window_start_state"]

                assert excess_cagr.index.equals(start_state.index)
                assert excess_cagr.notna().all()
                assert start_state.notna().all()

                for state in ["RISK_ON", "DEFENSIVE_MOMENTUM", "CASH_INVESTED", "CASH_TRUE"]:
                    state_mask = start_state == state
                    if not state_mask.any():
                        continue
                    state_excess = excess_cagr[state_mask]
                    print(f"  STATE: {state}")
                    print(f"    Mean / Median / Min / Max: "
                          f"{state_excess.mean()*100:>6.2f}% / "
                          f"{state_excess.median()*100:>6.2f}% / "
                          f"{state_excess.min()*100:>6.2f}% / "
                          f"{state_excess.max()*100:>6.2f}%")
                    print(f"    Window Count: {len(state_excess)}")

            print()
            print("ROLLING 3Y EXCESS RETURNS BY START CASH WEIGHT")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                excess_cagr = rolling_returns_df["rolling_3y_excess_cagr"]
                start_cash_weight = rolling_returns_df["rolling_window_start_cash_weight"]
                assert start_cash_weight.index.equals(excess_cagr.index)
                assert start_cash_weight.notna().all()

                bins = [
                    ("0-5%", (start_cash_weight >= 0.0) & (start_cash_weight < 0.05)),
                    ("5-20%", (start_cash_weight >= 0.05) & (start_cash_weight < 0.20)),
                    ("20-50%", (start_cash_weight >= 0.20) & (start_cash_weight < 0.50)),
                    ("50-80%", (start_cash_weight >= 0.50) & (start_cash_weight < 0.80)),
                    ("80-100%", (start_cash_weight >= 0.80) & (start_cash_weight <= 1.00)),
                ]
                for label, mask in bins:
                    if not mask.any():
                        continue
                    values = excess_cagr[mask]
                    print(f"  BIN: {label}")
                    print(f"    Mean / Median / Min / Max: "
                          f"{values.mean()*100:>6.2f}% / "
                          f"{values.median()*100:>6.2f}% / "
                          f"{values.min()*100:>6.2f}% / "
                          f"{values.max()*100:>6.2f}%")
                    print(f"    Window Count: {len(values)}")

            print()
            print("ROLLING 3Y EXCESS RETURNS BY CASH_TRUE TIME-IN-WINDOW")
            print("-" * 30)
            if rolling_returns_df is None or rolling_returns_df.empty:
                print("  N/A (insufficient history)")
            else:
                excess_cagr = rolling_returns_df["rolling_3y_excess_cagr"]
                cash_true_pct = rolling_returns_df["rolling_window_pct_days_cash_true"]
                assert cash_true_pct.index.equals(excess_cagr.index)
                assert cash_true_pct.notna().all()

                bins = [
                    ("0-10%", (cash_true_pct >= 0.0) & (cash_true_pct < 0.10)),
                    ("10-30%", (cash_true_pct >= 0.10) & (cash_true_pct < 0.30)),
                    ("30-60%", (cash_true_pct >= 0.30) & (cash_true_pct < 0.60)),
                    ("60-100%", (cash_true_pct >= 0.60) & (cash_true_pct <= 1.00)),
                ]
                for label, mask in bins:
                    if not mask.any():
                        continue
                    values = excess_cagr[mask]
                    print(f"  BIN: {label}")
                    print(f"    Mean / Median / Min / Max: "
                          f"{values.mean()*100:>6.2f}% / "
                          f"{values.median()*100:>6.2f}% / "
                          f"{values.min()*100:>6.2f}% / "
                          f"{values.max()*100:>6.2f}%")
                    print(f"    Window Count: {len(values)}")

            # Concentration / capacity diagnostics
            print()
            print("CONCENTRATION / CAPACITY FEASIBILITY")
            print("-" * 30)

            rebalance_rows = []
            for record in result.rebalance_records:
                output_state = record.state.name if hasattr(record.state, "name") else str(record.state)
                evaluated_branches = []
                risk_on_selected_count = np.nan
                defensive_selected_count_eval = np.nan
                if record.requested_state == "RISK_ON":
                    evaluated_branches.append("risk_on")
                    risk_on_selected_count = record.selected_count
                elif record.requested_state == "DEFENSIVE_MOMENTUM":
                    evaluated_branches.append("defensive")
                    defensive_selected_count_eval = record.defensive_selected_count
                elif record.requested_state in {"CASH_TRUE", "CASH_INVESTED", "CASH"}:
                    if config.cash_replace_mode == CashReplaceMode.DEFENSIVE:
                        evaluated_branches.append("cash_replace")
                        defensive_selected_count_eval = record.defensive_selected_count

                gate_label = record.concentration_gate_reason
                if gate_label == "risk_on_insufficient_concentration":
                    gate_label = "risk_on_insufficient_concentration"
                elif gate_label == "defensive_insufficient_concentration":
                    gate_label = "defensive_insufficient_concentration"

                rebalance_rows.append({
                    "date": record.date,
                    "requested_state": record.requested_state,
                    "output_state": output_state,
                    "concentration_forced_cash": record.concentration_forced_cash,
                    "concentration_gate_reason": gate_label,
                    "max_weight": record.max_weight,
                    "min_required": record.min_required,
                    "evaluated_branches": ",".join(evaluated_branches) if evaluated_branches else "none",
                    "risk_on_selected_count": risk_on_selected_count,
                    "defensive_selected_count_eval": defensive_selected_count_eval,
                    "selected_count": record.selected_count,
                    "defensive_selected_count": record.defensive_selected_count,
                    "invested_fraction": record.invested_fraction,
                    "cash_weight": 1.0 - record.invested_fraction,
                })
            rebalance_df = pd.DataFrame(rebalance_rows)

            eligibility_rows = []
            for record in result.eligibility_records:
                eligibility_rows.append({
                    "date": record.date,
                    "universe_count": record.universe_count,
                    "data_eligible_count": record.data_eligible_count,
                    "filter_passed_count": record.filter_passed_count,
                    "insufficient_history": record.insufficient_history,
                    "negative_returns": record.negative_returns,
                    "low_positive_months": record.low_positive_months,
                    "high_drawdown": record.high_drawdown,
                    "score_computation_failed": record.score_computation_failed,
                    "not_in_price_data": record.not_in_price_data,
                })
            eligibility_df = pd.DataFrame(eligibility_rows)
            if not eligibility_df.empty:
                eligibility_df = (
                    eligibility_df.sort_values("date")
                    .drop_duplicates(subset=["date"], keep="last")
                )

            if not rebalance_df.empty:
                merged_df = rebalance_df.merge(
                    eligibility_df,
                    on="date",
                    how="left",
                )
            else:
                merged_df = rebalance_df.copy()

            merged_df.to_csv(config.output_dir / "concentration_diagnostics.csv", index=False)

            concentration_gate_df = merged_df.copy()
            concentration_gate_df.rename(
                columns={
                    "date": "rebalance_date",
                },
                inplace=True,
            )
            concentration_gate_df.to_csv(
                config.output_dir / "concentration_gate_by_rebalance.csv",
                index=False,
            )

            total_rebalances = len(merged_df)
            if total_rebalances == 0:
                print("  N/A (no rebalances)")
            else:
                reason_counts = merged_df["concentration_gate_reason"].value_counts()
                for reason_label in [
                    "risk_on_insufficient_concentration",
                    "defensive_insufficient_concentration",
                    "cap_left_cash",
                ]:
                    count = int(reason_counts.get(reason_label, 0))
                    pct = (count / total_rebalances) * 100
                    print(f"{reason_label:<30} {count:>5d} ({pct:>5.1f}%)")

                other_reasons = [
                    r for r in reason_counts.index
                    if r not in {
                        "risk_on_insufficient_concentration",
                        "defensive_insufficient_concentration",
                        "cap_left_cash",
                        "",
                    }
                ]
                if other_reasons:
                    for reason_label in other_reasons:
                        count = int(reason_counts.get(reason_label, 0))
                        pct = (count / total_rebalances) * 100
                        print(f"{reason_label:<30} {count:>5d} ({pct:>5.1f}%)")

                cash_true_mask = merged_df["output_state"].isin(["CASH_TRUE", "CASH"])
                cash_true_df = merged_df[cash_true_mask]
                if not cash_true_df.empty:
                    concentration_pct = (cash_true_df["concentration_forced_cash"].mean() * 100)
                    requested_cash_mask = cash_true_df["requested_state"].isin(
                        ["CASH_TRUE", "CASH_INVESTED", "CASH"]
                    )
                    requested_cash_pct = requested_cash_mask.mean() * 100
                    print(f"\nCASH_TRUE concentration_forced_cash: {concentration_pct:>5.1f}%")
                    print(f"CASH_TRUE requested cash-regime: {requested_cash_pct:>5.1f}%")
                else:
                    print("\nCASH_TRUE concentration_forced_cash: N/A")

                forced_df = merged_df[merged_df["concentration_forced_cash"]]
                if not forced_df.empty:
                    risk_on_forced = forced_df[
                        forced_df["concentration_gate_reason"] == "risk_on_insufficient_concentration"
                    ]
                    if not risk_on_forced.empty:
                        risk_shortfall = (
                            risk_on_forced["min_required"] - risk_on_forced["risk_on_selected_count"]
                        )
                        print(
                            "\nRisk-on shortfall (min_required - selected_count): "
                            f"mean={risk_shortfall.mean():.2f}, "
                            f"median={risk_shortfall.median():.2f}, "
                            f"max={risk_shortfall.max():.2f}"
                        )
                    defensive_forced = forced_df[
                        forced_df["concentration_gate_reason"] == "defensive_insufficient_concentration"
                    ]
                    if not defensive_forced.empty:
                        defensive_shortfall = (
                            defensive_forced["min_required"] - defensive_forced["defensive_selected_count_eval"]
                        )
                        print(
                            "Defensive shortfall (min_required - defensive_selected_count): "
                            f"mean={defensive_shortfall.mean():.2f}, "
                            f"median={defensive_shortfall.median():.2f}, "
                            f"max={defensive_shortfall.max():.2f}"
                        )

                    eligibility_cols = [
                        "insufficient_history",
                        "negative_returns",
                        "low_positive_months",
                        "high_drawdown",
                        "score_computation_failed",
                        "not_in_price_data",
                    ]
                    eligible_means = {}
                    eligible_max = {}
                    for col in eligibility_cols:
                        if col in forced_df.columns:
                            eligible_means[col] = forced_df[col].mean()
                            eligible_max[col] = forced_df[col].max()
                    if eligible_means:
                        ranked = sorted(
                            eligible_means.items(),
                            key=lambda kv: (kv[1], eligible_max.get(kv[0], 0)),
                            reverse=True,
                        )
                        print("\nTop eligibility contributors (mean count):")
                        for col, mean_val in ranked[:3]:
                            max_val = eligible_max.get(col, 0)
                            print(f"  {col}: mean={mean_val:.2f}, max={max_val:.2f}")
                else:
                    print("\nNo concentration-forced events found.")

                if not forced_df.empty:
                    print("\nConcentration-forced debug (first 20)")
                    debug_rows = forced_df.head(20)
                    for _, row in debug_rows.iterrows():
                        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                        data_eligible = row.get("data_eligible_count", np.nan)
                        filter_passed = row.get("filter_passed_count", np.nan)
                        print(
                            f"{date_str} | {row['requested_state']} -> {row['output_state']} | "
                            f"{row['concentration_gate_reason']} | min_required={row['min_required']} | "
                            f"selected={row['risk_on_selected_count']} | defensive_selected={row['defensive_selected_count_eval']} | "
                            f"data_eligible={data_eligible} | filter_passed={filter_passed}"
                        )

            # Score computation failure diagnostics
            print()
            print("SCORE COMPUTATION FAILURE BREAKDOWN")
            print("-" * 30)

            funnel_rows = []
            for record in result.eligibility_funnel_records:
                funnel_rows.append({
                    "rebalance_date": record.date,
                    "universe_count": record.universe_count,
                    "data_eligible_count": record.data_eligible_count,
                    "warmup_skip": record.warmup_skip,
                    "insufficient_history_excluded": record.insufficient_history_excluded,
                    "history_eligible_count": record.history_eligible,
                    "history_ineligible_count": record.history_ineligible_count,
                    "score_attempted_count": record.score_attempted_count,
                    "score_success_count": record.score_success_count,
                    "score_computation_failed_after_gate": record.score_computation_failed_after_gate,
                    "filter_passed_count": record.filter_passed_count,
                    "selected_count": record.selected_count,
                    "invested_count": record.invested_count,
                    "final_state": record.final_state,
                    "concentration_forced_cash": record.concentration_forced_cash,
                })
            funnel_df = pd.DataFrame(funnel_rows)
            if not funnel_df.empty:
                funnel_df.to_csv(
                    config.output_dir / "eligibility_funnel_by_rebalance.csv",
                    index=False,
                )

            score_rows = []
            for record in result.score_failure_records:
                score_rows.append({
                    "rebalance_date": record.date,
                    "state_requested": record.requested_state,
                    "state_final": record.output_state,
                    "reason_code": record.reason_code,
                    "count": record.count,
                    "sample_symbols": record.sample_symbols,
                    "sample_note": record.sample_note,
                })

            score_df = pd.DataFrame(score_rows)
            if not score_df.empty:
                score_df.to_csv(
                    config.output_dir / "score_failure_diagnostics.csv",
                    index=False,
                )

                reason_pivot = score_df.pivot_table(
                    index="rebalance_date",
                    columns="reason_code",
                    values="count",
                    aggfunc="sum",
                ).fillna(0)

                if not funnel_df.empty:
                    merged_dates = pd.to_datetime(funnel_df["rebalance_date"])
                    reason_pivot = reason_pivot.reindex(merged_dates).fillna(0)

                reason_pivot.index = pd.to_datetime(reason_pivot.index)
                total_failures = reason_pivot.sum(axis=1)

                score_summary = pd.DataFrame({
                    "rebalance_date": reason_pivot.index,
                    "score_computation_failed": total_failures.values,
                })
                if not funnel_df.empty:
                    elig_cols = [
                        "rebalance_date",
                        "universe_count",
                        "data_eligible_count",
                        "filter_passed_count",
                    ]
                    score_summary = score_summary.merge(
                        funnel_df[elig_cols],
                        on="rebalance_date",
                        how="left",
                    )

                for reason_code in reason_pivot.columns:
                    score_summary[reason_code] = reason_pivot[reason_code].values

                sample_map = {}
                for reason_code in reason_pivot.columns:
                    sample_map[reason_code] = (
                        score_df[score_df["reason_code"] == reason_code]
                        .set_index("rebalance_date")["sample_symbols"]
                    )
                    score_summary[f"sample_{reason_code}"] = (
                        score_summary["rebalance_date"].map(sample_map[reason_code]).fillna("")
                    )

                score_summary.to_csv(
                    config.output_dir / "score_failure_by_rebalance.csv",
                    index=False,
                )

                reason_means = reason_pivot.mean().sort_values(ascending=False)
                top_reasons = reason_means.head(5)
                if not top_reasons.empty:
                    print("Across all rebalances (top 5 reasons):")
                    for reason, mean_val in top_reasons.items():
                        median_val = reason_pivot[reason].median()
                        max_val = reason_pivot[reason].max()
                        print(
                            f"  {reason}: mean={mean_val:.2f}, "
                            f"median={median_val:.2f}, max={max_val:.2f}"
                        )

                total_rebalances_all = len(reason_pivot)
                zero_fail_reb = int((total_failures == 0).sum())
                nonzero_fail_reb = int((total_failures > 0).sum())
                print(f"\nRebalances total: {total_rebalances_all}")
                print(f"Rebalances with zero failures: {zero_fail_reb}")
                print(f"Rebalances with >0 failures: {nonzero_fail_reb}")

                if (total_failures > 0).any():
                    top_dates = total_failures.sort_values(ascending=False).head(3)
                    print("Top 3 rebalance dates by failures:")
                    for date, total in top_dates.items():
                        top_reason = reason_pivot.loc[date].idxmax()
                        top_count = reason_pivot.loc[date, top_reason]
                        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                        print(
                            f"  {date_str} | total_fail={int(total)} | "
                            f"top_reason={top_reason} ({int(top_count)})"
                        )
                else:
                    print("Top 3 rebalance dates by failures: N/A")

                if not merged_df.empty:
                    forced_dates = pd.to_datetime(
                        merged_df.loc[merged_df["concentration_forced_cash"], "date"]
                    )
                    forced_pivot = reason_pivot.reindex(forced_dates).fillna(0)
                    forced_total = forced_pivot.sum(axis=1)
                    forced_zero = int((forced_total == 0).sum())
                    forced_nonzero = int((forced_total > 0).sum())
                    print("\nConcentration-forced subset:")
                    print(f"  Rebalances with zero failures: {forced_zero}")
                    print(f"  Rebalances with >0 failures: {forced_nonzero}")
                    if (forced_total > 0).any():
                        top_forced = forced_total.sort_values(ascending=False).head(3)
                        print("  Top 3 concentration-forced dates by failures:")
                        for date, total in top_forced.items():
                            top_reason = forced_pivot.loc[date].idxmax()
                            top_count = forced_pivot.loc[date, top_reason]
                            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                            print(
                                f"    {date_str} | total_fail={int(total)} | "
                                f"top_reason={top_reason} ({int(top_count)})"
                            )
                    else:
                        print("  Top 3 concentration-forced dates by failures: N/A")
            else:
                print("  N/A (no score failures recorded)")

            if not funnel_df.empty:
                print()
                print("ELIGIBILITY FUNNEL SUMMARY")
                print("-" * 30)
                total_rebalances = len(funnel_df)
                warmup_skipped = int(funnel_df["warmup_skip"].sum())
                non_warm = funnel_df[~funnel_df["warmup_skip"]]
                filter_zero = int((non_warm["filter_passed_count"] == 0).sum()) if not non_warm.empty else 0
                forced_cash_zero = int(
                    ((non_warm["concentration_forced_cash"]) & (non_warm["filter_passed_count"] == 0)).sum()
                ) if not non_warm.empty else 0
                print(f"Rebalances total: {total_rebalances}")
                print(f"Warmup skipped rebalances: {warmup_skipped}")
                print(f"Rebalances with filter_passed_count == 0: {filter_zero}")
                print(f"Forced cash due to zero eligibility: {forced_cash_zero}")

                non_warm = funnel_df[~funnel_df["warmup_skip"]]
                zero_selection = non_warm[non_warm["selected_count"] == 0] if not non_warm.empty else pd.DataFrame()
                zero_count = int(len(zero_selection))
                zero_pct = (zero_count / len(non_warm) * 100.0) if len(non_warm) else 0.0
                history_gate_enabled = config.history_gate_mode in ("strict", "lenient")
                warmup_zero = int(funnel_df[funnel_df["warmup_skip"] & (funnel_df["selected_count"] == 0)].shape[0])
                history_empty = int(
                    zero_selection[
                        history_gate_enabled
                        & (zero_selection["history_eligible_count"] == 0)
                    ].shape[0]
                ) if history_gate_enabled and not zero_selection.empty else 0
                eligibility_empty = int(
                    zero_selection[
                        (zero_selection["history_eligible_count"] > 0)
                        & (zero_selection["filter_passed_count"] == 0)
                    ].shape[0]
                ) if not zero_selection.empty else 0
                capacity_zero = int(
                    zero_selection[zero_selection["concentration_forced_cash"]].shape[0]
                ) if not zero_selection.empty else 0

                print()
                print("ZERO-SELECTION REBALANCES")
                print("-" * 30)
                print(f"Count: {zero_count} ({zero_pct:>5.1f}%)")
                print(f"  warmup: {warmup_zero}")
                print(f"  history_empty: {history_empty}")
                print(f"  eligibility_empty: {eligibility_empty}")
                print(f"  capacity: {capacity_zero}")

                # Per-rebalance funnel lines + diagnostics JSON payload
                rebalance_rows = []
                for record in result.rebalance_records:
                    rebalance_rows.append({
                        "rebalance_date": record.date,
                        "requested_state": record.requested_state,
                        "output_state": record.state.name,
                        "selected_count": record.selected_count,
                        "defensive_selected_count": record.defensive_selected_count,
                        "min_required": record.min_required,
                        "invested_fraction": record.invested_fraction,
                    })
                rebalance_df = pd.DataFrame(rebalance_rows)
                if not rebalance_df.empty:
                    rebalance_df["rebalance_date"] = pd.to_datetime(rebalance_df["rebalance_date"])
                merged_funnel = funnel_df.merge(
                    rebalance_df,
                    on="rebalance_date",
                    how="left",
                    suffixes=("_funnel", "_rebalance"),
                )
                def _jsonify_short_samples(samples):
                    safe = []
                    for sample in samples:
                        safe_item = {}
                        for key, value in sample.items():
                            if isinstance(value, (pd.Timestamp, datetime.date)):
                                safe_item[key] = str(value)
                            else:
                                safe_item[key] = value
                        safe.append(safe_item)
                    return safe

                if not merged_funnel.empty:
                    selected_count = merged_funnel["selected_count_funnel"]
                    defensive_selected_count = merged_funnel["defensive_selected_count"]
                    requested_state = merged_funnel["requested_state"]
                    selected_eval = np.where(
                        requested_state == "RISK_ON",
                        selected_count,
                        defensive_selected_count,
                    )
                    capacity_pass = np.where(
                        (merged_funnel["filter_passed_count"] > 0) & (selected_eval > 0),
                        selected_eval >= merged_funnel["min_required"],
                        np.nan,
                    )
                    merged_funnel["capacity_min_required"] = merged_funnel["min_required"]
                    merged_funnel["capacity_pass"] = capacity_pass
                    history_gate_enabled = config.history_gate_mode in ("strict", "lenient")
                    empty_reason = np.where(
                        merged_funnel["warmup_skip"],
                        "WARMUP_ACTIVE",
                        np.where(
                            history_gate_enabled & (merged_funnel["history_eligible_count"] == 0),
                            "HISTORY_GATE_EMPTY",
                            np.where(
                                merged_funnel["score_success_count"] == 0,
                                "SCORE_EMPTY",
                                np.where(
                                    merged_funnel["filter_passed_count"] == 0,
                                    "FILTER_EMPTY",
                                    "NOT_EMPTY",
                                ),
                            ),
                        ),
                    )
                    merged_funnel["empty_reason"] = np.where(
                        selected_count == 0, empty_reason, "NOT_EMPTY"
                    )

                    print()
                    print("ELIGIBILITY FUNNEL (Per Rebalance)")
                    print("-" * 30)
                    for _, row in merged_funnel.iterrows():
                        date_str = pd.to_datetime(row["rebalance_date"]).strftime("%Y-%m-%d")
                        cap_pass = row["capacity_pass"]
                        cap_pass_str = "NA" if pd.isna(cap_pass) else ("Y" if cap_pass else "N")
                        print(
                            f"{date_str} | universe={int(row['universe_count'])} | "
                            f"history_ok={int(row['history_eligible_count'])} | "
                            f"history_bad={int(row['history_ineligible_count'])} | "
                            f"score_attempted={int(row['score_attempted_count'])} | "
                            f"score_success={int(row['score_success_count'])} | "
                            f"filter_passed={int(row['filter_passed_count'])} | "
                            f"selected={int(row['selected_count_funnel'])} | "
                            f"min_required={int(row['capacity_min_required'])} | "
                            f"capacity_pass={cap_pass_str} | "
                            f"empty_reason={row['empty_reason']}"
                        )

                # Score rejection reasons for SCORE_EMPTY dates
                rejection_by_date = {
                    pd.to_datetime(r.date): r for r in result.score_rejection_records
                }
                score_rejection_rows = []
                score_rejection_json = {}
                short_window_rows = []
                if not merged_funnel.empty:
                    print()
                    for _, row in merged_funnel.iterrows():
                        if row["empty_reason"] != "SCORE_EMPTY":
                            continue
                        rebalance_date = pd.to_datetime(row["rebalance_date"])
                        rec = rejection_by_date.get(rebalance_date)
                        if rec is None:
                            logger.warning(
                                "Missing score rejection record for SCORE_EMPTY date %s",
                                rebalance_date.strftime("%Y-%m-%d"),
                            )
                            continue

                        full_counts = {k: int(rec.rejection_counts.get(k, 0)) for k in REJECTION_REASON_CODES}
                        attempted = int(rec.attempted)
                        ok = int(rec.ok)
                        sum_counts = sum(full_counts.values())

                        print(f"[{rebalance_date.strftime('%Y-%m-%d')}] SCORE_EMPTY root causes:")
                        print(f"  attempted={attempted}, ok={ok}")
                        hist_parts = ", ".join([f"{k}={v}" for k, v in full_counts.items()])
                        print(f"  histogram: {hist_parts}")
                        print(f"  sum(histogram)={sum_counts}")

                        if sum_counts != attempted:
                            delta = attempted - sum_counts
                            missing_keys = [k for k in REJECTION_REASON_CODES if k not in rec.rejection_counts]
                            extra_keys = [k for k in rec.rejection_counts if k not in REJECTION_REASON_CODES]
                            logger.error(
                                "SCORE_EMPTY histogram mismatch on %s (attempted=%d sum=%d delta=%d)",
                                rebalance_date.strftime("%Y-%m-%d"),
                                attempted,
                                sum_counts,
                                delta,
                            )
                            if delta != 0:
                                full_counts["REJ_OTHER"] = full_counts.get("REJ_OTHER", 0) + delta
                                sum_counts = sum(full_counts.values())
                                logger.error(
                                    "SCORE_EMPTY forcing delta into REJ_OTHER on %s (new_sum=%d)",
                                    rebalance_date.strftime("%Y-%m-%d"),
                                    sum_counts,
                                )
                            if missing_keys or extra_keys:
                                logger.error(
                                    "SCORE_EMPTY histogram keys on %s missing=%s extra=%s",
                                    rebalance_date.strftime("%Y-%m-%d"),
                                    ",".join(missing_keys) if missing_keys else "none",
                                    ",".join(extra_keys) if extra_keys else "none",
                                )
                            sample_pairs = []
                            for reason, samples in (rec.rejection_samples or {}).items():
                                for sample in samples:
                                    sample_pairs.append(f"{reason}:{sample}")
                            if sample_pairs:
                                logger.error(
                                    "SCORE_EMPTY sample outcomes on %s: %s",
                                    rebalance_date.strftime("%Y-%m-%d"),
                                    ", ".join(sample_pairs[:20]),
                                )

                        exception_hist = rec.exception_hist or {}
                        if exception_hist:
                            top_exc = sorted(exception_hist.items(), key=lambda kv: kv[1], reverse=True)[:3]
                            exc_parts = ", ".join([f"{k}={v}" for k, v in top_exc])
                        else:
                            exc_parts = "none"
                        print(f"  exceptions: {exc_parts}")

                        contexts = rec.rejection_sample_contexts.get("REJ_OTHER", []) if rec.rejection_sample_contexts else []
                        samples = rec.rejection_samples.get("REJ_OTHER", []) if rec.rejection_samples else []
                        print(f"  REJ_OTHER count: {full_counts.get('REJ_OTHER', 0)}")
                        if full_counts.get("REJ_OTHER", 0) > 0:
                            if contexts:
                                print(f"  REJ_OTHER samples: {', '.join(contexts[:10])}")
                            elif samples:
                                print(f"  REJ_OTHER samples: {', '.join(samples[:10])}")

                        short_samples = rec.short_window_samples or []
                        if short_samples:
                            required_vals = [s.get("required_len", 0) for s in short_samples]
                            actual_vals = [s.get("actual_len", 0) for s in short_samples]
                            actual_zero = sum(1 for v in actual_vals if v == 0)
                            uses_month_anchor = any(s.get("slicing_mode") == "monthly_resample" for s in short_samples)
                            print("  REJ_TOO_SHORT_WINDOW summary:")
                            print(
                                f"    required_len min/median/max: "
                                f"{int(np.min(required_vals))}/{int(np.median(required_vals))}/{int(np.max(required_vals))}"
                            )
                            print(
                                f"    actual_len   min/median/max: "
                                f"{int(np.min(actual_vals))}/{int(np.median(actual_vals))}/{int(np.max(actual_vals))}"
                            )
                            print(
                                f"    actual_len==0: {actual_zero} "
                                f"({(actual_zero / max(len(actual_vals), 1)) * 100:>5.1f}%)"
                            )
                            print(f"    uses_month_anchor: {uses_month_anchor}")

                            top_samples = sorted(
                                short_samples,
                                key=lambda s: (s.get("actual_len", 0), s.get("required_len", 0)),
                            )[:5]
                            for s in top_samples:
                                print(
                                    "    "
                                    f"{s.get('symbol')} | required={s.get('required_len')} "
                                    f"actual={s.get('actual_len')} "
                                    f"window_end={s.get('window_end_date')} "
                                    f"last_price={s.get('last_available_price_date')} "
                                    f"note={s.get('note')}"
                                )
                                missing_components = s.get("missing_components") or []
                                if missing_components:
                                    parts = []
                                    for comp in missing_components[:3]:
                                        parts.append(
                                            f"{comp.get('component')} "
                                            f"(req={comp.get('required_len')}, actual={comp.get('actual_len')})"
                                        )
                                    print(f"      missing_components: {', '.join(parts)}")
                                if s.get("required_monthly_points") is not None:
                                    print(
                                        f"      monthly_points={s.get('monthly_points')} "
                                        f"required_monthly_points={s.get('required_monthly_points')}"
                                    )

                            for sample in short_samples[:200]:
                                short_window_rows.append({
                                    "date": rebalance_date.strftime("%Y-%m-%d"),
                                    "symbol": sample.get("symbol"),
                                    "required_len": sample.get("required_len"),
                                    "actual_len": sample.get("actual_len"),
                                    "window_start": sample.get("window_start_date"),
                                    "window_end": sample.get("window_end_date"),
                                    "last_price_date": sample.get("last_available_price_date"),
                                    "earliest_price_date": sample.get("earliest_available_price_date"),
                                    "window_label": sample.get("window_label"),
                                    "slicing_mode": sample.get("slicing_mode"),
                                    "note": sample.get("note"),
                                })

                        row_dict = {
                            "date": rebalance_date.strftime("%Y-%m-%d"),
                            "attempted": attempted,
                            "ok": ok,
                            "reason_count_json": json.dumps(full_counts),
                        }
                        for reason in REJECTION_REASON_CODES:
                            row_dict[reason] = full_counts.get(reason, 0)
                        score_rejection_rows.append(row_dict)
                        safe_short_samples = _jsonify_short_samples(rec.short_window_samples or [])
                        score_rejection_json[rebalance_date.strftime("%Y-%m-%d")] = {
                            "attempted": attempted,
                            "ok": ok,
                            "rejections_full": full_counts,
                            "exceptions": exception_hist,
                            "samples": rec.rejection_sample_contexts or {},
                            "short_window_samples": safe_short_samples,
                        }

                if score_rejection_rows:
                    score_rejection_df = pd.DataFrame(score_rejection_rows)
                    score_rejection_df.to_csv(
                        config.output_dir / "score_rejection_reasons.csv",
                        index=False,
                    )
                    with open(config.output_dir / "score_rejection_reasons_by_date.json", "w") as f:
                        json.dump(score_rejection_json, f, indent=2)

                if short_window_rows:
                    short_window_df = pd.DataFrame(short_window_rows)
                    short_window_df.to_csv(
                        config.output_dir / "score_short_window_samples.csv",
                        index=False,
                    )

                # Rebalance-level cash driver counts
                cash_states = {"CASH_TRUE", "CASH_INVESTED", "CASH"}
                regime_requested_cash = merged_funnel["requested_state"].isin(cash_states)
                zero_selection_flag = merged_funnel["selected_count_funnel"] == 0
                cap_left_cash = (
                    (merged_funnel["invested_fraction"] < 0.999)
                    & (~merged_funnel["output_state"].isin(cash_states))
                    & (merged_funnel["selected_count_funnel"] > 0)
                )
                print()
                print("REBALANCE CASH DRIVERS")
                print("-" * 30)
                print(f"total_rebalances: {len(merged_funnel)}")
                print(f"regime_requested_cash_count: {int(regime_requested_cash.sum())}")
                print(f"zero_selection_count: {int(zero_selection_flag.sum())}")
                print(f"cap_left_cash_count: {int(cap_left_cash.sum())}")
                print(f"both_regime_and_zero_selection_count: {int((regime_requested_cash & zero_selection_flag).sum())}")
                print(f"both_cap_and_regime_count: {int((cap_left_cash & regime_requested_cash).sum())}")

                # Daily cash decomposition
                daily_state = result.daily_state
                daily_invested = result.daily_invested_weight
                daily_cash = result.daily_cash_weight

                if not daily_state.empty:
                    zero_selection_by_date = pd.Series(index=daily_state.index, dtype=bool)
                    record_idx = 0
                    current_zero = False
                    for current_date in daily_state.index:
                        if record_idx < len(result.rebalance_records) and current_date >= result.rebalance_records[record_idx].date:
                            rec = result.rebalance_records[record_idx]
                            current_zero = (rec.selected_count == 0) and (rec.defensive_selected_count == 0)
                            record_idx += 1
                        zero_selection_by_date[current_date] = current_zero

                    cash_regime_mask = daily_state.isin(cash_states)
                    zero_selection_mask = zero_selection_by_date & (~cash_regime_mask)
                    cap_mask = (
                        (~cash_regime_mask)
                        & (daily_invested > 0)
                        & (daily_invested < 1)
                        & (~zero_selection_mask)
                    )

                    cash_regime = daily_cash.where(cash_regime_mask, 0.0)
                    cash_zero = daily_cash.where(zero_selection_mask, 0.0)
                    cash_caps = daily_cash.where(cap_mask, 0.0)

                    cash_total = daily_cash
                    cash_sum_parts = cash_regime + cash_zero + cash_caps
                    cash_residual = (cash_total - cash_sum_parts).abs()

                    avg_cash_total = float(cash_total.mean())
                    avg_cash_regime = float(cash_regime.mean())
                    avg_cash_caps = float(cash_caps.mean())
                    avg_cash_zero = float(cash_zero.mean())
                    avg_cash_residual = float(cash_residual.mean())
                    max_cash_residual = float(cash_residual.max())

                    print()
                    print("CASH DECOMPOSITION (Daily, Weight-Based)")
                    print("-" * 30)
                    print(f"Avg Cash Total:         {avg_cash_total*100:>7.2f}%")
                    print(f"Avg Cash From Regime:   {avg_cash_regime*100:>7.2f}%")
                    print(f"Avg Cash From Caps:     {avg_cash_caps*100:>7.2f}%")
                    print(f"Avg Cash From Zero-Selection: {avg_cash_zero*100:>7.2f}%")
                    print(f"Sanity mean(abs(total - sum(parts))): {avg_cash_residual:.6f}")
                    if max_cash_residual > 1e-6:
                        logger.error(
                            "Cash decomposition residual exceeds tolerance: max=%.8f",
                            max_cash_residual,
                        )

                    # Persist diagnostics to metrics.json
                    metrics_path = config.output_dir / "metrics.json"
                    if metrics_path.exists():
                        import json
                        with open(metrics_path, "r") as f:
                            metrics_data = json.load(f)
                        metrics_data["diagnostics"] = metrics_data.get("diagnostics", {})
                        funnel_payload = merged_funnel[[
                            "rebalance_date",
                            "universe_count",
                            "history_eligible_count",
                            "history_ineligible_count",
                            "score_attempted_count",
                            "score_success_count",
                            "filter_passed_count",
                            "selected_count_funnel",
                            "capacity_min_required",
                            "capacity_pass",
                            "empty_reason",
                        ]].copy()
                        funnel_payload["rebalance_date"] = funnel_payload["rebalance_date"].dt.strftime("%Y-%m-%d")
                        metrics_data["diagnostics"]["eligibility_funnel"] = funnel_payload.to_dict(orient="records")
                        metrics_data["diagnostics"]["cash_decomposition"] = {
                            "avg_cash_total_daily": avg_cash_total,
                            "avg_cash_from_regime_daily": avg_cash_regime,
                            "avg_cash_from_caps_daily": avg_cash_caps,
                            "avg_cash_from_zero_selection_daily": avg_cash_zero,
                            "sanity_mean_abs_residual": avg_cash_residual,
                        }
                        metrics_data["diagnostics"]["rebalance_cash_drivers"] = {
                            "total_rebalances": int(len(merged_funnel)),
                            "regime_requested_cash_count": int(regime_requested_cash.sum()),
                            "zero_selection_count": int(zero_selection_flag.sum()),
                            "cap_left_cash_count": int(cap_left_cash.sum()),
                            "both_regime_and_zero_selection_count": int((regime_requested_cash & zero_selection_flag).sum()),
                            "both_cap_and_regime_count": int((cap_left_cash & regime_requested_cash).sum()),
                        }
                        with open(metrics_path, "w") as f:
                            json.dump(metrics_data, f, indent=2)

            # Re-entry blocker attribution (CASH_TRUE)
            print()
            print("RE-ENTRY BLOCKER ATTRIBUTION (CASH_TRUE)")
            print("-" * 30)

            blocker_rows = []
            canonical_blockers = [
                "WARMUP_ACTIVE",
                "PANIC_FLAG",
                "VOLATILITY_GATE",
                "MOMENTUM_CONDITION",
                "HISTORY_GATE_EMPTY",
                "ELIGIBILITY_EMPTY",
                "CAPACITY_FEASIBILITY",
            ]
            failing_counts = defaultdict(int)
            final_blocker_counts = defaultdict(int)

            episode_id = 0
            last_fail_map = None
            last_fail_reason = None
            debug_samples = 0
            debug_limit = 20
            unattributed_warns = 0
            unattributed_warn_limit = 20
            funnel_by_date = {
                pd.to_datetime(r.date): r for r in result.eligibility_funnel_records
            }

            def _signal_snapshot(record: "RebalanceRecord") -> dict:
                funnel = funnel_by_date.get(pd.to_datetime(record.date))
                warmup_active = bool(getattr(funnel, "warmup_skip", False))
                history_eligible_count = (
                    int(funnel.history_eligible) if funnel is not None else 0
                )
                filter_passed_count = (
                    int(funnel.filter_passed_count) if funnel is not None else 0
                )

                vol_ratio = record.features.vol_ratio
                bench_6m = record.features.benchmark_return_6m
                dd_3m = record.features.portfolio_drawdown_3m

                vol_ratio_missing = vol_ratio is None or (isinstance(vol_ratio, float) and np.isnan(vol_ratio))
                bench_6m_missing = bench_6m is None or (isinstance(bench_6m, float) and np.isnan(bench_6m))
                dd_3m_missing = dd_3m is None or (isinstance(dd_3m, float) and np.isnan(dd_3m))

                volatility_gate_pass = (not vol_ratio_missing) and (vol_ratio < config.panic_vol_ratio)
                momentum_pass = (not bench_6m_missing) and (bench_6m > 0)

                if config.panic_defensive_mode or config.cash_replace_mode == CashReplaceMode.DEFENSIVE:
                    if record.defensive_selected_count > 0:
                        defensive_pass = record.defensive_selected_count >= config.defensive_basket_size
                    else:
                        defensive_pass = True
                else:
                    defensive_pass = True

                if record.requested_state == "RISK_ON":
                    eval_count = record.selected_count
                else:
                    eval_count = record.defensive_selected_count
                capacity_evaluated = filter_passed_count > 0 and eval_count > 0
                capacity_pass = True
                if capacity_evaluated:
                    capacity_pass = eval_count >= record.min_required

                panic_active = False
                if not dd_3m_missing and dd_3m > config.panic_dd_threshold:
                    panic_active = True
                if not vol_ratio_missing and vol_ratio > config.panic_vol_ratio:
                    panic_active = True

                return {
                    "vol_gate_pass": volatility_gate_pass,
                    "momentum_pass": momentum_pass,
                    "defensive_pass": defensive_pass,
                    "capacity_pass": capacity_pass,
                    "capacity_evaluated": capacity_evaluated,
                    "panic_active": panic_active,
                    "vol_ratio_missing": vol_ratio_missing,
                    "bench_6m_missing": bench_6m_missing,
                    "dd_3m_missing": dd_3m_missing,
                    "warmup_active": warmup_active,
                    "history_gate_enabled": config.history_gate_mode in ("strict", "lenient"),
                    "history_eligible_count": history_eligible_count,
                    "filter_passed_count": filter_passed_count,
                }

            def _cash_true_reason(snapshot: dict) -> str:
                if snapshot["warmup_active"]:
                    return "WARMUP_ACTIVE"
                if snapshot["panic_active"]:
                    return "PANIC_FLAG"
                if not snapshot["vol_gate_pass"]:
                    return "VOLATILITY_GATE"
                if not snapshot["momentum_pass"]:
                    return "MOMENTUM_CONDITION"
                if snapshot["history_gate_enabled"] and snapshot["history_eligible_count"] == 0:
                    return "HISTORY_GATE_EMPTY"
                if snapshot["filter_passed_count"] == 0:
                    return "ELIGIBILITY_EMPTY"
                if not snapshot["capacity_pass"]:
                    return "CAPACITY_FEASIBILITY"
                raise AssertionError("Unreachable CASH_TRUE attribution")

            for i, record in enumerate(result.rebalance_records):
                prev_state = result.rebalance_records[i - 1].state if i > 0 else None
                curr_state = record.state

                if curr_state == MarketState.CASH_TRUE and prev_state != MarketState.CASH_TRUE:
                    episode_id += 1
                    last_fail_map = None
                    last_fail_reason = None

                if prev_state == MarketState.CASH_TRUE:
                    signal_map = _signal_snapshot(record)

                    if curr_state == MarketState.CASH_TRUE:
                        failing_reason = _cash_true_reason(signal_map)
                        if debug_samples < debug_limit:
                            debug_samples += 1
                            print(
                                f"{record.date.strftime('%Y-%m-%d')} | "
                                f"vol_gate_pass={signal_map['vol_gate_pass']} | "
                                f"momentum_pass={signal_map['momentum_pass']} | "
                                f"defensive_pass={signal_map['defensive_pass']} | "
                                f"capacity_pass={signal_map['capacity_pass']} | "
                                f"panic_active={signal_map['panic_active']} | "
                                f"failing={failing_reason}"
                            )
                        last_fail_map = signal_map
                        last_fail_reason = failing_reason
                        failing_counts[failing_reason] += 1
                        blocker_rows.append({
                            "rebalance_date": record.date.strftime("%Y-%m-%d"),
                            "cash_true_episode_id": episode_id,
                            "vol_gate_pass": signal_map["vol_gate_pass"],
                            "momentum_pass": signal_map["momentum_pass"],
                            "defensive_pass": signal_map["defensive_pass"],
                            "concentration_pass": signal_map["capacity_pass"],
                            "panic_active": signal_map["panic_active"],
                            "failing_signals": failing_reason,
                            "final_reentry_blocker": "",
                        })

                if prev_state == MarketState.CASH_TRUE and curr_state != MarketState.CASH_TRUE:
                    if last_fail_map is None:
                        prev_record = result.rebalance_records[i - 1]
                        last_fail_map = _signal_snapshot(prev_record)
                    signal_map = _signal_snapshot(record)
                    final_blocker = last_fail_reason or _cash_true_reason(last_fail_map)
                    final_blocker_counts[final_blocker] += 1
                    blocker_rows.append({
                        "rebalance_date": record.date.strftime("%Y-%m-%d"),
                        "cash_true_episode_id": episode_id,
                        "vol_gate_pass": signal_map["vol_gate_pass"],
                        "momentum_pass": signal_map["momentum_pass"],
                        "defensive_pass": signal_map["defensive_pass"],
                        "concentration_pass": signal_map["capacity_pass"],
                        "panic_active": signal_map["panic_active"],
                        "failing_signals": "",
                        "final_reentry_blocker": final_blocker,
                    })

            total_failures = sum(failing_counts.values())
            if total_failures > 0:
                for label in canonical_blockers:
                    count = failing_counts[label]
                    pct = (count / total_failures) * 100
                    print(f"{label:<24} {pct:>6.1f}%")
            else:
                print("  N/A (no CASH_TRUE episodes)")

            extra_failure_labels = [
                label for label in failing_counts.keys()
                if label not in canonical_blockers
            ]
            if extra_failure_labels:
                print("\nDiagnostics")
                for label in sorted(extra_failure_labels):
                    count = failing_counts[label]
                    pct = (count / total_failures) * 100 if total_failures else 0.0
                    print(f"{label:<24} {pct:>6.1f}%")

            total_final = sum(final_blocker_counts.values())
            if total_final > 0:
                print("\nFinal Re-entry Blocker Distribution")
                for label in canonical_blockers:
                    count = final_blocker_counts[label]
                    pct = (count / total_final) * 100
                    print(f"{label:<24} {pct:>6.1f}%")

            if blocker_rows:
                pd.DataFrame(blocker_rows).to_csv(
                    config.output_dir / "cash_true_reentry_blockers.csv",
                    index=False,
                )

            # CASH_TRUE drag KPIs
            print()
            print("CASH_TRUE DRAG ANALYSIS")
            print("-" * 30)

            cash_true_episodes = []
            mask_days = result.daily_state == "CASH_TRUE"
            current_start = None
            for day, is_cash in mask_days.items():
                if is_cash and current_start is None:
                    current_start = day
                elif not is_cash and current_start is not None:
                    cash_true_episodes.append({
                        "start_date": current_start,
                        "end_date": prev_day,
                    })
                    current_start = None
                prev_day = day

            if current_start is not None:
                cash_true_episodes.append({
                    "start_date": current_start,
                    "end_date": prev_day,
                })

            if cash_true_episodes:
                covered = pd.Series(False, index=result.daily_state.index)
                for ep in cash_true_episodes:
                    covered.loc[ep["start_date"]:ep["end_date"]] = True
                if not mask_days.equals(covered):
                    first_mismatch = (mask_days != covered)
                    first_date = first_mismatch[first_mismatch].index[0]
                    raise RuntimeError(
                        "CASH_TRUE episode coverage mismatch in daily state series "
                        f"(first mismatch: {first_date.date()})"
                    )

            episode_rows = []
            total_drag = 0.0
            for idx, ep in enumerate(cash_true_episodes, 1):
                start_date = ep["start_date"]
                end_date = ep["end_date"]
                end_idx = trading_dates[trading_dates <= end_date]
                if len(end_idx) == 0:
                    continue
                end_actual = end_idx[-1]
                start_actual = start_date
                strat_start = result.equity_curve.loc[start_actual]
                strat_end = result.equity_curve.loc[end_actual]
                mom50_curve = comparison_benchmark.data.reindex(result.equity_curve.index).ffill()
                mom50_start = mom50_curve.loc[start_actual]
                mom50_end = mom50_curve.loc[end_actual]

                strategy_return = (strat_end / strat_start) - 1.0
                mom50_return = (mom50_end / mom50_start) - 1.0
                cash_true_drag = mom50_return - strategy_return

                duration_days = (end_actual - start_actual).days + 1
                if duration_days <= 0:
                    raise RuntimeError("CASH_TRUE episode duration_days must be > 0")
                duration_months = duration_days / 30.4375
                drag_per_month = cash_true_drag / duration_months
                total_drag += cash_true_drag

                episode_rows.append({
                    "cash_true_episode_id": idx,
                    "start_date": start_actual.strftime("%Y-%m-%d"),
                    "end_date": end_actual.strftime("%Y-%m-%d"),
                    "duration_days": duration_days,
                    "duration_months": round(duration_months, 4),
                    "strategy_return_episode": round(strategy_return, 6),
                    "mom50_return_episode": round(mom50_return, 6),
                    "cash_true_drag": round(cash_true_drag, 6),
                    "cash_true_drag_per_month": round(drag_per_month, 6),
                })

            if episode_rows:
                df_episodes = pd.DataFrame(episode_rows)
                df_episodes.to_csv(config.output_dir / "cash_true_drag.csv", index=False)

                mean_drag = df_episodes["cash_true_drag"].mean()
                median_drag = df_episodes["cash_true_drag"].median()
                max_drag = df_episodes["cash_true_drag"].max()
                min_drag = df_episodes["cash_true_drag"].min()
                mean_drag_per_month = df_episodes["cash_true_drag_per_month"].mean()

                print(f"Episodes: {len(df_episodes)}")
                print(f"Mean Drag: {mean_drag*100:>7.2f}%")
                print(f"Median Drag: {median_drag*100:>7.2f}%")
                print(f"Max Drag: {max_drag*100:>7.2f}%")
                print(f"Min Drag: {min_drag*100:>7.2f}%")
                print(f"Mean Drag / Month: {mean_drag_per_month*100:>7.2f}%")

                total_strategy_return = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) - 1.0
                cash_true_drag_ratio = total_drag / total_strategy_return if total_strategy_return > 0 else 0.0
                print(f"\nCASH_TRUE DRAG RATIO: {cash_true_drag_ratio*100:>7.2f}%")
            else:
                print("Episodes: 0")

        logger.info("Backtest completed successfully!")
        return 0

    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        return 1

    except StagingError as e:
        logger.error(f"Data staging failed: {e}")
        return 1

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
