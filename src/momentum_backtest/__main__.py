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
from pathlib import Path
import subprocess
import sys
from typing import Optional

import pandas as pd

from .cli import parse_args, setup_logging, build_config
from .data.staging import stage_parquet_file, stage_ticker_csv, StagingError
from .data.universe import load_universe
from .data.benchmark import get_benchmark_data, get_trading_calendar, load_benchmark_from_csv
from .data.downloader import download_price_data, align_to_calendar, load_price_data_from_parquet
from .data.calendar import build_rebalance_calendar, validate_rebalance_calendar
from .data.sanitizer import sanitize_stock_data
from .engine.backtest import run_backtest
from .reporting.metrics import (
    compute_metrics,
    compute_rolling_excess_return,
    _compute_cagr,
    _compute_max_drawdown,
    _compute_volatility,
)
from .reporting.exporter import export_results, print_summary, export_rolling_excess_return
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
    1. Every rebalance date exists in benchmark trading dates
    2. Rebalance dates are strictly increasing
    3. Exactly one rebalance per calendar month

    Args:
        rebalance_calendar: The frozen rebalance calendar
        trading_dates: Benchmark-derived trading calendar

    Raises:
        ValidationError: If any invariant is violated (fatal bug)
    """
    import pandas as pd

    trading_set = set(trading_dates)

    # Invariant 1: Every rebalance date is a benchmark trading day
    for rebal_date in rebalance_calendar:
        if rebal_date not in trading_set:
            raise ValidationError(
                f"FATAL BUG: Rebalance date {rebal_date.date()} not in benchmark trading dates. "
                "This indicates a bug in calendar construction - rebalance dates must be "
                "derived from benchmark trading days, never from stock data."
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

        # Step 3: Get benchmark data and trading calendar
        logger.info("Step 3: Getting benchmark data...")
        if args.benchmark_csv is not None:
            # Load from local CSV file
            benchmark_result = load_benchmark_from_csv(
                args.benchmark_csv, config.start_date, config.end_date
            )
        else:
            # Download from Yahoo Finance
            benchmark_result = get_benchmark_data(config.start_date, config.end_date)

        logger.info(f"  Using benchmark: {benchmark_result.ticker} "
                    f"(coverage: {benchmark_result.coverage:.1%})")

        if benchmark_result.is_proxy:
            logger.warning("  WARNING: Using NIFTY 50 as proxy for NIFTY 500")

        trading_dates = get_trading_calendar(benchmark_result.data)
        logger.info(f"  Trading calendar: {len(trading_dates)} days")

        # Step 4: Build rebalance calendar FROM BENCHMARK TRADING DATES ONLY
        # This is the FROZEN calendar - never modified based on stock data
        freq_label = {1: "monthly", 2: "bi-monthly", 3: "quarterly"}[config.rebalance_months]
        logger.info(f"Step 4: Building rebalance calendar ({freq_label})...")
        rebalance_calendar = build_rebalance_calendar(
            trading_dates, config.start_date, config.end_date, config.rebalance_months
        )
        logger.info(f"  Rebalance dates: {len(rebalance_calendar)}")

        # INVARIANT: Every rebalance date must exist in benchmark trading dates
        validate_rebalance_calendar(rebalance_calendar, trading_dates)
        _validate_rebalance_invariants(rebalance_calendar, trading_dates)

        # Step 4b: Load comparison benchmark if provided (for metrics only)
        comparison_benchmark = None
        if args.comparison_benchmark_csv is not None:
            logger.info("Step 4b: Loading comparison benchmark...")
            comparison_benchmark = load_benchmark_from_csv(
                args.comparison_benchmark_csv, config.start_date, config.end_date,
            )
            logger.info(f"  Comparison benchmark: {comparison_benchmark.ticker} "
                        f"(coverage: {comparison_benchmark.coverage:.1%})")

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
            time_in_cash=result.time_in_cash,
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
            pct_time_true_cash=result.pct_time_true_cash,
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
                    time_in_cash=0.0,
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
                    pct_time_true_cash=0.0,
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
                    time_in_cash=result.time_in_cash,
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
                    pct_time_true_cash=result.pct_time_true_cash,
                )
                comparison_benchmark_stats = compute_metrics(
                    equity_curve=comp_curve,
                    benchmark_curve=comp_curve,
                    time_in_risk_on=0.0,
                    time_in_panic=0.0,
                    time_in_cash=0.0,
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
                    pct_time_true_cash=0.0,
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

        # Step 9b: Compute and export rolling 3-year excess return
        # Use comparison benchmark if provided, else primary benchmark
        rolling_benchmark = None
        rolling_benchmark_name = None

        if comparison_benchmark is not None:
            rolling_benchmark = comparison_benchmark.data
            rolling_benchmark_name = comparison_benchmark.ticker
        elif benchmark_result is not None:
            rolling_benchmark = benchmark_result.data
            rolling_benchmark_name = benchmark_result.ticker

        if rolling_benchmark is not None:
            logger.info("Step 9b: Computing rolling 3-year excess return...")
            # Normalize benchmark to same scale as equity curve
            bench_aligned = rolling_benchmark.reindex(result.equity_curve.index).dropna()
            if len(bench_aligned) > 0:
                bench_normalized = bench_aligned / bench_aligned.iloc[0]

                rolling_df, rolling_stats = compute_rolling_excess_return(
                    equity_curve=result.equity_curve,
                    benchmark_curve=bench_normalized,
                    rebalance_dates=rebalance_calendar,
                    benchmark_name=rolling_benchmark_name,
                    window_months=36,
                )

                export_rolling_excess_return(
                    rolling_df=rolling_df,
                    rolling_stats=rolling_stats,
                    output_dir=config.output_dir,
                )
        else:
            logger.warning("No benchmark available for rolling excess return calculation")

        # Print summary to console
        print()
        print_summary(metrics)

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
