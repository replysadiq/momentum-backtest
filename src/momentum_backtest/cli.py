"""
Command-line interface for the momentum backtest system.

Provides argument parsing and orchestrates the full backtest run.
"""

import argparse
from datetime import date, datetime
from pathlib import Path
import logging
import sys
from typing import Optional

from .config import BacktestConfig, CashEntryMode, CashReplaceMode


logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD."
        )


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: List of arguments (default: sys.argv)

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Institution-grade momentum backtesting system for Indian equities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings
  momentum-backtest --tickers-csv data/nifty500.csv

  # Custom date range
  momentum-backtest --tickers-csv data/nifty500.csv --start 2015-01-01 --end 2023-12-31

  # With transaction costs and custom max weight
  momentum-backtest --tickers-csv data/nifty500.csv --tc-bps 10 --max-weight 0.08

  # Enable drawdown filter
  momentum-backtest --tickers-csv data/nifty500.csv --use-dd-filter --dd-threshold 0.35
""",
    )

    # Required arguments
    parser.add_argument(
        "--tickers-csv",
        type=Path,
        required=True,
        help="Path to CSV file with NIFTY 500 ticker list",
    )

    # Data source
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=None,
        help="Path to parquet file with pre-downloaded OHLCV data. "
             "If provided, skips Yahoo Finance download.",
    )

    # Benchmark source
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Path to CSV file with primary benchmark index data (Date,Close columns). "
             "Used for trading calendar and state machine signals. "
             "If not provided, downloads from Yahoo Finance.",
    )
    parser.add_argument(
        "--comparison-benchmark-csv",
        type=Path,
        default=None,
        help="Path to CSV file with comparison benchmark (Date,Close columns). "
             "Used ONLY for performance comparison in metrics, not for signals. "
             "Useful for comparing against factor indices like NIFTY500 Momentum 50.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output files (default: output/)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=parse_date,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 10 years ago",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=None,
        help="End date (YYYY-MM-DD). Default: end of last year",
    )

    # Portfolio parameters
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of stocks to hold (default: 20)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.10,
        help="Maximum weight per stock (default: 0.10). Set to 0 to disable cap.",
    )

    # Drawdown filter
    parser.add_argument(
        "--use-dd-filter",
        action="store_true",
        help="Enable 12-month max drawdown filter (default: off)",
    )
    parser.add_argument(
        "--dd-threshold",
        type=float,
        default=0.40,
        help="Max drawdown threshold for filter (default: 0.40)",
    )

    # Scoring methodology
    parser.add_argument(
        "--nse-style",
        action="store_true",
        help="Use NSE NIFTY500 Momentum 50 scoring methodology "
             "(Z-score normalized, 50/50 6M/12M weighting)",
    )

    # Transaction costs
    parser.add_argument(
        "--tc-bps",
        type=float,
        default=0.0,
        help="Transaction cost in basis points (default: 0 = no costs)",
    )

    # Cash yield
    parser.add_argument(
        "--cash-rate-annual",
        type=float,
        default=0.0,
        help="Annualized cash yield for CASH/PANIC periods (default: 0.0). "
             "Set to 0.05 for 5%% T-bill yield in India.",
    )

    # State machine thresholds
    parser.add_argument(
        "--panic-dd",
        type=float,
        default=0.15,
        help="Portfolio drawdown to trigger PANIC (default: 0.15)",
    )
    parser.add_argument(
        "--panic-vol-ratio",
        type=float,
        default=2.0,
        help="Volatility ratio to trigger PANIC (default: 2.0)",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (very verbose)",
    )

    # === V2 Strategy Levers ===
    v2_group = parser.add_argument_group("V2 Strategy Levers")

    # Lever A: Defensive Momentum
    v2_group.add_argument(
        "--panic-defensive-mode",
        action="store_true",
        help="Enable defensive momentum: invest in low-vol stocks during PANIC "
             "instead of going to cash (Lever A)",
    )
    v2_group.add_argument(
        "--defensive-basket-size",
        type=int,
        default=15,
        help="Number of stocks in defensive basket (default: 15)",
    )

    # Lever B: Rank Buffer / Min Hold (mutually exclusive)
    buffer_group = v2_group.add_mutually_exclusive_group()
    buffer_group.add_argument(
        "--rank-buffer",
        type=float,
        default=0.0,
        help="Rank buffer: current holding only sold if new candidate's rank is "
             "this fraction better (e.g., 0.20 = 20%% better). Default: 0 (disabled)",
    )
    buffer_group.add_argument(
        "--min-hold-months",
        type=int,
        default=0,
        help="Minimum months to hold a stock before it can be sold. "
             "Default: 0 (disabled). Mutually exclusive with --rank-buffer",
    )

    # Lever D: Disable 6M Filter
    v2_group.add_argument(
        "--disable-6m-filter",
        action="store_true",
        help="Disable 6-month return > 0 eligibility filter (Lever D). "
             "Keeps: 12M return > 0, >=60%% positive months",
    )

    # Rebalance frequency
    parser.add_argument(
        "--rebalance-months",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Rebalance frequency in months: 1=monthly (default), 2=bi-monthly, 3=quarterly",
    )

    # === V3 Strategy Levers ===
    v3_group = parser.add_argument_group("V3 Strategy Levers")

    v3_group.add_argument(
        "--cash-entry-mode",
        type=str,
        choices=["baseline", "strict_dual", "strict_persist"],
        default="baseline",
        help="V3: Cash entry mode. "
             "'baseline' = original (6M ret <= 0), "
             "'strict_dual' = require 6M AND 3M ret <= 0, "
             "'strict_persist' = require baseline condition for 2 consecutive rebalances. "
             "(default: baseline)",
    )

    v3_group.add_argument(
        "--cash-replace-mode",
        type=str,
        choices=["none", "defensive"],
        default="none",
        help="V3.1: Cash replacement mode. "
             "'none' = hold cash when in CASH state (original), "
             "'defensive' = hold defensive momentum portfolio instead of cash. "
             "(default: none)",
    )

    # === V4 Breadth Overlay ===
    v4_group = parser.add_argument_group("V4 Breadth Overlay")

    v4_group.add_argument(
        "--enable-breadth-overlay",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        metavar="BOOL",
        help="V4: Enable breadth-based exposure scaling (default: true). "
             "Set to 'false' to reproduce legacy behavior exactly.",
    )
    v4_group.add_argument(
        "--breadth-lookback",
        type=int,
        default=63,
        help="Trading days for breadth return calculation (default: 63)",
    )
    v4_group.add_argument(
        "--breadth-ema-span",
        type=int,
        default=10,
        help="EMA smoothing span for breadth (default: 10)",
    )
    v4_group.add_argument(
        "--breadth-low",
        type=float,
        default=0.35,
        help="Breadth below this = confidence 0 (default: 0.35)",
    )
    v4_group.add_argument(
        "--breadth-high",
        type=float,
        default=0.65,
        help="Breadth above this = confidence 1 (default: 0.65)",
    )
    v4_group.add_argument(
        "--breadth-min-coverage",
        type=float,
        default=0.60,
        help="Minimum fraction of stocks required for breadth calc (default: 0.60)",
    )

    return parser.parse_args(args)


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def build_config(args: argparse.Namespace) -> BacktestConfig:
    """
    Build BacktestConfig from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        BacktestConfig instance
    """
    # Handle max_weight = 0 meaning disabled
    max_weight = args.max_weight if args.max_weight > 0 else None

    # Build config with defaults filled in
    config_kwargs = {
        "tickers_csv": args.tickers_csv,
        "output_dir": args.output_dir,
        "top_n_stocks": args.top_n,
        "max_weight": max_weight,
        "use_dd_filter": args.use_dd_filter,
        "dd_threshold": args.dd_threshold,
        "nse_style_scoring": args.nse_style,
        "tc_bps": args.tc_bps,
        "cash_rate_annual": args.cash_rate_annual,
        "panic_dd_threshold": args.panic_dd,
        "panic_vol_ratio": args.panic_vol_ratio,
        "verbose": args.verbose or args.debug,
        # V2 levers
        "panic_defensive_mode": args.panic_defensive_mode,
        "defensive_basket_size": args.defensive_basket_size,
        "rank_buffer": args.rank_buffer,
        "min_hold_months": args.min_hold_months,
        "disable_6m_filter": args.disable_6m_filter,
        "rebalance_months": args.rebalance_months,
        # V3 levers
        "cash_entry_mode": CashEntryMode(args.cash_entry_mode),
        # V3.1 levers
        "cash_replace_mode": CashReplaceMode(args.cash_replace_mode),
        # V4 breadth overlay
        "enable_breadth_overlay": args.enable_breadth_overlay,
        "breadth_lookback": args.breadth_lookback,
        "breadth_ema_span": args.breadth_ema_span,
        "breadth_low": args.breadth_low,
        "breadth_high": args.breadth_high,
        "breadth_min_coverage": args.breadth_min_coverage,
    }

    # Add dates if specified
    if args.start is not None:
        config_kwargs["start_date"] = args.start
    if args.end is not None:
        config_kwargs["end_date"] = args.end

    return BacktestConfig(**config_kwargs)
