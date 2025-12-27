"""
Benchmark index selection with automated fallback.

Tries candidates in priority order, validates coverage, and selects the best.
The benchmark trading calendar becomes the authoritative source for trading days.

Also supports loading benchmark from a local CSV file.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import List, NamedTuple, Optional, Union
import logging

import pandas as pd

from .downloader import download_single_ticker


logger = logging.getLogger(__name__)


# Benchmark candidates in priority order
BENCHMARK_CANDIDATES: List[str] = [
    "^CRSLDX",     # NIFTY 500 index (preferred if available)
    "^NSEI",       # NIFTY 50 index (reliable fallback)
]

# Minimum coverage threshold (as fraction of expected trading days)
MIN_COVERAGE_THRESHOLD = 0.98  # 98% coverage required


class BenchmarkResult(NamedTuple):
    """Result of benchmark selection."""
    ticker: str
    data: pd.Series
    coverage: float
    is_proxy: bool  # True if using NIFTY 50 as proxy for NIFTY 500
    fallback_chain: List[str]  # Tickers tried in order


def _estimate_expected_trading_days(start_date: date, end_date: date) -> int:
    """
    Estimate expected number of trading days in a date range.

    Uses approximately 250 trading days per year for India.
    """
    days = (end_date - start_date).days
    years = days / 365.25
    return int(years * 250)


def _check_benchmark_coverage(
    data: pd.Series,
    start_date: date,
    end_date: date,
) -> float:
    """
    Check what fraction of expected trading days have data.

    Args:
        data: Price series from benchmark
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        Coverage fraction (0.0 to 1.0)
    """
    expected = _estimate_expected_trading_days(start_date, end_date)
    if expected == 0:
        return 0.0

    # Count days within the backtest period
    mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
    actual = mask.sum()

    return actual / expected


def load_benchmark_from_csv(
    csv_path: Union[str, Path],
    start_date: date,
    end_date: date,
    ticker_name: Optional[str] = None,
) -> BenchmarkResult:
    """
    Load benchmark data from a local CSV file.

    Expected CSV format:
    - First column: Date (index)
    - Column named 'Close': Closing prices

    Args:
        csv_path: Path to the CSV file
        start_date: Backtest start date
        end_date: Backtest end date
        ticker_name: Optional name for the benchmark (default: derived from filename)

    Returns:
        BenchmarkResult with loaded data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    logger.info(f"Loading benchmark from local CSV: {csv_path}")

    # Derive ticker name from filename if not provided
    if ticker_name is None:
        ticker_name = csv_path.stem.upper()

    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    if "Close" not in df.columns:
        raise ValueError(f"CSV must have a 'Close' column. Found: {df.columns.tolist()}")

    # Extract Close prices as Series
    data = df["Close"].sort_index()

    # Filter to date range (with buffer for lookback)
    buffer_start = start_date - timedelta(days=365)  # 1 year buffer for momentum calc
    mask = (data.index >= pd.Timestamp(buffer_start)) & (data.index <= pd.Timestamp(end_date))
    data = data[mask]

    if len(data) == 0:
        raise ValueError(
            f"No data in CSV for date range {start_date} to {end_date}. "
            f"CSV date range: {df.index.min().date()} to {df.index.max().date()}"
        )

    coverage = _check_benchmark_coverage(data, start_date, end_date)

    logger.info(f"Loaded benchmark '{ticker_name}': {len(data)} days, coverage={coverage:.1%}")
    logger.info(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")

    return BenchmarkResult(
        ticker=ticker_name,
        data=data,
        coverage=coverage,
        is_proxy=False,  # Local file is authoritative
        fallback_chain=[ticker_name],
    )


def get_benchmark_data(
    start_date: date,
    end_date: date,
    candidates: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Get benchmark price data with automated fallback.

    Tries candidates in priority order:
    1. Download 10-year data for each candidate
    2. Check coverage (must be >= 98% of expected trading days)
    3. Select the first candidate that meets coverage threshold
    4. If no candidate meets threshold, use the one with best coverage

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        candidates: List of ticker symbols to try (default: BENCHMARK_CANDIDATES)

    Returns:
        BenchmarkResult with selected ticker, data, coverage, and proxy flag

    Raises:
        RuntimeError: If no benchmark data can be obtained
    """
    if candidates is None:
        candidates = BENCHMARK_CANDIDATES.copy()

    logger.info(f"Selecting benchmark from candidates: {candidates}")

    # Track which candidates we tried (for audit report)
    tried_candidates: List[str] = []
    best_result: Optional[BenchmarkResult] = None
    best_coverage = 0.0

    for ticker in candidates:
        logger.info(f"Trying benchmark candidate: {ticker}")
        tried_candidates.append(ticker)

        data = download_single_ticker(ticker, start_date, end_date, progress=False)

        if data is None or len(data) == 0:
            logger.warning(f"No data returned for {ticker}")
            continue

        coverage = _check_benchmark_coverage(data, start_date, end_date)
        logger.info(f"{ticker}: coverage = {coverage:.1%} ({len(data)} days)")

        # Determine if this is a proxy (NIFTY 50 instead of NIFTY 500)
        is_proxy = ticker == "^NSEI"

        result = BenchmarkResult(
            ticker=ticker,
            data=data,
            coverage=coverage,
            is_proxy=is_proxy,
            fallback_chain=tried_candidates.copy(),
        )

        # If coverage meets threshold, use this one
        if coverage >= MIN_COVERAGE_THRESHOLD:
            logger.info(f"Selected benchmark: {ticker} (coverage: {coverage:.1%})")
            if is_proxy:
                logger.warning(
                    f"Using {ticker} (NIFTY 50) as proxy for NIFTY 500. "
                    "This may introduce tracking error."
                )
            return result

        # Track best so far
        if coverage > best_coverage:
            best_coverage = coverage
            best_result = result

    # No candidate met threshold - use best available
    if best_result is not None:
        logger.warning(
            f"No benchmark met {MIN_COVERAGE_THRESHOLD:.0%} coverage threshold. "
            f"Using {best_result.ticker} with {best_result.coverage:.1%} coverage."
        )
        if best_result.is_proxy:
            logger.warning(
                f"Using {best_result.ticker} (NIFTY 50) as proxy for NIFTY 500. "
                "This may introduce tracking error."
            )
        # Update fallback_chain to include all tried candidates
        return BenchmarkResult(
            ticker=best_result.ticker,
            data=best_result.data,
            coverage=best_result.coverage,
            is_proxy=best_result.is_proxy,
            fallback_chain=tried_candidates,
        )

    raise RuntimeError(
        f"Failed to obtain benchmark data from any candidate: {candidates}. "
        "Check network connection and Yahoo Finance availability."
    )


def get_trading_calendar(benchmark_data: pd.Series) -> pd.DatetimeIndex:
    """
    Extract the trading calendar from benchmark data.

    The benchmark's trading dates become the authoritative source
    for what constitutes a valid trading day.

    Args:
        benchmark_data: Price series from the benchmark

    Returns:
        Sorted DatetimeIndex of trading dates
    """
    trading_dates = benchmark_data.index.sort_values()
    logger.info(f"Trading calendar: {len(trading_dates)} days from "
                f"{trading_dates[0].date()} to {trading_dates[-1].date()}")
    return trading_dates
