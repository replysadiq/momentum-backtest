"""
Data layer for the momentum backtest system.

Modules:
- universe: ticker list management
- downloader: Yahoo Finance data acquisition
- benchmark: Benchmark index selection with fallback
- calendar: Rebalance calendar construction (single source of truth)
- sanitizer: Data cleaning and eligibility checks
"""

from .universe import load_universe, validate_tickers
from .downloader import download_price_data, download_single_ticker
from .benchmark import get_benchmark_data, BENCHMARK_CANDIDATES
from .calendar import build_rebalance_calendar
from .sanitizer import sanitize_stock_data, check_stock_eligibility

__all__ = [
    "load_universe",
    "validate_tickers",
    "download_price_data",
    "download_single_ticker",
    "get_benchmark_data",
    "BENCHMARK_CANDIDATES",
    "build_rebalance_calendar",
    "sanitize_stock_data",
    "check_stock_eligibility",
]
