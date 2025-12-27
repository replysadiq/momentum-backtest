"""
Yahoo Finance data acquisition module.

Downloads daily Adj Close prices for stocks and benchmark.
Supports loading from pre-downloaded parquet files.
No forward-filling is applied - missing data is preserved for eligibility filtering.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

import pandas as pd
import yfinance as yf
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Extra buffer days for data download to ensure we have enough history
DOWNLOAD_BUFFER_DAYS = 400  # ~13 months buffer for lookback


def download_single_ticker(
    ticker: str,
    start_date: date,
    end_date: date,
    progress: bool = False,
) -> Optional[pd.Series]:
    """
    Download Adj Close price series for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g., 'RELIANCE.NS')
        start_date: Start date for data download
        end_date: End date for data download
        progress: Show yfinance progress bar

    Returns:
        Series of Adj Close prices indexed by date, or None if download fails

    Note:
        - Uses only 'Adj Close' field as mandated
        - Does NOT forward-fill missing values
        - Returns None on download failure (caller handles eligibility)
    """
    try:
        # Add buffer to start date for lookback calculations
        buffer_start = start_date - timedelta(days=DOWNLOAD_BUFFER_DAYS)

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(
            start=buffer_start.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            auto_adjust=False,  # We want explicit Adj Close
        )

        if hist.empty:
            logger.debug(f"No data returned for {ticker}")
            return None

        # Extract Adj Close only
        if "Adj Close" not in hist.columns:
            logger.warning(f"No 'Adj Close' column for {ticker}, columns: {list(hist.columns)}")
            return None

        adj_close = hist["Adj Close"].copy()

        # Convert index to date (remove timezone if present)
        adj_close.index = pd.to_datetime(adj_close.index).date
        adj_close.index = pd.DatetimeIndex(adj_close.index)
        adj_close.name = ticker

        # Drop any NaN values (do NOT forward-fill)
        adj_close = adj_close.dropna()

        if adj_close.empty:
            logger.debug(f"All NaN values for {ticker}")
            return None

        logger.debug(f"Downloaded {len(adj_close)} days for {ticker}: "
                     f"{adj_close.index[0].date()} to {adj_close.index[-1].date()}")

        return adj_close

    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def download_price_data(
    tickers: List[str],
    start_date: date,
    end_date: date,
    show_progress: bool = True,
) -> Dict[str, pd.Series]:
    """
    Download Adj Close price data for multiple tickers.

    Args:
        tickers: List of Yahoo Finance ticker symbols
        start_date: Start date for data download
        end_date: End date for data download
        show_progress: Show progress bar during download

    Returns:
        Dictionary mapping ticker symbols to their Adj Close price series.
        Tickers that fail to download are excluded (not in dict).

    Note:
        - Downloads include buffer period for lookback calculations
        - Missing data is NOT forward-filled
        - Failed downloads are logged but not raised as errors
    """
    logger.info(f"Downloading price data for {len(tickers)} tickers...")

    results: Dict[str, pd.Series] = {}
    failed: List[str] = []

    iterator = tqdm(tickers, desc="Downloading", disable=not show_progress)

    for ticker in iterator:
        series = download_single_ticker(ticker, start_date, end_date, progress=False)
        if series is not None and len(series) > 0:
            results[ticker] = series
        else:
            failed.append(ticker)

    success_rate = len(results) / len(tickers) * 100 if tickers else 0
    logger.info(f"Downloaded {len(results)}/{len(tickers)} tickers ({success_rate:.1f}% success)")

    if failed:
        logger.warning(f"Failed to download {len(failed)} tickers: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return results


def align_to_calendar(
    price_data: Dict[str, pd.Series],
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align all price series to a common trading calendar.

    Args:
        price_data: Dictionary of ticker -> price series
        trading_dates: DatetimeIndex of valid trading dates (from benchmark)

    Returns:
        DataFrame with tickers as columns and trading_dates as index.
        Missing values are NaN (NOT forward-filled).

    Note:
        - Only includes dates present in trading_dates
        - Stocks missing data on a trading day have NaN for that day
        - This preserves missing data for eligibility filtering
    """
    if not price_data:
        raise ValueError("No price data to align")

    # Build all series at once to avoid DataFrame fragmentation
    aligned_series = {
        ticker: series.reindex(trading_dates)
        for ticker, series in price_data.items()
    }
    aligned = pd.DataFrame(aligned_series, index=trading_dates)

    logger.info(f"Aligned {len(aligned.columns)} tickers to {len(aligned)} trading days")

    # Log coverage statistics
    coverage = aligned.notna().sum() / len(aligned) * 100
    logger.debug(f"Data coverage: min={coverage.min():.1f}%, median={coverage.median():.1f}%, max={coverage.max():.1f}%")

    return aligned


def load_price_data_from_parquet(
    parquet_path: Path,
    tickers: List[str],
    start_date: date,
    end_date: date,
) -> Dict[str, pd.Series]:
    """
    Load Adj Close price data from a pre-downloaded parquet file.

    Expects parquet with columns: date, adj_close, symbol (NSE format).
    Converts NSE symbols (RELIANCE) to Yahoo format (RELIANCE.NS).

    Args:
        parquet_path: Path to parquet file
        tickers: List of tickers to load (Yahoo format with .NS suffix)
        start_date: Start date for data
        end_date: End date for data

    Returns:
        Dictionary mapping ticker symbols to their Adj Close price series.
    """
    logger.info(f"Loading price data from parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Convert date column to datetime and remove timezone
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Add buffer for lookback calculations
    buffer_start = start_date - timedelta(days=DOWNLOAD_BUFFER_DAYS)

    # Filter date range
    df = df[(df["date"] >= pd.Timestamp(buffer_start)) & (df["date"] <= pd.Timestamp(end_date))]

    # Build mapping from Yahoo symbol to NSE symbol
    # Yahoo: RELIANCE.NS -> NSE: RELIANCE
    yahoo_to_nse = {}
    for ticker in tickers:
        if ticker.endswith(".NS"):
            nse_symbol = ticker[:-3]
        else:
            nse_symbol = ticker
        yahoo_to_nse[ticker] = nse_symbol

    # Get unique NSE symbols we need
    nse_symbols_needed = set(yahoo_to_nse.values())

    # Filter to only symbols we need
    df = df[df["symbol"].isin(nse_symbols_needed)]

    results: Dict[str, pd.Series] = {}
    loaded = 0
    missing = []

    for yahoo_ticker, nse_symbol in yahoo_to_nse.items():
        ticker_data = df[df["symbol"] == nse_symbol].copy()

        if ticker_data.empty:
            missing.append(yahoo_ticker)
            continue

        # Create series with date index
        ticker_data = ticker_data.sort_values("date")
        series = pd.Series(
            ticker_data["adj_close"].values,
            index=pd.DatetimeIndex(ticker_data["date"].values),
            name=yahoo_ticker,
        )

        # Drop NaN values
        series = series.dropna()

        if not series.empty:
            results[yahoo_ticker] = series
            loaded += 1
        else:
            missing.append(yahoo_ticker)

    success_rate = loaded / len(tickers) * 100 if tickers else 0
    logger.info(f"Loaded {loaded}/{len(tickers)} tickers from parquet ({success_rate:.1f}% success)")

    if missing:
        logger.warning(f"Missing {len(missing)} tickers in parquet: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    return results
