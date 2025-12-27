"""
Universe management for NIFTY 500 tickers.

Handles loading, normalization, and validation of the stock universe.

SURVIVORSHIP BIAS NOTE:
This system uses the CURRENT NIFTY 500 constituent list for the entire backtest period.
This introduces survivorship bias - stocks that were in the index historically but have
since been removed (due to poor performance, delisting, or other reasons) are excluded.
This typically overstates backtest performance by 1-3% annually.
To address this properly, one would need historical point-in-time constituent data.
"""

from pathlib import Path
from typing import List, Set
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def normalize_ticker(ticker: str) -> str:
    """
    Normalize a ticker symbol to Yahoo Finance format for NSE stocks.

    Args:
        ticker: Raw ticker symbol (e.g., 'RELIANCE', 'RELIANCE.NS', 'reliance')

    Returns:
        Normalized ticker with .NS suffix (e.g., 'RELIANCE.NS')
    """
    ticker = ticker.strip().upper()

    # Remove any existing suffix and re-add .NS
    if ticker.endswith(".NS"):
        return ticker
    if ticker.endswith(".BO"):
        # Convert BSE to NSE
        ticker = ticker[:-3]

    return f"{ticker}.NS"


def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate and deduplicate a list of tickers.

    Args:
        tickers: List of ticker symbols

    Returns:
        Deduplicated list of valid tickers

    Raises:
        ValueError: If no valid tickers remain after validation
    """
    if not tickers:
        raise ValueError("Empty ticker list provided")

    # Normalize all tickers
    normalized = [normalize_ticker(t) for t in tickers]

    # Check for duplicates
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    unique_tickers: List[str] = []

    for ticker in normalized:
        if ticker in seen:
            duplicates.add(ticker)
        else:
            seen.add(ticker)
            unique_tickers.append(ticker)

    if duplicates:
        logger.warning(f"Removed {len(duplicates)} duplicate tickers: {sorted(duplicates)[:10]}...")

    # Validate format
    invalid = [t for t in unique_tickers if not t.endswith(".NS")]
    if invalid:
        raise ValueError(f"Invalid ticker format (must end with .NS): {invalid[:5]}")

    if not unique_tickers:
        raise ValueError("No valid tickers after validation")

    logger.info(f"Validated {len(unique_tickers)} unique tickers")
    return unique_tickers


def load_universe(csv_path: Path) -> List[str]:
    """
    Load and validate the NIFTY 500 ticker universe from a CSV file.

    The CSV file should have a column containing ticker symbols.
    The function will look for columns named 'Symbol', 'Ticker', or use the first column.

    Args:
        csv_path: Path to the CSV file containing tickers

    Returns:
        List of validated, normalized ticker symbols with .NS suffix

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV is empty or has no valid tickers
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker CSV file not found: {csv_path}")

    logger.info(f"Loading universe from: {csv_path}")

    # Try reading with different possible formats
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")

    if df.empty:
        raise ValueError(f"Empty CSV file: {csv_path}")

    # Find the ticker column
    ticker_col = None
    for col_name in ["Symbol", "SYMBOL", "symbol", "Ticker", "TICKER", "ticker"]:
        if col_name in df.columns:
            ticker_col = col_name
            break

    if ticker_col is None:
        # Use first column
        ticker_col = df.columns[0]
        logger.info(f"No 'Symbol' or 'Ticker' column found, using first column: '{ticker_col}'")

    # Extract tickers
    raw_tickers = df[ticker_col].dropna().astype(str).tolist()

    if not raw_tickers:
        raise ValueError(f"No tickers found in column '{ticker_col}'")

    # Validate and normalize
    validated = validate_tickers(raw_tickers)

    # Log survivorship bias warning
    logger.warning(
        "SURVIVORSHIP BIAS: Using current NIFTY 500 constituents for entire backtest period. "
        "This may overstate returns by 1-3% annually as removed stocks are excluded."
    )

    return validated
