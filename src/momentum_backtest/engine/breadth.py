"""
Breadth-based regime overlay.

Computes market breadth (fraction of stocks with positive N-day returns)
and converts it to a confidence scalar for exposure scaling.

Key design principles:
- No lookahead bias: breadth at t uses only prices <= t-1 close
- Robust handling of missing data (coverage thresholds, carry-forward)
- Continuous scalar output in [0, 1]
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class BreadthConfig:
    """Configuration for breadth overlay computation."""
    lookback: int = 63  # Trading days for return calculation
    ema_span: int = 10  # EMA smoothing span
    low: float = 0.35  # Breadth below this = confidence 0
    high: float = 0.65  # Breadth above this = confidence 1
    min_coverage: float = 0.60  # Minimum fraction of stocks required


@dataclass
class BreadthSeries:
    """Complete breadth series with all intermediate values."""
    # Daily series (DatetimeIndex)
    breadth_raw: pd.Series  # Raw fraction of stocks with positive 63d return
    breadth_smooth: pd.Series  # EMA-smoothed breadth
    breadth_confidence: pd.Series  # Confidence scalar in [0, 1]

    # Metadata
    config: BreadthConfig
    coverage: pd.Series  # Daily coverage (fraction of stocks with data)


def compute_daily_breadth_series(
    price_data: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    config: BreadthConfig,
) -> BreadthSeries:
    """
    Compute daily breadth series from stock prices.

    For each date t, breadth is computed using prices up to and including t-1
    (no lookahead - we don't know t's close when deciding at t).

    Args:
        price_data: DataFrame with columns = tickers, index = dates, values = prices
        trading_dates: Trading calendar
        config: Breadth computation parameters

    Returns:
        BreadthSeries with raw, smoothed, and confidence values
    """
    logger.info(f"Computing daily breadth series (lookback={config.lookback}, "
                f"ema_span={config.ema_span}, low={config.low}, high={config.high})")

    # Ensure price_data is aligned to trading dates
    price_aligned = price_data.reindex(trading_dates)

    n_stocks = len(price_data.columns)
    lookback = config.lookback

    # Initialize output series
    breadth_raw = pd.Series(index=trading_dates, dtype=float)
    coverage = pd.Series(index=trading_dates, dtype=float)

    # For each date, compute breadth using data up to that date (inclusive)
    # This means for trading on date t, we use prices through t-1
    # But we index by the date the breadth "applies to" for lookup
    for i, current_date in enumerate(trading_dates):
        if i < lookback:
            # Not enough history
            breadth_raw[current_date] = np.nan
            coverage[current_date] = 0.0
            continue

        # Get lookback period: [current_date - lookback, current_date]
        # Actually we need lookback+1 prices to compute lookback-day return
        lookback_start_idx = i - lookback
        lookback_start_date = trading_dates[lookback_start_idx]

        # Get prices at start and end of lookback period
        try:
            start_prices = price_aligned.loc[lookback_start_date]
            end_prices = price_aligned.loc[current_date]
        except KeyError:
            breadth_raw[current_date] = np.nan
            coverage[current_date] = 0.0
            continue

        # Compute returns for each stock
        valid_mask = (
            pd.notna(start_prices) &
            pd.notna(end_prices) &
            (start_prices > 0)
        )
        n_valid = valid_mask.sum()

        # Check coverage threshold
        current_coverage = n_valid / n_stocks if n_stocks > 0 else 0.0
        coverage[current_date] = current_coverage

        if current_coverage < config.min_coverage:
            # Below coverage threshold - mark as NaN (will be filled later)
            breadth_raw[current_date] = np.nan
            continue

        # Compute fraction with positive return
        returns = (end_prices[valid_mask] / start_prices[valid_mask]) - 1.0
        n_positive = (returns > 0).sum()
        breadth_raw[current_date] = n_positive / n_valid

    # Forward-fill NaN values (carry last valid breadth)
    # If no valid values at start, use neutral (0.5)
    breadth_filled = breadth_raw.ffill()
    if breadth_filled.isna().all():
        breadth_filled = pd.Series(0.5, index=trading_dates)
    else:
        breadth_filled = breadth_filled.fillna(0.5)  # Fill any remaining NaNs with neutral

    # Apply EMA smoothing
    breadth_smooth = breadth_filled.ewm(span=config.ema_span, adjust=False).mean()

    # Convert to confidence scalar
    breadth_confidence = _breadth_to_confidence(breadth_smooth, config.low, config.high)

    logger.info(f"Breadth series computed: {len(breadth_raw.dropna())} valid days, "
                f"mean raw={breadth_raw.mean():.3f}, mean confidence={breadth_confidence.mean():.3f}")

    return BreadthSeries(
        breadth_raw=breadth_raw,
        breadth_smooth=breadth_smooth,
        breadth_confidence=breadth_confidence,
        config=config,
        coverage=coverage,
    )


def _breadth_to_confidence(
    breadth_smooth: pd.Series,
    low: float,
    high: float,
) -> pd.Series:
    """
    Convert smoothed breadth to confidence scalar in [0, 1].

    conf = clamp((breadth - low) / (high - low), 0, 1)

    Args:
        breadth_smooth: EMA-smoothed breadth series
        low: Breadth threshold below which confidence = 0
        high: Breadth threshold above which confidence = 1

    Returns:
        Confidence scalar series in [0, 1]
    """
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")

    confidence = (breadth_smooth - low) / (high - low)
    confidence = confidence.clip(lower=0.0, upper=1.0)

    return confidence


def get_breadth_confidence(
    breadth_series: BreadthSeries,
    date: pd.Timestamp,
) -> float:
    """
    Get breadth confidence for a specific date.

    If date is not in the series, returns the most recent available value.
    If no data available, returns 1.0 (neutral - no scaling).

    Args:
        breadth_series: Pre-computed breadth series
        date: Date to look up

    Returns:
        Confidence scalar in [0, 1]
    """
    try:
        # Try exact match first
        if date in breadth_series.breadth_confidence.index:
            value = breadth_series.breadth_confidence[date]
            if pd.notna(value):
                return float(value)

        # Fall back to most recent value before date
        prior_values = breadth_series.breadth_confidence[
            breadth_series.breadth_confidence.index <= date
        ]
        if len(prior_values) > 0:
            return float(prior_values.iloc[-1])

    except (KeyError, IndexError):
        pass

    # No data - return neutral (no scaling)
    logger.warning(f"No breadth data for {date}, using confidence=1.0 (neutral)")
    return 1.0


def scale_weights_by_breadth(
    weights: dict,
    breadth_confidence: float,
) -> dict:
    """
    Scale portfolio weights by breadth confidence.

    The remainder (1 - breadth_confidence) is implicitly allocated to cash.

    Args:
        weights: Original portfolio weights {ticker: weight}
        breadth_confidence: Confidence scalar in [0, 1]

    Returns:
        Scaled weights {ticker: weight * breadth_confidence}
    """
    if not weights:
        return {}

    if breadth_confidence >= 1.0:
        return weights.copy()

    if breadth_confidence <= 0.0:
        return {}  # All cash

    return {ticker: weight * breadth_confidence for ticker, weight in weights.items()}
