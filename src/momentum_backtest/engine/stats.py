"""
Canonical implementations for statistical computations.

This is the SINGLE SOURCE OF TRUTH for:
- Period returns
- Realized volatility
- Maximum drawdown
- Price lookups

All computations use EXCLUSIVE end boundaries to prevent lookahead bias.
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def get_last_price_before(
    prices: pd.Series,
    end_date: pd.Timestamp,
) -> Optional[float]:
    """
    Get the last available price strictly before end_date.

    Args:
        prices: Price series
        end_date: Exclusive end boundary

    Returns:
        Last price before end_date, or None if no data
    """
    valid_prices = prices[prices.index < end_date].dropna()
    if len(valid_prices) == 0:
        return None
    return float(valid_prices.iloc[-1])


def get_first_price_on_or_after(
    prices: pd.Series,
    start_date: pd.Timestamp,
) -> Optional[float]:
    """
    Get the first available price on or after start_date.

    Args:
        prices: Price series
        start_date: Inclusive start boundary

    Returns:
        First price on/after start_date, or None if no data
    """
    valid_prices = prices[prices.index >= start_date].dropna()
    if len(valid_prices) == 0:
        return None
    return float(valid_prices.iloc[0])


def total_return(
    prices: pd.Series,
    start_date: pd.Timestamp,
    end_date_exclusive: pd.Timestamp,
) -> Optional[float]:
    """
    Compute total return over a period with EXCLUSIVE end boundary.

    Return = (last_price_before_end / first_price_on_or_after_start) - 1

    Args:
        prices: Price series
        start_date: Inclusive start of period
        end_date_exclusive: Exclusive end of period (no lookahead)

    Returns:
        Total return as decimal, or None if insufficient data
    """
    # Get prices strictly within [start, end)
    mask = (prices.index >= start_date) & (prices.index < end_date_exclusive)
    period_prices = prices[mask].dropna()

    if len(period_prices) < 2:
        return None

    start_price = period_prices.iloc[0]
    end_price = period_prices.iloc[-1]

    if start_price <= 0:
        return None

    return (end_price / start_price) - 1.0


def total_return_months(
    prices: pd.Series,
    end_date_exclusive: pd.Timestamp,
    months: int,
) -> Optional[float]:
    """
    Compute total return for the last N months before end_date.

    Args:
        prices: Price series
        end_date_exclusive: Exclusive end of period
        months: Number of months to look back

    Returns:
        Total return as decimal, or None if insufficient data
    """
    start_date = end_date_exclusive - pd.DateOffset(months=months)
    return total_return(prices, start_date, end_date_exclusive)


def realized_vol(
    prices: pd.Series,
    end_date_exclusive: pd.Timestamp,
    months: int,
    annualize: bool = True,
    min_samples: int = 20,
    trading_dates: Optional[pd.DatetimeIndex] = None,
) -> Optional[float]:
    """
    Compute realized volatility with EXCLUSIVE end boundary.

    Uses log returns for accuracy.

    Args:
        prices: Price series
        end_date_exclusive: Exclusive end of period
        months: Number of months to look back
        annualize: Whether to annualize (default: True, uses sqrt(252))
        min_samples: Minimum number of returns required

    Returns:
        Volatility as decimal, or None if insufficient data
    """
    start_date = end_date_exclusive - pd.DateOffset(months=months)

    # Determine required samples from trading calendar if provided.
    if trading_dates is not None:
        required_samples = int(
            ((trading_dates >= start_date) & (trading_dates < end_date_exclusive)).sum()
        )
    else:
        required_samples = min_samples

    # Get prices strictly within [start, end)
    mask = (prices.index >= start_date) & (prices.index < end_date_exclusive)
    period_prices = prices[mask].dropna()

    if len(period_prices) < required_samples:
        return None

    # Compute log returns
    log_returns = np.log(period_prices / period_prices.shift(1)).dropna()

    if len(log_returns) < required_samples - 1:
        return None

    daily_vol = log_returns.std()

    if daily_vol <= 0:
        return None

    if annualize:
        return daily_vol * np.sqrt(252)

    return daily_vol


def max_drawdown(
    prices: pd.Series,
    end_date_exclusive: pd.Timestamp,
    months: int,
    min_samples: int = 5,
) -> Optional[float]:
    """
    Compute maximum drawdown with EXCLUSIVE end boundary.

    Args:
        prices: Price series
        end_date_exclusive: Exclusive end of period
        months: Number of months to look back
        min_samples: Minimum number of prices required

    Returns:
        Max drawdown as positive decimal (e.g., 0.20 for 20%), or None
    """
    start_date = end_date_exclusive - pd.DateOffset(months=months)

    # Get prices strictly within [start, end)
    mask = (prices.index >= start_date) & (prices.index < end_date_exclusive)
    period_prices = prices[mask].dropna()

    if len(period_prices) < min_samples:
        return None

    running_max = period_prices.expanding().max()
    drawdown = (period_prices - running_max) / running_max

    return abs(drawdown.min())


def positive_month_percentage(
    prices: pd.Series,
    end_date_exclusive: pd.Timestamp,
    months: int = 12,
) -> Optional[float]:
    """
    Compute percentage of positive months in lookback period.

    Args:
        prices: Price series
        end_date_exclusive: Exclusive end of period
        months: Number of months to look back

    Returns:
        Fraction of months with positive returns (0.0 to 1.0), or None
    """
    start_date = end_date_exclusive - pd.DateOffset(months=months)

    # Get prices strictly within [start, end)
    mask = (prices.index >= start_date) & (prices.index < end_date_exclusive)
    period_prices = prices[mask].dropna()

    if len(period_prices) < 20:
        return None

    # Resample to monthly and compute returns
    monthly_prices = period_prices.resample("ME").last().dropna()

    if len(monthly_prices) < 2:
        return None

    monthly_returns = monthly_prices.pct_change().dropna()

    if len(monthly_returns) == 0:
        return None

    positive_count = (monthly_returns > 0).sum()
    return positive_count / len(monthly_returns)


def compute_turnover(
    prev_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """
    Compute one-way portfolio turnover.

    Turnover = sum(|target_weight - prev_weight|) for all assets.

    This represents total notional traded as fraction of portfolio.
    - Full liquidation (invested -> cash): turnover = 1.0
    - Full rotation to disjoint set: turnover can approach 2.0

    Args:
        prev_weights: Previous portfolio weights (Series or dict-like)
        target_weights: Target portfolio weights (Series or dict-like)

    Returns:
        One-way turnover as decimal
    """
    # Convert to Series if needed
    if not isinstance(prev_weights, pd.Series):
        prev_weights = pd.Series(prev_weights)
    if not isinstance(target_weights, pd.Series):
        target_weights = pd.Series(target_weights)

    # Align indices - missing keys get 0 weight
    all_assets = prev_weights.index.union(target_weights.index)
    prev_aligned = prev_weights.reindex(all_assets, fill_value=0.0)
    target_aligned = target_weights.reindex(all_assets, fill_value=0.0)

    # One-way turnover = sum of absolute differences
    return float((target_aligned - prev_aligned).abs().sum())


def apply_transaction_cost(
    gross_return: float,
    turnover: float,
    tc_bps: float,
) -> float:
    """
    Apply transaction cost to gross return.

    Args:
        gross_return: Gross return as decimal
        turnover: One-way turnover
        tc_bps: Transaction cost in basis points

    Returns:
        Net return after costs
    """
    cost = (tc_bps / 10000.0) * turnover
    return gross_return - cost
