"""
Data sanitization and eligibility checking.

This module enforces the strict data quality rules:
- NO forward-filling of prices
- Missing data excludes stock for that rebalance only
- At least 13 months of clean historical data required
- No missing prices in lookback periods used for calculations

CRITICAL: Eligibility is evaluated per-stock, per-rebalance-date.
"""

from datetime import date
from typing import Dict, List, Set, Tuple
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

# Minimum historical data requirements (defaults)
DEFAULT_MIN_HISTORY_MONTHS = 13  # At least 13 months before rebalance date
DEFAULT_MOMENTUM_LOOKBACK_MONTHS = 12  # For 12-month momentum
DEFAULT_VOLATILITY_LOOKBACK_MONTHS = 6  # For 6-month volatility


def required_history_months(
    min_history_months: int,
    momentum_lookback_months: int,
    volatility_lookback_months: int,
) -> int:
    """Compute the minimum history window (months) required for features."""
    return max(min_history_months, momentum_lookback_months, volatility_lookback_months)


def history_gate_counts(
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    required_months: int,
) -> Tuple[int, int, int, int]:
    """
    Return (daily_available, daily_required, monthly_available, monthly_required)
    for the required history window ending at rebalance_date (exclusive).
    """
    start_date = rebalance_date - pd.DateOffset(months=required_months)
    expected_daily = trading_dates[(trading_dates >= start_date) & (trading_dates < rebalance_date)]
    daily_required = len(expected_daily)
    daily_available = prices.reindex(expected_daily).notna().sum()

    expected_monthly = pd.date_range(
        start=start_date,
        end=rebalance_date - pd.Timedelta(days=1),
        freq="ME",
    )
    monthly_required = len(expected_monthly)
    monthly_prices = prices.reindex(expected_daily).dropna()
    monthly_available = monthly_prices.resample("ME").last().dropna().shape[0]

    return int(daily_available), int(daily_required), int(monthly_available), int(monthly_required)


def history_gate_passes(
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    required_months: int,
    history_gate_mode: str,
    history_leniency: float,
) -> Tuple[bool, Dict[str, int]]:
    """
    Evaluate history sufficiency for a symbol at a rebalance date.
    Returns (passes, counts) where counts include available/required points.
    """
    first_valid = prices.first_valid_index()
    last_valid = prices.last_valid_index()

    if first_valid is None or last_valid is None:
        return False, {
            "daily_available": 0,
            "daily_required": 0,
            "monthly_available": 0,
            "monthly_required": 0,
        }

    if rebalance_date <= first_valid:
        return False, {
            "daily_available": 0,
            "daily_required": 0,
            "monthly_available": 0,
            "monthly_required": 0,
        }
    if rebalance_date > last_valid:
        return False, {
            "daily_available": 0,
            "daily_required": 0,
            "monthly_available": 0,
            "monthly_required": 0,
        }

    daily_available, daily_required, monthly_available, monthly_required = history_gate_counts(
        prices,
        rebalance_date,
        trading_dates,
        required_months,
    )

    if history_gate_mode == "lenient":
        daily_required = int(np.ceil(daily_required * (1.0 - history_leniency)))
        monthly_required = int(np.ceil(monthly_required * (1.0 - history_leniency)))

    passes = daily_available >= daily_required and monthly_available >= monthly_required
    return passes, {
        "daily_available": daily_available,
        "daily_required": daily_required,
        "monthly_available": monthly_available,
        "monthly_required": monthly_required,
    }


def get_eligible_stocks_with_history_gate(
    price_data: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    min_history_months: int,
    momentum_lookback_months: int,
    volatility_lookback_months: int,
    history_gate_mode: str,
    history_leniency: float,
) -> Tuple[Set[str], int, int]:
    """
    Apply history gate before scoring and return:
    (eligible_tickers, insufficient_history_excluded, eligible_after_history)
    """
    eligible: Set[str] = set()
    insufficient_history_excluded = 0
    eligible_after_history = 0
    required_months = required_history_months(
        min_history_months,
        momentum_lookback_months,
        volatility_lookback_months,
    )

    for ticker in price_data.columns:
        prices = price_data[ticker]
        history_ok, _ = history_gate_passes(
            prices,
            rebalance_date,
            trading_dates,
            required_months,
            history_gate_mode,
            history_leniency,
        )
        if not history_ok:
            insufficient_history_excluded += 1
            continue
        eligible_after_history += 1

        if history_gate_mode == "strict":
            is_eligible, _ = check_stock_eligibility(
                ticker,
                prices,
                rebalance_date,
                trading_dates,
                min_history_months=min_history_months,
                momentum_lookback_months=momentum_lookback_months,
                volatility_lookback_months=volatility_lookback_months,
            )
            if not is_eligible:
                continue
        eligible.add(ticker)

    return eligible, insufficient_history_excluded, eligible_after_history


def check_stock_eligibility(
    ticker: str,
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    momentum_lookback_months: int = DEFAULT_MOMENTUM_LOOKBACK_MONTHS,
    volatility_lookback_months: int = DEFAULT_VOLATILITY_LOOKBACK_MONTHS,
) -> Tuple[bool, str]:
    """
    Check if a stock is eligible for selection at a given rebalance date.

    Eligibility criteria:
    1. At least 13 months of clean historical data before rebalance date
    2. No missing prices in last 12 months (momentum calculation)
    3. No missing prices in last 6 months (volatility calculation)
    4. Ignore pre-listing/post-delisting gaps outside the ticker's valid price span

    Args:
        ticker: Stock ticker symbol
        prices: Price series for the stock (may have NaN values)
        rebalance_date: The rebalance date to check eligibility for
        trading_dates: Full trading calendar

    Returns:
        Tuple of (is_eligible, reason)
        - is_eligible: True if stock passes all eligibility checks
        - reason: Empty string if eligible, else explanation of failure
    """
    # Get trading dates before the rebalance date
    past_dates = trading_dates[trading_dates < rebalance_date]

    if len(past_dates) == 0:
        return False, "No historical trading dates"

    first_valid = prices.first_valid_index()
    last_valid = prices.last_valid_index()

    if first_valid is None or last_valid is None:
        return False, "No price data"

    # Exclude pre-listing and post-delisting periods
    if rebalance_date <= first_valid:
        return False, "Not listed yet"
    if rebalance_date > last_valid:
        return False, "Delisted"

    # Calculate lookback dates
    lookback_13m = rebalance_date - pd.DateOffset(months=min_history_months)
    lookback_12m = rebalance_date - pd.DateOffset(months=momentum_lookback_months)
    lookback_6m = rebalance_date - pd.DateOffset(months=volatility_lookback_months)

    if first_valid > lookback_13m:
        return False, f"Insufficient history: IPO < {min_history_months} months"

    # Check 1: At least 13 months of history
    history_start = max(lookback_13m, first_valid)
    history_mask = (prices.index >= history_start) & (prices.index < rebalance_date)
    history_prices = prices[history_mask]

    # Count expected trading days in the 13-month period
    expected_dates_13m = trading_dates[(trading_dates >= history_start) & (trading_dates < rebalance_date)]

    if len(history_prices) == 0:
        return False, f"No price data in 13-month lookback"

    history_aligned = history_prices.reindex(expected_dates_13m)
    missing_history = history_aligned.isna().sum()
    if missing_history > 0:
        return False, (
            f"Missing {missing_history} prices in {min_history_months}-month history period"
        )

    # Check 2: No missing prices in last 12 months
    start_12m = max(lookback_12m, first_valid)
    mask_12m = (prices.index >= start_12m) & (prices.index < rebalance_date)
    prices_12m = prices[mask_12m]
    expected_dates_12m = trading_dates[(trading_dates >= start_12m) & (trading_dates < rebalance_date)]

    # Reindex to expected dates and check for NaN
    prices_12m_aligned = prices_12m.reindex(expected_dates_12m)
    missing_12m = prices_12m_aligned.isna().sum()

    if missing_12m > 0:
        return False, f"Missing {missing_12m} prices in 12-month momentum period"

    # Check 3: No missing prices in last 6 months
    start_6m = max(lookback_6m, first_valid)
    mask_6m = (prices.index >= start_6m) & (prices.index < rebalance_date)
    prices_6m = prices[mask_6m]
    expected_dates_6m = trading_dates[(trading_dates >= start_6m) & (trading_dates < rebalance_date)]

    prices_6m_aligned = prices_6m.reindex(expected_dates_6m)
    missing_6m = prices_6m_aligned.isna().sum()

    if missing_6m > 0:
        return False, f"Missing {missing_6m} prices in 6-month volatility period"

    return True, ""


def get_eligible_stocks(
    price_data: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    momentum_lookback_months: int = DEFAULT_MOMENTUM_LOOKBACK_MONTHS,
    volatility_lookback_months: int = DEFAULT_VOLATILITY_LOOKBACK_MONTHS,
) -> Set[str]:
    """
    Get the set of eligible stocks for a given rebalance date.

    Args:
        price_data: DataFrame with stocks as columns, dates as index
        rebalance_date: The rebalance date to check eligibility for
        trading_dates: Full trading calendar

    Returns:
        Set of ticker symbols that pass all eligibility checks
    """
    eligible: Set[str] = set()
    ineligible_reasons: Dict[str, str] = {}

    for ticker in price_data.columns:
        prices = price_data[ticker]
        is_eligible, reason = check_stock_eligibility(
            ticker,
            prices,
            rebalance_date,
            trading_dates,
            min_history_months=min_history_months,
            momentum_lookback_months=momentum_lookback_months,
            volatility_lookback_months=volatility_lookback_months,
        )

        if is_eligible:
            eligible.add(ticker)
        else:
            ineligible_reasons[ticker] = reason

    logger.debug(
        f"Rebalance {rebalance_date.date()}: {len(eligible)}/{len(price_data.columns)} stocks eligible"
    )

    # Log summary of ineligibility reasons
    if ineligible_reasons:
        reason_counts: Dict[str, int] = {}
        for reason in ineligible_reasons.values():
            # Extract reason type (first part before colon or number)
            reason_type = reason.split(":")[0].split(" ")[0]
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        logger.debug(f"Ineligibility breakdown: {reason_counts}")

    return eligible


def sanitize_stock_data(
    price_data: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Sanitize stock price data WITHOUT forward-filling.

    This function:
    1. Aligns data to the trading calendar
    2. Logs coverage statistics
    3. Does NOT modify or fill any values

    Args:
        price_data: DataFrame with stocks as columns, dates as index
        trading_dates: Full trading calendar

    Returns:
        DataFrame aligned to trading_dates with NaN preserved

    Note:
        Eligibility filtering happens separately per rebalance date.
        This function only handles alignment and logging.
    """
    # Ensure index is DatetimeIndex
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.DatetimeIndex(price_data.index)

    # Reindex to trading calendar (introduces NaN for missing dates)
    aligned = price_data.reindex(trading_dates)

    # Log coverage statistics
    coverage = aligned.notna().sum() / len(aligned)
    logger.info(
        f"Data coverage after alignment: "
        f"min={coverage.min():.1%}, median={coverage.median():.1%}, max={coverage.max():.1%}"
    )

    # Count stocks with good coverage (>95%)
    good_coverage = (coverage > 0.95).sum()
    logger.info(f"{good_coverage}/{len(coverage)} stocks have >95% coverage")

    return aligned


def compute_returns(
    prices: pd.Series,
    trading_dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """
    Compute total return over a period.

    Args:
        prices: Price series
        trading_dates: Full trading calendar
        start_date: Start of period
        end_date: End of period (exclusive)

    Returns:
        Total return as a decimal (e.g., 0.10 for 10%)

    Raises:
        ValueError: If prices are missing at start or end
    """
    # Get trading dates in period
    period_dates = trading_dates[(trading_dates >= start_date) & (trading_dates < end_date)]

    if len(period_dates) < 2:
        raise ValueError(f"Not enough trading dates in period {start_date} to {end_date}")

    start_price = prices.get(period_dates[0])
    end_price = prices.get(period_dates[-1])

    if pd.isna(start_price) or pd.isna(end_price):
        raise ValueError(f"Missing prices at period boundaries")

    return (end_price / start_price) - 1.0


def compute_volatility(
    prices: pd.Series,
    trading_dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    annualize: bool = True,
) -> float:
    """
    Compute realized volatility over a period.

    Args:
        prices: Price series
        trading_dates: Full trading calendar
        start_date: Start of period
        end_date: End of period (exclusive)
        annualize: If True, annualize the volatility (default: True)

    Returns:
        Volatility as a decimal (e.g., 0.20 for 20%)

    Raises:
        ValueError: If insufficient data in period
    """
    # Get trading dates in period
    period_dates = trading_dates[(trading_dates >= start_date) & (trading_dates < end_date)]

    # Get prices for these dates
    period_prices = prices.reindex(period_dates).dropna()

    if len(period_prices) < 20:  # Need reasonable sample for volatility
        raise ValueError(f"Insufficient data for volatility: {len(period_prices)} days")

    # Compute daily log returns
    log_returns = np.log(period_prices / period_prices.shift(1)).dropna()

    if len(log_returns) == 0:
        raise ValueError("No returns to compute volatility")

    # Standard deviation of daily returns
    daily_vol = log_returns.std()

    if annualize:
        # Annualize using sqrt(252) for Indian markets
        return daily_vol * np.sqrt(252)

    return daily_vol
