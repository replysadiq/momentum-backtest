"""
Rebalance calendar construction - SINGLE SOURCE OF TRUTH.

This module builds the authoritative monthly rebalance schedule based on
the first trading day of each month, derived from benchmark trading dates.

The calendar is frozen before backtesting and used consistently for:
- Feature computation
- State machine evaluation
- Trade execution
"""

from datetime import date
from typing import List
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def build_rebalance_calendar(
    trading_dates: pd.DatetimeIndex,
    start_date: date,
    end_date: date,
) -> pd.DatetimeIndex:
    """
    Build the monthly rebalance calendar.

    Rebalance day = first trading day of each month.

    Algorithm:
    1. Generate all calendar month starts within [start_date, end_date]
    2. For each month start:
       - If the 1st is a trading day → use it
       - Else → use the next available trading day
    3. Freeze this calendar before backtesting

    Args:
        trading_dates: DatetimeIndex of valid trading days (from benchmark)
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        DatetimeIndex of rebalance dates (first trading day of each month)

    Raises:
        ValueError: If no valid rebalance dates can be generated

    Example:
        If Jan 1 is a Sunday and Jan 2 is a holiday,
        and Jan 3 is the first trading day,
        then Jan 3 is the rebalance date for January.
    """
    # Convert to pandas Timestamps for comparison
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Filter trading dates to the backtest period
    mask = (trading_dates >= start_ts) & (trading_dates <= end_ts)
    period_trading_dates = trading_dates[mask].sort_values()

    if len(period_trading_dates) == 0:
        raise ValueError(f"No trading dates in period {start_date} to {end_date}")

    # Generate all month starts in the period
    # Start from the beginning of start_date's month
    month_starts = pd.date_range(
        start=start_ts.replace(day=1),
        end=end_ts,
        freq="MS",  # Month Start
    )

    rebalance_dates: List[pd.Timestamp] = []

    for month_start in month_starts:
        # Find the first trading day >= month_start
        candidates = period_trading_dates[period_trading_dates >= month_start]

        if len(candidates) == 0:
            # No more trading days - we've reached the end
            logger.debug(f"No trading day found for month starting {month_start.date()}")
            continue

        first_trading_day = candidates[0]

        # Only include if it's still within the same month
        # (handles edge case where month start is at end of period)
        if first_trading_day.month == month_start.month:
            rebalance_dates.append(first_trading_day)
            logger.debug(f"Month {month_start.strftime('%Y-%m')}: rebalance on {first_trading_day.date()}")
        elif first_trading_day <= end_ts:
            # If month had no trading days, the first trading day of next month is used
            # But we skip this to avoid double-counting
            logger.debug(f"Month {month_start.strftime('%Y-%m')}: no trading days, skipping")

    if not rebalance_dates:
        raise ValueError(f"No valid rebalance dates generated for period {start_date} to {end_date}")

    calendar = pd.DatetimeIndex(rebalance_dates).sort_values()

    logger.info(
        f"Rebalance calendar built: {len(calendar)} dates from "
        f"{calendar[0].date()} to {calendar[-1].date()}"
    )

    # Log first few rebalance dates for verification
    logger.debug(f"First 5 rebalance dates: {[d.date() for d in calendar[:5]]}")

    return calendar


def get_lookback_dates(
    rebalance_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    months: int,
) -> pd.DatetimeIndex:
    """
    Get trading dates for the lookback period before a rebalance date.

    Args:
        rebalance_date: The rebalance date to look back from
        trading_dates: Full trading calendar
        months: Number of months to look back

    Returns:
        DatetimeIndex of trading dates in the lookback period (exclusive of rebalance_date)

    Note:
        Uses calendar months for lookback, then maps to trading days.
        This ensures consistent behavior across different month lengths.
    """
    # Calculate lookback start date (N months before rebalance date)
    lookback_start = rebalance_date - pd.DateOffset(months=months)

    # Get trading dates in the lookback period (exclusive of rebalance_date)
    mask = (trading_dates >= lookback_start) & (trading_dates < rebalance_date)
    return trading_dates[mask].sort_values()


def validate_rebalance_calendar(
    calendar: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
) -> None:
    """
    Validate that all rebalance dates are valid trading days.

    Args:
        calendar: Rebalance calendar to validate
        trading_dates: Full trading calendar

    Raises:
        AssertionError: If any rebalance date is not a trading day
    """
    trading_set = set(trading_dates)

    for rebal_date in calendar:
        assert rebal_date in trading_set, (
            f"Rebalance date {rebal_date.date()} is not a valid trading day"
        )

    logger.debug(f"Validated {len(calendar)} rebalance dates are all trading days")
