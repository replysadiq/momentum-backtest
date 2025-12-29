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
    rebalance_months: int = 1,
) -> pd.DatetimeIndex:
    """
    Build the rebalance calendar with configurable frequency.

    Rebalance day = first trading day of each selected month.

    Algorithm:
    1. Find anchor date: first trading day of the month on/after start_date
    2. From anchor, step N months at a time (where N = rebalance_months)
    3. For each selected month, use the first trading day of that month
    4. Continue until we exceed end_date

    Args:
        trading_dates: DatetimeIndex of valid trading days (from benchmark)
        start_date: Backtest start date
        end_date: Backtest end date
        rebalance_months: Rebalance frequency (1=monthly, 2=bi-monthly, 3=quarterly)

    Returns:
        DatetimeIndex of rebalance dates

    Raises:
        ValueError: If no valid rebalance dates can be generated

    Invariants:
        - Exactly one rebalance per selected rebalance month
        - No rebalance dates outside [start_date, end_date]
        - Strictly increasing dates
        - Rebalance months step exactly N months apart (no drift)

    Example:
        If rebalance_months=2 and start_date is Jan 15, 2020:
        - Anchor = first trading day of Jan 2020 (e.g., Jan 2)
        - Rebalances: Jan 2020, Mar 2020, May 2020, Jul 2020, ...
    """
    # Convert to pandas Timestamps for comparison
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Filter trading dates to the backtest period
    mask = (trading_dates >= start_ts) & (trading_dates <= end_ts)
    period_trading_dates = trading_dates[mask].sort_values()

    if len(period_trading_dates) == 0:
        raise ValueError(f"No trading dates in period {start_date} to {end_date}")

    # Step 1: Find anchor month (the month containing or after start_date)
    anchor_month_start = start_ts.replace(day=1)

    # Step 2: Find the first trading day of anchor month (this is our first rebalance)
    anchor_candidates = period_trading_dates[period_trading_dates >= anchor_month_start]
    if len(anchor_candidates) == 0:
        raise ValueError(f"No trading days found on/after {start_date}")

    first_trading_day = anchor_candidates[0]

    # If first trading day is in a different month, use that month as anchor
    if first_trading_day.month != anchor_month_start.month:
        anchor_month_start = first_trading_day.replace(day=1)

    rebalance_dates: List[pd.Timestamp] = []
    current_month = anchor_month_start
    month_step = 0

    while current_month <= end_ts:
        # Find the first trading day of current_month
        month_candidates = period_trading_dates[
            (period_trading_dates >= current_month) &
            (period_trading_dates.month == current_month.month) &
            (period_trading_dates.year == current_month.year)
        ]

        if len(month_candidates) > 0:
            rebal_date = month_candidates[0]
            # Only include if within [start_date, end_date]
            if rebal_date >= start_ts and rebal_date <= end_ts:
                rebalance_dates.append(rebal_date)
                logger.debug(
                    f"Month {current_month.strftime('%Y-%m')}: "
                    f"rebalance on {rebal_date.date()}"
                )
        else:
            logger.debug(
                f"Month {current_month.strftime('%Y-%m')}: no trading days, skipping"
            )

        # Step to next rebalance month
        month_step += rebalance_months
        current_month = anchor_month_start + pd.DateOffset(months=month_step)

    if not rebalance_dates:
        raise ValueError(
            f"No valid rebalance dates generated for period {start_date} to {end_date}"
        )

    calendar = pd.DatetimeIndex(rebalance_dates).sort_values()

    # Validate invariants
    _validate_calendar_invariants(calendar, start_ts, end_ts, rebalance_months)

    freq_label = {1: "monthly", 2: "bi-monthly", 3: "quarterly"}[rebalance_months]
    logger.info(
        f"Rebalance calendar built ({freq_label}): {len(calendar)} dates from "
        f"{calendar[0].date()} to {calendar[-1].date()}"
    )

    # Log first few rebalance dates for verification
    logger.debug(f"First 5 rebalance dates: {[d.date() for d in calendar[:5]]}")

    return calendar


def _validate_calendar_invariants(
    calendar: pd.DatetimeIndex,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    rebalance_months: int,
) -> None:
    """
    Validate calendar invariants.

    Raises:
        AssertionError: If any invariant is violated
    """
    # Invariant 1: No dates outside [start_date, end_date]
    assert calendar[0] >= start_ts, (
        f"First rebalance {calendar[0].date()} is before start_date {start_ts.date()}"
    )
    assert calendar[-1] <= end_ts, (
        f"Last rebalance {calendar[-1].date()} is after end_date {end_ts.date()}"
    )

    # Invariant 2: Strictly increasing
    for i in range(1, len(calendar)):
        assert calendar[i] > calendar[i - 1], (
            f"Calendar not strictly increasing: {calendar[i - 1].date()} >= {calendar[i].date()}"
        )

    # Invariant 3: Rebalance months step exactly N months apart
    if len(calendar) >= 2:
        for i in range(1, len(calendar)):
            prev_date = calendar[i - 1]
            curr_date = calendar[i]
            # Calculate month difference
            month_diff = (
                (curr_date.year - prev_date.year) * 12 +
                (curr_date.month - prev_date.month)
            )
            assert month_diff == rebalance_months, (
                f"Month step between {prev_date.date()} and {curr_date.date()} "
                f"is {month_diff}, expected {rebalance_months}"
            )


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
