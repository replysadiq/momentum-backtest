"""
Validation and invariant checking.

This module provides explicit checks for:
- No lookahead bias (strict slicing)
- No NaNs in computed features
- Weights sum to 1 (RISK_ON) or 0 (CASH/PANIC)
- Rebalance dates are valid trading days
- Data integrity

All checks abort execution on failure (fail-fast).
"""

from typing import Any, Dict, List, Set, Union
import logging

import pandas as pd

from .engine.state_machine import MarketState
from .engine.backtest import RebalanceRecord


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a validation invariant is violated."""
    pass


def _is_nan_safe(value: Any) -> bool:
    """
    Check if a value is NaN or None, safely handling all types.

    Uses pd.isna() which handles floats, None, NaT, and other types
    without raising TypeError (unlike np.isnan which crashes on non-floats).

    Args:
        value: Any value to check

    Returns:
        True if value is NaN, None, or NaT
    """
    return pd.isna(value)


def validate_rebalance_dates_are_trading_days(
    rebalance_calendar: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
) -> None:
    """
    Assert that all rebalance dates are valid trading days.

    Args:
        rebalance_calendar: Rebalance dates to validate
        trading_dates: Full trading calendar

    Raises:
        ValidationError: If any rebalance date is not a trading day
    """
    trading_set = set(trading_dates)
    invalid_dates = []

    for date in rebalance_calendar:
        if date not in trading_set:
            invalid_dates.append(date)

    if invalid_dates:
        raise ValidationError(
            f"Rebalance dates are not trading days: "
            f"{[d.strftime('%Y-%m-%d') for d in invalid_dates[:5]]}"
            f"{'...' if len(invalid_dates) > 5 else ''}"
        )

    logger.debug(f"Validated {len(rebalance_calendar)} rebalance dates are trading days")


def validate_no_nans_in_features(
    rebalance_records: List[RebalanceRecord],
) -> None:
    """
    Assert that no NaNs exist in computed features used for decisions.

    Args:
        rebalance_records: List of rebalance records to validate

    Raises:
        ValidationError: If any feature contains NaN
    """
    for record in rebalance_records:
        features = record.features

        # Check all feature values
        values = {
            "benchmark_return_6m": features.benchmark_return_6m,
            "benchmark_return_3m": features.benchmark_return_3m,
            "benchmark_vol_1m": features.benchmark_vol_1m,
            "benchmark_vol_6m": features.benchmark_vol_6m,
            "portfolio_drawdown_3m": features.portfolio_drawdown_3m,
        }

        for name, value in values.items():
            if _is_nan_safe(value):
                raise ValidationError(
                    f"NaN in feature '{name}' at rebalance date {record.date.strftime('%Y-%m-%d')}"
                )

    logger.debug(f"Validated no NaNs in {len(rebalance_records)} rebalance records")


def validate_weights_sum(
    rebalance_records: List[RebalanceRecord],
    tolerance: float = 1e-6,
) -> None:
    """
    Assert that weights sum to 1 (invested) or 0 (not invested).

    V3.1: CASH state with invested_flag=True is allowed to have weights (defensive replacement).

    Args:
        rebalance_records: List of rebalance records to validate
        tolerance: Acceptable deviation from target sum

    Raises:
        ValidationError: If weights don't sum correctly
    """
    # States where we should be invested (weights sum to 1 when stocks selected)
    invested_states = {MarketState.RISK_ON, MarketState.DEFENSIVE_MOMENTUM}

    for record in rebalance_records:
        weight_sum = sum(record.weights.values())
        n_stocks = len(record.weights)

        # V3.1: Check invested_flag - if True, this is an invested state regardless of state name
        is_invested = record.state in invested_states or record.invested_flag

        if is_invested:
            # RISK_ON, DEFENSIVE_MOMENTUM, or CASH with defensive replacement
            if n_stocks > 0:
                expected = 1.0
                if abs(weight_sum - expected) > tolerance:
                    raise ValidationError(
                        f"Weights sum to {weight_sum:.6f} (expected {expected}) "
                        f"at rebalance date {record.date.strftime('%Y-%m-%d')} in {record.state.name} state with {n_stocks} stocks"
                    )
            else:
                # No eligible stocks - effectively in cash, weights should be 0
                if weight_sum > tolerance:
                    raise ValidationError(
                        f"Weights sum to {weight_sum:.6f} but no stocks selected "
                        f"at rebalance date {record.date.strftime('%Y-%m-%d')} in {record.state.name} state"
                    )
        else:
            # PANIC or CASH (without defensive replacement) - should have no weights (sum = 0)
            expected = 0.0
            if abs(weight_sum - expected) > tolerance:
                raise ValidationError(
                    f"Weights sum to {weight_sum:.6f} (expected {expected}) "
                    f"at rebalance date {record.date.strftime('%Y-%m-%d')} in {record.state.name} state"
                )

    logger.debug(f"Validated weight sums for {len(rebalance_records)} rebalance records")


def validate_no_lookahead(
    price_data: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    tickers_used: Set[str],
) -> None:
    """
    Assert that only data before rebalance date was used for selection.

    This validates that the prices used for computation are strictly
    before the rebalance date.

    Args:
        price_data: Full price DataFrame
        rebalance_date: The rebalance date
        tickers_used: Set of tickers that were selected

    Raises:
        ValidationError: If data after rebalance date was accessed

    Note:
        This is a structural check. The actual enforcement happens
        in the data slicing during computation.
    """
    # Check that we have data before the rebalance date for selected tickers
    for ticker in tickers_used:
        if ticker not in price_data.columns:
            continue

        prices = price_data[ticker]
        before_rebalance = prices[prices.index < rebalance_date]

        if len(before_rebalance) == 0:
            raise ValidationError(
                f"No data before rebalance date {rebalance_date.strftime('%Y-%m-%d')} "
                f"for selected ticker {ticker} - possible lookahead"
            )


def validate_no_nans_in_equity_curve(
    equity_curve: pd.Series,
) -> None:
    """
    Assert that the equity curve has no NaN values.

    Args:
        equity_curve: Portfolio equity curve

    Raises:
        ValidationError: If any NaN values exist
    """
    nan_count = equity_curve.isna().sum()

    if nan_count > 0:
        nan_dates = equity_curve[equity_curve.isna()].index[:5]
        raise ValidationError(
            f"Equity curve contains {nan_count} NaN values. "
            f"First occurrences: {[d.strftime('%Y-%m-%d') for d in nan_dates]}"
        )

    logger.debug(f"Validated equity curve has no NaNs ({len(equity_curve)} values)")


def validate_state_transitions(
    rebalance_records: List[RebalanceRecord],
) -> None:
    """
    Validate that state transitions follow the allowed rules.

    Allowed transitions:
    - RISK_ON -> RISK_ON, PANIC, DEFENSIVE_MOMENTUM, CASH
    - PANIC -> PANIC, CASH
    - DEFENSIVE_MOMENTUM -> DEFENSIVE_MOMENTUM, CASH (V2 Lever A)
    - CASH -> CASH, RISK_ON

    Args:
        rebalance_records: List of rebalance records

    Raises:
        ValidationError: If an invalid transition occurred
    """
    allowed_transitions = {
        MarketState.RISK_ON: {MarketState.RISK_ON, MarketState.PANIC, MarketState.DEFENSIVE_MOMENTUM, MarketState.CASH},
        MarketState.PANIC: {MarketState.PANIC, MarketState.CASH},
        MarketState.DEFENSIVE_MOMENTUM: {MarketState.DEFENSIVE_MOMENTUM, MarketState.CASH},  # V2 Lever A
        MarketState.CASH: {MarketState.CASH, MarketState.RISK_ON},
    }

    for i in range(1, len(rebalance_records)):
        prev_state = rebalance_records[i - 1].state
        curr_state = rebalance_records[i].state
        date = rebalance_records[i].date

        if curr_state not in allowed_transitions[prev_state]:
            raise ValidationError(
                f"Invalid state transition {prev_state.name} -> {curr_state.name} "
                f"at {date.strftime('%Y-%m-%d')}"
            )

    logger.debug(f"Validated {len(rebalance_records)} state transitions")


def validate_price_data_coverage(
    price_data: pd.DataFrame,
    min_coverage: float = 0.5,
) -> None:
    """
    Validate that price data has reasonable coverage.

    Args:
        price_data: Stock price DataFrame
        min_coverage: Minimum fraction of stocks with >80% data

    Raises:
        ValidationError: If coverage is too low
    """
    coverage = price_data.notna().sum() / len(price_data)
    good_coverage_count = (coverage > 0.80).sum()
    coverage_ratio = good_coverage_count / len(coverage) if len(coverage) > 0 else 0

    if coverage_ratio < min_coverage:
        raise ValidationError(
            f"Insufficient data coverage: only {good_coverage_count}/{len(coverage)} "
            f"stocks have >80% data coverage (minimum {min_coverage:.0%} required)"
        )

    logger.debug(f"Data coverage validation passed: {coverage_ratio:.1%} with good coverage")


def run_all_validations(
    rebalance_calendar: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
    rebalance_records: List[RebalanceRecord],
    equity_curve: pd.Series,
    price_data: pd.DataFrame,
) -> None:
    """
    Run all validation checks.

    Args:
        rebalance_calendar: Rebalance dates
        trading_dates: Full trading calendar
        rebalance_records: List of rebalance records
        equity_curve: Portfolio equity curve
        price_data: Stock price DataFrame

    Raises:
        ValidationError: If any validation fails
    """
    logger.info("Running validation checks...")

    validate_rebalance_dates_are_trading_days(rebalance_calendar, trading_dates)
    validate_no_nans_in_features(rebalance_records)
    validate_weights_sum(rebalance_records)
    validate_no_nans_in_equity_curve(equity_curve)
    validate_state_transitions(rebalance_records)
    validate_price_data_coverage(price_data)

    logger.info("All validations passed")
