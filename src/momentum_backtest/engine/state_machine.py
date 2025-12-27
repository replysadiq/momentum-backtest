"""
Unified market state machine - PURE FUNCTION implementation.

States:
- RISK_ON: Actively invested in momentum portfolio
- PANIC: Emergency exit due to volatility or drawdown
- CASH: Defensive cash position due to negative benchmark trend

No hidden globals. All state transitions are deterministic based on inputs.
Uses the canonical stats module for return/volatility computations.
"""

from dataclasses import dataclass
from enum import Enum, auto
import logging

import pandas as pd

from .stats import total_return_months, realized_vol, max_drawdown


logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market regime states."""
    RISK_ON = auto()
    PANIC = auto()
    DEFENSIVE_MOMENTUM = auto()  # V2: Low-vol stocks when panic triggered
    CASH = auto()


@dataclass(frozen=True)
class StateFeatures:
    """
    Input features for state machine transitions.

    All values are computed as of the rebalance date, using only
    data available up to (but not including) that date.
    """
    benchmark_return_6m: float  # 6-month benchmark return
    benchmark_return_3m: float  # 3-month benchmark return
    benchmark_vol_1m: float     # 1-month realized volatility
    benchmark_vol_6m: float     # 6-month realized volatility
    portfolio_drawdown_3m: float  # Rolling 3-month portfolio drawdown

    @property
    def vol_ratio(self) -> float:
        """Ratio of 1-month to 6-month volatility."""
        if self.benchmark_vol_6m <= 0:
            logger.debug(f"benchmark_vol_6m={self.benchmark_vol_6m} <= 0, returning vol_ratio=0")
            return 0.0
        return self.benchmark_vol_1m / self.benchmark_vol_6m


@dataclass(frozen=True)
class StateThresholds:
    """
    Thresholds for state machine transitions.

    These are configurable parameters that define when transitions occur.
    """
    # RISK_ON -> PANIC thresholds
    panic_dd_threshold: float = 0.15     # Portfolio 3M DD > 15%
    panic_vol_ratio: float = 2.0         # Vol(1M) > 2.0 * Vol(6M)

    # PANIC -> CASH thresholds
    panic_exit_vol_ratio: float = 1.5    # Vol(1M) < 1.5 * Vol(6M)
    panic_exit_bench_ret: float = -0.05  # Bench 3M return > -5%

    # V2 Lever A: Defensive Momentum
    panic_defensive_mode: bool = False   # When True, PANIC -> DEFENSIVE_MOMENTUM


def next_state(
    prev_state: MarketState,
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """
    Compute the next market state based on current state and features.

    This is a PURE FUNCTION with no side effects or hidden state.

    Transition rules:

    From RISK_ON:
        -> PANIC if:
            - Portfolio 3M drawdown > panic_dd_threshold (15%) OR
            - Vol(1M) > panic_vol_ratio * Vol(6M) (2.0x)
        -> CASH if:
            - Benchmark 6M return <= 0

    From PANIC:
        -> CASH only if:
            - Vol(1M) < panic_exit_vol_ratio * Vol(6M) (1.5x) AND
            - Benchmark 3M return > panic_exit_bench_ret (-5%)

    From CASH:
        -> RISK_ON only if:
            - Benchmark 6M return > 0

    Args:
        prev_state: Previous market state
        features: Current state features (computed at rebalance date)
        thresholds: Transition thresholds

    Returns:
        Next market state
    """
    if prev_state == MarketState.RISK_ON:
        new_state = _transition_from_risk_on(features, thresholds)
        # V2 Lever A: Intercept PANIC -> DEFENSIVE_MOMENTUM when mode enabled
        if new_state == MarketState.PANIC and thresholds.panic_defensive_mode:
            logger.info("PANIC intercepted -> DEFENSIVE_MOMENTUM (panic_defensive_mode=True)")
            return MarketState.DEFENSIVE_MOMENTUM
        return new_state

    elif prev_state == MarketState.PANIC:
        return _transition_from_panic(features, thresholds)

    elif prev_state == MarketState.DEFENSIVE_MOMENTUM:
        return _transition_from_defensive_momentum(features, thresholds)

    elif prev_state == MarketState.CASH:
        return _transition_from_cash(features, thresholds)

    else:
        raise ValueError(f"Unknown state: {prev_state}")


def _transition_from_risk_on(
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """Handle transitions from RISK_ON state."""

    # Check PANIC conditions (checked first - emergency exit)
    panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
    panic_vol = features.vol_ratio > thresholds.panic_vol_ratio

    if panic_dd or panic_vol:
        reason = []
        if panic_dd:
            reason.append(f"DD={features.portfolio_drawdown_3m:.1%}>{thresholds.panic_dd_threshold:.0%}")
        if panic_vol:
            reason.append(f"VolRatio={features.vol_ratio:.2f}>{thresholds.panic_vol_ratio:.1f}")
        logger.info(f"RISK_ON -> PANIC: {', '.join(reason)}")
        return MarketState.PANIC

    # Check CASH condition
    if features.benchmark_return_6m <= 0:
        logger.info(f"RISK_ON -> CASH: Bench6M={features.benchmark_return_6m:.1%}<=0")
        return MarketState.CASH

    # Stay in RISK_ON
    return MarketState.RISK_ON


def _transition_from_panic(
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """Handle transitions from PANIC state."""

    # Can only exit PANIC to CASH (not directly to RISK_ON)
    vol_ok = features.vol_ratio < thresholds.panic_exit_vol_ratio
    ret_ok = features.benchmark_return_3m > thresholds.panic_exit_bench_ret

    if vol_ok and ret_ok:
        logger.info(
            f"PANIC -> CASH: VolRatio={features.vol_ratio:.2f}<{thresholds.panic_exit_vol_ratio:.1f}, "
            f"Bench3M={features.benchmark_return_3m:.1%}>{thresholds.panic_exit_bench_ret:.0%}"
        )
        return MarketState.CASH

    # Stay in PANIC
    return MarketState.PANIC


def _transition_from_defensive_momentum(
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """Handle transitions from DEFENSIVE_MOMENTUM state (V2 Lever A)."""

    # Same exit conditions as PANIC -> transition to CASH when vol normalizes
    vol_ok = features.vol_ratio < thresholds.panic_exit_vol_ratio
    ret_ok = features.benchmark_return_3m > thresholds.panic_exit_bench_ret

    if vol_ok and ret_ok:
        logger.info(
            f"DEFENSIVE_MOMENTUM -> CASH: VolRatio={features.vol_ratio:.2f}<{thresholds.panic_exit_vol_ratio:.1f}, "
            f"Bench3M={features.benchmark_return_3m:.1%}>{thresholds.panic_exit_bench_ret:.0%}"
        )
        return MarketState.CASH

    # Stay in DEFENSIVE_MOMENTUM
    return MarketState.DEFENSIVE_MOMENTUM


def _transition_from_cash(
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """Handle transitions from CASH state."""

    # Can only exit CASH to RISK_ON
    if features.benchmark_return_6m > 0:
        logger.info(f"CASH -> RISK_ON: Bench6M={features.benchmark_return_6m:.1%}>0")
        return MarketState.RISK_ON

    # Stay in CASH
    return MarketState.CASH


def compute_state_features(
    benchmark_prices: pd.Series,
    portfolio_equity: pd.Series,
    rebalance_date: pd.Timestamp,
) -> StateFeatures:
    """
    Compute state features for a given rebalance date.

    All computations use only data strictly before the rebalance date
    to avoid lookahead bias. Uses the canonical stats module.

    Args:
        benchmark_prices: Benchmark price series
        portfolio_equity: Portfolio equity curve
        rebalance_date: The rebalance date (exclusive boundary)

    Returns:
        StateFeatures for the rebalance date
    """
    # Use canonical stats module with rebalance_date as exclusive end
    # Benchmark returns
    bench_6m = total_return_months(benchmark_prices, rebalance_date, 6)
    bench_3m = total_return_months(benchmark_prices, rebalance_date, 3)

    # Benchmark volatility
    vol_1m = realized_vol(benchmark_prices, rebalance_date, 1, min_samples=5)
    vol_6m = realized_vol(benchmark_prices, rebalance_date, 6, min_samples=10)

    # Portfolio drawdown
    port_dd_3m = max_drawdown(portfolio_equity, rebalance_date, 3, min_samples=2)

    # Handle None returns with sensible defaults
    return StateFeatures(
        benchmark_return_6m=bench_6m if bench_6m is not None else 0.0,
        benchmark_return_3m=bench_3m if bench_3m is not None else 0.0,
        benchmark_vol_1m=vol_1m if vol_1m is not None else 0.0,
        benchmark_vol_6m=vol_6m if vol_6m is not None else 0.0,
        portfolio_drawdown_3m=port_dd_3m if port_dd_3m is not None else 0.0,
    )


def determine_initial_state(
    features: StateFeatures,
    thresholds: StateThresholds,
) -> MarketState:
    """
    Determine the initial state at the first rebalance date.

    Rules (evaluated in order):
    1. If panic condition true -> PANIC
    2. Else if benchmark 6M return <= 0 -> CASH
    3. Else -> RISK_ON

    Args:
        features: State features at first rebalance date
        thresholds: Transition thresholds

    Returns:
        Initial market state
    """
    # Check panic conditions
    panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
    panic_vol = features.vol_ratio > thresholds.panic_vol_ratio

    if panic_dd or panic_vol:
        # V2 Lever A: Use DEFENSIVE_MOMENTUM instead of PANIC when mode enabled
        if thresholds.panic_defensive_mode:
            logger.info(f"Initial state: DEFENSIVE_MOMENTUM (DD={features.portfolio_drawdown_3m:.1%}, "
                        f"VolRatio={features.vol_ratio:.2f})")
            return MarketState.DEFENSIVE_MOMENTUM
        logger.info(f"Initial state: PANIC (DD={features.portfolio_drawdown_3m:.1%}, "
                    f"VolRatio={features.vol_ratio:.2f})")
        return MarketState.PANIC

    # Check cash condition
    if features.benchmark_return_6m <= 0:
        logger.info(f"Initial state: CASH (Bench6M={features.benchmark_return_6m:.1%})")
        return MarketState.CASH

    # Default to RISK_ON
    logger.info(f"Initial state: RISK_ON (Bench6M={features.benchmark_return_6m:.1%})")
    return MarketState.RISK_ON
