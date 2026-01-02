"""
Unified market state machine - PURE FUNCTION implementation.

States:
- RISK_ON: Actively invested in momentum portfolio
- PANIC: Emergency exit due to volatility or drawdown
- DEFENSIVE_MOMENTUM: Low-vol stocks during panic (V2 Lever A)
- CASH_INVESTED: Cash regime, but invested in defensive replacement basket
- CASH_TRUE: Cash regime, truly in cash (no equity exposure)

No hidden globals. All state transitions are deterministic based on inputs.
Uses the canonical stats module for return/volatility computations.

================================================================================
CASH STATE TRANSITION RULES (V3 Documentation)
================================================================================

ENTRY CONDITIONS TO CASH:
-------------------------
1. From RISK_ON (direct entry):
   - baseline mode: benchmark_6m_return <= 0
   - strict_dual mode: benchmark_6m_return <= 0 AND benchmark_3m_return <= 0
   - strict_persist mode: baseline condition true for 2 consecutive rebalances

2. From PANIC or DEFENSIVE_MOMENTUM (post-crisis normalization):
   - vol_ratio < panic_exit_vol_ratio (default 1.5) AND
   - benchmark_3m_return > panic_exit_bench_ret (default -5%)
   - Note: This is NOT affected by cash_entry_mode (crisis exit path)

EXIT CONDITIONS FROM CASH:
--------------------------
- To RISK_ON: benchmark_6m_return > 0

PANIC_EXIT_BENCH_RET GATE:
--------------------------
- Used only for PANIC/DEFENSIVE_MOMENTUM -> CASH transition
- Prevents exiting crisis mode into CASH during severe drawdowns
- Does NOT affect RISK_ON -> CASH transition
================================================================================
"""

from dataclasses import dataclass
from enum import Enum, auto
import logging
from typing import Optional

import pandas as pd

from .stats import total_return_months, realized_vol, max_drawdown


logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market regime states."""
    RISK_ON = auto()
    PANIC = auto()
    DEFENSIVE_MOMENTUM = auto()  # V2: Low-vol stocks when panic triggered
    CASH_INVESTED = auto()
    CASH_TRUE = auto()
    CASH = auto()  # Legacy alias (do not emit by default)


def is_cash_regime(state: MarketState) -> bool:
    """Return True if state is any cash-regime variant."""
    return state in {MarketState.CASH_INVESTED, MarketState.CASH_TRUE, MarketState.CASH}


class CashEntryMode(Enum):
    """
    V3: Cash entry mode variants for controlling when strategy goes to CASH.

    - baseline: Original behavior (benchmark_6m_return <= 0)
    - strict_dual: Require BOTH 6M AND 3M returns <= 0
    - strict_persist: Require baseline condition for 2 consecutive rebalances
    """
    BASELINE = "baseline"
    STRICT_DUAL = "strict_dual"
    STRICT_PERSIST = "strict_persist"


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

    # V3: Cash entry mode (baseline, strict_dual, strict_persist)
    cash_entry_mode: CashEntryMode = CashEntryMode.BASELINE


@dataclass
class StateContext:
    """
    Mutable context for state machine (V3).

    Tracks state that persists across rebalance cycles, such as
    the cash_entry_counter for strict_persist mode.
    """
    # Counter for strict_persist mode: increments when cash condition is true
    cash_entry_counter: int = 0

    # V3: Reason for last CASH entry (for audit/debug)
    # Values: "bench_6m<=0", "bench_6m<=0 & bench_3m<=0", "persist_2", "crisis_exit", None
    cash_entry_reason: Optional[str] = None

    def reset_cash_counter(self) -> None:
        """Reset cash entry counter (called when condition becomes false)."""
        self.cash_entry_counter = 0

    def increment_cash_counter(self) -> None:
        """Increment cash entry counter (called when condition is true)."""
        self.cash_entry_counter += 1

    def set_cash_reason(self, reason: str) -> None:
        """Set the reason for entering CASH state."""
        self.cash_entry_reason = reason
        logger.info(f"CASH entry reason: {reason}")


def next_state(
    prev_state: MarketState,
    features: StateFeatures,
    thresholds: StateThresholds,
    context: Optional[StateContext] = None,
) -> MarketState:
    """
    Compute the next market state based on current state and features.

    This function is mostly pure, except for strict_persist mode which
    uses the context to track consecutive cash conditions.

    Transition rules:

    From RISK_ON:
        -> PANIC if:
            - Portfolio 3M drawdown > panic_dd_threshold (15%) OR
            - Vol(1M) > panic_vol_ratio * Vol(6M) (2.0x)
        -> CASH if (depends on cash_entry_mode):
            - baseline: Benchmark 6M return <= 0
            - strict_dual: Benchmark 6M return <= 0 AND 3M return <= 0
            - strict_persist: baseline condition for 2 consecutive rebalances

    From PANIC:
        -> CASH only if:
            - Vol(1M) < panic_exit_vol_ratio * Vol(6M) (1.5x) AND
            - Benchmark 3M return > panic_exit_bench_ret (-5%)

    From CASH_*:
        -> RISK_ON only if:
            - Benchmark 6M return > 0

    Args:
        prev_state: Previous market state
        features: Current state features (computed at rebalance date)
        thresholds: Transition thresholds
        context: Mutable state context (for strict_persist mode)

    Returns:
        Next market state
    """
    # Create default context if not provided
    if context is None:
        context = StateContext()

    if prev_state == MarketState.RISK_ON:
        new_state = _transition_from_risk_on(features, thresholds, context)
        # V2 Lever A: Intercept PANIC -> DEFENSIVE_MOMENTUM when mode enabled
        if new_state == MarketState.PANIC and thresholds.panic_defensive_mode:
            logger.info("PANIC intercepted -> DEFENSIVE_MOMENTUM (panic_defensive_mode=True)")
            return MarketState.DEFENSIVE_MOMENTUM
        return new_state

    elif prev_state == MarketState.PANIC:
        return _transition_from_panic(features, thresholds, context)

    elif prev_state == MarketState.DEFENSIVE_MOMENTUM:
        return _transition_from_defensive_momentum(features, thresholds, context)

    elif is_cash_regime(prev_state):
        # Reset cash counter when we're in CASH (we've already entered)
        context.reset_cash_counter()
        return _transition_from_cash(features, thresholds)

    else:
        raise ValueError(f"Unknown state: {prev_state}")


def _transition_from_risk_on(
    features: StateFeatures,
    thresholds: StateThresholds,
    context: StateContext,
) -> MarketState:
    """Handle transitions from RISK_ON state."""

    # Check PANIC conditions (checked first - emergency exit)
    # NOTE: PANIC triggers are NOT affected by cash_entry_mode
    panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
    panic_vol = features.vol_ratio > thresholds.panic_vol_ratio

    if panic_dd or panic_vol:
        reason = []
        if panic_dd:
            reason.append(f"DD={features.portfolio_drawdown_3m:.1%}>{thresholds.panic_dd_threshold:.0%}")
        if panic_vol:
            reason.append(f"VolRatio={features.vol_ratio:.2f}>{thresholds.panic_vol_ratio:.1f}")
        logger.info(f"RISK_ON -> PANIC: {', '.join(reason)}")
        # Reset cash counter on PANIC (different exit path)
        context.reset_cash_counter()
        return MarketState.PANIC

    # Check CASH condition based on cash_entry_mode (V3)
    mode = thresholds.cash_entry_mode

    # Baseline condition: benchmark 6M return <= 0
    baseline_cash_condition = features.benchmark_return_6m <= 0

    if mode == CashEntryMode.BASELINE:
        # Original behavior
        if baseline_cash_condition:
            context.set_cash_reason("bench_6m<=0")
            logger.info(f"RISK_ON -> CASH [baseline]: Bench6M={features.benchmark_return_6m:.1%}<=0")
            return MarketState.CASH_TRUE

    elif mode == CashEntryMode.STRICT_DUAL:
        # Require BOTH 6M AND 3M returns <= 0
        dual_condition = baseline_cash_condition and features.benchmark_return_3m <= 0
        if dual_condition:
            context.set_cash_reason("bench_6m<=0 & bench_3m<=0")
            logger.info(
                f"RISK_ON -> CASH [strict_dual]: Bench6M={features.benchmark_return_6m:.1%}<=0, "
                f"Bench3M={features.benchmark_return_3m:.1%}<=0"
            )
            return MarketState.CASH_TRUE
        elif baseline_cash_condition:
            # Log when baseline would have triggered but strict_dual blocked it
            logger.debug(
                f"RISK_ON: strict_dual blocked CASH entry "
                f"(Bench6M={features.benchmark_return_6m:.1%}<=0 but "
                f"Bench3M={features.benchmark_return_3m:.1%}>0)"
            )

    elif mode == CashEntryMode.STRICT_PERSIST:
        # Require baseline condition for 2 consecutive rebalances
        if baseline_cash_condition:
            context.increment_cash_counter()
            if context.cash_entry_counter >= 2:
                context.set_cash_reason("persist_2")
                logger.info(
                    f"RISK_ON -> CASH [strict_persist]: Bench6M={features.benchmark_return_6m:.1%}<=0 "
                    f"for {context.cash_entry_counter} consecutive periods"
                )
                return MarketState.CASH_TRUE
            else:
                logger.debug(
                    f"RISK_ON: strict_persist pending "
                    f"(counter={context.cash_entry_counter}/2, "
                    f"Bench6M={features.benchmark_return_6m:.1%}<=0)"
                )
        else:
            # Condition not met, reset counter
            context.reset_cash_counter()

    # Stay in RISK_ON
    return MarketState.RISK_ON


def _transition_from_panic(
    features: StateFeatures,
    thresholds: StateThresholds,
    context: StateContext,
) -> MarketState:
    """Handle transitions from PANIC state."""

    # Can only exit PANIC to CASH (not directly to RISK_ON)
    vol_ok = features.vol_ratio < thresholds.panic_exit_vol_ratio
    ret_ok = features.benchmark_return_3m > thresholds.panic_exit_bench_ret

    if vol_ok and ret_ok:
        context.set_cash_reason("crisis_exit")
        logger.info(
            f"PANIC -> CASH: VolRatio={features.vol_ratio:.2f}<{thresholds.panic_exit_vol_ratio:.1f}, "
            f"Bench3M={features.benchmark_return_3m:.1%}>{thresholds.panic_exit_bench_ret:.0%}"
        )
        return MarketState.CASH_TRUE

    # Stay in PANIC
    return MarketState.PANIC


def _transition_from_defensive_momentum(
    features: StateFeatures,
    thresholds: StateThresholds,
    context: StateContext,
) -> MarketState:
    """Handle transitions from DEFENSIVE_MOMENTUM state (V2 Lever A)."""

    # Same exit conditions as PANIC -> transition to CASH when vol normalizes
    vol_ok = features.vol_ratio < thresholds.panic_exit_vol_ratio
    ret_ok = features.benchmark_return_3m > thresholds.panic_exit_bench_ret

    if vol_ok and ret_ok:
        context.set_cash_reason("crisis_exit")
        logger.info(
            f"DEFENSIVE_MOMENTUM -> CASH: VolRatio={features.vol_ratio:.2f}<{thresholds.panic_exit_vol_ratio:.1f}, "
            f"Bench3M={features.benchmark_return_3m:.1%}>{thresholds.panic_exit_bench_ret:.0%}"
        )
        return MarketState.CASH_TRUE

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
    return MarketState.CASH_TRUE


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
    context: Optional[StateContext] = None,
) -> MarketState:
    """
    Determine the initial state at the first rebalance date.

    Rules (evaluated in order):
    1. If panic condition true -> PANIC (or DEFENSIVE_MOMENTUM if mode enabled)
    2. Else if cash condition true (depends on cash_entry_mode) -> CASH
    3. Else -> RISK_ON

    V3 cash_entry_mode affects CASH entry:
    - baseline: benchmark 6M return <= 0
    - strict_dual: benchmark 6M return <= 0 AND 3M return <= 0
    - strict_persist: baseline condition for 2 consecutive checks (can't enter on first check)

    Args:
        features: State features at first rebalance date
        thresholds: Transition thresholds
        context: Mutable state context (for strict_persist mode)

    Returns:
        Initial market state
    """
    if context is None:
        context = StateContext()

    # Check panic conditions (not affected by cash_entry_mode)
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

    # Check cash condition based on cash_entry_mode (V3)
    mode = thresholds.cash_entry_mode
    baseline_cash_condition = features.benchmark_return_6m <= 0

    if mode == CashEntryMode.BASELINE:
        if baseline_cash_condition:
            context.set_cash_reason("bench_6m<=0")
            logger.info(f"Initial state: CASH [baseline] (Bench6M={features.benchmark_return_6m:.1%})")
            return MarketState.CASH_TRUE

    elif mode == CashEntryMode.STRICT_DUAL:
        dual_condition = baseline_cash_condition and features.benchmark_return_3m <= 0
        if dual_condition:
            context.set_cash_reason("bench_6m<=0 & bench_3m<=0")
            logger.info(f"Initial state: CASH [strict_dual] (Bench6M={features.benchmark_return_6m:.1%}, "
                        f"Bench3M={features.benchmark_return_3m:.1%})")
            return MarketState.CASH_TRUE

    elif mode == CashEntryMode.STRICT_PERSIST:
        # For strict_persist, we can't enter CASH on first check (need 2 consecutive)
        # Just start the counter if condition is met
        if baseline_cash_condition:
            context.increment_cash_counter()
            logger.debug(f"Initial state: strict_persist counter started "
                        f"(counter={context.cash_entry_counter}/2)")
        # Always start in RISK_ON for strict_persist (need 2 checks to enter CASH)

    # Default to RISK_ON
    logger.info(f"Initial state: RISK_ON (Bench6M={features.benchmark_return_6m:.1%})")
    return MarketState.RISK_ON
