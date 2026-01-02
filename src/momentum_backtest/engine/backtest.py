"""
Main backtesting engine.

Orchestrates the simulation loop:
1. For each rebalance date
2. Compute state features
3. Determine state transition
4. If RISK_ON: select stocks and rebalance
5. If PANIC/CASH: liquidate to cash
6. Track equity curve, weights, states

CRITICAL: All return computations use EXCLUSIVE end boundaries to prevent lookahead bias.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CashReplaceMode
from ..data.sanitizer import (
    get_eligible_stocks_with_history_gate,
    required_history_months,
    history_gate_passes,
)
from .signals import (
    rank_stocks_by_momentum,
    EligibilityBreakdown,
    select_defensive_basket,
    SCORE_FAILURE_REASON_CODES,
)
from .portfolio import build_portfolio, build_defensive_portfolio, HoldingInfo
from .stats import compute_turnover, apply_transaction_cost
from .state_machine import (
    MarketState,
    StateFeatures,
    StateThresholds,
    StateContext,
    CashEntryMode,
    next_state,
    compute_state_features,
    determine_initial_state,
    is_cash_regime,
)


logger = logging.getLogger(__name__)


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event."""
    date: pd.Timestamp
    state: MarketState
    features: StateFeatures
    tickers: List[str]
    weights: Dict[str, float]
    turnover: float
    transaction_cost: float
    requested_state: str = ""
    transition_reason: str = ""
    transition_detail: str = ""
    prev_state: str = ""
    warmup_skip: bool = False
    # V3: Reason for CASH entry (None if not entering CASH this rebalance)
    cash_entry_reason: Optional[str] = None
    # V3.1: True if actually invested (not holding cash) - used for legacy CASH naming
    invested_flag: bool = False
    # V3.4: Concentration cap diagnostics
    selected_count: int = 0
    defensive_selected_count: int = 0
    max_weight: float = 0.0
    min_required: int = 0
    invested_fraction: float = 0.0
    concentration_gate_triggered: bool = False
    concentration_gate_reason: str = ""
    concentration_forced_cash: bool = False


@dataclass
class EligibilityRecord:
    """Eligibility breakdown for a single rebalance date."""
    date: pd.Timestamp
    state: MarketState
    universe_count: int
    data_eligible_count: int
    filter_passed_count: int
    selected_count: int
    insufficient_history: int
    negative_returns: int
    low_positive_months: int
    high_drawdown: int
    score_computation_failed: int
    not_in_price_data: int


@dataclass
class ScoreFailureRecord:
    """Diagnostics for score computation failures per rebalance."""
    date: pd.Timestamp
    requested_state: str
    output_state: str
    reason_code: str
    count: int
    sample_symbols: str
    sample_note: str = ""


@dataclass
class ScoreRejectionRecord:
    """Diagnostics for score rejection reasons per rebalance."""
    date: pd.Timestamp
    attempted: int
    ok: int
    rejection_counts: Dict[str, int]
    exception_hist: Dict[str, int]
    rejection_samples: Dict[str, List[str]]
    rejection_sample_contexts: Dict[str, List[str]]
    short_window_samples: List[Dict[str, object]]


@dataclass
class EligibilityFunnelRecord:
    """Eligibility funnel diagnostics per rebalance."""
    date: pd.Timestamp
    universe_count: int
    data_eligible_count: int
    warmup_skip: bool
    insufficient_history_excluded: int
    history_eligible: int
    history_ineligible_count: int
    score_attempted_count: int
    score_success_count: int
    score_computation_failed_after_gate: int
    filter_passed_count: int
    selected_count: int
    invested_count: int
    final_state: str
    concentration_forced_cash: bool


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    rebalance_records: List[RebalanceRecord]
    eligibility_records: List[EligibilityRecord]
    score_failure_records: List[ScoreFailureRecord]
    score_rejection_records: List[ScoreRejectionRecord]
    eligibility_funnel_records: List[EligibilityFunnelRecord]
    config: BacktestConfig

    # Computed at end
    total_turnover: float = 0.0
    total_transaction_costs: float = 0.0
    time_in_risk_on: float = 0.0
    time_in_panic: float = 0.0
    time_in_defensive_momentum: float = 0.0  # V2 Lever A
    # Panic event tracking (verify PANIC conditions are firing)
    panic_events_count: int = 0  # Total times PANIC condition was triggered
    panic_vol_ratio_triggers: int = 0  # Triggered by vol_ratio > threshold
    panic_dd_triggers: int = 0  # Triggered by portfolio DD > threshold
    panic_response_mode: str = "PANIC"  # "PANIC" or "DEFENSIVE_MOMENTUM"

    # Turnover split
    turnover_risk_on: float = 0.0
    turnover_defensive: float = 0.0
    turnover_cash_panic: float = 0.0  # Liquidation turnover

    # V3.5: Time in explicit cash states
    time_in_cash_invested: float = 0.0
    time_in_true_cash: float = 0.0

    # V3.4: Concentration diagnostics
    pct_time_concentration_gated: float = 0.0
    avg_invested_fraction_by_state: Dict[str, float] = field(default_factory=dict)
    avg_holdings_by_state: Dict[str, float] = field(default_factory=dict)
    pct_time_true_cash: float = 0.0
    avg_cash_weight_overall: float = 0.0
    avg_cash_weight_by_state: Dict[str, float] = field(default_factory=dict)

    # Daily diagnostics
    daily_state: pd.Series = field(default_factory=pd.Series)
    daily_invested_weight: pd.Series = field(default_factory=pd.Series)
    daily_cash_weight: pd.Series = field(default_factory=pd.Series)

    # Daily cash exposure metrics
    avg_cash_weight_overall_daily: float = 0.0
    avg_cash_weight_by_state_daily: Dict[str, float] = field(default_factory=dict)
    avg_invested_weight_by_state_daily: Dict[str, float] = field(default_factory=dict)
    pct_time_cash_regime_daily: float = 0.0
    pct_time_cash_true_daily: float = 0.0
    pct_time_cash_invested_daily: float = 0.0


# =============================================================================
# Pure helper functions for testability (Fix 5)
# =============================================================================


def compute_segment_return(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    start_date: pd.Timestamp,
    end_date_exclusive: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    cash_rate_annual: float = 0.0,
) -> float:
    """
    Compute portfolio return for a holding segment with EXCLUSIVE end boundary.

    The holding period is [start_date, end_date_exclusive).
    We use the price on start_date and the LAST price BEFORE end_date_exclusive.

    Args:
        price_data: Stock price DataFrame
        weights: Portfolio weights
        start_date: Inclusive start of period
        end_date_exclusive: Exclusive end of period (no lookahead)
        trading_dates: Trading calendar

    Returns:
        Gross portfolio return as decimal
    """
    if not weights:
        # Cash return
        if cash_rate_annual > 0:
            days_in_period = (end_date_exclusive - start_date).days
            return cash_rate_annual * (days_in_period / 365.0)
        return 0.0  # Cash return = 0

    # Get trading dates in [start, end) - EXCLUSIVE end
    period_dates = trading_dates[
        (trading_dates >= start_date) & (trading_dates < end_date_exclusive)
    ]

    if len(period_dates) < 2:
        return 0.0

    # Start price: first trading day >= start_date
    # End price: last trading day < end_date_exclusive
    start_date_actual = period_dates[0]
    end_date_actual = period_dates[-1]

    try:
        start_prices = price_data.loc[start_date_actual]
        end_prices = price_data.loc[end_date_actual]
    except KeyError:
        logger.warning(f"Missing price data for period {start_date} to {end_date_exclusive}")
        return 0.0

    # Compute weighted return
    portfolio_return = 0.0
    for ticker, weight in weights.items():
        if ticker not in price_data.columns:
            continue

        start_p = start_prices.get(ticker)
        end_p = end_prices.get(ticker)

        if pd.notna(start_p) and pd.notna(end_p) and start_p > 0:
            stock_return = (end_p / start_p) - 1.0
            portfolio_return += weight * stock_return
        else:
            # Missing data - assume flat (conservative)
            logger.debug(f"Missing price data for {ticker} in period")

    invested_fraction = sum(weights.values())
    if cash_rate_annual > 0 and invested_fraction < 1.0:
        days_in_period = (end_date_exclusive - start_date).days
        cash_return = cash_rate_annual * (days_in_period / 365.0)
        portfolio_return += (1.0 - invested_fraction) * cash_return

    return portfolio_return


def _diagnose_transition(
    prev_state: Optional[MarketState],
    next_state: MarketState,
    features: StateFeatures,
    thresholds: StateThresholds,
    prev_cash_counter: int,
) -> tuple[str, str]:
    """
    Diagnostics-only transition reason for state_log.csv.
    """
    if prev_state is None:
        return "initial_state", ""

    if prev_state == next_state:
        if prev_state == MarketState.CASH_TRUE:
            return "stay_cash", f"Bench6M={features.benchmark_return_6m:.2%}"
        if prev_state == MarketState.PANIC:
            return "stay_panic", f"VolRatio={features.vol_ratio:.2f} Bench3M={features.benchmark_return_3m:.2%}"
        if prev_state == MarketState.DEFENSIVE_MOMENTUM:
            return "stay_defensive", f"VolRatio={features.vol_ratio:.2f} Bench3M={features.benchmark_return_3m:.2%}"
        return "stay", ""

    if prev_state == MarketState.RISK_ON and next_state == MarketState.PANIC:
        panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
        panic_vol = features.vol_ratio > thresholds.panic_vol_ratio
        reasons = []
        if panic_dd:
            reasons.append("panic_dd")
        if panic_vol:
            reasons.append("panic_vol")
        return "panic_trigger", ",".join(reasons)

    if prev_state == MarketState.RISK_ON and next_state == MarketState.DEFENSIVE_MOMENTUM:
        panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
        panic_vol = features.vol_ratio > thresholds.panic_vol_ratio
        reasons = []
        if panic_dd:
            reasons.append("panic_dd")
        if panic_vol:
            reasons.append("panic_vol")
        return "panic_intercept_defensive", ",".join(reasons)

    if is_cash_regime(prev_state) and next_state == MarketState.CASH_TRUE:
        return "stay_cash", f"Bench6M={features.benchmark_return_6m:.2%}"

    if prev_state == MarketState.RISK_ON and next_state == MarketState.CASH_TRUE:
        mode = thresholds.cash_entry_mode
        if mode == CashEntryMode.BASELINE:
            return "bench_6m<=0", f"Bench6M={features.benchmark_return_6m:.2%}"
        if mode == CashEntryMode.STRICT_DUAL:
            return "bench_6m<=0 & bench_3m<=0", (
                f"Bench6M={features.benchmark_return_6m:.2%} "
                f"Bench3M={features.benchmark_return_3m:.2%}"
            )
        if mode == CashEntryMode.STRICT_PERSIST:
            return "persist_2", (
                f"Bench6M={features.benchmark_return_6m:.2%} "
                f"counter={prev_cash_counter + 1}"
            )

    if prev_state in (MarketState.PANIC, MarketState.DEFENSIVE_MOMENTUM) and next_state == MarketState.CASH_TRUE:
        return "crisis_exit", (
            f"VolRatio={features.vol_ratio:.2f} Bench3M={features.benchmark_return_3m:.2%}"
        )

    if is_cash_regime(prev_state) and next_state == MarketState.RISK_ON:
        return "bench_6m>0", f"Bench6M={features.benchmark_return_6m:.2%}"

    return "unknown_transition", ""


def compute_daily_return(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
    daily_cash_yield: float = 0.0,
) -> float:
    """
    Compute one-day portfolio return.

    Args:
        price_data: Stock price DataFrame
        weights: Portfolio weights
        prev_date: Previous trading date
        curr_date: Current trading date

    Returns:
        Daily return as decimal
    """
    if not weights:
        return daily_cash_yield

    day_return = 0.0
    for ticker, weight in weights.items():
        if ticker not in price_data.columns:
            continue

        try:
            prev_p = price_data.loc[prev_date, ticker]
            curr_p = price_data.loc[curr_date, ticker]

            if pd.notna(prev_p) and pd.notna(curr_p) and prev_p > 0:
                stock_return = (curr_p / prev_p) - 1.0
                day_return += weight * stock_return
        except KeyError:
            continue

    invested_fraction = sum(weights.values())
    if invested_fraction < 1.0 and daily_cash_yield > 0:
        day_return += (1.0 - invested_fraction) * daily_cash_yield

    return day_return


def update_equity(prev_equity: float, net_return: float) -> float:
    """
    Update equity with a return.

    Args:
        prev_equity: Previous equity value
        net_return: Net return (after costs) as decimal

    Returns:
        New equity value
    """
    return prev_equity * (1.0 + net_return)


# =============================================================================
# Main backtest function
# =============================================================================


def run_backtest(
    config: BacktestConfig,
    price_data: pd.DataFrame,
    benchmark_prices: pd.Series,
    trading_dates: pd.DatetimeIndex,
    rebalance_calendar: pd.DatetimeIndex,
    trading_calendar_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Run the complete backtest simulation.

    Args:
        config: Backtest configuration
        price_data: DataFrame of stock prices (columns = tickers)
        benchmark_prices: Benchmark price series
        trading_dates: Full trading calendar
        rebalance_calendar: Rebalance dates

    Returns:
        BacktestResult with equity curve, records, and statistics
    """
    logger.info(f"Starting backtest: {len(rebalance_calendar)} rebalance dates")
    if config.strategy_version in ("v2", "v3"):
        logger.info(f"Strategy {config.strategy_version} enabled with levers: {', '.join(config.enabled_levers)}")

    # Map config cash_entry_mode to state machine enum
    from ..config import CashEntryMode as ConfigCashEntryMode
    cash_mode_map = {
        ConfigCashEntryMode.BASELINE: CashEntryMode.BASELINE,
        ConfigCashEntryMode.STRICT_DUAL: CashEntryMode.STRICT_DUAL,
        ConfigCashEntryMode.STRICT_PERSIST: CashEntryMode.STRICT_PERSIST,
    }
    sm_cash_entry_mode = cash_mode_map[config.cash_entry_mode]

    # Initialize state machine thresholds (including V2 Lever A, V3 cash entry mode)
    thresholds = StateThresholds(
        panic_dd_threshold=config.panic_dd_threshold,
        panic_vol_ratio=config.panic_vol_ratio,
        panic_exit_vol_ratio=config.panic_exit_vol_ratio,
        panic_exit_bench_ret=config.panic_exit_bench_ret,
        panic_defensive_mode=config.panic_defensive_mode,  # V2 Lever A
        cash_entry_mode=sm_cash_entry_mode,  # V3
    )

    # Initialize state context for strict_persist mode (V3)
    state_context = StateContext()

    # Initialize tracking variables
    equity = 1.0  # Start with $1
    equity_curve: Dict[pd.Timestamp, float] = {}
    rebalance_records: List[RebalanceRecord] = []
    eligibility_records: List[EligibilityRecord] = []
    score_failure_records: List[ScoreFailureRecord] = []
    score_rejection_records: List[ScoreRejectionRecord] = []
    eligibility_funnel_records: List[EligibilityFunnelRecord] = []
    current_weights: Dict[str, float] = {}
    current_state: Optional[MarketState] = None

    # V2 Lever B: Track holdings for rank buffer / min hold
    current_holdings: Dict[str, HoldingInfo] = {}

    # Panic event tracking
    panic_events_count = 0
    panic_vol_ratio_triggers = 0
    panic_dd_triggers = 0

    # Turnover split tracking
    turnover_risk_on = 0.0
    turnover_defensive = 0.0
    turnover_cash_panic = 0.0

    # V3.4: Concentration gate threshold
    if config.max_weight is not None and config.max_weight > 0:
        min_required = int(np.ceil(1.0 / config.max_weight))
    else:
        min_required = 1

    def _resolve_cash_state(state: MarketState, invested_fraction: float) -> MarketState:
        """Resolve cash-regime states based on actual investment."""
        if is_cash_regime(state):
            return MarketState.CASH_INVESTED if invested_fraction > 0 else MarketState.CASH_TRUE
        return state

    # Universe count for eligibility reporting
    universe_count = len(price_data.columns)

    # Build benchmark equity curve (buy and hold)
    bench_start_date = rebalance_calendar[0]
    bench_start_price = benchmark_prices[benchmark_prices.index >= bench_start_date].iloc[0]
    benchmark_curve = benchmark_prices / bench_start_price

    required_months = required_history_months(
        config.min_history_months,
        config.momentum_lookback_months,
        config.volatility_lookback_months,
    )
    warmup_start_date: Optional[pd.Timestamp] = None
    if config.warmup_mode == "auto":
        history_series = trading_calendar_prices if trading_calendar_prices is not None else benchmark_prices
        for d in rebalance_calendar:
            ok, _ = history_gate_passes(
                history_series,
                d,
                trading_dates,
                required_months,
                config.history_gate_mode,
                config.history_leniency,
            )
            if ok:
                warmup_start_date = d
                break
        if warmup_start_date is None:
            raise ValueError("Warmup failed: insufficient trading calendar history for any rebalance date.")

    # Track portfolio equity between rebalances
    last_rebalance_date: Optional[pd.Timestamp] = None
    last_rebalance_equity = equity

    for i, rebalance_date in enumerate(rebalance_calendar):
        logger.debug(f"Processing rebalance {i+1}/{len(rebalance_calendar)}: {rebalance_date.date()}")
        prev_state_obj = current_state
        prev_state_name = current_state.name if current_state is not None else ""
        prev_cash_counter = state_context.cash_entry_counter if state_context is not None else 0
        data_eligible_count = 0
        if rebalance_date in price_data.index:
            data_eligible_count = int(price_data.loc[rebalance_date].notna().sum())

        # Step 1: Compute portfolio return since last rebalance
        # Uses EXCLUSIVE end boundary - no lookahead
        if last_rebalance_date is not None:
            if current_weights:
                # Invested: compute stock returns
                gross_return = compute_segment_return(
                    price_data,
                    current_weights,
                    last_rebalance_date,
                    rebalance_date,  # EXCLUSIVE - last price is day BEFORE this
                    trading_dates,
                    cash_rate_annual=config.cash_rate_annual,
                )
            else:
                # In cash: apply cash yield
                if config.cash_rate_annual > 0:
                    # Compute days in period and convert to return
                    days_in_period = (rebalance_date - last_rebalance_date).days
                    gross_return = config.cash_rate_annual * (days_in_period / 365.0)
                else:
                    gross_return = 0.0
            equity = update_equity(last_rebalance_equity, gross_return)

        # Record equity at rebalance date
        equity_curve[rebalance_date] = equity

        # Step 2: Build portfolio equity series for state machine
        port_equity_series = pd.Series(equity_curve)

        # Step 3: Compute state features
        features = compute_state_features(
            benchmark_prices,
            port_equity_series,
            rebalance_date,
        )

        warmup_skip = (
            config.warmup_mode == "auto"
            and warmup_start_date is not None
            and rebalance_date < warmup_start_date
        )
        if warmup_skip:
            logger.info(
                f"WARMUP_SKIP | date={rebalance_date.date()} | "
                "reason=insufficient_global_history"
            )
            current_state = MarketState.CASH_TRUE
            requested_state = MarketState.CASH_TRUE
            output_state = MarketState.CASH_TRUE
            new_weights = {}
            selected_tickers = []
            selected_count = 0
            defensive_selected_count = 0
            invested_fraction = 0.0
            concentration_gate_triggered = False
            concentration_gate_reason = ""
            invested_flag = False

            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=output_state,
                universe_count=universe_count,
                data_eligible_count=0,
                filter_passed_count=0,
                selected_count=0,
                insufficient_history=0,
                negative_returns=0,
                low_positive_months=0,
                high_drawdown=0,
                score_computation_failed=0,
                not_in_price_data=0,
            ))

            record = RebalanceRecord(
                date=rebalance_date,
                state=output_state,
                requested_state=requested_state.name,
                transition_reason="warmup_skip",
                transition_detail="insufficient_global_history",
                prev_state=prev_state_name,
                warmup_skip=True,
                features=features,
                tickers=selected_tickers,
                weights=new_weights.copy(),
                turnover=0.0,
                transaction_cost=0.0,
                cash_entry_reason=None,
                invested_flag=invested_flag,
                selected_count=selected_count,
                defensive_selected_count=defensive_selected_count,
                max_weight=float(config.max_weight) if config.max_weight is not None else 0.0,
                min_required=min_required,
                invested_fraction=invested_fraction,
                concentration_gate_triggered=concentration_gate_triggered,
                concentration_gate_reason=concentration_gate_reason,
                concentration_forced_cash=False,
            )
            rebalance_records.append(record)
            eligibility_funnel_records.append(EligibilityFunnelRecord(
                date=rebalance_date,
                universe_count=universe_count,
                data_eligible_count=data_eligible_count,
                warmup_skip=True,
                insufficient_history_excluded=0,
                history_eligible=0,
                history_ineligible_count=0,
                score_attempted_count=0,
                score_success_count=0,
                score_computation_failed_after_gate=0,
                filter_passed_count=0,
                selected_count=0,
                invested_count=0,
                final_state=output_state.name,
                concentration_forced_cash=False,
            ))

            current_state = output_state
            current_weights = new_weights
            last_rebalance_date = rebalance_date
            last_rebalance_equity = equity
            continue

        # Step 4: Check for PANIC conditions (track even if intercepted)
        panic_dd = features.portfolio_drawdown_3m > thresholds.panic_dd_threshold
        panic_vol = features.vol_ratio > thresholds.panic_vol_ratio

        if panic_dd or panic_vol:
            panic_events_count += 1
            if panic_vol:
                panic_vol_ratio_triggers += 1
                logger.debug(f"  PANIC trigger: vol_ratio={features.vol_ratio:.2f} > {thresholds.panic_vol_ratio}")
            if panic_dd:
                panic_dd_triggers += 1
                logger.debug(f"  PANIC trigger: DD={features.portfolio_drawdown_3m:.1%} > {thresholds.panic_dd_threshold:.0%}")

        # Step 5: Determine state transition
        if current_state is None:
            # First rebalance - determine initial state
            current_state = determine_initial_state(features, thresholds, state_context)
        else:
            current_state = next_state(current_state, features, thresholds, state_context)

        transition_reason, transition_detail = _diagnose_transition(
            prev_state=prev_state_obj,
            next_state=current_state,
            features=features,
            thresholds=thresholds,
            prev_cash_counter=prev_cash_counter,
        )
        # Logged after output_state is resolved (to capture CASH_INVESTED/CASH_TRUE split).

        # Step 6: Execute based on state
        new_weights: Dict[str, float] = {}
        selected_tickers: List[str] = []
        turnover = 0.0
        transaction_cost = 0.0
        invested_flag = False  # V3.1: True if actually holding equities
        selected_count = 0
        defensive_selected_count = 0
        invested_fraction = 0.0
        concentration_gate_triggered = False
        concentration_gate_reason = ""
        requested_state = current_state
        score_failure_payload = None
        insufficient_history_excluded = 0
        eligible_after_history = 0
        score_computation_failed_after_gate = 0
        filter_passed_count = 0
        score_attempted_count = 0
        score_success_count = 0

        # Convert current_weights to Series for turnover calculation
        prev_weights_series = pd.Series(current_weights) if current_weights else pd.Series(dtype=float)

        if current_state == MarketState.RISK_ON:
            invested_flag = True  # RISK_ON = invested
            # Get eligible stocks
            eligible, insufficient_history_excluded, eligible_after_history = (
                get_eligible_stocks_with_history_gate(
                    price_data,
                    rebalance_date,
                    trading_dates,
                    min_history_months=config.min_history_months,
                    momentum_lookback_months=config.momentum_lookback_months,
                    volatility_lookback_months=config.volatility_lookback_months,
                    history_gate_mode=config.history_gate_mode,
                    history_leniency=config.history_leniency,
                )
            )
            score_attempted_count = len(eligible)

            # Rank by momentum (returns RankingResult with scores and eligibility)
            # V2 Lever D: Pass disable_6m_filter
            ranking_result = rank_stocks_by_momentum(
                price_data,
                eligible,
                rebalance_date,
                trading_dates=trading_dates,
                universe_count=universe_count,
                use_dd_filter=config.use_dd_filter,
                dd_threshold=config.dd_threshold,
                nse_style=config.nse_style_scoring,
                disable_6m_filter=config.disable_6m_filter,  # V2 Lever D
            )

            score_failure_payload = {
                "breakdown": ranking_result.score_failure_breakdown,
                "samples": ranking_result.score_failure_samples,
                "notes": ranking_result.score_failure_notes,
            }
            score_success_count = len(ranking_result.scores)
            score_rejection_records.append(ScoreRejectionRecord(
                date=rebalance_date,
                attempted=ranking_result.rejection_attempted,
                ok=ranking_result.rejection_ok,
                rejection_counts=ranking_result.rejection_counts,
                exception_hist=ranking_result.rejection_exception_hist,
                rejection_samples=ranking_result.rejection_samples,
                rejection_sample_contexts=ranking_result.rejection_sample_contexts,
                short_window_samples=ranking_result.short_window_samples,
            ))

            # Build portfolio from ranked scores
            # V2 Lever B: Pass rank buffer / min hold params
            allocation, current_holdings = build_portfolio(
                ranking_result.scores,
                top_n=config.top_n_stocks,
                max_weight=config.max_weight,
                previous_weights=current_weights,
                rebalance_date=rebalance_date,
                current_holdings=current_holdings,  # V2 Lever B
                rank_buffer=config.rank_buffer,  # V2 Lever B
                min_hold_months=config.min_hold_months,  # V2 Lever B
            )

            logger.info(
                f"[{rebalance_date.date()}] Eligible={len(eligible)} "
                f"Selected={len(allocation.tickers)} "
                f"Top={allocation.tickers[:min(5, len(allocation.tickers))]}"
            )

            # Record eligibility data
            elig = ranking_result.eligibility
            score_computation_failed_after_gate = elig.score_computation_failed
            filter_passed_count = elig.filter_passed_count
            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=requested_state,
                universe_count=elig.universe_count,
                data_eligible_count=elig.data_eligible_count,
                filter_passed_count=elig.filter_passed_count,
                selected_count=len(allocation.tickers),
                insufficient_history=elig.insufficient_history,
                negative_returns=elig.negative_returns,
                low_positive_months=elig.low_positive_months,
                high_drawdown=elig.high_drawdown,
                score_computation_failed=elig.score_computation_failed,
                not_in_price_data=elig.not_in_price_data,
            ))

            new_weights = allocation.weights
            selected_tickers = allocation.tickers
            selected_count = len(allocation.tickers)

            if (
                filter_passed_count > 0
                and selected_count > 0
                and selected_count < min_required
            ):
                concentration_gate_triggered = True
                concentration_gate_reason = "risk_on_insufficient_concentration"
                invested_flag = False
                current_state = MarketState.CASH_TRUE
                new_weights = {}
                selected_tickers = []
                current_holdings = {}

            # Compute turnover using canonical function (Fix 2)
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_risk_on += turnover  # Track RISK_ON turnover

            # Apply transaction costs
            if turnover > 0:
                transaction_cost = apply_transaction_cost(0.0, turnover, config.tc_bps)
                equity = update_equity(equity, -abs(transaction_cost))
                equity_curve[rebalance_date] = equity

        elif current_state == MarketState.DEFENSIVE_MOMENTUM:
            invested_flag = True  # DEFENSIVE_MOMENTUM = invested
            # V2 Lever A: Hold low-volatility stocks during panic instead of cash
            eligible, insufficient_history_excluded, eligible_after_history = (
                get_eligible_stocks_with_history_gate(
                    price_data,
                    rebalance_date,
                    trading_dates,
                    min_history_months=config.min_history_months,
                    momentum_lookback_months=config.momentum_lookback_months,
                    volatility_lookback_months=config.volatility_lookback_months,
                    history_gate_mode=config.history_gate_mode,
                    history_leniency=config.history_leniency,
                )
            )

            # Select defensive basket (lowest volatility stocks with positive momentum)
            defensive_scores = select_defensive_basket(
                price_data,
                eligible,
                rebalance_date,
                basket_size=config.defensive_basket_size,
            )
            filter_passed_count = len(defensive_scores)

            # Build defensive portfolio using inverse-vol weighting
            allocation = build_defensive_portfolio(
                defensive_scores,
                max_weight=config.max_weight,
                previous_weights=current_weights,
                rebalance_date=rebalance_date,
            )

            # Clear holdings on state change (bypass rank buffer)
            current_holdings = {}

            new_weights = allocation.weights
            selected_tickers = allocation.tickers
            defensive_selected_count = len(allocation.tickers)

            if (
                filter_passed_count > 0
                and defensive_selected_count > 0
                and defensive_selected_count < min_required
            ):
                concentration_gate_triggered = True
                concentration_gate_reason = "defensive_insufficient_concentration"
                invested_flag = False
                new_weights = {}
                selected_tickers = []

            # Compute turnover
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_defensive += turnover  # Track DEFENSIVE_MOMENTUM turnover

            # Apply transaction costs
            if turnover > 0:
                transaction_cost = apply_transaction_cost(0.0, turnover, config.tc_bps)
                equity = update_equity(equity, -abs(transaction_cost))
                equity_curve[rebalance_date] = equity

            # Record eligibility (defensive basket selection)
            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=current_state,
                universe_count=universe_count,
                data_eligible_count=len(eligible),
                filter_passed_count=len(defensive_scores),
                selected_count=len(allocation.tickers),
                insufficient_history=0,
                negative_returns=0,
                low_positive_months=0,
                high_drawdown=0,
                score_computation_failed=0,
                not_in_price_data=0,
            ))

        elif is_cash_regime(current_state) and config.cash_replace_mode == CashReplaceMode.DEFENSIVE:
            # V3.1: CASH state with defensive replacement - hold defensive portfolio
            invested_flag = True

            eligible, insufficient_history_excluded, eligible_after_history = (
                get_eligible_stocks_with_history_gate(
                    price_data,
                    rebalance_date,
                    trading_dates,
                    min_history_months=config.min_history_months,
                    momentum_lookback_months=config.momentum_lookback_months,
                    volatility_lookback_months=config.volatility_lookback_months,
                    history_gate_mode=config.history_gate_mode,
                    history_leniency=config.history_leniency,
                )
            )
            score_attempted_count = 0
            score_success_count = 0

            # Select defensive basket (same as DEFENSIVE_MOMENTUM state)
            defensive_scores = select_defensive_basket(
                price_data,
                eligible,
                rebalance_date,
                basket_size=config.defensive_basket_size,
            )
            filter_passed_count = len(defensive_scores)

            # Build defensive portfolio
            allocation = build_defensive_portfolio(
                defensive_scores,
                max_weight=config.max_weight,
            )

            new_weights = allocation.weights
            selected_tickers = allocation.tickers
            defensive_selected_count = len(allocation.tickers)

            if (
                filter_passed_count > 0
                and defensive_selected_count > 0
                and defensive_selected_count < min_required
            ):
                concentration_gate_triggered = True
                concentration_gate_reason = "defensive_insufficient_concentration"
                invested_flag = False
                new_weights = {}
                selected_tickers = []

            # Compute turnover
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_defensive += turnover  # Track as defensive turnover

            # Apply transaction costs
            if turnover > 0:
                transaction_cost = apply_transaction_cost(0.0, turnover, config.tc_bps)
                equity = update_equity(equity, -abs(transaction_cost))
                equity_curve[rebalance_date] = equity

            # Record eligibility
            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=current_state,
                universe_count=universe_count,
                data_eligible_count=len(eligible),
                filter_passed_count=len(defensive_scores),
                selected_count=len(allocation.tickers),
                insufficient_history=0,
                negative_returns=0,
                low_positive_months=0,
                high_drawdown=0,
                score_computation_failed=0,
                not_in_price_data=0,
            ))

        else:
            # PANIC or CASH_TRUE (with no replacement) - liquidate to cash
            # Target weights = empty (all cash)
            target_weights_series = pd.Series(dtype=float)

            # Compute turnover: full liquidation = sum of all current weights (Fix 2)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_cash_panic += turnover  # Track CASH/PANIC turnover (liquidation)

            if turnover > 0:
                # Apply transaction costs for liquidation
                transaction_cost = apply_transaction_cost(0.0, turnover, config.tc_bps)
                equity = update_equity(equity, -abs(transaction_cost))
                equity_curve[rebalance_date] = equity

            # Clear holdings on state change (bypass rank buffer)
            current_holdings = {}

            new_weights = {}
            selected_tickers = []

            # Record eligibility (no stock selection in PANIC/CASH)
            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=requested_state,
                universe_count=universe_count,
                data_eligible_count=0,
                filter_passed_count=0,
                selected_count=0,
                insufficient_history=0,
                negative_returns=0,
                low_positive_months=0,
                high_drawdown=0,
                score_computation_failed=0,
                not_in_price_data=0,
            ))

        invested_fraction = sum(new_weights.values()) if new_weights else 0.0
        cap_epsilon = 1e-9
        if (
            invested_fraction < 1.0 - cap_epsilon
            and not concentration_gate_triggered
            and invested_fraction > cap_epsilon
        ):
            concentration_gate_triggered = True
            concentration_gate_reason = "cap_left_cash"

        resolved_state = _resolve_cash_state(current_state, invested_fraction)
        if config.legacy_cash_state_names and is_cash_regime(resolved_state):
            output_state = MarketState.CASH
        else:
            output_state = resolved_state

        if prev_state_obj is not None and is_cash_regime(prev_state_obj) and is_cash_regime(output_state):
            if prev_state_obj != output_state:
                transition_reason = "cash_state_resolution"
                transition_detail = f"invested_fraction={invested_fraction:.3f}"

        if output_state == MarketState.CASH:
            invested_flag = invested_fraction > 0
        else:
            invested_flag = output_state in {
                MarketState.RISK_ON,
                MarketState.DEFENSIVE_MOMENTUM,
                MarketState.CASH_INVESTED,
            }

        logger.info(
            "STATE_TRANSITION | date=%s | %s->%s | Bench6M=%.2f%% | Bench3M=%.2f%% | "
            "VolRatio=%.2f | reason=%s",
            rebalance_date.date(),
            prev_state_name or "NONE",
            output_state.name,
            features.benchmark_return_6m * 100.0,
            features.benchmark_return_3m * 100.0,
            features.vol_ratio,
            transition_reason,
        )

        # Step 6: Record rebalance event
        # Capture cash_entry_reason from context (only set when entering CASH)
        cash_reason = state_context.cash_entry_reason
        state_context.cash_entry_reason = None  # Clear after capturing

        record = RebalanceRecord(
            date=rebalance_date,
            state=output_state,
            requested_state=requested_state.name,
            transition_reason=transition_reason,
            transition_detail=transition_detail,
            prev_state=prev_state_name,
            warmup_skip=warmup_skip,
            features=features,
            tickers=selected_tickers,
            weights=new_weights.copy(),
            turnover=turnover,
            transaction_cost=abs(transaction_cost),
            cash_entry_reason=cash_reason,
            invested_flag=invested_flag,
            selected_count=selected_count,
            defensive_selected_count=defensive_selected_count,
            max_weight=float(config.max_weight) if config.max_weight is not None else 0.0,
            min_required=min_required,
            invested_fraction=invested_fraction,
            concentration_gate_triggered=concentration_gate_triggered,
            concentration_gate_reason=concentration_gate_reason,
            concentration_forced_cash=(
                concentration_gate_triggered
                and concentration_gate_reason in {
                    "risk_on_insufficient_concentration",
                    "defensive_insufficient_concentration",
                }
            ),
        )
        rebalance_records.append(record)

        if score_failure_payload is not None:
            for reason_code in SCORE_FAILURE_REASON_CODES:
                count = int(score_failure_payload["breakdown"].get(reason_code, 0))
                samples = score_failure_payload["samples"].get(reason_code, [])
                notes = score_failure_payload["notes"].get(reason_code, "")
                score_failure_records.append(ScoreFailureRecord(
                    date=rebalance_date,
                    requested_state=record.requested_state,
                    output_state=output_state.name,
                    reason_code=reason_code,
                    count=count,
                    sample_symbols=",".join(samples),
                    sample_note=notes,
                ))
            score_failure_payload = None

        if requested_state == MarketState.RISK_ON:
            selected_count_funnel = selected_count
        elif requested_state in {
            MarketState.DEFENSIVE_MOMENTUM,
            MarketState.CASH_TRUE,
            MarketState.CASH_INVESTED,
            MarketState.CASH,
        }:
            selected_count_funnel = defensive_selected_count
        else:
            selected_count_funnel = 0

        eligibility_funnel_records.append(EligibilityFunnelRecord(
            date=rebalance_date,
            universe_count=universe_count,
            data_eligible_count=data_eligible_count,
            warmup_skip=warmup_skip,
            insufficient_history_excluded=insufficient_history_excluded,
            history_eligible=eligible_after_history,
            history_ineligible_count=insufficient_history_excluded,
            score_attempted_count=score_attempted_count,
            score_success_count=score_success_count,
            score_computation_failed_after_gate=score_computation_failed_after_gate,
            filter_passed_count=filter_passed_count,
            selected_count=selected_count_funnel,
            invested_count=len(new_weights),
            final_state=output_state.name,
            concentration_forced_cash=record.concentration_forced_cash,
        ))

        # Update tracking
        current_state = output_state
        current_weights = new_weights
        last_rebalance_date = rebalance_date
        last_rebalance_equity = equity

        logger.debug(f"  State={current_state.name}, Equity={equity:.4f}, "
                     f"Stocks={len(selected_tickers)}, Turnover={turnover:.2%}")

    # Step 7: Fill in daily equity curve between rebalances (through end date)
    daily_equity = _build_daily_equity(
        equity_curve,
        price_data,
        rebalance_records,
        trading_dates,
        config.end_date,
        config.cash_rate_annual,
    )

    # Step 8: Compute summary statistics
    total_turnover = sum(r.turnover for r in rebalance_records)
    total_tc = sum(r.transaction_cost for r in rebalance_records)

    state_counts = {s: 0 for s in MarketState}
    for record in rebalance_records:
        state_counts[record.state] += 1

    n_rebalances = len(rebalance_records)

    cash_states = {MarketState.CASH_INVESTED, MarketState.CASH_TRUE, MarketState.CASH}
    cash_invested_count = sum(
        1 for r in rebalance_records
        if r.state in cash_states and r.invested_fraction > 1e-12
    )
    true_cash_count = sum(
        1 for r in rebalance_records
        if r.state in cash_states and r.invested_fraction <= 1e-12
    )

    concentration_gate_count = sum(1 for r in rebalance_records if r.concentration_gate_triggered)

    avg_invested_fraction_by_state: Dict[str, float] = {}
    avg_holdings_by_state: Dict[str, float] = {}
    avg_cash_weight_by_state: Dict[str, float] = {}
    for state in MarketState:
        state_records = [r for r in rebalance_records if r.state == state]
        if state_records:
            avg_invested_fraction_by_state[state.name] = float(
                np.mean([r.invested_fraction for r in state_records])
            )
            avg_holdings_by_state[state.name] = float(
                np.mean([len(r.weights) for r in state_records])
            )
            avg_cash_weight_by_state[state.name] = float(
                np.mean([1.0 - r.invested_fraction for r in state_records])
            )
        else:
            avg_invested_fraction_by_state[state.name] = 0.0
            avg_holdings_by_state[state.name] = 0.0
            avg_cash_weight_by_state[state.name] = 0.0

    avg_cash_weight_overall = float(np.mean([1.0 - r.invested_fraction for r in rebalance_records])) if rebalance_records else 0.0

    daily_state, daily_invested_weight, daily_cash_weight = _build_daily_diagnostics(
        rebalance_records,
        trading_dates,
        config.end_date,
    )

    cash_regime_mask = daily_state.isin(
        {MarketState.CASH_INVESTED.name, MarketState.CASH_TRUE.name, MarketState.CASH.name}
    )
    cash_invested_mask = cash_regime_mask & (daily_invested_weight > 0)
    cash_true_mask = cash_regime_mask & (daily_invested_weight <= 0)

    avg_cash_weight_overall_daily = float(daily_cash_weight.mean()) if not daily_cash_weight.empty else 0.0
    avg_cash_weight_by_state_daily = (
        daily_cash_weight.groupby(daily_state).mean().to_dict() if not daily_cash_weight.empty else {}
    )
    avg_invested_weight_by_state_daily = (
        daily_invested_weight.groupby(daily_state).mean().to_dict() if not daily_invested_weight.empty else {}
    )

    pct_time_cash_regime_daily = float(cash_regime_mask.mean()) if len(daily_state) else 0.0
    pct_time_cash_invested_daily = float(cash_invested_mask.mean()) if len(daily_state) else 0.0
    pct_time_cash_true_daily = float(cash_true_mask.mean()) if len(daily_state) else 0.0

    result = BacktestResult(
        equity_curve=daily_equity,
        benchmark_curve=benchmark_curve,
        rebalance_records=rebalance_records,
        eligibility_records=eligibility_records,
        score_failure_records=score_failure_records,
        score_rejection_records=score_rejection_records,
        eligibility_funnel_records=eligibility_funnel_records,
        config=config,
        total_turnover=total_turnover,
        total_transaction_costs=total_tc,
        time_in_risk_on=state_counts[MarketState.RISK_ON] / n_rebalances if n_rebalances else 0,
        time_in_panic=state_counts[MarketState.PANIC] / n_rebalances if n_rebalances else 0,
        time_in_defensive_momentum=state_counts[MarketState.DEFENSIVE_MOMENTUM] / n_rebalances if n_rebalances else 0,
        # Panic event tracking
        panic_events_count=panic_events_count,
        panic_vol_ratio_triggers=panic_vol_ratio_triggers,
        panic_dd_triggers=panic_dd_triggers,
        panic_response_mode="DEFENSIVE_MOMENTUM" if config.panic_defensive_mode else "PANIC",
        # Turnover split
        turnover_risk_on=turnover_risk_on,
        turnover_defensive=turnover_defensive,
        turnover_cash_panic=turnover_cash_panic,
        # V3.5: Explicit cash states
        time_in_cash_invested=cash_invested_count / n_rebalances if n_rebalances else 0,
        time_in_true_cash=true_cash_count / n_rebalances if n_rebalances else 0,
        # V3.4: Concentration diagnostics
        pct_time_concentration_gated=concentration_gate_count / n_rebalances if n_rebalances else 0.0,
        avg_invested_fraction_by_state=avg_invested_fraction_by_state,
        avg_holdings_by_state=avg_holdings_by_state,
        pct_time_true_cash=true_cash_count / n_rebalances if n_rebalances else 0.0,
        avg_cash_weight_overall=avg_cash_weight_overall,
        avg_cash_weight_by_state=avg_cash_weight_by_state,
        daily_state=daily_state,
        daily_invested_weight=daily_invested_weight,
        daily_cash_weight=daily_cash_weight,
        avg_cash_weight_overall_daily=avg_cash_weight_overall_daily,
        avg_cash_weight_by_state_daily=avg_cash_weight_by_state_daily,
        avg_invested_weight_by_state_daily=avg_invested_weight_by_state_daily,
        pct_time_cash_regime_daily=pct_time_cash_regime_daily,
        pct_time_cash_invested_daily=pct_time_cash_invested_daily,
        pct_time_cash_true_daily=pct_time_cash_true_daily,
    )

    # Build log message with state distribution
    state_log = (f"RISK_ON={result.time_in_risk_on:.1%}, "
                 f"PANIC={result.time_in_panic:.1%}")
    if result.time_in_defensive_momentum > 0:
        state_log += f", DEFENSIVE_MOMENTUM={result.time_in_defensive_momentum:.1%}"
    state_log += f", CASH_INVESTED={result.time_in_cash_invested:.1%}, CASH_TRUE={result.time_in_true_cash:.1%}"

    # Log panic events (verify PANIC conditions are firing)
    if panic_events_count > 0:
        response_mode = "DEFENSIVE_MOMENTUM" if config.panic_defensive_mode else "PANIC"
        logger.info(f"Panic events: {panic_events_count} total "
                    f"(vol_ratio: {panic_vol_ratio_triggers}, dd: {panic_dd_triggers}), "
                    f"response: {response_mode}")

    logger.info(f"Backtest complete: Final equity={daily_equity.iloc[-1]:.4f}, {state_log}")

    return result


def _build_daily_equity(
    rebalance_equity: Dict[pd.Timestamp, float],
    price_data: pd.DataFrame,
    records: List[RebalanceRecord],
    trading_dates: pd.DatetimeIndex,
    backtest_end_date: date,
    cash_rate_annual: float = 0.0,
) -> pd.Series:
    """
    Build daily equity curve by interpolating between rebalances.

    Extends through backtest_end_date to capture returns after the last rebalance.

    Args:
        rebalance_equity: Equity at each rebalance date
        price_data: Stock price DataFrame
        records: Rebalance records with weights
        trading_dates: Trading calendar
        backtest_end_date: Configured end date for the backtest
        cash_rate_annual: Annualized cash yield for periods in cash

    Returns:
        Daily equity curve
    """
    if not records:
        return pd.Series(dtype=float)

    # Filter trading dates to backtest period
    # Extend through backtest_end_date to capture returns after last rebalance
    start_date = records[0].date
    end_date = pd.Timestamp(backtest_end_date)
    period_dates = trading_dates[
        (trading_dates >= start_date) & (trading_dates <= end_date)
    ]

    daily_equity = pd.Series(index=period_dates, dtype=float)

    # Fill in equity for each day
    current_weights: Dict[str, float] = {}
    current_equity = 1.0
    record_idx = 0

    # Daily cash yield (convert annual to daily)
    daily_cash_yield = cash_rate_annual / 252.0 if cash_rate_annual > 0 else 0.0

    for i, current_date in enumerate(period_dates):
        # Check if this is a rebalance date
        if record_idx < len(records) and current_date >= records[record_idx].date:
            current_equity = rebalance_equity.get(records[record_idx].date, current_equity)
            current_weights = records[record_idx].weights.copy()
            record_idx += 1
            daily_equity[current_date] = current_equity

        elif current_weights and i > 0:
            # Interpolate based on portfolio performance
            prev_date = period_dates[i - 1]
            prev_equity = daily_equity.get(prev_date, current_equity)

            # Use pure function for daily return (Fix 5)
            day_return = compute_daily_return(
                price_data,
                current_weights,
                prev_date,
                current_date,
                daily_cash_yield=daily_cash_yield,
            )
            daily_equity[current_date] = update_equity(prev_equity, day_return)

        elif i > 0:
            # In cash - apply daily cash yield
            prev_date = period_dates[i - 1]
            prev_equity = daily_equity.get(prev_date, current_equity)
            daily_equity[current_date] = prev_equity * (1 + daily_cash_yield)

        else:
            # First day
            daily_equity[current_date] = current_equity

    return daily_equity.dropna()


def _build_daily_diagnostics(
    records: List[RebalanceRecord],
    trading_dates: pd.DatetimeIndex,
    backtest_end_date: date,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Build daily state and cash exposure series based on rebalance records.

    Returns:
        Tuple of (daily_state, daily_invested_weight, daily_cash_weight)
    """
    if not records:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    start_date = records[0].date
    end_date = pd.Timestamp(backtest_end_date)
    period_dates = trading_dates[
        (trading_dates >= start_date) & (trading_dates <= end_date)
    ]

    daily_state = pd.Series(index=period_dates, dtype=object)
    daily_invested_weight = pd.Series(index=period_dates, dtype=float)
    daily_cash_weight = pd.Series(index=period_dates, dtype=float)

    record_idx = 0
    current_state = records[0].state
    current_invested = records[0].invested_fraction

    for current_date in period_dates:
        if record_idx < len(records) and current_date >= records[record_idx].date:
            current_state = records[record_idx].state
            current_invested = records[record_idx].invested_fraction
            record_idx += 1

        daily_state[current_date] = current_state.name
        daily_invested_weight[current_date] = current_invested
        daily_cash_weight[current_date] = 1.0 - current_invested

    return daily_state, daily_invested_weight, daily_cash_weight
