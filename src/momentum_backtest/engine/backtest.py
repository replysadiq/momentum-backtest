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

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from ..config import BacktestConfig, CashReplaceMode
from ..data.sanitizer import get_eligible_stocks
from .signals import rank_stocks_by_momentum, EligibilityBreakdown, select_defensive_basket
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
)
from .breadth import (
    BreadthConfig,
    BreadthSeries,
    compute_daily_breadth_series,
    get_breadth_confidence,
    scale_weights_by_breadth,
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
    # V3: Reason for CASH entry (None if not entering CASH this rebalance)
    cash_entry_reason: Optional[str] = None
    # V3.1: True if actually invested (not holding cash) - for CASH state with defensive replacement
    invested_flag: bool = False
    # V4: Breadth overlay values
    breadth_raw: Optional[float] = None  # Raw fraction of stocks with positive 63d return
    breadth_smooth: Optional[float] = None  # EMA-smoothed breadth
    breadth_confidence: Optional[float] = None  # Confidence scalar in [0, 1]
    base_exposure: float = 1.0  # Exposure before breadth overlay (1.0 if invested, 0.0 if cash)
    final_exposure: float = 1.0  # Exposure after breadth overlay


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
class BacktestResult:
    """Complete result of a backtest run."""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    rebalance_records: List[RebalanceRecord]
    eligibility_records: List[EligibilityRecord]
    config: BacktestConfig

    # Computed at end
    total_turnover: float = 0.0
    total_transaction_costs: float = 0.0
    time_in_risk_on: float = 0.0
    time_in_panic: float = 0.0
    time_in_defensive_momentum: float = 0.0  # V2 Lever A
    time_in_cash: float = 0.0

    # Panic event tracking (verify PANIC conditions are firing)
    panic_events_count: int = 0  # Total times PANIC condition was triggered
    panic_vol_ratio_triggers: int = 0  # Triggered by vol_ratio > threshold
    panic_dd_triggers: int = 0  # Triggered by portfolio DD > threshold
    panic_response_mode: str = "PANIC"  # "PANIC" or "DEFENSIVE_MOMENTUM"

    # Turnover split
    turnover_risk_on: float = 0.0
    turnover_defensive: float = 0.0
    turnover_cash_panic: float = 0.0  # Liquidation turnover

    # V3.1: Time in CASH state but actually invested (defensive replacement)
    time_in_cash_invested: float = 0.0

    # V4: Breadth overlay series
    breadth_series: Optional[BreadthSeries] = None


# =============================================================================
# Pure helper functions for testability (Fix 5)
# =============================================================================


def compute_segment_return(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    start_date: pd.Timestamp,
    end_date_exclusive: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
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

    return portfolio_return


def compute_daily_return(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    prev_date: pd.Timestamp,
    curr_date: pd.Timestamp,
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
        return 0.0

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

    # Universe count for eligibility reporting
    universe_count = len(price_data.columns)

    # Build benchmark equity curve (buy and hold)
    bench_start_date = rebalance_calendar[0]
    bench_start_price = benchmark_prices[benchmark_prices.index >= bench_start_date].iloc[0]
    benchmark_curve = benchmark_prices / bench_start_price

    # V4: Pre-compute daily breadth series (if enabled)
    breadth_series: Optional[BreadthSeries] = None
    if config.enable_breadth_overlay:
        breadth_config = BreadthConfig(
            lookback=config.breadth_lookback,
            ema_span=config.breadth_ema_span,
            low=config.breadth_low,
            high=config.breadth_high,
            min_coverage=config.breadth_min_coverage,
        )
        breadth_series = compute_daily_breadth_series(
            price_data, trading_dates, breadth_config
        )
        logger.info(f"Breadth overlay enabled: lookback={config.breadth_lookback}, "
                    f"ema_span={config.breadth_ema_span}, low={config.breadth_low}, high={config.breadth_high}")

    # Track portfolio equity between rebalances
    last_rebalance_date: Optional[pd.Timestamp] = None
    last_rebalance_equity = equity

    for i, rebalance_date in enumerate(rebalance_calendar):
        logger.debug(f"Processing rebalance {i+1}/{len(rebalance_calendar)}: {rebalance_date.date()}")

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

        # Step 6: Execute based on state
        new_weights: Dict[str, float] = {}
        selected_tickers: List[str] = []
        turnover = 0.0
        transaction_cost = 0.0
        invested_flag = False  # V3.1: True if actually holding equities

        # Convert current_weights to Series for turnover calculation
        prev_weights_series = pd.Series(current_weights) if current_weights else pd.Series(dtype=float)

        if current_state == MarketState.RISK_ON:
            invested_flag = True  # RISK_ON = invested
            # Get eligible stocks
            eligible = get_eligible_stocks(price_data, rebalance_date, trading_dates)

            # Rank by momentum (returns RankingResult with scores and eligibility)
            # V2 Lever D: Pass disable_6m_filter
            ranking_result = rank_stocks_by_momentum(
                price_data,
                eligible,
                rebalance_date,
                universe_count=universe_count,
                use_dd_filter=config.use_dd_filter,
                dd_threshold=config.dd_threshold,
                nse_style=config.nse_style_scoring,
                disable_6m_filter=config.disable_6m_filter,  # V2 Lever D
            )

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

            # Record eligibility data
            elig = ranking_result.eligibility
            eligibility_records.append(EligibilityRecord(
                date=rebalance_date,
                state=current_state,
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

            # Compute turnover using canonical function (Fix 2)
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_risk_on += turnover  # Track RISK_ON turnover

            # Apply transaction costs
            transaction_cost = apply_transaction_cost(0.0, turnover, config.tc_bps)
            equity = update_equity(equity, -abs(transaction_cost))
            equity_curve[rebalance_date] = equity

        elif current_state == MarketState.DEFENSIVE_MOMENTUM:
            invested_flag = True  # DEFENSIVE_MOMENTUM = invested
            # V2 Lever A: Hold low-volatility stocks during panic instead of cash
            eligible = get_eligible_stocks(price_data, rebalance_date, trading_dates)

            # Select defensive basket (lowest volatility stocks with positive momentum)
            defensive_scores = select_defensive_basket(
                price_data,
                eligible,
                rebalance_date,
                basket_size=config.defensive_basket_size,
            )

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

            # Compute turnover
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_defensive += turnover  # Track DEFENSIVE_MOMENTUM turnover

            # Apply transaction costs
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

        elif current_state == MarketState.CASH and config.cash_replace_mode == CashReplaceMode.DEFENSIVE:
            # V3.1: CASH state with defensive replacement - hold defensive portfolio
            invested_flag = True

            eligible = get_eligible_stocks(price_data, rebalance_date, trading_dates)

            # Select defensive basket (same as DEFENSIVE_MOMENTUM state)
            defensive_scores = select_defensive_basket(
                price_data,
                eligible,
                rebalance_date,
                basket_size=config.defensive_basket_size,
            )

            # Build defensive portfolio
            allocation = build_defensive_portfolio(
                defensive_scores,
                max_weight=config.max_weight,
            )

            new_weights = allocation.weights
            selected_tickers = allocation.tickers

            # Compute turnover
            target_weights_series = pd.Series(new_weights) if new_weights else pd.Series(dtype=float)
            turnover = compute_turnover(prev_weights_series, target_weights_series)
            turnover_defensive += turnover  # Track as defensive turnover

            # Apply transaction costs
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
            # PANIC or CASH (with no replacement) - liquidate to cash
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
                state=current_state,
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

        # Step 6a: V4 Breadth overlay - scale weights if enabled and invested
        breadth_raw_val: Optional[float] = None
        breadth_smooth_val: Optional[float] = None
        breadth_conf_val: Optional[float] = None
        base_exposure = 1.0 if invested_flag else 0.0
        final_exposure = base_exposure

        if config.enable_breadth_overlay and breadth_series is not None:
            # Look up breadth values for this date
            if rebalance_date in breadth_series.breadth_raw.index:
                breadth_raw_val = breadth_series.breadth_raw.get(rebalance_date)
                if pd.notna(breadth_raw_val):
                    breadth_raw_val = float(breadth_raw_val)
                else:
                    breadth_raw_val = None
            if rebalance_date in breadth_series.breadth_smooth.index:
                breadth_smooth_val = float(breadth_series.breadth_smooth[rebalance_date])
            breadth_conf_val = get_breadth_confidence(breadth_series, rebalance_date)

            # Apply breadth scaling only to invested states
            if invested_flag and new_weights:
                # Scale weights by breadth confidence (remainder goes to cash)
                scaled_weights = scale_weights_by_breadth(new_weights, breadth_conf_val)
                final_exposure = base_exposure * breadth_conf_val

                # Recompute turnover with scaled weights (vs previous scaled weights)
                target_weights_series = pd.Series(scaled_weights) if scaled_weights else pd.Series(dtype=float)
                turnover = compute_turnover(prev_weights_series, target_weights_series)

                # Apply transaction costs for any change from scaling
                if turnover > 0:
                    additional_tc = apply_transaction_cost(0.0, turnover, config.tc_bps)
                    transaction_cost = abs(additional_tc)
                    equity = update_equity(equity, -abs(additional_tc))
                    equity_curve[rebalance_date] = equity

                # Track turnover by state
                if current_state == MarketState.RISK_ON:
                    turnover_risk_on += turnover
                elif current_state in (MarketState.DEFENSIVE_MOMENTUM, MarketState.CASH):
                    turnover_defensive += turnover

                new_weights = scaled_weights
                selected_tickers = list(new_weights.keys())

                raw_str = f"{breadth_raw_val:.3f}" if breadth_raw_val is not None else "N/A"
                smooth_str = f"{breadth_smooth_val:.3f}" if breadth_smooth_val is not None else "N/A"
                logger.debug(f"  Breadth: raw={raw_str}, smooth={smooth_str}, "
                            f"conf={breadth_conf_val:.3f}, exposure={final_exposure:.2%}")

        # Step 6b: Record rebalance event
        # Capture cash_entry_reason from context (only set when entering CASH)
        cash_reason = state_context.cash_entry_reason
        state_context.cash_entry_reason = None  # Clear after capturing

        record = RebalanceRecord(
            date=rebalance_date,
            state=current_state,
            features=features,
            tickers=selected_tickers,
            weights=new_weights.copy(),
            turnover=turnover,
            transaction_cost=abs(transaction_cost),
            cash_entry_reason=cash_reason,
            invested_flag=invested_flag,
            # V4: Breadth values
            breadth_raw=breadth_raw_val,
            breadth_smooth=breadth_smooth_val,
            breadth_confidence=breadth_conf_val,
            base_exposure=base_exposure,
            final_exposure=final_exposure,
        )
        rebalance_records.append(record)

        # Update tracking
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

    # V3.1: Count CASH periods where invested_flag=True (defensive replacement)
    cash_invested_count = sum(
        1 for r in rebalance_records
        if r.state == MarketState.CASH and r.invested_flag
    )

    result = BacktestResult(
        equity_curve=daily_equity,
        benchmark_curve=benchmark_curve,
        rebalance_records=rebalance_records,
        eligibility_records=eligibility_records,
        config=config,
        total_turnover=total_turnover,
        total_transaction_costs=total_tc,
        time_in_risk_on=state_counts[MarketState.RISK_ON] / n_rebalances if n_rebalances else 0,
        time_in_panic=state_counts[MarketState.PANIC] / n_rebalances if n_rebalances else 0,
        time_in_defensive_momentum=state_counts[MarketState.DEFENSIVE_MOMENTUM] / n_rebalances if n_rebalances else 0,
        time_in_cash=state_counts[MarketState.CASH] / n_rebalances if n_rebalances else 0,
        # Panic event tracking
        panic_events_count=panic_events_count,
        panic_vol_ratio_triggers=panic_vol_ratio_triggers,
        panic_dd_triggers=panic_dd_triggers,
        panic_response_mode="DEFENSIVE_MOMENTUM" if config.panic_defensive_mode else "PANIC",
        # Turnover split
        turnover_risk_on=turnover_risk_on,
        turnover_defensive=turnover_defensive,
        turnover_cash_panic=turnover_cash_panic,
        # V3.1: Time in CASH but invested
        time_in_cash_invested=cash_invested_count / n_rebalances if n_rebalances else 0,
        # V4: Breadth series
        breadth_series=breadth_series,
    )

    # Build log message with state distribution
    state_log = (f"RISK_ON={result.time_in_risk_on:.1%}, "
                 f"PANIC={result.time_in_panic:.1%}")
    if result.time_in_defensive_momentum > 0:
        state_log += f", DEFENSIVE_MOMENTUM={result.time_in_defensive_momentum:.1%}"
    state_log += f", CASH={result.time_in_cash:.1%}"

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
                price_data, current_weights, prev_date, current_date
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
