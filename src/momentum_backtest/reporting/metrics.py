"""
Performance metric calculations.

Computes standard portfolio metrics:
- CAGR (Compound Annual Growth Rate)
- Volatility (annualized)
- Maximum Drawdown
- Sharpe Ratio
- Calmar Ratio
- Sortino Ratio
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics."""
    # Return metrics
    total_return: float
    cagr: float

    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration_days: int

    # Risk-adjusted metrics
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float

    # Benchmark comparison
    benchmark_total_return: float
    benchmark_cagr: float
    benchmark_volatility: float
    benchmark_max_drawdown: float
    excess_return: float
    information_ratio: float

    # Portfolio statistics
    time_in_risk_on: float
    time_in_panic: float
    time_in_defensive_momentum: float  # V2 Lever A
    pct_time_cash_invested: float
    pct_time_true_cash: float
    pct_time_cash_regime_daily: float
    pct_time_cash_invested_daily: float
    pct_time_cash_true_daily: float
    total_turnover: float
    total_transaction_costs: float

    # Period info
    start_date: str
    end_date: str
    trading_days: int
    years: float

    # V2/V3 Strategy versioning
    strategy_version: str = "v1"
    enabled_levers: List[str] = field(default_factory=list)

    # V3: Cash entry mode
    cash_entry_mode: str = "baseline"

    # Panic event tracking (verify PANIC conditions are firing)
    panic_events_count: int = 0
    panic_vol_ratio_triggers: int = 0
    panic_dd_triggers: int = 0
    panic_response_mode: str = "PANIC"  # "PANIC" or "DEFENSIVE_MOMENTUM"

    # Turnover split
    turnover_risk_on: float = 0.0
    turnover_defensive: float = 0.0
    turnover_cash_panic: float = 0.0

    # V3.1: Cash replacement mode and invested time
    cash_replace_mode: str = "none"
    time_in_cash_invested: float = 0.0  # Legacy field (kept for backward compatibility)

    # V3.4: Concentration diagnostics
    pct_time_concentration_gated: float = 0.0
    avg_invested_fraction_by_state: dict = field(default_factory=dict)
    avg_holdings_by_state: dict = field(default_factory=dict)
    avg_cash_weight_overall_rebalance: float = 0.0
    avg_cash_weight_overall_daily: float = 0.0
    avg_cash_weight_by_state: dict = field(default_factory=dict)
    avg_cash_weight_by_state_daily: dict = field(default_factory=dict)
    avg_invested_weight_by_state_daily: dict = field(default_factory=dict)


def compute_metrics(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    time_in_risk_on: float,
    time_in_panic: float,
    pct_time_cash_invested: float,
    pct_time_true_cash: float,
    pct_time_cash_regime_daily: float,
    pct_time_cash_invested_daily: float,
    pct_time_cash_true_daily: float,
    total_turnover: float,
    total_transaction_costs: float,
    risk_free_rate: float = 0.0,
    time_in_defensive_momentum: float = 0.0,
    strategy_version: str = "v1",
    enabled_levers: Optional[List[str]] = None,
    # Panic event tracking
    panic_events_count: int = 0,
    panic_vol_ratio_triggers: int = 0,
    panic_dd_triggers: int = 0,
    panic_response_mode: str = "PANIC",
    # Turnover split
    turnover_risk_on: float = 0.0,
    turnover_defensive: float = 0.0,
    turnover_cash_panic: float = 0.0,
    # V3: Cash entry mode
    cash_entry_mode: str = "baseline",
    # V3.1: Cash replacement mode
    cash_replace_mode: str = "none",
    time_in_cash_invested: float = 0.0,
    pct_time_concentration_gated: float = 0.0,
    avg_invested_fraction_by_state: Optional[dict] = None,
    avg_holdings_by_state: Optional[dict] = None,
    avg_cash_weight_overall_rebalance: float = 0.0,
    avg_cash_weight_overall_daily: float = 0.0,
    avg_cash_weight_by_state: Optional[dict] = None,
    avg_cash_weight_by_state_daily: Optional[dict] = None,
    avg_invested_weight_by_state_daily: Optional[dict] = None,
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        equity_curve: Daily portfolio equity curve
        benchmark_curve: Daily benchmark equity curve
        time_in_risk_on: Fraction of time in RISK_ON state
        time_in_panic: Fraction of time in PANIC state
    pct_time_cash_invested: Fraction of time in CASH_INVESTED state
    pct_time_true_cash: Fraction of time in CASH_TRUE state
        total_turnover: Cumulative portfolio turnover
        total_transaction_costs: Cumulative transaction costs
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        time_in_defensive_momentum: Fraction of time in DEFENSIVE_MOMENTUM state (V2)
        strategy_version: Strategy version ("v1" or "v2")
        enabled_levers: List of enabled V2 levers

    Returns:
        PerformanceMetrics with all calculated metrics
    """
    if enabled_levers is None:
        enabled_levers = []
    if avg_invested_fraction_by_state is None:
        avg_invested_fraction_by_state = {}
    if avg_holdings_by_state is None:
        avg_holdings_by_state = {}
    if avg_cash_weight_by_state is None:
        avg_cash_weight_by_state = {}
    if avg_cash_weight_by_state_daily is None:
        avg_cash_weight_by_state_daily = {}
    if avg_invested_weight_by_state_daily is None:
        avg_invested_weight_by_state_daily = {}
    # Align equity curves to common dates
    common_dates = equity_curve.index.intersection(benchmark_curve.index)
    equity = equity_curve.reindex(common_dates)
    benchmark = benchmark_curve.reindex(common_dates)

    # Basic period info
    start_date = equity.index[0]
    end_date = equity.index[-1]
    trading_days = len(equity)
    years = trading_days / 252.0

    # Portfolio returns
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    cagr = _compute_cagr(equity)

    # Portfolio risk
    volatility = _compute_volatility(equity)
    max_dd, max_dd_duration = _compute_max_drawdown(equity)

    # Risk-adjusted metrics
    sharpe = _compute_sharpe_ratio(equity, risk_free_rate)
    calmar = cagr / max_dd if max_dd > 0 else 0.0
    sortino = _compute_sortino_ratio(equity, risk_free_rate)

    # Benchmark metrics
    bench_total_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    bench_cagr = _compute_cagr(benchmark)
    bench_vol = _compute_volatility(benchmark)
    bench_max_dd, _ = _compute_max_drawdown(benchmark)

    # Relative metrics
    excess_return = cagr - bench_cagr
    info_ratio = _compute_information_ratio(equity, benchmark)

    metrics = PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        benchmark_total_return=bench_total_return,
        benchmark_cagr=bench_cagr,
        benchmark_volatility=bench_vol,
        benchmark_max_drawdown=bench_max_dd,
        excess_return=excess_return,
        information_ratio=info_ratio,
        time_in_risk_on=time_in_risk_on,
        time_in_panic=time_in_panic,
        time_in_defensive_momentum=time_in_defensive_momentum,
        pct_time_cash_invested=pct_time_cash_invested,
        pct_time_true_cash=pct_time_true_cash,
        pct_time_cash_regime_daily=pct_time_cash_regime_daily,
        pct_time_cash_invested_daily=pct_time_cash_invested_daily,
        pct_time_cash_true_daily=pct_time_cash_true_daily,
        total_turnover=total_turnover,
        total_transaction_costs=total_transaction_costs,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        trading_days=trading_days,
        years=years,
        strategy_version=strategy_version,
        enabled_levers=enabled_levers,
        cash_entry_mode=cash_entry_mode,
        # Panic event tracking
        panic_events_count=panic_events_count,
        panic_vol_ratio_triggers=panic_vol_ratio_triggers,
        panic_dd_triggers=panic_dd_triggers,
        panic_response_mode=panic_response_mode,
        # Turnover split
        turnover_risk_on=turnover_risk_on,
        turnover_defensive=turnover_defensive,
        turnover_cash_panic=turnover_cash_panic,
        # V3.1: Cash replacement
        cash_replace_mode=cash_replace_mode,
        time_in_cash_invested=time_in_cash_invested,
        pct_time_concentration_gated=pct_time_concentration_gated,
        avg_invested_fraction_by_state=avg_invested_fraction_by_state,
        avg_holdings_by_state=avg_holdings_by_state,
        avg_cash_weight_overall_rebalance=avg_cash_weight_overall_rebalance,
        avg_cash_weight_overall_daily=avg_cash_weight_overall_daily,
        avg_cash_weight_by_state=avg_cash_weight_by_state,
        avg_cash_weight_by_state_daily=avg_cash_weight_by_state_daily,
        avg_invested_weight_by_state_daily=avg_invested_weight_by_state_daily,
    )

    return metrics


def _compute_cagr(equity: pd.Series) -> float:
    """Compute Compound Annual Growth Rate."""
    if len(equity) < 2:
        return 0.0

    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / 252.0

    if years <= 0 or total_return <= 0:
        return 0.0

    return total_return ** (1 / years) - 1


def _compute_volatility(equity: pd.Series) -> float:
    """Compute annualized volatility."""
    if len(equity) < 20:
        return 0.0

    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 10:
        return 0.0

    return daily_returns.std() * np.sqrt(252)


def _compute_max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """
    Compute maximum drawdown and its duration.

    Returns:
        Tuple of (max_drawdown, duration_in_days)
    """
    if len(equity) < 2:
        return 0.0, 0

    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    max_dd = abs(drawdown.min())

    # Compute duration of max drawdown
    trough_idx = drawdown.idxmin()

    # Find the peak before max drawdown: the date when equity first reached
    # the running_max value at trough time
    peak_value = running_max[trough_idx]
    peak_candidates = equity[equity >= peak_value]
    peak_idx = peak_candidates.index[0]  # First time equity reached this peak

    # Find recovery point (next time equity equals peak after trough)
    recovery_candidates = equity[equity.index > trough_idx]
    recovery_candidates = recovery_candidates[recovery_candidates >= peak_value]

    if len(recovery_candidates) > 0:
        recovery_idx = recovery_candidates.index[0]
        duration = (recovery_idx - peak_idx).days
    else:
        # Never recovered - compute duration from peak to end of backtest
        duration = (equity.index[-1] - peak_idx).days

    return max_dd, duration


def _compute_sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe Ratio."""
    if len(equity) < 20:
        return 0.0

    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 10:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess <= 0:
        return 0.0

    return (mean_excess / std_excess) * np.sqrt(252)


def _compute_sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sortino Ratio (downside deviation only)."""
    if len(equity) < 20:
        return 0.0

    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 10:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf

    # Downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]

    if len(negative_returns) < 5:
        return 0.0

    downside_std = negative_returns.std()

    if downside_std <= 0:
        return 0.0

    mean_excess = excess_returns.mean()
    return (mean_excess / downside_std) * np.sqrt(252)


def _compute_information_ratio(
    equity: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Compute Information Ratio (excess return / tracking error)."""
    if len(equity) < 20:
        return 0.0

    port_returns = equity.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()

    # Align
    common_idx = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.reindex(common_idx)
    bench_returns = bench_returns.reindex(common_idx)

    if len(port_returns) < 10:
        return 0.0

    excess_returns = port_returns - bench_returns
    tracking_error = excess_returns.std()

    if tracking_error <= 0:
        return 0.0

    mean_excess = excess_returns.mean()
    return (mean_excess / tracking_error) * np.sqrt(252)


def format_metrics_table(metrics: PerformanceMetrics) -> str:
    """
    Format metrics as a readable table for console output.

    Args:
        metrics: PerformanceMetrics object

    Returns:
        Formatted string table
    """
    # Build header with strategy version
    header = "BACKTEST PERFORMANCE SUMMARY"
    if metrics.strategy_version == "v2":
        header += f" (Strategy v2)"

    lines = [
        "=" * 60,
        header,
        "=" * 60,
        f"Period: {metrics.start_date} to {metrics.end_date}",
        f"Trading Days: {metrics.trading_days} ({metrics.years:.2f} years)",
    ]

    # Add enabled levers if v2
    if metrics.enabled_levers:
        lines.append(f"Enabled Levers: {', '.join(metrics.enabled_levers)}")

    lines.extend([
        "",
        "PORTFOLIO RETURNS",
        "-" * 30,
        f"  Total Return:     {metrics.total_return:>10.2%}",
        f"  CAGR:             {metrics.cagr:>10.2%}",
        f"  Volatility:       {metrics.volatility:>10.2%}",
        f"  Max Drawdown:     {metrics.max_drawdown:>10.2%}",
        f"  DD Duration:      {metrics.max_drawdown_duration_days:>10} days",
        "",
        "RISK-ADJUSTED METRICS",
        "-" * 30,
        f"  Sharpe Ratio:     {metrics.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:    {metrics.sortino_ratio:>10.2f}",
        f"  Calmar Ratio:     {metrics.calmar_ratio:>10.2f}",
        "",
        "BENCHMARK COMPARISON",
        "-" * 30,
        f"  Benchmark CAGR:   {metrics.benchmark_cagr:>10.2%}",
        f"  Benchmark Vol:    {metrics.benchmark_volatility:>10.2%}",
        f"  Benchmark MaxDD:  {metrics.benchmark_max_drawdown:>10.2%}",
        f"  Excess Return:    {metrics.excess_return:>10.2%}",
        f"  Info Ratio:       {metrics.information_ratio:>10.2f}",
        "",
        "STATE DISTRIBUTION",
        "-" * 30,
        f"  RISK_ON:          {metrics.time_in_risk_on:>10.1%}",
        f"  PANIC:            {metrics.time_in_panic:>10.1%}",
    ])

    # Include DEFENSIVE_MOMENTUM if used
    if metrics.time_in_defensive_momentum > 0:
        lines.append(f"  DEFENSIVE_MOM:    {metrics.time_in_defensive_momentum:>10.1%}")

    lines.extend([
        f"  CASH_INVESTED:    {metrics.pct_time_cash_invested:>10.1%}",
        f"  CASH_TRUE:        {metrics.pct_time_true_cash:>10.1%}",
        "",
        "TRADING ACTIVITY",
        "-" * 30,
        f"  Total Turnover:   {metrics.total_turnover:>10.2f}x",
        f"  Total TC:         {metrics.total_transaction_costs:>10.4f}",
        "=" * 60,
    ])

    return "\n".join(lines)


@dataclass
class RollingExcessStats:
    """Summary statistics for rolling 3-year excess return."""
    window_months: int
    frequency: str
    benchmark_used: str
    n_observations: int
    pct_positive: float
    mean_excess: float
    median_excess: float
    min_excess: float
    max_excess: float
    longest_negative_streak_months: int


def compute_rolling_excess_return(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    benchmark_name: str = "BENCHMARK",
    window_months: int = 36,
) -> tuple[pd.DataFrame, RollingExcessStats]:
    """
    Compute rolling N-year excess return (CAGR difference) vs benchmark.

    This metric is DIAGNOSTIC ONLY - for narrative and analysis.
    It should NOT be used for trading decisions or optimization.

    Args:
        equity_curve: Strategy equity curve (daily, normalized to start at 1.0)
        benchmark_curve: Benchmark equity curve (daily, same dates as equity)
        rebalance_dates: Dates to compute rolling metrics on (monthly)
        benchmark_name: Name of benchmark for labeling
        window_months: Rolling window in months (default: 36 = 3 years)

    Returns:
        Tuple of:
        - DataFrame with columns: date, rolling_3y_excess_cagr
        - RollingExcessStats with summary statistics
    """
    # Align benchmark to equity curve dates
    benchmark_aligned = benchmark_curve.reindex(equity_curve.index)

    # Calculate approximate trading days per month
    days_per_month = 21  # ~252/12
    window_days = window_months * days_per_month

    results = []

    for rebal_date in rebalance_dates:
        # Find the window start (approximately N months back)
        window_start_idx = equity_curve.index.get_indexer([rebal_date], method='ffill')[0]

        if window_start_idx < window_days:
            # Not enough history for full window
            continue

        # Get window bounds
        end_idx = window_start_idx
        start_idx = end_idx - window_days

        # Extract windows
        strat_window = equity_curve.iloc[start_idx:end_idx + 1]
        bench_window = benchmark_aligned.iloc[start_idx:end_idx + 1]

        # Skip if benchmark has missing data in window
        if bench_window.isna().any() or len(bench_window) < window_days * 0.9:
            continue

        # Compute CAGR for both
        strat_cagr = _compute_cagr(strat_window)
        bench_cagr = _compute_cagr(bench_window)

        # Excess return = Strategy CAGR - Benchmark CAGR
        excess_cagr = strat_cagr - bench_cagr

        results.append({
            'date': rebal_date,
            'rolling_3y_excess_cagr': excess_cagr,
            'strategy_cagr': strat_cagr,
            'benchmark_cagr': bench_cagr,
        })

    if not results:
        # Return empty results if not enough data
        empty_df = pd.DataFrame(columns=['date', 'rolling_3y_excess_cagr'])
        empty_stats = RollingExcessStats(
            window_months=window_months,
            frequency="monthly",
            benchmark_used=benchmark_name,
            n_observations=0,
            pct_positive=0.0,
            mean_excess=0.0,
            median_excess=0.0,
            min_excess=0.0,
            max_excess=0.0,
            longest_negative_streak_months=0,
        )
        return empty_df, empty_stats

    # Build DataFrame
    df = pd.DataFrame(results)

    # Compute summary statistics
    excess_values = df['rolling_3y_excess_cagr']
    n_obs = len(excess_values)
    n_positive = (excess_values > 0).sum()

    # Compute longest negative streak
    longest_neg_streak = _compute_longest_negative_streak(excess_values)

    stats = RollingExcessStats(
        window_months=window_months,
        frequency="monthly",
        benchmark_used=benchmark_name,
        n_observations=n_obs,
        pct_positive=n_positive / n_obs if n_obs > 0 else 0.0,
        mean_excess=float(excess_values.mean()),
        median_excess=float(excess_values.median()),
        min_excess=float(excess_values.min()),
        max_excess=float(excess_values.max()),
        longest_negative_streak_months=longest_neg_streak,
    )

    logger.debug(
        f"Rolling {window_months}M excess return: "
        f"{n_obs} observations, {stats.pct_positive:.1%} positive, "
        f"mean={stats.mean_excess:.2%}"
    )

    return df, stats


def compute_rolling_returns(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    window_months: int = 36,
) -> pd.DataFrame:
    """
    Compute rolling 3-year CAGR and total return for strategy and benchmark.

    Uses rebalance_dates as monthly observation points.
    """
    if len(rebalance_dates) == 0:
        return pd.DataFrame(columns=[
            "date",
            "strategy_rolling_3y_cagr",
            "mom50_rolling_3y_cagr",
            "rolling_3y_excess_cagr",
            "strategy_rolling_3y_total_return",
            "mom50_rolling_3y_total_return",
        ])

    dates = pd.to_datetime(rebalance_dates)
    equity = equity_curve.reindex(dates).dropna()
    bench = benchmark_curve.reindex(dates).dropna()

    common_dates = equity.index.intersection(bench.index)
    if len(common_dates) == 0:
        return pd.DataFrame(columns=[
            "date",
            "strategy_rolling_3y_cagr",
            "mom50_rolling_3y_cagr",
            "rolling_3y_excess_cagr",
            "strategy_rolling_3y_total_return",
            "mom50_rolling_3y_total_return",
        ])

    equity = equity.reindex(common_dates)
    bench = bench.reindex(common_dates)

    records = []
    for end_date in common_dates:
        start_date_target = end_date - pd.DateOffset(months=window_months)
        start_candidates = common_dates[common_dates <= start_date_target]
        if len(start_candidates) == 0:
            continue
        start_date = start_candidates[-1]

        strat_start = equity.loc[start_date]
        strat_end = equity.loc[end_date]
        bench_start = bench.loc[start_date]
        bench_end = bench.loc[end_date]

        if strat_start <= 0 or bench_start <= 0:
            continue

        strat_total = (strat_end / strat_start) - 1.0
        bench_total = (bench_end / bench_start) - 1.0
        strat_cagr = (strat_end / strat_start) ** (12.0 / window_months) - 1.0
        bench_cagr = (bench_end / bench_start) ** (12.0 / window_months) - 1.0

        records.append({
            "date": end_date,
            "rolling_window_start_date": start_date,
            "strategy_rolling_3y_cagr": strat_cagr,
            "mom50_rolling_3y_cagr": bench_cagr,
            "rolling_3y_excess_cagr": strat_cagr - bench_cagr,
            "strategy_rolling_3y_total_return": strat_total,
            "mom50_rolling_3y_total_return": bench_total,
            "rolling_3y_excess_total_return": strat_total - bench_total,
        })

    return pd.DataFrame(records)


def _compute_longest_negative_streak(series: pd.Series) -> int:
    """Compute the longest consecutive streak of negative values."""
    if len(series) == 0:
        return 0

    max_streak = 0
    current_streak = 0

    for value in series:
        if value < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak
