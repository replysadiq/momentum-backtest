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
    time_in_cash: float
    total_turnover: float
    total_transaction_costs: float

    # Period info
    start_date: str
    end_date: str
    trading_days: int
    years: float

    # V2 Strategy versioning
    strategy_version: str = "v1"
    enabled_levers: List[str] = field(default_factory=list)

    # Panic event tracking (verify PANIC conditions are firing)
    panic_events_count: int = 0
    panic_vol_ratio_triggers: int = 0
    panic_dd_triggers: int = 0
    panic_response_mode: str = "PANIC"  # "PANIC" or "DEFENSIVE_MOMENTUM"

    # Turnover split
    turnover_risk_on: float = 0.0
    turnover_defensive: float = 0.0
    turnover_cash_panic: float = 0.0


def compute_metrics(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    time_in_risk_on: float,
    time_in_panic: float,
    time_in_cash: float,
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
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.

    Args:
        equity_curve: Daily portfolio equity curve
        benchmark_curve: Daily benchmark equity curve
        time_in_risk_on: Fraction of time in RISK_ON state
        time_in_panic: Fraction of time in PANIC state
        time_in_cash: Fraction of time in CASH state
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
        time_in_cash=time_in_cash,
        total_turnover=total_turnover,
        total_transaction_costs=total_transaction_costs,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        trading_days=trading_days,
        years=years,
        strategy_version=strategy_version,
        enabled_levers=enabled_levers,
        # Panic event tracking
        panic_events_count=panic_events_count,
        panic_vol_ratio_triggers=panic_vol_ratio_triggers,
        panic_dd_triggers=panic_dd_triggers,
        panic_response_mode=panic_response_mode,
        # Turnover split
        turnover_risk_on=turnover_risk_on,
        turnover_defensive=turnover_defensive,
        turnover_cash_panic=turnover_cash_panic,
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
    peak_idx = running_max.idxmax()
    trough_idx = drawdown.idxmin()

    # Find recovery point (next time equity equals peak)
    recovery_candidates = equity[equity.index > trough_idx]
    recovery_candidates = recovery_candidates[recovery_candidates >= running_max[trough_idx]]

    if len(recovery_candidates) > 0:
        recovery_idx = recovery_candidates.index[0]
        duration = (recovery_idx - peak_idx).days
    else:
        # Never recovered
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
        f"  CASH:             {metrics.time_in_cash:>10.1%}",
        "",
        "TRADING ACTIVITY",
        "-" * 30,
        f"  Total Turnover:   {metrics.total_turnover:>10.2f}x",
        f"  Total TC:         {metrics.total_transaction_costs:>10.4f}",
        "=" * 60,
    ])

    return "\n".join(lines)
