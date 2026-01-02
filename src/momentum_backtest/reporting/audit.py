"""
Audit reports for backtest validation.

Provides:
- benchmark_info.json: Benchmark identity and coverage validation
- drawdown_attribution.csv: Worst drawdown analysis
- holdings_snapshot.csv: Complete holdings at each rebalance
- eligibility_coverage.csv: Stock filtering breakdown
"""

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import json
import logging

import numpy as np
import pandas as pd

from ..engine.backtest import RebalanceRecord
from ..engine.state_machine import MarketState


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkInfo:
    """Benchmark identity and coverage information."""
    ticker: str
    is_proxy: bool
    proxy_warning: str
    coverage_pct: float
    total_days: int
    first_date: str
    last_date: str
    backtest_start: str
    backtest_end: str
    fallback_chain_attempted: List[str]


@dataclass
class DrawdownEvent:
    """A single drawdown event with attribution."""
    start_date: str
    trough_date: str
    recovery_date: Optional[str]
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_to_trough_days: int
    duration_to_recovery_days: Optional[int]
    state_at_start: str
    state_at_trough: str
    states_during: List[str]
    tc_applied_during: float


# Note: EligibilityRecord is imported from backtest module when exporting


def export_benchmark_info(
    benchmark_ticker: str,
    is_proxy: bool,
    coverage: float,
    benchmark_data: pd.Series,
    backtest_start: date,
    backtest_end: date,
    fallback_chain: List[str],
    output_dir: Path,
) -> BenchmarkInfo:
    """
    Export benchmark identity and coverage validation.

    Args:
        benchmark_ticker: The selected benchmark ticker
    is_proxy: Whether using a proxy benchmark
        coverage: Coverage percentage
        benchmark_data: The benchmark price series
        backtest_start: Backtest start date
        backtest_end: Backtest end date
        fallback_chain: List of tickers tried
        output_dir: Output directory

    Returns:
        BenchmarkInfo object
    """
    proxy_warning = ""
    if is_proxy:
        proxy_warning = (
            "WARNING: Using a proxy benchmark. "
            "Calendar and regime decisions may differ from the intended benchmark."
        )

    info = BenchmarkInfo(
        ticker=benchmark_ticker,
        is_proxy=is_proxy,
        proxy_warning=proxy_warning,
        coverage_pct=round(coverage * 100, 2),
        total_days=len(benchmark_data),
        first_date=benchmark_data.index[0].strftime("%Y-%m-%d"),
        last_date=benchmark_data.index[-1].strftime("%Y-%m-%d"),
        backtest_start=backtest_start.isoformat(),
        backtest_end=backtest_end.isoformat(),
        fallback_chain_attempted=fallback_chain,
    )

    filepath = output_dir / "benchmark_info.json"
    with open(filepath, "w") as f:
        json.dump(asdict(info), f, indent=2)

    logger.info(f"  Wrote {filepath}")

    # Also log to console for visibility
    logger.info(f"  Benchmark: {benchmark_ticker} ({'PROXY' if is_proxy else 'PRIMARY'})")
    logger.info(f"  Coverage: {info.coverage_pct}% ({info.total_days} days)")
    logger.info(f"  Date range: {info.first_date} to {info.last_date}")
    if proxy_warning:
        logger.warning(f"  {proxy_warning}")

    return info


def compute_drawdown_attribution(
    equity_curve: pd.Series,
    rebalance_records: List[RebalanceRecord],
    top_n: int = 3,
) -> List[DrawdownEvent]:
    """
    Compute attribution for the worst drawdown events.

    Args:
        equity_curve: Daily portfolio equity curve
        rebalance_records: List of rebalance records with state info
        top_n: Number of worst drawdowns to analyze

    Returns:
        List of DrawdownEvent objects
    """
    # Compute running maximum and drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    # Build state lookup by date
    state_by_date = {}
    for record in rebalance_records:
        state_by_date[record.date] = record.state

    # Find all local minima (drawdown troughs)
    events = []

    # Simple approach: find periods where drawdown goes below -5%
    in_drawdown = False
    peak_date = None
    peak_equity = None

    for i, (dt, dd) in enumerate(drawdown.items()):
        if not in_drawdown and dd < -0.01:
            # Start of drawdown
            in_drawdown = True
            # Find the peak (running max before this)
            peak_idx = running_max[:dt].idxmax() if len(running_max[:dt]) > 0 else dt
            peak_date = peak_idx
            peak_equity = running_max[peak_idx]

        elif in_drawdown and dd >= -0.001:
            # Recovery
            trough_idx = drawdown[peak_date:dt].idxmin()
            trough_equity = equity_curve[trough_idx]
            trough_dd = drawdown[trough_idx]

            # Get states during the drawdown
            states_during = []
            tc_during = 0.0
            for record in rebalance_records:
                if peak_date <= record.date <= dt:
                    states_during.append(record.state.name)
                    tc_during += record.transaction_cost

            # Get state at key dates
            state_at_start = _get_state_at_date(peak_date, rebalance_records)
            state_at_trough = _get_state_at_date(trough_idx, rebalance_records)

            events.append(DrawdownEvent(
                start_date=peak_date.strftime("%Y-%m-%d"),
                trough_date=trough_idx.strftime("%Y-%m-%d"),
                recovery_date=dt.strftime("%Y-%m-%d"),
                peak_equity=round(peak_equity, 4),
                trough_equity=round(trough_equity, 4),
                drawdown_pct=round(abs(trough_dd) * 100, 2),
                duration_to_trough_days=(trough_idx - peak_date).days,
                duration_to_recovery_days=(dt - peak_date).days,
                state_at_start=state_at_start,
                state_at_trough=state_at_trough,
                states_during=list(set(states_during)),
                tc_applied_during=round(tc_during, 6),
            ))

            in_drawdown = False

    # Handle ongoing drawdown at end
    if in_drawdown and peak_date is not None:
        trough_idx = drawdown[peak_date:].idxmin()
        trough_equity = equity_curve[trough_idx]
        trough_dd = drawdown[trough_idx]

        states_during = []
        tc_during = 0.0
        for record in rebalance_records:
            if peak_date <= record.date:
                states_during.append(record.state.name)
                tc_during += record.transaction_cost

        state_at_start = _get_state_at_date(peak_date, rebalance_records)
        state_at_trough = _get_state_at_date(trough_idx, rebalance_records)

        events.append(DrawdownEvent(
            start_date=peak_date.strftime("%Y-%m-%d"),
            trough_date=trough_idx.strftime("%Y-%m-%d"),
            recovery_date=None,  # Not recovered
            peak_equity=round(peak_equity, 4),
            trough_equity=round(trough_equity, 4),
            drawdown_pct=round(abs(trough_dd) * 100, 2),
            duration_to_trough_days=(trough_idx - peak_date).days,
            duration_to_recovery_days=None,
            state_at_start=state_at_start,
            state_at_trough=state_at_trough,
            states_during=list(set(states_during)),
            tc_applied_during=round(tc_during, 6),
        ))

    # Sort by drawdown severity and take top N
    events.sort(key=lambda e: e.drawdown_pct, reverse=True)
    return events[:top_n]


def _get_state_at_date(
    dt: pd.Timestamp,
    records: List[RebalanceRecord],
) -> str:
    """Get the state that was active at a given date."""
    # Find the most recent rebalance before or on this date
    active_state = "UNKNOWN"
    for record in records:
        if record.date <= dt:
            active_state = record.state.name
        else:
            break
    return active_state


def export_drawdown_attribution(
    equity_curve: pd.Series,
    rebalance_records: List[RebalanceRecord],
    output_dir: Path,
) -> List[DrawdownEvent]:
    """
    Export drawdown attribution report.

    Args:
        equity_curve: Daily portfolio equity curve
        rebalance_records: List of rebalance records
        output_dir: Output directory

    Returns:
        List of worst drawdown events
    """
    events = compute_drawdown_attribution(equity_curve, rebalance_records)

    if not events:
        logger.info("  No significant drawdown events found")
        return []

    # Export to CSV
    rows = [asdict(e) for e in events]
    for row in rows:
        row["states_during"] = ",".join(row["states_during"])

    df = pd.DataFrame(rows)
    filepath = output_dir / "drawdown_attribution.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} events)")

    # Log worst drawdown for visibility
    worst = events[0]
    logger.info(f"  Worst drawdown: {worst.drawdown_pct}% from {worst.start_date} to {worst.trough_date}")
    logger.info(f"    State at start: {worst.state_at_start}, at trough: {worst.state_at_trough}")
    logger.info(f"    TC applied during: {worst.tc_applied_during:.6f}")
    if worst.recovery_date:
        logger.info(f"    Recovered by: {worst.recovery_date}")
    else:
        logger.info(f"    NOT YET RECOVERED")

    return events


def export_holdings_snapshot(
    rebalance_records: List[RebalanceRecord],
    output_dir: Path,
) -> None:
    """
    Export detailed holdings at each rebalance.

    Columns: date, state, rank, ticker, weight, score, turnover, tc_cost
    """
    rows = []

    for record in rebalance_records:
        if not record.tickers:
            # No holdings - just record the state
            rows.append({
                "date": record.date.strftime("%Y-%m-%d"),
                "state": record.state.name,
                "rank": 0,
                "ticker": "",
                "weight": 0.0,
                "score": 0.0,
                "turnover": record.turnover,
                "tc_cost": record.transaction_cost,
            })
        else:
            # Sort tickers by weight descending
            sorted_tickers = sorted(
                record.tickers,
                key=lambda t: record.weights.get(t, 0),
                reverse=True
            )

            for rank, ticker in enumerate(sorted_tickers, 1):
                rows.append({
                    "date": record.date.strftime("%Y-%m-%d"),
                    "state": record.state.name,
                    "rank": rank,
                    "ticker": ticker,
                    "weight": round(record.weights.get(ticker, 0), 6),
                    "score": 0.0,  # Score not stored in RebalanceRecord currently
                    "turnover": record.turnover if rank == 1 else 0.0,
                    "tc_cost": record.transaction_cost if rank == 1 else 0.0,
                })

    df = pd.DataFrame(rows)
    filepath = output_dir / "holdings_snapshot.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} rows)")


def export_eligibility_coverage(
    eligibility_records: List,
    output_dir: Path,
) -> None:
    """
    Export eligibility coverage breakdown at each rebalance.

    Args:
        eligibility_records: List of EligibilityRecord from backtest
        output_dir: Output directory
    """
    rows = []
    for r in eligibility_records:
        rows.append({
            "date": r.date.strftime("%Y-%m-%d") if hasattr(r.date, "strftime") else str(r.date),
            "state": r.state.name if hasattr(r.state, "name") else str(r.state),
            "universe_count": r.universe_count,
            "data_eligible_count": r.data_eligible_count,
            "filter_passed_count": r.filter_passed_count,
            "selected_count": r.selected_count,
            "insufficient_history": r.insufficient_history,
            "negative_returns": r.negative_returns,
            "low_positive_months": r.low_positive_months,
            "high_drawdown": r.high_drawdown,
            "score_computation_failed": r.score_computation_failed,
            "not_in_price_data": r.not_in_price_data,
        })

    df = pd.DataFrame(rows)
    filepath = output_dir / "eligibility_coverage.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"  Wrote {filepath} ({len(df)} rows)")
