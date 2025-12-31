"""
Portfolio construction: stock selection and weighting.

This module handles:
- Top-N stock selection by momentum score
- Inverse-volatility weighting
- Optional weight capping
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

from .signals import MomentumScore

if TYPE_CHECKING:
    from .signals import DefensiveScore


logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Portfolio allocation for a rebalance date."""
    date: pd.Timestamp
    tickers: List[str]
    weights: Dict[str, float]
    scores: Dict[str, float]
    turnover: float = 0.0


# =============================================================================
# V2 Lever B: Rank Buffer / Minimum Hold
# =============================================================================


@dataclass
class HoldingInfo:
    """Tracking information for a held stock (V2 Lever B)."""
    ticker: str
    entry_date: pd.Timestamp
    entry_rank: int
    months_held: int = 0


def select_top_stocks(
    ranked_scores: List[MomentumScore],
    top_n: int = 20,
) -> List[MomentumScore]:
    """
    Select top N stocks by momentum score.

    Args:
        ranked_scores: List of MomentumScore sorted by score descending
        top_n: Number of stocks to select

    Returns:
        Top N MomentumScore objects
    """
    selected = ranked_scores[:top_n]

    if len(selected) < top_n:
        logger.warning(
            f"Only {len(selected)} stocks available, requested {top_n}"
        )

    return selected


def select_top_stocks_with_buffer(
    ranked_scores: List[MomentumScore],
    top_n: int,
    current_holdings: Dict[str, HoldingInfo],
    rebalance_date: pd.Timestamp,
    rank_buffer: float = 0.0,
    min_hold_months: int = 0,
) -> Tuple[List[MomentumScore], Dict[str, HoldingInfo]]:
    """
    Select top N stocks with rank buffer and minimum hold constraints (V2 Lever B).

    Rank Buffer Logic:
    - Current holding at rank R is only replaced if new candidate is at rank <= R * (1 - buffer)
    - Example: buffer=0.20, current at rank 15 -> only replaced if candidate at rank 12 or better

    Minimum Hold Logic:
    - Stock held < min_hold_months cannot be sold regardless of rank

    Args:
        ranked_scores: All eligible stocks sorted by score descending
        top_n: Target portfolio size
        current_holdings: Dict of ticker -> HoldingInfo for current holdings
        rebalance_date: Current rebalance date
        rank_buffer: Relative rank improvement required (0.0 to disable)
        min_hold_months: Minimum holding period (0 to disable)

    Returns:
        Tuple of (selected MomentumScores, updated HoldingInfo dict)
    """
    # Build rank lookup: ticker -> rank (1-indexed)
    rank_lookup = {score.ticker: idx + 1 for idx, score in enumerate(ranked_scores)}
    score_lookup = {score.ticker: score for score in ranked_scores}

    # Separate current holdings into "must keep" and "can evaluate"
    must_keep: List[str] = []
    can_evaluate: List[str] = []

    for ticker, info in current_holdings.items():
        if ticker not in rank_lookup:
            # Stock no longer eligible (failed filters) - must sell
            logger.debug(f"Rank buffer: {ticker} no longer eligible, will be replaced")
            continue

        # Check minimum hold period
        if min_hold_months > 0 and info.months_held < min_hold_months:
            must_keep.append(ticker)
            logger.debug(f"Rank buffer: {ticker} protected by min hold ({info.months_held}/{min_hold_months} months)")
            continue

        can_evaluate.append(ticker)

    # Track which current holdings to keep based on rank buffer
    keep_from_current: List[str] = list(must_keep)

    # Evaluate each current holding against the rank buffer
    for ticker in can_evaluate:
        current_rank = rank_lookup[ticker]
        threshold_rank = current_rank * (1 - rank_buffer)

        # Check if there's a new stock that's sufficiently better
        better_exists = False
        for new_rank, score in enumerate(ranked_scores, 1):
            if new_rank >= current_rank:
                # No better candidates found
                break
            if score.ticker not in current_holdings:
                # Found a new stock that's better than threshold
                if new_rank <= threshold_rank:
                    better_exists = True
                    logger.debug(
                        f"Rank buffer: {ticker} (rank {current_rank}) will be replaced - "
                        f"better candidate at rank {new_rank} <= threshold {threshold_rank:.0f}"
                    )
                    break

        if not better_exists:
            keep_from_current.append(ticker)
            logger.debug(
                f"Rank buffer: {ticker} (rank {current_rank}) protected - "
                f"no new candidate better than threshold {threshold_rank:.0f}"
            )

    # Build final selection: start with protected holdings, fill rest from top ranks
    selected_tickers: List[str] = list(keep_from_current)
    slots_remaining = top_n - len(selected_tickers)

    # Fill remaining slots from top-ranked stocks
    for score in ranked_scores:
        if slots_remaining <= 0:
            break
        if score.ticker not in selected_tickers:
            selected_tickers.append(score.ticker)
            slots_remaining -= 1

    # Build output list of MomentumScores (in score-descending order)
    selected_scores = [score_lookup[t] for t in selected_tickers if t in score_lookup]
    selected_scores.sort(key=lambda x: x.score, reverse=True)

    # Update holding info
    updated_holdings: Dict[str, HoldingInfo] = {}
    for score in selected_scores:
        ticker = score.ticker
        if ticker in current_holdings:
            # Increment months held
            info = current_holdings[ticker]
            updated_holdings[ticker] = HoldingInfo(
                ticker=ticker,
                entry_date=info.entry_date,
                entry_rank=info.entry_rank,
                months_held=info.months_held + 1,
            )
        else:
            # New holding
            updated_holdings[ticker] = HoldingInfo(
                ticker=ticker,
                entry_date=rebalance_date,
                entry_rank=rank_lookup.get(ticker, 0),
                months_held=0,
            )

    return selected_scores, updated_holdings


def compute_inverse_vol_weights(
    selected_stocks: List[MomentumScore],
    max_weight: Optional[float] = 0.05,
) -> Dict[str, float]:
    """
    Compute inverse-volatility weights for selected stocks.

    Weight for stock i = (1/vol_i) / sum(1/vol_j for all j)

    Args:
        selected_stocks: List of MomentumScore for selected stocks
        max_weight: Maximum weight per stock (None to disable cap)

    Returns:
        Dictionary mapping ticker to weight (may sum to < 1.0 if capped)
    """
    if not selected_stocks:
        return {}

    # Use 6-month volatility for weighting
    inverse_vols = {}
    for stock in selected_stocks:
        if stock.vol_6m > 0:
            inverse_vols[stock.ticker] = 1.0 / stock.vol_6m

    if not inverse_vols:
        # Fallback to equal weight if all vols are zero
        equal_weight = 1.0 / len(selected_stocks)
        weights = {s.ticker: equal_weight for s in selected_stocks}
        if max_weight is not None:
            weights = _apply_weight_cap(weights, max_weight)
        return weights

    # Normalize to sum to 1
    total_inv_vol = sum(inverse_vols.values())
    weights = {ticker: iv / total_inv_vol for ticker, iv in inverse_vols.items()}

    # Apply weight cap if specified (leave remainder as cash)
    if max_weight is not None:
        weights = _apply_weight_cap(weights, max_weight)

    return weights


def _apply_weight_cap(
    weights: Dict[str, float],
    max_weight: float,
) -> Dict[str, float]:
    """
    Apply maximum weight cap without redistribution.

    Args:
        weights: Initial weights (should sum to 1)
        max_weight: Maximum allowed weight

    Returns:
        Capped weights (may sum to < 1.0)
    """
    if max_weight >= 1.0:
        return weights

    return {ticker: min(weight, max_weight) for ticker, weight in weights.items()}


def compute_turnover(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> float:
    """
    Compute one-way portfolio turnover.

    Turnover = sum of absolute weight changes / 2

    Args:
        old_weights: Previous portfolio weights
        new_weights: New portfolio weights

    Returns:
        One-way turnover (0.0 to 1.0)
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())

    total_change = 0.0
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_change += abs(new_w - old_w)

    # One-way turnover is half of total absolute change
    return total_change / 2.0


def build_portfolio(
    ranked_scores: List[MomentumScore],
    top_n: int = 20,
    max_weight: Optional[float] = 0.05,
    previous_weights: Optional[Dict[str, float]] = None,
    rebalance_date: Optional[pd.Timestamp] = None,
    # V2 Lever B parameters
    current_holdings: Optional[Dict[str, HoldingInfo]] = None,
    rank_buffer: float = 0.0,
    min_hold_months: int = 0,
) -> Tuple[PortfolioAllocation, Dict[str, HoldingInfo]]:
    """
    Build a portfolio from ranked momentum scores.

    Args:
        ranked_scores: List of MomentumScore sorted by score descending
        top_n: Number of stocks to select
        max_weight: Maximum weight per stock (None to disable)
        previous_weights: Previous portfolio weights for turnover calculation
        rebalance_date: Rebalance date for the allocation
        current_holdings: V2 Lever B - current holdings for rank buffer
        rank_buffer: V2 Lever B - rank improvement required (0.0 to disable)
        min_hold_months: V2 Lever B - minimum months to hold (0 to disable)

    Returns:
        Tuple of (PortfolioAllocation, updated HoldingInfo dict)
    """
    # If rank buffer or min hold enabled, use buffered selection
    if (rank_buffer > 0 or min_hold_months > 0) and current_holdings is not None:
        selected, updated_holdings = select_top_stocks_with_buffer(
            ranked_scores,
            top_n,
            current_holdings,
            rebalance_date,
            rank_buffer,
            min_hold_months,
        )
    else:
        # Standard selection
        selected = select_top_stocks(ranked_scores, top_n)
        # Create new holding info for all selected
        updated_holdings = {
            s.ticker: HoldingInfo(
                ticker=s.ticker,
                entry_date=rebalance_date,
                entry_rank=idx + 1,
                months_held=0,
            )
            for idx, s in enumerate(selected)
        }

    if not selected:
        logger.warning("No stocks selected for portfolio")
        return PortfolioAllocation(
            date=rebalance_date,
            tickers=[],
            weights={},
            scores={},
            turnover=1.0 if previous_weights else 0.0,
        ), {}

    # Compute weights
    weights = compute_inverse_vol_weights(selected, max_weight)

    # Compute turnover
    turnover = 0.0
    if previous_weights is not None:
        turnover = compute_turnover(previous_weights, weights)

    # Build allocation
    allocation = PortfolioAllocation(
        date=rebalance_date,
        tickers=[s.ticker for s in selected],
        weights=weights,
        scores={s.ticker: s.score for s in selected},
        turnover=turnover,
    )

    if weights:
        max_w = max(weights.values())
        invested_frac = sum(weights.values())
    else:
        max_w = 0.0
        invested_frac = 0.0

    logger.debug(
        f"Portfolio built: {len(allocation.tickers)} stocks, "
        f"max_weight={max_w:.2%}, invested={invested_frac:.2%}, turnover={turnover:.2%}"
    )

    return allocation, updated_holdings


# =============================================================================
# V2 Lever A: Defensive Portfolio Builder
# =============================================================================


def build_defensive_portfolio(
    defensive_scores: List["DefensiveScore"],
    max_weight: Optional[float] = 0.05,
    previous_weights: Optional[Dict[str, float]] = None,
    rebalance_date: Optional[pd.Timestamp] = None,
) -> PortfolioAllocation:
    """
    Build defensive portfolio weighted by inverse volatility (V2 Lever A).

    Uses the same inverse-vol weighting as regular portfolios.

    Args:
        defensive_scores: List of DefensiveScore from select_defensive_basket
        max_weight: Maximum weight per stock
        previous_weights: Previous weights for turnover calculation
        rebalance_date: Rebalance date

    Returns:
        PortfolioAllocation with defensive basket
    """
    if not defensive_scores:
        return PortfolioAllocation(
            date=rebalance_date,
            tickers=[],
            weights={},
            scores={},
            turnover=1.0 if previous_weights else 0.0,
        )

    # Compute inverse-vol weights
    inverse_vols = {s.ticker: 1.0 / s.vol_6m for s in defensive_scores if s.vol_6m > 0}

    if not inverse_vols:
        # Fallback to equal weight
        equal_weight = 1.0 / len(defensive_scores)
        weights = {s.ticker: equal_weight for s in defensive_scores}
    else:
        total_inv_vol = sum(inverse_vols.values())
        weights = {t: iv / total_inv_vol for t, iv in inverse_vols.items()}

    # Apply weight cap
    if max_weight is not None:
        weights = _apply_weight_cap(weights, max_weight)

    # Compute turnover
    turnover = 0.0
    if previous_weights is not None:
        turnover = compute_turnover(previous_weights, weights)

    return PortfolioAllocation(
        date=rebalance_date,
        tickers=[s.ticker for s in defensive_scores if s.ticker in weights],
        weights=weights,
        scores={s.ticker: -s.vol_6m for s in defensive_scores},  # Negative vol as "score"
        turnover=turnover,
    )
