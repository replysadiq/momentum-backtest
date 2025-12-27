"""
Momentum signal computation and stock-level eligibility filters.

This module implements the zero-indicator (price-only) momentum signals:
- Stock-level eligibility filters based on returns
- Multi-period risk-adjusted momentum scoring

Uses the canonical stats module for all return/volatility computations.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
import logging

import numpy as np
import pandas as pd

from .stats import (
    total_return_months,
    realized_vol,
    max_drawdown,
    positive_month_percentage,
)


logger = logging.getLogger(__name__)


class EligibilityBreakdown(NamedTuple):
    """Breakdown of stock eligibility for a rebalance date."""
    universe_count: int
    data_eligible_count: int
    filter_passed_count: int
    insufficient_history: int
    negative_returns: int
    low_positive_months: int
    high_drawdown: int
    score_computation_failed: int
    not_in_price_data: int


class RankingResult(NamedTuple):
    """Result of ranking stocks, including eligibility breakdown."""
    scores: List["MomentumScore"]
    eligibility: EligibilityBreakdown


@dataclass
class MomentumScore:
    """Momentum score and components for a stock."""
    ticker: str
    score: float
    return_12m: float
    return_6m: float
    return_3m: float
    return_1m: float
    vol_12m: float
    vol_6m: float
    vol_3m: float
    vol_1m: float
    positive_month_pct: float
    max_drawdown_12m: Optional[float] = None


def apply_eligibility_filters(
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
    use_dd_filter: bool = False,
    dd_threshold: float = 0.40,
    disable_6m_filter: bool = False,  # V2 Lever D
) -> Tuple[bool, str]:
    """
    Apply stock-level eligibility filters.

    Filters (all must pass):
    1. 12-month return > 0
    2. 6-month return > 0 (can be disabled with Lever D)
    3. >= 60% positive months in last 12 months
    4. (Optional) 12-month max drawdown <= threshold

    Args:
        prices: Price series for the stock
        rebalance_date: The rebalance date (exclusive boundary)
        use_dd_filter: Whether to apply drawdown filter
        dd_threshold: Maximum allowed drawdown (default: 0.40)
        disable_6m_filter: V2 Lever D - skip 6M return > 0 filter

    Returns:
        Tuple of (passes_filters, reason)
    """
    # Filter 1: 12-month return > 0
    ret_12m = total_return_months(prices, rebalance_date, 12)
    if ret_12m is None:
        return False, "Insufficient data for 12M return"
    if ret_12m <= 0:
        return False, f"12M return <= 0 ({ret_12m:.2%})"

    # Filter 2: 6-month return > 0 (optional with V2 Lever D)
    if not disable_6m_filter:
        ret_6m = total_return_months(prices, rebalance_date, 6)
        if ret_6m is None:
            return False, "Insufficient data for 6M return"
        if ret_6m <= 0:
            return False, f"6M return <= 0 ({ret_6m:.2%})"

    # Filter 3: >= 60% positive months
    pos_pct = positive_month_percentage(prices, rebalance_date, 12)
    if pos_pct is None:
        return False, "Insufficient data for positive month calculation"
    if pos_pct < 0.60:
        return False, f"Positive month % < 60% ({pos_pct:.1%})"

    # Filter 4: (Optional) Max drawdown
    if use_dd_filter:
        max_dd = max_drawdown(prices, rebalance_date, 12)
        if max_dd is None:
            return False, "Insufficient data for drawdown calculation"
        if max_dd > dd_threshold:
            return False, f"Max drawdown > {dd_threshold:.0%} ({max_dd:.1%})"

    return True, ""


def compute_momentum_score(
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
) -> Optional[MomentumScore]:
    """
    Compute the risk-adjusted momentum score for a stock.

    Score formula:
        Score = 0.40 * (12M return / 12M vol)
              + 0.30 * (6M return / 6M vol)
              + 0.20 * (3M return / 3M vol)
              + 0.10 * (1M return / 1M vol)

    Args:
        prices: Price series for the stock
        rebalance_date: The rebalance date (exclusive boundary)

    Returns:
        MomentumScore with all components, or None if data insufficient
    """
    ticker = prices.name if prices.name else "UNKNOWN"

    # Compute returns for all periods using canonical stats module
    ret_12m = total_return_months(prices, rebalance_date, 12)
    ret_6m = total_return_months(prices, rebalance_date, 6)
    ret_3m = total_return_months(prices, rebalance_date, 3)
    ret_1m = total_return_months(prices, rebalance_date, 1)

    # Compute volatilities for all periods using canonical stats module
    vol_12m = realized_vol(prices, rebalance_date, 12)
    vol_6m = realized_vol(prices, rebalance_date, 6)
    vol_3m = realized_vol(prices, rebalance_date, 3)
    vol_1m = realized_vol(prices, rebalance_date, 1)

    # Check all required values are present
    if any(x is None for x in [ret_12m, ret_6m, ret_3m, ret_1m, vol_12m, vol_6m, vol_3m, vol_1m]):
        return None

    # Avoid division by zero
    if any(v <= 0 for v in [vol_12m, vol_6m, vol_3m, vol_1m]):
        return None

    # Compute risk-adjusted momentum for each period
    sharpe_12m = ret_12m / vol_12m
    sharpe_6m = ret_6m / vol_6m
    sharpe_3m = ret_3m / vol_3m
    sharpe_1m = ret_1m / vol_1m

    # Weighted score
    score = (
        0.40 * sharpe_12m +
        0.30 * sharpe_6m +
        0.20 * sharpe_3m +
        0.10 * sharpe_1m
    )

    # Positive month percentage using canonical stats module
    pos_pct = positive_month_percentage(prices, rebalance_date, 12)
    if pos_pct is None:
        pos_pct = 0.0

    # Max drawdown using canonical stats module
    max_dd = max_drawdown(prices, rebalance_date, 12)

    return MomentumScore(
        ticker=ticker,
        score=score,
        return_12m=ret_12m,
        return_6m=ret_6m,
        return_3m=ret_3m,
        return_1m=ret_1m,
        vol_12m=vol_12m,
        vol_6m=vol_6m,
        vol_3m=vol_3m,
        vol_1m=vol_1m,
        positive_month_pct=pos_pct,
        max_drawdown_12m=max_dd,
    )


class NSEMomentumRatio(NamedTuple):
    """Intermediate NSE momentum ratio values before Z-score normalization."""
    ticker: str
    mr6: float   # 6-month momentum ratio (return / vol)
    mr12: float  # 12-month momentum ratio (return / vol)
    ret_6m: float
    ret_12m: float
    vol_6m: float
    vol_12m: float
    pos_pct: float
    max_dd: Optional[float]


def compute_nse_momentum_ratios(
    prices: pd.Series,
    rebalance_date: pd.Timestamp,
) -> Optional[NSEMomentumRatio]:
    """
    Compute NSE-style momentum ratios (before Z-score normalization).

    NSE Formula:
        MR12 = 12-month price return / annualized volatility
        MR6  = 6-month price return / annualized volatility

    Args:
        prices: Price series for the stock
        rebalance_date: The rebalance date (exclusive boundary)

    Returns:
        NSEMomentumRatio with MR6 and MR12, or None if data insufficient
    """
    ticker = prices.name if prices.name else "UNKNOWN"

    # Compute returns (only 6M and 12M for NSE style)
    ret_12m = total_return_months(prices, rebalance_date, 12)
    ret_6m = total_return_months(prices, rebalance_date, 6)

    # NSE uses 1-year volatility for both ratios
    vol_12m = realized_vol(prices, rebalance_date, 12)
    vol_6m = realized_vol(prices, rebalance_date, 6)

    # Check required values
    if any(x is None for x in [ret_12m, ret_6m, vol_12m, vol_6m]):
        return None

    # Avoid division by zero - use 12M vol as the denominator (per NSE spec)
    if vol_12m <= 0:
        return None

    # NSE Momentum Ratios
    mr12 = ret_12m / vol_12m
    mr6 = ret_6m / vol_12m  # NSE uses same vol (1-year) for both

    # Additional metrics
    pos_pct = positive_month_percentage(prices, rebalance_date, 12)
    if pos_pct is None:
        pos_pct = 0.0
    max_dd = max_drawdown(prices, rebalance_date, 12)

    return NSEMomentumRatio(
        ticker=ticker,
        mr6=mr6,
        mr12=mr12,
        ret_6m=ret_6m,
        ret_12m=ret_12m,
        vol_6m=vol_6m,
        vol_12m=vol_12m,
        pos_pct=pos_pct,
        max_dd=max_dd,
    )


def compute_nse_normalized_score(
    mr6: float,
    mr12: float,
    universe_mr6: List[float],
    universe_mr12: List[float],
) -> float:
    """
    Compute NSE-style normalized momentum score using Z-scores.

    NSE Formula:
        1. Z12 = (MR12 - mean_MR12) / std_MR12
        2. Z6 = (MR6 - mean_MR6) / std_MR6
        3. Weighted_Z = 0.5 * Z12 + 0.5 * Z6
        4. Score = (1 + Z) if Z >= 0 else 1/(1-Z)

    Args:
        mr6: Stock's 6-month momentum ratio
        mr12: Stock's 12-month momentum ratio
        universe_mr6: All MR6 values in the eligible universe
        universe_mr12: All MR12 values in the eligible universe

    Returns:
        Normalized momentum score
    """
    # Compute Z-scores
    mean_mr12 = np.mean(universe_mr12)
    std_mr12 = np.std(universe_mr12, ddof=1)  # Sample std
    mean_mr6 = np.mean(universe_mr6)
    std_mr6 = np.std(universe_mr6, ddof=1)

    # Avoid division by zero
    if std_mr12 <= 0:
        std_mr12 = 1.0
    if std_mr6 <= 0:
        std_mr6 = 1.0

    z12 = (mr12 - mean_mr12) / std_mr12
    z6 = (mr6 - mean_mr6) / std_mr6

    # Weighted average Z-score (50% each per NSE spec)
    weighted_z = 0.5 * z12 + 0.5 * z6

    # NSE non-linear transform
    if weighted_z >= 0:
        score = 1.0 + weighted_z
    else:
        # (1 - weighted_z)^-1 = 1 / (1 - weighted_z)
        score = 1.0 / (1.0 - weighted_z)

    return score


def rank_stocks_by_momentum(
    price_data: pd.DataFrame,
    eligible_tickers: Set[str],
    rebalance_date: pd.Timestamp,
    universe_count: int = 0,
    use_dd_filter: bool = False,
    dd_threshold: float = 0.40,
    nse_style: bool = False,
    disable_6m_filter: bool = False,  # V2 Lever D
) -> RankingResult:
    """
    Rank eligible stocks by momentum score.

    Stock exclusions are EXPECTED behavior when:
    - Stock lacks sufficient history (12M lookback)
    - Stock fails momentum filters (negative returns, low positive months)

    This is NOT a bug - rebalance dates are fixed based on benchmark trading days.
    Stocks are simply excluded from that rebalance if ineligible.

    Args:
        price_data: DataFrame with stocks as columns
        eligible_tickers: Set of data-eligible tickers
        rebalance_date: The rebalance date
        universe_count: Total universe size (for reporting)
        use_dd_filter: Whether to apply drawdown filter
        dd_threshold: Maximum allowed drawdown
        nse_style: Use NSE NIFTY500 Momentum 50 scoring (Z-score normalized)
        disable_6m_filter: V2 Lever D - skip 6M return > 0 filter

    Returns:
        RankingResult with scores and eligibility breakdown
    """
    scores: List[MomentumScore] = []

    # Track exclusion reasons for summary logging
    insufficient_history = 0
    negative_returns = 0
    low_positive_months = 0
    high_drawdown = 0
    score_computation_failed = 0
    not_in_data = 0

    if nse_style:
        # NSE-style scoring: two-pass approach
        # Pass 1: Compute momentum ratios for all eligible stocks
        nse_ratios: List[NSEMomentumRatio] = []

        for ticker in eligible_tickers:
            if ticker not in price_data.columns:
                not_in_data += 1
                continue

            prices = price_data[ticker]
            prices.name = ticker

            # Apply eligibility filters
            passes, reason = apply_eligibility_filters(
                prices, rebalance_date, use_dd_filter, dd_threshold, disable_6m_filter
            )

            if not passes:
                if "Insufficient data" in reason:
                    insufficient_history += 1
                elif "return <= 0" in reason:
                    negative_returns += 1
                elif "Positive month" in reason:
                    low_positive_months += 1
                elif "drawdown" in reason.lower():
                    high_drawdown += 1
                logger.debug(f"[STOCK EXCLUDED] {ticker}: {reason}")
                continue

            # Compute NSE momentum ratios
            ratios = compute_nse_momentum_ratios(prices, rebalance_date)
            if ratios is not None:
                nse_ratios.append(ratios)
            else:
                score_computation_failed += 1
                logger.debug(f"[STOCK EXCLUDED] {ticker}: could not compute NSE momentum ratios")

        # Pass 2: Compute Z-scores and final scores using universe stats
        if len(nse_ratios) >= 2:  # Need at least 2 stocks for Z-score
            universe_mr6 = [r.mr6 for r in nse_ratios]
            universe_mr12 = [r.mr12 for r in nse_ratios]

            for ratios in nse_ratios:
                nse_score = compute_nse_normalized_score(
                    ratios.mr6, ratios.mr12, universe_mr6, universe_mr12
                )

                # Create MomentumScore with NSE score
                # Note: 3M and 1M returns/vols not used in NSE style, set to 0
                scores.append(MomentumScore(
                    ticker=ratios.ticker,
                    score=nse_score,
                    return_12m=ratios.ret_12m,
                    return_6m=ratios.ret_6m,
                    return_3m=0.0,
                    return_1m=0.0,
                    vol_12m=ratios.vol_12m,
                    vol_6m=ratios.vol_6m,
                    vol_3m=0.0,
                    vol_1m=0.0,
                    positive_month_pct=ratios.pos_pct,
                    max_drawdown_12m=ratios.max_dd,
                ))
        elif len(nse_ratios) == 1:
            # Single stock: can't compute Z-score, use raw score of 1.0
            ratios = nse_ratios[0]
            scores.append(MomentumScore(
                ticker=ratios.ticker,
                score=1.0,
                return_12m=ratios.ret_12m,
                return_6m=ratios.ret_6m,
                return_3m=0.0,
                return_1m=0.0,
                vol_12m=ratios.vol_12m,
                vol_6m=ratios.vol_6m,
                vol_3m=0.0,
                vol_1m=0.0,
                positive_month_pct=ratios.pos_pct,
                max_drawdown_12m=ratios.max_dd,
            ))

    else:
        # Original scoring approach
        for ticker in eligible_tickers:
            if ticker not in price_data.columns:
                not_in_data += 1
                continue

            prices = price_data[ticker]
            prices.name = ticker

            # Apply eligibility filters
            passes, reason = apply_eligibility_filters(
                prices, rebalance_date, use_dd_filter, dd_threshold, disable_6m_filter
            )

            if not passes:
                # Categorize reason
                if "Insufficient data" in reason:
                    insufficient_history += 1
                elif "return <= 0" in reason:
                    negative_returns += 1
                elif "Positive month" in reason:
                    low_positive_months += 1
                elif "drawdown" in reason.lower():
                    high_drawdown += 1
                logger.debug(f"[STOCK EXCLUDED] {ticker}: {reason}")
                continue

            # Compute momentum score
            score = compute_momentum_score(prices, rebalance_date)

            if score is not None:
                scores.append(score)
            else:
                score_computation_failed += 1
                logger.debug(f"[STOCK EXCLUDED] {ticker}: could not compute momentum score")

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)

    # Build eligibility breakdown
    eligibility = EligibilityBreakdown(
        universe_count=universe_count if universe_count > 0 else len(eligible_tickers),
        data_eligible_count=len(eligible_tickers),
        filter_passed_count=len(scores),
        insufficient_history=insufficient_history,
        negative_returns=negative_returns,
        low_positive_months=low_positive_months,
        high_drawdown=high_drawdown,
        score_computation_failed=score_computation_failed,
        not_in_price_data=not_in_data,
    )

    # Log summary of exclusions (at INFO level for visibility)
    total_excluded = len(eligible_tickers) - len(scores)
    if total_excluded > 0:
        summary_parts = []
        if not_in_data > 0:
            summary_parts.append(f"not_in_price_data={not_in_data}")
        if insufficient_history > 0:
            summary_parts.append(f"insufficient_history={insufficient_history}")
        if negative_returns > 0:
            summary_parts.append(f"negative_returns={negative_returns}")
        if low_positive_months > 0:
            summary_parts.append(f"low_positive_months={low_positive_months}")
        if high_drawdown > 0:
            summary_parts.append(f"high_drawdown={high_drawdown}")
        if score_computation_failed > 0:
            summary_parts.append(f"score_computation_failed={score_computation_failed}")

        logger.info(
            f"[{rebalance_date.date()}] Stock eligibility: {len(scores)} passed, "
            f"{total_excluded} excluded ({', '.join(summary_parts)})"
        )

    logger.debug(f"Ranked {len(scores)} stocks by momentum score")

    return RankingResult(scores=scores, eligibility=eligibility)


# =============================================================================
# V2 Lever A: Defensive Basket Selection
# =============================================================================


class DefensiveScore(NamedTuple):
    """Score for defensive basket selection (V2 Lever A)."""
    ticker: str
    vol_6m: float
    return_12m: float


def select_defensive_basket(
    price_data: pd.DataFrame,
    eligible_tickers: Set[str],
    rebalance_date: pd.Timestamp,
    basket_size: int = 15,
) -> List[DefensiveScore]:
    """
    Select lowest-volatility stocks for defensive basket (V2 Lever A).

    Used in DEFENSIVE_MOMENTUM state to hold low-vol stocks instead of cash.

    Criteria:
    - 12M return > 0 (positive momentum)
    - Ranked by LOWEST 6M volatility
    - Top `basket_size` stocks selected

    Args:
        price_data: Stock price DataFrame
        eligible_tickers: Set of data-eligible tickers
        rebalance_date: Rebalance date
        basket_size: Number of stocks to select (default: 15)

    Returns:
        List of DefensiveScore sorted by vol_6m ascending (lowest vol first)
    """
    candidates: List[DefensiveScore] = []

    for ticker in eligible_tickers:
        if ticker not in price_data.columns:
            continue

        prices = price_data[ticker]
        prices.name = ticker

        # Only require 12M return > 0 (no 6M filter for defensive)
        ret_12m = total_return_months(prices, rebalance_date, 12)
        if ret_12m is None or ret_12m <= 0:
            continue

        # Get 6M volatility for ranking
        vol_6m = realized_vol(prices, rebalance_date, 6)
        if vol_6m is None or vol_6m <= 0:
            continue

        candidates.append(DefensiveScore(
            ticker=ticker,
            vol_6m=vol_6m,
            return_12m=ret_12m,
        ))

    # Sort by volatility ASCENDING (lowest vol first)
    candidates.sort(key=lambda x: x.vol_6m)

    selected = candidates[:basket_size]

    if selected:
        logger.info(
            f"Defensive basket: {len(selected)} stocks selected "
            f"(vol range: {selected[0].vol_6m:.2%} to {selected[-1].vol_6m:.2%})"
        )
    else:
        logger.warning("Defensive basket: no eligible stocks found")

    return selected
