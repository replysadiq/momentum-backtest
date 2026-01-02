"""
Momentum signal computation and stock-level eligibility filters.

This module implements the zero-indicator (price-only) momentum signals:
- Stock-level eligibility filters based on returns
- Multi-period risk-adjusted momentum scoring

Uses the canonical stats module for all return/volatility computations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
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


SCORE_FAILURE_REASON_CODES = [
    "insufficient_history",
    "nan_in_price",
    "nan_in_returns",
    "nan_in_momentum_window",
    "nan_in_vol_window",
    "zero_or_nan_vol",
    "non_finite_score",
    "exception_in_score_calc",
    "date_alignment_missing",
]

REJECTION_REASON_CODES = [
    "SCORE_OK",
    "REJ_EMPTY_WINDOW",
    "REJ_TOO_SHORT_WINDOW",
    "REJ_NAN_IN_WINDOW",
    "REJ_NAN_SCORE",
    "REJ_INF_SCORE",
    "REJ_VOL_ZERO_OR_NEG",
    "REJ_DIV_BY_ZERO",
    "REJ_EXCEPTION",
    "REJ_POSTFILTER_NEG_RET",
    "REJ_POSTFILTER_LOW_POS_MONTHS",
    "REJ_OTHER",
]


class RankingResult(NamedTuple):
    """Result of ranking stocks, including eligibility breakdown."""
    scores: List["MomentumScore"]
    eligibility: EligibilityBreakdown
    score_failure_breakdown: Dict[str, int]
    score_failure_samples: Dict[str, List[str]]
    score_failure_notes: Dict[str, str]
    rejection_attempted: int
    rejection_ok: int
    rejection_counts: Dict[str, int]
    rejection_samples: Dict[str, List[str]]
    rejection_exception_hist: Dict[str, int]
    rejection_sample_contexts: Dict[str, List[str]]
    short_window_samples: List[Dict[str, Any]]


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
    trading_dates: Optional[pd.DatetimeIndex] = None,
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
    vol_12m = realized_vol(prices, rebalance_date, 12, trading_dates=trading_dates)
    vol_6m = realized_vol(prices, rebalance_date, 6, trading_dates=trading_dates)
    vol_3m = realized_vol(prices, rebalance_date, 3, trading_dates=trading_dates)
    vol_1m = realized_vol(prices, rebalance_date, 1, trading_dates=trading_dates)

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
    trading_dates: Optional[pd.DatetimeIndex] = None,
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
    vol_12m = realized_vol(prices, rebalance_date, 12, trading_dates=trading_dates)
    vol_6m = realized_vol(prices, rebalance_date, 6, trading_dates=trading_dates)

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
    trading_dates: Optional[pd.DatetimeIndex] = None,
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
        nse_style: Use Momentum 50 scoring (Z-score normalized)
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
    score_failure_breakdown: Dict[str, int] = {k: 0 for k in SCORE_FAILURE_REASON_CODES}
    score_failure_samples: Dict[str, List[str]] = {k: [] for k in SCORE_FAILURE_REASON_CODES}
    score_failure_notes: Dict[str, set] = {k: set() for k in SCORE_FAILURE_REASON_CODES}
    rejection_counts: Dict[str, int] = {k: 0 for k in REJECTION_REASON_CODES}
    rejection_samples: Dict[str, List[str]] = {k: [] for k in REJECTION_REASON_CODES}
    rejection_exception_hist: Dict[str, int] = {}
    rejection_sample_contexts: Dict[str, List[str]] = {k: [] for k in REJECTION_REASON_CODES}
    short_window_samples: List[Dict[str, Any]] = []
    rejection_attempted = len(eligible_tickers)
    short_window_sample_limit = 200

    def _record_score_failure(reason: str, ticker: str, note: Optional[str] = None) -> None:
        nonlocal score_computation_failed
        if reason not in score_failure_breakdown:
            score_failure_breakdown[reason] = 0
            score_failure_samples[reason] = []
            score_failure_notes[reason] = set()
        score_computation_failed += 1
        score_failure_breakdown[reason] += 1
        if len(score_failure_samples[reason]) < 10:
            score_failure_samples[reason].append(ticker)
        if note:
            score_failure_notes[reason].add(note)

    def _record_rejection(
        reason: str,
        ticker: str,
        exc: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        if reason not in rejection_counts:
            rejection_counts[reason] = 0
            rejection_samples[reason] = []
            rejection_sample_contexts[reason] = []
        rejection_counts[reason] += 1
        if len(rejection_samples[reason]) < 5:
            rejection_samples[reason].append(ticker)
        if context and len(rejection_sample_contexts[reason]) < 10:
            rejection_sample_contexts[reason].append(f"{ticker}:{context}")
        if reason == "REJ_EXCEPTION" and exc:
            rejection_exception_hist[exc] = rejection_exception_hist.get(exc, 0) + 1

    def _record_short_window(sample: Dict[str, Any]) -> None:
        if len(short_window_samples) >= short_window_sample_limit:
            return
        short_window_samples.append(sample)

    def _short_window_context(
        prices: pd.Series,
        window: pd.Series,
        window_label: str,
        required_len: int,
        slicing_mode: str,
        note: str = "",
        monthly_points: Optional[int] = None,
        required_monthly_points: Optional[int] = None,
        missing_components: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        last_avail = prices[prices.index < rebalance_date].dropna()
        earliest = prices.dropna()
        rebalance_idx = prices.index.get_indexer([rebalance_date])[0]
        return {
            "symbol": prices.name if prices.name else "UNKNOWN",
            "rebalance_date": rebalance_date,
            "required_len": int(required_len),
            "actual_len": int(len(window)),
            "window_start_date": window.index.min() if len(window) > 0 else None,
            "window_end_date": window.index.max() if len(window) > 0 else None,
            "last_available_price_date": last_avail.index.max() if len(last_avail) > 0 else None,
            "earliest_available_price_date": earliest.index.min() if len(earliest) > 0 else None,
            "index_of_rebalance_in_series": int(rebalance_idx),
            "window_label": window_label,
            "slicing_mode": slicing_mode,
            "note": note,
            "monthly_points": int(monthly_points) if monthly_points is not None else None,
            "required_monthly_points": int(required_monthly_points) if required_monthly_points is not None else None,
            "missing_components": missing_components,
        }

    def _record_short_window_for_reason(prices: pd.Series, reason: str) -> None:
        reason_l = reason.lower()
        window_label = ""
        months = None
        required_len = 2
        slicing_mode = "price_mask"
        monthly_points = None
        required_monthly_points = None

        if "12m return" in reason_l:
            window_label = "ret_12m_window"
            months = 12
        elif "6m return" in reason_l:
            window_label = "ret_6m_window"
            months = 6
        elif "positive month" in reason_l:
            window_label = "pos_months_12m_window"
            months = 12
            required_len = 2
            slicing_mode = "monthly_resample"
        elif "drawdown" in reason_l:
            window_label = "drawdown_12m_window"
            months = 12
            required_len = 5

        if months is None:
            return

        start = rebalance_date - pd.DateOffset(months=months)
        mask = (prices.index >= start) & (prices.index < rebalance_date)
        window = prices[mask].dropna()

        if slicing_mode == "monthly_resample":
            monthly_window = window.resample("ME").last().dropna()
            monthly_points = monthly_window.shape[0]
            required_monthly_points = 2
            window = monthly_window

        sample = _short_window_context(
            prices=prices,
            window=window,
            window_label=window_label,
            required_len=required_len,
            slicing_mode=slicing_mode,
            note=reason,
            monthly_points=monthly_points,
            required_monthly_points=required_monthly_points,
        )
        _record_short_window(sample)

    def _required_trading_samples(months: int, fallback: int) -> int:
        if trading_dates is None:
            return fallback
        start = rebalance_date - pd.DateOffset(months=months)
        return int(((trading_dates >= start) & (trading_dates < rebalance_date)).sum())

    def _classify_rejection(prices: pd.Series) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        if rebalance_date not in prices.index:
            return "REJ_EMPTY_WINDOW", None, None

        def _window(months: int) -> pd.Series:
            start = rebalance_date - pd.DateOffset(months=months)
            mask = (prices.index >= start) & (prices.index < rebalance_date)
            return prices[mask]

        window_12m = _window(12)
        if window_12m.empty:
            return "REJ_EMPTY_WINDOW", None, None
        if len(window_12m) < 2:
            return "REJ_TOO_SHORT_WINDOW", None, _short_window_context(
                prices=prices,
                window=window_12m,
                window_label="ret_12m_window",
                required_len=2,
                slicing_mode="price_mask",
            )
        if window_12m.isna().any():
            return "REJ_NAN_IN_WINDOW", None, None

        returns_12m = window_12m.pct_change().iloc[1:]
        if returns_12m.empty:
            return "REJ_EMPTY_WINDOW", None, None
        if returns_12m.isna().any():
            return "REJ_NAN_IN_WINDOW", None, None

        for m in [6, 3, 1]:
            window = _window(m)
            if window.empty or len(window) < 2:
                return "REJ_TOO_SHORT_WINDOW", None, _short_window_context(
                    prices=prices,
                    window=window,
                    window_label=f"ret_{m}m_window",
                    required_len=2,
                    slicing_mode="price_mask",
                )
            if window.isna().any() or window.pct_change().iloc[1:].isna().any():
                return "REJ_NAN_IN_WINDOW", None, None

        for m in [12, 6]:
            window = _window(m)
            if window.empty or len(window) < 2:
                return "REJ_TOO_SHORT_WINDOW", None, _short_window_context(
                    prices=prices,
                    window=window,
                    window_label=f"vol_{m}m_window",
                    required_len=_required_trading_samples(m, 20),
                    slicing_mode="price_mask",
                )
            if window.isna().any():
                return "REJ_NAN_IN_WINDOW", None, None

        ret_12m = total_return_months(prices, rebalance_date, 12)
        ret_6m = total_return_months(prices, rebalance_date, 6)
        ret_3m = total_return_months(prices, rebalance_date, 3)
        ret_1m = total_return_months(prices, rebalance_date, 1)
        vol_12m = realized_vol(prices, rebalance_date, 12, trading_dates=trading_dates)
        vol_6m = realized_vol(prices, rebalance_date, 6, trading_dates=trading_dates)
        vol_3m = realized_vol(prices, rebalance_date, 3, trading_dates=trading_dates)
        vol_1m = realized_vol(prices, rebalance_date, 1, trading_dates=trading_dates)

        if any(x is None for x in [ret_12m, ret_6m, ret_3m, ret_1m, vol_12m, vol_6m, vol_3m, vol_1m]):
            missing_components = []
            first_window = window_12m
            for label, value, months, required_len in [
                ("ret_12m", ret_12m, 12, 2),
                ("ret_6m", ret_6m, 6, 2),
                ("ret_3m", ret_3m, 3, 2),
                ("ret_1m", ret_1m, 1, 2),
                ("vol_12m", vol_12m, 12, _required_trading_samples(12, 20)),
                ("vol_6m", vol_6m, 6, _required_trading_samples(6, 20)),
                ("vol_3m", vol_3m, 3, _required_trading_samples(3, 20)),
                ("vol_1m", vol_1m, 1, _required_trading_samples(1, 20)),
            ]:
                if value is None:
                    window = _window(months)
                    if not missing_components:
                        first_window = window
                    missing_components.append({
                        "component": label,
                        "required_len": required_len,
                        "actual_len": int(len(window)),
                    })

            first = missing_components[0] if missing_components else {"required_len": 2, "actual_len": 0}
            return "REJ_TOO_SHORT_WINDOW", None, _short_window_context(
                prices=prices,
                window=first_window,
                window_label="ret_or_vol_window",
                required_len=first["required_len"],
                slicing_mode="price_mask",
                note="ret_or_vol_none",
                missing_components=missing_components,
            )

        if any(v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)))
               for v in [vol_12m, vol_6m, vol_3m, vol_1m]):
            return "REJ_VOL_ZERO_OR_NEG", None, None

        if any(v == 0 for v in [vol_12m, vol_6m, vol_3m, vol_1m]):
            return "REJ_DIV_BY_ZERO", None, None
        if any(v < 0 for v in [vol_12m, vol_6m, vol_3m, vol_1m]):
            return "REJ_VOL_ZERO_OR_NEG", None, None

        return "REJ_OTHER", None, None

    if nse_style:
        # NSE-style scoring: two-pass approach
        # Pass 1: Compute momentum ratios for all eligible stocks
        nse_ratios: List[NSEMomentumRatio] = []

        for ticker in eligible_tickers:
            if ticker not in price_data.columns:
                not_in_data += 1
                _record_rejection("REJ_OTHER", ticker, context="not_in_data")
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
                    _record_rejection("REJ_TOO_SHORT_WINDOW", ticker, context=reason)
                    _record_short_window_for_reason(prices, reason)
                elif "return <= 0" in reason:
                    negative_returns += 1
                    _record_rejection("REJ_POSTFILTER_NEG_RET", ticker)
                elif "Positive month" in reason:
                    low_positive_months += 1
                    _record_rejection("REJ_POSTFILTER_LOW_POS_MONTHS", ticker)
                elif "drawdown" in reason.lower():
                    high_drawdown += 1
                    _record_rejection("REJ_OTHER", ticker, context="drawdown")
                logger.debug(f"[STOCK EXCLUDED] {ticker}: {reason}")
                continue

            # Compute NSE momentum ratios
            try:
                ratios = compute_nse_momentum_ratios(
                    prices,
                    rebalance_date,
                    trading_dates=trading_dates,
                )
            except Exception as exc:
                _record_score_failure("exception_in_score_calc", ticker, type(exc).__name__)
                _record_rejection("REJ_EXCEPTION", ticker, type(exc).__name__)
                logger.debug(
                    f"[STOCK EXCLUDED] {ticker}: exception in NSE momentum ratios ({type(exc).__name__})"
                )
                continue
            if ratios is not None:
                nse_ratios.append(ratios)
            else:
                reason, note, ctx = _classify_rejection(prices)
                if reason == "REJ_TOO_SHORT_WINDOW":
                    insufficient_history += 1
                if reason.startswith("REJ_"):
                    context = "classify_rejection" if reason == "REJ_OTHER" else None
                    _record_rejection(reason, ticker, context=context)
                if reason == "REJ_TOO_SHORT_WINDOW" and ctx:
                    _record_short_window(ctx)
                logger.debug(f"[STOCK EXCLUDED] {ticker}: could not compute NSE momentum ratios")

        # Pass 2: Compute Z-scores and final scores using universe stats
        if len(nse_ratios) >= 2:  # Need at least 2 stocks for Z-score
            universe_mr6 = [r.mr6 for r in nse_ratios]
            universe_mr12 = [r.mr12 for r in nse_ratios]

            for ratios in nse_ratios:
                nse_score = compute_nse_normalized_score(
                    ratios.mr6, ratios.mr12, universe_mr6, universe_mr12
                )
                if not np.isfinite(nse_score):
                    code = "REJ_INF_SCORE" if np.isinf(nse_score) else "REJ_NAN_SCORE"
                    _record_rejection(code, ratios.ticker)
                    continue

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
                _record_rejection("SCORE_OK", ratios.ticker)
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
            _record_rejection("SCORE_OK", ratios.ticker)

    else:
        # Original scoring approach
        for ticker in eligible_tickers:
            if ticker not in price_data.columns:
                not_in_data += 1
                _record_rejection("REJ_OTHER", ticker, context="not_in_data")
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
                    _record_rejection("REJ_TOO_SHORT_WINDOW", ticker, context=reason)
                    _record_short_window_for_reason(prices, reason)
                elif "return <= 0" in reason:
                    negative_returns += 1
                    _record_rejection("REJ_POSTFILTER_NEG_RET", ticker)
                elif "Positive month" in reason:
                    low_positive_months += 1
                    _record_rejection("REJ_POSTFILTER_LOW_POS_MONTHS", ticker)
                elif "drawdown" in reason.lower():
                    high_drawdown += 1
                    _record_rejection("REJ_OTHER", ticker, context="drawdown")
                logger.debug(f"[STOCK EXCLUDED] {ticker}: {reason}")
                continue

            # Compute momentum score
            try:
                score = compute_momentum_score(
                    prices,
                    rebalance_date,
                    trading_dates=trading_dates,
                )
            except Exception as exc:
                _record_score_failure("exception_in_score_calc", ticker, type(exc).__name__)
                _record_rejection("REJ_EXCEPTION", ticker, type(exc).__name__)
                logger.debug(
                    f"[STOCK EXCLUDED] {ticker}: exception in momentum score ({type(exc).__name__})"
                )
                continue

            if score is not None:
                if not np.isfinite(score.score):
                    code = "REJ_INF_SCORE" if np.isinf(score.score) else "REJ_NAN_SCORE"
                    _record_rejection(code, ticker)
                else:
                    scores.append(score)
                    _record_rejection("SCORE_OK", ticker)
            else:
                reason, note, ctx = _classify_rejection(prices)
                if reason == "REJ_TOO_SHORT_WINDOW":
                    insufficient_history += 1
                if reason.startswith("REJ_"):
                    context = "classify_rejection" if reason == "REJ_OTHER" else None
                    _record_rejection(reason, ticker, context=context)
                if reason == "REJ_TOO_SHORT_WINDOW" and ctx:
                    _record_short_window(ctx)
                logger.debug(f"[STOCK EXCLUDED] {ticker}: could not compute momentum score")

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)
    rejection_ok = len(scores)

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

    notes_final: Dict[str, str] = {}
    for reason, notes in score_failure_notes.items():
        if notes:
            notes_final[reason] = ",".join(sorted(notes))
        else:
            notes_final[reason] = ""

    return RankingResult(
        scores=scores,
        eligibility=eligibility,
        score_failure_breakdown=score_failure_breakdown,
        score_failure_samples=score_failure_samples,
        score_failure_notes=notes_final,
        rejection_attempted=rejection_attempted,
        rejection_ok=rejection_ok,
        rejection_counts=rejection_counts,
        rejection_samples=rejection_samples,
        rejection_exception_hist=rejection_exception_hist,
        rejection_sample_contexts=rejection_sample_contexts,
        short_window_samples=short_window_samples,
    )


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
