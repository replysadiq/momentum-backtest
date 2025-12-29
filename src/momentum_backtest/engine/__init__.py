"""
Engine layer for the momentum backtest system.

Modules:
- stats: Canonical statistical computations (single source of truth)
- signals: Momentum scoring and stock-level filters
- portfolio: Stock selection and weighting
- state_machine: Market regime state transitions (pure function)
- backtest: Main simulation loop
"""

from .stats import (
    total_return,
    total_return_months,
    realized_vol,
    max_drawdown,
    compute_turnover,
    apply_transaction_cost,
)
from .signals import compute_momentum_score, apply_eligibility_filters
from .portfolio import select_top_stocks, compute_inverse_vol_weights
from .state_machine import (
    MarketState,
    StateFeatures,
    StateThresholds,
    StateContext,
    CashEntryMode,
    next_state,
    compute_state_features,
)
from .backtest import run_backtest, BacktestResult

__all__ = [
    "total_return",
    "total_return_months",
    "realized_vol",
    "max_drawdown",
    "compute_turnover",
    "apply_transaction_cost",
    "compute_momentum_score",
    "apply_eligibility_filters",
    "select_top_stocks",
    "compute_inverse_vol_weights",
    "MarketState",
    "StateFeatures",
    "StateThresholds",
    "StateContext",
    "CashEntryMode",
    "next_state",
    "compute_state_features",
    "run_backtest",
    "BacktestResult",
]
