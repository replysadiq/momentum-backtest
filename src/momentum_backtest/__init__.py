"""
Momentum Backtest - Institution-grade momentum backtesting system for Indian equities.

This package implements a zero-indicator momentum strategy with:
- Monthly rebalancing on first trading day of each month
- Top-20 stock selection with inverse-volatility weighting
- Unified market state machine (RISK_ON / PANIC / CASH)
- Strict data sanitization and validation
"""

__version__ = "1.0.0"
