"""
Configuration dataclass for the momentum backtest system.

All parameters are validated at construction time.
Logging is intentionally NOT done here - it belongs in the CLI/main module.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional


class CashEntryMode(Enum):
    """
    V3: Cash entry mode variants for controlling when strategy goes to CASH.

    - baseline: Original behavior (benchmark_6m_return <= 0)
    - strict_dual: Require BOTH 6M AND 3M returns <= 0
    - strict_persist: Require baseline condition for 2 consecutive rebalances
    """
    BASELINE = "baseline"
    STRICT_DUAL = "strict_dual"
    STRICT_PERSIST = "strict_persist"


class CashReplaceMode(Enum):
    """
    V3.1: Cash replacement mode - what to hold when in CASH state.

    - none: Hold cash (original behavior)
    - defensive: Hold defensive momentum portfolio instead of cash
    """
    NONE = "none"
    DEFENSIVE = "defensive"


def _default_start_date() -> date:
    """Default start date: Jan 1 of (current_year - 10) for last 10 full calendar years."""
    today = date.today()
    # If we're in the middle of a year, use previous year as end
    end_year = today.year - 1 if today.month < 12 or today.day < 31 else today.year
    start_year = end_year - 9  # 10 years inclusive
    return date(start_year, 1, 1)


def _default_end_date() -> date:
    """Default end date: Dec 31 of last completed year."""
    today = date.today()
    end_year = today.year - 1 if today.month < 12 or today.day < 31 else today.year
    return date(end_year, 12, 31)


@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable configuration for the momentum backtest.

    All optional parameters have sensible defaults matching the specification.
    """

    # Optional: path to CSV file with universe tickers
    tickers_csv: Optional[Path]

    # Price column to use for returns/volatility computations
    price_column: str = "close"

    # Output directory for results
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Backtest period
    start_date: date = field(default_factory=_default_start_date)
    end_date: date = field(default_factory=_default_end_date)

    # Lookback windows (months)
    min_history_months: int = 13
    momentum_lookback_months: int = 12
    volatility_lookback_months: int = 6
    # History gate + warmup behavior
    history_gate_mode: str = "strict"  # strict|lenient
    history_leniency: float = 0.20
    warmup_mode: str = "auto"  # off|auto

    # Portfolio parameters
    top_n_stocks: int = 20
    max_weight: Optional[float] = 0.05  # None to disable cap

    # Drawdown filter (optional eligibility filter)
    use_dd_filter: bool = False
    dd_threshold: float = 0.40  # 40% max drawdown

    # Scoring methodology
    nse_style_scoring: bool = False  # Use NSE Momentum 50 scoring (Z-score normalized)

    # Transaction costs in basis points (0 = no costs)
    tc_bps: float = 0.0

    # Cash yield (annualized rate for cash/defensive periods)
    cash_rate_annual: float = 0.0  # Default 0%, set to 0.05 for 5% T-bill yield

    # State machine thresholds
    panic_dd_threshold: float = 0.15  # 15% portfolio drawdown triggers PANIC
    panic_vol_ratio: float = 2.0  # vol(1M) > 2.0 * vol(6M) triggers PANIC
    panic_exit_vol_ratio: float = 1.5  # vol(1M) < 1.5 * vol(6M) to exit PANIC
    panic_exit_bench_ret: float = -0.05  # bench 3M return > -5% to exit PANIC

    # Logging verbosity
    verbose: bool = False

    # === V2 Strategy Levers ===

    # Lever A: Defensive Momentum - hold low-vol stocks during PANIC instead of cash
    panic_defensive_mode: bool = False
    defensive_basket_size: int = 15  # Number of low-vol stocks in defensive basket

    # Lever B: Rank Buffer / Minimum Hold (mutually exclusive)
    rank_buffer: float = 0.0  # Relative rank improvement required (0.20 = 20%)
    min_hold_months: int = 0  # Minimum months to hold before selling

    # Lever D: Disable 6M Filter
    disable_6m_filter: bool = False  # Skip 6M return > 0 filter

    # Rebalance frequency (1 = monthly, 2 = bi-monthly, 3 = quarterly)
    rebalance_months: int = 1

    # V3: Cash entry mode (baseline, strict_dual, strict_persist)
    cash_entry_mode: CashEntryMode = CashEntryMode.BASELINE

    # V3.1: Cash replacement mode (none, defensive)
    cash_replace_mode: CashReplaceMode = CashReplaceMode.NONE

    # V3.5: Legacy CASH naming for compatibility
    legacy_cash_state_names: bool = False

    @property
    def strategy_version(self) -> str:
        """Compute strategy version based on enabled levers."""
        # V3.5: explicit cash states
        return "v3.5"
        # V3.1: cash_replace_mode != none
        if self.cash_replace_mode != CashReplaceMode.NONE:
            return "v3.1"
        # V3: cash_entry_mode != baseline
        if self.cash_entry_mode != CashEntryMode.BASELINE:
            return "v3"
        # V2: any of the V2 levers enabled
        if any([
            self.panic_defensive_mode,
            self.rank_buffer > 0,
            self.min_hold_months > 0,
            self.disable_6m_filter,
        ]):
            return "v2"
        return "v1"

    @property
    def enabled_levers(self) -> List[str]:
        """List of enabled V2/V3 levers for logging."""
        levers = []
        if self.panic_defensive_mode:
            levers.append(f"A:defensive_momentum(n={self.defensive_basket_size})")
        if self.rank_buffer > 0:
            levers.append(f"B:rank_buffer({self.rank_buffer:.0%})")
        if self.min_hold_months > 0:
            levers.append(f"B:min_hold({self.min_hold_months}mo)")
        if self.disable_6m_filter:
            levers.append("D:disable_6m_filter")
        # V3: cash entry mode
        if self.cash_entry_mode != CashEntryMode.BASELINE:
            levers.append(f"V3:cash_entry({self.cash_entry_mode.value})")
        # V3.1: cash replace mode
        if self.cash_replace_mode != CashReplaceMode.NONE:
            levers.append(f"V3.1:cash_replace({self.cash_replace_mode.value})")
        # V3.4: concentration cap
        if self.max_weight is not None:
            levers.append(f"V3.4:max_weight_cap({self.max_weight:.2f})")
        # V3.5: explicit cash states
        levers.append("V3.5:explicit_cash_states")
        return levers

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate tickers_csv exists if provided
        if self.tickers_csv is not None and not self.tickers_csv.exists():
            raise FileNotFoundError(f"Tickers CSV file not found: {self.tickers_csv}")

        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")

        if self.price_column not in ("close", "adj_close"):
            raise ValueError("price_column must be 'close' or 'adj_close'")

        if self.min_history_months < 1:
            raise ValueError("min_history_months must be >= 1")
        if self.momentum_lookback_months < 1:
            raise ValueError("momentum_lookback_months must be >= 1")
        if self.volatility_lookback_months < 1:
            raise ValueError("volatility_lookback_months must be >= 1")
        if self.history_gate_mode not in ("strict", "lenient"):
            raise ValueError("history_gate_mode must be 'strict' or 'lenient'")
        if not (0.0 <= self.history_leniency <= 0.5):
            raise ValueError("history_leniency must be in [0.0, 0.5]")
        if self.warmup_mode not in ("off", "auto"):
            raise ValueError("warmup_mode must be 'off' or 'auto'")

        # Validate numeric parameters
        if self.top_n_stocks < 1:
            raise ValueError(f"top_n_stocks must be >= 1, got {self.top_n_stocks}")

        if self.max_weight is not None and not (0 < self.max_weight <= 1):
            raise ValueError(f"max_weight must be in (0, 1], got {self.max_weight}")

        if not (0 < self.dd_threshold <= 1):
            raise ValueError(f"dd_threshold must be in (0, 1], got {self.dd_threshold}")

        if self.tc_bps < 0:
            raise ValueError(f"tc_bps must be >= 0, got {self.tc_bps}")

        if not (0 < self.panic_dd_threshold <= 1):
            raise ValueError(f"panic_dd_threshold must be in (0, 1], got {self.panic_dd_threshold}")

        # Validate V2 lever parameters
        if self.rank_buffer < 0 or self.rank_buffer > 0.5:
            raise ValueError(f"rank_buffer must be in [0, 0.5], got {self.rank_buffer}")

        if self.min_hold_months < 0 or self.min_hold_months > 12:
            raise ValueError(f"min_hold_months must be in [0, 12], got {self.min_hold_months}")

        if self.defensive_basket_size < 5 or self.defensive_basket_size > 50:
            raise ValueError(f"defensive_basket_size must be in [5, 50], got {self.defensive_basket_size}")

        # Validate rank_buffer and min_hold_months are mutually exclusive
        if self.rank_buffer > 0 and self.min_hold_months > 0:
            raise ValueError("rank_buffer and min_hold_months are mutually exclusive")

        # Validate rebalance_months
        if self.rebalance_months not in (1, 2, 3):
            raise ValueError(f"rebalance_months must be 1, 2, or 3, got {self.rebalance_months}")

    @property
    def tc_fraction(self) -> float:
        """Transaction cost as a fraction (tc_bps / 10000)."""
        return self.tc_bps / 10000.0
