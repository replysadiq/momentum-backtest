"""
Configuration dataclass for the momentum backtest system.

All parameters are validated at construction time.
Logging is intentionally NOT done here - it belongs in the CLI/main module.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional


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

    # Required: path to CSV file with NIFTY 500 tickers
    tickers_csv: Path

    # Output directory for results
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Backtest period
    start_date: date = field(default_factory=_default_start_date)
    end_date: date = field(default_factory=_default_end_date)

    # Portfolio parameters
    top_n_stocks: int = 20
    max_weight: Optional[float] = 0.10  # None to disable cap

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

    @property
    def strategy_version(self) -> str:
        """Compute strategy version based on enabled levers."""
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
        """List of enabled V2 levers for logging."""
        levers = []
        if self.panic_defensive_mode:
            levers.append(f"A:defensive_momentum(n={self.defensive_basket_size})")
        if self.rank_buffer > 0:
            levers.append(f"B:rank_buffer({self.rank_buffer:.0%})")
        if self.min_hold_months > 0:
            levers.append(f"B:min_hold({self.min_hold_months}mo)")
        if self.disable_6m_filter:
            levers.append("D:disable_6m_filter")
        return levers

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate tickers_csv exists
        if not self.tickers_csv.exists():
            raise FileNotFoundError(f"Tickers CSV file not found: {self.tickers_csv}")

        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")

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

        if self.defensive_basket_size < 5 or self.defensive_basket_size > 30:
            raise ValueError(f"defensive_basket_size must be in [5, 30], got {self.defensive_basket_size}")

        # Validate rank_buffer and min_hold_months are mutually exclusive
        if self.rank_buffer > 0 and self.min_hold_months > 0:
            raise ValueError("rank_buffer and min_hold_months are mutually exclusive")

    @property
    def tc_fraction(self) -> float:
        """Transaction cost as a fraction (tc_bps / 10000)."""
        return self.tc_bps / 10000.0
