# Momentum Backtest

**Regime-Aware Momentum with Defensive Replacement**

A tactical momentum strategy backtesting framework for Indian equities with state-machine-based risk management.

---

## 1. Strategy Objective

The objective of this strategy is to deliver **equity-like long-term returns** with **structurally lower drawdowns and volatility** than traditional momentum factor indices, while remaining **fully invested through most market environments**.

The strategy is explicitly **not designed to maximize upside in all conditions**.
It is designed to **optimize the asymmetry between upside participation and downside avoidance**.

---

## 2. Universe and Rebalancing

* **Universe:** Broad Indian equity universe
* **Selection:** Top-ranked momentum stocks based on multi-horizon risk-adjusted returns
* **Portfolio Size:** 30 stocks (balance between diversification and signal strength)
* **Rebalance Frequency:** **Every 2 months**
  * Chosen deliberately to reduce turnover, noise, and over-reaction
  * Monthly rebalance was tested and rejected due to excess churn
* **Weighting:** Inverse volatility (risk-balanced, not cap-weighted)

---

## 3. Regime Framework (Core Differentiator)

The strategy operates through a **state machine**, not a static allocation:

| State | Meaning | Portfolio Behavior |
|-------|---------|-------------------|
| RISK_ON | Clean momentum regime | Full momentum portfolio |
| DEFENSIVE_MOMENTUM | Volatility shock / fragile trend | Defensive momentum basket |
| CASH_INVESTED | Capital protection regime | Defensive replacement portfolio |
| CASH_TRUE | Capital protection regime | True cash (no equity exposure) |

**Migration note (v3.5):**
The legacy ambiguous CASH state was split into **CASH_INVESTED** and **CASH_TRUE** for clear auditability. No behavioral change.
Use `--legacy-cash-state-names` if you need the old `CASH` label for downstream tooling.

---

## 4. Benchmarking Philosophy

Single benchmark used for regime signals and diagnostics:

1. **Momentum 50 (Mom50)**
   * Used for market regime signals and trading calendar
   * Used for relative performance diagnostics
   * Represents a fully invested, high-octane momentum factor

The strategy is **not designed to dominate Momentum-50 in all regimes**.
It is designed to deliver a **different risk-return profile**.

---

## 5. The Decisive Diagnostic: Capture Ratios

To conclusively explain relative performance, **up-capture and down-capture ratios vs Momentum-50** were computed using **monthly, rebalance-aligned data**.

### Results

| Metric | Value |
|--------|-------|
| **Up-Capture Ratio** | **69.3%** |
| **Down-Capture Ratio** | **32.7%** |
| Up Months | 69.7% of periods |
| Down Months | 30.3% of periods |

### Interpretation (Unambiguous)

* When Momentum-50 rises, the strategy captures **~70% of the upside**
* When Momentum-50 falls, the strategy captures **only ~33% of the downside**
* **~67% of drawdowns are avoided**, at the cost of **~30% upside sacrifice**

This confirms:

> Relative underperformance vs Momentum-50 is **intentional** and driven by **downside protection**, not by structural weakness or missed trends.

---

## 6. What This Strategy Is — and Is Not

### This strategy **IS**:

* Cycle-aware
* Drawdown-controlled
* Volatility-managed
* Suitable for investors prioritizing capital preservation and behavioral robustness

### This strategy **IS NOT**:

* A leverage-free replica of Momentum-50
* Designed to win every bull-market leaderboard
* Optimized for short-term factor timing

Any comparison that ignores capture ratios is **incomplete**.

---

## 7. Empirical Outcomes (Long-Run)

### Performance Summary (2011-2025, 14+ years)

| Metric | Strategy | Mom50 |
|--------|----------|------------------|
| CAGR | 19.11% | 19.11% |
| Max Drawdown | 21.28% | ~40% |
| Up-Capture | 69.3% | 100% |
| Down-Capture | 32.7% | 100% |

Across long horizons:

* Max drawdown is reduced by **~40–45% vs Momentum-50**
* Volatility is materially lower
* Turnover is controlled via 2-month rebalance
* CAGR shortfall (when present) is fully explained by **intentional downside avoidance**

This is a **portfolio construction choice**, not a modeling error.

---

## 8. Governance Rule (Locked)

Going forward:

* **Up-capture and down-capture ratios are invariant diagnostics**
* Any strategy modification must report:
  * Δ Up-Capture
  * Δ Down-Capture
* Any change that materially worsens down-capture **without explicit mandate** is rejected, regardless of headline CAGR improvement

This prevents silent risk creep.

### Acceptance Bands

| Metric | Target Range |
|--------|--------------|
| Up-Capture | 65–75% |
| Down-Capture | ≤40% |

---

## 9. Final Positioning Statement

This strategy represents a **defensive momentum allocation**, not a pure momentum factor clone.

It is appropriate for:

* Long-term capital allocation
* Investors sensitive to deep drawdowns
* Portfolios where behavioral sustainability matters as much as terminal CAGR

The capture-ratio analysis **closes the debate** on relative performance.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd momentum_no_indicator

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.9+
- pandas
- numpy
- yfinance (for benchmark data download)

---

## Data Requirements

### 1. Stock Universe (CSV)

A CSV file listing the stock tickers to consider. Example: `data/universe.csv`

```csv
Symbol
RELIANCE.NS
TCS.NS
HDFCBANK.NS
...
```

### 2. Price Data (Parquet)

OHLCV data in Parquet format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading date |
| symbol | string | Stock ticker (e.g., "RELIANCE.NS") |
| open | float | Opening price |
| high | float | High price |
| low | float | Low price |
| close | float | Closing price |
| adj_close | float | Adjusted close price |
| volume | int | Trading volume |

**Data range**: At least 12 months before backtest start date (for momentum lookback).

### 3. Benchmark Data

Provide a local Mom50 benchmark file (parquet or CSV) via `--benchmark-csv` or `--benchmark-parquet`:

```csv
Date,Close
2015-01-01,9191.08
2015-01-02,9320.29
...
```

---

## Usage

### Recommended Configuration (V3.1)

```bash
python -m momentum_backtest \
  --tickers-csv data/universe.csv \
  --parquet-file data/ohlcv_mcap5k.parquet \
  --start 2011-01-01 \
  --end 2025-12-31 \
  --rebalance-months 2 \
  --top-n 30 \
  --panic-defensive-mode \
  --panic-vol-ratio 1.5 \
  --defensive-basket-size 30 \
  --cash-replace-mode defensive \
  --tc-bps 10 \
  --comparison-benchmark-csv data/mom50_benchmark.csv \
  --output-dir output/v3_recommended
```

### Price Column Examples

```bash
# Legacy (close)
python -m momentum_backtest \
  --tickers-csv data/universe.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --price-column close \
  --start 2016-01-01 \
  --end 2024-12-31 \
  --output-dir output/legacy_close

# vNext (adj_close)
python -m momentum_backtest \
  --tickers-csv data/universe.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --price-column adj_close \
  --start 2016-01-01 \
  --end 2024-12-31 \
  --output-dir output/vnext_adj_close
```

### Command Line Options

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--tickers-csv` | Path to CSV file with stock universe |
| `--parquet-file` | Path to Parquet file with OHLCV data |
| `--price-column` | Price column to use: `close` or `adj_close` |
| `--start` | Backtest start date (YYYY-MM-DD) |
| `--end` | Backtest end date (YYYY-MM-DD) |
| `--output-dir` | Directory for output files |

#### Strategy Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-n` | 20 | Number of stocks to hold |
| `--max-weight` | 0.05 | Maximum weight per stock (5%) |
| `--tc-bps` | 10 | Transaction costs in basis points |
| `--rebalance-months` | 1 | Rebalance frequency: 1=monthly, 2=bi-monthly |

#### V2/V3 Strategy Levers

| Argument | Default | Description |
|----------|---------|-------------|
| `--panic-defensive-mode` | False | Enable DEFENSIVE_MOMENTUM state |
| `--defensive-basket-size` | 15 | Number of stocks in defensive basket |
| `--panic-vol-ratio` | 2.0 | Vol ratio threshold for PANIC trigger |
| `--cash-replace-mode` | none | V3.1: `none` or `defensive` |
| `--cash-rate-annual` | 0.0 | Annualized cash yield (e.g., 0.05 for 5%) |

#### Benchmark Options

| Argument | Description |
|----------|-------------|
| `--benchmark-csv` | Path to local benchmark CSV (used for state machine signals) |
| `--comparison-benchmark-csv` | Secondary benchmark for performance comparison (e.g., Mom50) |

---

## State Machine Details

The strategy uses a **4-state machine** evaluated at each rebalance:

```
                         ┌─────────────┐
                         │   RISK_ON   │
                         │  (Momentum) │
                         └──────┬──────┘
                                │
                 Vol ratio > threshold OR
                 Portfolio DD > 15%
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────┐           ┌───────────────────────┐
    │       PANIC       │           │  DEFENSIVE_MOMENTUM   │
    │  (100% Cash)      │           │  (Low-vol basket)     │
    │                   │           │                       │
    │  V1: Default      │           │  V2+: --panic-        │
    │                   │           │       defensive-mode  │
    └─────────┬─────────┘           └───────────┬───────────┘
              │                                 │
              └────────────┬────────────────────┘
                           │
            Vol ratio < exit threshold AND
            Benchmark 3M return < -5%
                           │
                           ▼
                     ┌────────────────────┐
                     │ CASH_INVESTED/TRUE │
                     └─────────┬──────────┘
                            │
            Benchmark 6M > 0% AND 3M > 0%
                            │
                            ▼
                     ┌─────────────┐
                     │   RISK_ON   │
                     └─────────────┘
```

### V3.1: Cash Replace Mode

| Mode | Cash Regime Behavior |
|------|----------------------|
| `none` | CASH_TRUE (hold actual cash) |
| `defensive` | CASH_INVESTED (hold defensive momentum basket) |

When `--cash-replace-mode defensive`:
- Cash regime is explicitly labeled as CASH_INVESTED
- Portfolio holds defensive low-volatility stocks
- `pct_time_cash_invested` reported in metrics.json

### State Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `panic_vol_ratio` | 1.5 | 1-month vol / 6-month vol ratio |
| `panic_dd_threshold` | 0.15 | Portfolio drawdown threshold (15%) |
| `panic_exit_vol_ratio` | 1.5 | Vol ratio to exit PANIC/DEFENSIVE |
| `cash_exit_bench_6m` | 0.0 | Benchmark 6M return to exit CASH |
| `cash_exit_bench_3m` | 0.0 | Benchmark 3M return to exit CASH |

---

## Momentum Scoring

Stocks are scored using the NSE Momentum Index methodology:

```
Score = 0.5 × (12M Return / 12M Volatility) + 0.5 × (6M Return / 6M Volatility)
```

### Eligibility Filters

A stock must pass all filters to be eligible:
1. **History**: At least 13 months of price data
2. **12M Return**: Must be positive (> 0%)
3. **6M Return**: Must be positive (> 0%)
4. **Positive Months**: At least 50% of trailing 12 months must be positive
5. **Max Drawdown**: Less than 30% drawdown in trailing 12 months

### Portfolio Construction

- **Weighting**: Inverse volatility (lower vol = higher weight)
- **Max Weight Cap**: 5% per stock (configurable via `--max-weight`)

---

## Output Files

| File | Description |
|------|-------------|
| `metrics.json` | Performance metrics (CAGR, Sharpe, MaxDD, etc.) |
| `metrics_vs_comparison.json` | Capture ratios vs comparison benchmark |
| `capture_ratios.csv` | Monthly capture ratio time series |
| `equity_curve.csv` | Daily equity values |
| `rebalance_log.csv` | Holdings and trades at each rebalance |
| `state_log.csv` | State transitions with `invested_flag` |
| `holdings_snapshot.csv` | Detailed holdings at each rebalance |
| `rolling_excess_3y_vs_benchmark.csv` | Rolling 3-year excess return |
| `run_manifest.json` | Full configuration for reproducibility |

---

## Diagnostic Scripts

### Capture Ratio Analysis

```bash
python scripts/compute_capture_ratios.py output/v3_recommended
```

Computes up-capture and down-capture ratios vs Mom50.

### V3.1 Validation

```bash
python scripts/validate_v31_implementation.py
```

Validates:
1. Defensive CASH uses stock portfolio returns (not benchmark)
2. Defensive portfolio is distinct from Mom50
3. Attribution split by state (CAGR contribution, turnover)

---

## Project Structure

```
momentum_no_indicator/
├── src/
│   └── momentum_backtest/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── cli.py               # Command line interface
│       ├── config.py            # Configuration dataclass
│       ├── data/
│       │   ├── benchmark.py     # Benchmark data loading
│       │   ├── calendar.py      # Rebalance calendar
│       │   ├── downloader.py    # Price data loading
│       │   └── universe.py      # Stock universe
│       ├── engine/
│       │   ├── backtest.py      # Main backtest loop
│       │   ├── portfolio.py     # Portfolio construction
│       │   ├── signals.py       # Momentum scoring
│       │   ├── state_machine.py # State transitions
│       │   └── stats.py         # Return/vol calculations
│       └── reporting/
│           ├── exporter.py      # Output file generation
│           ├── metrics.py       # Performance metrics
│           └── validation.py    # Validation checks
├── scripts/
│   ├── compute_capture_ratios.py
│   ├── validate_v31_implementation.py
│   └── run_cash_replace_ablation.py
├── data/
│   ├── universe.csv
│   ├── ohlcv_mcap5k.parquet
│   └── mom50_benchmark.csv
├── output/
└── README.md
```

---

## Important Notes

### Survivorship Bias Warning

The backtest uses **current** constituents for the entire period. This introduces survivorship bias that may overstate returns by 1-3% annually.

### Transaction Costs

Default transaction cost is 10 basis points (0.10%) per trade, which includes:
- Brokerage fees
- STT (Securities Transaction Tax)
- Exchange fees
- Slippage

---

## License

[Add your license here]
