# Momentum Backtest

A tactical momentum strategy backtesting framework for Indian equities (NIFTY 500 universe) with state-machine-based risk management.

## Strategy Overview

This is a **monthly-rebalanced momentum strategy** with a 3-state risk management system that dynamically switches between:

1. **RISK_ON**: Full momentum exposure (top 20 stocks by momentum score)
2. **DEFENSIVE_MOMENTUM**: Low-volatility defensive basket during market stress
3. **CASH**: Full cash position during sustained bear markets

### Core Philosophy

The strategy aims to **match benchmark returns with significantly lower drawdowns** by:
- Capturing momentum alpha during bull markets
- Switching to defensive low-volatility stocks during volatility spikes
- Moving to cash during prolonged downturns

### Performance Summary (2016-2024, 9 years)

| Metric | Strategy (Best Config) | NIFTY 500 Mom 50 Benchmark |
|--------|------------------------|---------------------------|
| CAGR | 21.2% | 21.1% |
| Max Drawdown | 22.5% | 39.4% |
| Sharpe Ratio | 1.37 | 1.04 |
| Calmar Ratio | 0.94 | 0.54 |

**Result**: Same returns with 43% less drawdown and 32% better Sharpe ratio.

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

A CSV file listing the stock tickers to consider. Example: `data/nse_nifty500_current.csv`

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

The system automatically downloads benchmark data from Yahoo Finance:
- Primary: `^CRSLDX` (NIFTY 500 Index)
- Fallback: `^NSEI` (NIFTY 50 Index)

Alternatively, provide a local CSV with `--benchmark-csv`:

```csv
Date,Close
2015-01-01,9191.08
2015-01-02,9320.29
...
```

---

## Usage

### Basic Usage

```bash
# V1 Strategy (original)
python -m momentum_backtest \
  --tickers-csv data/nse_nifty500_current.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --start 2016-01-01 \
  --end 2024-12-31 \
  --output-dir output/v1_baseline

# V2 Strategy with Defensive Momentum (recommended)
python -m momentum_backtest \
  --tickers-csv data/nse_nifty500_current.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --start 2016-01-01 \
  --end 2024-12-31 \
  --panic-defensive-mode \
  --panic-vol-ratio 1.5 \
  --defensive-basket-size 20 \
  --cash-rate-annual 0.05 \
  --tc-bps 10 \
  --output-dir output/v2_recommended
```

### Command Line Options

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--tickers-csv` | Path to CSV file with stock universe |
| `--parquet-file` | Path to Parquet file with OHLCV data |
| `--start` | Backtest start date (YYYY-MM-DD) |
| `--end` | Backtest end date (YYYY-MM-DD) |
| `--output-dir` | Directory for output files |

#### Strategy Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-n` | 20 | Number of stocks to hold |
| `--max-weight` | 0.10 | Maximum weight per stock (10%) |
| `--tc-bps` | 10 | Transaction costs in basis points |

#### V2 Strategy Levers

| Argument | Default | Description |
|----------|---------|-------------|
| `--panic-defensive-mode` | False | Enable DEFENSIVE_MOMENTUM state |
| `--defensive-basket-size` | 15 | Number of stocks in defensive basket |
| `--panic-vol-ratio` | 2.0 | Vol ratio threshold for PANIC trigger |
| `--cash-rate-annual` | 0.0 | Annualized cash yield (e.g., 0.05 for 5%) |
| `--rank-buffer` | 0.0 | Rank buffer for reducing turnover |
| `--disable-6m-filter` | False | Disable 6-month return filter |

#### Benchmark Options

| Argument | Description |
|----------|-------------|
| `--benchmark-csv` | Path to local benchmark CSV |
| `--comparison-benchmark-csv` | Secondary benchmark for comparison |

---

## Strategy Details

### State Machine

The strategy uses a **4-state machine** evaluated at each monthly rebalance:

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
    │  V1: Default      │           │  V2: --panic-         │
    │                   │           │      defensive-mode   │
    └─────────┬─────────┘           └───────────┬───────────┘
              │                                 │
              └────────────┬────────────────────┘
                           │
            Vol ratio < exit threshold AND
            Benchmark 3M return < -5%
                           │
                           ▼
                     ┌─────────────┐
                     │    CASH     │
                     │  (T-bills)  │
                     └──────┬──────┘
                            │
            Benchmark 6M > 0% AND 3M > 0%
                            │
                            ▼
                     ┌─────────────┐
                     │   RISK_ON   │
                     └─────────────┘
```

**Key difference between V1 and V2:**

| Version | When PANIC triggers | Response | Holdings |
|---------|---------------------|----------|----------|
| **V1** (default) | Vol spike or DD breach | PANIC state | 0% equity (cash) |
| **V2** (`--panic-defensive-mode`) | Vol spike or DD breach | DEFENSIVE_MOMENTUM state | Low-vol stocks |

In V2, the PANIC **event** still fires (tracked as `panic_events_count`), but the **response** is DEFENSIVE_MOMENTUM instead of going to cash. This keeps you invested in low-volatility stocks during stress periods.

### State Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `panic_vol_ratio` | 1.5-2.0 | 1-month vol / 6-month vol ratio |
| `panic_dd_threshold` | 0.15 | Portfolio drawdown threshold (15%) |
| `panic_exit_vol_ratio` | 1.5 | Vol ratio to exit PANIC/DEFENSIVE |
| `cash_exit_bench_6m` | 0.0 | Benchmark 6M return to exit CASH |
| `cash_exit_bench_3m` | 0.0 | Benchmark 3M return to exit CASH |

### Momentum Scoring

Stocks are scored using the NSE Momentum Index methodology:

```
Score = 0.5 × (12M Return / 12M Volatility) + 0.5 × (6M Return / 6M Volatility)
```

### Eligibility Filters

A stock must pass all filters to be eligible:
1. **History**: At least 13 months of price data
2. **12M Return**: Must be positive (> 0%)
3. **6M Return**: Must be positive (> 0%) — can be disabled with `--disable-6m-filter`
4. **Positive Months**: At least 50% of trailing 12 months must be positive
5. **Max Drawdown**: Less than 30% drawdown in trailing 12 months

### Portfolio Construction

- **Weighting**: Inverse volatility (lower vol = higher weight)
- **Max Weight Cap**: 10% per stock (configurable)
- **Rebalance Frequency**: Monthly (first trading day of each month)

---

## Output Files

The backtest generates the following files in the output directory:

| File | Description |
|------|-------------|
| `metrics.json` | Performance metrics (CAGR, Sharpe, MaxDD, etc.) |
| `equity_curve.csv` | Daily equity values |
| `rebalance_log.csv` | Monthly holdings and trades |
| `state_log.csv` | State transitions over time |
| `holdings_snapshot.csv` | Detailed holdings at each rebalance |
| `eligibility_breakdown.csv` | Stock eligibility analysis |
| `drawdown_attribution.csv` | Drawdown event analysis |
| `run_manifest.json` | Full configuration for reproducibility |

---

## Configuration Presets

### Conservative (Default V1)
```bash
python -m momentum_backtest \
  --tickers-csv data/nse_nifty500_current.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --start 2016-01-01 --end 2024-12-31 \
  --tc-bps 10 \
  --output-dir output/conservative
```
- Standard momentum without defensive features
- Moves to CASH on extreme volatility only

### Balanced (Recommended)
```bash
python -m momentum_backtest \
  --tickers-csv data/nse_nifty500_current.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --start 2016-01-01 --end 2024-12-31 \
  --panic-defensive-mode \
  --panic-vol-ratio 1.5 \
  --defensive-basket-size 20 \
  --cash-rate-annual 0.05 \
  --tc-bps 10 \
  --output-dir output/balanced
```
- Defensive momentum during stress periods (20 low-vol stocks)
- 5% cash yield assumption
- More responsive to volatility spikes (1.5 threshold catches 7 events vs 1 at default 2.0)

### Aggressive
```bash
python -m momentum_backtest \
  --tickers-csv data/nse_nifty500_current.csv \
  --parquet-file data/ohlcv_yahoo.parquet \
  --start 2016-01-01 --end 2024-12-31 \
  --panic-defensive-mode \
  --panic-vol-ratio 2.0 \
  --top-n 25 \
  --disable-6m-filter \
  --tc-bps 10 \
  --output-dir output/aggressive
```
- Larger portfolio (25 stocks)
- Less responsive to volatility (only extreme events)
- Relaxed eligibility filters

---

## Vol Ratio Threshold Sensitivity

The `--panic-vol-ratio` parameter controls how often the strategy moves to defensive mode:

| Threshold | Events (9yr) | DEF_MOM Time | CAGR | MaxDD |
|-----------|--------------|--------------|------|-------|
| 2.0 | 1 | 2.8% | 19.5% | 22.7% |
| 1.8 | 2 | 3.7% | 20.0% | 22.7% |
| 1.5 | 7 | 10.2% | 21.2% | 22.5% |

Lower threshold = more defensive interventions = better risk-adjusted returns (in backtests).

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
│       │   ├── sanitizer.py     # Data quality checks
│       │   └── universe.py      # Stock universe
│       ├── engine/
│       │   ├── backtest.py      # Main backtest loop
│       │   ├── portfolio.py     # Portfolio construction
│       │   ├── signals.py       # Momentum scoring
│       │   ├── state_machine.py # State transitions
│       │   └── stats.py         # Return/vol calculations
│       └── reporting/
│           ├── audit.py         # Drawdown analysis
│           ├── exporter.py      # Output file generation
│           ├── metrics.py       # Performance metrics
│           └── validation.py    # Post-run validation
├── data/
│   ├── nse_nifty500_current.csv # Stock universe
│   ├── ohlcv_yahoo.parquet      # Price data
│   └── nifty500_momentum50_benchmark.csv
├── output/                       # Backtest results
└── README.md
```

---

## Important Notes

### Survivorship Bias Warning

The backtest uses **current** NIFTY 500 constituents for the entire period. This introduces survivorship bias that may overstate returns by 1-3% annually, as stocks that were removed from the index (due to poor performance, delisting, etc.) are excluded.

### Transaction Costs

Default transaction cost is 10 basis points (0.10%) per trade, which includes:
- Brokerage fees
- STT (Securities Transaction Tax)
- Exchange fees
- Slippage

Adjust with `--tc-bps` based on your actual trading costs.

### Cash Yield

The `--cash-rate-annual` parameter models the return earned while in CASH state. For Indian markets, 5% (0.05) is a reasonable assumption based on T-bill yields.

---

## License

[Add your license here]

---

## Acknowledgments

- NSE Momentum Index methodology for scoring approach
- Yahoo Finance for benchmark data
