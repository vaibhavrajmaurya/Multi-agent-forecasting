# `pipeline/` — Forecasting pipeline

The core of the project: feature engineering, the four forecasting agents, the online-learning
aggregator, the backtesting engine, and the financial metrics.

| Module | Responsibility |
|---|---|
| `features.py` | Build the model-ready feature matrix from raw prices — RSI, MACD, Bollinger width, lag features, rolling statistics, and VIX. Every feature is shifted by ≥1 day to prevent lookahead bias (the keystone `close = df["Close"].shift(1)`). |
| `agents.py` | `BaseAgent` abstract interface + the four specialists: **TrendAgent** (OLS on a deterministic trend + annual Fourier seasonality), **MomentumAgent** (XGBoost on lag/rolling/RSI/MACD), **VolatilityAgent** (XGBoost on Bollinger/volatility/VIX), **SequenceAgent** (2-layer LSTM on 30-day return sequences, with a fixed seed for reproducibility). |
| `aggregator.py` | `HedgeAggregator` — multiplicative-weights update with per-step loss normalization and **Fixed-Share** mixing so the ensemble stays regime-adaptive. `EqualWeightAggregator` — the naive averaging baseline. |
| `backtest.py` | `walk_forward_backtest()` — expanding-window, 3-year initial / 3-month steps (48 folds). Fits all agents per fold and updates Hedge weights day-by-day through each test window. `full_metrics_report()` compiles every model vs. Buy & Hold. |
| `metrics.py` | `sharpe_ratio`, `max_drawdown`, `directional_accuracy`, `information_ratio`, and `metrics_table` (with an optional conviction threshold). |
| `evaluate.py` | Lightweight agent-comparison harness (fit on 2010–2018, evaluate on 2019). |

## Agents at a glance

| Agent | Model | Inputs | Captures |
|---|---|---|---|
| TrendAgent | LinearRegression + `DeterministicProcess` + `CalendarFourier` | calendar only | long-run drift & seasonality |
| MomentumAgent | XGBoost | lags, rolling stats, RSI, MACD | short-term momentum |
| VolatilityAgent | XGBoost | Bollinger width, rolling std, VIX | volatility regime |
| SequenceAgent | 2-layer LSTM (PyTorch) | raw 30-day return sequence | non-linear temporal patterns |

## Run a module directly

Each module has a `__main__` smoke test:

```bash
python -m pipeline.features     # build & inspect the feature matrix
python -m pipeline.agents       # fit all agents on 2010–2018, print 2019 MAE
python -m pipeline.aggregator   # toy Hedge weight-evolution test
python -m pipeline.backtest     # full 48-fold backtest → results/*.csv
```

See [`../docs/REPORT.html`](../docs/REPORT.html) for a line-by-line walkthrough of each module.
