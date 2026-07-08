# Multi-Agent Financial Forecasting

**An online-learning ensemble for daily S&P 500 forecasting.** Four specialist models
(trend, momentum, volatility, and an LSTM) each forecast the next day's return; a
regret-minimizing **Hedge** aggregator (Freund–Schapire 1997) with **Fixed-Share**
adaptation (Herbster–Warmuth 1998) reweights them every day toward whichever models are
currently most accurate. Validated with a 48-fold walk-forward backtest on SPY (2010–2024).

<p align="center">
  <em>The adaptive ensemble outperforms naive equal-weighting by ~58% on Sharpe, and the
  system self-selects the agents that carry genuine signal.</em>
</p>

---

## Key results

48-fold walk-forward backtest, 2,992 out-of-sample days, reproducible (fixed seed +
deterministic cuDNN). Strategy: go long/short daily by predicted direction.

| Model | Directional Acc. | Sharpe | Max Drawdown | Info Ratio |
|---|---:|---:|---:|---:|
| Buy & Hold (benchmark) | 55.5% | **0.79** | 0.34 | 0.00 |
| SequenceAgent (LSTM) | 54.6% | **0.73** | 0.34 | −0.11 |
| TrendAgent | 52.8% | 0.45 | 0.38 | −0.42 |
| **Hedge Ensemble** | 52.3% | **0.38** | 0.33 | −0.46 |
| Equal Weight (naive baseline) | 51.9% | 0.24 | 0.31 | −0.58 |
| VolatilityAgent | 50.5% | 0.14 | 0.38 | −0.59 |
| MomentumAgent | 50.6% | −0.01 | 0.45 | −0.64 |

**What the numbers show**

- **The aggregation adds real value** — the adaptive Hedge ensemble beats naive
  equal-weighting by ~58% on Sharpe (0.38 vs 0.24).
- **The best agent is competitive with passive** — the LSTM reaches 0.73 Sharpe vs Buy &
  Hold's 0.79, and the ensemble self-selects it without supervision.
- **Technical-indicator agents are ~coin flips** (~50% directional accuracy) — exactly as
  weak-form market efficiency predicts on liquid daily data.
- **Beating passive Buy & Hold on daily index returns is near-impossible**, and not doing so
  is the scientifically honest result. The contribution here is **adaptive, regime-aware
  model selection** — not a market-beating trading strategy. Two design ideas
  (a P&L-aligned loss and a conviction threshold) were tested and **did not** hold up; both
  are documented as negative results in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md).

---

## Method

```
Yahoo Finance (SPY + VIX, 2010–2024)
        │   data/fetch.py
        ▼
Feature engineering ─ pipeline/features.py
   log returns · RSI · MACD · Bollinger width · lag & rolling stats · VIX
        │   (all shifted ≥1 day — no lookahead)
        ▼
Four agents ─ pipeline/agents.py
   TrendAgent (OLS + Fourier) · MomentumAgent (XGBoost) ·
   VolatilityAgent (XGBoost) · SequenceAgent (LSTM)
        │   (each makes an independent daily forecast)
        ▼
Hedge + Fixed-Share aggregator ─ pipeline/aggregator.py
   w_i ← w_i · exp(−η · L_i), normalize, then mix in α-uniform
        │
        ▼
Walk-forward backtest ─ pipeline/backtest.py + metrics.py
   48 folds · Sharpe · max drawdown · directional accuracy · information ratio
        │
        ▼
Streamlit dashboard ─ app.py
```

**Design choices.** Log returns (stationary, additive) as the target; walk-forward
validation (never shuffle time series); Hedge over simple averaging (adaptive, with an
O(√(T·log N)) no-regret bound); Fixed-Share on top (markets are non-stationary, so no agent
should ever die). Full rationale and a line-by-line code walkthrough are in
[`docs/REPORT.html`](docs/REPORT.html).

---

## Quick start

```bash
pip install -r requirements.txt
python -m pipeline.backtest     # downloads data, runs the 48-fold backtest, writes results/
streamlit run app.py            # launch the interactive dashboard
```

Requires Python 3.10+. The first run downloads SPY/VIX from Yahoo Finance; later runs use the
cached CSVs. The backtest is reproducible — re-running gives identical numbers.

---

## Repository structure

```
.
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/                   # data ingestion (yfinance) — see data/README.md
├── pipeline/               # features, agents, aggregator, backtest — see pipeline/README.md
├── notebooks/              # teaching notebooks (weeks 1–3) — see notebooks/README.md
├── results/                # backtest output CSVs (consumed by the dashboard)
└── docs/                   # report, methodology log, resources — see docs/README.md
    ├── REPORT.html         # complete technical report (printable to PDF)
    ├── CORRECTIONS.md      # debugging & methodology log (incl. negative results)
    └── resources/          # supporting PDFs
```

Each top-level package has its own `README.md` documenting its modules.

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/REPORT.html`](docs/REPORT.html) | Full technical report — every term, every module, each agent end-to-end, all results |
| [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) | Engineering & methodology log: every fix from the baseline, including two honest negative results |
| `pipeline/README.md` | The forecasting pipeline modules |
| `data/README.md` | Data ingestion and caching |
| `notebooks/README.md` | Teaching curriculum |

---

## References

- Freund, Y. & Schapire, R. E. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting.* JCSS 55(1), 119–139. — Hedge algorithm.
- Herbster, M. & Warmuth, M. K. (1998). *Tracking the best expert.* Machine Learning 32(2), 151–178. — Fixed-Share.
- Bollinger, J. (2002). *Bollinger on Bollinger Bands.* McGraw-Hill.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems.* — RSI.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
