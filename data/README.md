# `data/` — Data ingestion

Downloads and caches the raw market data the pipeline is built on.

| File | Responsibility |
|---|---|
| `fetch.py` | Downloads SPY and VIX daily data (2010–2024) from Yahoo Finance via `yfinance`, computes log returns (`ln(Close_t / Close_{t-1})`) and the 5-day VIX change, joins the two series on date, and caches them to CSV. |

## Key functions

- `fetch_spy()` / `fetch_vix()` — download and preprocess each series.
- `build_dataset()` — join SPY + VIX, save `spy_features.csv` and `vix.csv`.
- `load_or_fetch()` — load from cache if present, otherwise download fresh. Used by the rest
  of the pipeline so Yahoo Finance is hit at most once.

```bash
python -m data.fetch     # download, save CSVs, print sanity checks
```

**Note:** the generated CSVs are git-ignored (regenerated on demand). Prices use
`auto_adjust=True` so returns account for dividends and splits.
