# `notebooks/` — Teaching curriculum

This project was delivered as a mentored course (Stamatics, IIT Kanpur). The notebooks build
up the concepts behind the pipeline, week by week.

| Notebook | Topic |
|---|---|
| `Week1_Assignment.ipynb` | Time-series foundations — components (trend, seasonality, noise), pandas, decomposition, smoothing. |
| `Week2_Assignment.ipynb` | ML for forecasting — linear models, XGBoost, and hybrid approaches; reframing a time series as supervised regression. |
| `Week3_Assignment.ipynb` | Financial features & agents — log returns, lookahead bias, RSI/MACD/Bollinger from scratch, the agent (ABC) pattern, and an intro to the Hedge algorithm. |

Supporting reading material is in [`../docs/resources/`](../docs/resources/).

Together these notebooks map directly onto the production code in [`../pipeline/`](../pipeline/):
Week 2 → Trend & Momentum agents, Week 3 → feature engineering + the aggregator.
