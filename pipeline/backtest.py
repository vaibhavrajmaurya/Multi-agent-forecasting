"""
Walk-forward backtesting engine.
Expands training window by step_months each fold; tests on the next step_months.
Fits all agents + updates Hedge weights with realized losses at each step.
"""
import pathlib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.agents import BaseAgent
from pipeline.aggregator import HedgeAggregator, EqualWeightAggregator
from pipeline.metrics import metrics_table


def walk_forward_backtest(
    agents: list[BaseAgent],
    aggregator: HedgeAggregator | EqualWeightAggregator,
    feature_df: pd.DataFrame,
    initial_train_years: int = 3,
    step_months: int = 3,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest.

    Returns a DataFrame with columns:
      date, actual, <agent_name>_pred (×N), ensemble_pred,
      <agent_name>_weight (×N), fold
    """
    aggregator.reset()
    n = len(agents)
    agent_names = [a.name for a in agents]

    # Determine fold boundaries
    start = feature_df.index[0]
    train_end = start + relativedelta(years=initial_train_years) - relativedelta(days=1)

    records = []
    fold = 0

    while True:
        test_start = train_end + relativedelta(days=1)
        test_end   = train_end + relativedelta(months=step_months)

        if test_start > feature_df.index[-1]:
            break

        test_end = min(test_end, feature_df.index[-1])

        train_df = feature_df.loc[:train_end]
        test_df  = feature_df.loc[test_start:test_end]

        if len(train_df) < 50 or len(test_df) == 0:
            break

        # Fit all agents on expanding train window
        for agent in agents:
            agent.fit(train_df)

        # Predict + update Hedge step by step through the test window
        agent_preds_all = {name: agent.predict(test_df)
                           for name, agent in zip(agent_names, agents)}

        for i, (date, row) in enumerate(test_df.iterrows()):
            actual = row["log_return"]
            step_preds = [float(agent_preds_all[name][i]) for name in agent_names]

            ensemble = aggregator.aggregate(step_preds)
            aggregator.update(step_preds, actual)

            record = {"date": date, "actual": actual, "ensemble_pred": ensemble,
                      "fold": fold}
            for name, pred in zip(agent_names, step_preds):
                record[f"{name}_pred"] = pred
            for name, w in zip(agent_names, aggregator.weights):
                record[f"{name}_weight"] = w
            records.append(record)

        train_end += relativedelta(months=step_months)
        fold += 1

    return pd.DataFrame(records).set_index("date")


def run_baseline(feature_df: pd.DataFrame) -> pd.Series:
    """Buy-and-hold SPY: just the raw log returns over the backtest period."""
    return feature_df["log_return"]


def full_metrics_report(results_df: pd.DataFrame,
                        baseline: pd.Series,
                        agent_names: list[str]) -> pd.DataFrame:
    """
    Compute metrics_table for each agent, ensemble, equal-weight, and buy-and-hold.
    Returns a combined DataFrame sorted by Sharpe Ratio descending.
    """
    bh_returns = baseline.loc[results_df.index].values
    tables = []

    for name in agent_names:
        pred_col = f"{name}_pred"
        if pred_col in results_df.columns:
            tables.append(metrics_table(
                results_df["actual"].values,
                results_df[pred_col].values,
                benchmark=bh_returns,
                label=name,
            ))

    tables.append(metrics_table(
        results_df["actual"].values,
        results_df["ensemble_pred"].values,
        benchmark=bh_returns,
        label="Hedge Ensemble",
    ))

    # Same ensemble, but only trade on the most-confident half of days (sit in cash
    # otherwise). Tests whether high-|pred| days carry more edge than noise days.
    tables.append(metrics_table(
        results_df["actual"].values,
        results_df["ensemble_pred"].values,
        benchmark=bh_returns,
        label="Hedge (Conviction)",
        conviction=0.5,
    ))

    # Equal-weight ensemble (simple mean of agent predictions)
    agent_pred_cols = [f"{n}_pred" for n in agent_names if f"{n}_pred" in results_df.columns]
    eq_pred = results_df[agent_pred_cols].mean(axis=1).values
    tables.append(metrics_table(
        results_df["actual"].values,
        eq_pred,
        benchmark=bh_returns,
        label="Equal Weight",
    ))

    # Buy-and-hold baseline: always predict positive (always long)
    tables.append(metrics_table(
        results_df["actual"].values,
        np.abs(results_df["actual"].values),   # always long signal
        benchmark=bh_returns,
        label="Buy & Hold",
    ))

    combined = pd.concat(tables, ignore_index=True)
    return combined.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

    from data.fetch import load_or_fetch
    from pipeline.features import build_feature_matrix
    from pipeline.agents import TrendAgent, MomentumAgent, VolatilityAgent, SequenceAgent

    print("Loading data...")
    raw = load_or_fetch()
    df  = build_feature_matrix(raw)

    agents = [TrendAgent(), MomentumAgent(), VolatilityAgent(), SequenceAgent(epochs=25)]
    # loss_mode="mse": weight agents by predictive accuracy (mean-normalized squared
    # error). We also tested loss_mode="directional" (P&L-aligned), but daily direction
    # is near-random, so that loss chased noise and did not robustly help — see
    # CORRECTIONS.md. MSE gives a stable allocation that beats naive equal-weighting.
    # eta scales the update; alpha (Fixed-Share) keeps every agent revivable.
    aggregator = HedgeAggregator(n_agents=len(agents), eta=0.2, alpha=0.05,
                                 loss_mode="mse")

    print("Running walk-forward backtest (this may take a few minutes)...")
    results = walk_forward_backtest(agents, aggregator, df,
                                    initial_train_years=3, step_months=3)

    out = pathlib.Path(__file__).parents[1] / "results" / "backtest_results.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out)
    print(f"Saved {len(results)} rows to {out}")

    baseline = run_baseline(df)
    agent_names = [a.name for a in agents]
    report = full_metrics_report(results, baseline, agent_names)
    print("\n── Full Metrics Report ──")
    print(report.to_string(index=False))

    report.to_csv(out.parent / "metrics_report.csv", index=False)
