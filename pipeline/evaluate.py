"""
Day 4 evaluation harness.
Trains all 4 agents on 2010-2018, evaluates on 2019.
Prints MAE + directional accuracy per agent.
Flags if SequenceAgent MAE > 2x MomentumAgent MAE.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from data.fetch import load_or_fetch
from pipeline.features import build_feature_matrix
from pipeline.agents import TrendAgent, MomentumAgent, VolatilityAgent, SequenceAgent


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where the sign (up/down) is correct."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_agents(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    lstm_epochs: int = 20) -> pd.DataFrame:
    """
    Fit each agent on train_df, predict on test_df.
    Returns a DataFrame with MAE and directional accuracy per agent.
    """
    agents = [
        TrendAgent(),
        MomentumAgent(),
        VolatilityAgent(),
        SequenceAgent(epochs=lstm_epochs),
    ]

    y_true = test_df["log_return"].values
    records = []

    for agent in agents:
        print(f"  Training {agent.name}...", end=" ", flush=True)
        agent.fit(train_df)
        preds = agent.predict(test_df)
        mae = mean_absolute_error(y_true, preds)
        da  = directional_accuracy(y_true, preds)
        print(f"MAE={mae:.6f}  DirAcc={da:.3f}")
        records.append({
            "Agent": agent.name,
            "MAE": round(mae, 6),
            "Directional Accuracy": round(da, 4),
            "_agent_obj": agent,
        })

    results = pd.DataFrame(records)

    # ── Sanity check: LSTM should not be >2x worse than MomentumAgent ──────
    lstm_mae = results.loc[results["Agent"] == "SequenceAgent", "MAE"].values[0]
    mom_mae  = results.loc[results["Agent"] == "MomentumAgent", "MAE"].values[0]
    ratio = lstm_mae / mom_mae

    print(f"\n  LSTM/Momentum MAE ratio: {ratio:.2f}x")
    if ratio > 2.0:
        print("  ⚠ WARNING: LSTM is more than 2x worse than MomentumAgent.")
        print("    Check sequence length, normalization, or increase epochs.")
    else:
        print("  ✓ LSTM within acceptable range of MomentumAgent.")

    return results.drop(columns=["_agent_obj"])


def print_feature_importance(train_df: pd.DataFrame) -> None:
    """Print top-5 feature importance for MomentumAgent and VolatilityAgent."""
    print("\n── MomentumAgent Feature Importance ──")
    m = MomentumAgent()
    m.fit(train_df)
    print(m.feature_importance().head(5).to_string())

    print("\n── VolatilityAgent Feature Importance ──")
    v = VolatilityAgent()
    v.fit(train_df)
    print(v.feature_importance().head(5).to_string())


if __name__ == "__main__":
    print("Loading data...")
    raw = load_or_fetch()
    df  = build_feature_matrix(raw)

    train = df[df.index.year <= 2018]
    test  = df[df.index.year == 2019]

    print(f"Train: {train.index[0].date()} → {train.index[-1].date()}  "
          f"({len(train)} rows)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()}  "
          f"({len(test)} rows)\n")

    print("── Agent Evaluation ──")
    results = evaluate_agents(train, test, lstm_epochs=20)

    print("\n── Results Table ──")
    print(results.to_string(index=False))

    # Save to results/
    out = pathlib.Path(__file__).parents[1] / "results" / "agent_comparison_2019.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print_feature_importance(train)
