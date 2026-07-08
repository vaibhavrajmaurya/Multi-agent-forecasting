import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, annualization: int = 252) -> float:
    """Annualized Sharpe ratio: mean / std * sqrt(252)."""
    returns = np.asarray(returns, dtype=float)
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualization))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline as a positive fraction.
    equity_curve: cumulative value series (e.g. $10,000 compounded).
    """
    curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    return float(drawdown.max())


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with correct sign (up/down)."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_ratio(strategy_returns: np.ndarray,
                      benchmark_returns: np.ndarray,
                      annualization: int = 252) -> float:
    """
    Annualized Information Ratio: active return / tracking error.
    IR = mean(strategy - benchmark) / std(strategy - benchmark) * sqrt(252)
    """
    active = np.asarray(strategy_returns) - np.asarray(benchmark_returns)
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(annualization))


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray,
                  benchmark: np.ndarray | None = None,
                  label: str = "Model",
                  conviction: float = 0.0) -> pd.DataFrame:
    """
    Compute all metrics for one agent/ensemble and return as a single-row DataFrame.
    If benchmark is provided, also computes Information Ratio.

    conviction: fraction of days to sit OUT (in cash). 0.0 = trade every day.
      With conviction > 0 we stay flat on the lowest-|pred| days (no strong view)
      and trade only the most confident (1 - conviction) fraction. This is a genuine
      risk reduction — lower market exposure, not the zero-padding artifact: the flat
      days are a deliberate cash decision, not discarded predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Position: +1 long / -1 short by predicted direction, or 0 (cash) on
    # low-conviction days when a threshold is set.
    if conviction > 0:
        thresh = np.quantile(np.abs(y_pred), conviction)
        position = np.where(np.abs(y_pred) >= thresh, np.sign(y_pred), 0.0)
    else:
        position = np.sign(y_pred)

    strategy_returns = position * y_true
    equity_curve = 10_000 * np.exp(np.cumsum(strategy_returns))

    row = {
        "Label": label,
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 6),
        "Directional Accuracy": round(directional_accuracy(y_true, y_pred), 4),
        "Sharpe Ratio": round(sharpe_ratio(strategy_returns), 3),
        "Max Drawdown": round(max_drawdown(equity_curve), 4),
    }

    if benchmark is not None:
        row["Information Ratio"] = round(
            information_ratio(strategy_returns, benchmark), 3
        )

    return pd.DataFrame([row])
