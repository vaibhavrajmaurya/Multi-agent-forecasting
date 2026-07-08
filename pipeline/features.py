import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI on shifted prices to avoid lookahead bias."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line) tuple."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(series: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger Band width = (upper - lower) / mid."""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (upper - lower) / mid


def compute_lag_features(series: pd.Series, lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    return pd.DataFrame(
        {f"lag_{lag}": series.shift(lag) for lag in lags}
    )


def compute_rolling_features(series: pd.Series, windows=(5, 21)) -> pd.DataFrame:
    frames = {}
    for w in windows:
        frames[f"rolling_mean_{w}"] = series.shift(1).rolling(w).mean()
        frames[f"rolling_std_{w}"] = series.shift(1).rolling(w).std()
    return pd.DataFrame(frames)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must have columns: Close, log_return, vix_close, vix_change_5d.
    All features are lagged by at least 1 day to prevent lookahead bias.
    Returns a clean DataFrame with log_return as the prediction target.
    """
    close = df["Close"].shift(1)  # yesterday's close — safe to use as input

    rsi = compute_rsi(close).rename("rsi_14")
    macd_line, macd_signal = compute_macd(close)
    macd_signal = macd_signal.rename("macd_signal")
    bb_width = compute_bollinger(close).rename("bb_width")
    lag_feats = compute_lag_features(df["log_return"], lags=(1, 2, 3, 5, 10))
    roll_feats = compute_rolling_features(df["log_return"], windows=(5, 21))

    # VIX features: already in df but shift 1 to ensure no leakage
    vix_level = df["vix_close"].shift(1).rename("vix_level")
    vix_change = df["vix_change_5d"].shift(1).rename("vix_change_5d")

    feature_df = pd.concat(
        [rsi, macd_signal, bb_width, lag_feats, roll_feats, vix_level, vix_change,
         df["log_return"]],
        axis=1,
    )
    feature_df.dropna(inplace=True)
    return feature_df


if __name__ == "__main__":
    from data.fetch import load_or_fetch

    raw = load_or_fetch()
    features = build_feature_matrix(raw)
    print("Feature matrix shape:", features.shape)
    print(features.head())
    print("\nColumns:", list(features.columns))
    print("NaNs:", features.isna().sum().sum())
