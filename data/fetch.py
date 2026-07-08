import pathlib
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = pathlib.Path(__file__).parent
SPY_CSV = DATA_DIR / "spy_features.csv"
VIX_CSV = DATA_DIR / "vix.csv"

START = "2010-01-01"
END = "2024-12-31"


def fetch_spy() -> pd.DataFrame:
    df = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    df = df[["Close"]].copy()
    df.columns = ["Close"]
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def fetch_vix() -> pd.DataFrame:
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True, progress=False)
    vix = vix[["Close"]].copy()
    vix.columns = ["vix_close"]
    vix["vix_change_5d"] = vix["vix_close"].pct_change(5)
    vix.dropna(inplace=True)
    return vix


def build_dataset() -> pd.DataFrame:
    print("Downloading SPY...")
    spy = fetch_spy()
    print(f"  SPY rows: {len(spy)}")

    print("Downloading VIX...")
    vix = fetch_vix()
    print(f"  VIX rows: {len(vix)}")

    df = spy.join(vix, how="inner")
    df.dropna(inplace=True)
    print(f"  Merged rows: {len(df)}")

    spy.to_csv(SPY_CSV)
    vix.to_csv(VIX_CSV)
    print(f"Saved to {SPY_CSV}")
    return df


def load_or_fetch() -> pd.DataFrame:
    if SPY_CSV.exists() and VIX_CSV.exists():
        spy = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
        vix = pd.read_csv(VIX_CSV, index_col=0, parse_dates=True)
        return spy.join(vix, how="inner").dropna()
    return build_dataset()


if __name__ == "__main__":
    df = build_dataset()
    print("\nShape:", df.shape)
    print(df.head())
    print("\nNo NaNs:", df.isna().sum().sum() == 0)
    # Sanity check: log_return should not use today's close — already handled by shift(1)
    print("log_return range:", df["log_return"].min(), "to", df["log_return"].max())
