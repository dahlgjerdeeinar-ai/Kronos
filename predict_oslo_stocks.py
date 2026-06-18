"""
Fetch historical data for a Norwegian stock (Oslo Børs) via yfinance,
format it for KronosPredictor, run a forecast and print the results.

Usage:
    python predict_oslo_stocks.py [TICKER] [PRED_DAYS]

Example:
    python predict_oslo_stocks.py EQNR.OL 10
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# Make sure the model package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos-master"))
from model import Kronos, KronosTokenizer, KronosPredictor


# ── configuration ────────────────────────────────────────────────────────────
TICKER = sys.argv[1] if len(sys.argv) > 1 else "EQNR.OL"
PRED_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LOOKBACK_DAYS = 400          # rows of history fed to the model
INTERVAL = "1d"              # daily bars
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512


def fetch_ohlcv(ticker: str, n_days: int) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and return a clean DataFrame."""
    # download slightly more than needed to account for weekends/holidays
    raw = yf.download(
        ticker,
        period=f"{n_days + 100}d",
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the ticker symbol and your internet connection.")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.rename(columns={"adj close": "close"})
    required = ["open", "high", "low", "close", "volume"]
    raw = raw[required].dropna().reset_index()
    raw = raw.rename(columns={"Date": "timestamps", "Datetime": "timestamps",
                               "date": "timestamps", "index": "timestamps"})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"])

    # Kronos expects an 'amount' column (turnover); estimate if absent
    if "amount" not in raw.columns:
        raw["amount"] = raw["close"] * raw["volume"]

    return raw.tail(n_days).reset_index(drop=True)


def run_prediction(ticker: str, pred_days: int) -> pd.DataFrame:
    """Full pipeline: fetch → format → predict → return forecast DataFrame."""

    print(f"\n[1/4] Fetching {LOOKBACK_DAYS + pred_days} trading days for {ticker} …")
    total_days = LOOKBACK_DAYS + pred_days
    df = fetch_ohlcv(ticker, total_days)

    if len(df) < 50:
        raise ValueError(f"Only {len(df)} rows available – not enough history.")

    # Use whatever we actually got; adapt lookback accordingly
    actual_lookback = max(len(df) - pred_days, 50)
    pred_days = len(df) - actual_lookback

    x_df = df.loc[: actual_lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: actual_lookback - 1, "timestamps"]

    # Build future timestamps (business-day frequency)
    last_date = df["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_days
    )
    y_timestamp = pd.Series(future_dates, name="timestamps")

    print(f"[2/4] Loading model ({MODEL_NAME}) …")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)

    print(f"[3/4] Running forecast (pred_len={pred_days}) …")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_days,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    return pred_df


def print_forecast(ticker: str, pred_df: pd.DataFrame) -> None:
    """Pretty-print the forecast table."""
    print(f"\n[4/4] Forecast for {ticker} – next {len(pred_df)} trading days\n")
    print("=" * 72)
    header = f"{'Date':<14} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>14}"
    print(header)
    print("-" * 72)
    for ts, row in pred_df.iterrows():
        date_str = str(ts)[:10]
        print(
            f"{date_str:<14} "
            f"{row['open']:>10.2f} "
            f"{row['high']:>10.2f} "
            f"{row['low']:>10.2f} "
            f"{row['close']:>10.2f} "
            f"{row['volume']:>14.0f}"
        )
    print("=" * 72)
    print(f"\nFirst predicted close : {pred_df['close'].iloc[0]:.2f}")
    print(f"Last  predicted close : {pred_df['close'].iloc[-1]:.2f}")
    pct = (pred_df['close'].iloc[-1] / pred_df['close'].iloc[0] - 1) * 100
    print(f"Trend over period     : {pct:+.2f}%")


if __name__ == "__main__":
    pred_df = run_prediction(TICKER, PRED_DAYS)
    print_forecast(TICKER, pred_df)
