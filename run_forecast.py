"""Forecast SMOP.OL close prices for the next 5 days using Kronos-small."""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
KRONOS_DIR = ROOT / "Kronos-master"
KRONOS_ZIP = ROOT / "Kronos-master.zip"

if not KRONOS_DIR.exists() and KRONOS_ZIP.exists():
    with zipfile.ZipFile(KRONOS_ZIP) as zf:
        zf.extractall(ROOT)

sys.path.append(str(KRONOS_DIR))
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = "SMOP.OL"
LOOKBACK = 400
PRED_LEN = 5

# 1. Fetch historical price data
history = yf.download(TICKER, period="2y", interval="1d", auto_adjust=False, progress=False)
if history.empty:
    raise RuntimeError(f"No historical data returned for {TICKER}")

if isinstance(history.columns, pd.MultiIndex):
    history.columns = history.columns.get_level_values(0)

history = history.rename(columns=str.lower).reset_index()
history = history.rename(columns={"date": "timestamps"})
history["timestamps"] = pd.to_datetime(history["timestamps"])
history = history[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
history = history.tail(LOOKBACK).reset_index(drop=True)

x_df = history[["open", "high", "low", "close", "volume"]]
x_timestamp = history["timestamps"]
y_timestamp = pd.Series(pd.bdate_range(
    start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=PRED_LEN
))

# 2. Load model and tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 3. Instantiate predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 4. Run forecast
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# 5. Print predicted close prices
print(f"\nPredicted close prices for {TICKER} (next {PRED_LEN} trading days):")
for ts, close in zip(y_timestamp, pred_df["close"]):
    print(f"{ts.date()}: {close:.2f}")
