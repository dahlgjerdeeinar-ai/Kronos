import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KRONOS_DIR = ROOT / "Kronos-master"
KRONOS_ZIP = ROOT / "Kronos-master.zip"

if not KRONOS_DIR.exists() and KRONOS_ZIP.exists():
    with zipfile.ZipFile(KRONOS_ZIP) as zf:
        zf.extractall(ROOT)

sys.path.insert(0, str(KRONOS_DIR))
import yfinance as yf
import pandas as pd
from datetime import datetime
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS = ["STB.OL", "TEL.OL", "EQNR.OL", "NOVO-B.CO"]

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

for ticker in TICKERS:
    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.columns = ["open","high","low","close","volume"]
    df["timestamps"] = pd.to_datetime(df.index)
    df = df.reset_index(drop=True)
    recent_df = df.tail(100).reset_index(drop=True)
    x_df = recent_df[["open","high","low","close","volume"]]
    x_timestamp = recent_df["timestamps"]
    future_dates = pd.bdate_range(start=datetime.today(), periods=6)[1:]
    y_timestamp = pd.Series(future_dates)
    pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False)

    if ticker == "EQNR.OL":
        print(f"[DEBUG] {ticker} last 10 rows of historical input data:")
        print(x_df.tail(10))
        print(f"[DEBUG] {ticker} raw predicted close prices for next 5 days:")
        print(pred_df["close"].to_string())

    current_price = x_df["close"].iloc[-1]
    avg_forecast = pred_df["close"].mean()
    change_pct = ((avg_forecast - current_price) / current_price) * 100
    signal = "BUY" if change_pct > 3 else ("SELL" if change_pct < -4 else "HOLD")
    print(f"{ticker}: {current_price:.2f} -> {avg_forecast:.2f} ({change_pct:+.1f}%) | {signal}")
