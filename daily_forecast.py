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


def get_valuation_label(ev_ebitda):
    if ev_ebitda is None:
        return "N/A"
    if ev_ebitda < 5:
        return "Very cheap"
    if ev_ebitda <= 10:
        return "Fair"
    if ev_ebitda <= 15:
        return "Expensive"
    return "Very expensive"


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

    current_price = x_df["close"].iloc[-1]
    avg_forecast = pred_df["close"].mean()
    change_pct = ((avg_forecast - current_price) / current_price) * 100

    signal = "BUY" if change_pct > 3 else ("SELL" if change_pct < -4 else "HOLD")

    info = yf.Ticker(ticker).info
    ev_ebitda = info.get("enterpriseToEbitda")
    roic = info.get("returnOnEquity")  # proxy for ROIC when true ROIC isn't exposed by yfinance
    valuation_label = get_valuation_label(ev_ebitda)
    ev_ebitda_str = f"{ev_ebitda:.1f}" if ev_ebitda is not None else "N/A"
    roic_str = f"{roic * 100:.0f}%" if roic is not None else "N/A"

    print(f"{ticker}: {current_price:.2f} -> {avg_forecast:.2f} ({change_pct:+.1f}%) | {signal} | EV/EBITDA: {ev_ebitda_str} ({valuation_label}) | ROIC: {roic_str}")
