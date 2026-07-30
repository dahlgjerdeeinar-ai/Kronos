import json
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

# Suffix probe order for mapping bare screener symbols (e.g. "BIOMAR") to a
# tradable yfinance ticker: Denmark, Sweden, Finland, Norway.
NORDIC_SUFFIXES = [".CO", ".ST", ".HE", ".OL"]


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


def resolve_ticker(symbol):
    for suffix in NORDIC_SUFFIXES:
        candidate = f"{symbol}{suffix}"
        try:
            probe = yf.download(candidate, period="5d", interval="1d", auto_adjust=True, progress=False)
        except Exception:
            continue
        if isinstance(probe.columns, pd.MultiIndex):
            probe.columns = probe.columns.get_level_values(0)
        if not probe.empty and "Close" in probe.columns and probe["Close"].notna().any():
            return candidate
    return None


def forecast_ticker(predictor, ticker, future_dates):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return None
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    df.columns = ["open", "high", "low", "close", "volume"]
    df["timestamps"] = pd.to_datetime(df.index)
    df = df.reset_index(drop=True)
    recent_df = df.tail(100).reset_index(drop=True)
    x_df = recent_df[["open", "high", "low", "close", "volume"]]
    x_timestamp = recent_df["timestamps"]
    y_timestamp = pd.Series(future_dates)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    current_price = float(x_df["close"].iloc[-1])
    daily_prices = [float(p) for p in pred_df["close"].tolist()]
    avg_forecast = float(pred_df["close"].mean())
    change_pct = ((avg_forecast - current_price) / current_price) * 100

    signal = "BUY" if change_pct > 3 else ("SELL" if change_pct < -4 else "HOLD")

    info = yf.Ticker(ticker).info
    ev_ebitda = info.get("enterpriseToEbitda")
    roic = info.get("returnOnEquity")  # proxy for ROIC when true ROIC isn't exposed by yfinance
    valuation_label = get_valuation_label(ev_ebitda)

    return {
        "ticker": ticker,
        "current_price": current_price,
        "avg_forecast": avg_forecast,
        "change_pct": change_pct,
        "signal": signal,
        "ev_ebitda": ev_ebitda,
        "valuation_label": valuation_label,
        "roic": roic,
        "daily_prices": daily_prices,
    }


def run_forecast(screener_symbols=None):
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    future_dates = pd.bdate_range(start=datetime.today(), periods=6)[1:]
    dates = [d.strftime("%Y-%m-%d") for d in future_dates]

    results = []
    for ticker in TICKERS:
        result = forecast_ticker(predictor, ticker, future_dates)
        if result is not None:
            results.append(result)

    screener_forecasts = {}
    for symbol in screener_symbols or []:
        try:
            resolved = resolve_ticker(symbol)
            screener_forecasts[symbol] = forecast_ticker(predictor, resolved, future_dates) if resolved else None
        except Exception:
            screener_forecasts[symbol] = None

    return {"dates": dates, "tickers": results, "screener_forecasts": screener_forecasts}


if __name__ == "__main__":
    print(json.dumps(run_forecast(sys.argv[1:])))
