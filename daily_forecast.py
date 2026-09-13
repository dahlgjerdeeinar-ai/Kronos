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


def fetch_today_open(ticker):
    try:
        open_df = yf.download(ticker, period="2d", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return None
    if open_df.empty:
        return None
    if isinstance(open_df.columns, pd.MultiIndex):
        open_df.columns = open_df.columns.get_level_values(0)
    if "Open" not in open_df.columns or open_df["Open"].dropna().empty:
        return None
    return float(open_df["Open"].iloc[-1])


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

    last_date = df["timestamps"].max()
    today = pd.Timestamp(datetime.today().date())
    if last_date <= today:
        stale_trading_days = len(pd.bdate_range(start=last_date, end=today)) - 1
        if stale_trading_days > 1:
            print(
                f"[daily_forecast] WARNING: {ticker} data is stale -- last available date "
                f"{last_date.date()} is {stale_trading_days} trading days old",
                file=sys.stderr,
            )

    recent_df = df.tail(100).reset_index(drop=True)
    x_df = recent_df[["open", "high", "low", "close", "volume"]]
    x_timestamp = recent_df["timestamps"]
    y_timestamp = pd.Series(future_dates)

    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    ticker_obj = yf.Ticker(ticker)
    try:
        info = ticker_obj.info
    except Exception:
        info = {}

    # Always anchor on the exact same last close Kronos was given as input --
    # never a separately-fetched live price -- so the forecasted % change is
    # measured from Kronos's own baseline, not a moving target.
    current_price = float(x_df["close"].iloc[-1])
    today_open = fetch_today_open(ticker)
    gap_pct = (
        ((today_open - current_price) / current_price) * 100
        if today_open is not None and current_price
        else None
    )

    daily_prices = [float(p) for p in pred_df["close"].tolist()]
    avg_forecast = float(pred_df["close"].mean())
    change_pct = ((avg_forecast - current_price) / current_price) * 100

    signal = "BUY" if change_pct > 2 else ("SELL" if change_pct < -4 else "HOLD")

    ev_ebitda = info.get("enterpriseToEbitda")
    roic = info.get("returnOnEquity")  # proxy for ROIC when true ROIC isn't exposed by yfinance
    valuation_label = get_valuation_label(ev_ebitda)

    # predicted_prices: the 5-day-ahead Kronos forecast, each day's % change
    # measured against this same current_price baseline (never today_open).
    # actual_prices: the last 5 REAL trading days already in df -- no extra
    # yfinance call needed, this is exactly the tail of the data Kronos was
    # given as input.
    predicted_prices = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "price": p,
            "change_pct": ((p - current_price) / current_price * 100) if current_price else None,
        }
        for d, p in zip(future_dates, daily_prices)
    ]
    actual_tail = df.tail(5)
    actual_prices = [
        {"date": ts.strftime("%Y-%m-%d"), "price": float(c)}
        for ts, c in zip(actual_tail["timestamps"], actual_tail["close"])
    ]

    return {
        "ticker": ticker,
        "current_price": current_price,
        "today_open": today_open,
        "gap_pct": gap_pct,
        "avg_forecast": avg_forecast,
        "change_pct": change_pct,
        "signal": signal,
        "ev_ebitda": ev_ebitda,
        "valuation_label": valuation_label,
        "roic": roic,
        "daily_prices": daily_prices,
        "predicted_prices": predicted_prices,
        "actual_prices": actual_prices,
    }


FORECAST_HISTORY_PATH = ROOT / "forecast_history.json"
MAX_HISTORY_SNAPSHOTS = 10


def load_forecast_history():
    if not FORECAST_HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(FORECAST_HISTORY_PATH.read_text())
    except Exception:
        return {}


def save_forecast_history(history):
    FORECAST_HISTORY_PATH.write_text(json.dumps(history))


def record_forecast_snapshot(history, key, result, dates, snapshot_date):
    """Appends one day's forecast under `key` (a portfolio ticker like
    "STB.OL" or a bare screener symbol like "FLUG-B"). Stores the 5 future
    dates + predicted prices and the signal (per spec), plus change_pct and
    current_price -- the Kronos baseline for that day -- so a later lookup
    by symbol+date can render a forecast-vs-actual comparison without
    needing to recompute or re-fetch anything."""
    snapshot = {
        "snapshot_date": snapshot_date,
        "dates": dates,
        "daily_prices": result["daily_prices"],
        "signal": result["signal"],
        "change_pct": result["change_pct"],
        "current_price": result["current_price"],
    }
    history.setdefault(key, []).append(snapshot)
    history[key] = history[key][-MAX_HISTORY_SNAPSHOTS:]


def compute_daily_errors(key, history, actual_prices):
    """For each of the last 5 REAL trading days (actual_prices), finds the
    most recent past forecast snapshot that had predicted that exact date
    and compares it against the actual close. A date can appear in more
    than one past snapshot's 5-day window (e.g. both yesterday's and the
    day before's forecast cover today) -- the most recent snapshot is used
    since it's the freshest prediction for that date.

    Returns (daily_errors, mean_absolute_error_pct); the mean is None if
    nothing has matured yet (e.g. the very first run)."""
    snapshots_newest_first = sorted(history.get(key, []), key=lambda s: s.get("snapshot_date", ""), reverse=True)

    daily_errors = []
    for actual in actual_prices:
        target_date = actual["date"]
        actual_price = actual["price"]
        for snapshot in snapshots_newest_first:
            snap_dates = snapshot.get("dates", [])
            if target_date not in snap_dates:
                continue
            idx = snap_dates.index(target_date)
            predicted = snapshot.get("daily_prices", [None] * len(snap_dates))[idx]
            baseline = snapshot.get("current_price")
            if predicted is None or not actual_price:
                break
            daily_errors.append({
                "date": target_date,
                "predicted": predicted,
                "actual": actual_price,
                "change_pct": ((predicted - baseline) / baseline * 100) if baseline else None,
                "error_pct": abs(predicted - actual_price) / actual_price * 100.0,
            })
            break

    if not daily_errors:
        return daily_errors, None
    mean_error = sum(e["error_pct"] for e in daily_errors) / len(daily_errors)
    return daily_errors, mean_error


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

    # Kronos accuracy diagnostic + history recording, for portfolio tickers
    # AND screener candidates alike (keyed by ticker / bare screener symbol
    # respectively). For each, compare its own last 5 real trading days
    # (already in actual_prices, no extra fetch) against whatever past
    # snapshot had predicted those dates, then record today's forecast for
    # future comparisons. Purely a bookkeeping/reporting pass over the
    # model's outputs -- doesn't touch the model itself.
    forecast_history = load_forecast_history()
    today_str = datetime.today().strftime("%Y-%m-%d")

    def attach_accuracy(key, result):
        daily_errors, mean_error = compute_daily_errors(key, forecast_history, result["actual_prices"])
        result["daily_errors"] = daily_errors
        result["mean_absolute_error_pct"] = mean_error
        result["mape_warning"] = mean_error is not None and mean_error > 5
        record_forecast_snapshot(forecast_history, key, result, dates, today_str)

    for result in results:
        attach_accuracy(result["ticker"], result)

    for symbol, result in screener_forecasts.items():
        if result is not None:
            attach_accuracy(symbol, result)

    save_forecast_history(forecast_history)

    return {"dates": dates, "tickers": results, "screener_forecasts": screener_forecasts}


if __name__ == "__main__":
    print(json.dumps(run_forecast(sys.argv[1:])))
