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
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS = ["STB.OL", "TEL.OL", "EQNR.OL", "NOVO-B.CO"]

# Suffix probe order for mapping bare screener symbols (e.g. "BIOMAR") to a
# tradable yfinance ticker: Denmark, Sweden, Finland, Norway.
NORDIC_SUFFIXES = [".CO", ".ST", ".HE", ".OL"]

# Run predictor.predict() this many times per ticker and average the
# resulting close prices -- pure inference-time variance reduction (Kronos
# samples autoregressively), no change to the model itself.
SAMPLE_RUNS = 3

# Matches the Kronos paper's own example (examples/prediction_example.py),
# which uses lookback=400 against a max_context=512 predictor. Capped at
# however much of it yfinance actually returns.
MAX_LOOKBACK = 400

GAP_WARNING_THRESHOLD = 0.5  # flag a >50% single-day close-to-close move


def validate_input_data(df, ticker):
    """Sanity-checks the OHLCV window about to be handed to Kronos: no NaNs,
    no zero-volume days, no single-day 50%+ price gaps (a classic sign of
    unadjusted splits or bad data). Purely diagnostic -- logs to stderr and
    never blocks a forecast, since yfinance's auto_adjust=True already
    handles real splits/dividends upstream."""
    warnings = []

    nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
    if nan_counts.any():
        bad = {k: int(v) for k, v in nan_counts[nan_counts > 0].items()}
        warnings.append(f"{ticker}: NaN values present -- {bad}")

    zero_volume_days = int((df["volume"] == 0).sum())
    if zero_volume_days:
        warnings.append(f"{ticker}: {zero_volume_days} day(s) with zero volume")

    pct_change = df["close"].pct_change().abs()
    gap_mask = pct_change > GAP_WARNING_THRESHOLD
    if gap_mask.any():
        gap_dates = df.loc[gap_mask, "timestamps"].dt.strftime("%Y-%m-%d").tolist()
        warnings.append(
            f"{ticker}: {int(gap_mask.sum())} day(s) with a >{GAP_WARNING_THRESHOLD*100:.0f}% "
            f"close-to-close gap on {gap_dates}"
        )

    for w in warnings:
        print(f"[daily_forecast] DATA QUALITY WARNING: {w}", file=sys.stderr)
    return warnings


def ensure_naive_timestamps(series, label, ticker):
    """Kronos's calc_time_stamps() (model/kronos.py) calls
    x_timestamp.dt.minute/.dt.hour/etc, which requires a pandas Series of
    datetime64 dtype -- not a raw DatetimeIndex, which has no .dt accessor.
    We already pass Series everywhere (matching the official example), so
    this only guards against yfinance occasionally returning a
    timezone-aware index: tz-aware and tz-naive stamps mixed together would
    make hour/day arithmetic inconsistent between x_timestamp and
    y_timestamp, so strip tz info if present and log it."""
    series = pd.to_datetime(series)
    tz = getattr(series.dt, "tz", None)
    if tz is not None:
        print(
            f"[daily_forecast] TIMESTAMP WARNING: {ticker} {label} was timezone-aware "
            f"({tz}) -- converted to naive for Kronos compatibility",
            file=sys.stderr,
        )
        series = series.dt.tz_localize(None)
    return series


def print_kronos_usage_analysis():
    """Static comparison against Kronos-master/examples/prediction_example.py
    and model/kronos.py -- doesn't depend on any runtime data, so it's
    printed once per run rather than per ticker."""
    lines = [
        "=== KRONOS USAGE ANALYSIS (vs examples/prediction_example.py) ===",
        "Parameters: T=1.0 (match), top_p=0.9 (match), top_k=0/default (match), "
        "sample_count=1 per call (match; SAMPLE_RUNS averages multiple such calls "
        "externally, not via Kronos's own sample_count), max_context=512 (match).",
        "pred_len=5 here vs 120 in the example -- expected difference, the example's "
        "is arbitrary for its own intraday dataset and has no bearing on correctness.",
        "Input columns: ['open','high','low','close','volume'] passed. KronosPredictor "
        "only strictly requires price_cols=['open','high','low','close']; 'volume' is "
        "optional (defaults to 0.0). We omit 'amount' explicitly, but "
        "KronosPredictor.predict() auto-derives amount = volume * mean(price_cols) "
        "whenever volume is present and amount is not (model/kronos.py) -- exactly "
        "what we would compute ourselves, so this is not a gap.",
        "x_timestamp / y_timestamp: both passed as pandas Series of datetime64, "
        "matching the example -- calc_time_stamps() calls x_timestamp.dt.<attr>, "
        "which requires a Series (a raw DatetimeIndex has no .dt accessor and would "
        "raise). Per-ticker timezone-naive check logged below.",
        f"Lookback: MAX_LOOKBACK={MAX_LOOKBACK} (matches the example's own lookback=400, "
        "within the predictor's max_context=512), using df.tail(lookback) -- most recent "
        "data, not df.head(). Per-ticker lookback actually used (bounded by however much "
        "history yfinance returns) logged below.",
    ]
    for line in lines:
        print(f"[daily_forecast] {line}", file=sys.stderr)


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
        # 2y so there's comfortably enough history to reach MAX_LOOKBACK=400
        # trading days of context (~1.5y of actual trading days).
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
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

    lookback = min(MAX_LOOKBACK, len(df))
    print(
        f"[daily_forecast] ANALYSIS {ticker}: lookback={lookback} "
        f"(of {len(df)} trading days downloaded, capped at MAX_LOOKBACK={MAX_LOOKBACK})",
        file=sys.stderr,
    )
    recent_df = df.tail(lookback).reset_index(drop=True)  # most recent data, not df.head()
    x_df = recent_df[["open", "high", "low", "close", "volume"]]

    validate_input_data(recent_df, ticker)

    x_timestamp = ensure_naive_timestamps(recent_df["timestamps"], "x_timestamp", ticker)
    y_timestamp = ensure_naive_timestamps(pd.Series(future_dates), "y_timestamp", ticker)

    # Run inference SAMPLE_RUNS times and average the close-price path across
    # runs -- Kronos samples autoregressively (T/top_p), so repeated calls on
    # identical input vary; averaging reduces that variance without touching
    # the model. Reports the per-day std dev across runs as a variance signal.
    run_closes = []
    for _ in range(SAMPLE_RUNS):
        run_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )
        run_closes.append(run_df["close"].to_numpy(dtype=float))
    runs_array = np.array(run_closes)  # shape (SAMPLE_RUNS, 5)
    daily_prices = runs_array.mean(axis=0).tolist()
    daily_prices_std = runs_array.std(axis=0).tolist()
    print(
        f"[daily_forecast] ANALYSIS {ticker}: {SAMPLE_RUNS}-run std dev per day = "
        f"{[round(s, 4) for s in daily_prices_std]}, mean std = {np.mean(daily_prices_std):.4f}",
        file=sys.stderr,
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

    daily_prices = [float(p) for p in daily_prices]
    avg_forecast = float(np.mean(daily_prices))
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
    print_kronos_usage_analysis()

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # Always the next 5 trading days starting tomorrow, regardless of what
    # day it is or whether today itself is a trading day. bdate_range rolls
    # a weekend/holiday start forward to the next business day automatically.
    future_dates = pd.bdate_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=5)
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

    screener_successes = sum(1 for v in screener_forecasts.values() if v is not None)
    print(
        f"[daily_forecast] ANALYSIS summary: {len(results)}/{len(TICKERS)} portfolio tickers "
        f"and {screener_successes}/{len(screener_symbols or [])} screener candidates forecasted "
        f"successfully, each averaged over SAMPLE_RUNS={SAMPLE_RUNS} runs (see per-ticker lines above "
        "for lookback/variance/data-quality detail).",
        file=sys.stderr,
    )

    return {"dates": dates, "tickers": results, "screener_forecasts": screener_forecasts}


if __name__ == "__main__":
    print(json.dumps(run_forecast(sys.argv[1:])))
