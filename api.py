"""
Flask API for Kronos stock-price forecasting.

Endpoints
---------
POST /predict
    Body: {"ticker": "EQNR.OL", "pred_days": 10}
    Returns: {"ticker": ..., "predictions": [...]}

GET /health
    Returns: {"status": "ok"}

Run locally:
    python api.py

Or with gunicorn (used by Railway / Modal):
    gunicorn api:app --bind 0.0.0.0:${PORT:-8080}
"""

import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

# Make the Kronos model package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos-master"))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── app setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))
LOOKBACK_ROWS = int(os.environ.get("KRONOS_LOOKBACK", "400"))

# Lazy-loaded globals (loaded on first request or at startup)
_predictor: KronosPredictor | None = None


def get_predictor() -> KronosPredictor:
    global _predictor
    if _predictor is None:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        _predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
    return _predictor


def fetch_ohlcv(ticker: str, n_rows: int) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and return a clean DataFrame."""
    raw = yf.download(
        ticker,
        period=f"{n_rows + 200}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.rename(columns={"adj close": "close"})
    raw = raw[["open", "high", "low", "close", "volume"]].dropna().reset_index()
    raw = raw.rename(columns={"Date": "timestamps", "Datetime": "timestamps",
                               "date": "timestamps", "index": "timestamps"})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"])

    if "amount" not in raw.columns:
        raw["amount"] = raw["close"] * raw["volume"]

    return raw.tail(n_rows).reset_index(drop=True)


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    body = request.get_json(force=True, silent=True) or {}
    ticker = body.get("ticker", "EQNR.OL").strip().upper()
    pred_days = int(body.get("pred_days", 10))

    if pred_days < 1 or pred_days > 60:
        return jsonify({"error": "pred_days must be between 1 and 60"}), 400

    try:
        df = fetch_ohlcv(ticker, LOOKBACK_ROWS + pred_days)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Data fetch failed: {exc}"}), 502

    actual_lookback = max(len(df) - pred_days, 50)
    pred_days = len(df) - actual_lookback

    x_df = df.loc[: actual_lookback - 1,
                  ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: actual_lookback - 1, "timestamps"]

    last_date = df["timestamps"].iloc[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_days
    )
    y_timestamp = pd.Series(future_dates, name="timestamps")

    try:
        predictor = get_predictor()
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_days,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    records = []
    for ts, row in pred_df.iterrows():
        records.append({
            "date": str(ts)[:10],
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row["volume"]),
        })

    return jsonify({
        "ticker": ticker,
        "model": MODEL_NAME,
        "lookback_rows": actual_lookback,
        "pred_days": len(records),
        "predictions": records,
    })


# ── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Pre-load the model so the first request is fast
    print("Pre-loading Kronos model …")
    get_predictor()
    print(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
