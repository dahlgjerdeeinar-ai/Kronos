import sys
sys.path.insert(0, '.')
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
    lookback = min(100, len(df))
    x_df = df.loc[:lookback-1, ["open","high","low","close","volume"]]
    x_timestamp = df.loc[:lookback-1, "timestamps"]
    future_dates = pd.bdate_range(start=datetime.today(), periods=6)[1:]
    y_timestamp = pd.Series(future_dates)
    pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    current_price = df["close"].iloc[-1]
    avg_forecast = pred_df["close"].mean()
    change_pct = ((avg_forecast - current_price) / current_price) * 100
    signal = "BUY" if change_pct > 2 else ("SELL" if change_pct < -2 else "HOLD")
    print(f"{ticker}: {current_price:.2f} -> {avg_forecast:.2f} ({change_pct:+.1f}%) | {signal}")
