import sqlite3
import urllib.request
from pathlib import Path

DB_URL = "https://lseffer.github.io/stock_screener/stocks.db"
DB_PATH = Path(__file__).resolve().parent / "stocks.db"

MIN_MARKET_CAP = 300_000_000
MAX_EV_EBITDA = 15
TOP_N = 10


def get_recommendation(ev_ebitda_ratio):
    if ev_ebitda_ratio < 5:
        return "Strong Buy"
    if ev_ebitda_ratio < 10:
        return "Buy"
    return "Hold"


def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def main():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    stocks_cols = table_columns(cur, "stocks")
    prices_cols = table_columns(cur, "prices")
    required = {
        "stocks.isin": "isin" in stocks_cols,
        "stocks.symbol": "symbol" in stocks_cols,
        "stocks.name": "name" in stocks_cols,
        "prices.isin": "isin" in prices_cols,
        "prices.market_cap": "market_cap" in prices_cols,
        "prices.ev_ebitda_ratio": "ev_ebitda_ratio" in prices_cols,
    }
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        raise RuntimeError(
            f"Missing expected columns: {missing}\n"
            f"stocks columns: {stocks_cols}\nprices columns: {prices_cols}"
        )

    query = """
        SELECT s.symbol, s.name, p.market_cap, p.ev_ebitda_ratio, 1.0 / p.ev_ebitda_ratio AS value_score
        FROM stocks s
        JOIN prices p ON s.isin = p.isin
        WHERE p.market_cap >= ?
          AND p.ev_ebitda_ratio > 0
          AND p.ev_ebitda_ratio < ?
        ORDER BY p.ev_ebitda_ratio ASC
        LIMIT ?
    """
    rows = cur.execute(query, (MIN_MARKET_CAP, MAX_EV_EBITDA, TOP_N)).fetchall()

    print(f"{'Symbol':<10}{'Name':<30}{'Market Cap':>18}{'EV/EBITDA':>12}{'Recommendation':>16}")
    for symbol, name, market_cap, ev_ebitda_ratio, value_score in rows:
        recommendation = get_recommendation(ev_ebitda_ratio)
        print(f"{symbol:<10}{(name or '')[:29]:<30}{market_cap:>18,.0f}{ev_ebitda_ratio:>12.1f}{recommendation:>16}")

    conn.close()


if __name__ == "__main__":
    main()
