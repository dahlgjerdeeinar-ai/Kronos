import json
import sqlite3
import urllib.request
from pathlib import Path

DB_URL = "https://lseffer.github.io/stock_screener/stocks.db"
DB_PATH = Path(__file__).resolve().parent / "stocks.db"

MIN_MARKET_CAP = 50_000_000
TOP_N = 10

MAGIC_FORMULA_WEIGHT = 0.6
MOMENTUM_WEIGHT = 0.4

# Preference order for the "recency" column used to dedupe each time-series
# table down to one row per stock.
DATE_COLUMN_CANDIDATES = ["market_date", "trade_date", "dw_modified", "report_date", "period_end_date", "fiscal_date", "date"]
PRICE_COLUMN_CANDIDATES = ["close_price", "close", "adjusted_close", "adj_close", "price"]


def get_recommendation(ev_ebitda_ratio):
    if ev_ebitda_ratio < 5:
        return "Strong Buy"
    if ev_ebitda_ratio < 10:
        return "Buy"
    return "Hold"


def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def latest_per_isin_cte(table, date_col, alias):
    return f"""{alias} AS (
        SELECT * FROM (
            SELECT {table}.*, ROW_NUMBER() OVER (PARTITION BY isin ORDER BY {date_col} DESC) AS rn
            FROM {table}
        ) WHERE rn = 1
    )"""


def closest_to_date_cte(table, date_col, alias, target_date_expr):
    return f"""{alias} AS (
        SELECT * FROM (
            SELECT {table}.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY isin
                       ORDER BY ABS(JULIANDAY({date_col}) - JULIANDAY({target_date_expr})) ASC
                   ) AS rn
            FROM {table}
        ) WHERE rn = 1
    )"""


def run_screener():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    stocks_cols = table_columns(cur, "stocks")
    prices_cols = table_columns(cur, "prices")
    price_history_cols = table_columns(cur, "price_history")

    price_date_col = find_column(prices_cols, DATE_COLUMN_CANDIDATES)
    ph_date_col = find_column(price_history_cols, DATE_COLUMN_CANDIDATES)
    ph_close_col = find_column(price_history_cols, PRICE_COLUMN_CANDIDATES)

    required = {
        "stocks.isin": "isin" in stocks_cols,
        "stocks.symbol": "symbol" in stocks_cols,
        "stocks.name": "name" in stocks_cols,
        "stocks.sector": "sector" in stocks_cols,
        "prices.isin": "isin" in prices_cols,
        "prices.market_cap": "market_cap" in prices_cols,
        "prices.ev_ebitda_ratio": "ev_ebitda_ratio" in prices_cols,
        "prices date column": price_date_col is not None,
        "price_history.isin": "isin" in price_history_cols,
        "price_history date column": ph_date_col is not None,
        "price_history close-price column": ph_close_col is not None,
    }
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        raise RuntimeError(
            f"Missing expected columns/date fields: {missing}\n"
            f"stocks: {stocks_cols}\nprices: {prices_cols}\nprice_history: {price_history_cols}"
        )

    query = f"""
        WITH
        {latest_per_isin_cte("prices", price_date_col, "latest_prices")},
        {latest_per_isin_cte("price_history", ph_date_col, "latest_ph")},
        {closest_to_date_cte("price_history", ph_date_col, "base_ph", "DATE('now', '-6 months')")},
        momentum AS (
            SELECT
                l.isin,
                (l.{ph_close_col} - b.{ph_close_col}) / b.{ph_close_col} * 100.0 AS momentum_6m
            FROM latest_ph l
            JOIN base_ph b ON l.isin = b.isin
            WHERE b.{ph_close_col} > 0
        ),
        eligible AS (
            SELECT
                s.symbol,
                s.name,
                s.sector,
                p.market_cap,
                p.ev_ebitda_ratio,
                m.momentum_6m
            FROM stocks s
            JOIN latest_prices p ON s.isin = p.isin
            JOIN momentum m ON s.isin = m.isin
            WHERE p.market_cap >= ?
              AND p.ev_ebitda_ratio > 0
              AND m.momentum_6m > 0
              AND s.symbol NOT LIKE '%-TEMP'
        ),
        ranked AS (
            SELECT
                *,
                RANK() OVER (ORDER BY ev_ebitda_ratio ASC) AS magic_formula_rank,
                RANK() OVER (ORDER BY momentum_6m DESC) AS momentum_rank
            FROM eligible
        )
        SELECT
            symbol, name, sector, market_cap, ev_ebitda_ratio, momentum_6m,
            ({MAGIC_FORMULA_WEIGHT} * magic_formula_rank + {MOMENTUM_WEIGHT} * momentum_rank) AS combined_score
        FROM ranked
        ORDER BY combined_score ASC
        LIMIT ?
    """
    rows = cur.execute(query, (MIN_MARKET_CAP, TOP_N)).fetchall()
    conn.close()

    results = []
    for symbol, name, sector, market_cap, ev_ebitda_ratio, momentum_6m, combined_score in rows:
        results.append({
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "market_cap": market_cap,
            "ev_ebitda": ev_ebitda_ratio,
            "momentum_6m": momentum_6m,
            "recommendation": get_recommendation(ev_ebitda_ratio),
        })
    return results


if __name__ == "__main__":
    print(json.dumps(run_screener()))
