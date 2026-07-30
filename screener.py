import sqlite3
import urllib.request
from pathlib import Path

DB_URL = "https://lseffer.github.io/stock_screener/stocks.db"
DB_PATH = Path(__file__).resolve().parent / "stocks.db"

MIN_MARKET_CAP = 300_000_000
MAX_EV_EBITDA = 15
MIN_ROIC = 0.15
TOP_N = 10

# Preference order for the "recency" column used to dedupe each time-series
# table down to one row per stock.
DATE_COLUMN_CANDIDATES = ["market_date", "dw_modified", "report_date", "period_end_date", "fiscal_date", "date"]


def get_recommendation(ev_ebitda_ratio):
    if ev_ebitda_ratio < 5:
        return "Strong Buy"
    if ev_ebitda_ratio < 10:
        return "Buy"
    return "Hold"


def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def find_date_column(columns):
    for candidate in DATE_COLUMN_CANDIDATES:
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


def main():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    stocks_cols = table_columns(cur, "stocks")
    prices_cols = table_columns(cur, "prices")
    income_cols = table_columns(cur, "income_statements")
    balance_cols = table_columns(cur, "balance_sheet_statements")

    price_date_col = find_date_column(prices_cols)
    income_date_col = find_date_column(income_cols)
    balance_date_col = find_date_column(balance_cols)

    required = {
        "stocks.isin": "isin" in stocks_cols,
        "stocks.symbol": "symbol" in stocks_cols,
        "stocks.name": "name" in stocks_cols,
        "prices.isin": "isin" in prices_cols,
        "prices.market_cap": "market_cap" in prices_cols,
        "prices.ev_ebitda_ratio": "ev_ebitda_ratio" in prices_cols,
        "prices date column": price_date_col is not None,
        "income_statements.isin": "isin" in income_cols,
        "income_statements.ebit": "ebit" in income_cols,
        "income_statements date column": income_date_col is not None,
        "balance_sheet_statements.isin": "isin" in balance_cols,
        "balance_sheet_statements.total_assets": "total_assets" in balance_cols,
        "balance_sheet_statements.total_current_liabilities": "total_current_liabilities" in balance_cols,
        "balance_sheet_statements date column": balance_date_col is not None,
    }
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        raise RuntimeError(
            f"Missing expected columns/date fields: {missing}\n"
            f"stocks: {stocks_cols}\nprices: {prices_cols}\n"
            f"income_statements: {income_cols}\nbalance_sheet_statements: {balance_cols}"
        )

    query = f"""
        WITH
        {latest_per_isin_cte("prices", price_date_col, "latest_prices")},
        {latest_per_isin_cte("income_statements", income_date_col, "latest_income")},
        {latest_per_isin_cte("balance_sheet_statements", balance_date_col, "latest_balance")}
        SELECT
            s.symbol,
            s.name,
            p.market_cap,
            p.ev_ebitda_ratio,
            1.0 / p.ev_ebitda_ratio AS value_score,
            i.ebit / (b.total_assets - b.total_current_liabilities) AS roic
        FROM stocks s
        JOIN latest_prices p ON s.isin = p.isin
        JOIN latest_income i ON s.isin = i.isin
        JOIN latest_balance b ON s.isin = b.isin
        WHERE p.market_cap >= ?
          AND p.ev_ebitda_ratio > 0
          AND p.ev_ebitda_ratio < ?
          AND (b.total_assets - b.total_current_liabilities) > 0
          AND i.ebit / (b.total_assets - b.total_current_liabilities) >= ?
        ORDER BY p.ev_ebitda_ratio ASC
        LIMIT ?
    """
    rows = cur.execute(query, (MIN_MARKET_CAP, MAX_EV_EBITDA, MIN_ROIC, TOP_N)).fetchall()

    print(f"{'Symbol':<10}{'Name':<30}{'Market Cap':>18}{'EV/EBITDA':>12}{'ROIC':>8}{'Recommendation':>16}")
    for symbol, name, market_cap, ev_ebitda_ratio, value_score, roic in rows:
        recommendation = get_recommendation(ev_ebitda_ratio)
        print(
            f"{symbol:<10}{(name or '')[:29]:<30}{market_cap:>18,.0f}"
            f"{ev_ebitda_ratio:>12.1f}{roic * 100:>7.1f}%{recommendation:>16}"
        )

    conn.close()


if __name__ == "__main__":
    main()
