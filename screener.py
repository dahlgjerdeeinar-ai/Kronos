import sqlite3
import urllib.request
from pathlib import Path

DB_URL = "https://lseffer.github.io/stock_screener/stocks.db"
DB_PATH = Path(__file__).resolve().parent / "stocks.db"

CAP_TIERS = ("Small", "Mid", "Large")
MIN_ROIC_PCT = 20
MAX_EV_EBITDA = 15
MIN_PIOTROSKI = 7
TOP_N = 10


def find_column(columns, *keywords):
    for col in columns:
        lname = col.lower()
        if all(k in lname for k in keywords):
            return col
    return None


def main():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if not tables:
        raise RuntimeError("No tables found in stocks.db")
    table = tables[0]

    columns = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]

    # Column names in the source DB aren't documented, so match by keyword
    # instead of hardcoding names that might not exist.
    col_map = {
        "ticker": find_column(columns, "ticker") or find_column(columns, "symbol"),
        "name": find_column(columns, "name"),
        "tier": find_column(columns, "tier") or find_column(columns, "cap"),
        "roic": find_column(columns, "roic"),
        "ev_ebitda": find_column(columns, "ev", "ebitda"),
        "piotroski": find_column(columns, "piotroski") or find_column(columns, "f_score"),
        "magic": find_column(columns, "magic"),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not find columns for {missing} in table '{table}'. Available columns: {columns}")

    tier_placeholders = ",".join("?" for _ in CAP_TIERS)
    query = f"""
        SELECT {col_map['ticker']}, {col_map['name']}, {col_map['roic']}, {col_map['ev_ebitda']}, {col_map['magic']}
        FROM {table}
        WHERE {col_map['tier']} IN ({tier_placeholders})
          AND {col_map['roic']} > ?
          AND {col_map['ev_ebitda']} < ?
          AND {col_map['piotroski']} >= ?
        ORDER BY {col_map['magic']} DESC
        LIMIT ?
    """
    params = (*CAP_TIERS, MIN_ROIC_PCT, MAX_EV_EBITDA, MIN_PIOTROSKI, TOP_N)
    rows = cur.execute(query, params).fetchall()

    print(f"{'Ticker':<10}{'Company':<30}{'ROIC':>8}{'EV/EBITDA':>12}{'Magic Formula':>16}")
    for ticker, name, roic, ev_ebitda, magic in rows:
        print(f"{ticker:<10}{(name or '')[:29]:<30}{roic:>8.1f}{ev_ebitda:>12.1f}{magic:>16.2f}")

    conn.close()


if __name__ == "__main__":
    main()
