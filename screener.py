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


def get_schema(cur):
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    return {table: [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()] for table in tables}


def find_field(schema, *keywords):
    for table, columns in schema.items():
        col = find_column(columns, *keywords)
        if col:
            return table, col
    return None, None


def find_join_key(schema, table_a, table_b):
    common = set(schema[table_a]) & set(schema[table_b])
    if not common:
        return None
    for name in ("ticker", "symbol", "stock_id", "id"):
        for col in common:
            if col.lower() == name:
                return col
    return sorted(common)[0]


def main():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    schema = get_schema(cur)
    if not schema:
        raise RuntimeError("No tables found in stocks.db")

    # Fields can live in any table, so search all of them instead of
    # assuming a single table holds everything.
    field_keywords = {
        "ticker": [("ticker",), ("symbol",)],
        "name": [("name",)],
        "tier": [("tier",), ("cap",)],
        "roic": [("roic",)],
        "ev_ebitda": [("ev", "ebitda")],
        "piotroski": [("piotroski",), ("f_score",)],
        "magic": [("magic",)],
    }

    located = {}
    for field, keyword_options in field_keywords.items():
        for keywords in keyword_options:
            table, col = find_field(schema, *keywords)
            if table:
                located[field] = (table, col)
                break
        else:
            located[field] = (None, None)

    missing = [f for f, (t, c) in located.items() if t is None]
    if missing:
        raise RuntimeError(f"Could not locate columns for {missing}. Schema: {schema}")

    tables_needed = sorted({t for t, c in located.values()})
    base_table = tables_needed[0]
    from_clause = base_table
    for other in tables_needed[1:]:
        key = find_join_key(schema, base_table, other)
        if not key:
            raise RuntimeError(f"No common join key found between '{base_table}' and '{other}'. Schema: {schema}")
        from_clause += f" JOIN {other} ON {base_table}.{key} = {other}.{key}"

    select_exprs = {field: f"{table}.{col}" for field, (table, col) in located.items()}

    tier_placeholders = ",".join("?" for _ in CAP_TIERS)
    query = f"""
        SELECT {select_exprs['ticker']}, {select_exprs['name']}, {select_exprs['roic']}, {select_exprs['ev_ebitda']}, {select_exprs['magic']}
        FROM {from_clause}
        WHERE {select_exprs['tier']} IN ({tier_placeholders})
          AND {select_exprs['roic']} > ?
          AND {select_exprs['ev_ebitda']} < ?
          AND {select_exprs['piotroski']} >= ?
        ORDER BY {select_exprs['magic']} DESC
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
