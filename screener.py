"""Multi-factor Nordic equity quant screener.

Data availability constraints (from the confirmed schema) forced two
substitutions in the Piotroski F-Score, documented where they're computed:
  - "no dilution" (share count change) is omitted -- no shares-outstanding
    field exists anywhere in the confirmed schema. The remaining 8 criteria
    are summed and scaled to a 9-point-equivalent score.
  - "liquidity" uses cash / total_current_liabilities (a cash-ratio proxy)
    instead of a true current ratio, since total_current_assets isn't in
    the confirmed schema (only total_current_liabilities is).

"Gross Margin trend" (Quality, 8%) and "Gross margin improvement" (Growth,
4%) are the same underlying value (latest gross margin - prior gross
margin); the output schema only has one gross_margin_trend field, so it's
computed once and weighted into both categories (12% combined).
"""
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DB_URL = "https://lseffer.github.io/stock_screener/stocks.db"
DB_PATH = Path(__file__).resolve().parent / "stocks.db"

MIN_MARKET_CAP = 50_000_000
INITIAL_TOP_N = 20
FINAL_TOP_N = 10
MIN_VALID_FACTORS = 6

OSEBX_TICKER = "^OSEBX"

DATE_COLUMN_CANDIDATES = ["market_date", "trade_date", "dw_modified", "report_date", "period_end_date", "fiscal_date", "date"]
PRICE_COLUMN_CANDIDATES = ["close_price", "close", "adjusted_close", "adj_close", "price"]

# (factor_name, category, weight_pct, higher_is_better). gross_margin_trend
# intentionally appears twice (quality + growth) -- see module docstring.
FACTOR_WEIGHTS = [
    ("earnings_yield",     "value",     10, True),
    ("fcf_yield",          "value",      8, True),
    ("pb_ratio",           "value",      7, False),
    ("piotroski",          "quality",   10, True),
    ("gross_margin_trend", "quality",    8, True),
    ("debt_equity",        "quality",    7, False),
    ("momentum_12m",       "momentum",  12, True),
    ("momentum_6m",        "momentum",   8, True),
    ("high52w_ratio",      "momentum",   5, True),
    ("ivol",               "risk",      10, False),
    ("beta",               "risk",       5, False),
    ("revenue_growth",     "growth",     6, True),
    ("gross_margin_trend", "growth",     4, True),
]
FACTOR_NAMES = sorted({name for name, *_ in FACTOR_WEIGHTS})


def get_recommendation(quant_score):
    if quant_score >= 80:
        return "Strong Buy"
    if quant_score >= 60:
        return "Buy"
    if quant_score >= 40:
        return "Hold"
    return None  # excluded


def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def nth_per_isin_cte(table, date_col, alias, n):
    return f"""{alias} AS (
        SELECT * FROM (
            SELECT {table}.*, ROW_NUMBER() OVER (PARTITION BY isin ORDER BY {date_col} DESC) AS rn
            FROM {table}
        ) WHERE rn = {n}
    )"""


def in_clause(items):
    return "(" + ",".join("?" for _ in items) + ")"


def safe_div(a, b):
    try:
        if a is None or b is None or b == 0:
            return None
        return a / b
    except (TypeError, ZeroDivisionError):
        return None


# --------------------------------------------------------------------------
# Fundamentals-based factors
# --------------------------------------------------------------------------

def compute_piotroski(fin):
    try:
        roa_latest = safe_div(fin["ni_latest"], fin["assets_latest"])
        roa_prior = safe_div(fin["ni_prior"], fin["assets_prior"])
        cfo_latest = fin["cfo_latest"]
        accrual = safe_div(fin["ni_latest"] - fin["cfo_latest"], fin["assets_latest"]) if fin["cfo_latest"] is not None else None
        leverage_latest = safe_div(fin["liab_latest"], fin["assets_latest"])
        leverage_prior = safe_div(fin["liab_prior"], fin["assets_prior"])
        liquidity_latest = safe_div(fin["cash_latest"], fin["curr_liab_latest"])
        liquidity_prior = safe_div(fin["cash_prior"], fin["curr_liab_prior"])
        margin_latest = safe_div(fin["gp_latest"], fin["revenue_latest"])
        margin_prior = safe_div(fin["gp_prior"], fin["revenue_prior"])
        turnover_latest = safe_div(fin["revenue_latest"], fin["assets_latest"])
        turnover_prior = safe_div(fin["revenue_prior"], fin["assets_prior"])
    except KeyError:
        return None

    criteria = [
        roa_latest is not None and roa_latest > 0,
        cfo_latest is not None and cfo_latest > 0,
        roa_latest is not None and roa_prior is not None and roa_latest > roa_prior,
        accrual is not None and accrual < 0,
        leverage_latest is not None and leverage_prior is not None and leverage_latest < leverage_prior,
        liquidity_latest is not None and liquidity_prior is not None and liquidity_latest > liquidity_prior,
        margin_latest is not None and margin_prior is not None and margin_latest > margin_prior,
        turnover_latest is not None and turnover_prior is not None and turnover_latest > turnover_prior,
    ]
    known = [c for c in criteria if c is not None]
    if len(known) < 4:  # too little data to trust an 8-criteria score
        return None
    points = sum(1 for c in known if c)
    return points * (9.0 / len(known))


def compute_fundamental_factors(fin):
    earnings_yield = safe_div(fin["ebit_latest"], (fin["market_cap"] or 0) + (fin["liab_latest"] or 0) - (fin["cash_latest"] or 0))
    fcf = None
    if fin["cfo_latest"] is not None and fin["capex_latest"] is not None:
        fcf = fin["cfo_latest"] - fin["capex_latest"]
    fcf_yield = safe_div(fcf, fin["market_cap"])
    pb_ratio = safe_div(fin["market_cap"], fin["equity_latest"])

    margin_latest = safe_div(fin["gp_latest"], fin["revenue_latest"])
    margin_prior = safe_div(fin["gp_prior"], fin["revenue_prior"])
    gross_margin_trend = (margin_latest - margin_prior) if margin_latest is not None and margin_prior is not None else None

    revenue_growth = None
    if fin["revenue_latest"] is not None and fin["revenue_prior"]:
        revenue_growth = safe_div(fin["revenue_latest"] - fin["revenue_prior"], fin["revenue_prior"])

    return {
        "earnings_yield": earnings_yield,
        "fcf_yield": fcf_yield,
        "pb_ratio": pb_ratio,
        "gross_margin_trend": gross_margin_trend,
        "revenue_growth": revenue_growth,
        "piotroski": compute_piotroski(fin),
    }


# --------------------------------------------------------------------------
# Price-history-based factors
# --------------------------------------------------------------------------

def price_near(hist, target_date):
    if hist.empty:
        return None
    deltas = (hist["date"] - target_date).abs()
    return float(hist.loc[deltas.idxmin(), "close"])


def compute_price_factors(hist):
    """hist: DataFrame with columns [date, close] for one stock, sorted ascending."""
    if hist.empty:
        return {"momentum_6m": None, "momentum_12m": None, "high52w_ratio": None}
    hist = hist.sort_values("date")
    latest_date = hist["date"].iloc[-1]
    latest_price = float(hist["close"].iloc[-1])

    price_6m = price_near(hist, latest_date - pd.Timedelta(days=182))
    price_12m = price_near(hist, latest_date - pd.Timedelta(days=365))

    momentum_6m = safe_div(latest_price - price_6m, price_6m)
    momentum_12m = safe_div(latest_price - price_12m, price_12m)
    if momentum_6m is not None:
        momentum_6m *= 100.0
    if momentum_12m is not None:
        momentum_12m *= 100.0

    window = hist[hist["date"] >= latest_date - pd.Timedelta(weeks=52)]
    high52w = float(window["close"].max()) if not window.empty else None
    high52w_ratio = safe_div(latest_price, high52w)

    return {"momentum_6m": momentum_6m, "momentum_12m": momentum_12m, "high52w_ratio": high52w_ratio}


def compute_beta_ivol(stock_hist, market_hist):
    """Both are DataFrames with columns [date, close]. Returns (beta, ivol) or (None, None)."""
    if stock_hist.empty or market_hist.empty:
        return None, None
    merged = pd.merge(stock_hist, market_hist, on="date", suffixes=("_stock", "_mkt")).sort_values("date")
    if len(merged) < 30:
        return None, None
    stock_returns = merged["close_stock"].pct_change().dropna().to_numpy()
    market_returns = merged["close_mkt"].pct_change().dropna().to_numpy()
    n = min(len(stock_returns), len(market_returns))
    if n < 20:
        return None, None
    stock_returns, market_returns = stock_returns[-n:], market_returns[-n:]

    var_mkt = np.var(market_returns, ddof=1)
    if var_mkt == 0:
        return None, None
    cov = np.cov(stock_returns, market_returns, ddof=1)[0][1]
    beta = cov / var_mkt
    alpha = np.mean(stock_returns) - beta * np.mean(market_returns)
    residuals = stock_returns - (alpha + beta * market_returns)
    ivol = float(np.std(residuals, ddof=1))
    return float(beta), ivol


def fetch_debt_equity(yahoo_ticker):
    if not yahoo_ticker:
        return None
    try:
        return yf.Ticker(yahoo_ticker).info.get("debtToEquity")
    except Exception:
        return None


def fetch_osebx_history():
    try:
        df = yf.download(OSEBX_TICKER, period="14mo", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return pd.DataFrame({"date": pd.to_datetime(df.index).tz_localize(None), "close": df["Close"].values})


# --------------------------------------------------------------------------
# Percentile-rank normalization + per-stock weight redistribution
# --------------------------------------------------------------------------

def compute_quant_scores(df, factor_weights):
    """df: one row per stock, with columns for every unique factor name (NaN
    where missing) plus identification columns. Returns (df_with_scores,
    coverage_pct_by_factor)."""
    ranks = pd.DataFrame(index=df.index)
    higher_is_better = {}
    for name, _, _, hib in factor_weights:
        higher_is_better[name] = hib
    for name in {n for n, *_ in factor_weights}:
        if name in df.columns:
            ranks[name] = df[name].rank(pct=True, ascending=higher_is_better[name]) * 100.0
        else:
            ranks[name] = np.nan

    categories = {}
    for name, category, weight, _ in factor_weights:
        categories.setdefault(category, []).append((name, weight))

    total = len(df)
    coverage_count = {name: 0 for name in ranks.columns}
    scores = []
    valid_counts = []
    for idx in df.index:
        score = 0.0
        valid_names = set()
        for _category, items in categories.items():
            nominal_total = sum(w for _, w in items)
            available = [(n, w) for n, w in items if pd.notna(ranks.at[idx, n])]
            if not available:
                continue
            available_weight_sum = sum(w for _, w in available)
            scale = nominal_total / available_weight_sum
            for n, w in available:
                score += (w * scale) / 100.0 * ranks.at[idx, n]
                valid_names.add(n)
        for n in valid_names:
            coverage_count[n] += 1
        scores.append(score)
        valid_counts.append(len(valid_names))

    out = df.copy()
    out["quant_score"] = scores
    out["_valid_factor_count"] = valid_counts
    coverage_pct = {k: (v / total * 100.0 if total else 0.0) for k, v in coverage_count.items()}
    return out, coverage_pct


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_screener():
    urllib.request.urlretrieve(DB_URL, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    stocks_cols = table_columns(cur, "stocks")
    prices_cols = table_columns(cur, "prices")
    income_cols = table_columns(cur, "income_statements")
    balance_cols = table_columns(cur, "balance_sheet_statements")
    cashflow_cols = table_columns(cur, "cash_flow_statements")
    price_history_cols = table_columns(cur, "price_history")
    has_yahoo_ticker = "yahoo_ticker" in stocks_cols

    price_date_col = find_column(prices_cols, DATE_COLUMN_CANDIDATES)
    income_date_col = find_column(income_cols, DATE_COLUMN_CANDIDATES)
    balance_date_col = find_column(balance_cols, DATE_COLUMN_CANDIDATES)
    cashflow_date_col = find_column(cashflow_cols, DATE_COLUMN_CANDIDATES)
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
        "income_statements.isin": "isin" in income_cols,
        "income_statements.total_revenue": "total_revenue" in income_cols,
        "income_statements.gross_profit": "gross_profit" in income_cols,
        "income_statements.ebit": "ebit" in income_cols,
        "income_statements.net_income": "net_income" in income_cols,
        "income_statements date column": income_date_col is not None,
        "balance_sheet_statements.isin": "isin" in balance_cols,
        "balance_sheet_statements.total_assets": "total_assets" in balance_cols,
        "balance_sheet_statements.total_liab": "total_liab" in balance_cols,
        "balance_sheet_statements.total_stockholder_equity": "total_stockholder_equity" in balance_cols,
        "balance_sheet_statements.total_current_liabilities": "total_current_liabilities" in balance_cols,
        "balance_sheet_statements.cash": "cash" in balance_cols,
        "balance_sheet_statements date column": balance_date_col is not None,
        "cash_flow_statements.isin": "isin" in cashflow_cols,
        "cash_flow_statements.total_cash_from_operating_activities": "total_cash_from_operating_activities" in cashflow_cols,
        "cash_flow_statements.capital_expenditures": "capital_expenditures" in cashflow_cols,
        "cash_flow_statements date column": cashflow_date_col is not None,
        "price_history.isin": "isin" in price_history_cols,
        "price_history date column": ph_date_col is not None,
        "price_history close-price column": ph_close_col is not None,
    }
    missing = [k for k, ok in required.items() if not ok]
    if missing:
        raise RuntimeError(
            f"Missing expected columns/date fields: {missing}\n"
            f"stocks: {stocks_cols}\nprices: {prices_cols}\nincome_statements: {income_cols}\n"
            f"balance_sheet_statements: {balance_cols}\ncash_flow_statements: {cashflow_cols}\n"
            f"price_history: {price_history_cols}"
        )

    # ---- filter-funnel debug (independent counts, not cumulative) ----
    total_stocks = cur.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    latest_prices_cte = nth_per_isin_cte("prices", price_date_col, "lp", 1)
    pass_market_cap, pass_ev_ebitda, pass_not_temp = cur.execute(
        f"""
        WITH {latest_prices_cte}
        SELECT
            SUM(CASE WHEN lp.market_cap >= ? THEN 1 ELSE 0 END),
            SUM(CASE WHEN lp.ev_ebitda_ratio > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN s.symbol NOT LIKE '%-TEMP' THEN 1 ELSE 0 END)
        FROM stocks s JOIN lp ON s.isin = lp.isin
        """,
        (MIN_MARKET_CAP,),
    ).fetchone()
    print(f"[screener] total stocks in DB: {total_stocks}", file=sys.stderr)
    print(f"[screener] pass market_cap >= {MIN_MARKET_CAP:,}: {pass_market_cap}", file=sys.stderr)
    print(f"[screener] pass ev_ebitda_ratio > 0: {pass_ev_ebitda}", file=sys.stderr)
    print(f"[screener] pass symbol NOT LIKE '%-TEMP': {pass_not_temp}", file=sys.stderr)

    # ---- candidate pool: stocks + latest prices, SQL-side pre-filters ----
    candidate_rows = cur.execute(
        f"""
        WITH {latest_prices_cte}
        SELECT s.isin, s.symbol, s.name, s.sector,
               {"s.yahoo_ticker" if has_yahoo_ticker else "NULL"} AS yahoo_ticker,
               lp.market_cap, lp.ev_ebitda_ratio
        FROM stocks s
        JOIN lp ON s.isin = lp.isin
        WHERE lp.market_cap >= ?
          AND lp.ev_ebitda_ratio > 0
          AND s.symbol NOT LIKE '%-TEMP'
        """,
        (MIN_MARKET_CAP,),
    ).fetchall()
    candidates = {
        r[0]: {
            "isin": r[0], "symbol": r[1], "name": r[2], "sector": r[3],
            "yahoo_ticker": r[4], "market_cap": r[5], "ev_ebitda_ratio": r[6],
        }
        for r in candidate_rows
    }
    print(f"[screener] candidates after market_cap/ev_ebitda/-TEMP pre-filters: {len(candidates)}", file=sys.stderr)

    if not candidates:
        conn.close()
        return []

    isins = list(candidates.keys())

    # ---- fundamentals: latest + prior period per statement table ----
    income_rows = cur.execute(
        f"""
        WITH
        {nth_per_isin_cte("income_statements", income_date_col, "i1", 1)},
        {nth_per_isin_cte("income_statements", income_date_col, "i2", 2)}
        SELECT i1.isin, i1.total_revenue, i1.gross_profit, i1.ebit, i1.net_income,
               i2.total_revenue, i2.gross_profit, i2.net_income
        FROM i1 LEFT JOIN i2 ON i1.isin = i2.isin
        WHERE i1.isin IN {in_clause(isins)}
        """,
        isins,
    ).fetchall()
    balance_rows = cur.execute(
        f"""
        WITH
        {nth_per_isin_cte("balance_sheet_statements", balance_date_col, "b1", 1)},
        {nth_per_isin_cte("balance_sheet_statements", balance_date_col, "b2", 2)}
        SELECT b1.isin, b1.total_assets, b1.total_liab, b1.total_stockholder_equity,
               b1.total_current_liabilities, b1.cash,
               b2.total_assets, b2.total_liab, b2.total_current_liabilities, b2.cash
        FROM b1 LEFT JOIN b2 ON b1.isin = b2.isin
        WHERE b1.isin IN {in_clause(isins)}
        """,
        isins,
    ).fetchall()
    cashflow_rows = cur.execute(
        f"""
        WITH {nth_per_isin_cte("cash_flow_statements", cashflow_date_col, "c1", 1)}
        SELECT c1.isin, c1.total_cash_from_operating_activities, c1.capital_expenditures
        FROM c1
        WHERE c1.isin IN {in_clause(isins)}
        """,
        isins,
    ).fetchall()

    financials = {isin: {} for isin in isins}
    for isin, rev1, gp1, ebit1, ni1, rev2, gp2, ni2 in income_rows:
        financials[isin].update({
            "revenue_latest": rev1, "gp_latest": gp1, "ebit_latest": ebit1, "ni_latest": ni1,
            "revenue_prior": rev2, "gp_prior": gp2, "ni_prior": ni2,
        })
    for isin, assets1, liab1, equity1, curr_liab1, cash1, assets2, liab2, curr_liab2, cash2 in balance_rows:
        financials[isin].update({
            "assets_latest": assets1, "liab_latest": liab1, "equity_latest": equity1,
            "curr_liab_latest": curr_liab1, "cash_latest": cash1,
            "assets_prior": assets2, "liab_prior": liab2, "curr_liab_prior": curr_liab2, "cash_prior": cash2,
        })
    for isin, cfo1, capex1 in cashflow_rows:
        financials[isin].update({"cfo_latest": cfo1, "capex_latest": capex1})

    # ---- price history: last ~14 months per candidate, for momentum/52w/beta/ivol ----
    ph_rows = cur.execute(
        f"""
        SELECT isin, {ph_date_col}, {ph_close_col}
        FROM price_history
        WHERE isin IN {in_clause(isins)}
          AND {ph_date_col} >= DATE('now', '-14 months')
        """,
        isins,
    ).fetchall()
    conn.close()

    ph_by_isin = {}
    for isin, d, close in ph_rows:
        ph_by_isin.setdefault(isin, []).append((d, close))
    for isin in ph_by_isin:
        hist = pd.DataFrame(ph_by_isin[isin], columns=["date", "close"])
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date", "close"]).sort_values("date")
        ph_by_isin[isin] = hist

    # ---- assemble raw factor table (DB-only factors; debt_equity/beta/ivol filled in later) ----
    records = []
    for isin, cand in candidates.items():
        fin = {**{
            "revenue_latest": None, "gp_latest": None, "ebit_latest": None, "ni_latest": None,
            "revenue_prior": None, "gp_prior": None, "ni_prior": None,
            "assets_latest": None, "liab_latest": None, "equity_latest": None,
            "curr_liab_latest": None, "cash_latest": None,
            "assets_prior": None, "liab_prior": None, "curr_liab_prior": None, "cash_prior": None,
            "cfo_latest": None, "capex_latest": None,
        }, **financials.get(isin, {}), "market_cap": cand["market_cap"]}
        fund = compute_fundamental_factors(fin)

        hist = ph_by_isin.get(isin, pd.DataFrame(columns=["date", "close"]))
        price_factors = compute_price_factors(hist)

        record = {
            "isin": isin, "symbol": cand["symbol"], "name": cand["name"], "sector": cand["sector"],
            "yahoo_ticker": cand["yahoo_ticker"], "market_cap": cand["market_cap"],
            "ev_ebitda_ratio": cand["ev_ebitda_ratio"],
            "debt_equity": None, "beta": None, "ivol": None,
            **fund, **price_factors,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # ---- pre-filter: 6M momentum > 0 (needs the price-history pull above) ----
    before = len(df)
    df = df[df["momentum_6m"] > 0].reset_index(drop=True)
    print(f"[screener] pass 6M momentum > 0: {len(df)} (of {before} with price history joined)", file=sys.stderr)

    if df.empty:
        return []

    # ---- initial scoring (debt_equity/beta/ivol unavailable at this stage) -> top 20 ----
    scored, _ = compute_quant_scores(df, FACTOR_WEIGHTS)
    scored = scored[scored["_valid_factor_count"] >= MIN_VALID_FACTORS]
    top20 = scored.sort_values("quant_score", ascending=False).head(INITIAL_TOP_N).copy()
    print(f"[screener] candidates with >= {MIN_VALID_FACTORS} of {len(FACTOR_NAMES)} factors: {len(scored)}", file=sys.stderr)
    print(f"[screener] advancing top {len(top20)} to yfinance enrichment stage", file=sys.stderr)

    # ---- enrich top 20 with yfinance: debt/equity + beta/ivol vs OSEBX ----
    osebx_hist = fetch_osebx_history()
    for idx, row in top20.iterrows():
        top20.at[idx, "debt_equity"] = fetch_debt_equity(row["yahoo_ticker"])
        stock_hist = ph_by_isin.get(row["isin"], pd.DataFrame(columns=["date", "close"]))
        beta, ivol = compute_beta_ivol(stock_hist, osebx_hist)
        top20.at[idx, "beta"] = beta
        top20.at[idx, "ivol"] = ivol

    # ---- re-score the top 20 with all 12 factors, using percentile ranks within this subset ----
    rescored, coverage_pct = compute_quant_scores(top20, FACTOR_WEIGHTS)
    rescored = rescored[rescored["_valid_factor_count"] >= MIN_VALID_FACTORS]
    final = rescored.sort_values("quant_score", ascending=False).head(FINAL_TOP_N)

    print("[screener] factor coverage among top-20 candidates:", file=sys.stderr)
    for name, pct in sorted(coverage_pct.items(), key=lambda kv: kv[1]):
        print(f"[screener]   {name}: {pct:.0f}%", file=sys.stderr)
    worst5 = sorted(coverage_pct.items(), key=lambda kv: kv[1])[:5]
    print(f"[screener] top 5 factors with most missing data: {[n for n, _ in worst5]}", file=sys.stderr)

    results = []
    for _, row in final.iterrows():
        quant_score = round(float(row["quant_score"]), 1)
        recommendation = get_recommendation(quant_score)
        if recommendation is None:  # below 40 -> excluded
            continue
        results.append({
            "symbol": row["symbol"],
            "name": row["name"],
            "sector": row["sector"],
            "market_cap": row["market_cap"],
            "ev_ebitda": row["ev_ebitda_ratio"],
            "earnings_yield": row["earnings_yield"],
            "fcf_yield": row["fcf_yield"],
            "pb_ratio": row["pb_ratio"],
            "piotroski": row["piotroski"],
            "gross_margin_trend": row["gross_margin_trend"],
            "debt_equity": row["debt_equity"],
            "momentum_6m": row["momentum_6m"],
            "momentum_12m": row["momentum_12m"],
            "high52w_ratio": row["high52w_ratio"],
            "beta": row["beta"],
            "ivol": row["ivol"],
            "revenue_growth": row["revenue_growth"],
            "quant_score": quant_score,
            "recommendation": recommendation,
        })
    return results


if __name__ == "__main__":
    print(json.dumps(run_screener()))
