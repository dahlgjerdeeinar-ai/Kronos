import json
import os
import smtplib
import subprocess
import sys
from collections import Counter
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent

SCREENER_HISTORY_PATH = ROOT / "screener_history.json"
FORECAST_HISTORY_PATH = ROOT / "forecast_history.json"

BREVO_SMTP_HOST = "smtp-relay.brevo.com"
BREVO_SMTP_PORT = 587
BREVO_SMTP_USERNAME = "b3e1b9001@smtp-brevo.com"
SENDER_NAME = "Stock Analysis"

EMAIL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#e8e4dc;">
<div style="max-width:620px;margin:20px auto;border-radius:8px;overflow:hidden;border:1px solid #222;font-family:Georgia,serif;">

  <div style="background:#0a0f0a;padding:24px 28px;border-bottom:2px solid #00ff41;">
    <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#00C8FF;letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">Nordic Quant System · Daily Brief</div>
    <div style="font-size:22px;font-weight:normal;color:#ffffff;">Morning Memo</div>
    <div style="font-family:-apple-system,sans-serif;font-size:11px;color:#555;margin-top:4px;">{date} · 07:00 CET</div>
  </div>

  <div style="background:#faf8f3;padding:24px 28px;">

    <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;border-bottom:1px solid #222;padding-bottom:6px;">Topp nordiske kandidater</div>
    <table style="width:100%;font-size:13px;border-collapse:collapse;font-family:-apple-system,sans-serif;">
      <tr style="font-size:10px;color:#888;">
        <td style="padding:4px 0;">Selskap</td>
        <td style="padding:4px 8px;text-align:center;">Score</td>
        <td style="padding:4px 8px;text-align:center;">EV/E</td>
        <td style="padding:4px 8px;text-align:center;">Kronos</td>
        <td style="padding:4px 8px;text-align:right;">5D</td>
      </tr>
      {screener_rows}
    </table>

    <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin:24px 0 14px;border-bottom:1px solid #222;padding-bottom:6px;">Portefølje · 5-dagers prognose</div>
    <table style="width:100%;font-size:13px;border-collapse:collapse;font-family:-apple-system,sans-serif;">
      {portfolio_rows}
    </table>

    <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin:24px 0 14px;border-bottom:1px solid #222;padding-bottom:6px;">Daglig prisutvikling (prognose)</div>
    <table style="width:100%;font-size:12px;border-collapse:collapse;font-family:-apple-system,sans-serif;">
      {movement_rows}
    </table>

    {repeated_section}

    <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin:24px 0 14px;border-bottom:1px solid #222;padding-bottom:6px;">Kronos nøyaktighet (siste dager)</div>
    <table style="width:100%;font-size:13px;border-collapse:collapse;font-family:-apple-system,sans-serif;">
      {accuracy_rows}
    </table>
    {accuracy_warning}

    <div style="margin-top:24px;background:#f0ede6;border-left:3px solid #0a0f0a;padding:14px 16px;border-radius:0 4px 4px 0;">
      <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Markedsanalyse</div>
      <div style="font-size:13px;color:#333;line-height:1.8;font-style:italic;">{market_analysis}</div>
    </div>

    <details style="margin-top:24px;background:#f0ede6;border-left:3px solid #0a0f0a;padding:16px 20px;">
      <summary style="cursor:pointer;font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;">Om systemet &hellip;</summary>
      <div style="font-family:-apple-system,sans-serif;font-size:12px;color:#444;line-height:1.8;margin-top:10px;">
        <strong>Screening:</strong> 1400+ nordiske aksjer screenes daglig via lseffer Nordic Stock Screener.
        Aksjer rangeres etter en 12-faktor kvant-modell: Verdi (Earnings Yield, FCF Yield, P/B),
        Kvalitet (Piotroski F-Score, Gross Margin), Momentum (6M, 12M, 52W High),
        Vekst (omsetningsvekst, marginforbedring) og Risiko (Beta, IVOL).<br><br>
        <strong>Kronos:</strong> AI-modell trent på 12 milliarder kursdata-rekorder fra 45 globale børser.
        Spår kortsiktige kursbevegelser (5 dager) basert på historiske OHLCV-mønstre.
        Kronos er et timing-verktøy — ikke et seleksjonsverktøy.<br><br>
        Kronos analyserer de siste 400 handelsdagene (~1.5 år) med kursdata per aksje,
        og kjøres 3 ganger per aksje for å redusere tilfeldige variasjoner i prognosen.<br><br>
        <strong>Signaler:</strong><br>
        &#9679; BUY: Kronos spår over +2% vekst på 5 dager<br>
        &#9679; HOLD: Kronos spår mellom -4% og +2%<br>
        &#9679; SELL: Kronos spår under -4% fall på 5 dager<br><br>
        <strong>Quant Score:</strong> 0-100. Strong Buy &ge;80 &middot; Buy &ge;60 &middot; Hold &ge;40 &middot; Under 40 vises ikke.<br><br>
        <strong>Kronos-kompatibilitet:</strong> Kun aksjer med daglig volum &ge;50 000 og daglig volatilitet
        mellom 0.3-3% anbefales for Kronos-prognoser. Makro-sensitive sektorer (Energy, Basic Materials,
        Forsvar) er merket &#9888;&#65039; da Kronos ikke tar hensyn til geopolitiske hendelser.<br><br>
        <em>Dette er ikke finansiell rådgivning. Alle investeringsbeslutninger tas på eget ansvar.</em>
      </div>
    </details>

  </div>

  <div style="background:#0a0f0a;padding:12px 28px;font-family:-apple-system,sans-serif;font-size:10px;color:#444;display:flex;justify-content:space-between;">
    <span>Einar's Nordic Quant System</span>
    <span style="color:#00ff41;">&#9679; System operativt</span>
  </div>

</div>
</body>
</html>"""

SCREENER_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:10px 0;"><strong>{name}</strong><br><span style="font-size:11px;color:#888;">{sector} &middot; {momentum_6m} 6M</span></td>
  <td style="padding:10px 8px;text-align:center;"><span style="background:#0a0f0a;color:#ffffff;padding:3px 8px;border-radius:4px;font-size:12px;font-family:'Courier New',monospace;">{quant_score}</span></td>
  <td style="padding:10px 8px;text-align:center;color:#555;">{ev_ebitda}</td>
  <td style="padding:10px 8px;text-align:center;color:{signal_color};font-weight:bold;font-family:-apple-system,sans-serif;">&bull; {kronos_signal}</td>
  <td style="padding:10px 8px;text-align:right;color:{signal_color};font-weight:bold;font-family:-apple-system,sans-serif;">{change_pct}</td>
</tr>"""

PORTFOLIO_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:10px 0;font-weight:600;">{ticker}</td>
  <td style="padding:10px 8px;color:#888;">{current} &rarr; {forecast}
    <br><span style="font-size:10px;color:#aaa;">Yesterday close (Kronos baseline): {current} | Today open: {today_open} | Gap: {gap_pct}</span>
  </td>
  <td style="padding:10px 8px;text-align:center;color:{signal_color};font-weight:bold;">&bull; {signal}</td>
  <td style="padding:10px 8px;text-align:right;color:{signal_color};font-weight:bold;">{change_pct}</td>
  <td style="padding:10px 8px;text-align:right;color:#888;font-size:11px;">EV/E {ev_ebitda} &middot; ROIC {roic}</td>
</tr>"""

MOVEMENT_HEADER_DATE_CELL = """<td style="padding:4px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:9px;color:#888;white-space:nowrap;">{date}</td>"""
MOVEMENT_HEADER_FORECAST_DATE_CELL = """<td style="padding:4px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:9px;font-style:italic;color:#aaa;white-space:nowrap;">{date}</td>"""

MOVEMENT_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:8px 6px;vertical-align:top;">
    <span style="font-family:Georgia,serif;font-size:13px;font-weight:bold;color:#0a0f0a;">{symbol}</span><br>
    <span style="font-family:-apple-system,sans-serif;font-size:11px;color:#999;">{reason}</span>
  </td>
  {date_cells}
  <td style="padding:8px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:10px;font-style:italic;color:#555;white-space:nowrap;">{mae}</td>
</tr>"""

MOVEMENT_HISTORICAL_CELL = """<td style="padding:6px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:10px;white-space:nowrap;">{predicted}<br><span style="font-size:9px;color:{diff_color};">{actual}</span></td>"""

MOVEMENT_FORECAST_CELL = """<td style="padding:6px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:10px;font-style:italic;color:#999;white-space:nowrap;">{predicted}</td>"""

MOVEMENT_EMPTY_CELL = """<td style="padding:6px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:10px;color:#ccc;">&mdash;</td>"""

SIGNAL_COLOR_MAP = {"BUY": "#1a7a1a", "SELL": "#cc2222", "HOLD": "#b8860b"}

REPEATED_SECTION_WRAPPER = """
<div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin:24px 0 14px;border-bottom:1px solid #222;padding-bottom:6px;">Gjentatte screende aksjer</div>
<table style="width:100%;font-size:12px;border-collapse:collapse;">
  <tr style="font-size:10px;color:#888;font-family:-apple-system,sans-serif;">
    <td style="padding:4px 6px;">Dato</td>
    <td style="padding:4px 6px;text-align:right;">Kronos-pris</td>
    <td style="padding:4px 6px;text-align:right;">Kronos %</td>
    <td style="padding:4px 6px;text-align:right;">Faktisk</td>
    <td style="padding:4px 6px;text-align:right;">Avvik (pris)</td>
    <td style="padding:4px 6px;text-align:right;">Avvik (%)</td>
  </tr>
  {rows}
</table>"""

REPEATED_SYMBOL_HEADER = """
<tr>
  <td colspan="6" style="padding:14px 6px 4px;font-family:Georgia,serif;font-size:13px;font-weight:bold;color:#0a0f0a;">{name} <span style="font-family:-apple-system,sans-serif;font-size:11px;font-weight:normal;color:#888;">({symbol})</span> <span style="font-family:-apple-system,sans-serif;font-size:10px;font-weight:normal;color:#888;">&middot; quant score (siste): {quant_score}</span></td>
</tr>"""

REPEATED_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:8px 6px;font-family:-apple-system,sans-serif;">{date}</td>
  <td style="padding:8px 6px;text-align:right;font-family:-apple-system,sans-serif;">{predicted_price}</td>
  <td style="padding:8px 6px;text-align:right;font-family:-apple-system,sans-serif;">{forecast_pct}</td>
  <td style="padding:8px 6px;text-align:right;font-family:-apple-system,sans-serif;">{actual_close}</td>
  <td style="padding:8px 6px;text-align:right;color:{diff_color};font-weight:bold;font-family:-apple-system,sans-serif;">{diff_price}</td>
  <td style="padding:8px 6px;text-align:right;color:{diff_color};font-weight:bold;font-family:-apple-system,sans-serif;">{diff_pct}</td>
</tr>"""

ACCURACY_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:8px 0;font-weight:600;">{ticker}</td>
  <td style="padding:8px 8px;text-align:right;">{mape}</td>
</tr>"""

ACCURACY_WARNING = """
<div style="margin-top:10px;background:#fff8e1;border-left:3px solid #b8860b;padding:10px 14px;border-radius:0 4px 4px 0;font-family:-apple-system,sans-serif;font-size:12px;color:#7a5c00;">
  Kronos spår fra historiske sluttkurser. Store avvik kan skyldes gap ved børsåpning eller nyheter over natten.
</div>"""


def run_script(name, args=None, echo_stderr=False):
    """Note: does NOT use subprocess.run(check=True) -- that raises before
    stderr can ever be inspected/printed, which is why past failures in
    daily_forecast.py/screener.py never showed their real traceback in CI
    logs (just an opaque CalledProcessError). Print stderr ourselves and
    raise a clear error instead."""
    cmd = [sys.executable, str(ROOT / name)] + list(args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.stderr and (echo_stderr or result.returncode != 0):
        print(result.stderr.rstrip("\n"))
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with code {result.returncode}")
    return result.stdout.strip()


def signal_color(label):
    return SIGNAL_COLOR_MAP.get(label, "#888")  # covers "N/A" and anything unexpected


def fmt_pct(value, decimals=1):
    return f"{value:.{decimals}f}%" if value is not None else "N/A"


def fmt_signed_pct(value, decimals=1):
    return f"{value:+.{decimals}f}%" if value is not None else "N/A"


def fmt_num(value, decimals=1):
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def get_kronos_forecast(screener_forecasts, symbol):
    """Never raises: any lookup/shape problem falls back to all fields N/A/None."""
    try:
        forecast = screener_forecasts.get(symbol) or {}
    except Exception:
        forecast = {}
    return {
        "signal": forecast.get("signal", "N/A"),
        "change_pct": forecast.get("change_pct"),
        "current_price": forecast.get("current_price"),
        "avg_forecast": forecast.get("avg_forecast"),
        "daily_prices": forecast.get("daily_prices"),
    }


def build_screener_rows(rows, screener_forecasts):
    html_rows = []
    for row in rows:
        kf = get_kronos_forecast(screener_forecasts, row["symbol"])
        quant_score = row.get("quant_score")
        html_rows.append(SCREENER_ROW.format(
            name=row["name"],
            sector=row.get("sector") or "N/A",
            momentum_6m=fmt_signed_pct(row.get("momentum_6m")),
            quant_score=fmt_num(quant_score, 0),
            ev_ebitda=fmt_num(row.get("ev_ebitda")),
            signal_color=signal_color(kf["signal"]),
            kronos_signal=kf["signal"],
            change_pct=fmt_signed_pct(kf["change_pct"]),
        ))
    return "".join(html_rows)


def build_portfolio_rows(tickers):
    html_rows = []
    for t in tickers:
        roic_pct = t["roic"] * 100 if t.get("roic") is not None else None
        html_rows.append(PORTFOLIO_ROW.format(
            ticker=t["ticker"],
            current=fmt_num(t.get("current_price"), 2),
            forecast=fmt_num(t.get("avg_forecast"), 2),
            today_open=fmt_num(t.get("today_open"), 2),
            gap_pct=fmt_signed_pct(t.get("gap_pct")),
            signal_color=signal_color(t.get("signal")),
            signal=t.get("signal", "N/A"),
            change_pct=fmt_signed_pct(t.get("change_pct")),
            ev_ebitda=fmt_num(t.get("ev_ebitda")),
            roic=fmt_pct(roic_pct),
        ))
    return "".join(html_rows)


def build_accuracy_section(tickers):
    rows_html = []
    any_warning = False
    for t in tickers:
        mae = t.get("mean_absolute_error_pct")
        rows_html.append(ACCURACY_ROW.format(
            ticker=t["ticker"],
            mape=fmt_pct(mae) if mae is not None else "N/A (ingen historikk enda)",
        ))
        if t.get("mape_warning"):
            any_warning = True
    return "".join(rows_html), (ACCURACY_WARNING if any_warning else "")


# --------------------------------------------------------------------------
# Repeated screener candidates: rolling 3-day history tracker
# --------------------------------------------------------------------------

def load_screener_history():
    """screener_history.json is written by screener.py itself (it owns the
    symbol list + quant_score), not here -- by the time this subprocess
    call returns in main(), the file is already on disk."""
    if not SCREENER_HISTORY_PATH.exists():
        return []
    try:
        return json.loads(SCREENER_HISTORY_PATH.read_text())
    except Exception:
        return []


def load_forecast_history():
    """forecast_history.json is written by daily_forecast.py, keyed by
    portfolio ticker AND by bare screener symbol (see record_forecast_snapshot
    there) -- read here to look up what Kronos said about a repeated
    screener symbol on each historical date."""
    if not FORECAST_HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(FORECAST_HISTORY_PATH.read_text())
    except Exception:
        return {}


def find_repeated_symbols(history):
    counts = Counter()
    for entry in history:
        counts.update({c["symbol"] for c in entry.get("candidates", [])})
    return {symbol for symbol, n in counts.items() if n >= 2}


def resolve_yahoo_ticker(symbol, screener_forecasts):
    forecast = screener_forecasts.get(symbol) or {}
    ticker = forecast.get("ticker")
    if ticker:
        return ticker
    try:
        import daily_forecast  # heavy (loads Kronos model code) -- only import when the fallback is actually needed
        return daily_forecast.resolve_ticker(symbol)
    except Exception:
        return None


def fetch_recent_closes(ticker, days=3):
    try:
        hist = yf.download(ticker, period="10d", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return {}
    if hist.empty:
        return {}
    if hasattr(hist.columns, "get_level_values") and hist.columns.nlevels > 1:
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.tail(days)
    return {ts.strftime("%Y-%m-%d"): float(c) for ts, c in zip(hist.index, hist["Close"]) if pd.notna(c)}


def diff_color(diff_pct):
    if diff_pct is None:
        return "#888"
    if diff_pct <= 2:
        return "#1a7a1a"
    if diff_pct <= 5:
        return "#b8860b"
    return "#cc2222"


def find_predicted_price(forecast_history, symbol, target_date):
    """Among ALL past snapshots for `symbol` (not just the one made on
    target_date itself), finds the most recent one whose 5-day forecast
    window actually included target_date, and returns the day-specific
    predicted price for it -- the same "freshest snapshot wins" tie-break
    daily_forecast.py's compute_daily_errors uses when more than one past
    snapshot covered the same date.

    This replaces an earlier version of this section that compared the
    day's own baseline (current_price, i.e. roughly yesterday's close) to
    the actual close -- which is nearly meaningless since those two prices
    are almost always the same. What actually needs comparing is what
    Kronos had predicted for target_date versus what really happened."""
    snapshots_newest_first = sorted(forecast_history.get(symbol, []), key=lambda s: s.get("snapshot_date", ""), reverse=True)
    for snap in snapshots_newest_first:
        snap_dates = snap.get("dates", [])
        if target_date not in snap_dates:
            continue
        idx = snap_dates.index(target_date)
        daily_prices = snap.get("daily_prices", [])
        if idx >= len(daily_prices):
            continue
        predicted_price = daily_prices[idx]
        baseline = snap.get("current_price")
        change_pct = ((predicted_price - baseline) / baseline * 100) if baseline else None
        return {"predicted_price": predicted_price, "change_pct": change_pct, "signal": snap.get("signal")}
    return None


def build_repeated_section(screener_history, forecast_history, screener_forecasts):
    """Screener-only by construction: repeated symbols come from
    screener_history.json (screener.py's own candidate list). The Kronos
    forecast price shown for each historical date comes from
    forecast_history.json under that same screener symbol (never a
    portfolio ticker), via find_predicted_price above."""
    repeated = find_repeated_symbols(screener_history)
    if not repeated:
        return ""

    latest_quant_score = {}
    latest_name = {}
    for entry in screener_history:
        for c in entry.get("candidates", []):
            latest_quant_score[c["symbol"]] = c.get("quant_score")
            if c.get("name"):
                latest_name[c["symbol"]] = c["name"]

    rows_html = []
    for symbol in sorted(repeated):
        yahoo_ticker = resolve_yahoo_ticker(symbol, screener_forecasts)
        actual_closes = fetch_recent_closes(yahoo_ticker) if yahoo_ticker else {}
        rows_html.append(REPEATED_SYMBOL_HEADER.format(
            symbol=symbol,
            name=latest_name.get(symbol, symbol),
            quant_score=fmt_num(latest_quant_score.get(symbol), 0),
        ))
        for entry in screener_history:
            match = next((c for c in entry.get("candidates", []) if c["symbol"] == symbol), None)
            if match is None:
                continue
            entry_date = entry["date"]
            prediction = find_predicted_price(forecast_history, symbol, entry_date)
            predicted_price = prediction["predicted_price"] if prediction else None
            forecast_pct = prediction["change_pct"] if prediction else None

            actual_close = actual_closes.get(entry_date)
            diff_price = None
            diff_pct = None
            if actual_close is not None and predicted_price is not None and actual_close:
                diff_price = predicted_price - actual_close
                diff_pct = abs(diff_price) / actual_close * 100.0

            rows_html.append(REPEATED_ROW.format(
                date=entry_date,
                predicted_price=fmt_num(predicted_price, 2),
                forecast_pct=fmt_signed_pct(forecast_pct),
                actual_close=fmt_num(actual_close, 2),
                diff_color=diff_color(diff_pct),
                diff_price=fmt_num(diff_price, 2) if diff_price is not None else "N/A",
                diff_pct=fmt_pct(diff_pct),
            ))

    return REPEATED_SECTION_WRAPPER.format(rows="".join(rows_html))


def build_movement_header(historical_dates, future_dates):
    cells = ["<td style='padding:4px 6px;font-family:-apple-system,sans-serif;font-size:10px;color:#888;'>Ticker</td>"]
    cells += [MOVEMENT_HEADER_DATE_CELL.format(date=d) for d in historical_dates]
    cells += [MOVEMENT_HEADER_FORECAST_DATE_CELL.format(date=d) for d in future_dates]
    cells.append("<td style='padding:4px 4px;text-align:right;font-family:-apple-system,sans-serif;font-size:9px;color:#888;'>MAE</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def build_movement_row(symbol, result, reason, historical_dates, future_dates):
    """One row per symbol: historical columns show Kronos's predicted price
    for that date alongside the actual close (colored by how far off it
    was); future columns show only the pending forecast price, styled
    lighter/italic to mark it as not-yet-happened; a trailing MAE column
    summarizes the last 5 real trading days for that symbol."""
    errors_by_date = {e["date"]: e for e in result.get("daily_errors", [])}
    predicted_by_date = {p["date"]: p for p in result.get("predicted_prices", [])}

    date_cells = []
    for d in historical_dates:
        e = errors_by_date.get(d)
        if e is None:
            date_cells.append(MOVEMENT_EMPTY_CELL)
            continue
        date_cells.append(MOVEMENT_HISTORICAL_CELL.format(
            predicted=fmt_num(e.get("predicted"), 2),
            actual=fmt_num(e.get("actual"), 2),
            diff_color=diff_color(e.get("error_pct")),
        ))
    for d in future_dates:
        p = predicted_by_date.get(d)
        if p is None:
            date_cells.append(MOVEMENT_EMPTY_CELL)
            continue
        date_cells.append(MOVEMENT_FORECAST_CELL.format(predicted=fmt_num(p.get("price"), 2)))

    mae = result.get("mean_absolute_error_pct")
    return MOVEMENT_ROW.format(
        symbol=symbol,
        reason=reason,
        date_cells="".join(date_cells),
        mae=fmt_pct(mae) if mae is not None else "N/A",
    )


def build_movement_rows(tickers, screener_rows, screener_forecasts):
    screener_results = [
        (row, screener_forecasts.get(row["symbol"]))
        for row in screener_rows
        if screener_forecasts.get(row["symbol"])
    ]
    all_results = list(tickers) + [forecast for _, forecast in screener_results]

    # Historical columns: the union of every symbol's matured dates (Nordic
    # exchanges usually share the same trading calendar, but this stays
    # correct even if one doesn't -- a symbol missing a date just shows "-"
    # for it). Future columns: the 5-day forecast horizon, identical for
    # every symbol since it's computed once per run and shared by all of
    # them, so the first non-empty one is representative.
    historical_dates = sorted({e["date"] for r in all_results for e in r.get("daily_errors", [])})
    future_dates = next((
        [p["date"] for p in r["predicted_prices"]]
        for r in all_results if r.get("predicted_prices")
    ), [])

    blocks = [build_movement_header(historical_dates, future_dates)]

    for t in tickers:
        roic_pct = t["roic"] * 100 if t.get("roic") is not None else None
        reason = (
            f"EV/EBITDA: {fmt_num(t.get('ev_ebitda'))} &middot; "
            f"ROIC: {fmt_pct(roic_pct)} &middot; Signal: {t.get('signal', 'N/A')}"
        )
        blocks.append(build_movement_row(t["ticker"], t, reason, historical_dates, future_dates))

    for row, forecast in screener_results:
        reason = (
            f"Quant: {fmt_num(row.get('quant_score'), 0)} &middot; "
            f"EV/E: {fmt_num(row.get('ev_ebitda'))} &middot; "
            f"6M mom: {fmt_signed_pct(row.get('momentum_6m'))}"
        )
        blocks.append(build_movement_row(row["symbol"], forecast, reason, historical_dates, future_dates))

    return "".join(blocks)


def build_market_analysis(screener_rows, screener_forecasts):
    if not screener_rows:
        return "Ingen kandidater passerte screeningen i dag."

    scores = [r["quant_score"] for r in screener_rows if r.get("quant_score") is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    momentums = [r["momentum_6m"] for r in screener_rows if r.get("momentum_6m") is not None]
    avg_momentum = sum(momentums) / len(momentums) if momentums else None

    buy_count = sum(
        1 for r in screener_rows
        if get_kronos_forecast(screener_forecasts, r["symbol"])["signal"] == "BUY"
    )

    sectors = [r["sector"] for r in screener_rows if r.get("sector")]
    dominant_sector = Counter(sectors).most_common(1)[0][0] if sectors else "ukjent sektor"

    if avg_score is None:
        tone = "et usikkert"
    elif avg_score >= 65:
        tone = "et positivt"
    elif avg_score >= 50:
        tone = "et blandet"
    else:
        tone = "et forsiktig"

    score_text = f"{avg_score:.0f}" if avg_score is not None else "N/A"
    momentum_text = f"{avg_momentum:+.1f}%" if avg_momentum is not None else "N/A"
    buy_suffix = "er" if buy_count != 1 else ""

    return (
        f"Gjennomsnittlig quant-score blant {len(screener_rows)} kandidater er {score_text}, "
        f"med {buy_count} Kronos-KJØP-signal{buy_suffix} konsentrert i {dominant_sector}. "
        f"Gjennomsnittlig 6-måneders momentum ligger på {momentum_text}, {tone} bakteppe for dagen."
    )


def build_html_body(screener_rows, forecast_data, screener_history, forecast_history):
    today = date.today().isoformat()
    tickers = forecast_data["tickers"]
    screener_forecasts = forecast_data.get("screener_forecasts", {})
    accuracy_rows, accuracy_warning = build_accuracy_section(tickers)

    return EMAIL_TEMPLATE.format(
        date=today,
        screener_rows=build_screener_rows(screener_rows, screener_forecasts),
        repeated_section=build_repeated_section(screener_history, forecast_history, screener_forecasts),
        portfolio_rows=build_portfolio_rows(tickers),
        movement_rows=build_movement_rows(tickers, screener_rows, screener_forecasts),
        accuracy_rows=accuracy_rows,
        accuracy_warning=accuracy_warning,
        market_analysis=build_market_analysis(screener_rows, screener_forecasts),
    )


def build_text_body(screener_rows, forecast_data):
    screener_forecasts = forecast_data.get("screener_forecasts", {})
    lines = [f"DAILY STOCK ANALYSIS - {date.today().isoformat()}", "", "TOP NORDIC CANDIDATES"]
    for row in screener_rows:
        kf = get_kronos_forecast(screener_forecasts, row["symbol"])
        lines.append(
            f"{row['symbol']} {row['name']} ({row['sector']}) ev/ebitda={fmt_num(row['ev_ebitda'])} "
            f"momentum_6m={fmt_signed_pct(row.get('momentum_6m'))} momentum_12m={fmt_signed_pct(row.get('momentum_12m'))} "
            f"piotroski={fmt_num(row.get('piotroski'))} quant_score={fmt_num(row.get('quant_score'))} "
            f"({row['recommendation']}) | Kronos: "
            f"{fmt_num(kf['current_price'], 2)} -> {fmt_num(kf['avg_forecast'], 2)} "
            f"({fmt_signed_pct(kf['change_pct'])}) {kf['signal']}"
        )
    lines += ["", "PORTFOLIO - 5-DAY FORECAST"]
    for t in forecast_data["tickers"]:
        roic_pct = t["roic"] * 100 if t["roic"] is not None else None
        mae = t.get("mean_absolute_error_pct")
        lines.append(
            f"{t['ticker']}: {fmt_num(t['current_price'], 2)} -> {fmt_num(t['avg_forecast'], 2)} "
            f"({fmt_signed_pct(t['change_pct'])}) | {t['signal']} | EV/EBITDA: {fmt_num(t['ev_ebitda'])} "
            f"({t['valuation_label']}) | ROIC: {fmt_pct(roic_pct)}"
        )
        lines.append(
            f"    Yesterday close (Kronos baseline): {fmt_num(t.get('current_price'), 2)} | "
            f"Today open: {fmt_num(t.get('today_open'), 2)} | Gap: {fmt_signed_pct(t.get('gap_pct'))} | "
            f"MAE (last 5 days): {fmt_pct(mae) if mae is not None else 'N/A'}"
        )
    return "\n".join(lines)


def send_email(html_body, text_body):
    email_address = os.environ["GMAIL_ADRESS"]
    smtp_password = os.environ["BREVO_SMTP"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Stock Analysis - {date.today().isoformat()}"
    msg["From"] = formataddr((SENDER_NAME, email_address))
    msg["To"] = email_address
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(BREVO_SMTP_HOST, BREVO_SMTP_PORT) as server:
        server.starttls()
        server.login(BREVO_SMTP_USERNAME, smtp_password)
        server.send_message(msg)


def main():
    screener_rows = json.loads(run_script("screener.py", echo_stderr=True))
    screener_symbols = [row["symbol"] for row in screener_rows]
    forecast_data = json.loads(run_script("daily_forecast.py", screener_symbols))

    screener_history = load_screener_history()
    forecast_history = load_forecast_history()

    html_body = build_html_body(screener_rows, forecast_data, screener_history, forecast_history)
    text_body = build_text_body(screener_rows, forecast_data)

    print(text_body)
    send_email(html_body, text_body)


if __name__ == "__main__":
    main()
