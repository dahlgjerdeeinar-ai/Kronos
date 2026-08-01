import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
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

    <div style="margin-top:24px;background:#f0ede6;border-left:3px solid #0a0f0a;padding:14px 16px;border-radius:0 4px 4px 0;">
      <div style="font-family:-apple-system,sans-serif;font-size:10px;color:#555;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Markedsanalyse</div>
      <div style="font-size:13px;color:#333;line-height:1.8;font-style:italic;">{market_analysis}</div>
    </div>

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
  <td style="padding:10px 8px;text-align:center;"><span style="background:#0a0f0a;color:{score_color};padding:3px 8px;border-radius:4px;font-size:12px;font-family:'Courier New',monospace;">{quant_score}</span></td>
  <td style="padding:10px 8px;text-align:center;color:#555;">{ev_ebitda}</td>
  <td style="padding:10px 8px;text-align:center;color:{signal_color};font-weight:bold;font-family:-apple-system,sans-serif;">&bull; {kronos_signal}</td>
  <td style="padding:10px 8px;text-align:right;color:{signal_color};font-weight:bold;font-family:-apple-system,sans-serif;">{change_pct}</td>
</tr>"""

PORTFOLIO_ROW = """
<tr style="border-top:1px solid #e8e4dc;">
  <td style="padding:10px 0;font-weight:600;">{ticker}</td>
  <td style="padding:10px 8px;color:#888;">{current} &rarr; {forecast}</td>
  <td style="padding:10px 8px;text-align:center;color:{signal_color};font-weight:bold;">&bull; {signal}</td>
  <td style="padding:10px 8px;text-align:right;color:{signal_color};font-weight:bold;">{change_pct}</td>
  <td style="padding:10px 8px;text-align:right;color:#888;font-size:11px;">EV/E {ev_ebitda} &middot; ROIC {roic}</td>
</tr>"""

MOVEMENT_ROW_STYLE = "border-top:1px solid #e8e4dc;"
MOVEMENT_CELL_STYLE = "padding:8px 6px;font-family:-apple-system,sans-serif;"

SIGNAL_COLOR_MAP = {"BUY": "#1a7a1a", "SELL": "#cc2222", "HOLD": "#b8860b"}


def run_script(name, args=None, echo_stderr=False):
    cmd = [sys.executable, str(ROOT / name)] + list(args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    if echo_stderr and result.stderr:
        print(result.stderr.rstrip("\n"))
    return result.stdout.strip()


def score_color(quant_score):
    if quant_score is None:
        return "#888"
    if quant_score >= 80:
        return "#00ff41"
    if quant_score >= 60:
        return "#888800"
    return "#888"


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
            score_color=score_color(quant_score),
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
            signal_color=signal_color(t.get("signal")),
            signal=t.get("signal", "N/A"),
            change_pct=fmt_signed_pct(t.get("change_pct")),
            ev_ebitda=fmt_num(t.get("ev_ebitda")),
            roic=fmt_pct(roic_pct),
        ))
    return "".join(html_rows)


def build_movement_rows(dates, tickers, screener_rows, screener_forecasts):
    header_cells = "".join(
        f"<td style='{MOVEMENT_CELL_STYLE}color:#888;font-size:10px;'>{d}</td>" for d in dates
    )
    rows_html = [
        f"<tr style='{MOVEMENT_ROW_STYLE}'>"
        f"<td style='{MOVEMENT_CELL_STYLE}color:#888;font-size:10px;'>Ticker</td>{header_cells}</tr>"
    ]
    for t in tickers:
        cells = "".join(f"<td style='{MOVEMENT_CELL_STYLE}text-align:right;'>{fmt_num(p, 2)}</td>" for p in t["daily_prices"])
        rows_html.append(f"<tr style='{MOVEMENT_ROW_STYLE}'><td style='{MOVEMENT_CELL_STYLE}font-weight:600;'>{t['ticker']}</td>{cells}</tr>")
    for row in screener_rows:
        daily_prices = get_kronos_forecast(screener_forecasts, row["symbol"])["daily_prices"]
        if not daily_prices:
            continue
        cells = "".join(f"<td style='{MOVEMENT_CELL_STYLE}text-align:right;'>{fmt_num(p, 2)}</td>" for p in daily_prices)
        rows_html.append(f"<tr style='{MOVEMENT_ROW_STYLE}'><td style='{MOVEMENT_CELL_STYLE}font-weight:600;'>{row['symbol']}</td>{cells}</tr>")
    return "".join(rows_html)


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


def build_html_body(screener_rows, forecast_data):
    today = date.today().isoformat()
    dates = forecast_data["dates"]
    tickers = forecast_data["tickers"]
    screener_forecasts = forecast_data.get("screener_forecasts", {})

    return EMAIL_TEMPLATE.format(
        date=today,
        screener_rows=build_screener_rows(screener_rows, screener_forecasts),
        portfolio_rows=build_portfolio_rows(tickers),
        movement_rows=build_movement_rows(dates, tickers, screener_rows, screener_forecasts),
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
        lines.append(
            f"{t['ticker']}: {fmt_num(t['current_price'], 2)} -> {fmt_num(t['avg_forecast'], 2)} "
            f"({fmt_signed_pct(t['change_pct'])}) | {t['signal']} | EV/EBITDA: {fmt_num(t['ev_ebitda'])} "
            f"({t['valuation_label']}) | ROIC: {fmt_pct(roic_pct)}"
        )
    return "\n".join(lines)


def send_email(html_body, text_body):
    email_address = os.environ["GMAIL_ADRESS"]
    api_key = os.environ["BREVO_API"]

    payload = {
        "sender": {"name": SENDER_NAME, "email": email_address},
        "to": [{"email": email_address}],
        "subject": f"Daily Stock Analysis - {date.today().isoformat()}",
        "htmlContent": html_body,
        "textContent": text_body,
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(BREVO_API_URL, headers=headers, json=payload)
    response.raise_for_status()


def main():
    screener_rows = json.loads(run_script("screener.py", echo_stderr=True))
    screener_symbols = [row["symbol"] for row in screener_rows]
    forecast_data = json.loads(run_script("daily_forecast.py", screener_symbols))

    html_body = build_html_body(screener_rows, forecast_data)
    text_body = build_text_body(screener_rows, forecast_data)

    print(text_body)
    send_email(html_body, text_body)


if __name__ == "__main__":
    main()
