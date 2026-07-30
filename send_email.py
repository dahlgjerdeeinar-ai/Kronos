import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SENDER_NAME = "Stock Analysis"

EV_EBITDA_GUIDE = "EV/EBITDA guide: &lt;5 Very cheap | 5-10 Fair | 10-15 Expensive | &gt;15 Very expensive"

SIGNAL_COLORS = {
    "sell": ("#f8d7da", "#721c24"),
    "buy": ("#d4edda", "#155724"),
    "hold": ("#fff3cd", "#856404"),
}

TABLE_STYLE = "width:100%;border-collapse:collapse;margin:8px 0 20px;font-size:13px;"
TH_STYLE = "text-align:left;padding:8px 10px;background:#f2f2f2;border-bottom:2px solid #ddd;"
TD_STYLE = "padding:8px 10px;border-bottom:1px solid #eee;"
H2_STYLE = "font-size:16px;border-bottom:2px solid #1a1a2e;padding-bottom:6px;margin-top:28px;"


def run_script(name, args=None):
    cmd = [sys.executable, str(ROOT / name)] + list(args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def signal_style(label):
    lowered = (label or "").lower()
    for keyword, (bg, fg) in SIGNAL_COLORS.items():
        if keyword in lowered:
            return f"background-color:{bg};color:{fg};font-weight:bold;"
    return ""


def fmt_pct(value, decimals=1):
    return f"{value:.{decimals}f}%" if value is not None else "N/A"


def fmt_signed_pct(value, decimals=1):
    return f"{value:+.{decimals}f}%" if value is not None else "N/A"


def fmt_num(value, decimals=1):
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def th_row(columns):
    return "<tr>" + "".join(f"<th style='{TH_STYLE}'>{col}</th>" for col in columns) + "</tr>"


def get_kronos_forecast(screener_forecasts, symbol):
    """Never raises: any lookup/shape problem falls back to signal N/A, pct None."""
    try:
        forecast = screener_forecasts.get(symbol)
        if not forecast:
            return "N/A", None
        return forecast["signal"], forecast["change_pct"]
    except Exception:
        return "N/A", None


def build_screener_table(rows, screener_forecasts):
    header = th_row([
        "Symbol", "Company", "EV/EBITDA", "ROIC",
        "Fundamental Recommendation", "Kronos Signal", "5-Day Forecast %",
    ])
    body_rows = []
    for row in rows:
        recommendation = row["recommendation"]
        roic_pct = row["roic"] * 100 if row["roic"] is not None else None
        kronos_signal, forecast_change = get_kronos_forecast(screener_forecasts, row["symbol"])
        forecast_pct = fmt_signed_pct(forecast_change)
        body_rows.append(
            "<tr>"
            f"<td style='{TD_STYLE}'>{row['symbol']}</td>"
            f"<td style='{TD_STYLE}'>{row['name']}</td>"
            f"<td style='{TD_STYLE}'>{fmt_num(row['ev_ebitda_ratio'])}</td>"
            f"<td style='{TD_STYLE}'>{fmt_pct(roic_pct)}</td>"
            f"<td style='{TD_STYLE}{signal_style(recommendation)}'>{recommendation}</td>"
            f"<td style='{TD_STYLE}{signal_style(kronos_signal)}'>{kronos_signal}</td>"
            f"<td style='{TD_STYLE}'>{forecast_pct}</td>"
            "</tr>"
        )
    return f"<table style='{TABLE_STYLE}'>{header}{''.join(body_rows)}</table>"


def build_portfolio_table(tickers):
    header = th_row(["Ticker", "Current Price", "Forecast", "Change%", "Signal", "EV/EBITDA", "Valuation", "ROIC"])
    body_rows = []
    for t in tickers:
        roic_pct = t["roic"] * 100 if t["roic"] is not None else None
        body_rows.append(
            "<tr>"
            f"<td style='{TD_STYLE}'>{t['ticker']}</td>"
            f"<td style='{TD_STYLE}'>{fmt_num(t['current_price'], 2)}</td>"
            f"<td style='{TD_STYLE}'>{fmt_num(t['avg_forecast'], 2)}</td>"
            f"<td style='{TD_STYLE}'>{fmt_signed_pct(t['change_pct'])}</td>"
            f"<td style='{TD_STYLE}{signal_style(t['signal'])}'>{t['signal']}</td>"
            f"<td style='{TD_STYLE}'>{fmt_num(t['ev_ebitda'])}</td>"
            f"<td style='{TD_STYLE}'>{t['valuation_label']}</td>"
            f"<td style='{TD_STYLE}'>{fmt_pct(roic_pct)}</td>"
            "</tr>"
        )
    return f"<table style='{TABLE_STYLE}'>{header}{''.join(body_rows)}</table>"


def build_daily_movement_table(dates, tickers):
    header = th_row(["Ticker"] + dates)
    body_rows = []
    for t in tickers:
        cells = "".join(f"<td style='{TD_STYLE}'>{fmt_num(p, 2)}</td>" for p in t["daily_prices"])
        body_rows.append(f"<tr><td style='{TD_STYLE}'>{t['ticker']}</td>{cells}</tr>")
    return f"<table style='{TABLE_STYLE}'>{header}{''.join(body_rows)}</table>"


def build_html_body(screener_rows, forecast_data):
    today = date.today().isoformat()
    dates = forecast_data["dates"]
    tickers = forecast_data["tickers"]
    screener_forecasts = forecast_data.get("screener_forecasts", {})

    return f"""<html>
<body style="margin:0;padding:0;background-color:#ffffff;font-family:Arial,Helvetica,sans-serif;color:#1a1a1a;">
  <div style="max-width:900px;margin:0 auto;">
    <div style="background-color:#1a1a2e;color:#ffffff;padding:24px 28px;">
      <h1 style="margin:0;font-size:20px;">Daily Stock Analysis</h1>
      <div style="opacity:0.75;font-size:14px;margin-top:4px;">{today}</div>
    </div>
    <div style="padding:0 28px 28px;">
      <h2 style="{H2_STYLE}">Top Nordic Candidates (Fundamental Screening)</h2>
      {build_screener_table(screener_rows, screener_forecasts)}
      <p style="font-size:12px;color:#666;margin-top:-14px;">{EV_EBITDA_GUIDE}</p>

      <h2 style="{H2_STYLE}">Portfolio - 5-Day Forecast</h2>
      {build_portfolio_table(tickers)}

      <h2 style="{H2_STYLE}">Daily Price Movement (Predicted)</h2>
      {build_daily_movement_table(dates, tickers)}
    </div>
  </div>
</body>
</html>"""


def build_text_body(screener_rows, forecast_data):
    screener_forecasts = forecast_data.get("screener_forecasts", {})
    lines = [f"DAILY STOCK ANALYSIS - {date.today().isoformat()}", "", "TOP NORDIC CANDIDATES"]
    for row in screener_rows:
        roic_pct = row["roic"] * 100 if row["roic"] is not None else None
        kronos_signal, forecast_change = get_kronos_forecast(screener_forecasts, row["symbol"])
        forecast_pct = fmt_signed_pct(forecast_change)
        lines.append(
            f"{row['symbol']} {row['name']} ev/ebitda={fmt_num(row['ev_ebitda_ratio'])} "
            f"roic={fmt_pct(roic_pct)} {row['recommendation']} | Kronos: {kronos_signal} ({forecast_pct})"
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
    screener_rows = json.loads(run_script("screener.py"))
    screener_symbols = [row["symbol"] for row in screener_rows]
    forecast_data = json.loads(run_script("daily_forecast.py", screener_symbols))

    html_body = build_html_body(screener_rows, forecast_data)
    text_body = build_text_body(screener_rows, forecast_data)

    print(text_body)
    send_email(html_body, text_body)


if __name__ == "__main__":
    main()
