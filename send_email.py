import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SENDER_NAME = "Stock Analysis"

SEPARATOR = "═" * 39
EV_EBITDA_GUIDE = "EV/EBITDA guide: <5 Very cheap | 5-10 Fair | 10-15 Expensive | >15 Very expensive"
DAY_BY_DAY_PLACEHOLDER = (
    "(Per-day price breakdown not available -- daily_forecast.py currently "
    "reports only the 5-day average forecast per ticker.)"
)


def run_script(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / name)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def build_body(screener_output, forecast_output):
    today = date.today().isoformat()
    return f"""DAILY STOCK ANALYSIS - {today}

{SEPARATOR}
TOP NORDIC CANDIDATES (Fundamental Screening)
{SEPARATOR}
{screener_output}
{EV_EBITDA_GUIDE}

{SEPARATOR}
PORTFOLIO - 5-DAY FORECAST
{SEPARATOR}
{forecast_output}

{SEPARATOR}
DAILY PRICE MOVEMENT (predicted)
{SEPARATOR}
{DAY_BY_DAY_PLACEHOLDER}
"""


def send_email(body):
    email_address = os.environ["GMAIL_ADRESS"]
    api_key = os.environ["BREVO_API"]

    payload = {
        "sender": {"name": SENDER_NAME, "email": email_address},
        "to": [{"email": email_address}],
        "subject": f"Daily Stock Analysis - {date.today().isoformat()}",
        "textContent": body,
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = requests.post(BREVO_API_URL, headers=headers, json=payload)
    response.raise_for_status()


def main():
    screener_output = run_script("screener.py")
    forecast_output = run_script("daily_forecast.py")
    body = build_body(screener_output, forecast_output)
    print(body)
    send_email(body)


if __name__ == "__main__":
    main()
