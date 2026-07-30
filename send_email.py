import os
import smtplib
import ssl
import subprocess
import sys
from datetime import date
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent

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
    gmail_user = os.environ["GMAIL_USER"]
    gmail_password = os.environ["GMAIL_PASSWORD"]

    msg = MIMEText(body)
    msg["Subject"] = f"Daily Stock Analysis - {date.today().isoformat()}"
    msg["From"] = gmail_user
    msg["To"] = gmail_user

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(gmail_user, gmail_password)
        server.send_message(msg)


def main():
    screener_output = run_script("screener.py")
    forecast_output = run_script("daily_forecast.py")
    body = build_body(screener_output, forecast_output)
    print(body)
    send_email(body)


if __name__ == "__main__":
    main()
