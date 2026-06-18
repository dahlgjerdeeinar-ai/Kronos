# Kronos – Norwegian Stock Forecast API

## Quick start (local)

```bash
pip install -r requirements.txt
python predict_oslo_stocks.py EQNR.OL 10   # CLI forecast
python api.py                               # Flask API on :8080
```

Test the API:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "EQNR.OL", "pred_days": 10}'
```

## Deploy to Modal (serverless, free tier available)

```bash
pip install modal
modal setup          # authenticate once
modal deploy modal_app.py
```

Modal prints the public HTTPS URL after deployment, e.g.:
`https://your-username--kronos-forecast-api-flask-app.modal.run`

Test:
```bash
curl -X POST https://<slug>.modal.run/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "EQNR.OL", "pred_days": 10}'
```

## Deploy to Railway

1. Push this repo to GitHub.
2. Create a new Railway project → "Deploy from GitHub repo".
3. Select the repo; Railway auto-detects `Procfile` and `railway.json`.
4. No environment variables are required (defaults to Kronos-small).
5. Optional env vars:
   - `KRONOS_MODEL` – HuggingFace model ID (default `NeoQuasar/Kronos-small`)
   - `KRONOS_LOOKBACK` – rows of history to feed the model (default `400`)

## API reference

### `POST /predict`

Request body (JSON):

| Field      | Type   | Default    | Description                              |
|------------|--------|------------|------------------------------------------|
| ticker     | string | `EQNR.OL`  | Yahoo Finance ticker (Oslo Børs uses `.OL`) |
| pred_days  | int    | `10`       | Trading days to forecast (1–60)          |

Response (JSON):

```json
{
  "ticker": "EQNR.OL",
  "model": "NeoQuasar/Kronos-small",
  "lookback_rows": 394,
  "pred_days": 10,
  "predictions": [
    {
      "date": "2025-06-19",
      "open": 275.40,
      "high": 278.10,
      "low": 274.20,
      "close": 276.80,
      "volume": 4321000
    }
  ]
}
```

### `GET /health`

Returns `{"status": "ok"}` – used by Railway health checks.

## Supported Oslo Børs tickers (examples)

| Ticker    | Company               |
|-----------|-----------------------|
| EQNR.OL   | Equinor               |
| DNB.OL    | DNB Bank              |
| TEL.OL    | Telenor               |
| ORK.OL    | Orkla                 |
| MOWI.OL   | Mowi                  |
| YAR.OL    | Yara International    |
| NHY.OL    | Norsk Hydro           |
| SALM.OL   | SalMar                |
