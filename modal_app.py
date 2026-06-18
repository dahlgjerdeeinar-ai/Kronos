"""
Modal deployment for the Kronos stock-forecast API.

Deploy:
    modal deploy modal_app.py

Test:
    modal run modal_app.py

The public HTTPS URL is printed after deployment.
Endpoint: POST https://<your-slug>.modal.run/predict
"""

import os
import modal

# ── image ────────────────────────────────────────────────────────────────────
# Build a container image with all Python dependencies pre-installed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask",
        "gunicorn",
        "yfinance",
        "pandas==2.2.2",
        "numpy",
        "torch>=2.0.0",
        "einops==0.8.1",
        "huggingface_hub==0.33.1",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
        "matplotlib==3.9.3",
    )
    # Copy the entire local repo into the image
    .add_local_dir(".", remote_path="/app")
)

app = modal.App("kronos-forecast-api", image=image)

# ── persistent model cache ───────────────────────────────────────────────────
# Avoids re-downloading the ~200 MB model on every cold start.
model_cache = modal.Volume.from_name("kronos-model-cache", create_if_missing=True)


@app.function(
    volumes={"/root/.cache/huggingface": model_cache},
    # cpu=2 keeps costs low; swap for gpu="T4" to accelerate inference
    cpu=2,
    memory=4096,
    timeout=300,
    # Keep one warm instance to reduce cold-start latency
    min_concurrency=0,
    max_concurrency=5,
)
@modal.wsgi_app()
def flask_app():
    """Return the Flask WSGI app (Modal manages the ASGI/WSGI bridge)."""
    import sys
    sys.path.insert(0, "/app/Kronos-master")

    # Set working directory so relative imports inside the model package work
    os.chdir("/app")

    from api import app as flask_application

    # Pre-load the model inside the warm container
    from api import get_predictor
    get_predictor()

    return flask_application


# ── local test entrypoint ────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import json, urllib.request, urllib.error

    url = flask_app.get_url() + "/predict"
    payload = json.dumps({"ticker": "EQNR.OL", "pred_days": 5}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
