"""Step 8 - example HTTP client for the Ray Serve endpoint.

python client.py --url http://127.0.0.1:8000 --out outputs/serve
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os

import requests
from PIL import Image


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="outputs/serve")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = {}

    r = requests.get(f"{args.url}/health", timeout=30)
    print("GET /health", r.status_code, r.json())
    log["health"] = r.json()

    for label in [0, 1]:
        payload = {"label": label, "n": 8, "seed": 42, "return_grid": True}
        r = requests.post(f"{args.url}/generate", json=payload, timeout=60)
        body = r.json()
        print("POST /generate", payload, "->", r.status_code, sorted(body.keys()),
              "| X-Synthetic-Data:", r.headers.get("X-Synthetic-Data"))
        decode(body["grid"]).save(os.path.join(args.out, f"api_grid_label{label}.png"))
        img = decode(body["images"][0])
        print("   first image:", img.size, img.mode, "PNG metadata:", {k: img.info[k] for k in ("Synthetic", "Label", "Seed")})
        short = {k: v for k, v in body.items() if k not in ("images", "grid")}
        short["images"] = [body["images"][0][:60] + "...", f"... ({len(body['images'])} images)"]
        log[f"generate_label{label}"] = {"request": payload, "status": r.status_code, "response": short}

    # reproducibility: same seed -> identical bytes
    a = requests.post(f"{args.url}/generate", json={"label": "pneumonia", "n": 4, "seed": 7}, timeout=60).json()
    b = requests.post(f"{args.url}/generate", json={"label": 1, "n": 4, "seed": 7}, timeout=60).json()
    same = a["images"] == b["images"]
    print("same seed -> identical images:", same)
    log["seed_reproducible"] = same

    # validation: these must be rejected with HTTP 422
    bad = [{"label": 7, "n": 4}, {"label": "covid", "n": 4}, {"label": 0, "n": 1000},
           {"label": 0, "n": 0}, {"label": 1, "seed": -3}, {"label": 1, "size": 512},
           {"label": 0, "patient_id": "123"}]
    log["validation"] = []
    for p in bad:
        r = requests.post(f"{args.url}/generate", json=p, timeout=30)
        detail = r.json().get("detail", [{}])
        msg = detail[0].get("msg") if isinstance(detail, list) and detail else str(detail)
        print(f"invalid {json.dumps(p):40s} -> {r.status_code} {msg}")
        log["validation"].append({"request": p, "status": r.status_code, "msg": msg})

    with open(os.path.join(args.out, "client_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print("saved", os.path.join(args.out, "client_log.json"))


if __name__ == "__main__":
    main()
