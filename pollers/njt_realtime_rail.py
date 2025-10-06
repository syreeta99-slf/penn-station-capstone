#!/usr/bin/env python3
"""
NJ TRANSIT Rail – GTFS-RT poller (TripUpdates + VehiclePositions)

Env:
  NJT_ENV            : "test" or "prod"  (default: "prod")
  NJT_USERNAME       : NJT API username (Actions secret recommended)
  NJT_PASSWORD       : NJT API password (Actions secret recommended)
  NJT_OUT_DIR        : output folder for CSVs (default: data/realtime_njt)
  NJT_INCLUDE_VP     : "1" to fetch VehiclePositions too, else "0" (default: "1")
  NJT_FILE_PREFIX    : filename prefix (default: "njt_rail_rt")
  TZ                 : for logs (default: "UTC")

Notes:
- Token endpoint (/api/GTFSRT/getToken) is limited (~10/day). We cache a token to
  ~/.njt/token.json and reuse it for ~23h. Also auto-refresh once on 401.
- GTFS-RT endpoints (getTripUpdates, getVehiclePositions) have no daily limit.
"""

from __future__ import annotations

import os
import json
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings

# ------------------------- Config -------------------------

BASES = {
    "test": "https://testraildata.njtransit.com",
    "prod": "https://raildata.njtransit.com",
}

ENV = os.getenv("NJT_ENV", "prod").strip().lower()
BASE = BASES.get(ENV, BASES["prod"])

USERNAME = os.getenv("NJT_USERNAME", "").strip()
PASSWORD = os.getenv("NJT_PASSWORD", "").strip()

OUT_DIR = Path(os.getenv("NJT_OUT_DIR", "data/realtime_njt"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

INCLUDE_VP = os.getenv("NJT_INCLUDE_VP", "1") == "1"
FILE_PREFIX = os.getenv("NJT_FILE_PREFIX", "njt_rail_rt")

# New: per-runner, per-user token cache under home
TOKEN_PATH = Path(os.path.expanduser("~/.njt/token.json"))
TZNAME = os.getenv("TZ", "UTC")


# --------------------- Small log helpers ------------------

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


# ----------------------- Token cache ----------------------

def save_token(token: str, expiry_ts: float | None = None) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"token": token, "fetched_at": time.time(), "expiry_ts": expiry_ts}
    TOKEN_PATH.write_text(json.dumps(payload))


def load_token() -> dict | None:
    if TOKEN_PATH.exists():
        try:
            return json.loads(TOKEN_PATH.read_text())
        except Exception:
            return None
    return None


def token_is_valid(obj: dict | None) -> bool:
    if not obj or "token" not in obj:
        return False
    expiry_ts = obj.get("expiry_ts")
    if expiry_ts is None:
        # assume ~23h validity if API doesn’t supply expiry
        fetched = float(obj.get("fetched_at", 0))
        return (time.time() - fetched) < 23 * 3600
    return time.time() < float(expiry_ts)


# ----------------------- Token flow -----------------------

def _fetch_token_raw(style: str) -> requests.Response:
    """
    style: "multipart" (files=...) or "form" (data=...)
    """
    url = f"{BASE}/api/GTFSRT/getToken"
    timeout = 30
    if style == "multipart":
        return requests.post(
            url,
            files={"username": (None, USERNAME), "password": (None, PASSWORD)},
            timeout=timeout,
        )
    else:  # style == "form"
        return requests.post(
            url,
            data={"username": USERNAME, "password": PASSWORD},
            timeout=timeout,
        )


def fetch_token() -> str:
    """
    Robust token fetch: retries + payload-style fallback.
    Raises RuntimeError("DAILY_LIMIT") when the API says you’re over the 10/day cap.
    """
    if not USERNAME or not PASSWORD:
        raise RuntimeError("NJT_USERNAME/NJT_PASSWORD not set")

    attempts = [("multipart", 1.0), ("multipart", 2.0), ("form", 2.0), ("form", 4.0)]
    last = ""
    for style, backoff in attempts:
        try:
            r = _fetch_token_raw(style)
            if r.status_code == 200:
                try:
                    j = r.json()
                except Exception:
                    j = None
                tok = j.get("UserToken") if isinstance(j, dict) else None
                if tok:
                    return tok
                last = f"200 but no token: {str(j)[:200]}"
            elif r.status_code == 500 and "Daily usage limit" in r.text:
                # Explicit over-quota message
                raise RuntimeError("DAILY_LIMIT")
            else:
                last = f"{r.status_code} {r.text[:200]!r}"
        except RuntimeError:
            # bubble daily limit immediately
            raise
        except Exception as e:
            last = f"EXC {type(e).__name__}: {e}"
        time.sleep(backoff)
    raise RuntimeError(f"getToken failed after retries; last={last}")


def get_token() -> str | None:
    obj = load_token()
    if token_is_valid(obj):
        return obj["token"]
    try:
        tok = fetch_token()
        # If API returns an explicit expiry, capture it; otherwise rely on fetched_at window
        save_token(tok, expiry_ts=None)
        return tok
    except RuntimeError as e:
        if "DAILY_LIMIT" in str(e):
            print(f"[{_now_utc_str()}] [njt] Daily token limit reached; no new token until next UTC day.")
            return None
        raise


# ------------------- GTFS-RT retrieval --------------------

def post_proto_with_retry(endpoint: str, token: str, max_attempts: int = 5) -> tuple[bytes, str] | tuple[None, str]:
    """
    POST to an NJT GTFS-RT endpoint (token as form-data) with retries/backoff.

    - 200: return (content, token)
    - 401: one refresh attempt (fetch_token/ save & retry once)
    - 429/5xx: exponential backoff and retry
    - Other: raise RuntimeError

    Returns (None, token) only if refresh hit DAILY_LIMIT (so caller can skip run gracefully).
    """
    url = f"{BASE}{endpoint}"
    attempt = 0
    refreshed = False
    while attempt < max_attempts:
        attempt += 1
        try:
            r = requests.post(url, files={"token": (None, token)}, timeout=60)
            sc = r.status_code
            if sc == 200:
                return r.content, token

            if sc == 401 and not refreshed:
                print(f"[{_now_utc_str()}] [njt] {endpoint} got 401; attempting one token refresh…")
                try:
                    new_tok = fetch_token()
                    save_token(new_tok, expiry_ts=None)
                    token = new_tok
                except RuntimeError as e:
                    if "DAILY_LIMIT" in str(e):
                        print(f"[{_now_utc_str()}] [njt] token refresh blocked by daily limit; skipping this run.")
                        return None, token
                    raise
                refreshed = True
                # retry immediately after refresh
                continue

            if sc in (429, 500, 502, 503, 504):
                backoff = min(2 ** attempt, 20)
                print(f"[{_now_utc_str()}] [njt] {endpoint} {sc}; retry in {backoff}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                continue

            raise RuntimeError(f"{endpoint} failed: {sc} {r.text[:200]!r}")

        except requests.RequestException as e:
            backoff = min(2 ** attempt, 20)
            print(f"[{_now_utc_str()}] [njt] {endpoint} network {type(e).__name__}: {e}; sleep {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"{endpoint} failed after {max_attempts} attempts")


# ------------------- Protobuf parsers ---------------------

def parse_trip_updates(proto_bytes: bytes) -> pd.DataFrame:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(proto_bytes)

    rows = []
    now_utc = datetime.now(timezone.utc)
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip_id = tu.trip.trip_id or None
        route_id = tu.trip.route_id or None
        for stu in tu.stop_time_update:
            stop_id = stu.stop_id or None
            arr_epoch = stu.arrival.time if stu.HasField("arrival") else None
            dep_epoch = stu.departure.time if stu.HasField("departure") else None
            to_ts = lambda e: datetime.fromtimestamp(e, tz=timezone.utc) if e else None
            rows.append({
                "agency": "NJT",
                "entity_id": ent.id or None,
                "trip_id": trip_id,
                "route_id": route_id,
                "stop_id": stop_id,
                "RT_Arrival_UTC": to_ts(arr_epoch),
                "RT_Departure_UTC": to_ts(dep_epoch),
                "schedule_relationship": tu.trip.schedule_relationship if tu.trip else None,
                "observed_utc": now_utc,
            })
    return pd.DataFrame(rows)


def parse_vehicle_positions(proto_bytes: bytes) -> pd.DataFrame:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(proto_bytes)

    rows = []
    now_utc = datetime.now(timezone.utc)
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        pos = v.position if v.HasField("position") else None
        rows.append({
            "agency": "NJT",
            "entity_id": ent.id or None,
            "trip_id": v.trip.trip_id if v.HasField("trip") else None,
            "route_id": v.trip.route_id if v.HasField("trip") else None,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
            "lat": getattr(pos, "latitude", None) if pos else None,
            "lon": getattr(pos, "longitude", None) if pos else None,
            "bearing": getattr(pos, "bearing", None) if pos else None,
            "speed": getattr(pos, "speed", None) if pos else None,
            "current_stop_sequence": getattr(v, "current_stop_sequence", None),
            "current_status": getattr(v, "current_status", None),
            "timestamp_utc": datetime.fromtimestamp(getattr(v, "timestamp", 0), tz=timezone.utc) if getattr(v, "timestamp", 0) else None,
            "observed_utc": now_utc,
        })
    return pd.DataFrame(rows)


# ---------------------------- Main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-token", action="store_true", help="Fetch token, save to cache, then exit.")
    args = ap.parse_args()

    if not USERNAME or not PASSWORD:
        print("[njt] ERROR: set NJT_USERNAME/NJT_PASSWORD", file=sys.stderr)
        sys.exit(2)

    print(f"[{_now_utc_str()}] [njt] ENV={ENV}  BASE={BASE}")
    print(f"[{_now_utc_str()}] [njt] token path={TOKEN_PATH}")

    if args.warm_token:
        tok = get_token()
        if tok:
            print(f"[{_now_utc_str()}] [njt] Warmed token OK")
            sys.exit(0)
        else:
            print(f"[{_now_utc_str()}] [njt] Warm failed (daily limit?)")
            sys.exit(1)

    token = get_token()
    if not token:
        # daily limit hit; do not fail the workflow, just skip
        print(f"[{_now_utc_str()}] [njt] No token available this run; skipping.")
        sys.exit(0)

    # TripUpdates (required)
    tu_bytes_token = post_proto_with_retry("/api/GTFSRT/getTripUpdates", token)
    if tu_bytes_token[0] is None:
        # token refresh blocked by daily limit; skip this run
        sys.exit(0)
    tu_bytes, token = tu_bytes_token
    tu_df = parse_trip_updates(tu_bytes)

    # VehiclePositions (optional)
    vp_df = pd.DataFrame()
    if INCLUDE_VP:
        vp_bytes_token = post_proto_with_retry("/api/GTFSRT/getVehiclePositions", token)
        if vp_bytes_token[0] is None:
            # token refresh blocked by daily limit; still write TU
            vp_df = pd.DataFrame()
        else:
            vp_bytes, token = vp_bytes_token
            vp_df = parse_vehicle_positions(vp_bytes)

    # Write CSVs (UTC timestamp)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    tu_path = OUT_DIR / f"{FILE_PREFIX}_tripupdates_{ts}.csv"
    tu_df.to_csv(tu_path, index=False)
    print(f"[{_now_utc_str()}] [njt] wrote {len(tu_df):,} rows → {tu_path}")

    if INCLUDE_VP:
        vp_path = OUT_DIR / f"{FILE_PREFIX}_vehpos_{ts}.csv"
        vp_df.to_csv(vp_path, index=False)
        print(f"[{_now_utc_str()}] [njt] wrote {len(vp_df):,} rows → {vp_path}")


if __name__ == "__main__":
    main()
