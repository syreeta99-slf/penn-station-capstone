#!/usr/bin/env python3
"""
NJ TRANSIT Rail – GTFS-RT poller (TripUpdates + VehiclePositions)

Environment variables:
  NJT_ENV            : "test" or "prod"  (default: "prod")
  NJT_USERNAME       : NJT API username (Actions secret recommended)
  NJT_PASSWORD       : NJT API password (Actions secret recommended)
  NJT_OUT_DIR        : output folder for CSVs (default: data/realtime_njt)
  NJT_INCLUDE_VP     : "1" to fetch VehiclePositions too, else "0" (default: "1")
  NJT_FILE_PREFIX    : filename prefix (default: "njt_rail_rt")
  TZ                 : display/filename tz, currently only used for logs (default: "UTC")

Notes:
- Token endpoint (/api/GTFSRT/getToken) is limited (~10/day). We cache a token to
  .njt_token.json and reuse it for ~23h. Also auto-refresh once on 401.
- GTFS-RT endpoints (getTripUpdates, getVehiclePositions) have no daily limit.
"""

from __future__ import annotations

import os
import json
import time
import sys
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

TOKEN_CACHE = Path(".njt_token.json")  # persisted across runs via actions/cache
TZNAME = os.getenv("TZ", "UTC")

# --------------------- Small log helpers ------------------

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def _token_age_str() -> str:
    try:
        obj = json.loads(TOKEN_CACHE.read_text())
        age = int(time.time() - obj.get("ts", 0))
        return f"{age//3600}h{(age % 3600)//60}m"
    except Exception:
        return "unknown"

# ----------------------- Token flow -----------------------

def _load_token_cached() -> str | None:
    if TOKEN_CACHE.exists():
        try:
            data = json.loads(TOKEN_CACHE.read_text())
            token = data.get("token")
            ts = data.get("ts", 0)
            # treat as valid for ~23h
            if token and (time.time() - ts) < 23 * 3600:
                return token
        except Exception:
            pass
    return None

def _save_token_cached(token: str) -> None:
    TOKEN_CACHE.write_text(json.dumps({"token": token, "ts": time.time()}))

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
    Returns token string or raises RuntimeError.
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
            else:
                last = f"{r.status_code} {r.text[:200]!r}"
        except Exception as e:
            last = f"EXC {type(e).__name__}: {e}"
        time.sleep(backoff)
    raise RuntimeError(f"getToken failed after retries; last={last}")

def get_token() -> str:
    tok = _load_token_cached()
    if tok:
        return tok
    tok = fetch_token()
    _save_token_cached(tok)
    return tok

# ------------------- GTFS-RT retrieval --------------------

def post_proto_with_retry(endpoint: str, token: str, max_attempts: int = 5) -> tuple[bytes, str]:
    """
    POST to an NJT GTFS-RT endpoint (with token as form-data) with retries/backoff.

    Behavior:
      - 200: return (content, token).
      - 401: refresh token once, save, and retry immediately.
      - 429/5xx: exponential backoff and retry.
      - other: raise RuntimeError after first failure.

    Returns: (protobuf_bytes, token) -- token may be updated if refresh occurred.
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
                # likely expired token → refresh exactly once
                print(f"[{_now_utc_str()}] [njt] {endpoint} got 401; refreshing token…")
                token = fetch_token()
                _save_token_cached(token)
                refreshed = True
                continue
            if sc in (429, 500, 502, 503, 504):
                # transient → backoff
                backoff = min(2 ** attempt, 20)
                print(f"[{_now_utc_str()}] [njt] {endpoint} {sc}; retrying in {backoff}s (attempt {attempt}/{max_attempts})")
                time.sleep(backoff)
                continue
            # other codes → fail fast
            raise RuntimeError(f"{endpoint} failed: {sc} {r.text[:200]!r}")
        except requests.RequestException as e:
            # network hiccup → backoff
            backoff = min(2 ** attempt, 20)
            print(f"[{_now_utc_str()}] [njt] {endpoint} network error {type(e).__name__}: {e}; sleep {backoff}s")
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
    if not USERNAME or not PASSWORD:
        print("[njt] ERROR: set NJT_USERNAME/NJT_PASSWORD environment secrets", file=sys.stderr)
        sys.exit(2)

    print(f"[{_now_utc_str()}] [njt] ENV={ENV}  BASE={BASE}")
    print(f"[{_now_utc_str()}] [njt] token cache exists={TOKEN_CACHE.exists()} age={_token_age_str()}")

    token = get_token()

    # TripUpdates (required)
    tu_bytes, token = post_proto_with_retry("/api/GTFSRT/getTripUpdates", token)
    tu_df = parse_trip_updates(tu_bytes)

    # VehiclePositions (optional)
    vp_df = pd.DataFrame()
    if INCLUDE_VP:
        vp_bytes, token = post_proto_with_retry("/api/GTFSRT/getVehiclePositions", token)
        vp_df = parse_vehicle_positions(vp_bytes)

    # Write CSVs (UTC timestamp in filename for consistency)
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
