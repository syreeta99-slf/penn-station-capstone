#!/usr/bin/env python3
"""
NJ TRANSIT Rail – GTFS-RT poller (TripUpdates + VehiclePositions)

Env vars:
  NJT_ENV            : "test" or "prod"  (default: "prod")
  NJT_USERNAME       : your NJT API username
  NJT_PASSWORD       : your NJT API password
  NJT_OUT_DIR        : where to drop CSVs (default: data/realtime_njt)
  NJT_INCLUDE_VP     : "1" to also fetch VehiclePositions (default: "1")
  NJT_FILE_PREFIX    : filename prefix (default: "njt_rail_rt")
  TZ                 : display/filename tz (default: UTC)

Notes (from docs):
- Get a token via POST to /api/GTFSRT/getToken, limited to 10/day; cache and reuse for 24h.
- Real-time GTFS-RT calls (getTripUpdates / getVehiclePositions) have no daily limit. 
  (Use POST with form-data "token")  # sources: docs
"""

import os, json, time, sys
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings

# ---------------- Config ----------------
BASES = {
    "test": "https://testraildata.njtransit.com",
    "prod": "https://raildata.njtransit.com",
}

ENV = os.getenv("NJT_ENV", "prod").strip().lower()
BASE = BASES.get(ENV, BASES["prod"])

USERNAME = os.getenv("NJT_USERNAME")
PASSWORD = os.getenv("NJT_PASSWORD")

OUT_DIR = Path(os.getenv("NJT_OUT_DIR", "data/realtime_njt"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

INCLUDE_VP = os.getenv("NJT_INCLUDE_VP", "1") == "1"
FILE_PREFIX = os.getenv("NJT_FILE_PREFIX", "njt_rail_rt")

TOKEN_CACHE = Path(".njt_token.json")  # local file to avoid >10 token calls/day

TZNAME = os.getenv("TZ", "UTC")

# ---------------- Token helpers ----------------
def _load_token_cached():
    if TOKEN_CACHE.exists():
        try:
            data = json.loads(TOKEN_CACHE.read_text())
            token = data.get("token")
            ts = data.get("ts", 0)
            # consider token good for 23h (docs say ~24h)
            if time.time() - ts < 23*3600 and token:
                return token
        except Exception:
            pass
    return None

def _save_token_cached(token: str):
    TOKEN_CACHE.write_text(json.dumps({"token": token, "ts": time.time()}))

def fetch_token():
    """
    POST {BASE}/api/GTFSRT/getToken with form fields 'username','password'.
    Returns token string like {"Authenticated":"True","UserToken":"..."}.
    Call at most ~once/day.  (NJT limit: 10/day) 
    """
    url = f"{BASE}/api/GTFSRT/getToken"
    resp = requests.post(
        url,
        files={"username": (None, USERNAME or ""), "password": (None, PASSWORD or "")},
        timeout=30,
    )
    resp.raise_for_status()
    try:
        data = resp.json()
        token = data.get("UserToken") or ""
        if not token:
            raise RuntimeError(f"getToken failed: {data}")
        return token
    except ValueError:
        # Some error cases may return 'Null' body; handle as failure
        raise RuntimeError(f"getToken returned non-JSON: {resp.text[:200]}")

def get_token():
    tok = _load_token_cached()
    if tok:
        return tok
    tok = fetch_token()
    _save_token_cached(tok)
    return tok

# ---------------- GTFS-RT calls ----------------
def post_proto(endpoint: str, token: str) -> bytes:
    """
    POST token=form-data to a GTFSRT endpoint, return raw bytes (protobuf).
    E.g. /api/GTFSRT/getTripUpdates or /getVehiclePositions
    """
    url = f"{BASE}{endpoint}"
    r = requests.post(
        url,
        files={"token": (None, token)},
        timeout=60,
    )
    r.raise_for_status()
    return r.content

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

            def ts(e):
                return datetime.fromtimestamp(e, tz=timezone.utc) if e else None

            rows.append({
                "agency": "NJT",
                "entity_id": ent.id or None,
                "trip_id": trip_id,
                "route_id": route_id,
                "stop_id": stop_id,
                "RT_Arrival_UTC": ts(arr_epoch),
                "RT_Departure_UTC": ts(dep_epoch),
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
        lat = getattr(pos, "latitude", None) if pos else None
        lon = getattr(pos, "longitude", None) if pos else None
        bearing = getattr(pos, "bearing", None) if pos else None
        speed = getattr(pos, "speed", None) if pos else None

        rows.append({
            "agency": "NJT",
            "entity_id": ent.id or None,
            "trip_id": v.trip.trip_id if v.HasField("trip") else None,
            "route_id": v.trip.route_id if v.HasField("trip") else None,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
            "lat": lat,
            "lon": lon,
            "bearing": bearing,
            "speed": speed,
            "current_stop_sequence": v.current_stop_sequence if hasattr(v, "current_stop_sequence") else None,
            "current_status": v.current_status if hasattr(v, "current_status") else None,
            "timestamp_utc": datetime.fromtimestamp(v.timestamp, tz=timezone.utc) if hasattr(v, "timestamp") and v.timestamp else None,
            "observed_utc": now_utc,
        })
    return pd.DataFrame(rows)

def main():
    # sanity
    if not USERNAME or not PASSWORD:
        print("[njt] ERROR: set NJT_USERNAME/NJT_PASSWORD env vars", file=sys.stderr)
        sys.exit(2)

    token = get_token()

    # TripUpdates
    tu_bytes = post_proto("/api/GTFSRT/getTripUpdates", token)
    tu_df = parse_trip_updates(tu_bytes)

    # VehiclePositions (optional)
    if INCLUDE_VP:
        vp_bytes = post_proto("/api/GTFSRT/getVehiclePositions", token)
        vp_df = parse_vehicle_positions(vp_bytes)
    else:
        vp_df = pd.DataFrame()

    # write CSVs
    # Use UTC in filename; you can switch to America/New_York if preferred
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tu_path = OUT_DIR / f"{FILE_PREFIX}_tripupdates_{ts}.csv"
    tu_df.to_csv(tu_path, index=False)
    print(f"[njt] wrote {len(tu_df):,} rows → {tu_path}")

    if INCLUDE_VP:
        vp_path = OUT_DIR / f"{FILE_PREFIX}_vehpos_{ts}.csv"
        vp_df.to_csv(vp_path, index=False)
        print(f"[njt] wrote {len(vp_df):,} rows → {vp_path}")

if __name__ == "__main__":
    main()
