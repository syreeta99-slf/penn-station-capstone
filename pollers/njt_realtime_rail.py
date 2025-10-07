#!/usr/bin/env python3
"""
NJT realtime rail poller (GTFS-RT, Production)

- Auth:    POST multipart to /api/GTFSRT/getToken (username,password)
- Validate:POST multipart to /api/GTFSRT/isValidToken (token)
- Fetch:   POST multipart to /api/GTFSRT/getTripUpdates (token) -> protobuf bytes
- Parse:   google.transit.gtfs_realtime_pb2
- Output:  Append normalized rows to CSV per UTC day

Env vars:
  NJT_USERNAME   = (required) API username
  NJT_PASSWORD   = (required) API password
  NJT_RT_DIR     = (optional, default "data/njt_rt")
  NJT_TOKEN_DIR  = (optional, default "~/.njt")
  NJT_ENV        = (optional, default "prod")

Exit codes:
  0 = success or benign skip (e.g. daily token limit reached)
  1 = error (network, HTTP, parse, or missing creds)
"""

from __future__ import annotations

import os
import sys
import json
import enum
import pathlib
import logging
import datetime as dt
from typing import Optional, Tuple, List

import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings


# -----------------------------
# Config
# -----------------------------
USERNAME = os.getenv("NJT_USERNAME", "")
PASSWORD = os.getenv("NJT_PASSWORD", "")
TOKEN_DIR = pathlib.Path(os.getenv("NJT_TOKEN_DIR", str(pathlib.Path.home() / ".njt")))
TOKEN_PATH = TOKEN_DIR / "token.json"
OUT_DIR = pathlib.Path(os.getenv("NJT_RT_DIR", "data/njt_rt"))

# Hard-coded prod base (no NJT_BASE override needed anymore)
BASE = "https://raildata.njtransit.com"
EP_TOKEN = "/api/GTFSRT/getToken"
EP_VALIDATE = "/api/GTFSRT/isValidToken"
EP_TRIP_UPDATES = "/api/GTFSRT/getTripUpdates"
TIMEOUT = 45


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [njt] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S+0000]",
)
log = logging.getLogger("njt")


# -----------------------------
# Token status
# -----------------------------
class TokenStatus(enum.Enum):
    OK = enum.auto()
    DAILY_LIMIT = enum.auto()
    UNAVAILABLE = enum.auto()


# -----------------------------
# Token helpers
# -----------------------------
def _save_token(token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(
        json.dumps({"token": token, "saved_utc": dt.datetime.utcnow().isoformat()}),
        encoding="utf-8",
    )


def _read_token() -> Optional[str]:
    try:
        if TOKEN_PATH.exists():
            txt = TOKEN_PATH.read_text(encoding="utf-8").strip()
            if txt:
                return json.loads(txt).get("token")
    except Exception:
        pass
    return None


def _request_new_token() -> Tuple[TokenStatus, Optional[str]]:
    url = f"{BASE}{EP_TOKEN}"
    files = {"username": (None, USERNAME), "password": (None, PASSWORD)}
    try:
        r = requests.post(url, files=files, timeout=TIMEOUT)
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                data = {}
            token = data.get("UserToken") or data.get("token") or ""
            if token:
                return TokenStatus.OK, token
            if "daily token limit" in (r.text or "").lower():
                return TokenStatus.DAILY_LIMIT, None
            log.error(f"Token 200 without usable token; body={r.text[:200]!r}")
            return TokenStatus.UNAVAILABLE, None
        if r.status_code == 429:
            return TokenStatus.DAILY_LIMIT, None
        log.error(f"Token unexpected {r.status_code} {r.text[:200]!r}")
        return TokenStatus.UNAVAILABLE, None
    except requests.RequestException as e:
        log.error(f"Token request failed: {e}")
        return TokenStatus.UNAVAILABLE, None


def _validate_token(token: str) -> bool:
    try:
        r = requests.post(f"{BASE}{EP_VALIDATE}", files={"token": (None, token)}, timeout=TIMEOUT)
        return r.status_code == 200 and "true" in (r.text or "").lower()
    except requests.RequestException:
        return False


def get_token() -> Tuple[TokenStatus, Optional[str]]:
    tok = _read_token()
    if tok and _validate_token(tok):
        return TokenStatus.OK, tok
    if tok:
        try:
            TOKEN_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    status, token = _request_new_token()
    if status is TokenStatus.OK and token:
        _save_token(token)
        return status, token
    return status, None


# -----------------------------
# GTFS-RT fetch/parse
# -----------------------------
def _to_iso_utc(epoch: Optional[int | float]) -> Optional[str]:
    if epoch is None:
        return None
    ts = float(epoch)
    if ts > 1e11:
        ts /= 1000.0
    return (
        dt.datetime.utcfromtimestamp(ts)
        .replace(tzinfo=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _fetch_trip_updates(token: str) -> bytes:
    r = requests.post(f"{BASE}{EP_TRIP_UPDATES}", files={"token": (None, token)}, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"TripUpdates HTTP {r.status_code}: {r.text[:200]}")
    return r.content


def _parse_trip_updates(pb_bytes: bytes) -> List[dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(pb_bytes)

    source_minute = dt.datetime.utcnow().replace(second=0, microsecond=0, tzinfo=dt.timezone.utc)
    source_iso = source_minute.isoformat().replace("+00:00", "Z")

    rows: List[dict] = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip_id = tu.trip.trip_id or ""
        route_id = tu.trip.route_id or ""
        for stu in tu.stop_time_update:
            stop_id = getattr(stu, "stop_id", "")
            arr_epoch = stu.arrival.time if stu.HasField("arrival") else None
            dep_epoch = stu.departure.time if stu.HasField("departure") else None
            arr_iso = _to_iso_utc(arr_epoch)
            dep_iso = _to_iso_utc(dep_epoch)
            if not (arr_iso or dep_iso):
                continue
            rows.append(
                {
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "route_id": route_id,
                    "rt_arrival_utc": arr_iso,
                    "rt_departure_utc": dep_iso,
                    "source_minute_utc": source_iso,
                }
            )
    return rows


def _append_daily_csv(rows: List[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"njt_rt_{dt.date.today().strftime('%Y%m%d')}.csv"
    if not rows:
        log.info("No realtime rows parsed.")
        return
    df = pd.DataFrame.from_records(rows)
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=write_header)
    log.info(f"Wrote {len(df)} rows → {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not USERNAME or not PASSWORD:
        log.error("Missing NJT_USERNAME or NJT_PASSWORD.")
        sys.exit(1)

    status, token = get_token()
    if status is TokenStatus.OK and token:
        try:
            pb = _fetch_trip_updates(token)
            rows = _parse_trip_updates(pb)
            _append_daily_csv(rows)
            sys.exit(0)
        except Exception as e:
            log.error(f"Poll error: {e}")
            sys.exit(1)
    if status is TokenStatus.DAILY_LIMIT:
        log.info("Daily token limit reached; skipping until tomorrow.")
        sys.exit(0)
    log.error("Token unavailable; failing this run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
