#!/usr/bin/env python3
"""
NJT realtime rail poller (GTFS-RT, Production)

- Auth:    POST multipart to /api/GTFSRT/getToken (fields: username, password)
- Validate:POST multipart to /api/GTFSRT/isValidToken (field: token)
- Fetch:   POST multipart to /api/GTFSRT/getTripUpdates (field: token) -> protobuf bytes
- Parse:   google.transit.gtfs_realtime_pb2
- Output:  Append normalized rows to CSV per UTC day

Env vars:
  NJT_BASE=https://raildata.njtransit.com
  NJT_USERNAME=xxxxx
  NJT_PASSWORD=xxxxx
  NJT_RT_DIR=data/njt_rt
  NJT_TOKEN_DIR=~/.njt

Exit codes:
  0 = success or benign skip (e.g., daily token limit reached)
  1 = real error (e.g., network, unexpected HTTP, parsing)
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
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [njt] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S+0000]",
)
log = logging.getLogger("njt")


# -----------------------------
# Config
# -----------------------------
BASE = os.getenv("NJT_BASE", "https://raildata.njtransit.com").rstrip("/")
USERNAME = os.getenv("NJT_USERNAME", "")
PASSWORD = os.getenv("NJT_PASSWORD", "")

TOKEN_DIR = pathlib.Path(os.getenv("NJT_TOKEN_DIR", str(pathlib.Path.home() / ".njt")))
TOKEN_PATH = TOKEN_DIR / "token.json"

OUT_DIR = pathlib.Path(os.getenv("NJT_RT_DIR", "data/njt_rt"))

TIMEOUT = 45  # seconds

# Endpoints (prod)
EP_TOKEN = "/api/GTFSRT/getToken"
EP_VALIDATE = "/api/GTFSRT/isValidToken"
EP_TRIP_UPDATES = "/api/GTFSRT/getTripUpdates"


# -----------------------------
# Token status enum
# -----------------------------
class TokenStatus(enum.Enum):
    OK = enum.auto()
    DAILY_LIMIT = enum.auto()
    UNAVAILABLE = enum.auto()


# -----------------------------
# Helpers
# -----------------------------
def _log_env() -> None:
    log.info(f"BASE={BASE}")
    log.info(f"token path={TOKEN_PATH}")
    log.info(f"out dir={OUT_DIR}")


def _utc_midnight_next() -> str:
    now = dt.datetime.utcnow()
    nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1))
    return nxt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _save_token(token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(
        json.dumps({"token": token, "saved_utc": dt.datetime.utcnow().isoformat()}) + "\n",
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


def _to_iso_utc(epoch: Optional[int | float]) -> Optional[str]:
    if epoch is None:
        return None
    try:
        ts = float(epoch)
        if ts > 1e11:  # ms -> s
            ts = ts / 1000.0
        return (
            dt.datetime.utcfromtimestamp(ts)
            .replace(tzinfo=dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception:
        return None


# -----------------------------
# Auth (multipart/form-data)
# -----------------------------
def _request_new_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    POST multipart to getToken(username,password).
    Interpret non-200 or weird bodies conservatively.
    """
    url = f"{BASE}{EP_TOKEN}"
    files = {
        "username": (None, USERNAME),
        "password": (None, PASSWORD),
    }
    try:
        r = requests.post(url, files=files, timeout=TIMEOUT)
        log.info(f"POST {url} -> {r.status_code} {r.headers.get('Content-Type')}")
        if r.status_code == 200:
            # Expect JSON with a token field (often 'UserToken')
            try:
                data = r.json()
            except Exception:
                data = {}
            token = data.get("UserToken") or data.get("token") or ""
            if token:
                return TokenStatus.OK, token

            # Some servers reply 200 with message when daily limit hit
            body_lower = (r.text or "").lower()
            if "daily token limit" in body_lower:
                log.info("Daily token limit reached; no new token until next UTC day.")
                return TokenStatus.DAILY_LIMIT, None

            log.error(f"Token 200 without usable token; body={r.text[:300]!r}")
            return TokenStatus.UNAVAILABLE, None

        # Daily limit may also be 429
        if r.status_code == 429:
            log.info("Daily token limit (HTTP 429); skipping until next UTC day.")
            return TokenStatus.DAILY_LIMIT, None

        # Some misconfigurations return 204 or 404; log the preview
        log.error(f"Token endpoint unexpected status={r.status_code} body={r.text[:300]!r}")
        return TokenStatus.UNAVAILABLE, None

    except requests.RequestException as e:
        log.error(f"Token request failed: {e}")
        return TokenStatus.UNAVAILABLE, None


def _validate_token(token: str) -> bool:
    """
    POST multipart to isValidToken(token). Many servers return JSON-ish text; accept any 200 containing 'true'.
    """
    try:
        url = f"{BASE}{EP_VALIDATE}"
        r = requests.post(url, files={"token": (None, token)}, timeout=TIMEOUT)
        log.info(f"POST {url} -> {r.status_code} {r.headers.get('Content-Type')}")
        if r.status_code != 200:
            return False
        txt = (r.text or "").lower()
        # Accept a broad set of "true" representations
        return "true" in txt or '"valid":true' in txt or "'valid': true" in txt
    except requests.RequestException:
        return False


def get_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    Read cached token; if invalid or absent, request a new one.
    """
    tok = _read_token()
    if tok:
        if _validate_token(tok):
            return TokenStatus.OK, tok
        else:
            try:
                TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    status, token = _request_new_token()
    if status is TokenStatus.OK and token:
        _save_token(token)
        return TokenStatus.OK, token
    return status, None


# -----------------------------
# Fetch & parse GTFS-RT
# -----------------------------
def _fetch_trip_updates(token: str) -> bytes:
    url = f"{BASE}{EP_TRIP_UPDATES}"
    r = requests.post(url, files={"token": (None, token)}, timeout=TIMEOUT)
    log.info(f"POST {url} -> {r.status_code} {r.headers.get('Content-Type')}")
    if r.status_code != 200:
        raise RuntimeError(f"TripUpdates HTTP {r.status_code}: {r.text[:200]}")
    return r.content  # protobuf bytes


def _parse_trip_updates(pb_bytes: bytes) -> List[dict]:
    """
    Return rows:
      trip_id, stop_id, route_id, rt_arrival_utc, rt_departure_utc, source_minute_utc
    """
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
            stop_id = getattr(stu, "stop_id", "") or ""
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
        log.info("No realtime rows parsed from NJT TripUpdates.")
        return
    df = pd.DataFrame.from_records(
        rows,
        columns=[
            "trip_id",
            "stop_id",
            "route_id",
            "rt_arrival_utc",
            "rt_departure_utc",
            "source_minute_utc",
        ],
    )
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=write_header)
    log.info(f"Wrote {len(df)} NJT realtime rows → {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    _log_env()

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
            msg = str(e)
            log.error(f"Poll error: {msg[:300]}")
            # If token went stale mid-run, drop cache so next run re-issues
            if "401" in msg or "403" in msg:
                try:
                    TOKEN_PATH.unlink(missing_ok=True)
                except Exception:
                    pass
            sys.exit(1)

    if status is TokenStatus.DAILY_LIMIT:
        log.info("No token available this run; skipping.")
        log.info(f"Next token window: {_utc_midnight_next()}")
        sys.exit(0)

    log.error("Token unavailable due to error; failing this run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
