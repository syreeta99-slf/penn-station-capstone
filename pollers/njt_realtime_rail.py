#!/usr/bin/env python3
"""
NJT realtime rail poller
- Treats "Daily token limit reached" as a non-fatal outcome (exit 0)
- Uses consistent logging
- Writes/reads token at ~/.njt/token.json
- Polls NJT realtime endpoint and NORMALIZES to CSV schema required for downstream join:
    [trip_id, stop_id, route_id, rt_arrival_utc, rt_departure_utc, source_minute_utc]
- Appends rows to data/njt_rt/njt_rt_YYYYMMDD.csv (configurable via NJT_RT_DIR)
"""

from __future__ import annotations
import os
import sys
import json
import time
import enum
import logging
import pathlib
import datetime as dt
from typing import Tuple, Optional, Any, List

import requests
import pandas as pd  # NEW

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [njt] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S+0000]"
)
log = logging.getLogger("njt")

# -----------------------------
# Config
# -----------------------------
ENV = os.getenv("NJT_ENV", "prod")
BASE = os.getenv("NJT_BASE", "https://raildata.njtransit.com")
CLIENT_ID = os.getenv("NJT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("NJT_CLIENT_SECRET", "")
TOKEN_DIR = pathlib.Path(os.getenv("NJT_TOKEN_DIR", str(pathlib.Path.home() / ".njt")))
TOKEN_PATH = TOKEN_DIR / "token.json"
TIMEOUT = 30

# Optionally a small "warm" endpoint
WARM_ENDPOINT = "/RailDataWS/Authentication/ValidateToken"

# Where to write normalized CSVs
OUT_DIR = pathlib.Path(os.getenv("NJT_RT_DIR", "data/njt_rt"))

# Endpoint to poll (adjust if you have a different one)
REALTIME_ENDPOINT = "/RailDataWS/Trains/GetAll"

# -----------------------------
# Token status classification
# -----------------------------
class TokenStatus(enum.Enum):
    OK = enum.auto()            # Token available/valid
    DAILY_LIMIT = enum.auto()   # Daily token issuance limit reached (non-fatal)
    UNAVAILABLE = enum.auto()   # Network/other error (fatal)

# -----------------------------
# Helpers
# -----------------------------
def _utc_midnight_next() -> dt.datetime:
    now = dt.datetime.utcnow()
    nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1))
    return nxt

def _save_token(token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps({"token": token, "saved_utc": dt.datetime.utcnow().isoformat()}) + "\n")

def _read_token() -> Optional[str]:
    if not TOKEN_PATH.exists():
        return None
    try:
        data = json.loads(TOKEN_PATH.read_text().strip() or "{}")
        return data.get("token")
    except Exception:
        return None

def _log_env():
    log.info(f"ENV={ENV}  BASE={BASE}")
    log.info(f"token path={TOKEN_PATH}")
    log.info(f"out dir={OUT_DIR}")

def _to_iso_utc(val: Any) -> Optional[str]:
    """
    Coerce various timestamp formats to ISO-8601 UTC strings.
    Accepts ISO strings or epoch seconds/milliseconds (int/float/str).
    Returns None if it can't parse.
    """
    if val is None:
        return None
    try:
        # Already an ISO-like string?
        if isinstance(val, str):
            # fromisoformat cannot parse trailing 'Z'
            try:
                s = val
                if s.endswith("Z"):
                    dt_val = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                else:
                    dt_val = dt.datetime.fromisoformat(s)
                if dt_val.tzinfo is None:
                    dt_val = dt_val.replace(tzinfo=dt.timezone.utc)
                else:
                    dt_val = dt_val.astimezone(dt.timezone.utc)
                return dt_val.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                pass
        # Numeric epoch (seconds or ms)
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
            x = float(val)
            # Heuristic: treat >= 10^12 as milliseconds
            if x > 1e11:
                x = x / 1000.0
            dt_val = dt.datetime.utcfromtimestamp(x).replace(tzinfo=dt.timezone.utc)
            return dt_val.isoformat().replace("+00:00", "Z")
    except Exception:
        return None
    return None

def _get_first(d: dict, keys: List[str]) -> Optional[Any]:
    """Return the first present, truthy value from dict d for any key in keys."""
    for k in keys:
        if k in d and d[k] not in ("", None):
            return d[k]
    return None

# -----------------------------
# Token acquisition
# -----------------------------
def _request_new_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    Hit NJT auth; map 'daily limit' to DAILY_LIMIT without raising.
    Expected responses:
      - 200 with token in JSON
      - 429/400-ish with a body containing 'Daily token limit reached'
    """
    url = f"{BASE}/RailDataWS/Authentication/Authorize"
    payload = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        txt_lower = (r.text or "").lower()

        if r.status_code == 200:
            try:
                token = r.json().get("access_token") or r.json().get("token")
            except Exception:
                # Some implementations return plain text token
                token = r.text.strip()
            if token:
                return TokenStatus.OK, token

        # Known daily-limit signal (status or body text)
        if "daily token limit reached" in txt_lower or r.status_code in (429,):
            log.info("Daily token limit reached; no new token until next UTC day.")
            return TokenStatus.DAILY_LIMIT, None

        # Any other non-200 response
        log.error(f"Token endpoint unexpected status={r.status_code} body={r.text[:300]}")
        return TokenStatus.UNAVAILABLE, None

    except requests.RequestException as e:
        log.error(f"Token request failed: {e}")
        return TokenStatus.UNAVAILABLE, None

def get_token() -> Tuple[TokenStatus, Optional[str]]:
    """
    Try existing token; if absent, try to fetch one.
    DAILY_LIMIT => not an error; return (DAILY_LIMIT, None)
    """
    existing = _read_token()
    if existing:
        return TokenStatus.OK, existing

    status, token = _request_new_token()
    if status is TokenStatus.OK and token:
        _save_token(token)
        return TokenStatus.OK, token

    return status, None

# -----------------------------
# Warm path (lightweight check)
# -----------------------------
def warm(token: Optional[str]) -> bool:
    """
    Return True to keep CI green unless there's a real error.
    If token is None due to DAILY_LIMIT, we simply skip work.
    """
    if not token:
        log.info("No token available this run; skipping warm.")
        return True

    try:
        url = f"{BASE}{WARM_ENDPOINT}"
        hdrs = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=hdrs, timeout=TIMEOUT)
        if r.status_code == 200:
            log.info("Warm check ok.")
            return True
        # If token is invalid/expired, remove local file so next run can re-issue post-midnight
        if r.status_code in (401, 403):
            log.warning(f"Warm check unauthorized ({r.status_code}); clearing cached token.")
            try:
                TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return True  # Non-fatal; next run will attempt re-issue
        log.warning(f"Warm check unexpected status={r.status_code}; proceeding.")
        return True
    except requests.RequestException as e:
        log.error(f"Warm check failed: {e}")
        # Network/infra error—return False only if you want CI to fail on infra issues.
        return False

# -----------------------------
# Normalize & write
# -----------------------------
def _normalize_trips(data: Any) -> list[dict]:
    """
    Normalize the NJT JSON payload to the target schema rows.
    Target columns:
      trip_id, stop_id, route_id, rt_arrival_utc, rt_departure_utc, source_minute_utc
    """
    # Resolve 'trips' list out of various possible shapes
    trips = None
    if isinstance(data, dict):
        trips = data.get("Trips") or data.get("trains")
        if trips is None:
            # Sometimes dict keyed by id
            trips = list(data.values()) if data else []
    elif isinstance(data, list):
        trips = data
    else:
        trips = []

    if trips is None:
        trips = []

    source_minute = dt.datetime.utcnow().replace(second=0, microsecond=0, tzinfo=dt.timezone.utc)
    source_minute_iso = source_minute.isoformat().replace("+00:00", "Z")

    rows: list[dict] = []
    for trip in trips or []:
        if not isinstance(trip, dict):
            continue

        trip_id = _get_first(trip, ["TrainId", "trainId", "TripId", "trip_id", "id"])
        route_id = _get_first(trip, ["LineName", "lineName", "RouteId", "route_id", "line"])

        # Pull stops array (various possible keys)
        stops = _get_first(trip, ["Stops", "stops", "StopTimes", "stop_times"]) or []
        if isinstance(stops, dict):
            stops = list(stops.values())

        for stop in stops:
            if not isinstance(stop, dict):
                continue
            stop_id = _get_first(stop, ["StopId", "stop_id", "StopCode", "StationId", "stationId", "station"])

            arr_raw = _get_first(
                stop,
                ["ArrivalUTC", "arrival_utc", "ArrivalTime", "arrival_time", "arrive", "arr"]
            )
            dep_raw = _get_first(
                stop,
                ["DepartureUTC", "departure_utc", "DepartureTime", "departure_time", "depart", "dep"]
            )

            arr_iso = _to_iso_utc(arr_raw)
            dep_iso = _to_iso_utc(dep_raw)

            # write only if at least one timestamp is present
            if not (arr_iso or dep_iso):
                continue

            rows.append(
                {
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "route_id": route_id,
                    "rt_arrival_utc": arr_iso,
                    "rt_departure_utc": dep_iso,
                    "source_minute_utc": source_minute_iso,
                }
            )
    return rows

def _append_daily_csv(rows: list[dict]) -> None:
    """Append normalized rows to the daily CSV under OUT_DIR."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"njt_rt_{dt.date.today().strftime('%Y%m%d')}.csv"

    if not rows:
        log.info("No realtime rows parsed from NJT response.")
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
# Main poller work
# -----------------------------
def run_poller(token: Optional[str]) -> bool:
    """
    If token is None because of DAILY_LIMIT, we should skip and return True (non-fatal).
    Otherwise, poll realtime endpoint, normalize, and append to CSV.
    """
    if token is None:
        log.info("No token available this run; skipping poll.")
        return True

    try:
        hdrs = {"Authorization": f"Bearer {token}"}
        url = f"{BASE}{REALTIME_ENDPOINT}"
        r = requests.get(url, headers=hdrs, timeout=TIMEOUT)

        if r.status_code in (401, 403):
            log.warning(f"Poll unauthorized ({r.status_code}); clearing cached token.")
            try:
                TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            # Non-fatal; next run will re-issue after midnight UTC or next attempt
            return True

        if r.status_code != 200:
            log.error(f"Poll unexpected status={r.status_code}: {r.text[:300]}")
            return False

        data = r.json()
        rows = _normalize_trips(data)
        _append_daily_csv(rows)
        return True

    except requests.RequestException as e:
        log.error(f"Poll failed: {e}")
        return False
    except Exception as e:
        log.error(f"Polling/normalize error: {e}")
        return False

# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    _log_env()

    status, token = get_token()

    if status is TokenStatus.OK:
        ok = warm(token)
        if not ok:
            sys.exit(1)
        ok = run_poller(token)
        sys.exit(0 if ok else 1)

    if status is TokenStatus.DAILY_LIMIT:
        # Uniform, friendly message; keep step green
        log.info("No token available this run; skipping.")
        reset = _utc_midnight_next().strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info(f"Next token window: {reset}")
        sys.exit(0)

    # UNAVAILABLE or true error
    log.error("Token unavailable due to error; failing this run.")
    sys.exit(1)

if __name__ == "__main__":
    main()
