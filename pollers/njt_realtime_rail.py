#!/usr/bin/env python3
"""
NJT realtime rail poller (GTFS-RT, prod) — token-sparing version

Strategy:
- TRUST the cached token for the whole UTC day (no isValidToken calls).
- Only hit getToken if:
    a) no cached token, OR
    b) TripUpdates returns 401/403 (then retry once with a fresh token).
- If NJT returns the daily limit message, exit 0 (benign skip).

Env (required):
  NJT_USERNAME, NJT_PASSWORD

Env (optional):
  NJT_RT_DIR           default "data/njt_rt"
  NJT_TOKEN_DIR        default "~/.njt"
  NJT_FILTER_STOP_IDS  default "105" (comma-separated stop_ids to keep; "" means keep all)

Output:
  data/njt_rt/njt_rt_YYYYMMDD.csv with:
    trip_id, stop_id, route_id, rt_arrival_utc, rt_departure_utc, source_minute_utc
"""

from __future__ import annotations
import os, sys, json, pathlib, logging, datetime as dt
from typing import Optional, List, Tuple, Set

import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings

# -----------------------------
# Config
# -----------------------------
USERNAME = os.getenv("NJT_USERNAME", "")
PASSWORD = os.getenv("NJT_PASSWORD", "")
OUT_DIR = pathlib.Path(os.getenv("NJT_RT_DIR", "data/njt_rt"))
TOKEN_DIR = pathlib.Path(os.getenv("NJT_TOKEN_DIR", str(pathlib.Path.home() / ".njt")))
TOKEN_PATH = TOKEN_DIR / "token.json"
TOKEN_PATH_ENV = os.getenv("NJT_TOKEN_PATH", "").strip()
if TOKEN_PATH_ENV:
    TOKEN_PATH = pathlib.Path(os.path.expanduser(TOKEN_PATH_ENV))

FILTER_IDS_RAW = os.getenv("NJT_FILTER_STOP_IDS", "105")
FILTER_STOP_IDS: Set[str] = {s.strip() for s in FILTER_IDS_RAW.split(",") if s.strip()}

BASE = "https://raildata.njtransit.com"
EP_TOKEN = "/api/GTFSRT/getToken"
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
# Token cache helpers
# -----------------------------
def read_cached_token() -> Optional[str]:
    """Read token from disk. Log if we are going to reuse it."""
    try:
        if TOKEN_PATH.exists():
            raw = TOKEN_PATH.read_text(encoding="utf-8").strip()
            if raw:
                data = json.loads(raw)
                tok = data.get("token")
                saved = data.get("saved_utc")
                if tok:
                    log.info(f"Using cached token (saved_utc={saved}) from {TOKEN_PATH}")
                    return tok
    except Exception as e:
        log.warning(f"Could not read cached token: {e}")
    return None

def write_cached_token(token: str) -> None:
    """Atomic write to avoid partial files on ephemeral runners."""
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_PATH.with_suffix(".tmp")
    payload = {"token": token, "saved_utc": dt.datetime.utcnow().isoformat()}
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(TOKEN_PATH)  # atomic on POSIX
    
def clear_cached_token() -> None:
    try:
        TOKEN_PATH.unlink(missing_ok=True)
        log.info(f"Cleared cached token at {TOKEN_PATH}")
    except Exception as e:
        log.warning(f"Could not clear cached token: {e}")

def request_new_token() -> Tuple[Optional[str], str]:
    """Return (token_or_none, status_tag) where status_tag is 'ok', 'daily_limit', or 'error'."""
    url = f"{BASE}{EP_TOKEN}"
    files = {"username": (None, USERNAME), "password": (None, PASSWORD)}
    try:
        r = requests.post(url, files=files, timeout=TIMEOUT)
        body_lower = (r.text or "").lower()
        if r.status_code == 200:
            # Expected JSON: { "UserToken": "..." }
            try:
                data = r.json()
            except Exception:
                data = {}
            token = data.get("UserToken") or data.get("token") or ""
            if token:
                return token, "ok"
            if "daily token limit" in body_lower:
                return None, "daily_limit"
            log.error(f"Token 200 without usable token; body={r.text[:200]!r}")
            return None, "error"
        # Some servers return 429 or even 500 with the daily limit text
        if r.status_code in (429, 500) and "daily usage limit" in body_lower:
            log.info(r.text.strip()[:180])
            return None, "daily_limit"
        log.error(f"Token unexpected {r.status_code}: {r.text[:200]!r}")
        return None, "error"
    except requests.RequestException as e:
        log.error(f"Token request failed: {e}")
        return None, "error"

# -----------------------------
# GTFS-RT fetch / parse
# -----------------------------
def fetch_trip_updates(token: str) -> Tuple[Optional[bytes], int, str]:
    """Return (protobuf_bytes_or_none, http_status, body_preview)."""
    url = f"{BASE}{EP_TRIP_UPDATES}"
    try:
        r = requests.post(url, files={"token": (None, token)}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.content, r.status_code, ""
        return None, r.status_code, (r.text or "")[:200]
    except requests.RequestException as e:
        return None, 0, str(e)

def to_iso_utc(epoch) -> Optional[str]:
    if epoch is None or pd.isna(epoch):
        return None
    try:
        ts = float(epoch)
        if ts > 1e11:
            ts /= 1000.0
        return (
            dt.datetime.utcfromtimestamp(ts)
            .replace(tzinfo=dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception:
        return None

def parse_trip_updates(pb_bytes: bytes) -> List[dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(pb_bytes)

    source_min = dt.datetime.utcnow().replace(second=0, microsecond=0, tzinfo=dt.timezone.utc)
    source_iso = source_min.isoformat().replace("+00:00", "Z")

    rows: List[dict] = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip_id = tu.trip.trip_id or ""
        route_id = tu.trip.route_id or ""
        for stu in tu.stop_time_update:
            stop_id = getattr(stu, "stop_id", "") or ""
            arr = stu.arrival.time if stu.HasField("arrival") else None
            dep = stu.departure.time if stu.HasField("departure") else None
            arr_iso = to_iso_utc(arr)
            dep_iso = to_iso_utc(dep)
            if not (arr_iso or dep_iso):
                continue
            rows.append({
                "trip_id": trip_id,
                "stop_id": str(stop_id),
                "route_id": route_id,
                "rt_arrival_utc": arr_iso,
                "rt_departure_utc": dep_iso,
                "source_minute_utc": source_iso,
            })
    return rows

def apply_stop_filter(rows: List[dict]) -> List[dict]:
    if not FILTER_STOP_IDS:
        return rows
    keep = {s.strip() for s in FILTER_STOP_IDS}
    return [r for r in rows if r.get("stop_id") and str(r["stop_id"]) in keep]

def append_daily_csv(rows: List[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"njt_rt_{dt.date.today().strftime('%Y%m%d')}.csv"
    if not rows:
        log.info("No realtime rows after filtering; nothing to write.")
        return
    df = pd.DataFrame.from_records(rows, columns=[
        "trip_id","stop_id","route_id","rt_arrival_utc","rt_departure_utc","source_minute_utc"
    ])
    write_header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=write_header)
    log.info(f"Wrote {len(df)} NJT realtime rows → {out_path}")

# -----------------------------
# Main (token-sparing)
# -----------------------------
def main() -> None:
    log.info(f"token path={TOKEN_PATH}")
    log.info(f"out dir={OUT_DIR}")
    log.info(f"Filter stop_ids: {sorted(FILTER_STOP_IDS) if FILTER_STOP_IDS else 'ALL'}")

    if not USERNAME or not PASSWORD:
        log.error("Missing NJT_USERNAME or NJT_PASSWORD.")
        sys.exit(1)

    # 1) Try cached token first (no validation call)
    token = read_cached_token()
    tried_refresh = False

    for attempt in (1, 2):  # at most one refresh
        if not token:
            # no cached token → request one
            new_token, status = request_new_token()
            if status == "ok":
                token = new_token
                write_cached_token(token)
                log.info("Acquired new token (no cached token present).")
            elif status == "daily_limit":
                log.info("Daily token limit reached; skipping until next UTC day.")
                sys.exit(0)
            else:
                log.error("Token unavailable; failing this run.")
                sys.exit(1)

        # 2) Use token
        pb, status_code, preview = fetch_trip_updates(token)
        if pb is not None:
            rows = parse_trip_updates(pb)
            if FILTER_STOP_IDS:
                before = len(rows)
                rows = apply_stop_filter(rows)
                log.info(f"Parsed {before} rows; kept {len(rows)} after filter")
            append_daily_csv(rows)
            sys.exit(0)

        # 3) Handle auth errors → refresh once
        if status_code in (401, 403) and not tried_refresh:
            log.info(f"TripUpdates {status_code}; refreshing token once.")
            clear_cached_token()
            token = None
            tried_refresh = True
            continue  # retry with new token
        elif status_code in (429, 500) and "daily usage limit" in preview.lower():
            log.info(preview.strip())
            log.info("Daily token limit hit when fetching data; skipping.")
            sys.exit(0)
        else:
            log.error(f"TripUpdates HTTP {status_code}: {preview}")
            sys.exit(1)

if __name__ == "__main__":
    main()
