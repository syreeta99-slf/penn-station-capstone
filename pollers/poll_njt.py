#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from typing import List
from pathlib import Path
from hashlib import md5

import pandas as pd
import numpy as np
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

# =========================
# Config (env)
# =========================
STOP_ID = os.getenv("NJT_STOP_ID", "105")  # Penn Station (NJT)
TRIP_UPDATES_URL = os.getenv("NJT_GET_TRIP_UPDATES_URL", "https://raildata.njtransit.com/api/GTFSRT/getTripUpdates")

# rclone remote and Drive locations
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")
GDRIVE_DIR    = os.getenv("GDRIVE_DIR_NJT", "penn-station/njt")
MASTER_NAME   = os.getenv("NJT_MASTER_NAME", "njt_penn_master.csv")

SUBSYSTEM_TAG = "njt"  # used in _uid

TOKEN_PATH = pathlib.Path.home() / ".njt" / "token.json"

FIELDS = [
    "pull_utc","server_ts","trip_id","route_id","stop_id",
    "arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"
]

# =========================
# Utilities
# =========================
def ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def out_name() -> str:
    return f"njt_rt_{ts()}.csv"

def load_token() -> str:
    if not TOKEN_PATH.exists():
        print("[fatal] missing ~/.njt/token.json; run daily token workflow first", file=sys.stderr)
        sys.exit(2)
    try:
        js = json.loads(TOKEN_PATH.read_text())
        tok = js.get("UserToken") or js.get("token") or js.get("access_token")
        if not tok:
            raise ValueError("No token field found")
        return tok
    except Exception as e:
        print(f"[fatal] could not read token: {e}", file=sys.stderr)
        sys.exit(2)

def fetch_trip_updates(user_token: str) -> bytes:
    # NJT expects multipart/form-data with 'token' field for GTFS-RT endpoints
    files = {"token": (None, user_token)}
    r = requests.post(TRIP_UPDATES_URL, files=files, timeout=30)
    if r.status_code == 401:
        print("[warn] 401 Unauthorized with cached token; not reissuing in poll job", file=sys.stderr)
        sys.exit(78)
    r.raise_for_status()
    return r.content  # protobuf bytes

def parse_rows(feed_bytes: bytes, target_stop: str) -> List[dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(feed_bytes)
    rows: List[dict] = []
    server_ts = int(feed.header.timestamp or time.time())
    for ent in feed.entity:
        tu = ent.trip_update
        if not tu:
            continue
        for stu in tu.stop_time_update:
            if stu.stop_id != target_stop:
                continue
            arr = getattr(stu.arrival, "time", None) or None
            dep = getattr(stu.departure, "time", None) or None
            delay = getattr(stu.arrival, "delay", None) or getattr(stu.departure, "delay", None) or None
            rows.append({
                "pull_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "server_ts": server_ts,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,
                "stop_id": stu.stop_id,
                "arrival_time": arr,
                "departure_time": dep,
                "delay_sec": delay,
                "schedule_relationship": tu.trip.schedule_relationship,
                "entity_id": ent.id,
            })
    return rows

def write_csv(path: str, rows: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------- _uid utilities ----------
def get_event_epoch(row: dict):
    # Prefer explicit epoch times if present
    for k in ("arrival_time", "departure_time", "event_epoch"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                pass
    # Fallback to server_ts or pull_utc (epoch or ISO)
    for k in ("server_ts", "pull_utc"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)  # epoch numeric?
            except Exception:
                try:
                    return pd.to_datetime(v, utc=True).view("int64") / 1e9
                except Exception:
                    return np.nan
    return np.nan

def choose_tripish(row: dict) -> str:
    # pick a stable trip-ish identifier
    for k in ("trip_id", "entity_id", "vehicle_ref", "train_id", "block_id", "route_id"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: dict) -> str:
    stop_id = str(row.get("stop_id", "") or "")
    tripish = choose_tripish(row)
    evt = get_event_epoch(row)
    base = f"{SUBSYSTEM_TAG}|{stop_id}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

# ---------- rclone-based master update ----------
def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
    """
    Download master from Drive (if exists) -> concat -> add _uid -> dedup -> upload back.
    """
    if GDRIVE_REMOTE is None or GDRIVE_REMOTE.strip() == "":
        print("Missing GDRIVE_REMOTE_NAME env.", file=sys.stderr)
        sys.exit(2)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")

        # Attempt to pull existing master
        pulled = subprocess.run(
            ["rclone", "copyto", remote_master, local_master],
            capture_output=True, text=True
        )
        if pulled.returncode == 0 and os.path.exists(local_master):
            df_master = pd.read_csv(local_master)
        else:
            df_master = pd.DataFrame(columns=local_new_df.columns)

        # Backfill _uid if missing in existing master
        if len(df_master) and "_uid" not in df_master.columns:
            df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Ensure new pull has _uid
        if "_uid" not in local_new_df.columns:
            local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Align columns & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master = df_master.reindex(columns=all_cols)
        local_new_df = local_new_df.reindex(columns=all_cols)

        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # Prefer most recent observation per _uid if server_ts present
        if "server_ts" in combined.columns:
            combined = combined.sort_values("server_ts", na_position="last")
        combined = combined.drop_duplicates(subset=["_uid"], keep="last")

        # Save & push back to Drive
        combined.to_csv(local_master, index=False)
        subprocess.check_call(["rclone", "copyto", local_master, remote_master])
        print(f"✅ {SUBSYSTEM_TAG} master now {len(combined)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

# =========================
# Main
# =========================
def main():
    # Sanity
    if not GDRIVE_REMOTE:
        print("Missing GDRIVE_REMOTE_NAME env.", file=sys.stderr)
        sys.exit(2)

    # 1) Fetch + parse
    token = load_token()
    data = fetch_trip_updates(token)
    rows = parse_rows(data, STOP_ID)

    # 2) Raw snapshot (local + Drive)
    outpath = os.path.join("data", "njt", out_name())
    write_csv(outpath, rows)
    subprocess.run(
        ["rclone", "copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"],
        check=False
    )

    # 3) DataFrame for this pull → de-dupe immediate repeats → add _uid → update master
    df_poll = pd.DataFrame(rows, columns=FIELDS)

    # Drop immediate duplicates within this pull (optional)
    dedupe_key = [c for c in ["stop_id", "trip_id", "entity_id", "arrival_time", "departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    # Merge with Drive master via rclone and de-dup by _uid
    update_master_with_uid_rclone(df_poll)

    print(f"[ok] NJT poll (stop {STOP_ID}) completed with {len(rows)} rows")

if __name__ == "__main__":
    main()
