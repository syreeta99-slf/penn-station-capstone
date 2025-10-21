#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from typing import Set, List
from pathlib import Path
from hashlib import md5

import pandas as pd
import numpy as np
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

# =========================
# ENV / Config
# =========================
LIRR_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("LIRR_STOP_IDS","").split(",") if s.strip()}  # e.g., "237"
MTA_API_KEY    = os.getenv("MTA_API_KEY")  # optional but recommended
LIRR_FEED_URL  = os.getenv("LIRR_FEED_URL", "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/lirr%2Fgtfs-lirr")

# rclone remote + Drive paths
GDRIVE_REMOTE  = os.getenv("GDRIVE_REMOTE_NAME")              # required (e.g., "gdrive")
GDRIVE_DIR     = os.getenv("GDRIVE_DIR_LIRR", "penn-station/lirr")
MASTER_NAME    = os.getenv("LIRR_MASTER_NAME", "lirr_penn_master.csv")

SUBSYSTEM_TAG  = "lirr"  # used in _uid

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
    return f"lirr_rt_{ts()}.csv"

def fetch(url: str) -> gtfs_realtime_pb2.FeedMessage:
    headers = {"x-api-key": MTA_API_KEY} if MTA_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def rows_from_feed(feed: gtfs_realtime_pb2.FeedMessage, stops: Set[str]) -> List[dict]:
    rows = []
    server_ts = int(feed.header.timestamp or time.time())
    for e in feed.entity:
        tu = e.trip_update if e.HasField("trip_update") else None
        if not tu:
            continue
        for stu in tu.stop_time_update:
            if stu.stop_id not in stops:
                continue
            arr = getattr(stu.arrival, "time", None) or None
            dep = getattr(stu.departure, "time", None) or None
            delay = getattr(stu.arrival, "delay", None) or getattr(stu.departure, "delay", None) or None
            rows.append({
                "pull_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "server_ts": server_ts,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,  # may be blank for LIRR; keep for schema parity
                "stop_id": stu.stop_id,
                "arrival_time": arr,
                "departure_time": dep,
                "delay_sec": delay,
                "schedule_relationship": tu.trip.schedule_relationship,
                "entity_id": e.id,
            })
    return rows

def write_csv(path: str, rows: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------- _uid helpers ----------
def get_event_epoch(row: dict):
    # Prefer explicit epoch times
    for k in ("arrival_time","departure_time","event_epoch"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                pass
    # Fallback to server_ts or pull_utc (epoch or ISO)
    for k in ("server_ts","pull_utc"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)  # already epoch?
            except Exception:
                try:
                    return pd.to_datetime(v, utc=True).view("int64")/1e9
                except Exception:
                    return np.nan
    return np.nan

def choose_tripish(row: dict) -> str:
    for k in ("trip_id","entity_id","vehicle_ref","train_id","block_id","route_id"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: dict) -> str:
    stop_id = str(row.get("stop_id","") or "")
    tripish = choose_tripish(row)
    evt = get_event_epoch(row)
    base = f"{SUBSYSTEM_TAG}|{stop_id}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

# ---------- rclone-based master update ----------
def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
    """
    Download Drive master (if exists) -> concat -> ensure _uid -> dedup by _uid
    (prefer latest server_ts) -> upload back to Drive.
    """
    if not GDRIVE_REMOTE or not GDRIVE_REMOTE.strip():
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr)
        sys.exit(2)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")

        pulled = subprocess.run(
            ["rclone","copyto", remote_master, local_master],
            capture_output=True, text=True
        )

        if pulled.returncode == 0 and os.path.exists(local_master):
            df_master = pd.read_csv(local_master)
        else:
            df_master = pd.DataFrame(columns=local_new_df.columns)

        # Backfill _uid in existing master if missing
        if len(df_master) and "_uid" not in df_master.columns:
            df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Ensure new pull has _uid
        if "_uid" not in local_new_df.columns:
            local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Align columns & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master   = df_master.reindex(columns=all_cols)
        local_new_df= local_new_df.reindex(columns=all_cols)

        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # Prefer most recent observation per _uid if server_ts present
        if "server_ts" in combined.columns:
            combined = combined.sort_values("server_ts", na_position="last")
        combined = combined.drop_duplicates(subset=["_uid"], keep="last")

        combined.to_csv(local_master, index=False)
        subprocess.check_call(["rclone","copyto", local_master, remote_master])
        print(f"✅ {SUBSYSTEM_TAG} master now {len(combined)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

# =========================
# Main
# =========================
def main():
    if not LIRR_STOP_IDS:
        print("LIRR_STOP_IDS is required (e.g., '237').", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    # 1) Fetch & parse
    try:
        feed = fetch(LIRR_FEED_URL)
        rows = rows_from_feed(feed, LIRR_STOP_IDS)
    except Exception as e:
        print("[warn] fetch/parse failed:", e, file=sys.stderr)
        rows = []

    # 2) Raw snapshot (local + Drive)
    outpath = os.path.join("data","lirr", out_name())
    write_csv(outpath, rows)
    subprocess.run(
        ["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"],
        check=False
    )

    # 3) DataFrame for this pull → drop immediate dup rows → add _uid → update master
    df_poll = pd.DataFrame(rows, columns=FIELDS)

    # de-dupe identical rows within this pull (optional but helpful)
    dedupe_key = [c for c in ["stop_id","trip_id","entity_id","arrival_time","departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    # merge with Drive master (rclone), de-dup by _uid, upload back
    update_master_with_uid_rclone(df_poll)

    print(f"[ok] LIRR poll complete; stop_ids={sorted(LIRR_STOP_IDS)}")

if __name__ == "__main__":
    main()
