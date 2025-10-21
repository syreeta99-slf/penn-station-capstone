#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Poll MTA Bus Time (SIRI StopMonitoring) for selected stops and write:
#  - raw snapshot CSV (timestamped) to local + Drive
#  - merged master on Drive with stable _uid and de-dup (keep latest server_ts)

import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime
from pathlib import Path
from typing import List, Dict, Any, Set
from hashlib import md5

import requests
import pandas as pd
import numpy as np

# ---------- ENV ----------
API_KEY = os.getenv("MTA_API_KEY")  # required
BUS_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("BUS_STOP_IDS", "").split(",") if s.strip()}  # e.g. "MTA_401231,MTA_401361"
SIRI_URL = os.getenv("SIRI_STOPMON_URL", "https://bustime.mta.info/api/siri/stop-monitoring.json")

GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")  # required (rclone remote)
GDRIVE_DIR = os.getenv("GDRIVE_DIR_BUS", "penn-station/bus")
MASTER_NAME = os.getenv("BUS_MASTER_NAME", "bus_penn_master.csv")

SUBSYSTEM_TAG = "bus"  # used in _uid

# Output schema (aligned with your other modes)
FIELDS = [
    "pull_utc","server_ts","route_id","stop_id",
    "arrival_time","departure_time",
    "vehicle_ref","destination","visit_number","entity_id",
    "has_expected_arrival","has_expected_departure"
]

# ---------- helpers ----------
def utc_now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds")

def iso_to_epoch(iso_str: str | None) -> int | None:
    if not iso_str:
        return None
    try:
        # handles "...-04:00" or trailing "Z"
        dt = datetime.datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None

def fetch_stop(stop_id: str) -> Dict[str, Any]:
    r = requests.get(SIRI_URL, params={"key": API_KEY, "MonitoringRef": stop_id}, timeout=45)
    r.raise_for_status()
    return r.json()

def extract_rows(payload: Dict[str, Any], pull_iso: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sd = payload.get("Siri", {}).get("ServiceDelivery", {}) or {}
    server_ts = iso_to_epoch(sd.get("ResponseTimestamp")) or int(time.time())
    deliveries = sd.get("StopMonitoringDelivery", []) or []
    for d in deliveries:
        for v in (d.get("MonitoredStopVisit") or []):
            mvj = v.get("MonitoredVehicleJourney", {}) or {}
            mc  = mvj.get("MonitoredCall", {}) or {}
            rows.append({
                "pull_utc": pull_iso,
                "server_ts": server_ts,
                "route_id": mvj.get("LineRef"),
                "stop_id": mc.get("StopPointRef") or mvj.get("MonitoredCall", {}).get("StopPointRef"),
                "arrival_time": iso_to_epoch(mc.get("ExpectedArrivalTime")),
                "departure_time": iso_to_epoch(mc.get("ExpectedDepartureTime")),
                "vehicle_ref": mvj.get("VehicleRef"),
                "destination": mvj.get("DestinationName"),
                "visit_number": v.get("VisitNumber"),
                "entity_id": v.get("ItemIdentifier") or "",
                "has_expected_arrival": 1 if mc.get("ExpectedArrivalTime") else 0,
                "has_expected_departure": 1 if mc.get("ExpectedDepartureTime") else 0,
            })
    return rows

def write_csv(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

# ---------- _uid helpers ----------
def get_event_epoch(row: dict):
    # Prefer explicit event times when present
    for k in ("arrival_time","departure_time","event_epoch"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                pass
    # Fallback to server_ts or pull_utc
    for k in ("server_ts","pull_utc"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                try:
                    return pd.to_datetime(v, utc=True).view("int64")/1e9
                except Exception:
                    return np.nan
    return np.nan

def choose_tripish(row: dict) -> str:
    # For buses, VehicleRef is often the most stable; fall back to entity_id or route_id
    for k in ("vehicle_ref","entity_id","route_id"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: dict) -> str:
    stop_id = str(row.get("stop_id","") or "")
    tripish = choose_tripish(row)
    evt     = get_event_epoch(row)
    base    = f"{SUBSYSTEM_TAG}|{stop_id}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

# ---------- rclone-based master update (dedup by _uid) ----------
def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
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

        # Ensure this new pull has _uid
        if "_uid" not in local_new_df.columns:
            local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Align columns & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master    = df_master.reindex(columns=all_cols)
        local_new_df = local_new_df.reindex(columns=all_cols)

        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # Prefer most recent observation per _uid
        if "server_ts" in combined.columns:
            combined = combined.sort_values("server_ts", na_position="last")
        combined = combined.drop_duplicates(subset=["_uid"], keep="last")

        combined.to_csv(local_master, index=False)
        subprocess.check_call(["rclone","copyto", local_master, remote_master])
        print(f"✅ bus master now {len(combined)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

# ---------- main ----------
def main():
    if not API_KEY:
        print("MTA_API_KEY is required.", file=sys.stderr); sys.exit(2)
    if not BUS_STOP_IDS:
        print("BUS_STOP_IDS is required (comma list).", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    pull_iso = utc_now_iso()
    all_rows: List[Dict[str, Any]] = []

    for sid in BUS_STOP_IDS:
        try:
            payload = fetch_stop(sid)
            rows = extract_rows(payload, pull_iso)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[warn] stop {sid}: {e}", file=sys.stderr)

    # 1) raw snapshot (timestamped)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join("data", "bus", f"bus_rt_{ts}.csv")
    write_csv(outpath, all_rows)
    subprocess.run(
        ["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"],
        check=False
    )

    # 2) master merge with _uid + de-dup
    df_poll = pd.DataFrame(all_rows, columns=FIELDS)

    # drop identical rows within this pull (optional)
    dedupe_key = [c for c in ["stop_id","vehicle_ref","entity_id","arrival_time","departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    update_master_with_uid_rclone(df_poll)
    print(f"[ok] Bus poll complete; stops={len(BUS_STOP_IDS)} rows={len(all_rows)}")

if __name__ == "__main__":
    main()
