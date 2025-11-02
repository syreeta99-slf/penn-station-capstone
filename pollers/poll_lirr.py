#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, json, time, tempfile, subprocess, datetime, requests
from typing import Set, List, Dict, Any, Optional
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
                "route_id": tu.trip.route_id,  # may be blank; keep for schema parity
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
def add_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    # prefer departure for planning/connection logic; fall back to arrival
    if "departure_time" in df.columns and "arrival_time" in df.columns:
        df["event_time"] = df["departure_time"].where(df["departure_time"].notna(), df["arrival_time"])
    else:
        df["event_time"] = pd.NA

    df["event_time"] = df["event_time"].astype("Int64")
    df["event_ts_utc"] = pd.to_datetime(df["event_time"], unit="s", utc=True, errors="coerce")
    df["event_kind"] = np.where(df.get("departure_time", pd.Series(dtype="Int64")).notna(), "DEP",
                        np.where(df.get("arrival_time", pd.Series(dtype="Int64")).notna(), "ARR", pd.NA))
    return df

# ---------- stitching arrivals + departures ----------
def stitch_arrival_departure(df_raw: pd.DataFrame,
                             key_cols=("trip_id","stop_id"),
                             arrival_col="arrival_time",
                             departure_col="departure_time",
                             tolerance_minutes=90) -> pd.DataFrame:
    """
    Combine arrival-only and departure-only rows into one event row per trip/stop.
    Keeps singletons when no matching counterpart exists.
    """
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()

    # normalize types
    for c in [arrival_col, departure_col, "server_ts", "delay_sec"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    for c in list(key_cols) + ["route_id","schedule_relationship","entity_id","pull_utc","stop_id","trip_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    arr = df[df[arrival_col].notna()].copy()
    dep = df[df[departure_col].notna()].copy()

    arr["_arr_ts"] = pd.to_datetime(arr[arrival_col], unit="s", utc=True, errors="coerce")
    dep["_dep_ts"] = pd.to_datetime(dep[departure_col], unit="s", utc=True, errors="coerce")

    if arr.empty and dep.empty:
        return df

    # sort for merge_asof
    sort_keys = list(key_cols)
    if not arr.empty:
        arr = arr.sort_values(sort_keys + ["_arr_ts"])
    if not dep.empty:
        dep = dep.sort_values(sort_keys + ["_dep_ts"])

    stitched = pd.merge_asof(
        arr, dep,
        left_on="_arr_ts", right_on="_dep_ts",
        by=sort_keys,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        suffixes=("", "_dep")
    ) if (not arr.empty and not dep.empty) else arr.copy()

    # valid pairs: both times exist and dep >= arr
    if not stitched.empty and (departure_col + "_dep") in stitched.columns:
        valid_pair = (
            stitched[departure_col + "_dep"].notna() &
            stitched[arrival_col].notna() &
            (stitched[departure_col + "_dep"] >= stitched[arrival_col])
        )
        paired = stitched.loc[valid_pair].copy()
    else:
        paired = stitched.iloc[0:0].copy()

    def coalesce(a, b):
        return a.where(a.notna(), b)

    out_cols = df.columns.tolist()
    for tmpc in ("_arr_ts","_dep_ts"):
        if tmpc in out_cols:
            out_cols.remove(tmpc)
    rows_out = []

    # unified paired rows
    if len(paired):
        tmp = {}
        for c in out_cols:
            if c == arrival_col:
                tmp[c] = paired[arrival_col]
            elif c == departure_col and (departure_col + "_dep") in paired.columns:
                tmp[c] = paired[departure_col + "_dep"]
            elif c + "_dep" in paired.columns:
                tmp[c] = coalesce(paired[c], paired[c + "_dep"])
            else:
                tmp[c] = paired[c] if c in paired.columns else pd.NA
        rows_out.append(pd.DataFrame(tmp))

    # unpaired arrivals
    if not arr.empty:
        matched_arr_ids = set(paired["entity_id"].dropna()) if ("entity_id" in paired.columns and len(paired)) else set()
        ua = arr[~arr["entity_id"].isin(matched_arr_ids)] if "entity_id" in arr.columns else arr.iloc[0:0]
        if len(ua):
            rows_out.append(ua[out_cols])

    # unpaired departures
    if not dep.empty and "entity_id_dep" in (paired.columns if len(paired) else []):
        matched_dep_ids = set(paired["entity_id_dep"].dropna())
        ud = dep[~dep["entity_id"].isin(matched_dep_ids)] if "entity_id" in dep.columns else dep.iloc[0:0]
        if len(ud):
            rows_out.append(ud[out_cols])
    elif not dep.empty and len(paired) == 0:
        # no pairs at all -> keep departures as-is
        rows_out.append(dep[out_cols])

    out = pd.concat(rows_out, ignore_index=True) if rows_out else df.iloc[0:0].reindex(columns=out_cols)

    # readable timestamps
    out[arrival_col.replace("_time","_ts_utc")]   = pd.to_datetime(out[arrival_col],   unit="s", utc=True, errors="coerce")
    out[departure_col.replace("_time","_ts_utc")] = pd.to_datetime(out[departure_col], unit="s", utc=True, errors="coerce")

    # sanity: allow singletons; if both present, require dep >= arr
    ok = (
        out[departure_col].isna() |
        out[arrival_col].isna() |
        (out[departure_col] >= out[arrival_col])
    )
    out = out[ok].drop_duplicates(subset=list(key_cols) + [arrival_col, departure_col], keep="last")
    return out.reset_index(drop=True)

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
                return float(v)
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

# ---------- rclone-based master update (coalescing) ----------
def _last_valid(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else (s.iloc[-1] if len(s) else pd.NA)

def _as_Int64(series: pd.Series) -> pd.Series:
    try:
        return series.astype("Int64")
    except Exception:
        return series

def _add_readable_ts(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("arrival_time","departure_time"):
        if col in df.columns:
            out_col = col.replace("_time","_ts_utc")
            df[out_col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
    return df

def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
    """
    Download Drive master (if exists) -> concat -> coalesce by _uid (latest non-null per column) -> upload.
    """
    if not GDRIVE_REMOTE or not GDRIVE_REMOTE.strip():
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr)
        sys.exit(2)

    # normalize types
    for c in ("arrival_time","departure_time","server_ts","delay_sec"):
        if c in local_new_df.columns:
            local_new_df[c] = _as_Int64(local_new_df[c])
    for c in ("trip_id","route_id","stop_id","schedule_relationship","entity_id","pull_utc"):
        if c in local_new_df.columns:
            local_new_df[c] = local_new_df[c].astype("string")

    # Ensure _uid present
    if "_uid" not in local_new_df.columns:
        local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")

        pulled = subprocess.run(
            ["rclone","copyto", remote_master, local_master],
            capture_output=True, text=True
        )

        if pulled.returncode == 0 and os.path.exists(local_master):
            df_master = pd.read_csv(local_master, dtype={"trip_id":"string","route_id":"string",
                                                         "stop_id":"string","schedule_relationship":"string",
                                                         "entity_id":"string","pull_utc":"string","_uid":"string"},
                                    low_memory=False)
        else:
            df_master = pd.DataFrame(columns=local_new_df.columns)

        # Backfill _uid if missing historically
        if len(df_master) and "_uid" not in df_master.columns:
            df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Align columns & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master    = df_master.reindex(columns=all_cols)
        local_new_df = local_new_df.reindex(columns=all_cols)
        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # Sort and coalesce by _uid
        if "server_ts" in combined.columns:
            combined["server_ts"] = _as_Int64(combined["server_ts"])
        for c in ("arrival_time","departure_time","delay_sec"):
            if c in combined.columns:
                combined[c] = _as_Int64(combined[c])

        sort_cols = [c for c in ["_uid","server_ts"] if c in combined.columns]
        combined = combined.sort_values(sort_cols)

        agg_spec = {c: _last_valid for c in combined.columns if c != "_uid"}
        collapsed = (combined.groupby("_uid", as_index=False).agg(agg_spec))
        collapsed = _add_readable_ts(collapsed)
        collapsed = add_event_fields(collapsed)


def _print_time_mix(df, label="LIRR"):
    a = df["arrival_time"].notna() if "arrival_time" in df else pd.Series(dtype=bool)
    d = df["departure_time"].notna() if "departure_time" in df else pd.Series(dtype=bool)
    both = (a & d).mean() * 100
    only_a = (a & ~d).mean() * 100
    only_d = (~a & d).mean() * 100
    none = (~a & ~d).mean() * 100
    print(f"[{label}] both={both:.2f}% | only_arr={only_a:.2f}% | only_dep={only_d:.2f}% | none={none:.2f}%")

_print_time_mix(collapsed, "LIRR master")

        collapsed.to_csv(local_master, index=False)
        subprocess.check_call(["rclone","copyto", local_master, remote_master])
        print(f"✅ {SUBSYSTEM_TAG} master now {len(collapsed)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

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
    outpath = os.path.join("data","lirr","raw", out_name())
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    write_csv(outpath, rows)
    subprocess.run(
        ["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"],
        check=False
    )

    # 3) DataFrame for this pull → within-pull dedupe
    df_poll = pd.DataFrame(rows, columns=FIELDS)
    dedupe_key = [c for c in ["stop_id","trip_id","entity_id","arrival_time","departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    # 4) Stitch arrivals + departures into one event row per trip/stop
    df_events = stitch_arrival_departure(df_poll,
                                         key_cols=("trip_id","stop_id"),
                                         arrival_col="arrival_time",
                                         departure_col="departure_time",
                                         tolerance_minutes=90)

    # 5) Compute _uid AFTER stitching
    if "_uid" not in df_events.columns:
        df_events["_uid"] = df_events.apply(lambda r: make_uid(r.to_dict()), axis=1)

    # 6) Update master on Drive with coalescing
    update_master_with_uid_rclone(df_events)

    print(f"[ok] LIRR poll complete; stop_ids={sorted(LIRR_STOP_IDS)}")

if __name__ == "__main__":
    main()
