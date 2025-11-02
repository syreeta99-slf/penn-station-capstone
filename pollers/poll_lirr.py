#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, time, tempfile, subprocess, datetime, requests
from typing import Set, List, Dict
from pathlib import Path
from hashlib import md5

import pandas as pd
import numpy as np
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

# ========= ENV / CONFIG =========
LIRR_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("LIRR_STOP_IDS","").split(",") if s.strip()}  # e.g. "237"
MTA_API_KEY    = os.getenv("MTA_API_KEY")
LIRR_FEED_URL  = os.getenv("LIRR_FEED_URL", "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/lirr%2Fgtfs-lirr")

GDRIVE_REMOTE  = os.getenv("GDRIVE_REMOTE_NAME")                         # required
GDRIVE_DIR     = os.getenv("GDRIVE_DIR_LIRR", "penn-station/lirr")
MASTER_NAME    = os.getenv("LIRR_MASTER_NAME", "lirr_penn_master.csv")

SUBSYSTEM_TAG  = "lirr"

RAW_FIELDS = [
    "pull_utc","server_ts","trip_id","route_id","stop_id",
    "arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"
]

# ========= UTILS =========
def _ts_stamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _raw_name() -> str:
    return f"lirr_rt_{_ts_stamp()}.csv"

def _fetch_feed(url: str) -> gtfs_realtime_pb2.FeedMessage:
    headers = {"x-api-key": MTA_API_KEY} if MTA_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def _rows_from_feed(feed: gtfs_realtime_pb2.FeedMessage, stops: Set[str]) -> List[dict]:
    rows = []
    server_ts = int(feed.header.timestamp or time.time())
    now_iso = datetime.datetime.utcnow().isoformat(timespec="seconds")
    for e in feed.entity:
        if not e.HasField("trip_update"): 
            continue
        tu = e.trip_update
        for stu in tu.stop_time_update:
            if stops and stu.stop_id not in stops:
                continue
            arr = getattr(stu.arrival, "time", None) or None
            dep = getattr(stu.departure, "time", None) or None
            delay = getattr(stu.arrival, "delay", None) or getattr(stu.departure, "delay", None) or None
            rows.append({
                "pull_utc": now_iso,
                "server_ts": server_ts,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,
                "stop_id": stu.stop_id,
                "arrival_time": arr,
                "departure_time": dep,
                "delay_sec": delay,
                "schedule_relationship": tu.trip.schedule_relationship,
                "entity_id": e.id,
            })
    return rows

def _write_csv(path: str, rows: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ========= STITCHING (per-group asof) =========
def stitch_arrival_departure(
    df_raw: pd.DataFrame,
    key_cols=("trip_id","stop_id"),
    arrival_col="arrival_time",
    departure_col="departure_time",
    tolerance_minutes=90
) -> pd.DataFrame:
    """Pair arrival-only and departure-only rows per (trip_id, stop_id). Keeps singletons."""
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()

    # normalize dtypes
    for c in [arrival_col, departure_col, "server_ts", "delay_sec"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    for c in list(key_cols) + ["route_id","schedule_relationship","entity_id","pull_utc","stop_id","trip_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    arr = df[df[arrival_col].notna()].copy()
    dep = df[df[departure_col].notna()].copy()

    if not arr.empty:
        arr["_arr_ts"] = pd.to_datetime(arr[arrival_col], unit="s", utc=True, errors="coerce")
        arr = arr.dropna(subset=["_arr_ts"])
    if not dep.empty:
        dep["_dep_ts"] = pd.to_datetime(dep[departure_col], unit="s", utc=True, errors="coerce")
        dep = dep.dropna(subset=["_dep_ts"])

    # fast exits
    if arr.empty and dep.empty:
        out = df.iloc[0:0].reindex(columns=df.columns)
        out[arrival_col.replace("_time","_ts_utc")]   = pd.NaT
        out[departure_col.replace("_time","_ts_utc")] = pd.NaT
        return out
    if arr.empty:
        out = dep.copy()
        out[arrival_col.replace("_time","_ts_utc")]   = pd.to_datetime(out[arrival_col],   unit="s", utc=True, errors="coerce")
        out[departure_col.replace("_time","_ts_utc")] = pd.to_datetime(out[departure_col], unit="s", utc=True, errors="coerce")
        return out.reset_index(drop=True)
    if dep.empty:
        out = arr.copy()
        out[arrival_col.replace("_time","_ts_utc")]   = pd.to_datetime(out[arrival_col],   unit="s", utc=True, errors="coerce")
        out[departure_col.replace("_time","_ts_utc")] = pd.to_datetime(out[departure_col], unit="s", utc=True, errors="coerce")
        return out.reset_index(drop=True)

    # ensure keys exist & non-null per group
    for k in key_cols:
        if k not in arr.columns: arr[k] = pd.NA
        if k not in dep.columns: dep[k] = pd.NA
        arr[k] = arr[k].astype("string")
        dep[k] = dep[k].astype("string")
    arr = arr.dropna(subset=list(key_cols))
    dep = dep.dropna(subset=list(key_cols))

    out_cols = df.columns.tolist()
    for tmp in ("_arr_ts","_dep_ts"):
        if tmp in out_cols: out_cols.remove(tmp)

    pieces = []
    matched_arr_idx, matched_dep_idx = set(), set()

    # pair only common groups
    arr_groups = arr.groupby(list(key_cols), dropna=False, sort=False)
    dep_groups = dep.groupby(list(key_cols), dropna=False, sort=False)
    common = set(arr_groups.groups.keys()) & set(dep_groups.groups.keys())

    for key in common:
        ag = arr_groups.get_group(key).sort_values("_arr_ts").copy()
        dg = dep_groups.get_group(key).sort_values("_dep_ts").copy()
        ag["_left_idx"] = ag.index
        dg["_right_idx"] = dg.index

        merged = pd.merge_asof(
            ag, dg,
            left_on="_arr_ts", right_on="_dep_ts",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
            suffixes=("", "_dep")
        )

        dep_col_merged = f"{departure_col}_dep"
        valid = (
            merged.get(dep_col_merged).notna() &
            merged[arrival_col].notna() &
            (merged[dep_col_merged] >= merged[arrival_col])
        )

        if valid.any():
            m = merged.loc[valid].copy()
            matched_arr_idx.update(m["_left_idx"].tolist())
            if "_right_idx" in m.columns:
                matched_dep_idx.update(m["_right_idx"].dropna().astype(int).tolist())

            def coalesce(a, b): return a.where(a.notna(), b)

            row = {}
            for c in out_cols:
                if c == arrival_col:
                    row[c] = m[arrival_col]
                elif c == departure_col and dep_col_merged in m.columns:
                    row[c] = m[dep_col_merged]
                elif c + "_dep" in m.columns:
                    row[c] = coalesce(m[c], m[c + "_dep"])
                else:
                    row[c] = m[c] if c in m.columns else pd.NA
            pieces.append(pd.DataFrame(row))

    # unpaired
    if len(arr):
        ua = arr.loc[[i for i in arr.index if i not in matched_arr_idx], out_cols]
        if len(ua): pieces.append(ua)
    if len(dep):
        ud = dep.loc[[i for i in dep.index if i not in matched_dep_idx], out_cols]
        if len(ud): pieces.append(ud)

    out = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].reindex(columns=out_cols)

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

# ========= UID / EVENT FIELDS =========
def _get_event_epoch(row: Dict) -> float:
    for k in ("arrival_time","departure_time","event_epoch"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try: return float(v)
            except: pass
    for k in ("server_ts","pull_utc"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try: return float(v)
            except:
                try: return pd.to_datetime(v, utc=True).view("int64")/1e9
                except: return np.nan
    return np.nan

def _choose_tripish(row: Dict) -> str:
    for k in ("trip_id","entity_id","vehicle_ref","train_id","block_id","route_id"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: Dict) -> str:
    stop_id = str(row.get("stop_id","") or "")
    tripish = _choose_tripish(row)
    evt = _get_event_epoch(row)
    base = f"{SUBSYSTEM_TAG}|{stop_id}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

def add_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "departure_time" in df.columns and "arrival_time" in df.columns:
        df["event_time"] = df["departure_time"].where(df["departure_time"].notna(), df["arrival_time"])
    else:
        df["event_time"] = pd.NA
    df["event_time"] = df["event_time"].astype("Int64")
    df["event_ts_utc"] = pd.to_datetime(df["event_time"], unit="s", utc=True, errors="coerce")
    df["event_kind"] = np.where(df.get("departure_time", pd.Series(dtype="Int64")).notna(), "DEP",
                        np.where(df.get("arrival_time", pd.Series(dtype="Int64")).notna(), "ARR", pd.NA))
    return df

def _add_readable_ts(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("arrival_time","departure_time"):
        if col in df.columns:
            out_col = col.replace("_time","_ts_utc")
            df[out_col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
    return df

def _as_Int64(s: pd.Series) -> pd.Series:
    try: return s.astype("Int64")
    except: return s

def _last_valid(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else (s.iloc[-1] if len(s) else pd.NA)

def _print_time_mix(df: pd.DataFrame, label="LIRR"):
    a = df["arrival_time"].notna() if "arrival_time" in df.columns else pd.Series(dtype=bool)
    d = df["departure_time"].notna() if "departure_time" in df.columns else pd.Series(dtype=bool)
    both = (a & d).mean() * 100 if len(df) else 0.0
    only_a = (a & ~d).mean() * 100 if len(df) else 0.0
    only_d = (~a & d).mean() * 100 if len(df) else 0.0
    none = (~a & ~d).mean() * 100 if len(df) else 0.0
    print(f"[{label}] both={both:.2f}% | only_arr={only_a:.2f}% | only_dep={only_d:.2f}% | none={none:.2f}%")

# ========= MASTER UPDATE (Drive via rclone) =========
def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
    if not GDRIVE_REMOTE or not GDRIVE_REMOTE.strip():
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr)
        sys.exit(2)

    # dtypes
    for c in ("arrival_time","departure_time","server_ts","delay_sec"):
        if c in local_new_df.columns:
            local_new_df[c] = _as_Int64(local_new_df[c])
    for c in ("trip_id","route_id","stop_id","schedule_relationship","entity_id","pull_utc"):
        if c in local_new_df.columns:
            local_new_df[c] = local_new_df[c].astype("string")

    if "_uid" not in local_new_df.columns:
        local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        pulled = subprocess.run(["rclone","copyto", remote_master, local_master],
                                capture_output=True, text=True)

        if pulled.returncode == 0 and os.path.exists(local_master):
            df_master = pd.read_csv(local_master, dtype={"trip_id":"string","route_id":"string",
                                                         "stop_id":"string","schedule_relationship":"string",
                                                         "entity_id":"string","pull_utc":"string","_uid":"string"},
                                    low_memory=False)
        else:
            df_master = pd.DataFrame(columns=local_new_df.columns)

        if len(df_master) and "_uid" not in df_master.columns:
            df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # align & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master    = df_master.reindex(columns=all_cols)
        local_new_df = local_new_df.reindex(columns=all_cols)
        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # sort & coalesce
        if "server_ts" in combined.columns:
            combined["server_ts"] = _as_Int64(combined["server_ts"])
        for c in ("arrival_time","departure_time","delay_sec"):
            if c in combined.columns:
                combined[c] = _as_Int64(combined[c])

        combined = combined.sort_values([c for c in ["_uid","server_ts"] if c in combined.columns])
        agg_spec = {c: _last_valid for c in combined.columns if c != "_uid"}
        collapsed = combined.groupby("_uid", as_index=False).agg(agg_spec)

        # enrich & QA
        collapsed = _add_readable_ts(collapsed)
        collapsed = add_event_fields(collapsed)
        _print_time_mix(collapsed, "LIRR master")

        # write back
        collapsed.to_csv(local_master, index=False)
        subprocess.check_call(["rclone","copyto", local_master, remote_master])
        print(f"✅ {SUBSYSTEM_TAG} master now {len(collapsed)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

# ========= MAIN =========
def main():
    if not LIRR_STOP_IDS:
        print("LIRR_STOP_IDS is required (e.g., '237').", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    # 1) Fetch & parse
    try:
        feed = _fetch_feed(LIRR_FEED_URL)
        rows = _rows_from_feed(feed, LIRR_STOP_IDS)
    except Exception as e:
        print("[warn] fetch/parse failed:", e, file=sys.stderr)
        rows = []

    # 2) Raw snapshot (local + Drive)
    raw_local = os.path.join("data","lirr","raw", _raw_name())
    _write_csv(raw_local, rows)
    subprocess.run(["rclone","copyto", raw_local, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(raw_local)}"],
                   check=False)

    # 3) Frame & within-pull dedupe
    df_poll = pd.DataFrame(rows, columns=RAW_FIELDS)
    dedupe_key = [c for c in ["stop_id","trip_id","entity_id","arrival_time","departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    # 4) Stitch arrivals + departures
    df_events = stitch_arrival_departure(df_poll, key_cols=("trip_id","stop_id"),
                                         arrival_col="arrival_time", departure_col="departure_time",
                                         tolerance_minutes=90)

    # 5) UID after stitching
    if "_uid" not in df_events.columns:
        df_events["_uid"] = df_events.apply(lambda r: make_uid(r.to_dict()), axis=1)

    # 6) Update master on Drive
    update_master_with_uid_rclone(df_events)

    print(f"[ok] LIRR poll complete; stop_ids={sorted(LIRR_STOP_IDS)}")

if __name__ == "__main__":
    main()
