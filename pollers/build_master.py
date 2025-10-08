#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py — Rolling master builder for Penn Station interfaces
with NJT trip_id-agnostic schedule join.

Environment variables (with defaults):
  MTA_RT_DIR             default "data/realtime"
  NJT_RT_DIR             default "data/njt_rt"
  MASTER_OUT             default "data/curated/master_interface_dataset.csv"
  SERVICE_DATE           optional YYYY-MM-DD (for labeling/logs)
  APPEND_TO_MASTER       default "1"
  PENN_STOP_IDS          default "105"  # NJT stop_ids for NY Penn (comma-separated)
  SUBWAY_PENN_NODE       default "Subway_Penn"
  NJT_FROM_NODE          default "NJT_Rail"
  NJT_TOLERANCE_MIN      default "45"   # nearest schedule tolerance (minutes)

Notes:
- NJT RT vs static trip_id mismatch: we match by (route_id, stop_id, nearest time in local clock).
- Static GTFS times may exceed 24:00:00; handled as seconds since local midnight (>86400 allowed).
"""

from __future__ import annotations

import os
import io
import glob
import pathlib
import zipfile
import datetime as dt
from typing import List, Set

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# -------------------- config / env --------------------
MTA_RT_DIR         = os.getenv("MTA_RT_DIR", "data/realtime")
NJT_RT_DIR         = os.getenv("NJT_RT_DIR", "data/njt_rt")
MASTER_OUT         = os.getenv("MASTER_OUT", "data/curated/master_interface_dataset.csv")
SERVICE_DATE       = os.getenv("SERVICE_DATE", "").strip() or dt.date.today().isoformat()
APPEND_TO_MASTER   = os.getenv("APPEND_TO_MASTER", "1").strip() == "1"

# Penn filter + node labels
PENN_STOP_IDS_RAW  = os.getenv("PENN_STOP_IDS", "105")
PENN_STOP_IDS: Set[str] = {s.strip() for s in PENN_STOP_IDS_RAW.split(",") if s.strip()}
SUBWAY_PENN_NODE   = os.getenv("SUBWAY_PENN_NODE", "Subway_Penn")
NJT_FROM_NODE      = os.getenv("NJT_FROM_NODE", "NJT_Rail")

# Schedule join tolerance (minutes)
NJT_TOLERANCE_MIN  = int(os.getenv("NJT_TOLERANCE_MIN", "45"))

# File patterns
MTA_RT_GLOB        = "*.csv"          # subway poller files (unchanged here)
NJT_RT_GLOB        = "njt_rt_*.csv"   # NJT poller files

NY_TZ = ZoneInfo("America/New_York")

# -------------------- master schema --------------------
MASTER_COLUMNS = [
    "Interface_ID",
    "From_Node",
    "To_Node",
    "Link_Type",
    "Scheduled_Arrival",
    "RT_Arrival",
    "Arrival_Delay_Min",
    "Arrival_Delay_Min_Filled",
    "Scheduled_Departure",
    "RT_Departure",
    "Departure_Delay_Min",
    "Departure_Delay_Min_Filled",
    "Transfer_Gap_Min",
    "Missed_Transfer_Flag",
    "Avg_Flow_Volume",
    "Peak_Flow_Volume",
    "Daily_Ridership_Share_%",
    "Delay_Frequency_%",
    "Avg_Delay_Min",
    "Delay_Chain_Min",
    "Chain_Reaction_Factor",
    "Alt_Path_Available",
    "Criticality_Score",
    "Ped_Count",
    "Stress_Index",
    "Time_Period",
    "Day_of_Week",
    "Hour_of_Day",
    "Peak_Flag",
    "External_Pressure",
    "Incident_History",
    "Used_Scheduled_Fallback",
]

# -------------------- small helpers --------------------
def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    # Reorder
    return df[cols].copy()

def _to_dt_utc(series_like) -> pd.Series:
    return pd.to_datetime(series_like, utc=True, errors="coerce")

def _weekday_name(ts) -> str | pd.NA:
    try:
        if pd.isna(ts):
            return pd.NA
        t = pd.to_datetime(ts, utc=True)
        return t.day_name()
    except Exception:
        return pd.NA

def _hour_of_day(ts) -> int | pd.NA:
    try:
        if pd.isna(ts):
            return pd.NA
        t = pd.to_datetime(ts, utc=True)
        return int(t.hour)
    except Exception:
        return pd.NA

# -------------------- static schedule loading --------------------
def _latest_njt_static_zip() -> str | None:
    """
    Find the newest NJT static zip in gtfs_static/.
    Tries, in order: njt_rail_static*.zip, njt_rail*.zip, njt_*.zip, *.zip
    """
    root = pathlib.Path("gtfs_static")
    cands = (sorted(root.glob("njt_rail_static*.zip"))
             or sorted(root.glob("njt_rail*.zip"))
             or sorted(root.glob("njt_*.zip"))
             or sorted(root.glob("*.zip")))
    return str(cands[-1]) if cands else None

def _gtfs_time_to_seconds(hms: str | float | int | None) -> float:
    """Convert GTFS HH:MM:SS (may exceed 24:00:00) -> seconds since local midnight (>86400 allowed)."""
    if hms is None or (isinstance(hms, float) and np.isnan(hms)):
        return np.nan
    try:
        hh, mm, ss = str(hms).split(":")
        hh, mm, ss = int(hh), int(mm), int(ss)
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return np.nan

def load_njt_static_schedule() -> pd.DataFrame:
    """
    Read NJT static trips.txt + stop_times.txt into schedule rows with seconds.
    Returns columns: route_id, stop_id, trip_id, sched_arrival_sec, sched_departure_sec
    """
    zpath = _latest_njt_static_zip()
    if not zpath:
        print("[njt][sched] no NJT static zip found in gtfs_static/")
        return pd.DataFrame(columns=["route_id","stop_id","trip_id","sched_arrival_sec","sched_departure_sec"])

    with zipfile.ZipFile(zpath, "r") as z:
        try:
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")))
            st    = pd.read_csv(io.BytesIO(z.read("stop_times.txt")))
        except KeyError as e:
            print(f"[njt][sched] missing file in zip: {e}")
            return pd.DataFrame(columns=["route_id","stop_id","trip_id","sched_arrival_sec","sched_departure_sec"])

    # Normalize dtypes for join keys
    if "route_id" in trips:
        trips["route_id"] = trips["route_id"].astype(str).str.strip()
    if "trip_id" in trips:
        trips["trip_id"] = trips["trip_id"].astype(str).str.strip()

    for c in ("trip_id","stop_id"):
        if c in st.columns:
            st[c] = st[c].astype(str).str.strip()

    # Attach route_id onto stop_times
    st = st.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")

    # Convert GTFS times to seconds since local midnight
    st["sched_arrival_sec"]   = st["arrival_time"].map(_gtfs_time_to_seconds)
    st["sched_departure_sec"] = st["departure_time"].map(_gtfs_time_to_seconds)

    sched = st[["route_id","stop_id","trip_id","sched_arrival_sec","sched_departure_sec"]].dropna(subset=["route_id","stop_id"])
    # Ensure strings
    sched["route_id"] = sched["route_id"].astype(str).str.strip()
    sched["stop_id"]  = sched["stop_id"].astype(str).str.strip()
    return sched

# -------------------- RT ⇄ schedule time alignment --------------------
def _utc_to_local_seconds(ts_utc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    UTC -> (local_date, seconds since local midnight) using tz-aware arithmetic.
    Avoids deprecated .view; robust across DST.
    """
    s = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    s_local = s.dt.tz_convert(NY_TZ)
    local_date = s_local.dt.date
    # seconds since local midnight:
    secs = (s_local - s_local.dt.normalize()).dt.total_seconds()
    return local_date, secs

def _sec_to_local_utc(local_date: pd.Series, sec_col: pd.Series) -> pd.Series:
    """(local_date, seconds since local midnight) -> UTC timestamp."""
    base = pd.to_datetime(local_date.astype(str), errors="coerce").dt.tz_localize(NY_TZ)
    delta = pd.to_timedelta(pd.to_numeric(sec_col, errors="coerce").fillna(0), unit="s")
    local_dt = base + delta
    return local_dt.dt.tz_convert("UTC")

def join_njt_rt_to_schedule(rt: pd.DataFrame, sched: pd.DataFrame, tolerance_min: int = 45) -> pd.DataFrame:
    """
    Match NJT realtime rows to static schedule using (route_id, stop_id, nearest time).
    Inputs:
      rt    columns: route_id, stop_id, rt_arrival_utc, rt_departure_utc (UTC)
      sched columns: route_id, stop_id, trip_id, sched_arrival_sec, sched_departure_sec

    Returns rt enriched with:
      Scheduled_Arrival (UTC), Scheduled_Departure (UTC), Arrival_Delay_Min, Departure_Delay_Min, trip_id (matched)
    """
    if rt.empty or sched.empty:
        return rt.assign(Scheduled_Arrival=pd.NaT, Scheduled_Departure=pd.NaT,
                         Arrival_Delay_Min=pd.NA, Departure_Delay_Min=pd.NA, trip_id=pd.NA)

    # Normalize keys
    for c in ("route_id","stop_id"):
        rt[c] = rt[c].astype(str).str.strip()
        sched[c] = sched[c].astype(str).str.strip()

    # Convert RT times and compute local date / seconds (arrival preferred)
    rt["rt_arrival_utc"]   = pd.to_datetime(rt["rt_arrival_utc"], utc=True, errors="coerce")
    rt["rt_departure_utc"] = pd.to_datetime(rt["rt_departure_utc"], utc=True, errors="coerce")

    rt["local_date_arr"], rt["rt_arr_sec"] = _utc_to_local_seconds(rt["rt_arrival_utc"])  # may be NaN
    _, rt["rt_dep_sec"] = _utc_to_local_seconds(rt["rt_departure_utc"])                     # may be NaN

    # Numeric tolerance in seconds (keys are numeric seconds)
    tol_sec = float(tolerance_min * 60)

    # Arrival-based nearest match within each (route_id, stop_id)
    leftA = rt[["route_id","stop_id","rt_arrival_utc","rt_departure_utc","local_date_arr","rt_arr_sec","rt_dep_sec"]].copy()
    rightA = sched[["route_id","stop_id","trip_id","sched_arrival_sec"]].rename(columns={"sched_arrival_sec":"sched_key_sec"})

    mergedA = pd.merge_asof(
        leftA.sort_values("rt_arr_sec").astype({"rt_arr_sec":"float64"}),
        rightA.sort_values("sched_key_sec").astype({"sched_key_sec":"float64"}),
        by=["route_id","stop_id"],
        left_on="rt_arr_sec", right_on="sched_key_sec",
        direction="nearest", tolerance=tol_sec
    )

    # For rows not matched on arrival, try departure-based match
    unmatched = mergedA[mergedA["trip_id"].isna()].copy()
    if not unmatched.empty:
        rightD = sched[["route_id","stop_id","trip_id","sched_departure_sec"]].rename(columns={"sched_departure_sec":"sched_key_sec"})
        mergedD = pd.merge_asof(
            unmatched.sort_values("rt_dep_sec").astype({"rt_dep_sec":"float64"}),
            rightD.sort_values("sched_key_sec").astype({"sched_key_sec":"float64"}),
            by=["route_id","stop_id"],
            left_on="rt_dep_sec", right_on="sched_key_sec",
            direction="nearest", tolerance=tol_sec
        )
        # Fill back into mergedA
        idx = unmatched.index
        if "trip_id" in mergedD.columns:
            mergedA.loc[idx, "trip_id"] = mergedD["trip_id"].values
        if "sched_key_sec" in mergedD.columns:
            mergedA.loc[idx, "sched_key_sec"] = mergedD["sched_key_sec"].values

    # Reconstruct scheduled UTC timestamps from (local_date, seconds)
    mergedA["Scheduled_Arrival"] = _sec_to_local_utc(mergedA["local_date_arr"], mergedA["sched_key_sec"])
    # (Optional) Scheduled_Departure second pass could be added; keep NaT for now
    mergedA["Scheduled_Departure"] = pd.NaT

    # Delays (minutes): RT - Scheduled
    mergedA["Arrival_Delay_Min"] = (mergedA["rt_arrival_utc"] - mergedA["Scheduled_Arrival"]) / pd.Timedelta(minutes=1)
    mergedA["Departure_Delay_Min"] = pd.NA

    return mergedA[[
        "route_id","stop_id","rt_arrival_utc","rt_departure_utc",
        "Scheduled_Arrival","Scheduled_Departure",
        "Arrival_Delay_Min","Departure_Delay_Min","trip_id"
    ]]

# -------------------- IO helpers --------------------
def read_glob_csv(folder: str, pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] skip {p}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -------------------- NJT realtime → interfaces --------------------
def build_njt_interfaces() -> pd.DataFrame:
    df = read_glob_csv(NJT_RT_DIR, NJT_RT_GLOB)
    if df.empty:
        print("[njt] no NJT realtime files found")
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # Expected columns; fill missing defensively
    expected = ["trip_id", "stop_id", "route_id", "rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize key types
    df["stop_id"]  = df["stop_id"].astype(str).str.strip()
    df["trip_id"]  = df["trip_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()

    # Strict Penn filter
    if PENN_STOP_IDS:
        before = len(df)
        df = df[df["stop_id"].isin(PENN_STOP_IDS)].copy()
        print(f"[njt] filtered NJT rows to Penn: {len(df)}/{before}")

    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # Choose best realtime timestamp (prefer arrival, fallback to departure)
    df["RT_Arrival"]   = df["rt_arrival_utc"]
    df["RT_Departure"] = df["rt_departure_utc"]
    best_ts = df["RT_Arrival"].fillna(df["RT_Departure"])
    best_dt = _to_dt_utc(best_ts)

    stamp_min = best_dt.dt.strftime("%Y%m%d_%H%M")
    interface_id = (
        "NJT_Rail_" + SUBWAY_PENN_NODE + "_" +
        stamp_min.fillna("NA") + "_" +
        df["trip_id"].fillna("") + "_" +
        df["stop_id"].fillna("")
    )

    out = pd.DataFrame({
        "Interface_ID": interface_id,
        "From_Node": NJT_FROM_NODE,
        "To_Node": SUBWAY_PENN_NODE,
        "Link_Type": "Rail-Subway",
        "Scheduled_Arrival": pd.NaT,
        "RT_Arrival": _to_dt_utc(df["RT_Arrival"]),
        "Arrival_Delay_Min": pd.NA,
        "Arrival_Delay_Min_Filled": pd.NA,
        "Scheduled_Departure": pd.NaT,
        "RT_Departure": _to_dt_utc(df["RT_Departure"]),
        "Departure_Delay_Min": pd.NA,
        "Departure_Delay_Min_Filled": pd.NA,
        "Transfer_Gap_Min": pd.NA,
        "Missed_Transfer_Flag": pd.NA,
        "Avg_Flow_Volume": pd.NA,
        "Peak_Flow_Volume": pd.NA,
        "Daily_Ridership_Share_%": pd.NA,
        "Delay_Frequency_%": pd.NA,
        "Avg_Delay_Min": pd.NA,
        "Delay_Chain_Min": pd.NA,
        "Chain_Reaction_Factor": pd.NA,
        "Alt_Path_Available": pd.NA,
        "Criticality_Score": pd.NA,
        "Ped_Count": pd.NA,
        "Stress_Index": pd.NA,
        "Time_Period": SERVICE_DATE,
        "Day_of_Week": best_dt.apply(_weekday_name),
        "Hour_of_Day": best_dt.apply(_hour_of_day),
        "Peak_Flag": pd.NA,
        "External_Pressure": pd.NA,
        "Incident_History": pd.NA,
        "Used_Scheduled_Fallback": "0",

        # Keep keys for schedule enrichment
        "trip_id": df["trip_id"],
        "stop_id": df["stop_id"],
        "route_id": df["route_id"],
    })

    # Keep join keys temporarily
    out = out.reindex(columns=MASTER_COLUMNS + ["trip_id","stop_id","route_id"])

    print(f"[njt] built {len(out)} NJT interface rows")

    # Enrich with schedule if static is present
    sched = load_njt_static_schedule()
    if not sched.empty:
        print("[njt][sched] attempting trip_id-agnostic schedule match …")
        # Prepare minimal RT frame for join
        rt_join = out[["route_id","stop_id","RT_Arrival","RT_Departure"]].rename(columns={
            "RT_Arrival":"rt_arrival_utc",
            "RT_Departure":"rt_departure_utc",
        }).copy()

        matched = join_njt_rt_to_schedule(rt_join, sched, tolerance_min=NJT_TOLERANCE_MIN)
        print(f"[njt][sched] matched {matched['Scheduled_Arrival'].notna().sum()} rows within ±{NJT_TOLERANCE_MIN} min")

        matched = matched.rename(columns={
            "rt_arrival_utc": "RT_Arrival",
            "rt_departure_utc": "RT_Departure",
        })

        # Merge matched schedule columns back onto interfaces
        out = out.merge(
            matched,
            on=["route_id","stop_id","RT_Arrival","RT_Departure"],
            how="left",
            suffixes=("","")
        )

        # Prefer filled schedule values
        def _coalesce(dst, a, b):
            if a in out.columns and b in out.columns:
                out[dst] = out[a].combine_first(out[b])
                with contextlib.suppress(Exception):
                    out.drop(columns=[a,b], inplace=True)
        import contextlib
        _coalesce("Scheduled_Arrival", "Scheduled_Arrival_x", "Scheduled_Arrival_y")
        _coalesce("Scheduled_Departure", "Scheduled_Departure_x", "Scheduled_Departure_y")
        _coalesce("Arrival_Delay_Min", "Arrival_Delay_Min_x", "Arrival_Delay_Min_y")
        _coalesce("Departure_Delay_Min", "Departure_Delay_Min_x", "Departure_Delay_Min_y")

    else:
        print("[njt][sched] no static schedule available — leaving Scheduled_* blank")

    # Remove join keys from final master
    out.drop(columns=["trip_id","stop_id","route_id"], inplace=True, errors="ignore")
    out = _ensure_columns(out, MASTER_COLUMNS)
    return out

# -------------------- (placeholder) MTA → interfaces --------------------
def build_mta_interfaces_passthrough() -> pd.DataFrame:
    """
    Placeholder to keep subway logic unchanged in this script.
    If you already have an MTA builder elsewhere, replace this stub
    or leave it empty to avoid breaking the pipeline.
    """
    return pd.DataFrame(columns=MASTER_COLUMNS)

# -------------------- write / append --------------------
def append_master(df: pd.DataFrame) -> None:
    out_path = pathlib.Path(MASTER_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("[master] nothing to write")
        return
    df = _ensure_columns(df, MASTER_COLUMNS)
    write_header = not out_path.exists()
    mode = "a" if (APPEND_TO_MASTER and out_path.exists()) else "w"
    df.to_csv(out_path, index=False, mode=mode, header=write_header or (mode == "w"))
    print(f"[master] wrote {len(df)} rows -> {out_path} (mode={mode})")

# -------------------- main --------------------
def main() -> None:
    print("[t0] starting build_master")
    print(f"[cfg] SERVICE_DATE={SERVICE_DATE}  MASTER_OUT={MASTER_OUT}")
    print(f"[cfg] MTA_RT_DIR={MTA_RT_DIR}  NJT_RT_DIR={NJT_RT_DIR}")
    print(f"[cfg] PENN_STOP_IDS={sorted(PENN_STOP_IDS) if PENN_STOP_IDS else 'ALL'}  NJT_TOLERANCE_MIN={NJT_TOLERANCE_MIN}")
    print(f"[cfg] SUBWAY_PENN_NODE={SUBWAY_PENN_NODE}  NJT_FROM_NODE={NJT_FROM_NODE}")

    njt_ifaces = build_njt_interfaces()
    mta_ifaces = build_mta_interfaces_passthrough()

    master_chunk = pd.concat([mta_ifaces, njt_ifaces], ignore_index=True)
    if master_chunk.empty:
        print("[master] no interfaces produced this run")
    append_master(master_chunk)
    print("[t1] build_master finished")

if __name__ == "__main__":
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except Exception:
        pass
    main()
