from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Master — unified pipeline for Penn Station interfaces

What this version does
- NJT: fills Scheduled_Arrival & Scheduled_Departure via trip_id-agnostic merge
  using (route_id, stop_id, nearest local scheduled time). Robust to RT/static
  trip_id mismatches. Computes Arrival/Departure delays.
- MTA/Subway: preserves any existing Scheduled_Arrival / Scheduled_Departure
  fields in your RT CSVs (case-insensitive detection + common name variants).
  Robust UTC parsing even if strings are naive (assumes America/New_York).
- Optional MTA static fallback (off by default): if env MTA_STATIC_ZIP is set
  and Scheduled_* are missing/blank, compute schedules via nearest-time match
  like NJT.

Other correctness:
- Safe merge_back (suffix + coalesce) and numeric `merge_asof` tolerance.
- TZ-safe conversions (no `.view` warnings).
- Auto-detects the latest NJT static ZIP from `gtfs_static/`.

Env vars (defaults):
- MTA_RT_DIR (data/realtime) – subway RT export dir
- NJT_RT_DIR (data/njt_rt) – NJT RT export dir
- MASTER_OUT (data/curated/master_interface_dataset.csv)
- SERVICE_DATE (today) – label only
- APPEND_TO_MASTER (1) – append/write
- PENN_STOP_IDS (105) – comma-separated NJT stop_ids for NY Penn
- SUBWAY_PENN_NODE (Subway_Penn)
- NJT_FROM_NODE (NJT_Rail)
- MTA_FROM_NODE (MTA_Subway)
- NJT_TOLERANCE_MIN (45)
- MTA_STATIC_ZIP (unset) – set path like gtfs_static/mta_subway_static.zip to enable MTA static fallback
"""

import os
import io
import glob
import pathlib
import zipfile
import contextlib
import datetime as dt
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# -------------------- config / env --------------------
MTA_RT_DIR         = os.getenv("MTA_RT_DIR", "data/realtime")
NJT_RT_DIR         = os.getenv("NJT_RT_DIR", "data/njt_rt")
MASTER_OUT         = os.getenv("MASTER_OUT", "data/curated/master_interface_dataset.csv")
SERVICE_DATE       = os.getenv("SERVICE_DATE", "").strip() or dt.date.today().isoformat()
APPEND_TO_MASTER   = os.getenv("APPEND_TO_MASTER", "1").strip() == "1"

PENN_STOP_IDS_RAW  = os.getenv("PENN_STOP_IDS", "105")
PENN_STOP_IDS: Set[str] = {s.strip() for s in PENN_STOP_IDS_RAW.split(",") if s.strip()}
SUBWAY_PENN_NODE   = os.getenv("SUBWAY_PENN_NODE", "Subway_Penn")
NJT_FROM_NODE      = os.getenv("NJT_FROM_NODE", "NJT_Rail")
MTA_FROM_NODE      = os.getenv("MTA_FROM_NODE", "MTA_Subway")

NJT_TOLERANCE_MIN  = int(os.getenv("NJT_TOLERANCE_MIN", "45"))
MTA_STATIC_ZIP     = os.getenv("MTA_STATIC_ZIP", "").strip()  # optional

# File patterns
MTA_RT_GLOB        = "*.csv"          # subway poller files (we preserve fields if present)
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

# -------------------- utilities --------------------
def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
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

# -------------------- robust parsers --------------------
def parse_to_utc_any(s: pd.Series) -> pd.Series:
    """
    Robustly parse timestamps into UTC:
    - if already tz-aware: convert to UTC
    - if naive string/datetime: assume America/New_York then to UTC
    - if numeric: treat as seconds since epoch UTC
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    # First try normal parse
    dt1 = pd.to_datetime(s, errors="coerce", utc=False)
    # For tz-aware entries, convert to UTC; for naive, localize NY then convert
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    mask_valid = dt1.notna()
    if mask_valid.any():
        dt_sub = dt1[mask_valid]
        # Determine which are tz-aware (have tzinfo)
        tz_aware = getattr(dt_sub.dt, "tz", None) is not None
        if tz_aware:
            out.loc[mask_valid] = dt_sub.dt.tz_convert("UTC")
        else:
            # localize as NY then convert
            out.loc[mask_valid] = dt_sub.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    # Handle numeric epoch seconds that failed above
    numeric = pd.to_numeric(s, errors="coerce")
    mask_num = numeric.notna() & out.isna()
    if mask_num.any():
        out.loc[mask_num] = pd.to_datetime(numeric[mask_num], unit="s", utc=True, errors="coerce")
    return out

# -------------------- NJT static schedule loading --------------------
def _latest_njt_static_zip() -> str | None:
    root = pathlib.Path("gtfs_static")
    cands = (sorted(root.glob("njt_rail_static*.zip"))
             or sorted(root.glob("njt_rail*.zip"))
             or sorted(root.glob("njt_*.zip"))
             or sorted(root.glob("*.zip")))
    return str(cands[-1]) if cands else None

def _gtfs_time_to_seconds(hms: str | float | int | None) -> float:
    if hms is None or (isinstance(hms, float) and np.isnan(hms)):
        return np.nan
    try:
        hh, mm, ss = str(hms).split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return np.nan

def load_njt_static_schedule() -> pd.DataFrame:
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
    for c in ("route_id","trip_id"):
        if c in trips.columns:
            trips[c] = trips[c].astype(str).str.strip()
    for c in ("trip_id","stop_id"):
        if c in st.columns:
            st[c] = st[c].astype(str).str.strip()
    st = st.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")
    st["sched_arrival_sec"]   = st["arrival_time"].map(_gtfs_time_to_seconds)
    st["sched_departure_sec"] = st["departure_time"].map(_gtfs_time_to_seconds)
    sched = st[["route_id","stop_id","trip_id","sched_arrival_sec","sched_departure_sec"]].dropna(subset=["route_id","stop_id"])
    sched["route_id"] = sched["route_id"].astype(str).str.strip()
    sched["stop_id"]  = sched["stop_id"].astype(str).str.strip()
    return sched

# -------------------- NJT RT ⇄ schedule alignment --------------------
def _utc_to_local_seconds(ts_utc: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    s_local = s.dt.tz_convert(NY_TZ)
    local_date = s_local.dt.date
    secs = (s_local - s_local.dt.normalize()).dt.total_seconds()
    return local_date, secs

def _sec_to_local_utc(local_date: pd.Series, sec_col: pd.Series) -> pd.Series:
    """(local_date, seconds since local midnight) -> UTC timestamp. NaN seconds => NaT."""
    base = pd.to_datetime(local_date.astype(str), errors="coerce").dt.tz_localize(NY_TZ)
    secs = pd.to_numeric(sec_col, errors="coerce")
    delta = pd.to_timedelta(secs, unit="s")
    local_dt = base + delta
    return local_dt.dt.tz_convert("UTC")

def join_njt_rt_to_schedule(rt: pd.DataFrame, sched: pd.DataFrame, tolerance_min: int = 45) -> pd.DataFrame:
    """
    Match NJT realtime rows to static schedule using (route_id, stop_id, nearest time).
    Inputs:
      rt    columns: route_id, stop_id, rt_arrival_utc, rt_departure_utc (UTC)
      sched columns: route_id, stop_id, trip_id, sched_arrival_sec, sched_departure_sec

    Returns rt enriched with:
      Scheduled_Arrival (UTC), Scheduled_Departure (UTC),
      Arrival_Delay_Min, Departure_Delay_Min, trip_id (matched)
    """
    if rt.empty or sched.empty:
        return rt.assign(Scheduled_Arrival=pd.NaT, Scheduled_Departure=pd.NaT,
                         Arrival_Delay_Min=pd.NA, Departure_Delay_Min=pd.NA, trip_id=pd.NA)

    # normalize keys
    for c in ("route_id","stop_id"):
        rt[c] = rt[c].astype(str).str.strip()
        sched[c] = sched[c].astype(str).str.strip()

    # to UTC and local seconds
    rt["rt_arrival_utc"]   = pd.to_datetime(rt["rt_arrival_utc"], utc=True, errors="coerce")
    rt["rt_departure_utc"] = pd.to_datetime(rt["rt_departure_utc"], utc=True, errors="coerce")

    rt["local_date_arr"], rt["rt_arr_sec"] = _utc_to_local_seconds(rt["rt_arrival_utc"])
    rt["local_date_dep"], rt["rt_dep_sec"] = _utc_to_local_seconds(rt["rt_departure_utc"])

    tol_sec = float(tolerance_min * 60)

    # Arrival-based nearest match
    leftA  = rt[["route_id","stop_id","rt_arrival_utc","rt_departure_utc","local_date_arr","rt_arr_sec","rt_dep_sec"]].copy()
    rightA = sched[["route_id","stop_id","trip_id","sched_arrival_sec"]].rename(columns={"sched_arrival_sec":"sched_key_sec"})
    mergedA = pd.merge_asof(
        leftA.sort_values("rt_arr_sec").astype({"rt_arr_sec":"float64"}),
        rightA.sort_values("sched_key_sec").astype({"sched_key_sec":"float64"}),
        by=["route_id","stop_id"],
        left_on="rt_arr_sec", right_on="sched_key_sec",
        direction="nearest", tolerance=tol_sec
    )

    # Fallback: departure-based for those still unmatched
    unmatched = mergedA[mergedA["trip_id"].isna()].copy()
    if not unmatched.empty:
        rightD_fb = sched[["route_id","stop_id","trip_id","sched_departure_sec"]].rename(columns={"sched_departure_sec":"sched_key_sec"})
        mergedD_fb = pd.merge_asof(
            unmatched.sort_values("rt_dep_sec").astype({"rt_dep_sec":"float64"}),
            rightD_fb.sort_values("sched_key_sec").astype({"sched_key_sec":"float64"}),
            by=["route_id","stop_id"],
            left_on="rt_dep_sec", right_on="sched_key_sec",
            direction="nearest", tolerance=tol_sec
        )
        idx = unmatched.index
        if "trip_id" in mergedD_fb.columns:
            mergedA.loc[idx, "trip_id"] = mergedD_fb["trip_id"].values
        if "sched_key_sec" in mergedD_fb.columns:
            mergedA.loc[idx, "sched_key_sec"] = mergedD_fb["sched_key_sec"].values

    # Build Scheduled_Arrival UTC
    mergedA["Scheduled_Arrival"] = _sec_to_local_utc(mergedA["local_date_arr"], mergedA["sched_key_sec"])

    # Departure-based nearest match for Scheduled_Departure
    dep_left  = rt[["route_id","stop_id","rt_departure_utc","local_date_dep","rt_dep_sec"]].copy()
    dep_right = sched[["route_id","stop_id","trip_id","sched_departure_sec"]].rename(columns={"sched_departure_sec":"sched_dep_key_sec"})
    dep_join = pd.merge_asof(
        dep_left.sort_values("rt_dep_sec").astype({"rt_dep_sec":"float64"}),
        dep_right.sort_values("sched_dep_key_sec").astype({"sched_dep_key_sec":"float64"}),
        by=["route_id","stop_id"],
        left_on="rt_dep_sec", right_on="sched_dep_key_sec",
        direction="nearest", tolerance=tol_sec
    )
    dep_join["Scheduled_Departure"] = _sec_to_local_utc(dep_join["local_date_dep"], dep_join["sched_dep_key_sec"])
    dep_join = dep_join.rename(columns={"trip_id":"trip_id_dep"})

    mergedA = mergedA.merge(
        dep_join[["route_id","stop_id","rt_departure_utc","Scheduled_Departure","trip_id_dep"]],
        on=["route_id","stop_id","rt_departure_utc"],
        how="left"
    )

    # Prefer arrival trip_id; fill from departure pass if missing
    if "trip_id_dep" in mergedA.columns:
        mergedA["trip_id"] = mergedA["trip_id"].fillna(mergedA["trip_id_dep"])
        mergedA.drop(columns=["trip_id_dep"], inplace=True, errors=True)

    # Delays
    mergedA["Arrival_Delay_Min"]   = (mergedA["rt_arrival_utc"]   - mergedA["Scheduled_Arrival"])   / pd.Timedelta(minutes=1)
    mergedA["Departure_Delay_Min"] = (mergedA["rt_departure_utc"] - mergedA["Scheduled_Departure"]) / pd.Timedelta(minutes=1)

    return mergedA[[
        "route_id","stop_id","rt_arrival_utc","rt_departure_utc",
        "Scheduled_Arrival","Scheduled_Departure",
        "Arrival_Delay_Min","Departure_Delay_Min","trip_id"
    ]]

# -------------------- NJT realtime → interfaces --------------------
def build_njt_interfaces() -> pd.DataFrame:
    df = read_glob_csv(NJT_RT_DIR, NJT_RT_GLOB)
    if df.empty:
        print("[njt] no NJT realtime files found")
        return pd.DataFrame(columns=MASTER_COLUMNS)

    expected = ["trip_id", "stop_id", "route_id", "rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    df["stop_id"]  = df["stop_id"].astype(str).str.strip()
    df["trip_id"]  = df["trip_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()

    if PENN_STOP_IDS:
        before = len(df)
        df = df[df["stop_id"].isin(PENN_STOP_IDS)].copy()
        print(f"[njt] filtered NJT rows to Penn: {len(df)}/{before}")

    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    df["RT_Arrival"]   = df["rt_arrival_utc"]
    df["RT_Departure"] = df["rt_departure_utc"]
    best_ts = _to_dt_utc(df["RT_Arrival"].fillna(df["RT_Departure"]))
    stamp_min = best_ts.dt.strftime("%Y%m%d_%H%M")
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
        "Day_of_Week": best_ts.apply(_weekday_name),
        "Hour_of_Day": best_ts.apply(_hour_of_day),
        "Peak_Flag": pd.NA,
        "External_Pressure": pd.NA,
        "Incident_History": pd.NA,
        "Used_Scheduled_Fallback": "0",
        "trip_id": df["trip_id"],
        "stop_id": df["stop_id"],
        "route_id": df["route_id"],
    })

    out = out.reindex(columns=MASTER_COLUMNS + ["trip_id","stop_id","route_id"])
    print(f"[njt] built {len(out)} NJT interface rows")

    sched = load_njt_static_schedule()
    if not sched.empty:
        print("[njt][sched] attempting trip_id-agnostic schedule match …")
        rt_join = out[["route_id","stop_id","RT_Arrival","RT_Departure"]].rename(columns={
            "RT_Arrival":"rt_arrival_utc",
            "RT_Departure":"rt_departure_utc",
        }).copy()
        matched = join_njt_rt_to_schedule(rt_join, sched, tolerance_min=NJT_TOLERANCE_MIN)
        print(f"[njt][sched] matched {matched['Scheduled_Arrival'].notna().sum()} rows within ±{NJT_TOLERANCE_MIN} min")

        matched = matched.rename(columns={"rt_arrival_utc": "RT_Arrival", "rt_departure_utc": "RT_Departure"})
        keep_cols = [
            "route_id", "stop_id", "RT_Arrival", "RT_Departure",
            "Scheduled_Arrival", "Scheduled_Departure",
            "Arrival_Delay_Min", "Departure_Delay_Min", "trip_id"
        ]
        matched_small = matched[[c for c in keep_cols if c in matched.columns]].copy()

        out = out.merge(
            matched_small,
            on=["route_id","stop_id","RT_Arrival","RT_Departure"],
            how="left",
            suffixes=("", "_m")
        )

        for c in ["Scheduled_Arrival", "Scheduled_Departure", "Arrival_Delay_Min", "Departure_Delay_Min", "trip_id"]:
            cm = f"{c}_m"
            if cm in out.columns:
                if c in out.columns:
                    out[c] = out[c].combine_first(out[cm])
                else:
                    out[c] = out[cm]
                out.drop(columns=[cm], inplace=True, errors="ignore")
    else:
        print("[njt][sched] no static schedule available — leaving Scheduled_* blank")

    out.drop(columns=["trip_id","stop_id","route_id"], inplace=True, errors="ignore")
    out = _ensure_columns(out, MASTER_COLUMNS)
    return out

# -------------------- Optional MTA static schedule (fallback like NJT) --------------------
def maybe_load_mta_static_schedule() -> pd.DataFrame:
    """Load MTA static schedule if MTA_STATIC_ZIP is provided. Else return empty."""
    if not MTA_STATIC_ZIP:
        return pd.DataFrame()
    zpath = pathlib.Path(MTA_STATIC_ZIP)
    if not zpath.exists():
        print(f"[mta][sched] static zip not found: {MTA_STATIC_ZIP}")
        return pd.DataFrame()
    with zipfile.ZipFile(zpath, "r") as z:
        try:
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")))
            st    = pd.read_csv(io.BytesIO(z.read("stop_times.txt")))
        except KeyError as e:
            print(f"[mta][sched] missing file in zip: {e}")
            return pd.DataFrame()
    for c in ("route_id","trip_id"):
        if c in trips.columns:
            trips[c] = trips[c].astype(str).str.strip()
    for c in ("trip_id","stop_id"):
        if c in st.columns:
            st[c] = st[c].astype(str).str.strip()
    st = st.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")
    st["sched_arrival_sec"]   = st["arrival_time"].map(_gtfs_time_to_seconds)
    st["sched_departure_sec"] = st["departure_time"].map(_gtfs_time_to_seconds)
    sched = st[["route_id","stop_id","trip_id","sched_arrival_sec","sched_departure_sec"]].dropna(subset=["stop_id"])
    sched["stop_id"]  = sched["stop_id"].astype(str).str.strip()
    sched["route_id"] = sched["route_id"].astype(str).str.strip()
    return sched

def join_generic_nearest(rt_arrival_utc: pd.Series, rt_departure_utc: pd.Series,
                         stop_id: pd.Series, route_id: pd.Series,
                         sched: pd.DataFrame, tolerance_min: int = 45) -> Tuple[pd.Series, pd.Series]:
    """
    Generic nearest-time join to compute (Scheduled_Arrival, Scheduled_Departure) from a GTFS-like schedule.
    Assumes stop_id/route_id in both sides (route_id optional).
    """
    if sched.empty:
        return pd.Series(pd.NaT, index=rt_arrival_utc.index), pd.Series(pd.NaT, index=rt_departure_utc.index)

    # Build a small RT frame
    rt = pd.DataFrame({
        "route_id": route_id.astype(str).fillna(""),
        "stop_id":  stop_id.astype(str).fillna(""),
        "rt_arrival_utc":   _to_dt_utc(rt_arrival_utc),
        "rt_departure_utc": _to_dt_utc(rt_departure_utc),
    })

    # Local seconds
    rt["local_date_arr"], rt["rt_arr_sec"] = _utc_to_local_seconds(rt["rt_arrival_utc"])
    rt["local_date_dep"], rt["rt_dep_sec"] = _utc_to_local_seconds(rt["rt_departure_utc"])

    tol_sec = float(tolerance_min * 60)

    # Arrival
    leftA  = rt[["route_id","stop_id","rt_arrival_utc","local_date_arr","rt_arr_sec"]].copy()
    rightA = sched[["route_id","stop_id","sched_arrival_sec"]].rename(columns={"sched_arrival_sec":"sched_key_sec"})
    mergedA = pd.merge_asof(
        leftA.sort_values("rt_arr_sec").astype({"rt_arr_sec":"float64"}),
        rightA.sort_values("sched_key_sec").astype({"sched_key_sec":"float64"}),
        by=["route_id","stop_id"],
        left_on="rt_arr_sec", right_on="sched_key_sec",
        direction="nearest", tolerance=tol_sec
    )
    sched_arrival = _sec_to_local_utc(mergedA["local_date_arr"], mergedA["sched_key_sec"])

    # Departure
    leftD  = rt[["route_id","stop_id","rt_departure_utc","local_date_dep","rt_dep_sec"]].copy()
    rightD = sched[["route_id","stop_id","sched_departure_sec"]].rename(columns={"sched_departure_sec":"sched_dep_key_sec"})
    mergedD = pd.merge_asof(
        leftD.sort_values("rt_dep_sec").astype({"rt_dep_sec":"float64"}),
        rightD.sort_values("sched_dep_key_sec").astype({"sched_dep_key_sec":"float64"}),
        by=["route_id","stop_id"],
        left_on="rt_dep_sec", right_on="sched_dep_key_sec",
        direction="nearest", tolerance=tol_sec
    )
    sched_departure = _sec_to_local_utc(mergedD["local_date_dep"], mergedD["sched_dep_key_sec"])
    return sched_arrival, sched_departure

# -------------------- MTA/Subway realtime → interfaces --------------------
def build_mta_interfaces() -> pd.DataFrame:
    df = read_glob_csv(MTA_RT_DIR, MTA_RT_GLOB)
    if df.empty:
        print("[mta] no MTA realtime files found")
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # Flexible column resolver (case-insensitive)
    def pick(colnames: list[str]) -> pd.Series:
        lower = {c.lower(): c for c in df.columns}
        for name in colnames:
            if name.lower() in lower:
                return df[lower[name.lower()]]
        return pd.Series([pd.NA] * len(df))

    # Pull RT and Scheduled columns from common variants (case-insensitive)
    rt_arr_s   = pick(["RT_Arrival","rt_arrival_utc","rt_arrival","realtime_arrival_utc"])
    rt_dep_s   = pick(["RT_Departure","rt_departure_utc","rt_departure","realtime_departure_utc"])
    sch_arr_s  = pick(["Scheduled_Arrival","scheduled_arrival_utc","scheduled_arrival","sched_arrival","sch_arrival_utc"])
    sch_dep_s  = pick(["Scheduled_Departure","scheduled_departure_utc","scheduled_departure","sched_departure","sch_departure_utc"])
    stop_id_s  = pick(["stop_id","Stop_ID","stopId"])
    route_id_s = pick(["route_id","Route_ID","routeId"])

    # Robust parsing
    rt_arr   = parse_to_utc_any(rt_arr_s)
    rt_dep   = parse_to_utc_any(rt_dep_s)
    sched_arr = parse_to_utc_any(sch_arr_s)
    sched_dep = parse_to_utc_any(sch_dep_s)

    # If Scheduled_* absent/empty and MTA_STATIC_ZIP present, compute via static fallback
    need_fallback = MTA_STATIC_ZIP and (sched_arr.isna().all() and sched_dep.isna().all())
    if need_fallback:
        print("[mta][sched] Scheduled_* missing; attempting static fallback")
        mta_sched = maybe_load_mta_static_schedule()
        if not mta_sched.empty:
            sa, sd = join_generic_nearest(rt_arr, rt_dep, stop_id_s.astype(str), route_id_s.astype(str), mta_sched, tolerance_min=NJT_TOLERANCE_MIN)
            sched_arr = sa
            sched_dep = sd
        else:
            print("[mta][sched] static fallback unavailable; leaving Scheduled_* blank")

    # Compute delays when both RT and Scheduled exist
    arr_delay = (rt_arr - sched_arr) / pd.Timedelta(minutes=1)
    dep_delay = (rt_dep - sched_dep) / pd.Timedelta(minutes=1)

    best_ts = rt_arr.fillna(rt_dep)
    stamp_min = best_ts.dt.strftime("%Y%m%d_%H%M")
    interface_id = "MTA_Subway_" + SUBWAY_PENN_NODE + "_" + stamp_min.fillna("NA")

    out = pd.DataFrame({
        "Interface_ID": interface_id,
        "From_Node": df.get("From_Node", pd.Series([MTA_FROM_NODE]*len(df))),
        "To_Node": df.get("To_Node", pd.Series([SUBWAY_PENN_NODE]*len(df))),
        "Link_Type": df.get("Link_Type", pd.Series(["Subway-Subway"]*len(df))),
        "Scheduled_Arrival": sched_arr,
        "RT_Arrival": rt_arr,
        "Arrival_Delay_Min": arr_delay,
        "Arrival_Delay_Min_Filled": pd.NA,
        "Scheduled_Departure": sched_dep,
        "RT_Departure": rt_dep,
        "Departure_Delay_Min": dep_delay,
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
        "Day_of_Week": best_ts.apply(_weekday_name),
        "Hour_of_Day": best_ts.apply(_hour_of_day),
        "Peak_Flag": pd.NA,
        "External_Pressure": pd.NA,
        "Incident_History": pd.NA,
        "Used_Scheduled_Fallback": "0",
    })

    out = _ensure_columns(out, MASTER_COLUMNS)
    print(f"[mta] built {len(out)} MTA interface rows (preserved/parsed Scheduled_*; static fallback={'on' if need_fallback else 'off'})")
    return out

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
    print(f"[cfg] SUBWAY_PENN_NODE={SUBWAY_PENN_NODE}  NJT_FROM_NODE={NJT_FROM_NODE}  MTA_FROM_NODE={MTA_FROM_NODE}")
    if MTA_STATIC_ZIP:
        print(f"[cfg] MTA_STATIC_ZIP={MTA_STATIC_ZIP}")

    njt_ifaces = build_njt_interfaces()
    mta_ifaces = build_mta_interfaces()

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
