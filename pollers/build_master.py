#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py — Rolling master builder for Penn Station interfaces

- Reads NJT realtime CSVs from NJT_RT_DIR and filters strictly to Penn by stop_id (default "105").
- Emits NJT interfaces even if NJT static is not available.
- Appends to MASTER_OUT while preserving the wide schema you’ve been using.

Env vars (same as your workflow):
  MTA_RT_DIR               default "data/realtime"
  NJT_RT_DIR               default "data/njt_rt"
  MASTER_OUT               default "data/curated/master_interface_dataset.csv"
  SERVICE_DATE             optional override (YYYY-MM-DD) else inferred UTC today
  APPEND_TO_MASTER         default "1"
  PENN_STOP_IDS            default "105"     # comma-separated NJT stop_ids to keep
  SUBWAY_PENN_NODE         default "Subway_Penn"
  NJT_FROM_NODE            default "NJT_Rail"
"""

from __future__ import annotations
import os
import glob
import pathlib
import datetime as dt
from typing import List, Set

import pandas as pd

# ---------- config / env ----------
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

# file patterns
MTA_RT_GLOB        = "*.csv"          # subway poller files (untouched here)
NJT_RT_GLOB        = "njt_rt_*.csv"   # NJT poller daily files

# ---------- schema helpers ----------
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
        t = pd.to_datetime(ts, utc=True)  # <-- correct function (no trailing underscore)
        return t.tz_convert("UTC").day_name()
    except Exception:
        return pd.NA

def _hour_of_day(ts) -> int | pd.NA:
    try:
        if pd.isna(ts):
            return pd.NA
        t = pd.to_datetime(ts, utc=True)  # <-- correct function (no trailing underscore)
        return int(t.tz_convert("UTC").hour)
    except Exception:
        return pd.NA

# ---------- IO ----------
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

# ---------- NJT realtime → interfaces ----------
def build_njt_interfaces() -> pd.DataFrame:
    """
    Create NJT interfaces at Penn directly from realtime (no static needed).
    Interface_ID: NJT_Rail_<To_Node>_<YYYYMMDD_HHMM>_<trip_id>_<stop_id>
    """
    df = read_glob_csv(NJT_RT_DIR, NJT_RT_GLOB)
    if df.empty:
        print("[njt] no NJT realtime files found")
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # Expected columns; fill missing defensively
    expected = ["trip_id", "stop_id", "route_id", "rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    df["stop_id"]  = df["stop_id"].astype(str)
    df["trip_id"]  = df["trip_id"].astype(str)
    df["route_id"] = df["route_id"].astype(str)

    # strict Penn filter
    if PENN_STOP_IDS:
        keep = {s.strip() for s in PENN_STOP_IDS}
        before = len(df)
        df = df[df["stop_id"].isin(keep)].copy()
        print(f"[njt] filtered NJT rows to Penn: {len(df)}/{before}")

    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # choose best real-time timestamp (prefer arrival, fallback to departure)
    df["RT_Arrival"]   = df["rt_arrival_utc"]
    df["RT_Departure"] = df["rt_departure_utc"]
    best_ts = df["RT_Arrival"].fillna(df["RT_Departure"])
    best_dt = _to_dt_utc(best_ts)

    # derive identifiers
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
    })

    out = _ensure_columns(out, MASTER_COLUMNS)
    print(f"[njt] built {len(out)} NJT interface rows")
    return out

# ---------- (placeholder) MTA → interfaces ----------
def build_mta_interfaces_passthrough() -> pd.DataFrame:
    """No-op placeholder to keep your subway logic unchanged elsewhere."""
    return pd.DataFrame(columns=MASTER_COLUMNS)

# ---------- write / append ----------
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

# ---------- main ----------
def main() -> None:
    print("[t0] starting build_master")
    print(f"[cfg] SERVICE_DATE={SERVICE_DATE}  MASTER_OUT={MASTER_OUT}")
    print(f"[cfg] MTA_RT_DIR={MTA_RT_DIR}  NJT_RT_DIR={NJT_RT_DIR}")
    print(f"[cfg] PENN_STOP_IDS={sorted(PENN_STOP_IDS) if PENN_STOP_IDS else 'ALL'}")
    print(f"[cfg] SUBWAY_PENN_NODE={SUBWAY_PENN_NODE}  NJT_FROM_NODE={NJT_FROM_NODE}")

    njt_ifaces = build_njt_interfaces()
    mta_ifaces = build_mta_interfaces_passthrough()

    master_chunk = pd.concat([mta_ifaces, njt_ifaces], ignore_index=True)
    if master_chunk.empty:
        print("[master] no interfaces produced this run")
    append_master(master_chunk)
    print("[t1] build_master finished")

if __name__ == "__main__":
    # (Optional) avoid downcasting surprises in new pandas
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except Exception:
        pass
    main()
