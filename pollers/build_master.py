#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master.py — Rolling master builder for Penn Station interfaces

Key behavior for your current issue:
- Reads NJT realtime CSVs from NJT_RT_DIR and filters strictly to Penn by stop_id (default "105").
- Emits NJT interfaces even if NJT static is not available.
- Appends to MASTER_OUT while preserving your wide schema.

Env vars (same as your workflow):
  MTA_RT_DIR               default "data/realtime"
  NJT_RT_DIR               default "data/njt_rt"
  MASTER_OUT               default "data/curated/master_interface_dataset.csv"
  SERVICE_DATE             optional override (YYYY-MM-DD) else inferred UTC today
  TOL_MIN                  default "15"      # retained but not used in this minimal NJT-first build
  TRANSFER_WINDOW_MIN      default "20"
  MISSED_THRESHOLD_MIN     default "2"
  APPEND_TO_MASTER         default "1"
  PENN_STOP_IDS            default "105"     # comma-separated strings for NJT stop_ids to keep
  SUBWAY_PENN_NODE         default "Subway_Penn"  # target node label to connect NJT to
  NJT_FROM_NODE            default "NJT_Rail"     # source node label

Assumptions:
- NJT realtime CSV has columns: trip_id, stop_id, route_id, rt_arrival_utc, rt_departure_utc, source_minute_utc
- MTA realtime CSVs (if present) are left as-is (no changes required to enable NJT interfaces)
"""

from __future__ import annotations
import os
import sys
import glob
import json
import math
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
MTA_RT_GLOB        = "*.csv"          # your subway poller already writes under data/realtime
NJT_RT_GLOB        = "njt_rt_*.csv"   # NJT poller daily file(s)

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

def _to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _weekday_name(ts: pd.Timestamp | float | str) -> str | pd.NA:
    try:
        if pd.isna(ts):
            return pd.NA
        t = pd.to_datetime(ts, utc=True)
        return t.tz_convert("UTC").day_name()
    except Exception:
        return pd.NA

def _hour_of_day(ts: pd.Timestamp | float | str) -> int | pd.NA:
    try:
        if pd.isna(ts):
            return pd.NA
        t = pd.to_datetime_
