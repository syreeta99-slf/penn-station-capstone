#!/usr/bin/env python3
"""
pollers/build_master.py  —  REWRITTEN

Builds the event-level Master Interface Dataset by joining:
- Static GTFS (scheduled) from gtfs_static/subway_all_YYYYMMDD.zip
- Realtime subway polls from data/realtime/subway_rt_*.csv
- Realtime NJT polls from data/njt_rt/njt_rt_*.csv   (new)

Generates three interface types and APPENDS to:
- data/curated/master_interface_dataset.csv
Also writes a parquet alongside for convenience.

Interface types produced:
  1) NJT ↔ Subway (1/2/3)
  2) NJT ↔ Subway (A/C/E)
  3) Subway (1/2/3) ↔ Subway (A/C/E)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, io, json, zipfile, subprocess, sys, re, glob

import pandas as pd

# ----------------------- PATHS & ENV -----------------------
ROOT = Path(".")
DATA_DIR = ROOT / "data"
REALTIME_DIR = DATA_DIR / "realtime"
CURATED_DIR = DATA_DIR / "curated"
STATIC_DIR = ROOT / "gtfs_static"
CONFIG_PATH = ROOT / "config" / "penn_stops.json"

CURATED_DIR.mkdir(parents=True, exist_ok=True)

# Env knobs
RT_DIR = Path(os.getenv("RT_DIR", str(REALTIME_DIR)))  # existing MTA subway RT dir (data/realtime)
MTA_RT_GLOB = os.getenv("MTA_RT_GLOB", "subway_rt_*.csv")
NJT_RT_DIR = Path(os.getenv("NJT_RT_DIR", "data/njt_rt"))  # NEW: where NJT poller writes
NJT_RT_GLOB = os.getenv("NJT_RT_GLOB", "njt_rt_*.csv")

RT_MAX_FILES = int(os.getenv("RT_MAX_FILES", "60"))  # newest MTA RT CSVs to consider
SERVICE_DATE_OVERRIDE = os.getenv("SERVICE_DATE", "")  # YYYY-MM-DD or empty to infer
TRANSFER_WINDOW_MIN = int(os.getenv("TRANSFER_WINDOW_MIN", "20"))
MISSED_THRESHOLD_MIN = float(os.getenv("MISSED_THRESHOLD_MIN", "2"))
ALLOW_STATIC_REFRESH = os.getenv("ALLOW_STATIC_REFRESH", "1")

AM_PEAK = range(6, 10)   # 06–09 local
PM_PEAK = range(16, 20)  # 16–19 local

# ----------------------- ROUTE GROUPS ----------------------
ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["4", "5", "6", "7", "S"]),
    "Subway_ACE": set(list("ACE")),
}

# --- NJT: conservative known line codes (expand as needed) ---
NJT_ROUTES = {
    "NEC", "NJCL", "RVL", "ME", "MOBO", "M&E", "MNE", "PVL", "ACL", "RARV", "RBUS"
}

def _looks_like_njt(route_id: str) -> bool:
    rid = str(route_id or "")
    if rid in NJT_ROUTES:
        return True
    if rid.startswith("NJT_"):
        return True
    if rid.isalpha() and rid.upper() == rid and 2 <= len(rid) <= 5:
        return rid not in ROUTE_GROUPS["Subway_ACE"] and rid not in ROUTE_GROUPS["Subway_123"]
    return False

def route_to_node(route_id: str) -> str:
    if pd.isna(route_id) or route_id is None:
        return "Subway_Unknown"
    rid = str(route_id)

    for node, routes in ROUTE_GROUPS.items():
        if rid in routes:
            return node

    if _looks_like_njt(rid):
        return "NJT_Rail"
    return "Subway_Other"

# ------------------- STATIC TIME PARSING -------------------
def parse_gtfs_time_to_dt(hms: str, service_date: str) -> pd.Timestamp:
    try:
        h, m, s = map(int, str(hms).split(":"))
    except Exception:
        return pd.NaT
    extra_days, h = divmod(h, 24)
    base = pd.Timestamp(service_date).tz_localize("UTC")
    return base + pd.Timedelta(days=extra_days, hours=h, minutes=m, seconds=s)

# --------------------- INPUT HELPERS -----------------------
def _uniq_cols(base: list[str], extras: list[str]) -> list[str]:
    seen = set(base)
    out = list(base)
    for c in extras:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def latest_static_zip() -> Path:
    zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips and ALLOW_STATIC_REFRESH == "1":
        print("[build_master] No static ZIPs found. Attempting refresh...")
        subprocess.check_call([sys.executable, "pollers/mta_static_refresh.py"])
        zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips:
        raise FileNotFoundError("No subway_all_*.zip in gtfs_static/.")
    return zips[-1]

def read_txt_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)))

def build_stop_alias_map(stops_df: pd.DataFrame) -> dict:
    alias = {}
    for _, r in stops_df.iterrows():
        sid = str(r["stop_id"])
        parent = r.get("parent_station")
        parent = str(parent) if pd.notna(parent) else sid
        alias[sid] = parent
    return alias

def infer_service_date_from_rt(rt_df: pd.DataFrame) -> str:
    ts = pd.concat([rt_df.get("rt_arrival_utc"), rt_df.get("rt_departure_utc")], ignore_index=True).dropna()
    if ts.empty:
        return datetime.utcnow().date().isoformat()
    ts_local = ts.dt.tz_convert("America/New_York")
    return ts_local.dt.date.min().isoformat()

def latest_njt_static_zip() -> Path:
    zips = sorted(STATIC_DIR.glob("njt_rail_static*.zip")) or sorted(STATIC_DIR.glob("njt_rail_static.zip"))
    if not zips:
        raise FileNotFoundError("Place NJT GTFS static zip as gtfs_static/njt_rail_static.zip")
    return zips[-1]

def load_scheduled_njt_at_penn(service_date: str, njt_stop_ids: list) -> pd.DataFrame:
    zpath = latest_njt_static_zip()
    trips = read_txt_from_zip(zpath, "trips.txt")
    stop_times = read_txt_from_zip(zpath, "stop_times.txt")

    stop_times["stop_id_norm"] = stop_times["stop_id"].astype(str)
    njt_norm = {str(s) for s in njt_stop_ids}
    st = stop_times[stop_times["stop_id_norm"].isin(njt_norm)].copy()

    st["Scheduled_Arrival"]   = st["arrival_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    st["route_id"] = st["route_id"].astype(str)
    return st[["trip_id","stop_id","stop_id_norm","route_id","Scheduled_Arrival","Scheduled_Departure"]]

# ------------------- LOAD REALTIME -------------------------
RT_COLS = ["trip_id","stop_id","route_id","rt_arrival_utc","rt_departure_utc","source_minute_utc"]

def _read_rt_glob(folder: Path, pattern: str, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(folder.glob(pattern))
    if max_files and max_files > 0:
        files = files[-max_files:]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str)
            for c in RT_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            dfs.append(df[RT_COLS])
        except Exception as e:
            print(f"[warn] skip {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=RT_COLS)

def load_realtime_union_mta_njt() -> pd.DataFrame:
    mta = _read_rt_glob(RT_DIR, MTA_RT_GLOB, RT_MAX_FILES)
    njt = _read_rt_glob(NJT_RT_DIR, NJT_RT_GLOB, None)
    df = pd.concat([mta, njt], ignore_index=True).drop_duplicates()
    # normalize times
    for col in ["rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # normalize stop ids via subway static alias when available
    try:
        zpath = latest_static_zip()
        stops = read_txt_from_zip(zpath, "stops.txt")
        alias = build_stop_alias_map(stops)
        df["stop_id_norm"] = df["stop_id"].astype(str).map(alias).fillna(df["stop_id"].astype(str))
    except Exception:
        df["stop_id_norm"] = df["stop_id"].astype(str)
    # exact dedupe by minute bucket
    if {"trip_id","stop_id_norm","source_minute_utc"} <= set(df.columns):
        df["dedupe_key"] = (
            df["trip_id"].astype(str) + "|" +
            df["stop_id_norm"].astype(str) + "|" +
            df["source_minute_utc"].astype(str)
        )
        before = len(df)
        df = df.drop_duplicates("dedupe_key", keep="last").drop(columns=["dedupe_key"])
        print(f"[build_master] De-dup RT (trip/stop/min): {before} → {len(df)}")
    return df


# ------------------- LOAD STATIC (PENN) --------------------
def load_scheduled_at_penn(service_date: str, penn_stop_ids: list) -> pd.DataFrame:
    zpath = latest_static_zip()
    print(f"[build_master] Using static file: {zpath.name}")

    trips = read_txt_from_zip(zpath, "trips.txt")
    stop_times = read_txt_from_zip(zpath, "stop_times.txt")
    stops = read_txt_from_zip(zpath, "stops.txt")

    alias = build_stop_alias_map(stops)
    stop_times["stop_id_norm"] = stop_times["stop_id"].astype(str).map(alias).fillna(stop_times["stop_id"].astype(str))
    penn_norm = {alias.get(str(s), str(s)) for s in penn_stop_ids}

    st = stop_times[stop_times["stop_id_norm"].isin(penn_norm)].copy()
    st["Scheduled_Arrival"]   = st["arrival_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")

    cols = ["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Arrival", "Scheduled_Departure"]
    return st[cols]

# -------------------- ASOF PREP/SORT -----------------------
def _prep_for_asof(df: pd.DataFrame, ts_col: str, by_cols: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    df = df.copy()

    if by_cols:
        for c in by_cols:
            df[c] = df[c].astype(str)
    if "stop_id_norm" in df.columns:
        df["stop_id_norm"] = df["stop_id_norm"].astype(str)
    if "route_id" in df.columns:
        df["route_id"] = df["route_id"].astype(str)

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        if getattr(df[ts_col].dtype, "tz", None) is None:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    drop_cols = (by_cols or []) + [ts_col]
    df = df.dropna(subset=drop_cols)

    sort_cols = (by_cols or []) + [ts_col]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df

def _group_keys_intersection(left: pd.DataFrame, right: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    if not by_cols:
        return pd.DataFrame({"_dummy": [1]})
    lkeys = left[by_cols].drop_duplicates()
    rkeys = right[by_cols].drop_duplicates()
    both = lkeys.merge(rkeys, on=by_cols, how="inner")
    return both

def _groupwise_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    by_cols: list[str],
    tolerance: pd.Timedelta,
    direction: str = "nearest",
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame()

    if not by_cols:
        l = left.sort_values([left_on], kind="mergesort").reset_index(drop=True)
        r = right.sort_values([right_on], kind="mergesort").reset_index(drop=True)
        r = r.drop(columns=[c for c in by_cols if c in r.columns])
        return pd.merge_asof(
            l, r,
            left_on=left_on, right_on=right_on,
            direction=direction, tolerance=tolerance,
            allow_exact_matches=allow_exact_matches
        )

    keys = _group_keys_intersection(left, right, by_cols)
    out = []
    for _, keyrow in keys.iterrows():
        lmask = pd.Series(True, index=left.index)
        rmask = pd.Series(True, index=right.index)
        for c in by_cols:
            val = keyrow[c]
            lmask &= (left[c] == val)
            rmask &= (right[c] == val)

        lg = left.loc[lmask]
        rg = right.loc[rmask]
        if lg.empty or rg.empty:
            continue

        lg = lg.sort_values([left_on], kind="mergesort").reset_index(drop=True)
        rg = rg.sort_values([right_on], kind="mergesort").reset_index(drop=True)
        rg2 = rg.drop(columns=[c for c in by_cols if c in rg.columns])

        merged = pd.merge_asof(
            lg, rg2,
            left_on=left_on, right_on=right_on,
            direction=direction, tolerance=tolerance,
            allow_exact_matches=allow_exact_matches
        )
        out.append(merged)

    if not out:
        return pd.DataFrame(columns=list(set(left.columns) | set(right.columns)))
    return pd.concat(out, ignore_index=True)

# -------------------- TIME-BASED MATCH --------------------
def join_static_with_rt_time_based(sched_df: pd.DataFrame, rt_df: pd.DataFrame, tolerance_min: int = 30) -> pd.DataFrame:
    tol = pd.Timedelta(minutes=tolerance_min)

    for col in ["trip_id","stop_id","stop_id_norm","route_id","Scheduled_Arrival","Scheduled_Departure"]:
        if col not in sched_df.columns:
            raise KeyError(f"[join] sched_df missing: {col}")
    for col in ["trip_id","stop_id","stop_id_norm","route_id"]:
        if col not in rt_df.columns:
            raise KeyError(f"[join] rt_df missing: {col}")

    candidate_by = ["stop_id_norm", "route_id"]
    by_cols = [c for c in candidate_by if c in sched_df.columns and c in rt_df.columns]

    rtA_cols    = _uniq_cols(by_cols, ["trip_id","stop_id","stop_id_norm","route_id","rt_arrival_utc"])
    rtD_cols    = _uniq_cols(by_cols, ["trip_id","stop_id","stop_id_norm","route_id","rt_departure_utc"])
    schedA_cols = _uniq_cols(by_cols, ["Scheduled_Arrival"])
    schedD_cols = _uniq_cols(by_cols, ["Scheduled_Departure"])

    rtA    = _prep_for_asof(rt_df[rtA_cols].rename(columns={"rt_arrival_utc":"RT_Arrival"}),          "RT_Arrival",          by_cols)
    rtD    = _prep_for_asof(rt_df[rtD_cols].rename(columns={"rt_departure_utc":"RT_Departure"}),      "RT_Departure",        by_cols)
    schedA = _prep_for_asof(sched_df[schedA_cols],                                                   "Scheduled_Arrival",   by_cols)
    schedD = _prep_for_asof(sched_df[schedD_cols],                                                   "Scheduled_Departure", by_cols)

    leftA  = rtA[_uniq_cols(by_cols, ["RT_Arrival","trip_id","stop_id"])].rename(columns={"trip_id":"trip_id_arr","stop_id":"stop_id_arr"})
    rightA = schedA[_uniq_cols(by_cols, ["Scheduled_Arrival"])]

    leftD  = rtD[_uniq_cols(by_cols, ["RT_Departure","trip_id","stop_id"])].rename(columns={"trip_id":"trip_id_dep","stop_id":"stop_id_dep"})
    rightD = schedD[_uniq_cols(by_cols, ["Scheduled_Departure"])]

    print(f"[join] by_cols={by_cols}  rtA={len(rtA)}  schedA={len(schedA)}  rtD={len(rtD)}  schedD={len(schedD)}")

    a = _groupwise_asof(leftA, rightA, "RT_Arrival",     "Scheduled_Arrival",   by_cols, tol, "nearest", True) if not (leftA.empty or rightA.empty) else pd.DataFrame()
    d = _groupwise_asof(leftD, rightD, "RT_Departure",   "Scheduled_Departure", by_cols, tol, "nearest", True) if not (leftD.empty or rightD.empty) else pd.DataFrame()

    if a.empty and d.empty:
        return pd.DataFrame(columns=[
            "trip_id","route_id","stop_id","stop_id_norm",
            "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
            "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
        ])

    a = a.copy(); d = d.copy()
    if "trip_id_arr" in a.columns and "trip_id_dep" in d.columns:
        a["trip_id"] = a["trip_id_arr"]
        d["trip_id"] = d["trip_id_dep"]
        merge_keys = [*(by_cols or []), "trip_id"]
    else:
        a["rt_min_bucket"] = pd.to_datetime(a["RT_Arrival"], utc=True, errors="coerce").dt.floor("min")
        d["rt_min_bucket"] = pd.to_datetime(d["RT_Departure"], utc=True, errors="coerce").dt.floor("min")
        merge_keys = [*(by_cols or []), "rt_min_bucket"]

    print(f"[combine] merge_keys={merge_keys}")

    out = pd.merge(a, d, on=merge_keys, how="outer", suffixes=("_arr","_dep"), copy=False)

    for col in ["Scheduled_Arrival","RT_Arrival","Scheduled_Departure","RT_Departure"]:
        if col not in out.columns:
            out[col] = pd.NaT

    def _coalesce(cols):
        ser = pd.Series(pd.NA, index=out.index, dtype="object")
        for c in cols:
            if c in out.columns:
                ser = ser.combine_first(out[c])
        return ser

    out["trip_id"]  = _coalesce(["trip_id_arr", "trip_id_dep", "trip_id"])
    out["route_id"] = _coalesce(["route_id"])
    out["stop_id"]  = _coalesce(["stop_id_arr","stop_id_dep","stop_id"])
    if "stop_id_norm" not in out.columns:
        out["stop_id_norm"] = _coalesce(["stop_id_norm"])

    out["Arrival_Delay_Min"] = (
        pd.to_datetime(out["RT_Arrival"], utc=True, errors="coerce")
        - pd.to_datetime(out["Scheduled_Arrival"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0

    out["Departure_Delay_Min"] = (
        pd.to_datetime(out["RT_Departure"], utc=True, errors="coerce")
        - pd.to_datetime(out["Scheduled_Departure"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0

    keep = [
        "trip_id","route_id","stop_id","stop_id_norm",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[keep]
    return out

# --------------- INTERFACE CONSTRUCTION -------------------
def build_interfaces_123_ace(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    df["From_Node"] = df["route_id"].apply(route_to_node)

    def other(node: str) -> str | None:
        if node == "Subway_123": return "Subway_ACE"
        if node == "Subway_ACE": return "Subway_123"
        return None

    df["Target_Node"] = df["From_Node"].apply(other)

    df["Best_Arrival"]   = df["RT_Arrival"].combine_first(df["Scheduled_Arrival"])
    df["Best_Departure"] = df["RT_Departure"].combine_first(df["Scheduled_Departure"])
    df["Used_Scheduled_Fallback"] = df["RT_Arrival"].isna() | df["RT_Departure"].isna()

    arrivals   = df[df["Best_Arrival"].notna()].copy()
    departures = df[df["Best_Departure"].notna()].copy()
    departures["To_Node"] = departures["route_id"].apply(route_to_node)

    out_rows = []
    for _, a in arrivals.iterrows():
        tgt = a["Target_Node"]
        if not tgt:
            continue
        mask = (
            (departures["To_Node"] == tgt) &
            (departures["Best_Departure"] >= a["Best_Arrival"]) &
            (departures["Best_Departure"] <= a["Best_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
        )
        cand = departures.loc[mask].sort_values("Best_Departure").head(1)
        arr_min = pd.Timestamp(a["Best_Arrival"]).tz_convert("UTC").strftime("%Y%m%d_%H%M")
        iid = f"{a['From_Node']}_{tgt}_{arr_min}"

        if cand.empty:
            out_rows.append({
                "Interface_ID": iid,
                "From_Node": a["From_Node"],
                "To_Node": tgt,
                "Link_Type": "Subway-Subway",
                "Scheduled_Arrival": a.get("Scheduled_Arrival"),
                "RT_Arrival": a.get("RT_Arrival"),
                "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
                "Scheduled_Departure": pd.NaT,
                "RT_Departure": pd.NaT,
                "Departure_Delay_Min": pd.NA,
                "Transfer_Gap_Min": pd.NA,
                "Missed_Transfer_Flag": True,
                "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
            })
            continue

        d = cand.iloc[0]
        gap_min = (pd.Timestamp(d["Best_Departure"]) - pd.Timestamp(a["Best_Arrival"])).total_seconds() / 60.0
        out_rows.append({
            "Interface_ID": iid,
            "From_Node": a["From_Node"],
            "To_Node": tgt,
            "Link_Type": "Subway-Subway",
            "Scheduled_Arrival": a.get("Scheduled_Arrival"),
            "RT_Arrival": a.get("RT_Arrival"),
            "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
            "Scheduled_Departure": d.get("Scheduled_Departure"),
            "RT_Departure": d.get("RT_Departure"),
            "Departure_Delay_Min": d.get("Departure_Delay_Min"),
            "Transfer_Gap_Min": gap_min,
            "Missed_Transfer_Flag": (gap_min < MISSED_THRESHOLD_MIN),
            "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
        })

    return pd.DataFrame(out_rows)

def build_interfaces_rail_to_subway(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    df["From_Node"] = df["route_id"].apply(route_to_node)
    df["Best_Arrival"]   = df["RT_Arrival"].combine_first(df["Scheduled_Arrival"])
    df["Best_Departure"] = df["RT_Departure"].combine_first(df["Scheduled_Departure"])
    df["Used_Scheduled_Fallback"] = df["RT_Arrival"].isna() | df["RT_Departure"].isna()

    arrivals = df[(df["Best_Arrival"].notna()) & (df["From_Node"] == "NJT_Rail")].copy()

    departures = df[df["Best_Departure"].notna()].copy()
    departures["To_Node"] = departures["route_id"].apply(route_to_node)
    departures = departures[departures["To_Node"].isin(["Subway_123","Subway_ACE"])]

    out = []
    for _, a in arrivals.iterrows():
        cand = departures[
            (departures["Best_Departure"] >= a["Best_Arrival"]) &
            (departures["Best_Departure"] <= a["Best_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
        ].sort_values("Best_Departure").head(1)

        arr_min = pd.Timestamp(a["Best_Arrival"]).tz_convert("UTC").strftime("%Y%m%d_%H%M")
        if cand.empty:
            out.append({
                "Interface_ID": f"NJT_Rail_Subway_{arr_min}",
                "From_Node": "NJT_Rail",
                "To_Node": "Subway",
                "Link_Type": "Rail-Subway",
                "Scheduled_Arrival": a.get("Scheduled_Arrival"),
                "RT_Arrival": a.get("RT_Arrival"),
                "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
                "Scheduled_Departure": pd.NaT,
                "RT_Departure": pd.NaT,
                "Departure_Delay_Min": pd.NA,
                "Transfer_Gap_Min": pd.NA,
                "Missed_Transfer_Flag": True,
                "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
            })
            continue

        d = cand.iloc[0]
        gap = (pd.Timestamp(d["Best_Departure"]) - pd.Timestamp(a["Best_Arrival"])).total_seconds() / 60.0
        out.append({
            "Interface_ID": f"NJT_Rail_{d['To_Node']}_{arr_min}",
            "From_Node": "NJT_Rail",
            "To_Node": d["To_Node"],
            "Link_Type": "Rail-Subway",
            "Scheduled_Arrival": a.get("Scheduled_Arrival"),
            "RT_Arrival": a.get("RT_Arrival"),
            "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
            "Scheduled_Departure": d.get("Scheduled_Departure"),
            "RT_Departure": d.get("RT_Departure"),
            "Departure_Delay_Min": d.get("Departure_Delay_Min"),
            "Transfer_Gap_Min": gap,
            "Missed_Transfer_Flag": (gap < MISSED_THRESHOLD_MIN),
            "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
        })

    return pd.DataFrame(out)

# ---------------- TIME FEATURES & FIELDS ------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    when = (df["RT_Arrival"].combine_first(df["RT_Departure"])
                     .combine_first(df["Scheduled_Arrival"])
                     .combine_first(df["Scheduled_Departure"]))
    when_local = pd.to_datetime(when, utc=True).dt.tz_convert("America/New_York")

    df["Day_of_Week"] = when_local.dt.day_name()
    df["Hour_of_Day"] = when_local.dt.hour
    df["Peak_Flag"] = df["Hour_of_Day"].isin(list(AM_PEAK) + list(PM_PEAK))
    df["Time_Period"] = pd.Categorical(
        ["AM peak" if h in AM_PEAK else "PM peak" if h in PM_PEAK else "Off-peak" for h in df["Hour_of_Day"]],
        categories=["AM peak", "PM peak", "Off-peak"]
    )
    return df

def add_placeholders_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Arrival_Delay_Min", "Departure_Delay_Min"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["Delay_Chain_Min"] = (
        pd.to_numeric(df["Arrival_Delay_Min"], errors="coerce").fillna(0) +
        pd.to_numeric(df["Departure_Delay_Min"], errors="coerce").fillna(0)
    )
    for col in [
        "Avg_Flow_Volume","Peak_Flow_Volume","Daily_Ridership_Share_%",
        "Delay_Frequency_%","Avg_Delay_Min","Chain_Reaction_Factor",
        "Alt_Path_Available","Criticality_Score","Ped_Count","Stress_Index",
        "External_Pressure","Incident_History"
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df

# ---------------------- MAIN ------------------------------
def main():
    print("[t0] starting build_master")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/penn_stops.json")
    cfg = json.loads(CONFIG_PATH.read_text())

    # Pull stop ids; keep subway mandatory, NJT optional
    subway_penn_ids = cfg.get("subway_penn_stops", [])
    if not subway_penn_ids:
        raise RuntimeError("'subway_penn_stops' is empty in config/penn_stops.json")
    njt_penn_ids = cfg.get("njt_penn_stops", [])  # optional

    # Debug: show input row counts from both folders
    mta_count = len(_read_rt_glob(Path(MTA_RT_DIR), MTA_RT_GLOB))
    njt_count = len(_read_rt_glob(Path(NJT_RT_DIR), NJT_RT_GLOB))
    print(f"[build_master] RT union sizes: MTA={mta_count}  NJT={njt_count}")
  

    # Load union RT (MTA + NJT)
    rt = load_realtime_union_mta_njt()
    if rt.empty:
        print("[build_master] No realtime data found; exiting gracefully.")
        return

    # Service date
    service_date = SERVICE_DATE_OVERRIDE or infer_service_date_from_rt(rt)
    print(f"[build_master] Using service_date: {service_date}")

    # Subway scheduled @ Penn
    sched_sub = load_scheduled_at_penn(service_date, subway_penn_ids)

    # NJT scheduled @ Penn (optional)
    try:
        sched_njt = load_scheduled_njt_at_penn(service_date, njt_penn_ids) if njt_penn_ids else pd.DataFrame()
    except FileNotFoundError:
        print("[build_master] NJT static not found; continuing with subway static only.")
        sched_njt = pd.DataFrame()

    # Union scheduled
    sched = pd.concat([sched_sub, sched_njt], ignore_index=True) if not sched_njt.empty else sched_sub

    # Filter static to RT time window ±60 min (if RT has timestamps)
    rt_times = pd.concat([rt.get("rt_arrival_utc"), rt.get("rt_departure_utc")], ignore_index=True).dropna()
    if not rt_times.empty:
        rt_min, rt_max = rt_times.min(), rt_times.max()
        pad = pd.Timedelta(minutes=60)
        before = len(sched)
        sched = sched[
            sched["Scheduled_Arrival"].between(rt_min - pad, rt_max + pad) |
            sched["Scheduled_Departure"].between(rt_min - pad, rt_max + pad)
        ].copy()
        print(f"[build_master] Filtered static to RT window: {before} → {len(sched)} rows")
    else:
        print("[build_master] No RT times; skipping static time window filter.")

    print(f"[build_master] RT counts: "
          f"arrivals={rt['rt_arrival_utc'].notna().sum()} "
          f"departures={rt['rt_departure_utc'].notna().sum()}")

    # Time-based merge (nearest within tolerance)
    events_at_penn = join_static_with_rt_time_based(sched, rt, tolerance_min=30)

    # ---------- BUILD INTERFACES ----------
    # 1) Subway ↔ Subway (1/2/3 ↔ A/C/E)
    interfaces_sub = build_interfaces_123_ace(events_at_penn)

    # 2) NJT ↔ Subway (both buckets, earliest feasible departure chosen)
    interfaces_rail = build_interfaces_rail_to_subway(events_at_penn)

    # 3) Combine
    interfaces = pd.concat([interfaces_sub, interfaces_rail], ignore_index=True)

    # 4) De-dup identical pairs across reruns
    before = len(interfaces)
    interfaces = interfaces.drop_duplicates(
        subset=[
            "From_Node","To_Node","Link_Type",
            "Scheduled_Arrival","RT_Arrival",
            "Scheduled_Departure","RT_Departure"
        ],
        keep="last"
    )
    print(f"[build_master] De-dup interfaces: {before} → {len(interfaces)}")

    # 5) Enrich & finalize columns
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders_and_scores(interfaces)

    cols = [
        "Interface_ID","From_Node","To_Node","Link_Type",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min",
        "Transfer_Gap_Min","Missed_Transfer_Flag",
        "Avg_Flow_Volume","Peak_Flow_Volume","Daily_Ridership_Share_%",
        "Delay_Frequency_%","Avg_Delay_Min","Delay_Chain_Min","Chain_Reaction_Factor",
        "Alt_Path_Available","Criticality_Score","Ped_Count","Stress_Index",
        "Time_Period","Day_of_Week","Hour_of_Day","Peak_Flag",
        "External_Pressure","Incident_History","Used_Scheduled_Fallback"
    ]
    for c in cols:
        if c not in interfaces.columns:
            interfaces[c] = pd.NA
    interfaces = interfaces[cols]

    # ---------------- Append to rolling master ----------------
    roll_csv  = CURATED_DIR / "master_interface_dataset.csv"
    roll_parq = CURATED_DIR / "master_interface_dataset.parquet"

    def _dedup_key(df: pd.DataFrame) -> pd.Series:
        # Stable key: (Interface_ID, To_Node, RT_Departure minute)
        rt_dep_min = pd.to_datetime(df.get("RT_Departure"), utc=True, errors="coerce").dt.floor("min").astype(str)
        return (df.get("Interface_ID").astype(str) + "|" +
                df.get("To_Node").astype(str) + "|" +
                rt_dep_min.fillna("NaT"))

    # If an old master exists, read → concat → dedupe → write (APPEND semantics)
    if roll_parq.exists() or roll_csv.exists():
        try:
            old = pd.read_parquet(roll_parq)
            print(f"[build_master] Loaded previous master (parquet): {len(old)} rows")
        except Exception:
            if roll_csv.exists():
                old = pd.read_csv(roll_csv, low_memory=False)
                print(f"[build_master] Loaded previous master (csv): {len(old)} rows")
            else:
                old = pd.DataFrame(columns=interfaces.columns)

        if not old.empty:
            old["_dedup_key"] = _dedup_key(old)
        else:
            old["_dedup_key"] = pd.Series([], dtype="object")
        new = interfaces.copy()
        new["_dedup_key"] = _dedup_key(new)

        combined = (pd.concat([old, new], ignore_index=True)
                      .drop_duplicates("_dedup_key", keep="last")
                      .drop(columns=["_dedup_key"], errors="ignore"))
        interfaces = combined
        print(f"[build_master] Appended; total {len(interfaces)} rows")
    else:
        print("[build_master] Creating master for the first time.")

    # Save CSV + Parquet
    interfaces.to_csv(roll_csv, index=False)
    try:
        interfaces.to_parquet(roll_parq, index=False)
    except Exception as e:
        print(f"[build_master] Parquet write skipped: {e}")

    print(f"[build_master] Wrote {len(interfaces)} rows → {roll_csv} / {roll_parq}")

if __name__ == "__main__":
    main()
