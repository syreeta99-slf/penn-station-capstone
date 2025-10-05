#!/usr/bin/env python3
"""
pollers/build_master.py

Build the event-level Master Interface Dataset by joining:
- Static GTFS (scheduled) from gtfs_static/subway_all_YYYYMMDD.zip
- Realtime subway polls from data/realtime/subway_rt_*.csv (or RT_DIR)

Outputs (always):
- data/curated/master_interface_dataset.csv
- data/curated/master_interface_dataset.parquet

Append mode (if APPEND_TO_MASTER=1):
- Reads the prior master, appends new, de-dupes on a stable key, writes back.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, io, json, zipfile, subprocess, sys, textwrap

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
RT_DIR = Path(os.getenv("RT_DIR", str(REALTIME_DIR)))
RT_MAX_FILES = int(os.getenv("RT_MAX_FILES", "60"))  # how many newest CSVs to consider (set big for backfill)
SERVICE_DATE_OVERRIDE = os.getenv("SERVICE_DATE", "")  # YYYY-MM-DD or empty to infer from RT
TRANSFER_WINDOW_MIN = int(os.getenv("TRANSFER_WINDOW_MIN", "20"))
MISSED_THRESHOLD_MIN = float(os.getenv("MISSED_THRESHOLD_MIN", "2"))
ALLOW_STATIC_REFRESH = os.getenv("ALLOW_STATIC_REFRESH", "1")
APPEND_TO_MASTER = os.getenv("APPEND_TO_MASTER", "0") == "1"

AM_PEAK = range(6, 10)          # 06–09 local
PM_PEAK = range(16, 20)         # 16–19 local

# Route → Node mapping (tune as needed)
ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["4", "5", "6", "7", "S"]),
    "Subway_ACE": set(list("ACE")),
}

# ----------------------- HELPERS ---------------------------
def route_to_node(route_id: str) -> str:
    if pd.isna(route_id) or route_id is None:
        return "Subway_Unknown"
    rid = str(route_id)
    for node, routes in ROUTE_GROUPS.items():
        if rid in routes:
            return node
    return "Subway_Other"


def parse_gtfs_time_to_dt(hms: str, service_date: str) -> pd.Timestamp:
    """Handle 24+ hour GTFS times relative to service_date (UTC)."""
    try:
        h, m, s = map(int, str(hms).split(":"))
    except Exception:
        return pd.NaT
    extra_days, h = divmod(h, 24)
    base = pd.Timestamp(service_date).tz_localize("UTC")
    return base + pd.Timedelta(days=extra_days, hours=h, minutes=m, seconds=s)


def latest_static_zip() -> Path:
    zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips and ALLOW_STATIC_REFRESH == "1":
        print("[build_master] No static ZIPs found. Attempting one-time refresh...")
        subprocess.check_call([sys.executable, "pollers/mta_static_refresh.py"])
        zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips:
        raise FileNotFoundError("No subway_all_*.zip in gtfs_static/.")
    return zips[-1]


def read_txt_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)))


def infer_service_date_from_rt(rt_df: pd.DataFrame) -> str:
    ts = pd.concat([rt_df.get("rt_arrival_utc"), rt_df.get("rt_departure_utc")], ignore_index=True)
    ts = ts.dropna()
    if ts.empty:
        return datetime.utcnow().date().isoformat()
    ts_local = ts.dt.tz_convert("America/New_York")
    return ts_local.dt.date.min().isoformat()


def build_stop_alias_map(stops_df: pd.DataFrame) -> dict:
    alias = {}
    for _, r in stops_df.iterrows():
        sid = str(r["stop_id"])
        parent = r.get("parent_station")
        parent = str(parent) if pd.notna(parent) else sid
        alias[sid] = parent
    return alias


# ------------------- LOAD INPUTS --------------------------
def load_realtime_events() -> pd.DataFrame:
    files = sorted(RT_DIR.glob("subway_rt_*.csv"))
    if not files:
        raise FileNotFoundError(f"No realtime CSVs in {RT_DIR}.")
    if RT_MAX_FILES > 0:
        files = files[-RT_MAX_FILES:]

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"[warn] skip {f.name}: {e}")
    if not dfs:
        raise RuntimeError("No readable realtime CSVs.")
    df = pd.concat(dfs, ignore_index=True)

    # normalize times
    for col in ["rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # normalize stop ids via static
    zpath = latest_static_zip()
    stops = read_txt_from_zip(zpath, "stops.txt")
    alias = build_stop_alias_map(stops)
    df["stop_id_norm"] = df["stop_id"].astype(str).map(alias).fillna(df["stop_id"].astype(str))

    # de-dupe by exact (trip_id, stop_id_norm, source_minute_utc)
    if {"trip_id","stop_id_norm","source_minute_utc"} <= set(df.columns):
        df["dedupe_key"] = (
            df["trip_id"].astype(str) + "|" +
            df["stop_id_norm"].astype(str) + "|" +
            df["source_minute_utc"].astype(str)
        )
        before = len(df)
        df = df.drop_duplicates("dedupe_key", keep="last").drop(columns=["dedupe_key"])
        print(f"[build_master] De-dup events (exact): {before} → {len(df)}")

    return df


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
    st["Scheduled_Arrival"] = st["arrival_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    cols = ["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Arrival", "Scheduled_Departure"]
    return st[cols]


# -------------------- ASOF PREP / SORT --------------------
def _prep_for_asof(df: pd.DataFrame, ts_col: str, by_cols: list[str] | None = None) -> pd.DataFrame:
    """
    General-purpose preparation for pd.merge_asof:
      - cast stop_id_norm to string if present (stable key shape)
      - ensure ts_col is datetime
      - drop rows with NaN/NaT in join keys
      - sort by by_cols (if provided) THEN ts_col
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    df = df.copy()

    if "stop_id_norm" in df.columns:
        df["stop_id_norm"] = df["stop_id_norm"].astype(str)

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    drop_cols = (by_cols or []) + [ts_col]
    df = df.dropna(subset=drop_cols)

    sort_cols = (by_cols or []) + [ts_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def _assert_sorted(df: pd.DataFrame, ts_col: str, by_cols: list[str] | None = None, label: str = "") -> None:
    """Optional guard to fail fast with a clear message if sorting precondition breaks."""
    if df is None or df.empty:
        return
    sort_cols = (by_cols or []) + [ts_col]
    ok = True
    if by_cols:
        # check groupwise monotonicity
        bad_groups = []
        for k, g in df.groupby(by_cols, dropna=False, sort=False):
            if not g[ts_col].is_monotonic_increasing:
                bad_groups.append(k)
                if len(bad_groups) >= 5:
                    break
        ok = (len(bad_groups) == 0)
        if not ok:
            raise ValueError(f"[asof-guard] {label}: time not sorted within groups for {ts_col}; bad groups (first 5): {bad_groups}")
    else:
        if not df[ts_col].is_monotonic_increasing:
            raise ValueError(f"[asof-guard] {label}: {ts_col} not globally sorted.")


# -------------------- TIME-BASED MATCH --------------------
def join_static_with_rt_time_based(sched_df: pd.DataFrame, rt_df: pd.DataFrame, tolerance_min: int = 30) -> pd.DataFrame:
    """
    Nearest-time join between realtime and scheduled stop events at Penn.
    Uses merge_asof on both arrivals and departures, with multi-key group-aware sorting.
    """
    tol = pd.Timedelta(minutes=tolerance_min)

    # Columns present check (defensive)
    for needed in [
        "trip_id","stop_id","stop_id_norm","route_id",
        "Scheduled_Arrival","Scheduled_Departure"
    ]:
        if needed not in sched_df.columns:
            raise KeyError(f"[join] sched_df is missing required column: {needed}")

    for needed in ["trip_id","stop_id","stop_id_norm","route_id","rt_arrival_utc","rt_departure_utc"]:
        if needed not in rt_df.columns:
            # allow missing arrivals or departures; we’ll handle empties
            if needed in ("rt_arrival_utc","rt_departure_utc"):
                continue
            raise KeyError(f"[join] rt_df is missing required column: {needed}")

    # Build slim views
    rtA = (rt_df[["trip_id","stop_id","stop_id_norm","route_id","rt_arrival_utc"]]
           .rename(columns={"rt_arrival_utc":"RT_Arrival"}))
    rtD = (rt_df[["trip_id","stop_id","stop_id_norm","route_id","rt_departure_utc"]]
           .rename(columns={"rt_departure_utc":"RT_Departure"}))

    schedA = sched_df[["trip_id","stop_id","stop_id_norm","route_id","Scheduled_Arrival"]].copy()
    schedD = sched_df[["trip_id","stop_id","stop_id_norm","route_id","Scheduled_Departure"]].copy()

    # Choose grouping keys that exist on both sides (stable & specific first)
    candidate_by = ["stop_id_norm", "route_id"]
    by_cols = [c for c in candidate_by if c in sched_df.columns and c in rt_df.columns]

    # Prepare both sides (casts, drops, sorts)
    rtA = _prep_for_asof(rtA, "RT_Arrival", by_cols)
    rtD = _prep_for_asof(rtD, "RT_Departure", by_cols)
    schedA = _prep_for_asof(schedA, "Scheduled_Arrival", by_cols)
    schedD = _prep_for_asof(schedD, "Scheduled_Departure", by_cols)

    # Optional sanity guards (clear fail if preconditions break)
    _assert_sorted(rtA, "RT_Arrival", by_cols, label="rtA")
    _assert_sorted(schedA, "Scheduled_Arrival", by_cols, label="schedA")
    _assert_sorted(rtD, "RT_Departure", by_cols, label="rtD")
    _assert_sorted(schedD, "Scheduled_Departure", by_cols, label="schedD")

    # merge_asof for arrivals and departures (left = RT)
    if rtA.empty or schedA.empty:
        a = pd.DataFrame()
    else:
        a = pd.merge_asof(
            rtA, schedA,
            left_on="RT_Arrival", right_on="Scheduled_Arrival",
            by=by_cols if by_cols else None,
            direction="nearest", tolerance=tol, allow_exact_matches=True
        )

    if rtD.empty or schedD.empty:
        d = pd.DataFrame()
    else:
        d = pd.merge_asof(
            rtD, schedD,
            left_on="RT_Departure", right_on="Scheduled_Departure",
            by=by_cols if by_cols else None,
            direction="nearest", tolerance=tol, allow_exact_matches=True
        )

    # If both were empty, return a coherent empty schema
    if a.empty and d.empty:
        return pd.DataFrame(columns=[
            "trip_id","route_id","stop_id","stop_id_norm",
            "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
            "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
        ])

    # Build combined view.
    # Start from union of keys seen in either frame (on the chosen by_cols).
    if by_cols:
        base_keys = list(by_cols)
    else:
        base_keys = ["stop_id_norm"] if "stop_id_norm" in (a.columns if not a.empty else d.columns) else []

    if base_keys:
        base = pd.concat(
            [
                a[base_keys] if not a.empty else pd.DataFrame(columns=base_keys),
                d[base_keys] if not d.empty else pd.DataFrame(columns=base_keys),
            ],
            ignore_index=True,
        ).drop_duplicates()
    else:
        # fallback: simple single-row base to outer-merge into
        base = pd.DataFrame({"_join_base": [1]})

    out = base
    if not a.empty:
        out = out.merge(a, on=base_keys, how="left") if base_keys else out.merge(a, left_on="_join_base", right_index=True, how="left")
    if not d.empty:
        out = out.merge(d, on=base_keys, how="left", suffixes=("_arr","_dep")) if base_keys else out.merge(d, left_on="_join_base", right_index=True, suffixes=("_arr","_dep"), how="left")

    # Ensure expected columns exist
    for c in ["trip_id_arr","stop_id_arr","route_id_arr","trip_id_dep","stop_id_dep","route_id_dep",
              "Scheduled_Arrival","RT_Arrival","Scheduled_Departure","RT_Departure"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["trip_id"] = out.get("trip_id_arr").combine_first(out.get("trip_id_dep"))
    out["route_id"] = out.get("route_id_arr").combine_first(out.get("route_id_dep"))
    out["stop_id"]  = out.get("stop_id_arr").combine_first(out.get("stop_id_dep"))

    out["Arrival_Delay_Min"] = (
        pd.to_datetime(out["RT_Arrival"], utc=True) - pd.to_datetime(out["Scheduled_Arrival"], utc=True)
    ).dt.total_seconds() / 60.0

    out["Departure_Delay_Min"] = (
        pd.to_datetime(out["RT_Departure"], utc=True) - pd.to_datetime(out["Scheduled_Departure"], utc=True)
    ).dt.total_seconds() / 60.0

    keep = [
        "trip_id","route_id","stop_id","stop_id_norm",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
    ]
    return out[keep]


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

    # best known timestamps
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
        # Find the *soonest* departure on the target side within window
        mask = (
            (departures["To_Node"] == tgt) &
            (departures["Best_Departure"] >= a["Best_Arrival"]) &
            (departures["Best_Departure"] <= a["Best_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
        )
        cand = departures.loc[mask].sort_values("Best_Departure").head(1)

        # Interface_ID per arrival minute
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

    out = pd.DataFrame(out_rows)
    return out


# ---------------- TIME FEATURES & FIELDS ------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    when = (df["RT_Arrival"].combine_first(df["RT_Departure"])
                     .combine_first(df["Scheduled_Arrival"])
                     .combine_first(df["Scheduled_Departure"]))
    when_local = pd.to_datetime(when, utc=True).dt.tz_convert("America/New_York")

    df["Day_of_World"] = when_local.dt.day_name()   # (typo-proofing kept original name below too)
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
    # Load config for the Penn stops
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/penn_stops.json")
    cfg = json.loads(CONFIG_PATH.read_text())
    penn_ids = cfg.get("subway_penn_stops", [])
    if not penn_ids:
        raise RuntimeError("'subway_penn_stops' is empty in config/penn_stops.json")

    # Load RT
    rt = load_realtime_events()

    # Service date
    service_date = infer_service_date_from_rt(rt)
    if SERVICE_DATE_OVERRIDE:
        service_date = SERVICE_DATE_OVERRIDE
    print(f"[build_master] Using service_date: {service_date}")

    # Static scheduled rows for Penn
    sched = load_scheduled_at_penn(service_date, penn_ids)

    # Filter static to RT time window ±60 min (if RT has timestamps)
    rt_times = pd.concat([rt.get("rt_arrival_utc"), rt.get("rt_departure_utc")], ignore_index=True).dropna()
    if not rt_times.empty:
        rt_min, rt_max = rt_times.min(), rt_times.max()
        pad = pd.Timedelta(minutes=60)
        rt_min, rt_max = rt_min - pad, rt_max + pad
        before = len(sched)
        sched = sched[
            sched["Scheduled_Arrival"].between(rt_min, rt_max) |
            sched["Scheduled_Departure"].between(rt_min, rt_max)
        ].copy()
        print(f"[build_master] Filtered static to RT window: {before} → {len(sched)} rows")
    else:
        print("[build_master] No RT times; skipping static time window filter.")

    print(f"[build_master] RT non-null counts: "
          f"arr={rt['rt_arrival_utc'].notna().sum()} "
          f"dep={rt['rt_departure_utc'].notna().sum()} "
          f"stop_norm_missing={(rt['stop_id_norm'].isna().sum() if 'stop_id_norm' in rt.columns else 'NA')}")

    # Time-based merge (nearest within tolerance)
    events_at_penn = join_static_with_rt_time_based(sched, rt, tolerance_min=30)

    # Simple QC preview
    total_sched = len(sched)
    rt_rows = len(rt)
    evt = len(events_at_penn)
    with_rt_arr = events_at_penn["RT_Arrival"].notna().sum() if "RT_Arrival" in events_at_penn.columns else 0
    with_rt_dep = events_at_penn["RT_Departure"].notna().sum() if "RT_Departure" in events_at_penn.columns else 0
    print(f"[build_master] sched_rows={total_sched}  rt_rows={rt_rows}  time-matched={evt} "
          f"with_RT_Arr={with_rt_arr} with_RT_Dep={with_rt_dep}")

    # Persist raw join for debug (optional)
    try:
        (CURATED_DIR / "_debug_events_at_penn.csv").write_text(
            events_at_penn.to_csv(index=False))
    except Exception:
        pass

    # Guard
    required = ["trip_id","stop_id","route_id","Scheduled_Arrival","Scheduled_Departure","RT_Arrival","RT_Departure"]
    missing = [c for c in required if c not in events_at_penn.columns]
    if missing:
        raise RuntimeError(f"Missing columns after join: {missing}. Got: {list(events_at_penn.columns)}")

    print("[build_master] Preview of merged columns:")
    print(events_at_penn[required].head(8))

    # Build interfaces
    interfaces = build_interfaces_123_ace(events_at_penn)
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders_and_scores(interfaces)

    # Reorder/ensure columns
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

    # QC snippets
    if not interfaces.empty:
        by_link = interfaces.groupby(["From_Node","To_Node"]).size()
        print("\n[qc] Interfaces by link:")
        print(by_link)

        by_time = interfaces["Time_Period"].value_counts(dropna=False)
        print("\n[qc] Time buckets:")
        print(by_time)

        miss_rate = interfaces["Missed_Transfer_Flag"].mean()
        print("\n[qc] Missed-transfer rate:")
        print(f"Missed_Transfer_Flag rate: {miss_rate:.1%}")

        # tiny sample
        print("\n[qc] Sample:")
        print(interfaces.head(5)[["Interface_ID","From_Node","To_Node","RT_Arrival","RT_Departure","Transfer_Gap_Min","Missed_Transfer_Flag"]])

    # ---------------- Rolling append (if enabled) ----------------
    roll_csv  = CURATED_DIR / "master_interface_dataset.csv"
    roll_parq = CURATED_DIR / "master_interface_dataset.parquet"

    def _dedup_key(df: pd.DataFrame) -> pd.Series:
        rt_dep_min = pd.to_datetime(df.get("RT_Departure"), utc=True, errors="coerce").dt.floor("min").astype(str)
        return (df.get("Interface_ID").astype(str) + "|" +
                df.get("To_Node").astype(str) + "|" +
                rt_dep_min.fillna("NaT"))

    if APPEND_TO_MASTER and (roll_csv.exists() or roll_parq.exists()):
        # Load previous rolling dataset (prefer parquet)
        try:
            old = pd.read_parquet(roll_parq)
            print(f"[build_master] Loaded previous rolling master (parquet): {len(old)} rows")
        except Exception:
            if roll_csv.exists():
                old = pd.read_csv(roll_csv, low_memory=False)
                print(f"[build_master] Loaded previous rolling master (csv): {len(old)} rows")
            else:
                old = pd.DataFrame(columns=interfaces.columns)
                print("[build_master] No previous rolling master found; starting fresh")
        # De-dup
        new = interfaces.copy()
        if not old.empty:
            old["_dedup_key"] = _dedup_key(old)
        else:
            old["_dedup_key"] = pd.Series([], dtype="object")
        new["_dedup_key"] = _dedup_key(new)

        combined = (pd.concat([old, new], ignore_index=True)
                      .drop_duplicates("_dedup_key", keep="last")
                      .drop(columns=["_dedup_key"], errors="ignore"))
        interfaces = combined
        print(f"[build_master] Rolling append enabled → {len(interfaces)} rows total")
    else:
        print("[build_master] Rolling append disabled or no prior file; writing fresh.")

    # ---------------- Save (rolling master) ----------------
    interfaces.to_csv(roll_csv, index=False)
    try:
        interfaces.to_parquet(roll_parq, index=False)
    except Exception as e:
        print(f"[build_master] Parquet write skipped: {e}")

    print(f"[build_master] Wrote {len(interfaces)} rows → {roll_csv} / {roll_parq}")


if __name__ == "__main__":
    main()
