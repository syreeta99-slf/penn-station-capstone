#!/usr/bin/env python3
"""
pollers/build_master.py

Builds the event-level Master Interface Dataset by joining:
- Static GTFS (scheduled) from gtfs_static/subway_all_YYYYMMDD.zip
- Realtime subway polls from data/realtime/subway_rt_*.csv

Outputs:
- data/curated/master_interface_dataset.csv
- data/curated/master_interface_dataset.parquet
"""

from pathlib import Path
from datetime import datetime
import io, json, zipfile
import pandas as pd
import subprocess, sys

# ----------------------- PATHS -----------------------
DATA_DIR = Path("data")
REALTIME_DIR = DATA_DIR / "realtime"
CURATED_DIR = DATA_DIR / "curated"
STATIC_DIR = Path("gtfs_static")
CONFIG_PATH = Path("config/penn_stops.json")

CURATED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- PARAMETERS ---------------------
TRANSFER_WINDOW_MIN = 25       # arrivals → next departure must be within this window
MISSED_THRESHOLD_MIN = 3       # < 3 min gap (or no departure) ⇒ missed/unsafe

AM_PEAK = range(6, 10)         # 06–09
PM_PEAK = range(16, 20)        # 16–19

ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["4", "5", "6", "7", "S"]),
    "Subway_ACE": set(list("ACE")),
}

# -------------------- HELPERS ------------------------
def route_to_node(route_id: str) -> str:
    if pd.isna(route_id) or route_id is None:
        return "Subway_Unknown"
    rid = str(route_id)
    for node, routes in ROUTE_GROUPS.items():
        if rid in routes:
            return node
    return "Subway_Other"


def parse_gtfs_time_to_dt(hms: str, service_date: str) -> pd.Timestamp:
    try:
        h, m, s = map(int, str(hms).split(":"))
    except Exception:
        return pd.NaT
    extra_days, h = divmod(h, 24)
    base = pd.Timestamp(service_date).tz_localize("UTC")
    return base + pd.Timedelta(days=extra_days, hours=h, minutes=m, seconds=s)


def latest_static_zip() -> Path:
    zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips:
        print("[build_master] No static ZIPs found in gtfs_static/. Attempting one-time refresh...")
        subprocess.check_call([sys.executable, "pollers/mta_static_refresh.py"])
        zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
        if not zips:
            raise FileNotFoundError("Static refresh ran but no subway_all_*.zip present in gtfs_static/.")
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

# ------------------- LOAD INPUTS ---------------------
def load_realtime_events() -> pd.DataFrame:
    files = sorted(REALTIME_DIR.glob("subway_rt_*.csv"))
    if not files:
        raise FileNotFoundError("No realtime CSVs in data/realtime/. Run the realtime poller first.")
    dfs = [pd.read_csv(f) for f in files[-60:]]
    df = pd.concat(dfs, ignore_index=True)

    for col in ["rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    zpath = latest_static_zip()
    stops = read_txt_from_zip(zpath, "stops.txt")
    alias = build_stop_alias_map(stops)
    df["stop_id_norm"] = df["stop_id"].astype(str).map(alias).fillna(df["stop_id"].astype(str))

    df["dedupe_key"] = (
        df["trip_id"].astype(str) + "|" +
        df["stop_id_norm"].astype(str) + "|" +
        df["source_minute_utc"].astype(str)
    )
    df = df.drop_duplicates("dedupe_key", keep="last").drop(columns=["dedupe_key"])
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
    return st[["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Arrival", "Scheduled_Departure"]]

# -------------------- TIME-BASED MATCH ------------------
def join_static_with_rt_time_based(sched_df: pd.DataFrame, rt_df: pd.DataFrame, tolerance_min: int = 30) -> pd.DataFrame:
    """
    Match RT events to static schedules by stop_id_norm + nearest time within tolerance (minutes).
    Uses pre-renamed columns (no suffix guessing), enforces global sort by 'on' keys, and returns a stable schema.
    """
    tol = pd.Timedelta(minutes=tolerance_min)

    # ---- Build RT frames and pre-rename to unique names ----
    rtA = rt_df[["trip_id", "stop_id", "stop_id_norm", "route_id", "rt_arrival_utc"]].rename(
        columns={
            "trip_id": "trip_id_arr",
            "stop_id": "stop_id_arr",
            "route_id": "route_id_arr",
            "rt_arrival_utc": "RT_Arrival",
        }
    ).copy()

    rtD = rt_df[["trip_id", "stop_id", "stop_id_norm", "route_id", "rt_departure_utc"]].rename(
        columns={
            "trip_id": "trip_id_dep",
            "stop_id": "stop_id_dep",
            "route_id": "route_id_dep",
            "rt_departure_utc": "RT_Departure",
        }
    ).copy()

    # ---- Build schedule frames and pre-rename to unique names ----
    schedA = sched_df[["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Arrival"]].rename(
        columns={
            "trip_id": "trip_id_sched_arr",
            "stop_id": "stop_id_sched_arr",
            "route_id": "route_id_sched_arr",
        }
    ).copy()

    schedD = sched_df[["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Departure"]].rename(
        columns={
            "trip_id": "trip_id_sched_dep",
            "stop_id": "stop_id_sched_dep",
            "route_id": "route_id_sched_dep",
        }
    ).copy()

    # ---- Drop rows with null merge keys ----
    rtA = rtA.dropna(subset=["stop_id_norm", "RT_Arrival"]).copy()
    rtD = rtD.dropna(subset=["stop_id_norm", "RT_Departure"]).copy()
    schedA = schedA.dropna(subset=["stop_id_norm", "Scheduled_Arrival"]).copy()
    schedD = schedD.dropna(subset=["stop_id_norm", "Scheduled_Departure"]).copy()

    # ---- Align dtypes: time cols tz-aware UTC, by-key as str ----
    if not rtA.empty:
        rtA["RT_Arrival"] = pd.to_datetime(rtA["RT_Arrival"], utc=True, errors="coerce")
    if not rtD.empty:
        rtD["RT_Departure"] = pd.to_datetime(rtD["RT_Departure"], utc=True, errors="coerce")
    if not schedA.empty:
        schedA["Scheduled_Arrival"] = pd.to_datetime(schedA["Scheduled_Arrival"], utc=True, errors="coerce")
    if not schedD.empty:
        schedD["Scheduled_Departure"] = pd.to_datetime(schedD["Scheduled_Departure"], utc=True, errors="coerce")

    for df_ in (rtA, rtD, schedA, schedD):
        if not df_.empty:
            df_["stop_id_norm"] = df_["stop_id_norm"].astype(str)

    def _empty_out():
        return pd.DataFrame(columns=[
            "trip_id", "route_id", "stop_id", "stop_id_norm",
            "Scheduled_Arrival", "RT_Arrival", "Arrival_Delay_Min",
            "Scheduled_Departure", "RT_Departure", "Departure_Delay_Min"
        ])

    # ---- IMPORTANT: sort by the 'on' key ONLY (global monotonic), stable, reset index ----
    if not rtA.empty:
        rtA = rtA.sort_values("RT_Arrival", kind="mergesort").reset_index(drop=True)
    if not rtD.empty:
        rtD = rtD.sort_values("RT_Departure", kind="mergesort").reset_index(drop=True)
    if not schedA.empty:
        schedA = schedA.sort_values("Scheduled_Arrival", kind="mergesort").reset_index(drop=True)
    if not schedD.empty:
        schedD = schedD.sort_values("Scheduled_Departure", kind="mergesort").reset_index(drop=True)

    # ---- Merge arrivals (no suffixes needed since we pre-renamed) ----
    if rtA.empty or schedA.empty:
        matchedA = pd.DataFrame(columns=[
            "stop_id_norm", "RT_Arrival",
            "trip_id_arr", "stop_id_arr", "route_id_arr",
            "trip_id_sched_arr", "stop_id_sched_arr", "route_id_sched_arr",
            "Scheduled_Arrival",
        ])
    else:
        matchedA = pd.merge_asof(
            rtA, schedA,
            left_on="RT_Arrival", right_on="Scheduled_Arrival",
            by="stop_id_norm", direction="nearest", tolerance=tol,
        )

    # ---- Merge departures ----
    if rtD.empty or schedD.empty:
        matchedD = pd.DataFrame(columns=[
            "stop_id_norm", "RT_Departure",
            "trip_id_dep", "stop_id_dep", "route_id_dep",
            "trip_id_sched_dep", "stop_id_sched_dep", "route_id_sched_dep",
            "Scheduled_Departure",
        ])
    else:
        matchedD = pd.merge_asof(
            rtD, schedD,
            left_on="RT_Departure", right_on="Scheduled_Departure",
            by="stop_id_norm", direction="nearest", tolerance=tol,
        )

    # ---- If both are empty, return a typed empty result ----
    if matchedA.empty and matchedD.empty:
        return _empty_out()

    # ---- Build union base keyed by stop_id_norm ----
    base = pd.DataFrame({
        "stop_id_norm": pd.concat(
            [
                matchedA.get("stop_id_norm", pd.Series(dtype=object)),
                matchedD.get("stop_id_norm", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        )
    }).dropna().drop_duplicates()

    # ---- Join matched pieces (pre-renamed columns so no suffix games) ----
    out = base.merge(matchedA, on="stop_id_norm", how="left")
    out = out.merge(matchedD, on="stop_id_norm", how="left", suffixes=("_arr", "_dep"))

    # ---- Ensure expected columns exist ----
    for c in ["trip_id_arr", "stop_id_arr", "route_id_arr", "trip_id_dep", "stop_id_dep", "route_id_dep",
              "Scheduled_Arrival", "RT_Arrival", "Scheduled_Departure", "RT_Departure"]:
        if c not in out.columns:
            out[c] = pd.NA

    # ---- Choose a single trip_id/stop_id/route_id to carry forward (arrival-preferred) ----
    out["trip_id"]  = out["trip_id_arr"].combine_first(out["trip_id_dep"])
    out["route_id"] = out["route_id_arr"].combine_first(out["route_id_dep"])
    out["stop_id"]  = out["stop_id_arr"].combine_first(out["stop_id_dep"])

    # ---- Compute delays safely (handles NaT) ----
    out["Arrival_Delay_Min"] = (
        pd.to_datetime(out["RT_Arrival"], utc=True) - pd.to_datetime(out["Scheduled_Arrival"], utc=True)
    ).dt.total_seconds() / 60.0
    out["Departure_Delay_Min"] = (
        pd.to_datetime(out["RT_Departure"], utc=True) - pd.to_datetime(out["Scheduled_Departure"], utc=True)
    ).dt.total_seconds() / 60.0

    # ---- Filled delays: fall back to scheduled when RT is missing ----
    out["Used_Arrival"] = out["RT_Arrival"].combine_first(out["Scheduled_Arrival"])
    out["Used_Departure"] = out["RT_Departure"].combine_first(out["Scheduled_Departure"])

    out["Arrival_Delay_Min_Filled"] = (
        pd.to_datetime(out["Used_Arrival"], utc=True) - pd.to_datetime(out["Scheduled_Arrival"], utc=True)
    ).dt.total_seconds() / 60.0

    out["Departure_Delay_Min_Filled"] = (
        pd.to_datetime(out["Used_Departure"], utc=True) - pd.to_datetime(out["Scheduled_Departure"], utc=True)
    ).dt.total_seconds() / 60.0

 
    # ---- Final schema ----
    keep = [
        "trip_id", "route_id", "stop_id", "stop_id_norm",
        "Scheduled_Arrival", "RT_Arrival", "Arrival_Delay_Min",
        "Scheduled_Departure", "RT_Departure", "Departure_Delay_Min",
        "Used_Arrival", "Used_Departure",
        "Arrival_Delay_Min_Filled", "Departure_Delay_Min_Filled",
    ]

    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA

    return out[keep]

# --------------- INTERFACE CONSTRUCTION --------------
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

    # Pick best timestamps (RT preferred, else scheduled)
    df["Best_Arrival"]   = df["RT_Arrival"].combine_first(df["Scheduled_Arrival"])
    df["Best_Departure"] = df["RT_Departure"].combine_first(df["Scheduled_Departure"])
    df["Used_Scheduled_Fallback"] = df["RT_Arrival"].isna() | df["RT_Departure"].isna()

    arrivals = df[df["Best_Arrival"].notna()].copy()
    departures = df[df["Best_Departure"].notna()].copy()
    departures["To_Node"] = departures["route_id"].apply(route_to_node)

    out_rows = []

    def _safe_delay_minutes(rt_ts, sched_ts):
        if pd.isna(rt_ts) and pd.isna(sched_ts):
            return pd.NA
        if pd.isna(rt_ts) and pd.notna(sched_ts):
            return 0.0
        if pd.notna(rt_ts) and pd.isna(sched_ts):
            return 0.0
        return (pd.to_datetime(rt_ts, utc=True) - pd.to_datetime(sched_ts, utc=True)).total_seconds() / 60.0

    def _prefer(*vals):
        for v in vals:
            if not pd.isna(v):
                return v
        return pd.NA
        
    for _, a in arrivals.iterrows():
        tgt = a["Target_Node"]
        if not tgt:
            continue

    # localize arrival
    arr_ts_local = pd.Timestamp(a["Best_Arrival"]).tz_convert("America/New_York")
    arr_local_day = arr_ts_local.date()

    # candidate departure timestamps in local tz
    dep_local = departures["Best_Departure"].dt.tz_convert("America/New_York")
    dep_local_day = dep_local.dt.date

    # allow same-day OR next-day within 90 minutes of midnight
    next_day_ok = (
        (dep_local_day == (arr_local_day + pd.Timedelta(days=1)).date()) &
        (dep_local.dt.hour * 60 + dep_local.dt.minute <= 90)
    )

    same_day = dep_local_day == arr_local_day

    mask = (
        (departures["To_Node"] == tgt) &
        (same_day | next_day_ok) &
        (departures["Best_Departure"] >= a["Best_Arrival"]) &
        (departures["Best_Departure"] <= a["Best_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
    )

    cand = departures.loc[mask].sort_values("Best_Departure").head(1)


        iid = f"{a['From_Node']}_{tgt}_{pd.Timestamp(a['Best_Arrival']).tz_convert('UTC').strftime('%Y%m%d_%H%M')}"

        if cand.empty:
            # compute filled arrival delay from a (RT if present else scheduled vs scheduled)
            arr_used = a["RT_Arrival"] if pd.notna(a["RT_Arrival"]) else a["Scheduled_Arrival"]
            arr_delay_filled = _prefer(
                a.get("Arrival_Delay_Min"),
                _safe_delay_minutes(a.get("RT_Arrival"), a.get("Scheduled_Arrival")),
                0.0
            )
            dep_delay_filled = pd.NA  # no departure chosen


            out_rows.append({
                "Interface_ID": iid,
                "From_Node": a["From_Node"],
                "To_Node": tgt,
                "Link_Type": "Subway-Subway",
                "Scheduled_Arrival": a.get("Scheduled_Arrival"),
                "RT_Arrival": a.get("RT_Arrival"),
                "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
                "Arrival_Delay_Min_Filled": arr_delay_filled,
                "Scheduled_Departure": pd.NaT,
                "RT_Departure": pd.NaT,
                "Departure_Delay_Min": pd.NA,
                "Departure_Delay_Min_Filled": pd.NA,
                "Transfer_Gap_Min": pd.NA,
                "Missed_Transfer_Flag": True,
                "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
            })
            continue

        d = cand.iloc[0]

        # compute transfer gap
        gap_min = (pd.Timestamp(d["Best_Departure"]) - pd.Timestamp(a["Best_Arrival"])).total_seconds() / 60.0

        # compute filled delays for arrival and departure
        arr_used = a["RT_Arrival"] if pd.notna(a["RT_Arrival"]) else a["Scheduled_Arrival"]
        dep_used = d["RT_Departure"] if pd.notna(d["RT_Departure"]) else d["Scheduled_Departure"]

        arr_sched = a.get("Scheduled_Arrival")
        dep_sched = d.get("Scheduled_Departure")

        arr_delay_filled = _prefer(
            a.get("Arrival_Delay_Min"),
            _safe_delay_minutes(a.get("RT_Arrival"), a.get("Scheduled_Arrival")),
            0.0
        )

        dep_delay_filled = _prefer(
            d.get("Departure_Delay_Min"),
            _safe_delay_minutes(d.get("RT_Departure"), d.get("Scheduled_Departure")),
            0.0
        )

        # If dep delay is NA or 0 due to missing RT, propagate arrival delay as a proxy
        if (pd.isna(dep_delay_filled) or dep_delay_filled == 0) and pd.notna(arr_delay_filled):
            dep_delay_filled = float(arr_delay_filled)


        out_rows.append({
            "Interface_ID": iid,
            "From_Node": a["From_Node"],
            "To_Node": tgt,
            "Link_Type": "Subway-Subway",
            "Scheduled_Arrival": a.get("Scheduled_Arrival"),
            "RT_Arrival": a.get("RT_Arrival"),
            "Arrival_Delay_Min": a.get("Arrival_Delay_Min"),
            "Arrival_Delay_Min_Filled": arr_delay_filled,
            "Scheduled_Departure": d.get("Scheduled_Departure"),
            "RT_Departure": d.get("RT_Departure"),
            "Departure_Delay_Min": d.get("Departure_Delay_Min"),
            "Departure_Delay_Min_Filled": dep_delay_filled,
            "Transfer_Gap_Min": gap_min,
            "Missed_Transfer_Flag": (gap_min < MISSED_THRESHOLD_MIN),
            "Used_Scheduled_Fallback": bool(a["Used_Scheduled_Fallback"])
        })

    return pd.DataFrame(out_rows)

# ---------------- TIME FEATURES & FIELDS -------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Build a 4-column candidate matrix, each coerced to tz-aware UTC datetimes
    when_candidates = pd.concat(
        [
            pd.to_datetime(df.get("RT_Arrival"), utc=True, errors="coerce"),
            pd.to_datetime(df.get("RT_Departure"), utc=True, errors="coerce"),
            pd.to_datetime(df.get("Scheduled_Arrival"), utc=True, errors="coerce"),
            pd.to_datetime(df.get("Scheduled_Departure"), utc=True, errors="coerce"),
        ],
        axis=1,
    )

    # Take the first non-null across the 4 columns
    when = when_candidates.bfill(axis=1).iloc[:, 0]

    # Convert to local time for feature extraction
    when_local = when.dt.tz_convert("America/New_York")

    df["Day_of_Week"] = when_local.dt.day_name()
    df["Hour_of_Day"] = when_local.dt.hour
    df["Peak_Flag"] = df["Hour_of_Day"].isin(list(AM_PEAK) + list(PM_PEAK))
    df["Time_Period"] = pd.Categorical(
        ["AM peak" if h in AM_PEAK else "PM peak" if h in PM_PEAK else "Off-peak"
         for h in df["Hour_of_Day"]],
        categories=["AM peak", "PM peak", "Off-peak"]
    )
    return df

def add_placeholders_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        for c in [
            "Arrival_Delay_Min","Departure_Delay_Min",
            "Arrival_Delay_Min_Filled","Departure_Delay_Min_Filled",
            "Delay_Chain_Min","Avg_Flow_Volume","Peak_Flow_Volume",
            "Daily_Ridership_Share_%","Delay_Frequency_%","Avg_Delay_Min",
            "Chain_Reaction_Factor","Alt_Path_Available","Criticality_Score",
            "Ped_Count","Stress_Index","External_Pressure","Incident_History",
        ]:
            if c not in df.columns:
                df[c] = pd.NA
        return df

    # Ensure both original and filled delay cols exist
    for col in [
        "Arrival_Delay_Min","Departure_Delay_Min",
        "Arrival_Delay_Min_Filled","Departure_Delay_Min_Filled",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Use *filled* delays when available, else fallback to originals, else 0
    arr = pd.to_numeric(df["Arrival_Delay_Min_Filled"], errors="coerce")
    dep = pd.to_numeric(df["Departure_Delay_Min_Filled"], errors="coerce")

    arr = arr.fillna(pd.to_numeric(df["Arrival_Delay_Min"], errors="coerce"))
    dep = dep.fillna(pd.to_numeric(df["Departure_Delay_Min"], errors="coerce"))

    df["Delay_Chain_Min"] = arr.fillna(0) + dep.fillna(0)

    # Placeholders
    for col in [
        "Avg_Flow_Volume","Peak_Flow_Volume","Daily_Ridership_Share_%",
        "Delay_Frequency_%","Avg_Delay_Min","Chain_Reaction_Factor",
        "Alt_Path_Available","Criticality_Score","Ped_Count","Stress_Index",
        "External_Pressure","Incident_History"
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df

# ----------------------- MAIN ------------------------
def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/penn_stops.json")
    cfg = json.loads(CONFIG_PATH.read_text())
    penn_ids = cfg.get("subway_penn_stops", [])
    if not penn_ids:
        raise RuntimeError("'subway_penn_stops' is empty in config/penn_stops.json")

    # Load inputs
    rt = load_realtime_events()
    service_date = infer_service_date_from_rt(rt)
    print(f"[build_master] Using service_date: {service_date}")

    sched = load_scheduled_at_penn(service_date, penn_ids)

    # Filter static by RT time window (+/- 60min) to reduce noise
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
        print("[build_master] No RT times found; skipping static time window filter.")

    # Diagnostics for RT before merge
    print(f"[build_master] RT non-null counts: "
          f"arr={rt['rt_arrival_utc'].notna().sum()} "
          f"dep={rt['rt_departure_utc'].notna().sum()} "
          f"stop_norm_missing={(rt['stop_id_norm'].isna().sum() if 'stop_id_norm' in rt.columns else 'NA')}")

    # Time-based join
    events_at_penn = join_static_with_rt_time_based(sched, rt, tolerance_min=30)

    # ---------- A) Exact de-dup on events ----------
    event_dedup_keys = [
        "stop_id_norm", "trip_id", "route_id", "stop_id",
        "Scheduled_Arrival", "RT_Arrival",
        "Scheduled_Departure", "RT_Departure",
    ]
    before_evt = len(events_at_penn)
    events_at_penn = events_at_penn.drop_duplicates(subset=event_dedup_keys, keep="first")
    print(f"[build_master] De-dup events (exact): {before_evt} → {len(events_at_penn)}")

    # ---------- B) Optional soft de-dup (rounded to minute) ----------
    # Use this ONLY if you observe near-identical duplicates caused by timestamp jitter
    # evt = events_at_penn.copy()
    # for c in ["RT_Arrival", "RT_Departure", "Scheduled_Arrival", "Scheduled_Departure"]:
    #     evt[c] = pd.to_datetime(evt[c], utc=True, errors="coerce").dt.floor("min")
    # before_soft = len(evt)
    # evt = evt.drop_duplicates(subset=event_dedup_keys, keep="first")
    # print(f"[build_master] De-dup events (rounded to min): {before_soft} → {len(evt)}")
    # events_at_penn = evt

    # Post-join diagnostics
    total_sched = len(sched)
    rt_rows = len(rt)
    evt_rows = len(events_at_penn)
    with_rt_arr = events_at_penn["RT_Arrival"].notna().sum()
    with_rt_dep = events_at_penn["RT_Departure"].notna().sum()
    print(f"[build_master] sched_rows={total_sched}  rt_rows={rt_rows}  time-matched={evt_rows} "
          f"with_RT_Arr={with_rt_arr} with_RT_Dep={with_rt_dep}")

    # Debug preview & save joined events
    required = ["trip_id","stop_id","route_id","Scheduled_Arrival","Scheduled_Departure","RT_Arrival","RT_Departure"]
    missing = [c for c in required if c not in events_at_penn.columns]
    if missing:
        raise RuntimeError(f"Missing columns after join: {missing}. Got: {list(events_at_penn.columns)}")

    print("[build_master] Preview of merged columns:")
    print(events_at_penn[required].head(8))
    events_at_penn.to_csv("data/curated/_debug_events_at_penn.csv", index=False)

    # Build interfaces
    interfaces = build_interfaces_123_ace(events_at_penn)
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders_and_scores(interfaces)

  # ---------- QC Diagnostics (place right before final column order & save) ----------
    print("\n[qc] Interfaces by link:")
    print(interfaces.groupby(["From_Node","To_Node"]).size().sort_values(ascending=False).head(10))

    print("\n[qc] Time buckets:")
    print(interfaces["Time_Period"].value_counts(dropna=False))

    print("\n[qc] Missed-transfer rate:")
    mt_rate = (interfaces["Missed_Transfer_Flag"] == True).mean()
    print(f"Missed_Transfer_Flag rate: {0.0 if pd.isna(mt_rate) else mt_rate:.1%}")

    print("\n[qc] Delay summary (min):")
    for c in ["Arrival_Delay_Min_Filled","Departure_Delay_Min_Filled","Transfer_Gap_Min","Delay_Chain_Min"]:
        s = pd.to_numeric(interfaces.get(c), errors="coerce")
        n = int(s.notna().sum())
    if n == 0:
        print(f"{c}: n=0")
    else:
        print(f"{c}: n={n} mean={s.mean():.2f} p50={s.median():.2f} p90={s.quantile(0.9):.2f}")

    # Gap quantiles (mins)
    g = pd.to_numeric(interfaces.get("Transfer_Gap_Min"), errors="coerce")
    if g.notna().any():
        qs = g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
        print("\n[qc] Transfer_Gap_Min quantiles (10/25/50/75/90):")
        print(qs.to_string())
    else:
        print("\n[qc] Transfer_Gap_Min: n=0")

    # Negative vs positive delays (filled)
    for name in ["Arrival_Delay_Min_Filled", "Departure_Delay_Min_Filled"]:
        s = pd.to_numeric(interfaces.get(name), errors="coerce").dropna()
        if s.empty:
            print(f"[qc] {name}: n=0")
        else:
            neg = int((s < 0).sum()); pos = int((s > 0).sum())
            print(f"[qc] {name}: n={len(s)} neg={neg} pos={pos} mean={s.mean():.2f} p50={s.median():.2f}")


    print("\n[qc] Sample:")
    print(interfaces.head(5)[[
        "Interface_ID","From_Node","To_Node","RT_Arrival","RT_Departure",
        "Arrival_Delay_Min_Filled","Departure_Delay_Min_Filled",
        "Transfer_Gap_Min","Missed_Transfer_Flag"
    ]])
# ---------- End QC Diagnostics ----------

    # ---------- C) De-dup the interface rows ----------
    before_if = len(interfaces)
    interfaces = interfaces.drop_duplicates(subset=[
        "Interface_ID", "From_Node", "To_Node",
        "Scheduled_Arrival", "RT_Arrival",
        "Scheduled_Departure", "RT_Departure",
    ], keep="first")
    print(f"[build_master] De-dup interfaces: {before_if} → {len(interfaces)}")

    # Final event-level schema order
    cols = [
        "Interface_ID","From_Node","To_Node","Link_Type",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Arrival_Delay_Min_Filled",  # new
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min",
        "Departure_Delay_Min_Filled",  # new
        "Transfer_Gap_Min","Missed_Transfer_Flag",
        "Avg_Flow_Volume","Peak_Flow_Volume","Daily_Ridership_Share_%",
        "Delay_Frequency_%","Avg_Delay_Min","Delay_Chain_Min","Chain_Reaction_Factor",
        "Alt_Path_Available","Criticality_Score","Ped_Count","Stress_Index",
        "Time_Period","Day_of_Week","Hour_of_Day","Peak_Flag",
        "External_Pressure","Incident_History"
    ]
    for c in cols:
        if c not in interfaces.columns:
            interfaces[c] = pd.NA
    interfaces = interfaces[cols]

    # Save outputs
    out_csv = CURATED_DIR / "master_interface_dataset.csv"
    out_parq = CURATED_DIR / "master_interface_dataset.parquet"
    interfaces.to_csv(out_csv, index=False)
    interfaces.to_parquet(out_parq, index=False)
    print(f"[build_master] Wrote {len(interfaces)} rows → {out_csv} / {out_parq}")


if __name__ == "__main__":
    main()

