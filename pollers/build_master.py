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
from datetime import datetime, timedelta, timezone
import io, json, zipfile, glob
import pandas as pd
import subprocess, sys

# ----------------------- PATHS -----------------------
DATA_DIR     = Path("data")
REALTIME_DIR = DATA_DIR / "realtime"
CURATED_DIR  = DATA_DIR / "curated"
STATIC_DIR   = Path("gtfs_static")
CONFIG_PATH  = Path("config/penn_stops.json")

CURATED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- PARAMETERS ---------------------
# Transfer matching window & threshold (minutes)
TRANSFER_WINDOW_MIN  = 20       # arrivals → next departure must be within this window
MISSED_THRESHOLD_MIN = 2        # < 2 min gap (or no departure) ⇒ missed/unsafe

# Peak time buckets (local ET)
AM_PEAK = range(6, 10)          # 06–09
PM_PEAK = range(16, 20)         # 16–19

# route_id → node label mapping
ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["4", "5", "6", "7", "S"]),  # adjust as desired
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
    """
    Convert GTFS HH:MM:SS (can exceed 24:00:00) to tz-aware UTC Timestamp on given service_date.
    """
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
        # run the refresher; let it raise if it fails
        subprocess.check_call([sys.executable, "pollers/mta_static_refresh.py"])
        zips = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all*.zip"))
        if not zips:
            raise FileNotFoundError("Static refresh ran but no subway_all_*.zip present in gtfs_static/.")
    return zips[-1]

def read_txt_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)))

def infer_service_date_from_rt(rt_df: pd.DataFrame) -> str:
    """
    Infer a service date from realtime timestamps:
    - merge arrival & departure times
    - convert UTC→America/New_York
    - pick earliest local date (covers overnight service)
    """
    ts = pd.concat([rt_df.get("rt_arrival_utc"), rt_df.get("rt_departure_utc")], ignore_index=True)
    ts = ts.dropna()
    if ts.empty:
        return datetime.utcnow().date().isoformat()
    ts_local = ts.dt.tz_convert("America/New_York")
    return ts_local.dt.date.min().isoformat()

def build_stop_alias_map(stops_df: pd.DataFrame) -> dict:
    """
    Map child stop_id -> parent_station (fallback to itself).
    Use to normalize both static and RT stop_ids before matching.
    """
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
    # widen window to improve matching
    dfs = [pd.read_csv(f) for f in files[-60:]]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure tz-aware UTC
    for col in ["rt_arrival_utc", "rt_departure_utc", "source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Normalize RT stop ids using alias map from latest static
    zpath = latest_static_zip()
    stops = read_txt_from_zip(zpath, "stops.txt")
    alias = build_stop_alias_map(stops)
    df["stop_id_norm"] = df["stop_id"].astype(str).map(alias).fillna(df["stop_id"].astype(str))

    # Deduplicate by (trip_id, normalized stop, minute bucket)
    df["dedupe_key"] = (
        df["trip_id"].astype(str) + "|" +
        df["stop_id_norm"].astype(str) + "|" +
        df["source_minute_utc"].astype(str)
    )
    df = df.drop_duplicates("dedupe_key", keep="last").drop(columns=["dedupe_key"])
    return df

def load_scheduled_at_penn(service_date: str, penn_stop_ids: list) -> pd.DataFrame:
    """
    Read latest static ZIP, return scheduled rows for Penn stop IDs only
    with tz-aware UTC Scheduled_Arrival/Departure, and normalized stop_id.
    """
    zpath = latest_static_zip()
    print(f"[build_master] Using static file: {zpath.name}")

    trips      = read_txt_from_zip(zpath, "trips.txt")
    stop_times = read_txt_from_zip(zpath, "stop_times.txt")
    stops      = read_txt_from_zip(zpath, "stops.txt")

    alias = build_stop_alias_map(stops)
    stop_times["stop_id_norm"] = stop_times["stop_id"].astype(str).map(alias).fillna(stop_times["stop_id"].astype(str))
    penn_norm = {alias.get(str(s), str(s)) for s in penn_stop_ids}

    st = stop_times[stop_times["stop_id_norm"].isin(penn_norm)].copy()
    st["Scheduled_Arrival"]   = st["arrival_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].astype(str).apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    return st[["trip_id", "stop_id", "stop_id_norm", "route_id", "Scheduled_Arrival", "Scheduled_Departure"]]

# -------------------- TIME-BASED MATCH ------------------
def join_static_with_rt_time_based(sched_df: pd.DataFrame, rt_df: pd.DataFrame, tolerance_min: int = 30) -> pd.DataFrame:
    """
    Match RT events to static schedules by stop_id_norm + nearest time within tolerance (minutes).
    Drops rows with null keys before merge_asof to avoid errors.
    """
    tol = pd.Timedelta(minutes=tolerance_min)

    # Prepare RT frames (arrival / departure), keep only rows with non-null keys
    rtA = rt_df[["trip_id","stop_id","stop_id_norm","route_id","rt_arrival_utc"]].rename(
        columns={"rt_arrival_utc":"RT_Arrival"}).copy()
    rtD = rt_df[["trip_id","stop_id","stop_id_norm","route_id","rt_departure_utc"]].rename(
        columns={"rt_departure_utc":"RT_Departure"}).copy()

    # Drop rows with NaT/NaN in merge keys
    rtA = rtA.dropna(subset=["stop_id_norm", "RT_Arrival"]).copy()
    rtD = rtD.dropna(subset=["stop_id_norm", "RT_Departure"]).copy()

    # Prepare schedule frames & drop rows with null keys
    schedA = sched_df.dropna(subset=["stop_id_norm", "Scheduled_Arrival"]).copy()
    schedD = sched_df.dropna(subset=["stop_id_norm", "Scheduled_Departure"]).copy()

    # If either side is empty, short-circuit to an empty result (avoid merge_asof errors)
    def _empty_out():
        return pd.DataFrame(columns=[
            "trip_id","route_id","stop_id","stop_id_norm",
            "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
            "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
        ])

    # Sort as required by merge_asof: first by 'by' key, then by 'on' key
    if not rtA.empty:    rtA = rtA.sort_values(["stop_id_norm","RT_Arrival"])
    if not rtD.empty:    rtD = rtD.sort_values(["stop_id_norm","RT_Departure"])
    if not schedA.empty: schedA = schedA.sort_values(["stop_id_norm","Scheduled_Arrival"])
    if not schedD.empty: schedD = schedD.sort_values(["stop_id_norm","Scheduled_Departure"])

    # Merge arrivals
    if rtA.empty or schedA.empty:
        matchedA = pd.DataFrame(columns=[
            "stop_id_norm","RT_Arrival","trip_id_sched","route_id","stop_id_sched","Scheduled_Arrival"
        ])
    else:
        matchedA = pd.merge_asof(
            rtA, schedA,
            left_on="RT_Arrival", right_on="Scheduled_Arrival",
            by="stop_id_norm", direction="nearest", tolerance=tol,
            suffixes=("_rt","_sched")
        )

    # Merge departures
    if rtD.empty or schedD.empty:
        matchedD = pd.DataFrame(columns=[
            "stop_id_norm","RT_Departure","trip_id_sched","route_id","stop_id_sched","Scheduled_Departure"
        ])
    else:
        matchedD = pd.merge_asof(
            rtD, schedD,
            left_on="RT_Departure", right_on="Scheduled_Departure",
            by="stop_id_norm", direction="nearest", tolerance=tol,
            suffixes=("_rt","_sched")
        )

    # If both are empty, return an empty frame with the expected columns
    if matchedA.empty and matchedD.empty:
        return _empty_out()

    # Build union base keyed by stop_id_norm + available times
    base = pd.DataFrame({
        "stop_id_norm": pd.concat(
            [matchedA.get("stop_id_norm", pd.Series(dtype=object)),
             matchedD.get("stop_id_norm", pd.Series(dtype=object))],
            ignore_index=True
        )
    }).dropna().drop_duplicates()

    # Attach matched columns
    useA_cols = [c for c in ["stop_id_norm","RT_Arrival","trip_id_sched","route_id","stop_id_sched","Scheduled_Arrival"] if c in matchedA.columns]
    useD_cols = [c for c in ["stop_id_norm","RT_Departure","trip_id_sched","route_id","stop_id_sched","Scheduled_Departure"] if c in matchedD.columns]

    out = base
    if useA_cols:
        out = out.merge(matchedA[useA_cols], on="stop_id_norm", how="left")
    else:
        for c in ["RT_Arrival","trip_id_sched","route_id","stop_id_sched","Scheduled_Arrival"]:
            out[c] = pd.NA

    if useD_cols:
        out = out.merge(matchedD[useD_cols], on="stop_id_norm", how="left", suffixes=("_arr","_dep"))
    else:
        for c in ["RT_Departure","trip_id_sched_dep","route_id_dep","stop_id_sched_dep","Scheduled_Departure"]:
            out[c] = pd.NA

    # Unify columns (guard against missing)
    rename_map = {}
    if "trip_id_sched" in out.columns:           rename_map["trip_id_sched"] = "trip_id_arr"
    if "stop_id_sched" in out.columns:           rename_map["stop_id_sched"] = "stop_id_arr"
    if "route_id" in out.columns and "route_id_arr" not in out.columns:
        # Arrival leg route_id column may be simply 'route_id' depending on merge outcome
        rename_map["route_id"] = "route_id_arr"
    if "trip_id_sched_dep" in out.columns:       rename_map["trip_id_sched_dep"] = "trip_id_dep"
    if "stop_id_sched_dep" in out.columns:       rename_map["stop_id_sched_dep"] = "stop_id_dep"
    out = out.rename(columns=rename_map)

    # Ensure expected columns exist
    for c in ["trip_id_arr","stop_id_arr","route_id_arr","trip_id_dep","stop_id_dep","route_id_dep",
              "Scheduled_Arrival","RT_Arrival","Scheduled_Departure","RT_Departure"]:
        if c not in out.columns:
            out[c] = pd.NA

    # Pick a single trip_id/stop_id/route_id to carry forward (arrival-preferred)
    out["trip_id"]  = out["trip_id_arr"].combine_first(out["trip_id_dep"])
    out["route_id"] = out["route_id_arr"].combine_first(out["route_id_dep"])
    out["stop_id"]  = out["stop_id_arr"].combine_first(out["stop_id_dep"])

    # Delays
    out["Arrival_Delay_Min"]   = (pd.to_datetime(out["RT_Arrival"], utc=True)   - pd.to_datetime(out["Scheduled_Arrival"], utc=True)).dt.total_seconds() / 60.0
    out["Departure_Delay_Min"] = (pd.to_datetime(out["RT_Departure"], utc=True) - pd.to_datetime(out["Scheduled_Departure"], utc=True)).dt.total_seconds() / 60.0

    keep = [
        "trip_id","route_id","stop_id","stop_id_norm",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min"
    ]
    return out[keep]

# --------------- INTERFACE CONSTRUCTION --------------
def build_interfaces_123_ace(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build interface events: for each arrival on one node, find the next departure
    on the *other* node within TRANSFER_WINDOW_MIN. Prefer RT times; fall back to Scheduled
    so we still produce rows when RT is missing. Adds Used_Scheduled_Fallback flag.
    """
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    df["From_Node"] = df["route_id"].apply(route_to_node)

    def other(node: str) -> str | None:
        if node == "Subway_123": return "Subway_ACE"
        if node == "Subway_ACE": return "Subway_123"
        return None

    df["Target_Node"] = df["From_Node"].apply(other)

    # Best available times
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

        base_ts = a["Best_Arrival"]
        iid = f"{a['From_Node']}_{tgt}_{pd.Timestamp(base_ts).tz_convert('UTC').strftime('%Y%m%d_%H%M')}"

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

# ---------------- TIME FEATURES & FIELDS -------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    when = df["RT_Arrival"].combine_first(df["RT_Departure"]).combine_first(
        df["Scheduled_Arrival"]).combine_first(df["Scheduled_Departure"])
    when_local = pd.to_datetime(when, utc=True).dt.tz_convert("America/New_York")

    df["Day_of_Week"] = when_local.dt.day_name()
    df["Hour_of_Day"] = when_local.dt.hour
    df["Peak_Flag"]   = df["Hour_of_Day"].isin(list(AM_PEAK) + list(PM_PEAK))
    df["Time_Period"] = pd.Categorical(
        ["AM peak" if h in AM_PEAK else "PM peak" if h in PM_PEAK else "Off-peak"
         for h in df["Hour_of_Day"]],
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

# ----------------------- MAIN ------------------------
def main():
    # Load config with Penn stop_ids
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/penn_stops.json")
    cfg = json.loads(CONFIG_PATH.read_text())
    penn_ids = cfg.get("subway_penn_stops", [])
    if not penn_ids:
        raise RuntimeError("'subway_penn_stops' is empty in config/penn_stops.json")

    # Load realtime (tz-aware UTC) and infer a service date to materialize static HH:MM:SS
    rt = load_realtime_events()
    service_date = infer_service_date_from_rt(rt)
    print(f"[build_master] Using service_date: {service_date}")

    # Load scheduled rows at Penn (tz-aware UTC)
    sched = load_scheduled_at_penn(service_date, penn_ids)

    # Filter static to RT time window (+/- 60 min) to reduce noise
    rt_times = pd.concat([rt.get("rt_arrival_utc"), rt.get("rt_departure_utc")], ignore_index=True).dropna()
    if not rt_times.empty:
        rt_min = rt_times.min()
        rt_max = rt_times.max()
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

    # Time-based match (stop_id_norm + nearest time)
    events_at_penn = join_static_with_rt_time_based(sched, rt, tolerance_min=30)

    # ... any cleaning of rt (renaming cols, normalizing stop ids, etc.)
    print(f"[build_master] RT non-null counts: "
      f"arr={rt['rt_arrival_utc'].notna().sum()} "
      f"dep={rt['rt_departure_utc'].notna().sum()} "
      f"stop_norm_missing={(rt['stop_id_norm'].isna().sum() if 'stop_id_norm' in rt.columns else 'NA')}")

# now run the join
events_at_penn = join_static_with_rt_time_based(sched, rt, tolerance_min=30)


    # Diagnostics
    total_sched = len(sched)
    rt_rows = len(rt)
    evt = len(events_at_penn)
    with_rt_arr = events_at_penn["RT_Arrival"].notna().sum()
    with_rt_dep = events_at_penn["RT_Departure"].notna().sum()
    print(f"[build_master] sched_rows={total_sched}  rt_rows={rt_rows}  time-matched={evt} "
          f"with_RT_Arr={with_rt_arr} with_RT_Dep={with_rt_dep}")

    # Save joined events (debug)
    events_at_penn.to_csv("data/curated/_debug_events_at_penn.csv", index=False)

    # Guardrail
    required = [
        "trip_id", "stop_id", "route_id",
        "Scheduled_Arrival", "Scheduled_Departure",
        "RT_Arrival", "RT_Departure"
    ]
    missing = [c for c in required if c not in events_at_penn.columns]
    if missing:
        raise RuntimeError(f"Missing columns after join: {missing}. Got: {list(events_at_penn.columns)}")

    print("[build_master] Preview of merged columns:")
    print(events_at_penn[required].head(8))

    # Build Subway_123 ↔ Subway_ACE interface events
    interfaces = build_interfaces_123_ace(events_at_penn)

    # Add time buckets & placeholder risk/flow fields
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders_and_scores(interfaces)

    # Final event-level schema order
    cols = [
        "Interface_ID","From_Node","To_Node","Link_Type",
        "Scheduled_Arrival","RT_Arrival","Arrival_Delay_Min",
        "Scheduled_Departure","RT_Departure","Departure_Delay_Min",
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

    # Save
    out_csv = CURATED_DIR / "master_interface_dataset.csv"
    out_parq = CURATED_DIR / "master_interface_dataset.parquet"
    interfaces.to_csv(out_csv, index=False)
    interfaces.to_parquet(out_parq, index=False)

    print(f"[build_master] Wrote {len(interfaces)} rows → {out_csv} / {out_parq}")

if __name__ == "__main__":
    main()
