import json, zipfile, io
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta, timezone
import pandas as pd

# ---------------- CONFIG ----------------
CURATED_DIR   = Path("data/curated")
REALTIME_DIR  = Path("data/realtime")
STATIC_DIR    = Path("gtfs_static")
CONFIG_PATH   = Path("config/penn_stops.json")

# Interface / timing config
TRANSFER_WINDOW_MIN  = 20       # arrivals → next departure must be within this window
MISSED_THRESHOLD_MIN = 2        # if gap < 2 min (or none found) => missed/unsafe

# Time buckets (America/New_York)
AM_PEAK = range(6, 10)          # 06–09
PM_PEAK = range(16, 20)         # 16–19

# Map route_id → node label
ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["4","5","6","7","S"]),  # include lines you want in this node
    "Subway_ACE": set(list("ACE")),
}
# ---------------------------------------

def route_to_node(route_id: str) -> str:
    if pd.isna(route_id) or route_id is None:
        return "Subway_Unknown"
    for node, routes in ROUTE_GROUPS.items():
        if str(route_id) in routes:
            return node
    return "Subway_Other"

def latest_static_zip():
    zips = sorted(STATIC_DIR.glob("subway_all_*.zip"))
    if not zips:
        zips = sorted(STATIC_DIR.glob("subway_all*.zip"))
    if not zips:
        raise FileNotFoundError("No static ZIPs in gtfs_static/. Run the static refresh first.")
    return zips[-1]

def read_txt_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, "r") as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)))

def parse_gtfs_time_to_dt(hms: str, service_date: str) -> datetime:
    # GTFS times can exceed 24:00:00 → wrap to next day(s). Return UTC datetime.
    h, m, s = map(int, hms.split(":"))
    extra_days, h = divmod(h, 24)
    base = datetime.fromisoformat(service_date).replace(tzinfo=timezone.utc)
    return base + timedelta(days=extra_days, hours=h, minutes=m, seconds=s)

def infer_service_date_from_zip(zip_path: Path) -> str:
    # Try: subway_all_YYYYMMDD.zip → YYYY-MM-DD; else today (UTC)
    name = zip_path.stem
    for token in name.split("_"):
        if token.isdigit() and len(token) == 8:
            return f"{token[:4]}-{token[4:6]}-{token[6:]}"
    return datetime.utcnow().date().isoformat()

def load_realtime_events():
    files = sorted(REALTIME_DIR.glob("subway_rt_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files[-12:]]  # last few pulls
    df = pd.concat(dfs, ignore_index=True)

    # Dedup by snapshot minute per trip+stop
    df["key"] = df["trip_id"].astype(str) + "|" + df["stop_id"].astype(str) + "|" + df["source_minute_utc"].astype(str)
    df = df.drop_duplicates("key", keep="last").drop(columns=["key"])

    for col in ["rt_arrival_utc","rt_departure_utc","source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def build_scheduled_at_penn(zip_path, penn_stop_ids, service_date):
    trips = read_txt_from_zip(zip_path, "trips.txt")
    stop_times = read_txt_from_zip(zip_path, "stop_times.txt")

    st = stop_times[stop_times["stop_id"].isin(set(penn_stop_ids))].copy()
    st["Scheduled_Arrival"]   = st["arrival_time"].apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")
    st["node"] = st["route_id"].apply(route_to_node)
    return st[["trip_id","stop_id","route_id","node","Scheduled_Arrival","Scheduled_Departure"]]

def join_static_with_rt(sched_df, rt_df):
    # rename RT columns to target names
    rt2 = rt_df.rename(columns={"rt_arrival_utc":"RT_Arrival","rt_departure_utc":"RT_Departure"})
    rt2 = rt2[["trip_id","stop_id","route_id","RT_Arrival","RT_Departure"]]

    df = sched_df.merge(rt2, on=["trip_id","stop_id"], how="left")

    # per-stop delays
    df["Arrival_Delay_Min"]   = (df["RT_Arrival"]   - df["Scheduled_Arrival"]).dt.total_seconds() / 60.0
    df["Departure_Delay_Min"] = (df["RT_Departure"] - df["Scheduled_Departure"]).dt.total_seconds() / 60.0
    return df

def build_interfaces_123_to_ace(event_stops_df):
    """
    For each arrival at Penn on one node, find the next departure at Penn on the other node
    within TRANSFER_WINDOW_MIN. Produces 1 row per transfer opportunity.
    """
    df = event_stops_df.copy()
    df["From_Node"] = df["route_id"].apply(route_to_node)

    def other(node):
        return "Subway_ACE" if node == "Subway_123" else ("Subway_123" if node == "Subway_ACE" else None)
    df["Target_Node"] = df["From_Node"].apply(other)

    arrivals   = df[df["RT_Arrival"].notna()].copy()
    departures = df[df["RT_Departure"].notna()].copy()
    departures["To_Node"] = departures["route_id"].apply(route_to_node)

    out = []
    for _, a in arrivals.iterrows():
        tgt = a["Target_Node"]
        if not tgt: 
            continue
        mask = (
            (departures["To_Node"] == tgt) &
            (departures["RT_Departure"] >= a["RT_Arrival"]) &
            (departures["RT_Departure"] <= a["RT_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
        )
        cand = departures.loc[mask].sort_values("RT_Departure").head(1)

        interface_id = f"{a['From_Node']}_{tgt}_{a['RT_Arrival'].strftime('%Y%m%d_%H%M')}" if pd.notna(a["RT_Arrival"]) else \
                       f"{a['From_Node']}_{tgt}_{a['Scheduled_Arrival'].strftime('%Y%m%d_%H%M')}"

        if cand.empty:
            out.append({
                "Interface_ID": interface_id,
                "From_Node": a["From_Node"],
                "To_Node": tgt,
                "Link_Type": "Subway-Subway",
                "Scheduled_Arrival": a["Scheduled_Arrival"],
                "RT_Arrival": a["RT_Arrival"],
                "Arrival_Delay_Min": a["Arrival_Delay_Min"],
                "Scheduled_Departure": pd.NaT,
                "RT_Departure": pd.NaT,
                "Departure_Delay_Min": pd.NA,
                "Transfer_Gap_Min": pd.NA,
                "Missed_Transfer_Flag": True,
            })
            continue

        d = cand.iloc[0]
        gap = (d["RT_Departure"] - a["RT_Arrival"]).total_seconds()/60.0
        out.append({
            "Interface_ID": interface_id,
            "From_Node": a["From_Node"],
            "To_Node": tgt,
            "Link_Type": "Subway-Subway",
            "Scheduled_Arrival": a["Scheduled_Arrival"],
            "RT_Arrival": a["RT_Arrival"],
            "Arrival_Delay_Min": a["Arrival_Delay_Min"],
            "Scheduled_Departure": d["Scheduled_Departure"],
            "RT_Departure": d["RT_Departure"],
            "Departure_Delay_Min": d["Departure_Delay_Min"],
            "Transfer_Gap_Min": gap,
            "Missed_Transfer_Flag": (gap < MISSED_THRESHOLD_MIN),
        })
    return pd.DataFrame(out)

def add_time_features(df):
    if df.empty: return df
    # event time preference order
    when = df["RT_Arrival"].combine_first(df["RT_Departure"]).combine_first(
        df["Scheduled_Arrival"]).combine_first(df["Scheduled_Departure"])
    # Convert to America/New_York for labeling
    when_local = when.dt.tz_convert("America/New_York")
    df["Day_of_Week"] = when_local.dt.day_name()
    df["Hour_of_Day"] = when_local.dt.hour
    df["Peak_Flag"]   = df["Hour_of_Day"].isin(list(AM_PEAK) + list(PM_PEAK))
    df["Time_Period"] = pd.Categorical(
        ["AM peak" if h in AM_PEAK else "PM peak" if h in PM_PEAK else "Off-peak" for h in df["Hour_of_Day"]],
        categories=["AM peak","PM peak","Off-peak"]
    )
    return df

def add_placeholders(df):
    # You will backfill these later from counts/alerts/capacity models
    df["Avg_Flow_Volume"] = pd.NA
    df["Peak_Flow_Volume"] = pd.NA
    df["Daily_Ridership_Share_%"] = pd.NA
    df["Delay_Frequency_%"] = pd.NA
    df["Avg_Delay_Min"] = pd.NA
    df["Delay_Chain_Min"] = (df["Arrival_Delay_Min"].fillna(0) + df["Departure_Delay_Min"].fillna(0))
    df["Chain_Reaction_Factor"] = pd.NA
    df["Alt_Path_Available"] = pd.NA
    df["Criticality_Score"] = pd.NA
    df["Ped_Count"] = pd.NA
    df["Stress_Index"] = pd.NA
    df["External_Pressure"] = pd.NA
    df["Incident_History"] = pd.NA
    return df

def main():
    # Config + inputs
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config/penn_stops.json")
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    penn_ids = cfg.get("subway_penn_stops", [])
    if not penn_ids:
        raise RuntimeError("'subway_penn_stops' is empty in config/penn_stops.json")

    rt = load_realtime_events()
    if rt.empty:
        print("No realtime CSVs yet (data/realtime). Run the poller first.")
        return

    zip_path = latest_static_zip()
    service_date = infer_service_date_from_zip(zip_path)

    # Scheduled at Penn + join with RT for delays
    sched = build_scheduled_at_penn(zip_path, penn_ids, service_date)
    events_at_penn = join_static_with_rt(sched, rt)

    # Build Subway_123 <-> Subway_ACE transfer interfaces
    interfaces = build_interfaces_123_to_ace(events_at_penn)

    # Time features + placeholders
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders(interfaces)

    # Order columns (event-level schema)
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

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    interfaces.to_csv(CURATED_DIR / "master_interface_dataset.csv", index=False)
    interfaces.to_parquet(CURATED_DIR / "master_interface_dataset.parquet", index=False)
    print(f"Wrote {len(interfaces)} interface rows → data/curated/master_interface_dataset.*")

if __name__ == "__main__":
    main()
