import json, zipfile, io
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

# ---------- CONFIG (edit as needed) ----------
CURATED_DIR = Path("data/curated")
REALTIME_DIR = Path("data/realtime")
STATIC_DIR = Path("gtfs_static")

# If multiple static ZIPs exist, we’ll use the newest by date in filename
STATIC_SUBWAY_ZIPS = sorted(STATIC_DIR.glob("subway_all_*.zip")) or sorted(STATIC_DIR.glob("subway_all.zip*"))
SERVICE_DATE = None  # set like "2025-10-15" to force; else pick "today" in NYC
TRANSFER_WINDOW_MIN = 20             # arrivals → next departure window
MISSED_THRESHOLD_MIN = 2             # < 2 min is considered a missed/unsafe transfer
AM_PEAK = range(6, 10)               # 06–09
PM_PEAK = range(16, 20)              # 16–19

# Map GTFS route_id → node labels you’ve defined
ROUTE_GROUPS = {
    "Subway_123": set(list("123") + ["S"]),  # include S as you prefer
    "Subway_ACE": set(list("ACE")),
}
# --------------------------------------------

def load_config():
    with open("config/penn_stops.json","r") as f:
        return json.load(f)

def nyc_today_yyyy_mm_dd():
    # Runner is UTC; convert to America/New_York if you want stricter alignment.
    # For most cases, using UTC "today" is OK when we’re just building a sample day.
    return datetime.utcnow().date().isoformat()

def choose_static_zip():
    if not STATIC_SUBWAY_ZIPS:
        raise FileNotFoundError("No subway static ZIPs found in gtfs_static/. Run the static refresh job first.")
    return STATIC_SUBWAY_ZIPS[-1]

def read_txt_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, "r") as zf:
        return pd.read_csv(io.BytesIO(zf.read(member)))

def parse_gtfs_time_to_dt(hms: str, service_date: str) -> datetime:
    # GTFS times can exceed 24:00:00; handle wrap
    h, m, s = map(int, hms.split(":"))
    extra_days, h = divmod(h, 24)
    base = datetime.fromisoformat(service_date).replace(tzinfo=timezone.utc)
    return base + timedelta(days=extra_days, hours=h, minutes=m, seconds=s)

def route_to_node(route_id: str) -> str:
    if route_id is None or pd.isna(route_id):
        return "Subway_Unknown"
    for node, routes in ROUTE_GROUPS.items():
        if route_id in routes:
            return node
    return "Subway_Other"

def load_latest_realtime():
    files = sorted(REALTIME_DIR.glob("subway_rt_*.csv"))
    if not files:
        return pd.DataFrame()
    # Use last few pulls to be safe
    dfs = [pd.read_csv(f) for f in files[-12:]]
    df = pd.concat(dfs, ignore_index=True)
    # Dedup: keep latest per (trip_id, stop_id, source_minute_utc)
    df["key"] = df["trip_id"].astype(str) + "|" + df["stop_id"].astype(str) + "|" + df["source_minute_utc"].astype(str)
    df = df.drop_duplicates("key", keep="last").drop(columns=["key"])
    # Parse times
    for col in ["rt_arrival_utc","rt_departure_utc","source_minute_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def build_scheduled_at_penn(zip_path, penn_stop_ids, service_date):
    stops = read_txt_from_zip(zip_path, "stops.txt")
    trips = read_txt_from_zip(zip_path, "trips.txt")
    stop_times = read_txt_from_zip(zip_path, "stop_times.txt")

    st = stop_times[stop_times["stop_id"].isin(set(penn_stop_ids))].copy()
    # parse scheduled times for a specific service date
    st["Scheduled_Arrival"] = st["arrival_time"].apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st["Scheduled_Departure"] = st["departure_time"].apply(lambda x: parse_gtfs_time_to_dt(x, service_date))
    st = st.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")
    st["From_Node"] = st["route_id"].apply(route_to_node)  # we’ll repurpose later
    st["To_Node"] = st["From_Node"]
    st = st[["trip_id","stop_id","route_id","Scheduled_Arrival","Scheduled_Departure","From_Node","To_Node"]]
    return st

def compute_delays_at_penn(scheduled_df, realtime_df):
    # Join RT to scheduled by (trip_id, stop_id). RT may have arrival or departure or both.
    rt = realtime_df.rename(columns={
        "rt_arrival_utc":"RT_Arrival",
        "rt_departure_utc":"RT_Departure"
    })[["trip_id","stop_id","route_id","RT_Arrival","RT_Departure"]].copy()

    # If route_id missing in RT, fill from scheduled
    df = scheduled_df.merge(rt, on=["trip_id","stop_id"], how="left", suffixes=("",""))

    # Compute delays (minutes)
    df["Arrival_Delay_Min"] = (df["RT_Arrival"] - df["Scheduled_Arrival"]).dt.total_seconds() / 60.0
    df["Departure_Delay_Min"] = (df["RT_Departure"] - df["Scheduled_Departure"]).dt.total_seconds() / 60.0

    # Keep rows that have at least an RT_Arrival or RT_Departure
    has_rt = df["RT_Arrival"].notna() | df["RT_Departure"].notna()
    return df.loc[has_rt].copy()

def build_interfaces_subway_to_subway(events_df):
    """
    Build interface events between Subway_123 and Subway_ACE.
    For each *arrival* from either node, find the next *departure* on the other node within window.
    """
    df = events_df.copy()
    df["From_Node"] = df["route_id"].apply(route_to_node)
    # Label the counterpart node for transfer pairing
    def other_node(n):
        if n == "Subway_123": return "Subway_ACE"
        if n == "Subway_ACE": return "Subway_123"
        return None
    df["Target_Node"] = df["From_Node"].apply(other_node)

    # ARRIVALS = rows with RT_Arrival
    arrivals = df[df["RT_Arrival"].notna()].copy()
    # DEPARTURES = rows with RT_Departure
    departures = df[df["RT_Departure"].notna()].copy()
    departures["To_Node"] = departures["route_id"].apply(route_to_node)

    rows = []
    for idx, a in arrivals.iterrows():
        if not a["Target_Node"]:
            continue
        # find next departure at Penn on the target node within window
        mask = (departures["To_Node"] == a["Target_Node"]) & \
               (departures["RT_Departure"] >= a["RT_Arrival"]) & \
               (departures["RT_Departure"] <= a["RT_Arrival"] + pd.Timedelta(minutes=TRANSFER_WINDOW_MIN))
        cand = departures.loc[mask].sort_values("RT_Departure").head(1)
        if cand.empty:
            # record a "missed" with no departure found
            rows.append({
                "Interface_ID": f"{a['From_Node']}_{a['Target_Node']}_{a['RT_Arrival'].strftime('%Y%m%d_%H%M')}",
                "From_Node": a["From_Node"],
                "To_Node": a["Target_Node"],
                "Link_Type": "Subway-Subway",
                "Scheduled_Arrival": a["Scheduled_Arrival"],
                "RT_Arrival": a["RT_Arrival"],
                "Arrival_Delay_Min": a["Arrival_Delay_Min"],
                "Scheduled_Departure": pd.NaT,
                "RT_Departure": pd.NaT,
                "Departure_Delay_Min": pd.NA,
                "Transfer_Gap_Min": pd.NA,
                "Missed_Transfer_Flag": True
            })
            continue

        d = cand.iloc[0]
        gap = (d["RT_Departure"] - a["RT_Arrival"]).total_seconds()/60.0
        rows.append({
            "Interface_ID": f"{a['From_Node']}_{a['Target_Node']}_{a['RT_Arrival'].strftime('%Y%m%d_%H%M')}",
            "From_Node": a["From_Node"],
            "To_Node": a["Target_Node"],
            "Link_Type": "Subway-Subway",
            "Scheduled_Arrival": a["Scheduled_Arrival"],
            "RT_Arrival": a["RT_Arrival"],
            "Arrival_Delay_Min": a["Arrival_Delay_Min"],
            "Scheduled_Departure": d["Scheduled_Departure"],
            "RT_Departure": d["RT_Departure"],
            "Departure_Delay_Min": d["Departure_Delay_Min"],
            "Transfer_Gap_Min": gap,
            "Missed_Transfer_Flag": (gap < MISSED_THRESHOLD_MIN)
        })

    return pd.DataFrame(rows)

def add_time_features(df):
    if df.empty: return df
    # Use arrival (or departure if arrival missing) as event time
    when = df["RT_Arrival"].combine_first(df["RT_Departure"])
    # fall back to scheduled if needed
    when = when.combine_first(df["Scheduled_Arrival"]).combine_first(df["Scheduled_Departure"])
    # Day/Hour in ET-ish (we’re using UTC; for precision, convert to America/New_York)
    dt = when.dt.tz_convert("America/New_York")
    df["Day_of_Week"] = dt.dt.day_name()
    df["Hour_of_Day"] = dt.dt.hour
    df["Peak_Flag"] = df["Hour_of_Day"].isin(list(AM_PEAK) + list(PM_PEAK))
    df["Time_Period"] = pd.Categorical(
        pd.Series(["AM peak" if h in AM_PEAK else "PM peak" if h in PM_PEAK else "Off-peak" for h in df["Hour_of_Day"]]),
        categories=["AM peak","PM peak","Off-peak"]
    )
    return df

def add_placeholders_and_scores(df):
    # Placeholders to be backfilled later with counts/aggregations
    df["Avg_Flow_Volume"] = pd.NA
    df["Peak_Flow_Volume"] = pd.NA
    df["Daily_Ridership_Share_%"] = pd.NA
    # Frequency & averages can be computed as aggregations by (From_Node, To_Node, Time_Period)
    # but we leave NA in the event-level table and compute in a separate rollup
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
    cfg = load_config()
    subway_penn_ids = cfg.get("subway_penn_stops", [])
    if not subway_penn_ids:
        raise RuntimeError("config/penn_stops.json has no 'subway_penn_stops'. Fill this first.")

    # Pick service date (today by default)
    service_date = SERVICE_DATE or nyc_today_yyyy_mm_dd()

    # Load static + realtime
    zip_path = choose_static_zip()
    rt = load_latest_realtime()
    if rt.empty:
        print("No realtime CSVs found in data/realtime/ yet. Run the realtime poller first.")
        return

    # Build scheduled arrivals/departures at Penn, then compute delays with RT
    sched = build_scheduled_at_penn(zip_path, subway_penn_ids, service_date)
    events = compute_delays_at_penn(sched, rt)

    # Build interface rows for Subway_123 <-> Subway_ACE
    interfaces = build_interfaces_subway_to_subway(events)

    # Add time features & placeholders
    interfaces = add_time_features(interfaces)
    interfaces = add_placeholders_and_scores(interfaces)

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    out_parq = CURATED_DIR / "master_interface_dataset.parquet"
    out_csv  = CURATED_DIR / "master_interface_dataset.csv"
    interfaces.to_parquet(out_parq, index=False)
    interfaces.to_csv(out_csv, index=False)
    print(f"Wrote {len(interfaces)} interface rows → {out_csv} / {out_parq}")

if __name__ == "__main__":
    main()
