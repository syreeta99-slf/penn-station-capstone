#!/usr/bin/env python3
"""
pollers/build_master.py
Build the Master Interface Dataset by joining static GTFS (scheduled)
with realtime subway data (Penn Station stops).
"""

import pathlib, zipfile, io, json
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = pathlib.Path("data")
STATIC_DIR = pathlib.Path("gtfs_static")
CURATED_DIR = DATA_DIR / "curated"
REALTIME_DIR = DATA_DIR / "realtime"
CONFIG_FILE = pathlib.Path("config/penn_stops.json")

CURATED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load static GTFS from latest ZIP
# --------------------------------------------------
def load_static_subway():
    zips = sorted(STATIC_DIR.glob("subway_all_*.zip"))
    if not zips:
        raise FileNotFoundError("No subway static ZIP found in gtfs_static/. Run static refresh first.")
    latest = zips[-1]
    print("Using static file:", latest)

    with zipfile.ZipFile(latest, "r") as zf:
        stops = pd.read_csv(io.BytesIO(zf.read("stops.txt")))
        stop_times = pd.read_csv(io.BytesIO(zf.read("stop_times.txt")))
        trips = pd.read_csv(io.BytesIO(zf.read("trips.txt")))

    sched = stop_times.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")
    sched = sched.merge(stops[["stop_id","stop_name"]], on="stop_id", how="left")

    # convert HH:MM:SS to proper datetimes
    def to_dt(x):
        try:
            return datetime.strptime(x, "%H:%M:%S")
        except Exception:
            return pd.NaT

    sched["arr_dt"] = pd.to_datetime(sched["arrival_time"], errors="coerce")
    sched["dep_dt"] = pd.to_datetime(sched["departure_time"], errors="coerce")

    sched = sched.rename(columns={
        "arr_dt": "Scheduled_Arrival",
        "dep_dt": "Scheduled_Departure"
    })
    return sched

# --------------------------------------------------
# Load realtime CSVs
# --------------------------------------------------
def load_realtime():
    csvs = sorted(REALTIME_DIR.glob("subway_rt_*.csv"))
    if not csvs:
        raise FileNotFoundError("No realtime CSVs found in data/realtime/. Run the realtime poller first.")
    latest = csvs[-1]
    print("Using realtime file:", latest)
    df = pd.read_csv(latest, parse_dates=["rt_arrival_utc","rt_departure_utc"])
    return df

# --------------------------------------------------
# Join static + realtime
# --------------------------------------------------
def join_static_with_rt(sched_df, rt_df):
    rt2 = rt_df.rename(columns={
        "rt_arrival_utc": "RT_Arrival",
        "rt_departure_utc": "RT_Departure"
    })
    rt2 = rt2[["trip_id","stop_id","route_id","RT_Arrival","RT_Departure"]]

    df = sched_df.merge(rt2, on=["trip_id","stop_id"], how="left", suffixes=("_sched","_rt"))

    # unify route_id
    if "route_id_rt" in df.columns or "route_id_sched" in df.columns:
        df["route_id"] = df.get("route_id_rt").combine_first(df.get("route_id_sched"))
        for c in ["route_id_rt","route_id_sched"]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

    # delays
    df["Arrival_Delay_Min"] = (df["RT_Arrival"] - df["Scheduled_Arrival"]).dt.total_seconds() / 60.0
    df["Departure_Delay_Min"] = (df["RT_Departure"] - df["Scheduled_Departure"]).dt.total_seconds() / 60.0

    return df

# --------------------------------------------------
# Build interface events between 123 and ACE lines
# --------------------------------------------------
def build_interfaces_123_to_ace(df):
    def route_to_node(route_id):
        if str(route_id) in ["1","2","3"]:
            return "Subway_123"
        elif str(route_id) in ["A","C","E"]:
            return "Subway_ACE"
        else:
            return "Other"

    df["From_Node"] = df["route_id"].apply(route_to_node)

    # Example: simplistic next-departure matching within 30 min
    results = []
    arrivals = df[df["From_Node"]=="Subway_123"].copy()
    departures = df[df["route_id"].isin(["A","C","E"])].copy()

    for _, arr in arrivals.iterrows():
        mask = (departures["Scheduled_Departure"] >= arr["Scheduled_Arrival"]) & \
               (departures["Scheduled_Departure"] <= arr["Scheduled_Arrival"] + timedelta(minutes=30))
        cand = departures[mask].sort_values("Scheduled_Departure").head(1)
        if not cand.empty:
            dep = cand.iloc[0]
            results.append({
                "From_Node": arr["From_Node"],
                "To_Node": "Subway_ACE",
                "Scheduled_Arrival": arr["Scheduled_Arrival"],
                "RT_Arrival": arr["RT_Arrival"],
                "Scheduled_Departure": dep["Scheduled_Departure"],
                "RT_Departure": dep["RT_Departure"],
                "Arrival_Delay_Min": arr["Arrival_Delay_Min"],
                "Departure_Delay_Min": dep["Departure_Delay_Min"],
                "Transfer_Gap_Min": (dep["Scheduled_Departure"] - arr["Scheduled_Arrival"]).total_seconds()/60.0
            })
    return pd.DataFrame(results)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    sched = load_static_subway()
    rt = load_realtime()
    df = join_static_with_rt(sched, rt)

    # Guardrail: check for required columns
    required = [
        "trip_id","stop_id","route_id",
        "Scheduled_Arrival","Scheduled_Departure",
        "RT_Arrival","RT_Departure"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns after join: {missing}. Got: {list(df.columns)}")

    print("Preview of merged events_at_penn:")
    print(df[required].head())

    # Build example interface events
    interfaces = build_interfaces_123_to_ace(df)

    # Save outputs
    out_csv = CURATED_DIR / "master_interface_dataset.csv"
    out_parquet = CURATED_DIR / "master_interface_dataset.parquet"
    interfaces.to_csv(out_csv, index=False)
    interfaces.to_parquet(out_parquet, index=False)

    print("Saved master_interface_dataset with", len(interfaces), "rows")

if __name__ == "__main__":
    main()
