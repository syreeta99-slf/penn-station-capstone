import os
import pathlib
import zipfile
import pandas as pd
import numpy as np
import datetime
import pytz

NY_TZ = pytz.timezone("America/New_York")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _utc_to_local_seconds(ts_utc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """UTC -> (local_date, seconds since local midnight).
    Uses tz-aware arithmetic; avoids deprecated .view and dtype issues.
    """
    s = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    s_local = s.dt.tz_convert(NY_TZ)
    local_date = s_local.dt.date
    # Seconds since local midnight using normalize() (keeps timezone)
    secs = (s_local - s_local.dt.normalize()).dt.total_seconds()
    return local_date, secs

# --------------------------------------------------
# NJT static loader
# --------------------------------------------------
def load_njt_static(static_zip: str, penn_stop_ids: list[str]) -> pd.DataFrame:
    """Load NJT static GTFS stop_times filtered to Penn Station."""
    with zipfile.ZipFile(static_zip, "r") as zf:
        with zf.open("stop_times.txt") as f:
            stop_times = pd.read_csv(f)
        with zf.open("trips.txt") as f:
            trips = pd.read_csv(f)

    # Merge stop_times with trips for route_id
    merged = stop_times.merge(trips, on="trip_id", how="left")
    merged = merged[merged["stop_id"].astype(str).isin(penn_stop_ids)]

    # Convert HH:MM:SS to seconds past midnight
    def _hms_to_sec(x):
        try:
            h, m, s = map(int, str(x).split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return np.nan

    merged["arr_sec"] = merged["arrival_time"].apply(_hms_to_sec)
    merged["dep_sec"] = merged["departure_time"].apply(_hms_to_sec)
    merged["sched_key_sec"] = merged["arr_sec"]

    return merged[["trip_id", "route_id", "stop_id", "sched_key_sec", "arrival_time", "departure_time"]]

# --------------------------------------------------
# Join realtime with static schedule
# --------------------------------------------------
def join_njt_rt_to_schedule(rt_df: pd.DataFrame, sched: pd.DataFrame, tolerance_min: int = 45) -> pd.DataFrame:
    """Join NJT realtime arrivals to static schedule using nearest-time match."""
    if rt_df.empty or sched.empty:
        return pd.DataFrame()

    # Expand rt_df with join keys
    rt_df = rt_df.copy()
    rt_df["route_id"] = rt_df["route_id"].astype(str)
    rt_df["stop_id"] = rt_df["stop_id"].astype(str)

    # Convert tolerance to numeric seconds
    tol_sec = float(tolerance_min * 60)

    # Ensure float dtype for merge_asof
    rt_df = rt_df.sort_values("rt_arr_sec").astype({"rt_arr_sec": "float64"})
    sched = sched.sort_values("sched_key_sec").astype({"sched_key_sec": "float64"})

    mergedA = pd.merge_asof(
        rt_df,
        sched,
        by=["route_id", "stop_id"],
        left_on="rt_arr_sec", right_on="sched_key_sec",
        direction="nearest", tolerance=tol_sec
    )

    return mergedA

# --------------------------------------------------
# Build NJT interfaces
# --------------------------------------------------
def build_njt_interfaces():
    """Stub for NJT interface build (replace with full pipeline logic)."""
    # Example structure
    rt = pd.DataFrame({
        "trip_id": ["t1"],
        "route_id": ["13"],
        "stop_id": ["105"],
        "rt_arr_sec": [36000],  # 10:00 AM
    })

    static_zip = "data/njt_static/njt_rail_static.zip"
    sched = load_njt_static(static_zip, ["105"])

    matched = join_njt_rt_to_schedule(rt, sched)
    print("[njt][sched] matched", len(matched), "rows")
    return matched

# --------------------------------------------------
# Main entrypoint
# --------------------------------------------------
def main():
    print("[t0] starting build_master")
    njt_ifaces = build_njt_interfaces()
    print(njt_ifaces.head())

if __name__ == "__main__":
    main()
