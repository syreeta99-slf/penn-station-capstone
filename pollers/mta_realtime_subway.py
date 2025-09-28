import json, requests, pandas as pd, pathlib
from datetime import datetime, timezone
from google.transit import gtfs_realtime_pb2

# Realtime URLs (protobuf)
RT_URLS = {
    "nyct_all": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
    "nyct_ace": "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
}

# We'll load Penn Station stop_ids from config/penn_stops.json
def load_penn_stops():
    try:
        with open("config/penn_stops.json","r") as f:
            cfg = json.load(f)
        return set(cfg.get("subway_penn_stops", []))
    except FileNotFoundError:
        # If you haven't created it yet, collect everything (we'll filter later)
        return set()

def fetch_feed(url):
    feed = gtfs_realtime_pb2.FeedMessage()
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    feed.ParseFromString(r.content)
    return feed

def to_rows(feed, penn_set):
    rows = []
    now = datetime.now(timezone.utc)
    minute = now.replace(second=0, microsecond=0).isoformat()
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip_id = tu.trip.trip_id
        route_id = tu.trip.route_id
        for stu in tu.stop_time_update:
            stop_id = stu.stop_id
            # If you haven't filled penn_stops yet, you can temporarily keep all rows
            if penn_set and stop_id not in penn_set:
                continue
            arr_ts = getattr(stu.arrival, "time", None)
            dep_ts = getattr(stu.departure, "time", None)
            arr_iso = datetime.fromtimestamp(arr_ts, tz=timezone.utc).isoformat() if arr_ts else None
            dep_iso = datetime.fromtimestamp(dep_ts, tz=timezone.utc).isoformat() if dep_ts else None
            rows.append({
                "source_minute_utc": minute,
                "trip_id": trip_id,
                "route_id": route_id,
                "stop_id": stop_id,
                "rt_arrival_utc": arr_iso,
                "rt_departure_utc": dep_iso,
            })
    return rows

if __name__ == "__main__":
    penn_set = load_penn_stops()
    frames = []
    for label, url in RT_URLS.items():
        feed = fetch_feed(url)
        frames.append(pd.DataFrame(to_rows(feed, penn_set)))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    outdir = pathlib.Path("data/realtime")
    outdir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        df.to_csv(outdir / f"subway_rt_{ts}.csv", index=False)
        print("Wrote", outdir / f"subway_rt_{ts}.csv")
    else:
        print("No realtime rows (check penn_stops.json or off-peak)")
