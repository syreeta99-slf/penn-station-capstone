#!/usr/bin/env python3
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from typing import Set, List
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

# --- ENV ---
LIRR_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("LIRR_STOP_IDS","").split(",") if s.strip()}  # e.g., "237"
MTA_API_KEY = os.getenv("MTA_API_KEY")  # optional but recommended
LIRR_FEED_URL = os.getenv("LIRR_FEED_URL", "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/lirr%2Fgtfs-lirr")

GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")  # required
GDRIVE_DIR = os.getenv("GDRIVE_DIR_LIRR", "penn-station/lirr")
MASTER_NAME = os.getenv("LIRR_MASTER_NAME", "lirr_penn_master.csv")

FIELDS = ["pull_utc","server_ts","trip_id","route_id","stop_id","arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"]

def ts(): return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
def out_name(): return f"lirr_rt_{ts()}.csv"

def fetch(url: str):
    headers = {"x-api-key": MTA_API_KEY} if MTA_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def rows_from_feed(feed, stops: Set[str]) -> List[dict]:
    rows = []
    server_ts = int(feed.header.timestamp or time.time())
    for e in feed.entity:
        # Prefer TripUpdate if present; vehicle positions don't include stop_time_updates
        tu = e.trip_update if e.HasField("trip_update") else None
        if not tu:
            continue
        for stu in tu.stop_time_update:
            if stu.stop_id not in stops:
                continue
            arr = getattr(stu.arrival, "time", None) or None
            dep = getattr(stu.departure, "time", None) or None
            delay = getattr(stu.arrival, "delay", None) or getattr(stu.departure, "delay", None) or None
            rows.append({
                "pull_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "server_ts": server_ts,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,  # For LIRR this may be blank; keeping column for schema parity
                "stop_id": stu.stop_id,
                "arrival_time": arr,
                "departure_time": dep,
                "delay_sec": delay,
                "schedule_relationship": tu.trip.schedule_relationship,
                "entity_id": e.id,
            })
    return rows

def write_csv(path, rows):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def lsjson(remote):
    cp = subprocess.run(["rclone","lsjson",remote], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    return json.loads(cp.stdout)

def append_to_master(local_new_csv):
    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    exists = lsjson(remote_master)
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        if exists:
            subprocess.check_call(["rclone","copyto", remote_master, local_master])
            with open(local_master, "a", newline="") as out, open(local_new_csv, "r", newline="") as newf:
                reader = csv.reader(newf); next(reader, None)  # skip header
                for r in reader:
                    out.write(",".join(str(x) for x in r) + "\n")
        else:
            subprocess.check_call(["cp", local_new_csv, local_master])
        subprocess.check_call(["rclone","copyto", local_master, remote_master])

def main():
    if not LIRR_STOP_IDS:
        print("LIRR_STOP_IDS is required (e.g., '237').", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    try:
        feed = fetch(LIRR_FEED_URL)
        rows = rows_from_feed(feed, LIRR_STOP_IDS)
    except Exception as e:
        print("[warn] fetch/parse failed:", e, file=sys.stderr)
        rows = []

    outpath = os.path.join("data","lirr", out_name())
    write_csv(outpath, rows)
    # raw drop
    subprocess.run(["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"], check=False)
    # append to master
    append_to_master(outpath)
    print("LIRR poll appended:", f"{GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__":
    main()
