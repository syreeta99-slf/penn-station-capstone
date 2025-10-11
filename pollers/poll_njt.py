#!/usr/bin/env python3
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

NJT_RT_URL = os.getenv("NJT_RT_URL")
NJT_API_TOKEN = os.getenv("NJT_API_TOKEN")
NJT_USERNAME = os.getenv("NJT_USERNAME")
NJT_PASSWORD = os.getenv("NJT_PASSWORD")
TARGET_STOP_ID = os.getenv("NJT_STOP_ID", "105")
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")
GDRIVE_DIR = os.getenv("GDRIVE_DIR_NJT", "penn-station/njt")
MASTER_NAME = os.getenv("NJT_MASTER_NAME", "njt_penn_master.csv")

FIELDS = ["pull_utc","server_ts","trip_id","route_id","stop_id","arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"]

def ts() -> str: return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
def out_name() -> str: return f"njt_rt_{ts()}.csv"

def fetch_feed():
    headers = {}
    auth = None
    if NJT_API_TOKEN:
        headers["Authorization"] = f"Bearer {NJT_API_TOKEN}"
    elif NJT_USERNAME and NJT_PASSWORD:
        auth = (NJT_USERNAME, NJT_PASSWORD)  # HTTP Basic
    r = requests.get(NJT_RT_URL, headers=headers, auth=auth, timeout=30)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def rows_for_stop(feed, stop_id):
    rows = []
    server_ts = int(feed.header.timestamp or time.time())
    for e in feed.entity:
        tu = e.trip_update
        if not tu: continue
        for stu in tu.stop_time_update:
            if stu.stop_id != stop_id: continue
            arr = getattr(stu.arrival, "time", None) or None
            dep = getattr(stu.departure, "time", None) or None
            delay = getattr(stu.arrival, "delay", None) or getattr(stu.departure, "delay", None) or None
            rows.append({
                "pull_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                "server_ts": server_ts,
                "trip_id": tu.trip.trip_id,
                "route_id": tu.trip.route_id,
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
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows: w.writerow(r)

def lsjson(remote):
    cp = subprocess.run(["rclone","lsjson",remote], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip(): return []
    return json.loads(cp.stdout)

def append_to_master(local_new_csv):
    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    exists = lsjson(remote_master)
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        if exists:
            subprocess.check_call(["rclone","copyto", remote_master, local_master])
            with open(local_master, "a", newline="") as out, open(local_new_csv, "r", newline="") as newf:
                reader = csv.reader(newf); next(reader, None)
                for r in reader: out.write(",".join(str(x) for x in r) + "\n")
        else:
            subprocess.check_call(["cp", local_new_csv, local_master])
        subprocess.check_call(["rclone","copyto", local_master, remote_master])

def main():
    if not (NJT_RT_URL and GDRIVE_REMOTE):
        print("Missing NJT_RT_URL or GDRIVE_REMOTE_NAME", file=sys.stderr); sys.exit(2)
    try:
        feed = fetch_feed()
        rows = rows_for_stop(feed, TARGET_STOP_ID)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); rows = []
    outpath = os.path.join("data","njt", out_name()); write_csv(outpath, rows)
    subprocess.run(["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"], check=False)
    append_to_master(outpath)
    print("NJT poll appended:", f"{GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__": main()
