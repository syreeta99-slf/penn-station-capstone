#!/usr/bin/env python3
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from typing import List
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

# Config (env)
STOP_ID = os.getenv("NJT_STOP_ID", "105")  # Penn Station (NJT)
TRIP_UPDATES_URL = os.getenv("NJT_GET_TRIP_UPDATES_URL", "https://raildata.njtransit.com/api/GTFSRT/getTripUpdates")
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")
GDRIVE_DIR = os.getenv("GDRIVE_DIR_NJT", "penn-station/njt")
MASTER_NAME = os.getenv("NJT_MASTER_NAME", "njt_penn_master.csv")

TOKEN_PATH = pathlib.Path.home() / ".njt" / "token.json"
FIELDS = [
    "pull_utc","server_ts","trip_id","route_id","stop_id",
    "arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"
]

def ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def out_name() -> str:
    return f"njt_rt_{ts()}.csv"

def load_token() -> str:
    if not TOKEN_PATH.exists():
        print("[fatal] missing ~/.njt/token.json; run daily token workflow first", file=sys.stderr)
        sys.exit(2)
    try:
        js = json.loads(TOKEN_PATH.read_text())
        tok = js.get("UserToken") or js.get("token") or js.get("access_token")
        if not tok:
            raise ValueError("No token field found")
        return tok
    except Exception as e:
        print(f"[fatal] could not read token: {e}", file=sys.stderr)
        sys.exit(2)

def fetch_trip_updates(user_token: str) -> bytes:
    # NJT expects multipart/form-data with 'token' field for GTFS-RT endpoints
    files = {"token": (None, user_token)}
    r = requests.post(TRIP_UPDATES_URL, files=files, timeout=30)
    if r.status_code == 401:
        print("[warn] 401 Unauthorized with cached token; not reissuing in poll job", file=sys.stderr)
        sys.exit(78)
    r.raise_for_status()
    return r.content  # protobuf bytes

def parse_rows(feed_bytes: bytes, target_stop: str) -> List[dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(feed_bytes)
    rows: List[dict] = []
    server_ts = int(feed.header.timestamp or time.time())
    for ent in feed.entity:
        tu = ent.trip_update
        if not tu:
            continue
        for stu in tu.stop_time_update:
            if stu.stop_id != target_stop:
                continue
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
                "entity_id": ent.id,
            })
    return rows

def write_csv(path: str, rows: List[dict]) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def lsjson(remote_path: str):
    cp = subprocess.run(["rclone","lsjson",remote_path], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return []

def append_to_master(local_new_csv: str) -> None:
    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    exists = lsjson(remote_master)
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        if exists:
            subprocess.check_call(["rclone","copyto", remote_master, local_master])
            with open(local_master, "a", newline="") as out, open(local_new_csv, "r", newline="") as newf:
                reader = csv.reader(newf)
                next(reader, None)  # skip header
                for r in reader:
                    out.write(",".join(str(x) for x in r) + "\n")
        else:
            subprocess.check_call(["cp", local_new_csv, local_master])
        subprocess.check_call(["rclone","copyto", local_master, remote_master])

def main():
    if not GDRIVE_REMOTE:
        print("Missing GDRIVE_REMOTE_NAME env.", file=sys.stderr)
        sys.exit(2)

    token = load_token()
    data = fetch_trip_updates(token)
    rows = parse_rows(data, STOP_ID)

    outpath = os.path.join("data","njt", out_name())
    write_csv(outpath, rows)

    # raw audit copy
    subprocess.run(["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"], check=False)

    append_to_master(outpath)
    print(f"[ok] NJT poll (stop {STOP_ID}) appended to {GDRIVE_DIR}/{MASTER_NAME} with {len(rows)} rows")

if __name__ == "__main__":
    main()
