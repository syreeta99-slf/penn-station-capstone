#!/usr/bin/env python3
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, requests
from typing import Set, List
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings protobuf

SUBWAY_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("SUBWAY_STOP_IDS","").split(",") if s.strip()}
MTA_API_KEY = os.getenv("MTA_API_KEY")  # optional
SUBWAY_FEEDS = json.loads(os.getenv("SUBWAY_FEEDS_JSON", '["https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs","https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"]'))
SUBSYSTEM_TAG = "subway"
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")
GDRIVE_DIR = os.getenv("GDRIVE_DIR_SUBWAY", "penn-station/subway")
MASTER_NAME = os.getenv("SUBWAY_MASTER_NAME", "subway_penn_master.csv")

FIELDS = ["pull_utc","server_ts","trip_id","route_id","stop_id","arrival_time","departure_time","delay_sec","schedule_relationship","entity_id"]

def get_event_epoch(row: dict):
    for k in ("arrival_time","departure_time","event_epoch"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                pass
    for k in ("server_ts","pull_utc"):
        v = row.get(k)
        if v is not None and str(v) != "":
            try:
                return float(v)
            except Exception:
                try:
                    return pd.to_datetime(v, utc=True).view("int64")/1e9
                except Exception:
                    return np.nan
    return np.nan

def choose_tripish(row: dict):
    for k in ("trip_id","entity_id","vehicle_ref","train_id","block_id","route_id"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: dict):
    stop_id = str(row.get("stop_id","") or "")
    tripish = choose_tripish(row)
    evt = get_event_epoch(row)
    base = f"{SUBSYSTEM_TAG}|{stop_id}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

def ts(): return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
def out_name(): return f"subway_rt_{ts()}.csv"

def fetch(url):
    headers = {"x-api-key": MTA_API_KEY} if MTA_API_KEY else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed

def rows_from_feed(feed, stops: Set[str]) -> List[dict]:
    rows = []; server_ts = int(feed.header.timestamp or time.time())
    for e in feed.entity:
        tu = e.trip_update
        if not tu: continue
        for stu in tu.stop_time_update:
            if stu.stop_id not in stops: continue
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
    if not SUBWAY_STOP_IDS:
        print("SUBWAY_STOP_IDS is required.", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    all_rows = []
    for url in SUBWAY_FEEDS:
        try:
            feed = fetch(url); all_rows.extend(rows_from_feed(feed, SUBWAY_STOP_IDS))
        except Exception as e:
            print("[warn]", e, file=sys.stderr)

    df_poll["_uid"] = df_poll.apply(lambda r: make_uid(r.to_dict()), axis=1)

master_path = Path(MASTER_NAME)
if master_path.exists():
    df_master = pd.read_csv(master_path)
    if "_uid" not in df_master.columns:
        df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)
else:
    df_master = pd.DataFrame(columns=df_poll.columns)

# merge + dedup
all_cols = sorted(set(df_master.columns) | set(df_poll.columns))
df_master = df_master.reindex(columns=all_cols)
df_poll = df_poll.reindex(columns=all_cols)
combined = pd.concat([df_master, df_poll], ignore_index=True)

if "server_ts" in combined.columns:
    combined = combined.sort_values("server_ts", na_position="last")
combined = combined.drop_duplicates(subset=["_uid"], keep="last")

combined.to_csv(master_path, index=False)
print(f"✅ {SUBSYSTEM_TAG} master now {len(combined)} rows")


    outpath = os.path.join("data","subway", out_name()); write_csv(outpath, all_rows)
    subprocess.run(["rclone","copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"], check=False)
    append_to_master(outpath)
    print("Subway poll appended:", f"{GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__": main()
