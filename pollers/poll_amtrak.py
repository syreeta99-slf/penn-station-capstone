#!/usr/bin/env python3
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, urllib.request
from typing import List, Dict, Any

# ---- Config (env), mirroring your NJT pattern ----
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")                # required
GDRIVE_DIR = os.getenv("GDRIVE_DIR_AMTRAK", "penn-station/amtrak")
MASTER_NAME = os.getenv("AMTRAK_MASTER_NAME", "amtrak_penn_master.csv")
STATION_CODE = os.getenv("AMTRAK_STATION_CODE", "NYP")         # Moynihan/Penn
TRAINS_URL = os.getenv("AMTRAKER_TRAINS_URL", "https://api-v3.amtraker.com/v3/trains")
STATIONS_URL = os.getenv("AMTRAKER_STATIONS_URL", "https://api-v3.amtraker.com/v3/stations")

OUT_DIR = pathlib.Path("data/amtrak")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    # Keep it NJT-like where applicable
    "pull_utc","server_ts","train_number","route_name","station_code",
    "arrival_time","departure_time","delay_sec","status","entity_id"
]

def ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def out_name() -> str:
    return f"amtrak_rt_{ts()}.csv"

def http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent":"capstone-noncommercial"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())

def extract_rows(trains: Dict[str, Any], station_code: str) -> List[dict]:
    rows = []
    server_ts = int(time.time())
    pull_utc = datetime.datetime.utcnow().isoformat(timespec="seconds")
    for train_num, variants in (trains or {}).items():
        if not isinstance(variants, list):
            continue
        for idx, t in enumerate(variants):
            stations_list = t.get("stations") or []
            # find this train's NYP stop, if any
            nyp_stop = None
            for st in stations_list:
                code = (st.get("code") or st.get("station") or "").upper()
                if code == station_code:
                    nyp_stop = st; break
            if not nyp_stop:
                continue

            # arrival/departure may be epoch seconds when present
            arr = nyp_stop.get("eta") or nyp_stop.get("estArr") or nyp_stop.get("arrival") or nyp_stop.get("estArrival")
            dep = nyp_stop.get("etd") or nyp_stop.get("estDep") or nyp_stop.get("departure") or nyp_stop.get("estDeparture")
            arrival_time = int(arr) if isinstance(arr, (int,float)) else None
            departure_time = int(dep) if isinstance(dep, (int,float)) else None

            # delay (approx) — some feeds expose minutes late; convert to seconds if possible
            late_min = t.get("late") or nyp_stop.get("late")
            try:
                delay_sec = int(late_min) * 60 if late_min is not None else None
            except Exception:
                delay_sec = None

            status = t.get("status") or t.get("lastVal") or ""
            entity_id = f"{train_num}_{idx}"

            rows.append({
                "pull_utc": pull_utc,
                "server_ts": server_ts,
                "train_number": str(train_num),
                "route_name": t.get("name") or t.get("route") or "",
                "station_code": station_code,
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "delay_sec": delay_sec,
                "status": status,
                "entity_id": entity_id,
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

    # fetch live data
    trains = http_json(TRAINS_URL)
    # stations endpoint not strictly required here, but keep for parity/testing
    # stations = http_json(STATIONS_URL)

    rows = extract_rows(trains, STATION_CODE)

    outpath = os.path.join("data","amtrak", out_name())
    write_csv(outpath, rows)

    # raw audit copy to Drive
    subprocess.run([
        "rclone","copyto",
        outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"
    ], check=False)

    append_to_master(outpath)
    print(f"[ok] Amtrak poll (station {STATION_CODE}) appended to {GDRIVE_DIR}/{MASTER_NAME} with {len(rows)} rows")

if __name__ == "__main__":
    main()
