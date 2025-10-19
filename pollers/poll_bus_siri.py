#!/usr/bin/env python3
# Poll MTA Bus Time (SIRI StopMonitoring) for selected stops and append to a Drive master CSV.
import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime
import requests
from typing import List, Dict, Any, Set

# ---------- ENV ----------
API_KEY = os.getenv("MTA_API_KEY")  # required
BUS_STOP_IDS: Set[str] = {s.strip() for s in os.getenv("BUS_STOP_IDS", "").split(",") if s.strip()}  # e.g. "MTA_401231,MTA_401361"
SIRI_URL = os.getenv("SIRI_STOPMON_URL", "https://bustime.mta.info/api/siri/stop-monitoring.json")

GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")  # required
GDRIVE_DIR = os.getenv("GDRIVE_DIR_BUS", "penn-station/bus")
MASTER_NAME = os.getenv("BUS_MASTER_NAME", "bus_penn_master.csv")

# Output schema (aligned with your other modes)
FIELDS = [
    "pull_utc","server_ts","route_id","stop_id",
    "arrival_time","departure_time",
    "vehicle_ref","destination","visit_number","entity_id",
    "has_expected_arrival","has_expected_departure"
]

def utc_now_iso():
    return datetime.datetime.utcnow().isoformat(timespec="seconds")

def iso_to_epoch(iso_str: str | None) -> int | None:
    if not iso_str:
        return None
    try:
        # Handles e.g. 2025-10-18T20:12:38.407-04:00 or ...Z
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None

def fetch_stop(stop_id: str) -> Dict[str, Any]:
    r = requests.get(SIRI_URL, params={"key": API_KEY, "MonitoringRef": stop_id}, timeout=45)
    r.raise_for_status()
    return r.json()

def extract_rows(payload: Dict[str, Any], pull_iso: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sd = payload.get("Siri", {}).get("ServiceDelivery", {}) or {}
    server_ts = iso_to_epoch(sd.get("ResponseTimestamp")) or int(time.time())
    deliveries = sd.get("StopMonitoringDelivery", []) or []
    for d in deliveries:
        for v in (d.get("MonitoredStopVisit") or []):
            mvj = v.get("MonitoredVehicleJourney", {}) or {}
            mc  = mvj.get("MonitoredCall", {}) or {}
            rows.append({
                "pull_utc": pull_iso,
                "server_ts": server_ts,
                "route_id": mvj.get("LineRef"),
                "stop_id": mc.get("StopPointRef") or mvj.get("MonitoredCall", {}).get("StopPointRef"),
                "arrival_time": iso_to_epoch(mc.get("ExpectedArrivalTime")),
                "departure_time": iso_to_epoch(mc.get("ExpectedDepartureTime")),
                "vehicle_ref": mvj.get("VehicleRef"),
                "destination": mvj.get("DestinationName"),
                "visit_number": v.get("VisitNumber"),
                "entity_id": v.get("ItemIdentifier") or "",
                "has_expected_arrival": 1 if mc.get("ExpectedArrivalTime") else 0,
                "has_expected_departure": 1 if mc.get("ExpectedDepartureTime") else 0,
            })
    return rows

def write_csv(path: str, rows: List[Dict[str, Any]]):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

def lsjson(remote: str):
    cp = subprocess.run(["rclone","lsjson",remote], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    return json.loads(cp.stdout)

def append_to_master(local_new_csv: str):
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
    if not API_KEY:
        print("MTA_API_KEY is required.", file=sys.stderr); sys.exit(2)
    if not BUS_STOP_IDS:
        print("BUS_STOP_IDS is required (comma list).", file=sys.stderr); sys.exit(2)
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    pull_iso = utc_now_iso()
    all_rows: List[Dict[str, Any]] = []

    for sid in BUS_STOP_IDS:
        try:
            payload = fetch_stop(sid)
            rows = extract_rows(payload, pull_iso)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[warn] stop {sid}: {e}", file=sys.stderr)

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join("data", "bus", f"bus_rt_{ts}.csv")
    write_csv(outpath, all_rows)

    # Upload raw and append master
    subprocess.run(["rclone", "copyto", outpath, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(outpath)}"], check=False)
    append_to_master(outpath)
    print("Bus poll appended:", f"{GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__":
    main()
