#!/usr/bin/env python3
"""
Poll Amtrak (unofficial via Amtraker v3) for NYP, write timestamped raw CSV locally,
ship raw to Google Drive, and append rows to a remote rolling master CSV on Drive.

Pattern:
- local timestamped raw: data/amtrak_rt/raw/amtrak_rt_<STAMP>.csv
- remote raw: <GDRIVE_REMOTE_NAME>:<GDRIVE_DIR_AMTRAK>/raw/...
- remote master append: <GDRIVE_REMOTE_NAME>:<GDRIVE_DIR_AMTRAK>/<AMTRAK_MASTER_NAME>

ENV (set in workflow):
  GDRIVE_REMOTE_NAME    e.g., "googledrive" (required)
  GDRIVE_DIR_AMTRAK     e.g., "penn-station/amtrak" (default)
  AMTRAK_MASTER_NAME    e.g., "amtrak_penn_master.csv" (default)
  AMTRAK_STATION_CODE   default "NYP" (Moynihan / New York Penn)
  AMTRAKER_TRAINS_URL   default "https://api-v3.amtraker.com/v3/trains"
  AMTRAKER_STATIONS_URL default "https://api-v3.amtraker.com/v3/stations"
"""

import os, sys, csv, json, time, tempfile, pathlib, subprocess, datetime, urllib.request
from typing import List, Dict, Any, Optional
from datetime import datetime as dt, date
from zoneinfo import ZoneInfo

# -------- Config from env (align to NJT pattern) --------
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")                         # required
GDRIVE_DIR = os.getenv("GDRIVE_DIR_AMTRAK", "penn-station/amtrak")
MASTER_NAME = os.getenv("AMTRAK_MASTER_NAME", "amtrak_penn_master.csv")
STATION_CODE = (os.getenv("AMTRAK_STATION_CODE") or "NYP").upper()

TRAINS_URL = os.getenv("AMTRAKER_TRAINS_URL", "https://api-v3.amtraker.com/v3/trains")
STATIONS_URL = os.getenv("AMTRAKER_STATIONS_URL", "https://api-v3.amtraker.com/v3/stations")

OUT_DIR = pathlib.Path("data/amtrak_rt/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NY_TZ = ZoneInfo("America/New_York")

# Keep fields parallel to your NJT raw shape where it makes sense
FIELDS = [
    "pull_utc","server_ts","train_number","route_name","station_code",
    "arrival_time","departure_time","delay_sec","status","entity_id"
]

# -------- Small utils --------
def now_stamp() -> str:
    return dt.utcnow().strftime("%Y%m%d_%H%M%S")

def raw_filename() -> str:
    return f"amtrak_rt_{now_stamp()}.csv"

def http_json(url: str, retries: int = 2, timeout: int = 45) -> Any:
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"capstone-noncommercial"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            # Backoff (1s, then 3s)
            time.sleep(1 + attempt * 2)
    raise last_err

def parse_time_any(val) -> Optional[int]:
    """
    Normalize many time formats to epoch seconds (int, UTC-based):
      - epoch seconds (int/float)
      - epoch milliseconds (int/float, auto-divide)
      - ISO8601 strings ("2025-10-13T14:23:00-04:00" or "...Z")
      - 'HH:MM' strings (assumed *today* in America/New_York)
    Returns None if cannot parse.
    """
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        x = int(val)
        # If likely ms (>> year 2033 in seconds), convert
        return x // 1000 if x > 2_000_000_000 else x
    s = str(val).strip()
    # ISO8601
    try:
        iso = s.replace("Z", "+00:00")
        dti = dt.fromisoformat(iso)
        if dti.tzinfo is None:
            dti = dti.replace(tzinfo=NY_TZ)
        return int(dti.timestamp())
    except Exception:
        pass
    # HH:MM
    if ":" in s and len(s) <= 5:
        try:
            hh, mm = map(int, s.split(":")[:2])
            dti = dt.combine(date.today(), dt.min.time(), NY_TZ).replace(hour=hh, minute=mm)
            return int(dti.timestamp())
        except Exception:
            return None
    return None

def extract_rows(trains: Dict[str, Any], station_code: str) -> List[dict]:
    """
    Build NJT-like rows from the Amtraker JSON. Only include trains whose stoplist contains station_code (NYP).
    Robustly parse arrival/dep; fallback to scheduled when live is absent.
    """
    rows: List[dict] = []
    server_ts = int(time.time())
    pull_utc = dt.utcnow().isoformat(timespec="seconds")

    for train_num, variants in (trains or {}).items():
        if not isinstance(variants, list):
            continue
        for idx, t in enumerate(variants):
            stations_list = t.get("stations") or []
            nyp_stop = None
            for st in stations_list:
                code = (st.get("code") or st.get("station") or "").upper()
                if code == station_code:
                    nyp_stop = st; break
            if not nyp_stop:
                continue

            # Collect lots of candidates; take first that parses
            arr_candidates = [
                nyp_stop.get("eta"), nyp_stop.get("estArr"),
                nyp_stop.get("arrival"), nyp_stop.get("estArrival"),
                nyp_stop.get("schArr")   # scheduled fallback
            ]
            dep_candidates = [
                nyp_stop.get("etd"), nyp_stop.get("estDep"),
                nyp_stop.get("departure"), nyp_stop.get("estDeparture"),
                nyp_stop.get("schDep")   # scheduled fallback
            ]
            arrival_time = next((x for x in (parse_time_any(v) for v in arr_candidates) if x is not None), None)
            departure_time = next((x for x in (parse_time_any(v) for v in dep_candidates) if x is not None), None)

            # delay: many variants expose "late" in minutes; convert to seconds if numeric
            late_min = t.get("late") or nyp_stop.get("late")
            delay_sec: Optional[int] = None
            try:
                if late_min is not None:
                    delay_sec = int(float(late_min)) * 60
            except Exception:
                delay_sec = None

            status = t.get("status") or t.get("lastVal") or ""
            entity_id = f"{train_num}_{idx}"  # stable per train object in this pull

            # Optional debug: if times missing, print keys once (comment out in production)
            # if not arrival_time and not departure_time:
            #     print("[debug] NYP stop had no times; keys:", list(nyp_stop.keys()))

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

def rclone_lsjson(remote_path: str):
    cp = subprocess.run(["rclone", "lsjson", remote_path], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return []

def append_to_remote_master(local_new_csv: str) -> None:
    """
    Copy the remote master down, append rows (skip header), copy back up.
    """
    if not GDRIVE_REMOTE:
        print("ERROR: GDRIVE_REMOTE_NAME is not set.", file=sys.stderr)
        sys.exit(2)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    exists = rclone_lsjson(remote_master)

    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        if exists:
            # Download existing master
            subprocess.check_call(["rclone", "copyto", remote_master, local_master])

            # Append new rows (skip header line)
            with open(local_master, "a", newline="") as out, open(local_new_csv, "r", newline="") as newf:
                reader = csv.reader(newf)
                next(reader, None)  # skip header
                for r in reader:
                    out.write(",".join(str(x) for x in r) + "\n")
        else:
            # No master yet; initialize with the new file
            subprocess.check_call(["cp", local_new_csv, local_master])

        # Upload updated master
        subprocess.check_call(["rclone", "copyto", local_master, remote_master])

def main():
    if not GDRIVE_REMOTE:
        print("ERROR: GDRIVE_REMOTE_NAME not provided.", file=sys.stderr)
        sys.exit(2)

    # Fetch live data (with light retry/backoff)
    trains = http_json(TRAINS_URL, retries=2)

    # Build NYP-only rows (robust parsing + scheduled fallback)
    rows = extract_rows(trains, STATION_CODE)

    # Write local timestamped raw
    outpath = OUT_DIR / raw_filename()
    write_csv(str(outpath), rows)
    print(f"[raw] wrote {len(rows)} rows → {outpath}")

    # Copy raw to Drive raw folder (best-effort)
    subprocess.run(
        ["rclone", "copyto", str(outpath), f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{outpath.name}"],
        check=False
    )

    # Append to remote rolling master on Drive
    append_to_remote_master(str(outpath))
    print(f"[master] appended to {GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__":
    main()
