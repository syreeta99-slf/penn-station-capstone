#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Amtrak poller (via Amtraker v3) for NYP:
- writes timestamped raw CSV locally and to Drive (/raw/)
- merges into Drive master with stable _uid and de-dup (coalesces latest non-null values)

ENV (set in workflow):
  GDRIVE_REMOTE_NAME    e.g., "gdrive" (required)
  GDRIVE_DIR_AMTRAK     e.g., "penn-station/amtrak" (default)
  AMTRAK_MASTER_NAME    e.g., "amtrak_penn_master.csv" (default)
  AMTRAK_STATION_CODE   default "NYP"
  AMTRAKER_TRAINS_URL   default "https://api-v3.amtraker.com/v3/trains"
  AMTRAKER_STATIONS_URL default "https://api-v3.amtraker.com/v3/stations" (unused)
"""

import os, sys, csv, json, time, tempfile, subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime as dt, date
from pathlib import Path
from hashlib import md5

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# ---------------- Env / Config ----------------
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")                         # required
GDRIVE_DIR    = os.getenv("GDRIVE_DIR_AMTRAK", "penn-station/amtrak")
MASTER_NAME   = os.getenv("AMTRAK_MASTER_NAME", "amtrak_penn_master.csv")
STATION_CODE  = (os.getenv("AMTRAK_STATION_CODE") or "NYP").upper()

TRAINS_URL    = os.getenv("AMTRAKER_TRAINS_URL", "https://api-v3.amtraker.com/v3/trains")
STATIONS_URL  = os.getenv("AMTRAKER_STATIONS_URL", "https://api-v3.amtraker.com/v3/stations")

OUT_DIR = Path("data/amtrak_rt/raw"); OUT_DIR.mkdir(parents=True, exist_ok=True)
NY_TZ   = ZoneInfo("America/New_York")
SUBSYSTEM_TAG = "amtrak"

FIELDS = [
    "pull_utc","server_ts","train_number","route_name","station_code",
    "arrival_time","departure_time","delay_sec","status","entity_id"
]

# ---------------- Small utils ----------------
def now_stamp() -> str:
    return dt.utcnow().strftime("%Y%m%d_%H%M%S")

def raw_filename() -> str:
    return f"amtrak_rt_{now_stamp()}.csv"

def http_json(url: str, retries: int = 2, timeout: int = 45) -> Any:
    last_err = None
    for attempt in range(retries + 1):
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent":"capstone-noncommercial"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt * 2)
    raise last_err

def parse_time_any(val) -> Optional[int]:
    """
    Parse epoch(s), ISO strings, or 'HH:MM' into epoch seconds (UTC).
    """
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        x = int(val)
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
    # HH:MM (assume today NY time)
    if ":" in s and len(s) <= 5:
        try:
            hh, mm = map(int, s.split(":")[:2])
            dti = dt.combine(date.today(), dt.min.time(), NY_TZ).replace(hour=hh, minute=mm)
            return int(dti.timestamp())
        except Exception:
            return None
    return None

# ---------------- Row extraction ----------------
def extract_rows(trains: Dict[str, Any], station_code: str) -> List[dict]:
    rows: List[dict] = []
    server_ts = int(time.time())
    pull_utc = dt.utcnow().isoformat(timespec="seconds")

    for train_num, variants in (trains or {}).items():
        if not isinstance(variants, list):
            continue
        for idx, t in enumerate(variants):
            # find this train's NYP stop entry
            nyp_stop = None
            for st in (t.get("stations") or []):
                code = (st.get("code") or st.get("station") or "").upper()
                if code == station_code:
                    nyp_stop = st; break
            if not nyp_stop:
                continue

            arr_candidates = [nyp_stop.get(k) for k in ("eta","estArr","arrival","estArrival","schArr")]
            dep_candidates = [nyp_stop.get(k) for k in ("etd","estDep","departure","estDeparture","schDep")]
            arrival_time   = next((x for x in (parse_time_any(v) for v in arr_candidates) if x is not None), None)
            departure_time = next((x for x in (parse_time_any(v) for v in dep_candidates) if x is not None), None)

            late_min = t.get("late") or nyp_stop.get("late")
            delay_sec = None
            try:
                if late_min is not None:
                    delay_sec = int(float(late_min)) * 60
            except Exception:
                delay_sec = None

            status    = t.get("status") or t.get("lastVal") or ""
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------------- _uid helpers ----------------
def get_event_epoch(row: dict):
    # Prefer arrival/departure/event_epoch if present; else fallback to server_ts/pull_utc
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

def choose_tripish(row: dict) -> str:
    for k in ("train_number","entity_id","route_name"):
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return ""

def make_uid(row: dict) -> str:
    stopish = str(row.get("station_code","") or "")
    tripish = choose_tripish(row)
    evt     = get_event_epoch(row)
    base    = f"{SUBSYSTEM_TAG}|{stopish}|{tripish}|{int(evt) if pd.notna(evt) else ''}"
    return md5(base.encode("utf-8")).hexdigest()

# ---------------- Helpers for safe coalescing ----------------
def _last_valid(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if len(s2) else (s.iloc[-1] if len(s) else pd.NA)

def _as_int64(series: pd.Series) -> pd.Series:
    # keep pandas nullable Int64 (not numpy int64) to preserve NA
    try:
        return series.astype("Int64")
    except Exception:
        return series

def _add_readable_ts(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("arrival_time","departure_time"):
        if col in df.columns:
            out_col = col.replace("_time","_ts_utc")
            df[out_col] = pd.to_datetime(df[col], unit="s", utc=True, errors="coerce")
    return df

# ---------------- rclone-based master update ----------------
def update_master_with_uid_rclone(local_new_df: pd.DataFrame) -> None:
    if not GDRIVE_REMOTE or not GDRIVE_REMOTE.strip():
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr)
        sys.exit(2)

    # Normalize dtypes on new pull
    for c in ("arrival_time","departure_time","server_ts","delay_sec"):
        if c in local_new_df.columns:
            local_new_df[c] = _as_int64(local_new_df[c])
    for c in ("train_number","route_name","station_code","status","entity_id","pull_utc"):
        if c in local_new_df.columns:
            local_new_df[c] = local_new_df[c].astype("string")

    # Ensure new pull has _uid
    if "_uid" not in local_new_df.columns:
        local_new_df["_uid"] = local_new_df.apply(lambda r: make_uid(r.to_dict()), axis=1)

    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")

        # Pull existing master if present
        pulled = subprocess.run(
            ["rclone","copyto", remote_master, local_master],
            capture_output=True, text=True
        )

        if pulled.returncode == 0 and os.path.exists(local_master):
            # Read with minimal inference; we’ll recast below
            df_master = pd.read_csv(local_master, dtype={"train_number":"string",
                                                         "route_name":"string",
                                                         "station_code":"string",
                                                         "status":"string",
                                                         "entity_id":"string",
                                                         "_uid":"string"},
                                    low_memory=False)
        else:
            df_master = pd.DataFrame(columns=local_new_df.columns)

        # Backfill _uid if missing in existing master
        if len(df_master) and "_uid" not in df_master.columns:
            df_master["_uid"] = df_master.apply(lambda r: make_uid(r.to_dict()), axis=1)

        # Align columns & union
        all_cols = sorted(set(df_master.columns) | set(local_new_df.columns))
        df_master    = df_master.reindex(columns=all_cols)
        local_new_df = local_new_df.reindex(columns=all_cols)

        combined = pd.concat([df_master, local_new_df], ignore_index=True)

        # Sort so later rows have newer server_ts; then coalesce by _uid using "last valid" values
        if "server_ts" in combined.columns:
            combined["server_ts"] = _as_int64(combined["server_ts"])
        for c in ("arrival_time","departure_time","delay_sec"):
            if c in combined.columns:
                combined[c] = _as_int64(combined[c])

        # ensure deterministic order before coalescing
        sort_cols = [c for c in ["_uid","server_ts"] if c in combined.columns]
        combined = combined.sort_values(sort_cols)

        # Build aggregation spec: prefer last non-null per column
        agg_spec = {}
        for c in combined.columns:
            if c == "_uid":
                continue
            agg_spec[c] = _last_valid

        collapsed = (combined
                     .groupby("_uid", as_index=False)
                     .agg(agg_spec))

        # Add readable timestamps for convenience
        collapsed = _add_readable_ts(collapsed)

        # Persist
        local_master_tmp = local_master
        collapsed.to_csv(local_master_tmp, index=False)
        subprocess.check_call(["rclone","copyto", local_master_tmp, remote_master])
        print(f"✅ amtrak master now {len(collapsed)} rows at {GDRIVE_DIR}/{MASTER_NAME}")

# ---------------- Main ----------------
def main():
    if not GDRIVE_REMOTE:
        print("ERROR: GDRIVE_REMOTE_NAME not provided.", file=sys.stderr)
        sys.exit(2)

    # 1) Fetch live trains JSON
    trains = http_json(TRAINS_URL, retries=2)

    # 2) Build NYP rows
    rows = extract_rows(trains, STATION_CODE)

    # 3) Write raw snapshot locally and to Drive (/raw/)
    outpath = OUT_DIR / raw_filename()
    write_csv(str(outpath), rows)
    subprocess.run(
        ["rclone","copyto", str(outpath), f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{outpath.name}"],
        check=False
    )

    # 4) Build DF, drop intra-pull dupes, add _uid, and update master on Drive
    df_poll = pd.DataFrame(rows, columns=FIELDS)

    # Optional within-pull de-dupe
    dedupe_key = [c for c in ["station_code","train_number","entity_id","arrival_time","departure_time"] if c in df_poll.columns]
    if dedupe_key:
        df_poll = df_poll.drop_duplicates(subset=dedupe_key, keep="last")

    update_master_with_uid_rclone(df_poll)

    print(f"[ok] Amtrak poll complete; station={STATION_CODE}, rows={len(rows)}")

if __name__ == "__main__":
    main()
