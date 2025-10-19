#!/usr/bin/env python3
# Fetch NYC DOT Bi-Annual Pedestrian Counts and append to a Drive master CSV.
import os, sys, csv, tempfile, pathlib, subprocess, datetime, requests, json

# ---------------- ENV ----------------
NYC_PEDS_URL = os.getenv(
    "NYC_PEDS_URL",
    "https://data.cityofnewyork.us/resource/2de2-6x2h.csv?$limit=500000"
)

# Case-insensitive match tokens; comma-separated (e.g., "PENN,34 ST,31 ST,33 ST,7 AV,8 AV")
FILTER_SUBSTRINGS = [s.strip().lower() for s in os.getenv("PEDS_FILTER", "").split(",") if s.strip()]

# Columns (case-insensitive header names) to apply filtering against.
# Defaults cover typical NYC DOT naming; adjust via repo var PEDS_FILTER_COLUMNS if needed.
PEDS_FILTER_COLUMNS = [s.strip().lower() for s in os.getenv(
    "PEDS_FILTER_COLUMNS",
    "location_name,location,intersection,streetname,street_1,street_2,corridor"
).split(",") if s.strip()]

GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE_NAME")          # required
GDRIVE_DIR     = os.getenv("GDRIVE_DIR_PEDS", "penn-station/pedestrians")
MASTER_NAME    = os.getenv("PEDS_MASTER_NAME", "pedestrian_master.csv")

# ---------------- HELPERS ----------------
def now_ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def ensure_dirs():
    pathlib.Path("data/pedestrians").mkdir(parents=True, exist_ok=True)

def rclone_lsjson(remote):
    cp = subprocess.run(["rclone","lsjson",remote], capture_output=True, text=True)
    if cp.returncode != 0 or not cp.stdout.strip():
        return []
    return json.loads(cp.stdout)

def filtered_copy(source_path, dest_path, filters, preferred_cols):
    """
    Copy CSV keeping header + rows where ANY filter token appears (case-insensitive)
    in ANY of the selected columns. If none of those columns exist, fallback to whole-row match.
    If 'filters' is empty, copy as-is (no filtering).
    """
    if not filters:
        subprocess.check_call(["cp", source_path, dest_path])
        return

    with open(source_path, "r", encoding="utf-8", errors="ignore", newline="") as inf, \
         open(dest_path,   "w", encoding="utf-8", errors="ignore", newline="") as outf:

        reader = csv.reader(inf)
        writer = csv.writer(outf)

        try:
            header = next(reader)
        except StopIteration:
            return

        writer.writerow(header)

        lower_header = [h.strip().lower() for h in header]
        col_indices = [i for i, h in enumerate(lower_header) if h in preferred_cols]
        use_whole_row = (len(col_indices) == 0)

        for row in reader:
            if use_whole_row:
                text = ",".join(row).lower()
                if any(tok in text for tok in filters):
                    writer.writerow(row)
            else:
                haystacks = [str(row[i]).lower() for i in col_indices if 0 <= i < len(row)]
                if any(any(tok in h for h in haystacks) for tok in filters):
                    writer.writerow(row)

def append_master(local_csv):
    remote_master = f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/{MASTER_NAME}"
    exists = rclone_lsjson(remote_master)

    with tempfile.TemporaryDirectory() as td:
        local_master = os.path.join(td, "master.csv")
        if exists:
            subprocess.check_call(["rclone","copyto", remote_master, local_master])
            with open(local_master, "a", encoding="utf-8", errors="ignore") as out, \
                 open(local_csv, "r", encoding="utf-8", errors="ignore") as newf:
                first = True
                for line in newf:
                    if first:
                        first = False
                        continue
                    out.write(line)
        else:
            subprocess.check_call(["cp", local_csv, local_master])

        subprocess.check_call(["rclone","copyto", local_master, remote_master])

def main():
    if not GDRIVE_REMOTE:
        print("GDRIVE_REMOTE_NAME is required.", file=sys.stderr); sys.exit(2)

    ensure_dirs()

    # 1) Download latest CSV
    out_raw = f"data/pedestrians/ped_counts_{now_ts()}.csv"
    resp = requests.get(NYC_PEDS_URL, timeout=120)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    # 2) Column-based filtering (keeps header); falls back to whole-row if needed
    out_filtered = f"data/pedestrians/ped_counts_filtered_{now_ts()}.csv"
    filtered_copy(tmp_path, out_filtered, FILTER_SUBSTRINGS, PEDS_FILTER_COLUMNS)

    # 3) Canonical raw snapshot (the kept set)
    subprocess.check_call(["cp", out_filtered, out_raw])

    # 4) Upload raw
    subprocess.run(
        ["rclone","copyto", out_raw, f"{GDRIVE_REMOTE}:{GDRIVE_DIR}/raw/{os.path.basename(out_raw)}"],
        check=False
    )

    # 5) Append to master
    append_master(out_raw)

    print("Pedestrian counts appended:", f"{GDRIVE_DIR}/{MASTER_NAME}")

if __name__ == "__main__":
    import os
    main()
