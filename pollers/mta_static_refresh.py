#!/usr/bin/env python3
import pathlib, requests, datetime, zipfile, io, sys

# Official static GTFS ZIPs (ordered by preference)
STATIC_FEEDS = [
    ("subway_all", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"),
    ("subway_supplemented", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"),
    # Add others here if you want fallback options
]

OUT_DIR = pathlib.Path("gtfs_static")
REQUIRED_FILES = {"trips.txt", "stop_times.txt", "stops.txt"}

def fetch(url: str, timeout=60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def validate(buf: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(buf), "r") as z:
            names = set(z.namelist())
            missing = REQUIRED_FILES - names
            if missing:
                raise ValueError(f"zip missing required files: {sorted(missing)}")
    except zipfile.BadZipFile as e:
        raise ValueError(f"invalid zip: {e}") from e

def write_zip(buf: bytes, name_prefix: str) -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    dated = OUT_DIR / f"{name_prefix}_{today}.zip"
    dated.write_bytes(buf)
    # also write/update a stable alias for easy auto-detect
    alias_name = "mta_subway_static.zip" if "subway" in name_prefix else f"{name_prefix}.zip"
    (OUT_DIR / alias_name).write_bytes(buf)
    return dated

def prune_old(prefixes=("subway_all", "subway_supplemented"), keep=3):
    for pfx in prefixes:
        zips = sorted(OUT_DIR.glob(f"{pfx}_*.zip"))
        for old in zips[:-keep]:
            try: old.unlink()
            except Exception: pass

def main():
    last_ok = None
    for name, url in STATIC_FEEDS:
        try:
            print(f"[mta_static_refresh] downloading {name} from {url}")
            buf = fetch(url)
            print(f"[mta_static_refresh] {len(buf):,} bytes; validating…")
            validate(buf)
            path = write_zip(buf, name)
            print(f"[mta_static_refresh] wrote: {path}")
            last_ok = name
            break
        except Exception as e:
            print(f"[mta_static_refresh] WARN: {name} failed: {e}", file=sys.stderr)

    if not last_ok:
        print("[mta_static_refresh] ERROR: all feeds failed", file=sys.stderr)
        sys.exit(2)

    prune_old()
    print("[mta_static_refresh] done.")

if __name__ == "__main__":
    main()
