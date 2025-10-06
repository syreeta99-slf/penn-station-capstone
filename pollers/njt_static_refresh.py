#!/usr/bin/env python3
from __future__ import annotations
import os, sys, io, zipfile
from pathlib import Path
from datetime import datetime, timezone
import requests

def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[njt_static] {now} {msg}")

def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def looks_like_zip(blob: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            _ = zf.infolist()
        return True
    except Exception:
        return False

def main():
    url = os.getenv("NJT_GTFS_STATIC_URL", "").strip()
    out_dir = Path(os.getenv("OUT_DIR", "gtfs_static_njt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not url:
        print("ERROR: set NJT_GTFS_STATIC_URL to the rail GTFS static zip.", file=sys.stderr)
        sys.exit(2)

    log(f"Downloading static GTFS from {url}")
    blob = fetch_bytes(url)
    if not looks_like_zip(blob):
        print("ERROR: downloaded file is not a valid zip.", file=sys.stderr)
        sys.exit(3)

    ymd = datetime.now(timezone.utc).strftime("%Y%m%d")
    dated = out_dir / f"rail_all_{ymd}.zip"
    latest = out_dir / "rail_latest.zip"

    dated.write_bytes(blob)
    latest.write_bytes(blob)

    try:
        import io as _io, zipfile as _zip
        with _zip.ZipFile(_io.BytesIO(blob), "r") as zf:
            names = [i.filename for i in zf.infolist()][:10]
        log(f"Saved {dated.name} ({len(blob)/1e6:.1f} MB). Sample entries: {names}")
    except Exception:
        log(f"Saved {dated.name} ({len(blob)/1e6:.1f} MB)")

    print(f"Saved {dated}")
    print(f"Saved {latest}")

if __name__ == "__main__":
    main()
