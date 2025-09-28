import pathlib, requests, datetime

# Official static GTFS ZIPs
STATIC_FEEDS = {
    "subway_all": "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip",
    "subway_supplemented": "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip",
    # You can add LIRR or others later if needed
}

def download(url, out_path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

if __name__ == "__main__":
    root = pathlib.Path("gtfs_static")
    root.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().strftime("%Y%m%d")

    for name, url in STATIC_FEEDS.items():
        out = root / f"{name}_{today}.zip"
        download(url, out)
        print("Saved", out)
