import pandas as pd, pathlib
from glob import glob

CUR = pathlib.Path("data/curated")
CUR.mkdir(parents=True, exist_ok=True)

def load_latest_realtime():
    files = sorted(glob("data/realtime/subway_rt_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files[-6:]]  # last few runs
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df = load_latest_realtime()
    if df.empty:
        print("No realtime to curate")
    else:
        # simple dedup: keep the last record per (trip_id, stop_id, source_minute_utc)
        df["key"] = df["trip_id"].astype(str) + "|" + df["stop_id"].astype(str) + "|" + df["source_minute_utc"].astype(str)
        df = df.drop_duplicates("key", keep="last").drop(columns=["key"])

        # save curated tables your analysts can use
        df.to_parquet(CUR / "master_interface_dataset.parquet", index=False)
        df.to_csv(CUR / "master_interface_dataset.csv", index=False)
        print("Wrote curated master dataset")
