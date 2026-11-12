"""Pack the chart images of the 300-row validation sample into one small parquet file.

The trained model is evaluated on the validation sample, so it needs image features for those charts too.
This writes the same format as make_subset_images.py (columns: chart_hash, image), so
extract_image_features.py can read it. About 280 charts.

Usage:
    python utils/make_val_images.py
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "datasets/HuggingFaceM4/ChartQA/data/val-00000-of-00001-0f11003c77497969.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", type=Path, default=ROOT / "results/val_sample_300.csv")
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets/val_sample/val_sample_300_images.parquet")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_csv)
    images = pd.read_parquet(args.val_path, columns=["image"])
    picked = images.iloc[sample["row_idx"].to_numpy()]["image"].tolist()
    packed = pd.DataFrame({"chart_hash": [hashlib.md5(x["bytes"]).hexdigest() for x in picked], "image": picked})
    packed = packed.drop_duplicates("chart_hash").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    packed.to_parquet(args.out, index=False)
    print(f"Saved {len(packed)} charts to {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
