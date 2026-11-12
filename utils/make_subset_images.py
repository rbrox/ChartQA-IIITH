"""Pack the chart images of the training subset into one small parquet file.

The subset CSV lists 3000 questions that use about 2800 different charts spread over three large
train files. This script copies just those charts (each once) into a single file, so only that
small file has to be uploaded to Kaggle for the UniChart run.

Output columns: chart_hash, image (bytes). Saved under datasets/ (not tracked by git).

Usage:
    python utils/make_subset_images.py
    python utils/make_subset_images.py --subset-csv results/train_subset_3000.csv
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets/HuggingFaceM4/ChartQA/data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-csv", type=Path, default=ROOT / "results/train_subset_3000.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets/train_subset/train_subset_3000_images.parquet")
    args = parser.parse_args()

    subset = pd.read_csv(args.subset_csv)
    first_use = subset.drop_duplicates("chart_hash")  # one image per chart
    files = sorted(glob.glob(str(DATA_DIR / "train-*.parquet")))

    parts = []
    for shard, path in enumerate(files):
        wanted = first_use[first_use["shard"] == shard]
        if wanted.empty:
            continue
        images = pd.read_parquet(path, columns=["image"])
        picked = images.iloc[wanted["row_in_shard"].to_numpy()].reset_index(drop=True)
        parts.append(pd.DataFrame({"chart_hash": wanted["chart_hash"].to_numpy(), "image": picked["image"].to_numpy()}))

    packed = pd.concat(parts, ignore_index=True)
    assert len(packed) == subset["chart_hash"].nunique(), "a chart is missing"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    packed.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved {len(packed)} charts to {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
