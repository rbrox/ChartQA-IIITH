"""Make the shared training subset for the first trained model (Checkpoint 1).

Draws a stratified random sample of the TRAIN split (half human-written questions, half
machine-generated) and saves it as a CSV, so everyone trains on exactly the same rows.
Each row records which train file and which row inside it holds the chart, plus a hash of the
chart image, so charts that appear several times are only extracted once.

It also checks that no chart in the subset appears in the validation split.

Usage:
    python utils/make_train_subset.py                 # 3000 rows -> results/train_subset_3000.csv
    python utils/make_train_subset.py --n 5000 --seed 7
"""
import argparse
import glob
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets/HuggingFaceM4/ChartQA/data"
STRATA_COL = "human_or_machine"


def load_with_hashes(pattern):
    """Every row of the files matching `pattern`, with its file number, position in the file and image hash."""
    parts = []
    for shard, path in enumerate(sorted(glob.glob(str(DATA_DIR / pattern)))):
        df = pd.read_parquet(path, columns=["image", "query", "label", STRATA_COL])
        df["chart_hash"] = [hashlib.md5(x["bytes"]).hexdigest() for x in df["image"]]
        df["shard"] = shard
        df["row_in_shard"] = range(len(df))
        parts.append(df.drop(columns="image"))
    df = pd.concat(parts, ignore_index=True)
    df["row_idx"] = df.index  # position in all matching files taken in order; unique id for each row
    return df


def make_subset(train, n, seed):
    """n rows, equal numbers from each group (human-written, machine-generated)."""
    per_group = n // train[STRATA_COL].nunique()
    parts = [train[train[STRATA_COL] == g].sample(n=per_group, random_state=seed) for g in sorted(train[STRATA_COL].unique())]
    subset = pd.concat(parts).sort_values(["shard", "row_in_shard"]).reset_index(drop=True)
    subset["label"] = subset["label"].apply(lambda x: "|".join(map(str, x)))  # list -> text for CSV
    return subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or ROOT / "results" / f"train_subset_{args.n}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    train = load_with_hashes("train-*.parquet")
    subset = make_subset(train, args.n, args.seed)

    val_hashes = set(load_with_hashes("val-*.parquet")["chart_hash"])
    leaked = subset["chart_hash"].isin(val_hashes).sum()

    subset.to_csv(out, index=False)
    print(f"Saved {len(subset)} rows to {out}")
    print(subset[STRATA_COL].value_counts().sort_index().to_string())
    print(f"unique charts in the subset: {subset['chart_hash'].nunique()}")
    print(f"subset rows whose chart also appears in validation: {leaked}")
