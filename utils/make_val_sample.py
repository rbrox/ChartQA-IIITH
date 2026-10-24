"""Make a shared, stratified sample of the ChartQA validation split.

Everyone (DePlot run, UniChart run) loads the saved CSV instead of re-sampling,
so both models are scored on exactly the same rows.

Usage:
    python utils/make_val_sample.py                 # 300 rows -> results/val_sample_300.csv
    python utils/make_val_sample.py --n 500 --seed 7
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "datasets/HuggingFaceM4/ChartQA/data/val-00000-of-00001-0f11003c77497969.parquet"
STRATA_COL = "human_or_machine"


def make_sample(val_path: Path, n: int, seed: int) -> pd.DataFrame:
    # Skip the image column: we only need text to pick rows, and it is big.
    df = pd.read_parquet(val_path, columns=["query", "label", STRATA_COL])
    df.index.name = "row_idx"  # position in the val parquet, used later to fetch the image
    df = df.reset_index()

    # Each group gets the share it has in the full set (50/50 here), via largest remainder.
    shares = df[STRATA_COL].value_counts(normalize=True) * n
    per_group = shares.astype(int)
    leftover = n - per_group.sum()
    for group in (shares - per_group).sort_values(ascending=False).index[:leftover]:
        per_group[group] += 1

    parts = [
        df[df[STRATA_COL] == group].sample(n=int(k), random_state=seed)
        for group, k in per_group.items()
    ]
    sample = pd.concat(parts).sort_values("row_idx").reset_index(drop=True)
    sample["label"] = sample["label"].apply(lambda x: "|".join(map(str, x)))  # list -> text for CSV
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or ROOT / "results" / f"val_sample_{args.n}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    sample = make_sample(args.val_path, args.n, args.seed)
    sample.to_csv(out, index=False)
    print(f"Saved {len(sample)} rows to {out}")
    print(sample[STRATA_COL].value_counts().to_string())
