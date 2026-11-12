"""Combine the rule-based labels with the verified LLM recipes into one final label file.

Rules:
- extract, yes_no and count_rows labels from the label-maker are kept as they are
- a compute label is kept only if the LLM recipe independently reproduced the true answer (the recipe is stored)
- an unusable question becomes a compute label if a verified recipe exists, otherwise it stays unusable

Usage:
    python utils/merge_labels.py
    python utils/merge_labels.py --labels results/unichart_val_300_labels.jsonl \
        --recipes results/unichart_val_300_recipes.jsonl --out results/unichart_val_300_final_labels.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def merge(labels, recipes):
    verified = recipes[recipes["verified"]].set_index("row_idx")["recipe"].to_dict()
    kinds, details = [], []
    for row in labels.itertuples(index=False):
        if row.kind == "compute":
            if row.row_idx in verified:
                kinds.append("compute")
                details.append({"recipe": verified[row.row_idx], "source": "rule and recipe agree"})
            else:
                kinds.append("unusable")
                details.append({"reason": "rule-based compute label not confirmed by an LLM recipe"})
        elif row.kind == "unusable" and row.row_idx in verified:
            kinds.append("compute")
            details.append({"recipe": verified[row.row_idx], "source": "recovered by recipe"})
        else:
            kinds.append(row.kind)
            details.append(row.detail)
    out = labels.copy()
    out["kind"] = kinds
    out["detail"] = details
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "results/unichart_train_3000_labels.jsonl")
    parser.add_argument("--recipes", type=Path, default=ROOT / "results/unichart_train_3000_recipes.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/unichart_train_3000_final_labels.jsonl")
    args = parser.parse_args()

    labels = pd.read_json(args.labels, lines=True, dtype=False)
    recipes = pd.read_json(args.recipes, lines=True, dtype=False)
    final = merge(labels, recipes)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in final.to_dict("records"):
            f.write(json.dumps(rec) + "\n")

    print(f"{len(final)} rows -> {args.out}")
    overall = (final["kind"].value_counts(normalize=True) * 100).rename("all")
    by_group = pd.crosstab(final["kind"], final["human_or_machine"], normalize="columns") * 100
    print("\nfinal kind, % of rows (group 0 = human-written, 1 = machine-generated)")
    print(pd.concat([overall, by_group], axis=1).round(1).to_string())
    print("\ncounts:", final["kind"].value_counts().to_dict())
    usable = final["kind"] != "unusable"
    print(f"usable: {usable.sum()} of {len(final)} ({usable.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
