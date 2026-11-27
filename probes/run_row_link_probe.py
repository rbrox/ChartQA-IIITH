"""Row-link probe: do the frozen text vectors know which value belongs to which row name?

Each cell is embedded alone by MiniLM. If "India" and "8.2" are in the same row, does anything in their two
vectors show it? Two tests (read-only on the saved vectors):
    no training   for a value cell, is the most similar row-name vector the one from its own row? (chance = 1 / rows)
    probe         given the vector of a row name and of a value cell, are they from the same row?
                  negatives are another row's name from the SAME table (50/50, chance 50%).
                  input forms: the two vectors side by side, and side by side plus their product
Shuffled-label control as in the other probes. Items of one chart stay on one side of the train/test split.

Usage:
    python probes/run_row_link_probe.py
"""
import torch  # must be imported before numpy and pandas on Windows
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_probes import ROOT, chart_split, fit_probe, load  # noqa: E402

OUT = ROOT / "probes/results/row_link_probe_results.json"


def main():
    d = load("train")
    emb = d["cand_emb"].astype(np.float32)
    rng = np.random.default_rng(0)
    pair_name, pair_val, y, groups = [], [], [], []
    sim_hits, sim_chance = [], []
    seen = set()
    for i, m in enumerate(d["meta"]):
        if m["chart_hash"] in seen:
            continue
        seen.add(m["chart_hash"])
        lo = d["offsets"][i]
        names = {c["row"]: lo + j for j, c in enumerate(m["cand"]) if c["kind"] == "row_label"}
        if len(names) < 3:
            continue
        rows = sorted(names)
        for j, c in enumerate(m["cand"]):
            if c["kind"] != "value" or c["row"] not in names:
                continue
            others = [r for r in rows if r != c["row"]]
            other = others[rng.integers(len(others))]
            pair_name += [names[c["row"]], names[other]]
            pair_val += [lo + j, lo + j]
            y += [1, 0]
            groups += [m["chart_hash"]] * 2
            sims = np.array([emb[names[r]] @ emb[lo + j] for r in rows])
            sim_hits.append(int(rows[int(np.argmax(sims))] == c["row"]))
            sim_chance.append(1 / len(rows))
    results = {"no_training": {"value_cells": len(sim_hits), "top1": float(np.mean(sim_hits)), "chance": float(np.mean(sim_chance))}}
    print(f"No training: the most similar row name to a value cell is its own row's name {np.mean(sim_hits) * 100:.1f}% of the time "
          f"(chance {np.mean(sim_chance) * 100:.1f}%, {len(sim_hits)} value cells)")

    pair_name, pair_val, y, groups = np.array(pair_name), np.array(pair_val), np.array(y), np.array(groups)
    sel = np.sort(rng.permutation(len(y) // 2)[:15000])
    sel = np.concatenate([sel * 2, sel * 2 + 1])  # keep each positive with its negative
    a, b, y, groups = emb[pair_name[sel]], emb[pair_val[sel]], y[sel], groups[sel]
    te = chart_split(groups)
    print("Probe: is this row name from the same row as this value cell? (50/50, chance 50%)")
    for label, x in (("side by side", np.concatenate([a, b], 1)), ("side by side + product", np.concatenate([a, b, a * b], 1))):
        lin, base = fit_probe(x[~te], y[~te], x[te], y[te], 2)
        mlp, _ = fit_probe(x[~te], y[~te], x[te], y[te], 2, hidden=256)
        ctrl, _ = fit_probe(x[~te], y[~te], x[te], y[te], 2, shuffle_labels=True)
        results[label] = {"test": int(te.sum()), "baseline": base, "linear": lin, "mlp": mlp, "control_shuffled": ctrl}
        print(f"  {label:<26} baseline {base * 100:5.1f}% | linear {lin * 100:5.1f}% | MLP {mlp * 100:5.1f}% | control {ctrl * 100:5.1f}%  (n={int(te.sum())})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
