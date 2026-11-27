"""Probes: can the information a question needs be read out of the frozen encoder vectors?

A probe is a very small classifier trained on frozen vectors to predict one property. If it cannot beat the
baseline (always guess the most common answer), the property is not easily readable from the vector.
Nothing here changes the main pipeline: it only READS the saved vectors, tables and recipes.

Probes (text vectors = frozen MiniLM, image vectors = UniChart's frozen encoder):
    retrieval        no training at all: is the answer cell the closest cell to the question?
    number order     given two cells' vectors from one column, which value is larger?
    number size      given one cell's vector, how large is its value (power of ten)?
    question intent  (control, should succeed) question vector -> kind of answer
    operation        question vector -> sum, difference, average, ratio or count
    image structure  chart image vector -> number of data series, number of rows, percent values or not

Every probe is also run with shuffled training labels (a control: it should fall to the baseline).

Usage:
    python probes/run_probes.py
"""
import torch  # must be imported before numpy and pandas on Windows, or its DLLs fail to load
import torch.nn as nn
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from build_labels import cell_number  # noqa: E402
from extract_image_features import load_features  # noqa: E402

EMB = ROOT / "datasets/embeddings"
OUT = ROOT / "probes/results/probe_results.json"
KINDS = ["extract", "compute", "yes_no", "count_rows"]
OPS = ["sum", "difference", "average", "ratio", "count_if"]


def load(name):
    with open(EMB / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def chart_split(hashes, test_fraction=0.2, seed=0):
    """True for items in the test part. Items of the same chart always stay together."""
    uniq = sorted(set(hashes))
    rng = np.random.default_rng(seed)
    test = set(rng.permutation(uniq)[: int(len(uniq) * test_fraction)])
    return np.array([h in test for h in hashes])


def fit_probe(x_tr, y_tr, x_te, y_te, n_classes, hidden=0, shuffle_labels=False, steps=300, seed=0):
    """Accuracy of a linear (hidden=0) or one-hidden-layer probe, plus the baseline of always guessing the commonest class."""
    torch.manual_seed(seed)
    x_tr, x_te = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(x_te, dtype=torch.float32)
    y_tr, y_te = torch.as_tensor(y_tr, dtype=torch.long), torch.as_tensor(y_te, dtype=torch.long)
    baseline = float((y_te == torch.bincount(y_tr, minlength=n_classes).argmax()).float().mean())  # always guess the commonest class
    if shuffle_labels:
        y_tr = y_tr[torch.randperm(len(y_tr))]
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    d = x_tr.shape[1]
    model = nn.Linear(d, n_classes) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2 if hidden == 0 else 3e-3, weight_decay=1e-2)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x_te).argmax(1)
    return float((pred == y_te).float().mean()), baseline


def run_probe(name, x, y, hashes, n_classes, note, results, x_test=None, y_test=None):
    """Linear and small-MLP probes, each with a shuffled-label control. Split by chart unless a test set is given."""
    x, y = np.asarray(x), np.asarray(y)
    if x_test is None:
        te = chart_split(hashes)
        x_tr, y_tr, x_te, y_te = x[~te], y[~te], x[te], y[te]
    else:
        x_tr, y_tr, x_te, y_te = x, y, np.asarray(x_test), np.asarray(y_test)
    lin, base = fit_probe(x_tr, y_tr, x_te, y_te, n_classes)
    mlp, _ = fit_probe(x_tr, y_tr, x_te, y_te, n_classes, hidden=256)
    ctrl, _ = fit_probe(x_tr, y_tr, x_te, y_te, n_classes, shuffle_labels=True)
    results[name] = {"train": int(len(y_tr)), "test": int(len(y_te)), "classes": n_classes, "baseline": base,
                     "linear": lin, "mlp": mlp, "control_shuffled": ctrl, "note": note}
    print(f"  {name:<34} baseline {base * 100:5.1f}% | linear {lin * 100:5.1f}% | small MLP {mlp * 100:5.1f}% | shuffled-label control {ctrl * 100:5.1f}%  (test n={len(y_te)})")


# ---------- retrieval: no training ----------

def retrieval(d, results):
    ranks, chance, gold_kind = [], [], []
    for i, m in enumerate(d["meta"]):
        if m["kind"] != "extract" or not m["gold"]:
            continue
        lo, hi = d["offsets"][i], d["offsets"][i + 1]
        sims = d["cand_emb"][lo:hi].astype(np.float32) @ d["q_emb"][i].astype(np.float32)  # vectors are normalised: cosine
        order = np.argsort(-sims)
        ranks.append(1 + min(int(np.where(order == g)[0][0]) for g in m["gold"]))
        chance.append(len(m["gold"]) / (hi - lo))
        gold_kind.append(m["cand"][m["gold"][0]]["kind"])
    ranks, chance, gold_kind = np.array(ranks), np.array(chance), np.array(gold_kind)
    out = {"questions": int(len(ranks)), "top1": float((ranks == 1).mean()), "top3": float((ranks <= 3).mean()),
           "mean_reciprocal_rank": float((1 / ranks).mean()), "chance_top1": float(chance.mean()), "by_answer_cell": {}}
    for k in ("value", "row_label", "header"):
        sel = gold_kind == k
        if sel.any():
            out["by_answer_cell"][k] = {"questions": int(sel.sum()), "top1": float((ranks[sel] == 1).mean()), "chance_top1": float(chance[sel].mean())}
    results["retrieval"] = out
    print(f"  retrieval (no training): closest cell to the question is the answer cell {out['top1'] * 100:.1f}% of the time "
          f"(chance {out['chance_top1'] * 100:.1f}%), answer in the top 3: {out['top3'] * 100:.1f}%, mean reciprocal rank {out['mean_reciprocal_rank']:.2f}")
    for k, v in out["by_answer_cell"].items():
        print(f"      when the answer is a {k:<9} cell: top-1 {v['top1'] * 100:5.1f}% (chance {v['chance_top1'] * 100:.1f}%, {v['questions']} questions)")


# ---------- numbers ----------

def number_probes(d, results):
    rng = np.random.default_rng(0)
    pairs_a, pairs_b, y, charts = [], [], [], []
    sizes_idx, sizes_y, sizes_chart = [], [], []
    seen = set()
    for i, m in enumerate(d["meta"]):
        if m["chart_hash"] in seen:  # one table per chart
            continue
        seen.add(m["chart_hash"])
        lo = d["offsets"][i]
        cols = {}
        for j, c in enumerate(m["cand"]):
            if c["kind"] != "value":
                continue
            v = cell_number(c["answer"])
            if v is None:
                continue
            cols.setdefault(c["col"], []).append((lo + j, v))
            if v != 0:
                sizes_idx.append(lo + j)
                sizes_y.append(int(np.clip(np.floor(np.log10(abs(v))), -1, 4)) + 1)  # power of ten, 0..5
                sizes_chart.append(m["chart_hash"])
        for items in cols.values():
            if len(items) < 2:
                continue
            idx = rng.permutation(len(items))
            for a, b in zip(idx[::2], idx[1::2]):  # disjoint pairs from the same column
                if items[a][1] == items[b][1]:
                    continue
                pairs_a.append(items[a][0]), pairs_b.append(items[b][0])
                y.append(int(items[a][1] > items[b][1]))
                charts.append(m["chart_hash"])
    emb = d["cand_emb"].astype(np.float32)
    pairs_a, pairs_b = np.array(pairs_a), np.array(pairs_b)
    sel = rng.permutation(len(y))[:30000]
    run_probe("number order (which cell is larger)", np.concatenate([emb[pairs_a[sel]], emb[pairs_b[sel]]], 1),
              np.array(y)[sel], np.array(charts)[sel], 2,
              "two cells from one column; chance is 50%", results)
    sel = rng.permutation(len(sizes_y))[:30000]
    run_probe("number size (power of ten of a value)", emb[np.array(sizes_idx)[sel]], np.array(sizes_y)[sel],
              np.array(sizes_chart)[sel], 6, "one cell; classes are 0.1-1, 1-10, 10-100, 100-1k, 1k-10k, 10k and up", results)


# ---------- the question vector ----------

def question_probes(train, val, results):
    def kinds(d):
        keep = [i for i, m in enumerate(d["meta"]) if m["kind"] in KINDS]
        return d["q_emb"][keep].astype(np.float32), np.array([KINDS.index(d["meta"][i]["kind"]) for i in keep]), keep
    x_tr, y_tr, _ = kinds(train)
    x_te, y_te, _ = kinds(val)
    run_probe("question intent (control)", x_tr, y_tr, None, len(KINDS),
              "question vector -> extract, compute, yes/no or count rows; trained on the training subset, tested on validation",
              results, x_te, y_te)

    def ops(d):
        keep = [i for i, m in enumerate(d["meta"]) if m["kind"] == "compute" and m["detail"]["recipe"]["op"] in OPS]
        return d["q_emb"][keep].astype(np.float32), np.array([OPS.index(d["meta"][i]["detail"]["recipe"]["op"]) for i in keep])
    x_tr, y_tr = ops(train)
    x_te, y_te = ops(val)
    run_probe("calculation type (from the question)", x_tr, y_tr, None, len(OPS),
              "question vector -> sum, difference, average, ratio or count; trained on training, tested on validation", results, x_te, y_te)


# ---------- the image vector ----------

def image_probes(train, results):
    folder = ROOT / "results/image_features"
    if not (folder.exists() and any(folder.glob("features_part_*.npz"))):
        print("  image features not found (git-ignored): skipped")
        return
    hashes, feats = load_features(folder)
    feats = feats.astype(np.float32)
    by_chart = {}
    for m in train["meta"]:
        if m["chart_hash"] not in by_chart:
            n_header = sum(1 for c in m["cand"] if c["kind"] == "header" and c["answer"].strip())
            by_chart[m["chart_hash"]] = {"series": min(max(n_header - 1, 1), 3) - 1, "rows": 0 if m["n_rows"] <= 5 else (1 if m["n_rows"] <= 10 else 2),
                                         "percent": int(any("%" in c["answer"] for c in m["cand"]))}
    keep = [i for i, h in enumerate(hashes) if h in by_chart]
    hs = [hashes[i] for i in keep]
    whole, grid = feats[keep, 0], feats[keep, 1:].reshape(len(keep), -1)
    note = "targets are read from UniChart's table for the chart (so they are approximate)"
    for label, key, k in (("data series: 1, 2 or 3+", "series", 3), ("rows: up to 5, 6 to 10, 11+", "rows", 3), ("values shown as percentages or not", "percent", 2)):
        y = np.array([by_chart[h][key] for h in hs])
        run_probe(f"image, whole-chart vector: {label}", whole, y, hs, k, note, results)
        run_probe(f"image, 4x4 grid vectors: {label}", grid, y, hs, k, note, results)


def main():
    train, val = load("train"), load("val")
    results = {}
    print("Retrieval with the frozen text vectors, no training (training subset):")
    retrieval(train, results)
    print("\nProbes on the frozen text vectors:")
    number_probes(train, results)
    question_probes(train, val, results)
    print("\nProbes on the frozen image vectors:")
    image_probes(train, results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
