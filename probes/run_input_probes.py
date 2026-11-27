"""Input-level probes: which PARTS of the input (the extracted table and the chart image) are missing from the frozen vectors?

Each probe predicts one property of the input from a frozen vector, with the answer key taken from the extracted table.
A probe that stays near the baseline means that part of the input is not easily readable from the vector.

Text (frozen MiniLM vector of one table cell):
    leading digit of the number      the exact digits, not only the size
    number of decimal places         formatting detail
    position of a row name           first, middle or last third of the table
    largest value in its column      a single cell, balanced 50/50 (chance 50%)
Image (UniChart's frozen encoder, kept as one whole-chart vector, a 4x4 grid or an 8x8 grid):
    size of the largest value        power of ten
    trend                            is the last row's value above the first row's (series 1)
    where the largest value is       first, middle or last third of the rows
Comparing the three image forms shows whether detail is lost by our pooling (whole vector) rather than by the encoder.

Nothing here changes the main pipeline: it only READS saved vectors and tables.

Usage:
    python probes/run_input_probes.py
"""
import torch  # must be imported before numpy and pandas on Windows, or its DLLs fail to load
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_probes import ROOT, chart_split, fit_probe, load, load_features, cell_number  # noqa: E402

OUT = ROOT / "probes/results/input_probe_results.json"


def run(name, x, y, groups, n_classes, note, results, mlp=True, test_split=None):
    x, y = np.asarray(x), np.asarray(y)
    te = chart_split(groups) if test_split is None else test_split
    x_tr, y_tr, x_te, y_te = x[~te], y[~te], x[te], y[te]
    lin, base = fit_probe(x_tr, y_tr, x_te, y_te, n_classes)
    mlp_acc = fit_probe(x_tr, y_tr, x_te, y_te, n_classes, hidden=256)[0] if mlp else None
    ctrl, _ = fit_probe(x_tr, y_tr, x_te, y_te, n_classes, shuffle_labels=True)
    results[name] = {"train": int(len(y_tr)), "test": int(len(y_te)), "classes": n_classes, "baseline": base, "linear": lin,
                     "mlp": mlp_acc, "control_shuffled": ctrl, "note": note}
    m = f"{mlp_acc * 100:5.1f}%" if mlp_acc is not None else "  n/a "
    print(f"  {name:<58} baseline {base * 100:5.1f}% | linear {lin * 100:5.1f}% | MLP {m} | control {ctrl * 100:5.1f}%  (n={len(y_te)})")


def decimals_of(text):
    m = re.search(r"\.(\d+)", text)
    return min(len(m.group(1)) if m else 0, 2)


# ---------- text: one table cell at a time ----------

def text_probes(d, results):
    rng = np.random.default_rng(0)
    emb = d["cand_emb"].astype(np.float32)
    lead, lead_y, lead_g = [], [], []
    dec_y = []
    pos_idx, pos_y, pos_g = [], [], []
    mx_idx, mx_y, mx_g = [], [], []
    seen = set()
    for i, m in enumerate(d["meta"]):
        if m["chart_hash"] in seen:
            continue
        seen.add(m["chart_hash"])
        lo = d["offsets"][i]
        n_rows = m["n_rows"]
        cols = {}
        for j, c in enumerate(m["cand"]):
            if c["kind"] == "value":
                v = cell_number(c["answer"])
                if v is None:
                    continue
                cols.setdefault(c["col"], []).append((lo + j, v))
                if v != 0:
                    digits = re.sub(r"[^1-9]", "", f"{abs(v):.10g}".replace("e", "0"))  # first non-zero digit
                    lead.append(lo + j), lead_y.append(int(digits[0]) - 1), lead_g.append(m["chart_hash"])
                    dec_y.append(decimals_of(c["answer"]))
            elif c["kind"] == "row_label" and n_rows >= 3:
                frac = (c["row"] - 1) / max(n_rows - 1, 1)
                pos_idx.append(lo + j), pos_y.append(0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)), pos_g.append(m["chart_hash"])
        for items in cols.values():
            if len(items) >= 3:
                vals = np.array([v for _, v in items])
                for (idx, v) in items:
                    mx_idx.append(idx), mx_y.append(int(v == vals.max() and (vals == v).sum() == 1)), mx_g.append(m["chart_hash"])
    lead, lead_y, lead_g = np.array(lead), np.array(lead_y), np.array(lead_g)
    sel = rng.permutation(len(lead))[:30000]
    run("text: leading digit of the number (1 to 9)", emb[lead[sel]], lead_y[sel], lead_g[sel], 9, "exact digits, not only size", results)
    run("text: number of decimal places (0, 1, 2+)", emb[lead[sel]], np.array(dec_y)[sel], lead_g[sel], 3, "formatting detail", results)
    pos_idx, pos_y, pos_g = np.array(pos_idx), np.array(pos_y), np.array(pos_g)
    sel = rng.permutation(len(pos_idx))[:30000]
    run("text: position of a row name (first, middle, last third)", emb[pos_idx[sel]], pos_y[sel], pos_g[sel], 3, "a row-name cell on its own", results)
    mx_idx, mx_y, mx_g = np.array(mx_idx), np.array(mx_y), np.array(mx_g)
    pos_sel, neg_sel = np.where(mx_y == 1)[0], np.where(mx_y == 0)[0]
    neg_sel = rng.permutation(neg_sel)[: len(pos_sel)]  # balance 50/50
    both = np.concatenate([pos_sel, neg_sel])
    run("text: is this cell the largest in its column (50/50)", emb[mx_idx[both]], mx_y[both], mx_g[both], 2, "one cell; chance 50%", results)


# ---------- image: one chart at a time ----------

def image_probes(d, results):
    by_chart = {}
    for m in d["meta"]:
        if m["chart_hash"] in by_chart:
            continue
        col1 = sorted([(c["row"], cell_number(c["answer"])) for c in m["cand"] if c["kind"] == "value" and c["col"] == 1 and cell_number(c["answer"]) is not None])
        if len(col1) < 3:
            continue
        vals = np.array([v for _, v in col1])
        if vals.max() <= 0 or vals[0] == vals[-1]:
            continue
        frac = int(np.argmax(vals)) / (len(vals) - 1)
        by_chart[m["chart_hash"]] = {"size": int(np.clip(np.floor(np.log10(vals.max())), -1, 4)) + 1, "trend": int(vals[-1] > vals[0]),
                                     "where": 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2), "unique_max": int((vals == vals.max()).sum() == 1)}
    forms = {}
    for label, folder, grid in (("whole-chart vector", "image_features", None), ("4x4 grid", "image_features", 4), ("8x8 grid", "image_features_8x8", 8)):
        path = ROOT / "results" / folder
        if not (path.exists() and any(path.glob("features_part_*.npz"))):
            print(f"  {label}: features not found, skipped")
            continue
        hashes, feats = load_features(path)
        feats = feats.astype(np.float32)
        keep = [i for i, h in enumerate(hashes) if h in by_chart]
        forms[label] = ([hashes[i] for i in keep], feats[keep, 0] if grid is None else feats[keep, 1:].reshape(len(keep), -1))
    for target, k, desc in (("size", 6, "size of the largest value (power of ten)"), ("trend", 2, "is the last row's value above the first row's (series 1)"),
                            ("where", 3, "where the largest value is: first, middle or last third of the rows")):
        for label, (hs, x) in forms.items():
            y = np.array([by_chart[h][target] for h in hs])
            ok = np.array([by_chart[h]["unique_max"] == 1 for h in hs]) if target == "where" else np.ones(len(hs), bool)
            run(f"image, {label}: {desc}", x[ok], y[ok], np.array(hs)[ok], k, "answer key from UniChart's table", results, mlp=(label != "8x8 grid"))


def main():
    train = load("train")
    results = {}
    print("Text probes (frozen MiniLM vector of one cell):")
    text_probes(train, results)
    print("\nImage probes (frozen UniChart encoder):")
    image_probes(train, results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
