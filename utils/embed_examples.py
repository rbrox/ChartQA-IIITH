"""Embed every question and table candidate with a frozen MiniLM, and cache the result.

For each split (train subset, validation sample) this saves, under datasets/embeddings/:
    question embeddings, candidate embeddings, 12 simple per-candidate features, and the metadata the
    model needs (candidate kinds and positions, the raw answer text of each cell, gold candidates, true labels).

The embeddings are computed once; training the small head afterwards is fast.

Usage:
    python utils/embed_examples.py
"""
import torch  # noqa: F401  (must be imported before numpy and pandas on Windows, or its DLLs fail to load)
import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np

from build_candidates import KINDS, build_examples
from build_labels import cell_number

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = ROOT / "datasets/embeddings"
BASE_FEAT = 12  # 3 for the candidate kind + 9 numeric and position features
N_FEAT = BASE_FEAT + 5  # plus 5 question-match features

SPLITS = {
    "train": (ROOT / "results/unichart_train_3000.jsonl", ROOT / "results/unichart_train_3000_final_labels.jsonl"),
    "val": (ROOT / "results/unichart_val_300.jsonl", ROOT / "results/unichart_val_300_final_labels.jsonl"),
}


def _rank_features(values):
    """For a list of numbers: (percentile 0..1, is the maximum, is the minimum) for each."""
    n = len(values)
    order = np.argsort(np.argsort(values))
    pct = order / (n - 1) if n > 1 else np.full(n, 0.5)
    return pct, np.array(values) == max(values), np.array(values) == min(values)


def candidate_features(cands, n_rows):
    """[len(cands), BASE_FEAT] float32. Gives the model numbers it cannot get from text embeddings:
    where a value ranks within its column, whether it is the largest or smallest, and where its row ranks."""
    feats = np.zeros((len(cands), BASE_FEAT), dtype=np.float32)
    n_cols = max([c["col"] for c in cands] + [0]) + 1
    col_values, row_first = {}, {}
    for i, c in enumerate(cands):
        feats[i, KINDS.index(c["kind"])] = 1.0
        if c["kind"] == "value":
            v = cell_number(c["answer"])
            if v is not None:
                col_values.setdefault(c["col"], []).append((i, v))
                row_first.setdefault(c["row"], v)
    for col, items in col_values.items():
        pct, is_max, is_min = _rank_features([v for _, v in items])
        for (i, _), p, mx, mn in zip(items, pct, is_max, is_min):
            feats[i, 4], feats[i, 5], feats[i, 6] = p, mx, mn  # col_pct, col_is_max, col_is_min
            feats[i, 3] = 1.0  # has_number
    if row_first:
        rows = sorted(row_first)
        pct, is_max, is_min = _rank_features([row_first[r] for r in rows])
        by_row = {r: (p, mx, mn) for r, p, mx, mn in zip(rows, pct, is_max, is_min)}
        for i, c in enumerate(cands):
            if c["row"] in by_row:
                feats[i, 7], feats[i, 8], feats[i, 9] = by_row[c["row"]]  # row_pct, row_is_max, row_is_min
    for i, c in enumerate(cands):
        feats[i, 10] = c["row"] / max(n_rows, 1)
        feats[i, 11] = c["col"] / n_cols
    return feats


def _words(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def match_features(example, q_vec, cand_vecs):
    """[len(cands), 5]: how well the question matches each candidate, its row name and its column name.

    cosine similarity of the question with the candidate, its row label and its column header,
    and the share of the row label's and column header's words that appear in the question.
    """
    cands = example["candidates"]
    row_idx = {c["row"]: i for i, c in enumerate(cands) if c["kind"] == "row_label"}
    col_idx = {c["col"]: i for i, c in enumerate(cands) if c["kind"] == "header"}
    q_words = _words(example["query"])
    sim = cand_vecs.astype(np.float32) @ q_vec.astype(np.float32)  # vectors are normalised, so this is the cosine
    out = np.zeros((len(cands), 5), dtype=np.float32)
    for i, c in enumerate(cands):
        out[i, 0] = sim[i]
        r, h = row_idx.get(c["row"]), col_idx.get(c["col"])
        if r is not None:
            words = _words(cands[r]["answer"])
            out[i, 1] = sim[r]
            out[i, 3] = len(words & q_words) / len(words) if words else 0.0
        if h is not None:
            words = _words(cands[h]["answer"])
            out[i, 2] = sim[h]
            out[i, 4] = len(words & q_words) / len(words) if words else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    for split, (tables, labels) in SPLITS.items():
        examples = build_examples(tables, labels)
        texts = sorted({e["query"] for e in examples} | {c["text"] for e in examples for c in e["candidates"]})
        index = {t: i for i, t in enumerate(texts)}
        print(f"{split}: {len(examples)} questions, {len(texts)} distinct strings to embed")
        start = time.time()
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
        print(f"  embedded in {time.time() - start:.0f}s")

        q_emb = np.stack([emb[index[e["query"]]] for e in examples])
        offsets = np.cumsum([0] + [len(e["candidates"]) for e in examples])
        cand_emb = np.concatenate([emb[[index[c["text"]] for c in e["candidates"]]] for e in examples])
        cand_feat = np.concatenate([
            np.concatenate([candidate_features(e["candidates"], e["n_rows"]),
                            match_features(e, q_emb[k], cand_emb[offsets[k]:offsets[k + 1]])], axis=1)
            for k, e in enumerate(examples)])
        meta = [{k: v for k, v in e.items() if k != "candidates"} | {
            "cand": [{k: c[k] for k in ("answer", "kind", "row", "col")} for c in e["candidates"]]} for e in examples]
        with open(args.out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump({"q_emb": q_emb, "cand_emb": cand_emb, "cand_feat": cand_feat, "offsets": offsets, "meta": meta}, f)
        print(f"  saved {args.out_dir / (split + '.pkl')}")


if __name__ == "__main__":
    main()
