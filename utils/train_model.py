"""Train and evaluate the first version of the chart-QA answer-selection model.

Frozen parts (not trained): MiniLM text embeddings of the question and every table cell, and, optionally,
UniChart's image features. Trained parts (small): an extract head that scores each table cell as the answer,
a router that decides the kind of answer (extract, compute, yes/no, count rows), and a yes/no head.

Version 1 answers: extract, yes/no and count-rows questions. Compute questions get no answer yet.

Usage:
    python utils/embed_examples.py          # once
    python utils/train_model.py             # table and question only
    python utils/train_model.py --use-image # also uses UniChart image features
"""
import torch  # must be imported before numpy and pandas on Windows, or its DLLs fail to load
import torch.nn as nn
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

from answer_and_score import relaxed_correct
from embed_examples import N_FEAT, OUT_DIR
from extract_image_features import load_features

ROOT = Path(__file__).resolve().parent.parent
ROUTER_KINDS = ["extract", "compute", "yes_no", "count_rows"]


# ---------- data ----------

def load_split(name, use_image):
    with open(OUT_DIR / f"{name}.pkl", "rb") as f:
        d = pickle.load(f)
    d["n"] = len(d["meta"])
    if use_image:
        folder = ROOT / ("results/image_features" if name == "train" else "results/image_features_val")
        hashes, feats = load_features(folder)
        by_hash = {h: f[0] for h, f in zip(hashes, feats.astype(np.float32))}  # whole-chart vector
        d["img"] = np.stack([by_hash[m["chart_hash"]] for m in d["meta"]])
    return d


def standardise_images(train, *others):
    mean, std = train["img"].mean(0), train["img"].std(0) + 1e-6
    for d in (train, *others):
        d["img"] = ((d["img"] - mean) / std).astype(np.float32)


def batch(d, idx, use_image):
    """Pad the candidates of the chosen questions into tensors."""
    sizes = [d["offsets"][i + 1] - d["offsets"][i] for i in idx]
    cmax = max(sizes)
    dim = d["q_emb"].shape[1]
    cand = np.zeros((len(idx), cmax, dim), dtype=np.float32)
    feat = np.zeros((len(idx), cmax, N_FEAT), dtype=np.float32)
    mask = np.zeros((len(idx), cmax), dtype=bool)
    gold = np.zeros((len(idx), cmax), dtype=bool)
    for b, i in enumerate(idx):
        lo, hi = d["offsets"][i], d["offsets"][i + 1]
        cand[b, :hi - lo] = d["cand_emb"][lo:hi]
        feat[b, :hi - lo] = d["cand_feat"][lo:hi]
        mask[b, :hi - lo] = True
        gold[b, d["meta"][i]["gold"]] = True
    q = torch.from_numpy(d["q_emb"][idx].astype(np.float32))
    out = {"q": q, "cand": torch.from_numpy(cand), "feat": torch.from_numpy(feat), "mask": torch.from_numpy(mask),
           "gold": torch.from_numpy(gold), "mean_c": (torch.from_numpy(cand).sum(1) / torch.tensor(sizes).unsqueeze(1))}
    if use_image:
        out["img"] = torch.from_numpy(d["img"][idx])
    return out


# ---------- model ----------

class Head(nn.Module):
    def __init__(self, dim=384, hid=256, img_dim=1024, use_image=False, drop=0.2):
        super().__init__()
        self.use_image = use_image
        if use_image:
            self.img = nn.Sequential(nn.Linear(img_dim, hid), nn.ReLU(), nn.Dropout(drop))
        qdim = dim + (hid if use_image else 0)
        self.q_proj = nn.Linear(qdim, hid)
        self.c_proj = nn.Linear(dim, hid)
        self.scorer = nn.Sequential(nn.Linear(hid * 4 + N_FEAT, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, 1))
        self.router = nn.Sequential(nn.Linear(qdim + dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, len(ROUTER_KINDS)))
        self.yesno = nn.Sequential(nn.Linear(qdim + dim, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, 2))

    def forward(self, b):
        q = torch.cat([b["q"], self.img(b["img"])], -1) if self.use_image else b["q"]
        qh = self.q_proj(q).unsqueeze(1).expand(-1, b["cand"].size(1), -1)
        ch = self.c_proj(b["cand"])
        x = torch.cat([qh * ch, (qh - ch).abs(), ch, qh, b["feat"]], -1)
        scores = self.scorer(x).squeeze(-1).masked_fill(~b["mask"], -1e9)
        pooled = torch.cat([q, b["mean_c"]], -1)
        return scores, self.router(pooled), self.yesno(pooled)


def losses(model, b, kinds, yn):
    """kinds: router class per question (-1 = no label); yn: 0/1 for yes-no questions (-1 otherwise)."""
    scores, router, yesno = model(b)
    total = scores.new_zeros(())
    is_extract = kinds == ROUTER_KINDS.index("extract")
    if is_extract.any():
        logp = torch.log_softmax(scores[is_extract], -1)
        gold = b["gold"][is_extract]
        total = total - torch.logsumexp(logp.masked_fill(~gold, -1e9), -1).mean()
    has_kind = kinds >= 0
    if has_kind.any():
        total = total + nn.functional.cross_entropy(router[has_kind], kinds[has_kind])
    has_yn = yn >= 0
    if has_yn.any():
        total = total + nn.functional.cross_entropy(yesno[has_yn], yn[has_yn])
    return total


def targets(d, idx):
    kinds = torch.tensor([ROUTER_KINDS.index(d["meta"][i]["kind"]) if d["meta"][i]["kind"] in ROUTER_KINDS else -1 for i in idx])
    yn = torch.tensor([(d["meta"][i]["detail"]["answer"] == "yes") * 1 if d["meta"][i]["kind"] == "yes_no" else -1 for i in idx])
    return kinds, yn


# ---------- answering and scoring ----------

@torch.no_grad()
def predict(model, d, use_image, oracle=False):
    """Answer string for every question. oracle=True uses the true answer kind instead of the router (to isolate the heads)."""
    model.eval()
    answers, routed = [], []
    for start in range(0, d["n"], 64):
        idx = list(range(start, min(start + 64, d["n"])))
        b = batch(d, idx, use_image)
        scores, router, yesno = model(b)
        for k, i in enumerate(idx):
            m = d["meta"][i]
            kind = m["kind"] if oracle and m["kind"] in ROUTER_KINDS else ROUTER_KINDS[int(router[k].argmax())]
            if oracle and m["kind"] not in ROUTER_KINDS:
                kind = "extract"  # unusable questions still get a guess from the extract head
            routed.append(kind)
            if kind == "extract":
                answers.append(m["cand"][int(scores[k].argmax())]["answer"])
            elif kind == "yes_no":
                answers.append("yes" if int(yesno[k].argmax()) == 1 else "no")
            elif kind == "count_rows":
                answers.append(str(m["n_rows"]))
            else:
                answers.append("")  # compute is not answered in version 1
    return answers, routed


def evaluate(answers, d):
    ok = np.array([bool(a) and relaxed_correct(a, m["label"]) for a, m in zip(answers, d["meta"])])
    group = np.array([m["human_or_machine"] for m in d["meta"]])
    kind = np.array([m["kind"] for m in d["meta"]])
    res = {"overall": ok.mean(), "human-written": ok[group == 0].mean(), "machine": ok[group == 1].mean()}
    for k in ["extract", "compute", "yes_no", "count_rows", "unusable"]:
        if (kind == k).any():
            res[f"true kind = {k}"] = ok[kind == k].mean()
    return res, ok


# ---------- training ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-image", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    train, val = load_split("train", args.use_image), load_split("val", args.use_image)
    if args.use_image:
        standardise_images(train, val)

    # hold out 10% of the training charts as a dev set for choosing the epoch
    charts = sorted({m["chart_hash"] for m in train["meta"]})
    random.Random(args.seed).shuffle(charts)
    dev_charts = set(charts[: len(charts) // 10])
    dev_idx = [i for i, m in enumerate(train["meta"]) if m["chart_hash"] in dev_charts]
    tr_idx = [i for i, m in enumerate(train["meta"]) if m["chart_hash"] not in dev_charts]
    usable = lambda ids: [i for i in ids if train["meta"][i]["kind"] in ROUTER_KINDS and (
        train["meta"][i]["kind"] != "extract" or train["meta"][i]["gold"])]
    tr_idx, dev_use = usable(tr_idx), usable(dev_idx)
    print(f"training questions {len(tr_idx)}, dev questions {len(dev_use)}, image features: {args.use_image}")

    model = Head(use_image=args.use_image)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(tr_idx)
        for s in range(0, len(tr_idx), 64):
            idx = tr_idx[s:s + 64]
            kinds, yn = targets(train, idx)
            loss = losses(model, batch(train, idx, args.use_image), kinds, yn)
            opt.zero_grad(), loss.backward(), opt.step()
        # dev score: top-1 of the extract head on extract questions, plus router accuracy
        model.eval()
        with torch.no_grad():
            b = batch(train, dev_use, args.use_image)
            scores, router, _ = model(b)
            kinds, _ = targets(train, dev_use)
            ex = kinds == ROUTER_KINDS.index("extract")
            top1 = b["gold"][ex].float().gather(1, scores[ex].argmax(1, keepdim=True)).mean().item()
            racc = (router.argmax(1) == kinds).float().mean().item()
        score = top1 + racc
        if score > best:
            best, best_state, best_epoch = score, {k: v.clone() for k, v in model.state_dict().items()}, epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:>3}: loss {loss.item():.3f} | dev extract top-1 {top1:.3f} | dev router accuracy {racc:.3f}")
    model.load_state_dict(best_state)
    print(f"best epoch {best_epoch} (dev extract top-1 + router accuracy = {best:.3f})")

    results = {}
    for name, oracle in (("router", False), ("oracle", True)):
        answers, routed = predict(model, val, args.use_image, oracle=oracle)
        res, _ = evaluate(answers, val)
        results[name] = res
        print(f"\nvalidation (300 questions), kind chosen by {'the true label (oracle)' if oracle else 'the router'}:")
        for k, v in res.items():
            print(f"  {k:<22} {v * 100:5.1f}%")
        if not oracle:
            true = [m["kind"] for m in val["meta"]]
            usable_val = [(t, r) for t, r in zip(true, routed) if t in ROUTER_KINDS]
            racc = np.mean([t == r for t, r in usable_val])
            print(f"  router accuracy on labelled questions: {racc * 100:.1f}% ({len(usable_val)} questions)")
            results["router_accuracy"] = racc

    out = args.out or ROOT / "results" / f"model_v1_{'image' if args.use_image else 'text'}.json"
    json.dump({k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v)) for k, v in results.items()},
              open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
