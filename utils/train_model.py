"""Train and evaluate the chart-QA answer-selection model.

Frozen parts (not trained): MiniLM text embeddings of the question and every table cell, and, optionally,
UniChart's image features. Trained parts (small):
    router        decides the kind of answer: extract, compute, yes/no or count rows
    extract head  scores each table cell as the answer
    yes/no head   answers yes or no
    compute head  picks the operation (sum, difference, average, ratio, count above/below) and the table cells
                  to apply it to; the arithmetic itself is done in code

Version 1 had no compute head (compute questions got no answer). Version 2 adds it; --no-compute gives version 1.

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
import re
from pathlib import Path

import numpy as np

from answer_and_score import relaxed_correct
from build_labels import cell_number
from embed_examples import N_FEAT, OUT_DIR
from extract_image_features import load_features

ROOT = Path(__file__).resolve().parent.parent
ROUTER_KINDS = ["extract", "compute", "yes_no", "count_rows"]
COMPUTE_OPS = ["sum", "difference", "average", "ratio", "count_if"]
KEYWORDS = [  # words in the question that hint at the operation; given to the compute head as 0/1 flags
    r"\b(sum|total|add|added|combined|together|altogether)\b",
    r"\b(differ\w*|subtract|minus|how much (more|less|higher|lower)|gap|increase\w*|decrease\w*|moved|change\w*)\b",
    r"\b(average|mean)\b",
    r"\b(ratio|times|divid\w*|per)\b",
    r"\bhow many\b|\bnumber of\b",
    r"\b(above|below|over|under|exceed\w*|more than|less than|greater|fewer|at least|at most|equal\w*)\b",
]
N_KW = len(KEYWORDS)
COND_WORDS = [
    (">=", r"\bat least\b|\bor (more|exceeds?)\b|\bgreater than or equal\b|\bexceeds? or\b|\bequals? to .* or exceeds?\b"),
    ("<=", r"\bat most\b|\bno more than\b|\bor (less|fewer)\b"),
    (">", r"\babove\b|\bover\b|\bmore th[ae]n\b|\bgreater th[ae]n\b|\bexceed\w*\b|\bhigher th[ae]n\b|\blarger th[ae]n\b|\bat least\b"),
    ("<", r"\bbelow\b|\bunder\b|\bless th[ae]n\b|\bfewer th[ae]n\b|\blower th[ae]n\b|\bsmaller th[ae]n\b"),
    ("==", r"\bequals?\b|\brepeats?\b|\bsame as\b|\bexactly\b|\bis \d"),
]


# ---------- data ----------

def parse_condition(question):
    """(comparison, number) for 'how many ... above 40' style questions, or None."""
    q = question.lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", q)
    if not numbers:
        return None
    for cmp_, pattern in COND_WORDS:
        if re.search(pattern, q):
            return cmp_, float(numbers[-1])
    return None


def keyword_flags(question):
    return [1.0 if re.search(p, question, re.I) else 0.0 for p in KEYWORDS]


def attach_compute_targets(d):
    """For compute questions: the operation index and the candidate indexes of the table cells the recipe uses."""
    for m in d["meta"]:
        m["op"], m["operands"] = -1, []
        if m["kind"] == "compute":
            recipe = m["detail"]["recipe"]
            if recipe["op"] in COMPUTE_OPS:
                where = {(c["row"], c["col"]): i for i, c in enumerate(m["cand"])}
                cells = [where[tuple(p)] for p in recipe.get("cells", []) if tuple(p) in where]
                if cells:
                    m["op"], m["operands"] = COMPUTE_OPS.index(recipe["op"]), cells


def load_split(name, image_mode, grid=4):
    with open(OUT_DIR / f"{name}.pkl", "rb") as f:
        d = pickle.load(f)
    d["n"] = len(d["meta"])
    attach_compute_targets(d)
    if image_mode != "none":
        base = "results/image_features" + ("" if grid == 4 else f"_{grid}x{grid}")  # 4x4 keeps the original folder names
        folder = ROOT / (base if name == "train" else base + "_val")
        hashes, feats = load_features(folder)
        by_hash = {h: f for h, f in zip(hashes, feats.astype(np.float32))}  # per chart: [whole-chart vector, 16 regions]
        d["img"] = np.stack([by_hash[m["chart_hash"]][0] for m in d["meta"]])    # [N, 1024]
        d["grid"] = np.stack([by_hash[m["chart_hash"]][1:] for m in d["meta"]])  # [N, 16, 1024]
    return d


def standardise_images(train, *others):
    mean, std = train["img"].mean(0), train["img"].std(0) + 1e-6
    flat = train["grid"].reshape(-1, train["grid"].shape[-1])
    gmean, gstd = flat.mean(0), flat.std(0) + 1e-6
    for d in (train, *others):
        d["img"] = ((d["img"] - mean) / std).astype(np.float32)
        d["grid"] = ((d["grid"] - gmean) / gstd).astype(np.float32)


def batch(d, idx, image_mode):
    """Pad the candidates of the chosen questions into tensors."""
    sizes = [d["offsets"][i + 1] - d["offsets"][i] for i in idx]
    cmax = max(sizes)
    dim = d["q_emb"].shape[1]
    cand = np.zeros((len(idx), cmax, dim), dtype=np.float32)
    feat = np.zeros((len(idx), cmax, N_FEAT), dtype=np.float32)
    mask = np.zeros((len(idx), cmax), dtype=bool)
    gold = np.zeros((len(idx), cmax), dtype=bool)
    operand = np.zeros((len(idx), cmax), dtype=bool)
    for b, i in enumerate(idx):
        lo, hi = d["offsets"][i], d["offsets"][i + 1]
        cand[b, :hi - lo] = d["cand_emb"][lo:hi]
        feat[b, :hi - lo] = d["cand_feat"][lo:hi]
        mask[b, :hi - lo] = True
        gold[b, d["meta"][i]["gold"]] = True
        operand[b, d["meta"][i]["operands"]] = True
    q = torch.from_numpy(d["q_emb"][idx].astype(np.float32))
    out = {"q": q, "cand": torch.from_numpy(cand), "feat": torch.from_numpy(feat), "mask": torch.from_numpy(mask),
           "gold": torch.from_numpy(gold), "operand": torch.from_numpy(operand),
           "mean_c": (torch.from_numpy(cand).sum(1) / torch.tensor(sizes).unsqueeze(1)),
           "kw": torch.tensor([keyword_flags(d["meta"][i]["query"]) for i in idx])}
    if image_mode == "mean":
        out["img"] = torch.from_numpy(d["img"][idx])
    elif image_mode == "grid":
        out["grid"] = torch.from_numpy(d["grid"][idx])
    return out


# ---------- model ----------

class Head(nn.Module):
    def __init__(self, dim=384, hid=256, img_dim=1024, image_mode="none", drop=0.2, grid_dim=64):
        super().__init__()
        self.image_mode = image_mode
        if image_mode == "mean":  # version tried first: the whole-chart vector joins the question for every head
            self.img = nn.Sequential(nn.Linear(img_dim, hid), nn.ReLU(), nn.Dropout(drop))
        extra = 0
        if image_mode == "grid":  # the question attends over 16 regions of the chart; only the cell scorers see the result
            self.grid_proj = nn.Sequential(nn.Linear(img_dim, grid_dim), nn.ReLU(), nn.Dropout(drop))
            self.grid_q = nn.Linear(dim, grid_dim)
            extra = grid_dim
        qdim = dim + (hid if image_mode == "mean" else 0)
        self.q_proj = nn.Linear(qdim, hid)
        self.c_proj = nn.Linear(dim, hid)

        def mlp(n_in, n_out):
            return nn.Sequential(nn.Linear(n_in, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, n_out))

        self.scorer = mlp(hid * 4 + N_FEAT + extra, 1)             # extract head: one score per table cell
        self.operand_scorer = mlp(hid * 4 + N_FEAT + extra, 1)     # compute head: is this cell used in the calculation
        self.router = mlp(qdim + dim, len(ROUTER_KINDS))
        self.yesno = mlp(qdim + dim, 2)
        self.op_head = mlp(qdim + dim + N_KW, len(COMPUTE_OPS))

    def forward(self, b):
        q = torch.cat([b["q"], self.img(b["img"])], -1) if self.image_mode == "mean" else b["q"]
        qh = self.q_proj(q).unsqueeze(1).expand(-1, b["cand"].size(1), -1)
        ch = self.c_proj(b["cand"])
        parts = [qh * ch, (qh - ch).abs(), ch, qh, b["feat"]]
        if self.image_mode == "grid":
            regions = self.grid_proj(b["grid"])                                                  # [B, 16, dim]
            attn = torch.softmax((self.grid_q(b["q"]).unsqueeze(1) * regions).sum(-1) / regions.size(-1) ** 0.5, -1)
            context = (attn.unsqueeze(-1) * regions).sum(1)                                      # what the question looks at
            parts.append(context.unsqueeze(1).expand(-1, b["cand"].size(1), -1))
        x = torch.cat(parts, -1)
        scores = self.scorer(x).squeeze(-1).masked_fill(~b["mask"], -1e9)
        operand = self.operand_scorer(x).squeeze(-1)
        pooled = torch.cat([q, b["mean_c"]], -1)
        return {"scores": scores, "operand": operand, "router": self.router(pooled), "yesno": self.yesno(pooled),
                "op": self.op_head(torch.cat([pooled, b["kw"]], -1))}


def losses(model, b, kinds, yn, ops, use_compute=True):
    """kinds: router class per question (-1 = no label); yn: 0/1 for yes-no questions (-1 otherwise);
    ops: compute operation per question (-1 otherwise)."""
    out = model(b)
    total = out["scores"].new_zeros(())
    is_extract = kinds == ROUTER_KINDS.index("extract")
    if is_extract.any():
        logp = torch.log_softmax(out["scores"][is_extract], -1)
        total = total - torch.logsumexp(logp.masked_fill(~b["gold"][is_extract], -1e9), -1).mean()
    has_kind = kinds >= 0
    if has_kind.any():
        total = total + nn.functional.cross_entropy(out["router"][has_kind], kinds[has_kind])
    has_yn = yn >= 0
    if has_yn.any():
        total = total + nn.functional.cross_entropy(out["yesno"][has_yn], yn[has_yn])
    has_op = ops >= 0
    if use_compute and has_op.any():
        total = total + nn.functional.cross_entropy(out["op"][has_op], ops[has_op])
        logits, target, valid = out["operand"][has_op], b["operand"][has_op].float(), b["mask"][has_op]
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=torch.tensor(4.0))
        total = total + (bce * valid).sum() / valid.sum()
    return total


def targets(d, idx):
    kinds = torch.tensor([ROUTER_KINDS.index(d["meta"][i]["kind"]) if d["meta"][i]["kind"] in ROUTER_KINDS else -1 for i in idx])
    yn = torch.tensor([(d["meta"][i]["detail"]["answer"] == "yes") * 1 if d["meta"][i]["kind"] == "yes_no" else -1 for i in idx])
    ops = torch.tensor([d["meta"][i]["op"] for i in idx])
    return kinds, yn, ops


# ---------- answering and scoring ----------

def mention_order(m, picked, question):
    """Order picked cells by where their row names appear in the question (for ratios: the first named is the top)."""
    q = question.lower()
    names = {c["row"]: c["answer"].lower() for c in m["cand"] if c["kind"] == "row_label"}

    def key(i):
        c = m["cand"][i]
        pos = q.find(names.get(c["row"], "\0"))
        return (pos if pos >= 0 else 10 ** 6, c["row"], c["col"])

    return sorted(picked, key=key)


def compute_answer(m, op_logits, operand_logits, numeric):
    """Run the predicted operation on the predicted cells. Returns an answer string, or '' if it cannot be done."""
    op = COMPUTE_OPS[int(op_logits.argmax())]
    probs = torch.sigmoid(operand_logits).masked_fill(~numeric, 0.0)
    ranked = [int(i) for i in probs.argsort(descending=True) if numeric[int(i)]]
    if not ranked:
        return ""
    if op in ("difference", "ratio"):
        picked = ranked[:2]
    else:
        picked = [i for i in ranked if probs[i] > 0.5] or ranked[:2 if op in ("sum", "average") else 1]
    if op in ("difference", "ratio", "sum", "average") and len(picked) < 2:
        return ""
    nums = {i: cell_number(m["cand"][i]["answer"]) for i in picked}
    if op == "ratio":
        picked = mention_order(m, picked, m["query"])
    vals = [nums[i] for i in picked]
    if op == "sum":
        result = sum(vals)
    elif op == "average":
        result = sum(vals) / len(vals)
    elif op == "difference":
        result = abs(vals[0] - vals[1])
    elif op == "ratio":
        if not vals[1]:
            return ""
        result = vals[0] / vals[1]
    else:  # count_if
        cond = parse_condition(m["query"])
        if cond is None:
            return ""
        cmp_, value = cond
        test = {">": lambda a: a > value, "<": lambda a: a < value, ">=": lambda a: a >= value,
                "<=": lambda a: a <= value, "==": lambda a: a == value}[cmp_]
        result = sum(test(v) for v in vals)
    return f"{result:.6g}"


@torch.no_grad()
def predict(model, d, image_mode, oracle=False, use_compute=True):
    """Answer string for every question. oracle=True uses the true answer kind instead of the router (to isolate the heads)."""
    model.eval()
    answers, routed = [], []
    for start in range(0, d["n"], 64):
        idx = list(range(start, min(start + 64, d["n"])))
        b = batch(d, idx, image_mode)
        out = model(b)
        numeric = b["mask"] & (b["feat"][:, :, 3] > 0)
        for k, i in enumerate(idx):
            m = d["meta"][i]
            kind = m["kind"] if oracle and m["kind"] in ROUTER_KINDS else ROUTER_KINDS[int(out["router"][k].argmax())]
            if oracle and m["kind"] not in ROUTER_KINDS:
                kind = "extract"  # unusable questions still get a guess from the extract head
            routed.append(kind)
            if kind == "extract":
                answers.append(m["cand"][int(out["scores"][k].argmax())]["answer"])
            elif kind == "yes_no":
                answers.append("yes" if int(out["yesno"][k].argmax()) == 1 else "no")
            elif kind == "count_rows":
                answers.append(str(m["n_rows"]))
            elif use_compute:
                answers.append(compute_answer(m, out["op"][k], out["operand"][k], numeric[k]))
            else:
                answers.append("")
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
    parser.add_argument("--use-image", action="store_true", help="same as --image-mode mean")
    parser.add_argument("--image-mode", choices=["none", "mean", "grid"], default="none",
                        help="none: no image; mean: whole-chart vector; grid: question attends over 16 chart regions")
    parser.add_argument("--image-grid", type=int, default=4, help="regions per side in the saved image features (4 or 8)")
    parser.add_argument("--no-compute", action="store_true", help="version 1: compute questions get no answer")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    use_compute = not args.no_compute
    image_mode = "mean" if args.use_image else args.image_mode

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    train, val = load_split("train", image_mode, args.image_grid), load_split("val", image_mode, args.image_grid)
    if image_mode != "none":
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
    print(f"training questions {len(tr_idx)}, dev questions {len(dev_use)}, image mode: {image_mode}, compute head: {use_compute}")

    model = Head(image_mode=image_mode)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(tr_idx)
        for s in range(0, len(tr_idx), 64):
            idx = tr_idx[s:s + 64]
            kinds, yn, ops = targets(train, idx)
            loss = losses(model, batch(train, idx, image_mode), kinds, yn, ops, use_compute)
            opt.zero_grad(), loss.backward(), opt.step()
        # dev score: top-1 of the extract head on extract questions, router accuracy, and operation accuracy
        model.eval()
        with torch.no_grad():
            b = batch(train, dev_use, image_mode)
            out = model(b)
            kinds, _, ops = targets(train, dev_use)
            ex = kinds == ROUTER_KINDS.index("extract")
            top1 = b["gold"][ex].float().gather(1, out["scores"][ex].argmax(1, keepdim=True)).mean().item()
            racc = (out["router"].argmax(1) == kinds).float().mean().item()
            has_op = ops >= 0
            opacc = (out["op"][has_op].argmax(1) == ops[has_op]).float().mean().item() if has_op.any() else 0.0
        score = top1 + racc + (opacc if use_compute else 0.0)
        if score > best:
            best, best_state, best_epoch = score, {k: v.clone() for k, v in model.state_dict().items()}, epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:>3}: loss {loss.item():.3f} | dev extract top-1 {top1:.3f} | dev router accuracy {racc:.3f} | dev operation accuracy {opacc:.3f}")
    model.load_state_dict(best_state)
    print(f"best epoch {best_epoch}")

    results = {}
    for name, oracle in (("router", False), ("oracle", True)):
        answers, routed = predict(model, val, image_mode, oracle=oracle, use_compute=use_compute)
        res, ok = evaluate(answers, val)
        results[name] = res
        if not oracle:
            answers_path = (args.out or ROOT / "results" / f"model_{'v2' if use_compute else 'v1'}_{image_mode}").with_suffix(".answers.jsonl")
            with open(answers_path, "w", encoding="utf-8") as f:
                for m, a_, o_, r_ in zip(val["meta"], answers, ok, routed):
                    f.write(json.dumps({"row_idx": m["row_idx"], "query": m["query"], "label": m["label"], "true_kind": m["kind"],
                                        "routed_kind": r_, "answer": a_, "correct": bool(o_), "group": m["human_or_machine"]}) + "\n")
        print(f"\nvalidation (300 questions), kind chosen by {'the true label (oracle)' if oracle else 'the router'}:")
        for k, v in res.items():
            print(f"  {k:<22} {v * 100:5.1f}%")
        if not oracle:
            true = [m["kind"] for m in val["meta"]]
            usable_val = [(t, r) for t, r in zip(true, routed) if t in ROUTER_KINDS]
            racc = np.mean([t == r for t, r in usable_val])
            print(f"  router accuracy on labelled questions: {racc * 100:.1f}% ({len(usable_val)} questions)")
            results["router_accuracy"] = racc

    version = "v2" if use_compute else "v1"
    out_path = args.out or ROOT / "results" / f"model_{version}_{ {'none': 'text', 'mean': 'image', 'grid': 'grid' + str(args.image_grid)}[image_mode] }.json"
    json.dump({k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v)) for k, v in results.items()},
              open(out_path, "w"), indent=1)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
