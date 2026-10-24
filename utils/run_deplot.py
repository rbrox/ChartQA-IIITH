"""Run DePlot (chart -> table) on the shared validation sample.

Reads the row list from results/val_sample_300.csv, runs DePlot once per unique
chart, and appends one JSON line per row to the output file. Safe to stop and
re-run: rows already in the output file are skipped.

Usage (local):
    python utils/run_deplot.py --limit 10
Usage (Kaggle):
    python utils/run_deplot.py \
        --val-path /kaggle/input/<dataset>/val-00000-of-00001-0f11003c77497969.parquet \
        --sample-csv /kaggle/input/<dataset>/val_sample_300.csv \
        --out /kaggle/working/deplot_val_300.jsonl
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "datasets/HuggingFaceM4/ChartQA/data/val-00000-of-00001-0f11003c77497969.parquet"
SAMPLE_CSV = ROOT / "results/val_sample_300.csv"
PROMPT = "Generate underlying data table of the figure below:"


def load_deplot(device: str):
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    proc = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device).eval()
    return proc, model


def deplot_table(proc, model, device: str, img: Image.Image) -> str:
    import torch

    inputs = proc(images=img, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)
    return proc.decode(out[0], skip_special_tokens=True).replace("<0x0A>", "\n")


def main():
    # When run with `!python` in a notebook, stdout is a pipe and Python buffers it, so progress would appear only at the end.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--sample-csv", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--out", type=Path, default=ROOT / "results/deplot_val_300.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N sample rows")
    parser.add_argument("--dry-run", action="store_true", help="skip the model, only check data/resume logic")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample_csv)
    if args.limit:
        sample = sample.head(args.limit)

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["row_idx"] for line in f if line.strip()}
    todo = sample[~sample["row_idx"].isin(done)]
    print(f"{len(sample)} sample rows, {len(done)} already done, {len(todo)} to run")
    if todo.empty:
        return

    val = pd.read_parquet(args.val_path, columns=["image"])

    device = "cpu"
    proc = model = None
    if not args.dry_run:
        import torch

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("device:", device)
        proc, model = load_deplot(device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}  # chart hash -> (table, seconds), so a chart with several questions is read once
    start = time.time()
    # A dry run must not write empty tables to the real file, or a later real run would skip those rows.
    out_target = os.devnull if args.dry_run else args.out
    with open(out_target, "a", encoding="utf-8") as f:
        for n, row in enumerate(todo.itertuples(index=False), 1):
            img_bytes = val.iloc[row.row_idx]["image"]["bytes"]
            chart_hash = hashlib.md5(img_bytes).hexdigest()
            if chart_hash not in cache:
                if args.dry_run:
                    cache[chart_hash] = ("", 0.0)
                else:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    t0 = time.time()
                    table = deplot_table(proc, model, device, img)
                    cache[chart_hash] = (table, round(time.time() - t0, 2))
            table, secs = cache[chart_hash]
            f.write(json.dumps({
                "row_idx": int(row.row_idx),
                "chart_hash": chart_hash,
                "query": row.query,
                "label": row.label,
                "human_or_machine": int(row.human_or_machine),
                "deplot_table": table,
                "deplot_s": secs,
            }) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} rows, {len(cache)} charts read, {time.time() - start:.0f}s elapsed")
    print("Dry run, nothing saved" if args.dry_run else f"Saved to {args.out}")


if __name__ == "__main__":
    main()
