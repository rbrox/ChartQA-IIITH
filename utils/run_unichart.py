"""Run UniChart (chart -> table) on the shared validation sample.

Reads the row list from results/val_sample_300.csv, runs UniChart once per unique
chart, and appends one JSON line per row to the output file. Safe to stop and
re-run: rows already in the output file are skipped.

Usage (local):
    python utils/run_unichart.py --limit 10
Usage (Kaggle):
    python utils/run_unichart.py \
        --val-path /kaggle/input/<dataset>/val-00000-of-00001-0f11003c77497969.parquet \
        --sample-csv /kaggle/input/<dataset>/val_sample_300.csv \
        --out /kaggle/working/unichart_val_300.jsonl
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
MODEL_NAME = "ahmed-masry/unichart-base-960"
PROMPT = "<extract_data_table> <s_answer>"


def load_unichart(device: str):
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    proc = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device).eval()
    return proc, model


def unichart_table(proc, model, device: str, img: Image.Image) -> str:
    import torch

    decoder_ids = proc.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = proc(img, return_tensors="pt").pixel_values
    with torch.no_grad():
        out = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=4,
            return_dict_in_generate=True,
        )
    seq = proc.tokenizer.batch_decode(out.sequences)[0]
    seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
    return seq.split("<s_answer>")[-1].strip()


def main():
    # When run with `!python` in a notebook, stdout is a pipe and Python buffers it, so progress would appear only at the end.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", type=Path, default=VAL_PATH)
    parser.add_argument("--sample-csv", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--out", type=Path, default=ROOT / "results/unichart_val_300.jsonl")
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
        proc, model = load_unichart(device)

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
                    table = unichart_table(proc, model, device, img)
                    cache[chart_hash] = (table, round(time.time() - t0, 2))
            table, secs = cache[chart_hash]
            f.write(json.dumps({
                "row_idx": int(row.row_idx),
                "chart_hash": chart_hash,
                "query": row.query,
                "label": row.label,
                "human_or_machine": int(row.human_or_machine),
                "unichart_table": table,
                "unichart_s": secs,
            }) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} rows, {len(cache)} charts read, {time.time() - start:.0f}s elapsed")
    print("Dry run, nothing saved" if args.dry_run else f"Saved to {args.out}")


if __name__ == "__main__":
    main()
