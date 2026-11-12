"""Save UniChart's image-encoder features for each chart (no table is generated, so this is fast).

These are the frozen image features for the trained model. For every chart it saves:
    - one pooled vector for the whole chart (mean over all image positions)
    - a coarse grid of pooled vectors (default 4x4), so some position information is kept

Input is the packed images file from make_subset_images.py (columns: chart_hash, image).
Features are written in chunks, so the run can be stopped and restarted.

Usage (Kaggle, after the 5-chart test):
    python utils/extract_image_features.py \
        --chart-images /kaggle/input/<dataset>/train_subset_3000_images.parquet \
        --out-dir /kaggle/working/image_features --limit 5

Load the result with load_features() below.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "ahmed-masry/unichart-base-960"
FEATURE_DIM = 1024  # width of UniChart's image encoder output; the real value is read from the model when it runs


def load_features(out_dir):
    """Returns (chart_hashes, features) with features shaped [charts, 1 + grid*grid, dim], float16.

    Position 0 is the whole-chart mean; positions 1.. are the grid cells in row order.
    """
    parts = sorted(Path(out_dir).glob("features_part_*.npz"))
    hashes, feats = [], []
    for p in parts:
        z = np.load(p, allow_pickle=False)
        hashes += [str(h) for h in z["chart_hash"]]
        feats.append(z["features"])
    return hashes, np.concatenate(feats)


def pool(tokens, grid):
    """tokens: [N, dim] image-encoder output for one chart -> [1 + grid*grid, dim]."""
    import torch

    n, dim = tokens.shape
    side = int(round(n ** 0.5))
    whole = tokens.mean(dim=0, keepdim=True)
    if side * side != n:  # not a square layout: keep only the whole-chart vector, repeat it for the grid cells
        return torch.cat([whole, whole.repeat(grid * grid, 1)])
    img = tokens.reshape(side, side, dim).permute(2, 0, 1).unsqueeze(0)  # [1, dim, side, side]
    cells = torch.nn.functional.adaptive_avg_pool2d(img, grid).squeeze(0).reshape(dim, grid * grid).T
    return torch.cat([whole, cells])


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart-images", type=Path, default=ROOT / "datasets/train_subset/train_subset_3000_images.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/image_features")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=250, help="charts per saved file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="skip the model and save zeros, to check the file handling")
    args = parser.parse_args()

    charts = pd.read_parquet(args.chart_images)
    if args.limit:
        charts = charts.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(charts) + args.chunk - 1) // args.chunk

    def chunk_done(i):
        """A chunk counts as done only if its file holds every chart it should (a --limit test run leaves a short one)."""
        path = args.out_dir / f"features_part_{i:04d}.npz"
        if not path.exists():
            return False
        expected = min(args.chunk, len(charts) - i * args.chunk)
        return len(np.load(path, allow_pickle=False)["chart_hash"]) == expected

    todo = [i for i in range(n_chunks) if not chunk_done(i)]
    print(f"{len(charts)} charts in {n_chunks} chunks, {n_chunks - len(todo)} already done")
    if not todo:
        return

    model = proc = device = None
    if not args.dry_run:
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("device:", device)
        proc = DonutProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device).eval()

    start = time.time()
    for i in todo:
        block = charts.iloc[i * args.chunk:(i + 1) * args.chunk]
        feats = []
        for item in block["image"]:
            if args.dry_run:
                feats.append(np.zeros((1 + args.grid ** 2, FEATURE_DIM), dtype=np.float16))
                continue
            img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            pixel_values = proc(img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                tokens = model.encoder(pixel_values=pixel_values).last_hidden_state[0]
            feats.append(pool(tokens, args.grid).cpu().numpy().astype(np.float16))
        np.savez_compressed(args.out_dir / f"features_part_{i:04d}.npz",
                            chart_hash=block["chart_hash"].to_numpy().astype(str), features=np.stack(feats))
        print(f"chunk {i + 1}/{n_chunks} saved, {time.time() - start:.0f}s elapsed")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
