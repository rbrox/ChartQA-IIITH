"""Ask an LLM for the 'recipe' that turns a table into the answer, and keep it only if it reproduces the true answer.

For questions the rule-based label-maker could not label (calculations, counts, odd cases), DeepSeek is shown
the table and the question (NOT the answer) and writes a recipe in a fixed format, for example
    {"op": "average", "cells": [[4, 1], [6, 1]]}
The recipe is then run in code on the table numbers. If the result equals the true answer (to the number of
decimals the answer is written with), the recipe becomes a training label. Otherwise it is thrown away.

The true answer is only used to check labels. Nothing here is used at test time.

Usage:
    python utils/propose_recipes.py --limit 5
    python utils/propose_recipes.py
"""
import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_and_score import ask_llm  # noqa: E402
from build_labels import cell_number, decimals, parse_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

OPS = ["sum", "difference", "average", "ratio", "max", "min", "median", "count", "count_if"]

PROMPT = """You are given a table extracted from a chart and a question about it. Write a recipe that computes the answer from the table cells.

The table (row index, then the cells; row 0 is the header; columns are counted from 0):
{grid}

Question: {question}

Reply with ONLY one JSON object in this form, nothing else:
{{"op": "<operation>", "cells": [[row, col], ...], "cmp": "<comparison>", "value": <number>}}

Operations:
- sum: add the listed cells
- difference: first cell minus second cell, as a positive number
- average: mean of the listed cells
- ratio: first cell divided by second cell
- max, min, median: of the listed cells
- count: how many cells are listed (for "how many bars/years/items" questions)
- count_if: how many of the listed cells satisfy the comparison; also give "cmp" (one of <, <=, >, >=, ==) and "value"
Use only numeric cells. "cmp" and "value" are only for count_if.
If the question cannot be answered with one of these operations from the table, reply {{"op": "none"}}."""


def grid_text(header, rows):
    lines = [f"r0: " + " | ".join(header)]
    lines += [f"r{i}: " + " | ".join(row) for i, row in enumerate(rows, start=1)]
    return "\n".join(lines)


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_recipe(recipe, header, rows):
    """Run the recipe on the table. Returns a number, or None if it cannot be run."""
    grid = [header] + rows
    op = recipe.get("op")
    if op not in OPS:
        return None
    nums = []
    for pos in recipe.get("cells", []):
        try:
            r, c = int(pos[0]), int(pos[1])
            n = cell_number(grid[r][c])
        except (IndexError, TypeError, ValueError):
            return None
        if n is None:
            return None
        nums.append(n)
    if not nums:
        return None
    if op == "sum":
        return sum(nums)
    if op == "average":
        return sum(nums) / len(nums)
    if op == "max":
        return max(nums)
    if op == "min":
        return min(nums)
    if op == "median":
        return statistics.median(nums)
    if op == "count":
        return float(len(nums))
    if op in ("difference", "ratio"):
        if len(nums) != 2:
            return None
        if op == "difference":
            return abs(nums[0] - nums[1])
        return nums[0] / nums[1] if nums[1] else None
    if op == "count_if":
        cmp_ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
                   ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}
        fn, value = cmp_ops.get(recipe.get("cmp")), recipe.get("value")
        if fn is None or not isinstance(value, (int, float)):
            return None
        return float(sum(fn(n, value) for n in nums))
    return None


def matches(result, answer):
    value = cell_number(answer)
    if result is None or value is None:
        return False
    return abs(result - value) <= 0.5 * 10 ** -decimals(answer) + 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=ROOT / "results/unichart_val_300.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "results/unichart_val_300_labels.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/unichart_val_300_recipes.jsonl")
    parser.add_argument("--table-col", default="unichart_table")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--kinds", default="unusable,compute", help="which label-maker kinds to try")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("[ERROR] DEEPSEEK_API_KEY not found in .env")

    tables = pd.read_json(args.tables, lines=True, dtype=False)[["row_idx", args.table_col]]
    labels = pd.read_json(args.labels, lines=True, dtype=False)
    df = labels[labels["kind"].isin(args.kinds.split(","))].merge(tables, on="row_idx")
    if args.limit:
        df = df.head(args.limit)

    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["row_idx"] for line in f if line.strip()}
    todo = df[~df["row_idx"].isin(done)]
    print(f"{len(df)} rows, {len(done)} already done, {len(todo)} to run")

    with open(args.out, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo.itertuples(index=False), 1):
            header, rows = parse_table(getattr(r, args.table_col))
            prompt = PROMPT.format(grid=grid_text(header, rows), question=r.query)
            reply = ask_llm(prompt, key, args.model, max_tokens=200)
            recipe = parse_json(reply) or {"op": "none"}
            result = run_recipe(recipe, header, rows)
            f.write(json.dumps({
                "row_idx": int(r.row_idx), "query": r.query, "label": r.label,
                "human_or_machine": int(r.human_or_machine), "kind_before": r.kind,
                "recipe": recipe, "result": result, "verified": bool(matches(result, r.label)),
            }) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}")

    res = pd.read_json(args.out, lines=True, dtype=False)
    print("\nverified recipes by what the label-maker said before:")
    print(res.groupby("kind_before")["verified"].agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
