"""Work out what the model should learn for each question, from its extracted table and the true answer.

For every row it decides one of:
    extract    the answer is a cell in the table (a name or a number); the matching cells are recorded
    yes_no     the answer is Yes or No
    count_rows the answer equals the number of rows in the table
    compute    the answer is a simple calculation on table numbers (sum, difference, average, ratio...)
    unusable   none of the above fits (for example a colour question, or a wrong table)

Usage:
    python utils/build_labels.py                                   # UniChart validation tables
    python utils/build_labels.py --tables results/unichart_train_3000.jsonl \
        --out results/unichart_train_3000_labels.jsonl
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_and_score import clean_unichart  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NUMBER_CELL = re.compile(r"^-?\d+(\.\d+)?\s*[a-z%]{0,3}$")

# keywords in the question that hint at which calculation is meant
OP_WORDS = {
    "sum": re.compile(r"\b(sum|total|add|added|combined|together|altogether)\b"),
    "difference": re.compile(r"\b(difference|subtract|minus|how much (more|less|higher|lower)|gap)\b"),
    "average": re.compile(r"\b(average|mean)\b"),
    "ratio": re.compile(r"\b(ratio|times|divide|per)\b"),
}


# ---------- reading a table ----------

def parse_table(text):
    """UniChart table text -> (header, rows). Rows are lists of cell strings."""
    rows = [[c.strip() for c in line.split("|")] for line in clean_unichart(text).split("\n") if line.strip()]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def cell_number(cell):
    """The number in a cell, or None. Handles %, $, commas and a short unit such as k."""
    t = cell.strip().lower().replace(",", "").replace("$", "")
    if not NUMBER_CELL.match(t):
        return None
    return float(re.match(r"-?\d+(\.\d+)?", t).group())


def normalise(text):
    return re.sub(r"\s+", " ", str(text).lower().replace("*", "")).strip()


def close(a, b, tol):
    return abs(a - b) <= 1e-9 + tol * abs(b)


# ---------- finding the answer ----------

def find_cells(header, rows, answer):
    """Cells (row, col) that hold the answer. Numbers may differ by a x100 scale (percent against fraction)."""
    hits = []
    value = cell_number(answer)
    target = normalise(answer)
    grid = [header] + rows
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if value is not None:
                n = cell_number(cell)
                # exact match (a x100 scale is allowed). A loose tolerance made 2013 and 2015 match an answer of 2014.
                if n is not None and any(abs(s - value) <= 1e-9 + 1e-6 * abs(value) for s in (n, n / 100, n * 100)):
                    hits.append((r, c))
            elif normalise(cell) == target:
                hits.append((r, c))
    return hits


def value_numbers(rows):
    """Numbers in the value cells (everything except the first, label, column) with their positions."""
    out = []
    for r, row in enumerate(rows, start=1):  # row 0 of the grid is the header
        for c, cell in enumerate(row[1:], start=1):
            n = cell_number(cell)
            if n is not None:
                out.append(((r, c), n))
    return out


def decimals(answer):
    """Number of decimal places written in the answer ('16.3' -> 1, '7' -> 0)."""
    m = re.search(r"\.(\d+)", str(answer))
    return len(m.group(1)) if m else 0


def mentioned(label, query):
    """True if at least half of the label's words (3+ letters or digits) appear in the question."""
    words = re.findall(r"[a-z0-9]{3,}", normalise(label))
    if not words:
        return False
    q_words = set(re.findall(r"[a-z0-9]{3,}", normalise(query)))
    return sum(w in q_words for w in words) / len(words) >= 0.5


def find_calculation(header, rows, query, answer):
    """A calculation on table numbers that gives the answer, or None.

    Strict on purpose, because a wrong 'compute' label would teach the model something false:
    - the question must contain a word for the calculation (sum, difference, average, ratio...)
    - the result must match the answer to the number of decimals the answer is written with
    - the table entries used must be the ones the question talks about
    """
    value = cell_number(answer)
    if value is None:
        return None
    q = query.lower()
    hinted = {op for op, pattern in OP_WORDS.items() if pattern.search(q)}
    if not hinted:
        return None
    nums = value_numbers(rows)
    tol = 0.5 * 10 ** -decimals(answer) + 1e-9
    grid = [header] + rows

    def row_label(pos):
        return grid[pos[0]][0] if grid[pos[0]] else ""

    def col_label(pos):
        return header[pos[1]] if pos[1] < len(header) else ""

    def talks_about(pos):
        return mentioned(row_label(pos), query) or mentioned(col_label(pos), query)

    def ok(op, result, positions):
        return op in hinted and abs(result - value) <= tol and all(talks_about(p) for p in positions)

    for (pa, a), (pb, b) in itertools.combinations(nums, 2):
        for op, result in (("sum", a + b), ("difference", abs(a - b)), ("ratio", a / b if b else None), ("ratio", b / a if a else None)):
            if result is not None and ok(op, result, (pa, pb)):
                return {"op": op, "operands": [list(pa), list(pb)]}
    columns = {}
    for pos, n in nums:
        columns.setdefault(pos[1], []).append((pos, n))
    for col, items in columns.items():
        vals = [n for _, n in items]
        positions = [p for p, _ in items]
        header_mentioned = mentioned(col_label(positions[0]), query)
        for op, result in (("sum", sum(vals)), ("average", sum(vals) / len(vals)), ("difference", max(vals) - min(vals))):
            if len(vals) >= 3 and header_mentioned and op in hinted and abs(result - value) <= tol:
                return {"op": op, "operands": [list(p) for p in positions]}
    return None


def label_row(query, label, table_text):
    labels = str(label).split("|")
    header, rows = parse_table(table_text)
    first = labels[0].strip()

    if len(labels) > 1 or first.startswith("["):
        parts = [p.strip() for p in first.strip("[]").split(",")] if first.startswith("[") else [p.strip() for p in labels]
        found = [find_cells(header, rows, p) for p in parts]
        if all(found):
            return "extract", {"answer_parts": parts, "cells": [h[0] for h in found]}
        return "unusable", {"reason": "list answer not fully in table"}
    if first.lower() in ("yes", "no"):
        return "yes_no", {"answer": first.lower()}

    value = cell_number(first)
    if value is not None and "how many" in query.lower() and value == int(value) and value <= 20:
        # small whole numbers match some cell by chance, so treat these as counting questions
        if value == len(rows):
            return "count_rows", {"rows": len(rows)}
        return "unusable", {"reason": "counting question (needs a count, not a cell)"}

    hits = find_cells(header, rows, first)
    if hits:
        return "extract", {"cells": [list(h) for h in hits]}
    calc = find_calculation(header, rows, query, first)
    if calc:
        return "compute", calc
    return "unusable", {"reason": "answer not in table and no reliable calculation fits"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=ROOT / "results/unichart_val_300.jsonl")
    parser.add_argument("--table-col", default="unichart_table")
    parser.add_argument("--out", type=Path, default=ROOT / "results/unichart_val_300_labels.jsonl")
    args = parser.parse_args()

    df = pd.read_json(args.tables, lines=True, dtype=False)
    kinds, details = [], []
    for r in df.itertuples(index=False):
        kind, detail = label_row(r.query, r.label, getattr(r, args.table_col))
        kinds.append(kind)
        details.append(detail)
    df["kind"] = kinds
    df["detail"] = details

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["row_idx", "query", "label", "human_or_machine", "kind", "detail"]
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in df[keep].to_dict("records"):
            f.write(json.dumps(rec) + "\n")

    print(f"{len(df)} rows -> {args.out}")
    print("\nkind, % of rows (group 0 = human-written, 1 = machine-generated)")
    overall = (df["kind"].value_counts(normalize=True) * 100).rename("all")
    by_group = pd.crosstab(df["kind"], df["human_or_machine"], normalize="columns") * 100
    print(pd.concat([overall, by_group], axis=1).round(1).to_string())
    usable = df["kind"] != "unusable"
    print(f"\nrows with a usable training label: {usable.sum()} of {len(df)} ({usable.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
