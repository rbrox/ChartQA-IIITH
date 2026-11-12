"""Turn each extracted table into a list of candidate answers (its cells), with the gold cells marked.

The extract head of the model scores every candidate against the question and picks the best one.
Cell positions follow build_labels.py: row 0 is the header row, rows 1.. are data rows, columns count from 0.

Candidate kinds:
    header     a cell in the header row (a column name)
    row_label  the first cell of a data row (a row name, such as a country or a year)
    value      any other cell of a data row; its text also carries its row name and column name
"""
import json
from pathlib import Path

import pandas as pd

from build_labels import parse_table

MAX_CANDIDATES = 200
KINDS = ["header", "row_label", "value"]


def table_candidates(table_text, max_candidates=MAX_CANDIDATES):
    """List of dicts: text (what gets embedded), answer (the raw cell text), kind, row, col."""
    header, rows = parse_table(table_text)
    cands = []
    for c, cell in enumerate(header):
        if cell:
            cands.append({"text": cell, "answer": cell, "kind": "header", "row": 0, "col": c})
    for r, row in enumerate(rows, start=1):
        label = row[0] if row else ""
        for c, cell in enumerate(row):
            if not cell:
                continue
            if c == 0:
                cands.append({"text": cell, "answer": cell, "kind": "row_label", "row": r, "col": 0})
            else:
                col_name = header[c] if c < len(header) else ""
                cands.append({"text": f"{label} | {col_name} | {cell}", "answer": cell, "kind": "value", "row": r, "col": c})
    return cands[:max_candidates], len(rows)


def build_examples(tables_path, labels_path, table_col="unichart_table"):
    """One dict per question: the question, the true answer, its kind, its candidates and the gold candidate indexes."""
    tables = pd.read_json(tables_path, lines=True, dtype=False)[["row_idx", "chart_hash", table_col]]
    labels = pd.read_json(labels_path, lines=True, dtype=False)
    df = labels.merge(tables, on="row_idx")
    examples = []
    for r in df.itertuples(index=False):
        cands, n_rows = table_candidates(getattr(r, table_col))
        gold = []
        if r.kind == "extract":
            positions = {(c["row"], c["col"]): i for i, c in enumerate(cands)}
            cells = r.detail.get("cells", [])
            if cells and not isinstance(cells[0], list):
                cells = [cells]
            gold = sorted({positions[tuple(p)] for p in cells if tuple(p) in positions})
        examples.append({
            "row_idx": int(r.row_idx), "chart_hash": r.chart_hash, "query": r.query, "label": str(r.label),
            "human_or_machine": int(r.human_or_machine), "kind": r.kind, "n_rows": n_rows,
            "candidates": cands, "gold": gold, "detail": r.detail,
        })
    return examples
