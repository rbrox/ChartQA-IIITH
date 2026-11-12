"""Ask an LLM to answer each question from its extracted chart table, then score the answers.

Works on any results file that has a table column (DePlot, UniChart, ...), so every
extractor goes through the same LLM, prompt and scoring.

Setup: put your key in the project's .env file as
    DEEPSEEK_API_KEY=sk-...

Usage:
    python utils/answer_and_score.py --limit 5            # small test first
    python utils/answer_and_score.py                      # all rows
    python utils/answer_and_score.py --tables results/unichart_val_300.jsonl \
        --table-col unichart_table --out results/unichart_val_300_answers.jsonl
    python utils/answer_and_score.py --dry-run --limit 2  # print prompts, no API call
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://api.deepseek.com/chat/completions"

PROMPT = """Below is a data table that was extracted from a chart. Answer the question using only the table.
Reply with just the final answer: a number, or a short word or phrase such as Yes or No. No explanation.

Table:
{table}

Question: {question}
Answer:"""


# ---------- scoring (ChartQA relaxed accuracy) ----------

def _normalise(text):
    return str(text).strip().lower().replace(",", "").replace("$", "").replace("%", "").rstrip(".").strip()


def _to_number(text):
    try:
        return float(_normalise(text))
    except ValueError:
        return None


def relaxed_correct(prediction, labels):
    """True if the prediction matches any accepted label.

    Numbers count as correct within 5% of the label; everything else needs an exact
    match after lowercasing and removing $ , % and spaces. `labels` can be a list or a
    '|'-joined string (as saved in the sample CSV).
    """
    if not isinstance(labels, (list, tuple)):
        labels = str(labels).split("|")
    pred_num = _to_number(prediction)
    for label in labels:
        label_num = _to_number(label)
        if pred_num is not None and label_num is not None:
            if label_num == 0:
                if pred_num == 0:
                    return True
            elif abs(pred_num - label_num) / abs(label_num) <= 0.05:
                return True
        elif _normalise(prediction) == _normalise(label):
            return True
    return False


# ---------- LLM call ----------

def ask_llm(prompt, key, model, retries=5, max_tokens=64):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            resp = None
        if resp is not None and resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        if resp is not None and resp.status_code in (401, 402, 403):
            sys.exit(f"[ERROR] API refused the request ({resp.status_code}): {resp.text[:200]}\n"
                     "Check the key in .env, and that the account has balance.")
        time.sleep(2 ** attempt)  # rate limit or server hiccup: wait and retry
    raise RuntimeError("API kept failing after retries")


def clean_unichart(table):
    """UniChart writes a whole table on one line with ' & ' between rows, and sometimes adds <tags>."""
    table = re.sub(r"<[^>]*>", " ", table)
    return "\n".join(row.strip() for row in table.split(" & ") if row.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=ROOT / "results/deplot_val_300.jsonl")
    parser.add_argument("--table-col", default="deplot_table")
    parser.add_argument("--out", type=Path, default=ROOT / "results/deplot_val_300_answers.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the prompts, make no API calls")
    parser.add_argument("--clean-unichart", action="store_true",
                        help="put UniChart's '&'-joined rows on separate lines and drop <tags> before prompting")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key and not args.dry_run:
        sys.exit("[ERROR] DEEPSEEK_API_KEY not found. Add it to the .env file in the project root.")

    rows = pd.read_json(args.tables, lines=True, dtype=False)  # keep labels as text, not auto-converted numbers
    if args.limit:
        rows = rows.head(args.limit)

    def get_table(r):
        table = getattr(r, args.table_col)
        return clean_unichart(table) if args.clean_unichart else table

    done = set()
    if args.out.exists() and not args.dry_run:
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["row_idx"] for line in f if line.strip()}
    todo = rows[~rows["row_idx"].isin(done)]
    print(f"{len(rows)} rows, {len(done)} already answered, {len(todo)} to run")

    if args.dry_run:
        for r in todo.head(2).itertuples(index=False):
            print("-----\n" + PROMPT.format(table=get_table(r), question=r.query))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo.itertuples(index=False), 1):
            answer = ask_llm(PROMPT.format(table=get_table(r), question=r.query), key, args.model)
            f.write(json.dumps({
                "row_idx": int(r.row_idx),
                "query": r.query,
                "label": r.label,
                "human_or_machine": int(r.human_or_machine),
                "answer": answer,
                "correct": bool(relaxed_correct(answer, r.label)),
            }) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} answered")

    result = pd.read_json(args.out, lines=True, dtype=False)
    print(f"\nAccuracy: {result['correct'].mean():.1%} on {len(result)} rows")
    print(result.groupby("human_or_machine")["correct"].mean().map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
