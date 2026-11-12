"""Helpers for looking at what ChartQA answers and questions look like.

Used by data_exploration/answer_space.ipynb to scope the answer space of the trained model:
should answers be picked out of the extracted table, or predicted from a fixed set of classes or bins?
"""
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets/HuggingFaceM4/ChartQA/data"

COMPUTE = re.compile(
    r"\b(sum|total|add|added|difference|subtract|average|mean|median|mode|ratio|product|multiply|divide|"
    r"combined|minus|times|percentage change|how much (more|less|higher|lower)|gap)\b")
SUPERLATIVE = re.compile(
    r"\b(highest|lowest|largest|smallest|most|least|maximum|minimum|max|min|peak|biggest|leading|top|second|third)\b")
COUNT = re.compile(
    r"\bhow many\b|\bnumber of (bars|colors|colours|categories|countries|years|segments|lines|slices|items)\b")
NUMBER_IN_TEXT = re.compile(r"-?\d[\d,]*\.?\d*")


def load_split(pattern, with_text_only=True):
    """Load question, label and group for every file matching `pattern` (images are skipped)."""
    files = sorted(glob.glob(str(DATA_DIR / pattern)))
    df = pd.concat([pd.read_parquet(f, columns=["query", "label", "human_or_machine"]) for f in files], ignore_index=True)
    df["labels"] = df["label"].apply(lambda x: [str(v) for v in list(x)])
    df["atype"] = df["labels"].apply(answer_type)
    df["op"] = [operation(q, t) for q, t in zip(df["query"], df["atype"])]
    return df


def to_number(text):
    t = str(text).strip().lower().replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def answer_type(labels):
    """number, yes/no, text (a name or category) or list (several answers)."""
    if len(labels) > 1 or labels[0].strip().startswith("["):
        return "list"
    answer = labels[0].strip().lower()
    if answer in ("yes", "no"):
        return "yes/no"
    if to_number(answer) is not None:
        return "number"
    return "text"


def operation(query, atype):
    """What the question asks the model to do, from keywords in the question."""
    q = query.lower()
    if atype == "yes/no":
        return "yes/no (compare or check)"
    if COMPUTE.search(q):
        return "compute (sum, difference, average, ratio...)"
    if COUNT.search(q):
        return "how many (a count or a value)"
    if SUPERLATIVE.search(q):
        return "pick highest/lowest (superlative)"
    return "look up a value or name"


def numbers_in(table):
    out = []
    for match in NUMBER_IN_TEXT.findall(table):
        try:
            out.append(float(match.replace(",", "")))
        except ValueError:
            pass
    return out


def answer_in_table(labels, table):
    """Rough check that the answer can be read straight out of the table (no computing needed).

    Text: the answer string appears in the table. Numbers: some number in the table equals the answer,
    also allowing a x100 or /100 scale difference (percent against fraction).
    """
    answer = labels[0].strip().lower()
    value = to_number(answer)
    if value is None:
        return answer.replace("*", "").strip() in table.lower().replace("*", "")
    for x in numbers_in(table):
        for scaled in (x, x / 100, x * 100):
            if abs(scaled - value) <= 1e-9 + 1e-6 * abs(value):
                return True
    return False
