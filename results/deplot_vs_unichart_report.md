# DePlot vs UniChart: baseline comparison

Pipeline: chart image -> extractor (DePlot or UniChart) -> table -> LLM + question -> answer -> relaxed accuracy.

Both extractors ran on the **same 300 validation rows** (`results/val_sample_300.csv`, 150 human-written + 150 machine-generated questions, 279 unique charts). The LLM (DeepSeek `deepseek-chat`, temperature 0), the prompt and the scoring (`utils/answer_and_score.py`) were identical for both.

## 1. Accuracy

| Extractor | Overall | Human-written (group 0) | Machine-generated (group 1) |
|---|---|---|---|
| DePlot | **81.0%** | **74.0%** | 88.0% |
| UniChart, raw table | 74.0% | 62.0% | 86.0% |
| UniChart, rows split | 77.0% | 65.3% | **88.7%** |

"Rows split" = UniChart's one-line table (rows joined by ` & `) is put on separate lines and stray `<tags>` are removed before the LLM sees it. The extractor output is unchanged.

Group 0 is most likely the human-written questions (judged from the wording; not confirmed against the dataset card).

## 2. Head to head (question by question)

| | DePlot right | DePlot wrong |
|---|---|---|
| UniChart raw right | 211 | 11 |
| UniChart raw wrong | 32 | 46 |

- DePlot alone right on 32, UniChart alone right on 11 (sign test p = 0.002).
- Against UniChart with rows split: DePlot alone 24, UniChart alone 12 (p = 0.07, borderline).
- On machine-generated questions the two are tied. The gap is on human-written questions.

## 3. Speed (different Kaggle sessions, so approximate)

| | Seconds per chart (mean) |
|---|---|
| DePlot | 7.27 (median 4.63, max 34.9) |
| UniChart | 1.17 |

UniChart is about 6 times faster.

## 4. Why the answers were wrong (sorted by cause)

Sorted by hand, mostly from the question, label and table text. Only a few charts were checked against the image. Files: `wrong_answers_sorted.csv` (DePlot), `unichart_wrong_answers_sorted.csv` (UniChart, rows split).

| Cause | DePlot (57 wrong) | UniChart (69 wrong) |
|---|---|---|
| Table wrong or missing data | 11 | **23** |
| Visual question (colour, position), table cannot answer | 12 | 15 |
| Label or question is questionable | 14 | 14 |
| Right answer, strict scoring (format) | 9 | 9 |
| LLM error with a good table | 11 | 8 |

- Visual, questionable-label and strict-scoring cases (35 of DePlot's 57) are mostly about the question, not the extractor. They show up for both models.
- The extra UniChart errors are table errors: 23 against 11. DePlot got 19 of those 23 questions right.

## 5. Format of the wrong answers

| | DePlot | UniChart raw | UniChart rows split |
|---|---|---|---|
| Wrong answers | 57 | 78 | 69 |
| LLM refused ("can't answer from the table") | 6 | 7 | 9 |
| Wrong answer is a number (label also a number) | 30 | 40 | 34 |
| Wrong answer is text | 19 | 28 | 24 |
| Answer contains "%" | 6 | 23 | 18 |
| Answer lists several values ("A and B") | 7 | 5 | 8 |

- Wrong numbers are mostly far from the label. Only 6 to 9 of the wrong numbers are within 20% of the label, and 3 to 4 are off by exactly 10x or 100x (for example 12.4 against 0.124, a percent against a fraction).
- UniChart keeps "%" signs in its tables, so its answers carry them more often. Scoring ignores "%", so this does not change correctness.
- Refusals come mostly from colour and position questions.

### Typical wrong-answer types

| Type | Example |
|---|---|
| Scale mismatch (scored wrong, answer right) | 12.4 against label 0.124 |
| Footnote or extra text | "China" against "China*" or "China (1928)" |
| List or ratio written differently | "17:11" against 1.545 |
| Several values given, label has one | "18.3 and 18.7" against 18.7 |
| Colour or position question | "How many shades of Green does a bar show?" |
| Label disagrees with the chart | Netherlands 2020: label 20.1, chart shows 7 for 2020 and 20.1 for 2019 |
| Table error | "<1" read as 1, a missing Transport row, a dropped digit (9.2 for 91.9) |

## 6. UniChart table problems seen

- Whole table on one line, rows joined by ` & ` (299 of 300 tables). Fixed by splitting the rows (section 1).
- 24 of 300 tables contain stray `<tags>`.
- Degenerate output: "About About About ...", or a table of all zeros.
- Chart title lost, so questions about the heading cannot be answered.
- A column dropped (a "NET" column), or two charts' data mixed into one table.
- Columns swapped (oppose and favor reversed on the fracking chart).
- Long labels cut off with "..." and misspellings ("nonepithilia").
- Numbers slightly different from DePlot's, which breaks sums and differences.

DePlot's table problems are listed in the notebook (section 5): small stacked-bar segments dropped, legend text turned into fake rows, "<1" read as 1, values estimated from bar height.

## 7. Teammate's UniChart notebook vs our run

`Unichart.ipynb` (teammate) was compared with `utils/run_unichart.py` (used for the 300-row run).

**Same:** model (`ahmed-masry/unichart-base-960`), prompt (`<extract_data_table> <s_answer>`), 4-beam search, `use_cache`, max length from the decoder config, the output post-processing (strip special tokens, split on `<s_answer>`), full precision. The two extract tables the same way, so the model setup is not a source of difference.

**Different:**

| | Teammate's notebook | `run_unichart.py` |
|---|---|---|
| Rows | 100, random seed 24 | The shared 300 rows |
| Overlap with the shared 300 | about 8 rows (my reconstruction of their sample) | all |
| Saves the tables | No (only timings and a crude check) | Yes, one JSON line per row |
| Row positions kept | No (`reset_index(drop=True)`) | Yes (`row_idx`) |
| Resume after a stop | No | Yes |
| Summary cell | Crashes (asks for `deplot_s` and `deplot_has_label`, which do not exist) | n/a |
| Last cell | Would crash (reads `unichart_table`, which was never saved) | n/a |
| Where it ran | Mac (`mps`) | Kaggle T4 |

**Findings**
1. The generation code matches, so results from either should agree for the same charts.
2. The teammate's notebook cannot be compared with the DePlot results as it stands: a different 100-row sample with only about 8 rows in common, and no saved tables to pass to the LLM.
3. The comparison in this report does not depend on that notebook. It uses `run_unichart.py` on the shared sample.
4. Useful follow-up: have the teammate run `utils/run_unichart.py` on the Mac for 20 or 30 rows and compare the tables with the Kaggle output. Small differences are expected (GPU against Mac, floating point), but large ones would point to a setup problem.

## 8. Caveats

- 300 questions, 150 per group: group-level scores can move by 6 to 7 points.
- The sorting by cause is my judgment. Only a handful of charts were checked against the image.
- The "rows split" cleaning also removed stray tags, so its effect cannot be separated from the row splitting.
- DeepSeek at temperature 0 is close to repeatable but not exactly, so a few answers can change between runs.
- Relaxed accuracy counts numbers within 5% as right, including years, which inflates scores slightly.

## Files

- Tables: `deplot_val_300.jsonl`, `unichart_val_300.jsonl`
- Answers: `deplot_val_300_answers.jsonl`, `unichart_val_300_answers.jsonl`, `unichart_val_300_answers_clean.jsonl`
- Sorted errors: `wrong_answers_sorted.csv`, `unichart_wrong_answers_sorted.csv`
- Scripts: `utils/run_deplot.py`, `utils/run_unichart.py`, `utils/answer_and_score.py`
- Notebook: `data_exploration/baseline_progress.ipynb`
