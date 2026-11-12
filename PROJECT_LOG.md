# Project log: ChartQA

A running record of what we decided, why, where the project stands, and what comes next. Newest decisions go at the bottom of the decision log. Dates before 2026-11-11 are approximate.

Last updated: 2026-11-11

---

## 1. Where we are now

**Done**
- Looked at the data and built a shared validation sample (300 rows).
- Ran DePlot and UniChart on that sample (Kaggle GPU), had DeepSeek answer from the tables, and scored with relaxed accuracy. This is the **baseline**.
- Sorted the wrong answers by cause (first version of the OCR-failure vs reasoning-failure breakdown).
- Compared DePlot and UniChart (report in `results/deplot_vs_unichart_report.md`).
- Decided the answer space (extractive, see D8) and wrote `data_exploration/answer_space.ipynb`.
- Chose the training subset (3,000 questions, 2,768 charts) and packed its images for Kaggle.
- Wrote the label-maker (`utils/build_labels.py`) and tested it on the 300 validation tables: 81.7% of questions get a usable training label.
- Tried LLM-proposed recipes to label the questions the rules could not (`utils/propose_recipes.py`): 15 of the 55 unusable questions recovered with a verified recipe, and 3 of the 18 rule-based compute labels looked wrong (see D12).

**Extraction and labels for the training subset (done)**
- UniChart ran on all 2,768 charts (3,000 questions) on Kaggle in about 53 minutes. Output `results/unichart_train_3000.jsonl` checked against the subset file: every row present, no empty tables, 233 tables with a stray `<tag>`, 187 with a number cut off after the decimal point.
- The label-maker on those tables (after the number-matching fix, D16): **2,325 of 3,000 (77.5%) usable** (extract 2,014, yes/no 149, compute 138, count rows 24); 675 unusable. Usable share is 90% for machine-generated and 65% for human-written questions.

- Recipe labelling on the training tables (813 DeepSeek calls): 181 of the 675 unusable questions recovered with a verified recipe (27%; 178 of them human-written), and 115 of the 138 rule-based compute labels confirmed. After merging (`utils/merge_labels.py`): **2,483 of 3,000 (82.8%) usable** (extract 2,014, compute 296, yes/no 149, count rows 24), 517 unusable. Final labels in `results/unichart_train_3000_final_labels.jsonl`.

- Image features for all 2,768 training charts and the 279 validation charts are in `results/image_features/` and `results/image_features_val/` (17 vectors of 1,024 numbers per chart, checked: right charts, right order, no NaN, no overlap between train and validation). A first run had left 245 training charts missing because a short test chunk counted as done; the script now checks chunk size.

**Ready but not yet run**
- Nothing is waiting on a Kaggle run now.

**Not started**
- The trainable model, evaluation of the trained model, deployment.

## 2. Next immediate step

1. Build the **compute head** (sum, difference, average, ratio, max, min, median, count, count above or below a value, with operands chosen from table cells). 296 training examples exist in `results/unichart_train_3000_final_labels.jsonl`. Compute questions are about 10% of validation and score 0% now.
2. Look at the extract head's mistakes on validation and improve it (v1 gets about 62% on extract-type questions).
3. Revisit the image input only for the extract head or for colour questions, since pooled features hurt overall.
4. Commit and push the new files (the user pushes).

## 3. The pipeline we are building

```
chart image -> UniChart -> table text ─┐
                                       ├─> frozen encoders -> trained head -> answer
question ──────────────────────────────┘
```

Planned trained part: a router that decides the answer kind, an extract head (pick a table cell), a compute head (sum, difference, average, ratio) and a yes/no head. The router and heads are proposed, not yet confirmed (see open questions).

Problem statement checkpoints: 1 = preliminary model on a subset, 2 = full dataset, 3 = deployment testing on a subset, 4 = fine-tuning and final deployment.

## 4. Results so far

All on the same 300 validation questions (150 human-written, 150 machine-generated), same LLM, prompt and scoring.

| Setup | Overall | Human-written | Machine-generated |
|---|---|---|---|
| DePlot + DeepSeek | 81.0% | 74.0% | 88.0% |
| UniChart + DeepSeek, raw table | 74.0% | 62.0% | 86.0% |
| UniChart + DeepSeek, rows split | 77.0% | 65.3% | 88.7% |

- UniChart is about 6 times faster than DePlot (about 1.2 s against 7.3 s per chart).
- Why answers were wrong (DePlot 57, UniChart 69): table errors 11 vs 23, visual (colour or position) questions 12 vs 15, questionable labels 14 vs 14, strict scoring 9 vs 9, LLM errors 11 vs 8. Sorting done by hand, mostly from the table text.
- Answer space (all train): 80.7% numbers, 16.6% text, 2.1% yes/no. About 76% of validation answers can be read straight from DePlot's table (74% from UniChart's).

### Findings from the training-data steps (3,000-question subset)

- **Extraction scales.** UniChart took about 53 minutes for 2,768 charts (1.15 s per chart), so all 18,271 training charts would take about 5.8 hours (an estimate). The quirks seen on validation hold on training: 7.8% of tables have a stray `<tag>` (8% on validation) and 6.2% have a number cut off after the decimal point (4.7%).
- **Most questions can be turned into extractive targets, but not all.** Rule-based labels: 77.5% usable (80.0% on validation). Usable share is 90% for machine-generated and 65% for human-written questions.
- **LLM recipes replicate and clean the labels.** 181 of 675 unusable questions recovered (27%, the same as on validation), 178 of them human-written. They confirmed 115 of 138 rule-based compute labels and exposed 23 false ones (17%). Compute examples went from 138 to 296.
- **Final label mix** (2,483 of 3,000 usable, 82.8%). Human-written: extract 44%, compute 20%, yes/no 10%, count 1.5%, unusable 25%. Machine-generated: extract 90%, unusable 10%. So extraction alone covers under half of the human-written questions; the compute and yes/no heads are needed.
- **No leakage.** No training chart appears in the validation set (checked by image hash and in the feature files).
- **Not proven yet:** that a trained model works, that image features help, and that every recovered label is correct (they match the true answer, which can still happen by coincidence; about 12 checked by eye).

### Model v1 (router, extract head, yes/no head; compute not answered yet)

Frozen MiniLM text embeddings plus simple numeric and question-match features; trained on the 2,483 usable training questions (10% of charts held out to pick the epoch); scored on the 300 validation questions with relaxed accuracy. Three seeds each (`results/model_v1_*.json`).

| | Overall | Human-written | Machine-generated | Extract-type questions |
|---|---|---|---|---|
| Text only | 48.9% (48.3, 49.0, 49.3) | 33.3% | 64.4% | 61.7% |
| Text + UniChart image features | 45.3% (43.7, 45.3, 47.0) | 25.8% | 64.9% | 58.9% |
| Baseline UniChart + DeepSeek (rows split) | 77.0% | 65.3% | 88.7% | n/a |
| Baseline DePlot + DeepSeek | 81.0% | 74.0% | 88.0% | n/a |

- The router picks the right answer kind for about 93% of labelled questions (about 90% with image features).
- Compute questions get no answer yet, so they score 0%; they are about 10% of validation.
- **Pooled image features did not help.** Every text-only run beat every image run overall. They hurt the human-written group (26% vs 33%) and the router, and are about equal on machine-generated questions. With about 2,000 training questions a 1,024-number image vector adds noise. If the image is used again, try it only in the extract head or for colour questions.
- The first version (without question-match features) scored 38.3%; adding the similarity and word-overlap features gave 48.3%.

## 5. Decision log

| # | When | Decision | Why |
|---|---|---|---|
| D1 | late Oct | Build a zero-shot baseline first (extractor, then an LLM answers from the table) before training anything. | It gives a reference number to beat, and it shows whether failures come from the table or the reasoning. |
| D2 | late Oct | Use the train split to learn, the validation split to check while building, and the test split once at the end. | Keeps the final score honest. |
| D3 | late Oct | One shared stratified validation sample: 300 rows, seed 42, 150 per `human_or_machine` group, saved as `results/val_sample_300.csv`. Everyone loads the CSV and nobody re-runs the script. | Both extractors are scored on identical rows. A seed alone can pick different rows under different library versions. |
| D4 | late Oct | Run the heavy extraction on Kaggle GPUs (T4). Keep VS Code as the main IDE and notebook, and move files in and out. | No local GPU. |
| D5 | late Oct | DeepSeek (`deepseek-chat`, temperature 0) is the baseline LLM. Key in `.env` (git-ignored), $2 top-up. | Cheap and easy; about 300 calls cost a few cents. |
| D6 | late Oct | Score with ChartQA relaxed accuracy: numbers within 5%, text exact after lowercasing and stripping $ , %. | Standard metric. Known quirk: counts years within 5% as right. |
| D7 | early Nov | For the baseline, split UniChart's one-line rows (joined by ` & `) onto separate lines and drop stray `<tags>` before the LLM sees them. | Raised UniChart from 74% to 77%. |
| D8 | 2026-11-11 | The model **extracts** answers from the table. No binned or fixed-class answers. | Numbers are too spread out (9,675 distinct values, about 250 bins of 10% width) and text answers are a long tail (2,604 distinct, the top 1,000 cover 66%). About 90% of look-up answers sit in the table. |
| D9 | 2026-11-11 | Use **UniChart only** for training, validation and the demo. | About 6 hours for all training charts against about 37 for DePlot; about 1.2 s per chart suits a demo; train and test tables must come from the same extractor. |
| D10 | 2026-11-11 | Training subset: 3,000 questions from the train split, half human-written and half machine-generated, 2,768 unique charts, none in validation. Files: `results/train_subset_3000.csv`, images in `datasets/train_subset/train_subset_3000_images.parquet` (git-ignored). | Half and half because validation is 50/50 and human-written questions are the hard ones (more computing, all the yes/no). Enough for Checkpoint 1. |
| D11 | 2026-11-11 | Training labels come from a rule-based label-maker that marks where each answer sits in the table (extract, compute, yes/no, count rows, unusable). Strict rules for calculations. | The data only gives final answers, and the model needs a target cell. Loose rules produced false labels, so they were tightened (first version said 90% usable, the strict one 81.7%). |

| D12 | 2026-11-11 | **Experiment, not yet adopted:** for questions the rules cannot label, ask DeepSeek for a recipe (operation plus table cells) without showing the answer, run it in code, and keep it only if it reproduces the true answer. | The data gives only final answers, and rules cannot cover every calculation. Results on the 300 validation tables: 15 of 55 unusable questions recovered (12 of 33 calculation or other, 3 of 13 counting, 0 of 9 needing the image); the recipe also reproduced 15 of 18 rule-based compute labels, and the other 3 looked like coincidental rule matches (for example a column header such as "value" made unrelated cells pass the rule). Estimated usable labels: about 257 of 300 (86%) if the 3 suspect labels are dropped. |

| D13 | 2026-11-11 | The frozen **image encoder is UniChart's own encoder**. Its features are saved by a separate encoder-only pass (`utils/extract_image_features.py`) after the table extraction finishes: one whole-chart vector plus a 4x4 grid of pooled vectors per chart. | The problem statement requires frozen encoders for the image, text and question but names none. UniChart's encoder is built for charts and is already in the pipeline. An encoder-only pass skips the slow table writing, so it should take a fraction of the extraction time (about 5 to 15 minutes, a guess). Image input targets the questions that need colours or positions (8 of the 40 unlabelled; 6.4% of train questions mention a colour, shade or legend). |

| D14 | 2026-11-11 | Use the **final labels**: rule-based extract, yes/no and count labels as they are; compute labels only where an LLM recipe independently reproduced the true answer; recipes recover some unusable questions. Files: `utils/merge_labels.py`, `results/unichart_train_3000_final_labels.jsonl`. | A wrong label teaches the model something false, so unconfirmed compute labels are dropped (22 of 136 on the training set; 3 of 18 on validation). The recipes recovered 28% of unusable training questions, almost all human-written. |

| D15 | 2026-11-11 | Leave the questions that need the image (colour, shade, legend, position; about 136 of the 484 unlabelled training questions) **unlabelled for now**. Revisit later, possibly with a vision-capable model proposing recipes that are checked against the true answer. | Image features exist but nothing uses them yet, and no vision-capable model is set up (DeepSeek chat is text-only). The first model should be built and measured before spending effort on this slice (about 4.5% of the subset). |

| D16 | 2026-11-11 | Frozen **text encoder: all-MiniLM-L6-v2**, run locally on CPU (torch and sentence-transformers installed in `.venv`). Fixed a **bug in the label-maker's number matching**: it counted numbers within 0.1% as equal, so a year such as 2014 matched cells reading 2013 and 2015. Matching is now exact (a x100 scale is still allowed). Labels, recipes and final labels were rebuilt. | MiniLM is small and standard (embedding all 3,000 questions and their table cells takes a few minutes on CPU). The bug created false extract labels (33 training rows and 5 validation rows became unusable or changed kind) and also inflated the earlier "answer is readable from the table" share (now 75.0% DePlot, 72.3% UniChart, was 76.0 and 74.3). |

### Working agreements
- Commit only when asked, and never push. The user pushes manually.
- Run notebooks from the top before committing, so saved outputs are current.
- Leave `README.md` and `Chart_extraction.ipynb` alone unless asked.
- Plain, simple language in explanations.

## 6. Open questions

- Confirm the router with an extract head, a compute head and a yes/no head (proposed, not yet agreed).
- Adopt the verified-recipe labelling (D12) for the training set, and then tighten the rule-based compute labels (the column-header check in `build_labels.py` lets unrelated cells pass).
- Which frozen **text** encoder to use for the question and table (the image encoder is decided, D13). Ask the mentor whether project 13 used particular encoders; the problem statement names none.
- How the image is used (parked, see D15): classify small answer sets (colour names, small counts), add series colours to the table headers, or full attention over image regions. The first two are proposed for Checkpoint 2.
- Who does what with the teammate: extraction on Kaggle, or labels and model.
- Which checkpoint is next and the date of the next Sunday session.
- Whether the test split is 50/50 human and machine (not checked).
- Whether to run DePlot on validation only (about 2 hours for 1,055 charts) to complete the DePlot against UniChart comparison. Proposed.

## 7. Known caveats

- 300 validation questions: group scores can move by 6 to 7 points.
- Wrong-answer sorting is my judgment, mostly from table text; only a few charts were checked against the image.
- The tests of the rules and counts used the validation tables; the UniChart model call has not been run on this PC (no torch), only on Kaggle.
- The teammate's `Unichart.ipynb` uses a different 100-row sample (about 8 rows overlap) and does not save its tables, so it cannot be compared with these results. Same model settings otherwise.

- The 40 questions still unlabelled after the recipe experiment (sorted in `results/unrecovered_40_sorted.csv`): 17 UniChart table wrong or missing, 8 need the image, 6 are answers the table contains but my tools missed (fixable: add argmax and argmin recipes that return a row name, read units like "n mi", loosen the small-number "how many" rule), 5 questionable question or label, 4 where UniChart dropped the last decimal digit.
- UniChart cuts numbers off after the decimal point (for example "376." for 376.1) in 14 of 300 tables (23 cells). The lost digit cannot be recovered. It is a UniChart extraction failure to report.

- On this Windows machine `torch` must be imported before numpy and pandas, or it fails with a DLL error (WinError 1114). `embed_examples.py` and `train_model.py` do this at the top. A first pip install also failed partway because another Python process was running in the same environment; keep the environment idle while installing.

## 8. Files

| What | Where |
|---|---|
| Scripts | `utils/make_val_sample.py`, `run_deplot.py`, `run_unichart.py`, `answer_and_score.py`, `answer_space.py`, `make_train_subset.py`, `make_subset_images.py`, `build_labels.py` |
| Model v1 | `utils/build_candidates.py`, `utils/embed_examples.py` (writes `datasets/embeddings/`, git-ignored), `utils/train_model.py` (results in `results/model_v1_*.json`) |
| Image features | `utils/make_val_images.py` (packs the validation charts), `utils/extract_image_features.py` (writes `results/image_features/features_part_*.npz`; load with `load_features`) |
| Notebooks | `data_exploration/baseline_progress.ipynb`, `answer_space.ipynb` |
| Report | `results/deplot_vs_unichart_report.md` |
| Samples | `results/val_sample_300.csv`, `results/train_subset_3000.csv` |
| Tables and answers | `results/deplot_val_300*.jsonl`, `results/unichart_val_300*.jsonl` |
| Sorted errors | `results/wrong_answers_sorted.csv`, `results/unichart_wrong_answers_sorted.csv` |
| Labels | `results/unichart_val_300_labels.jsonl`, `results/unichart_train_3000_labels.jsonl`, `results/unichart_train_3000_recipes.jsonl`, `results/unichart_train_3000_final_labels.jsonl` (use this one; script `utils/merge_labels.py`), `results/unichart_val_300_recipes.jsonl` (LLM recipes with a verified flag; script `utils/propose_recipes.py`) |

## 9. How to update this log

After any decision, result or change of plan: add a row to the decision log (what, why), update sections 1 and 2, and move answered questions out of section 6.
