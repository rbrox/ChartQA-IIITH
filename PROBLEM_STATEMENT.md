CV-08 · Computer Vision │ Visual Reasoning │ Document & Chart Understanding
14. Chart Question Answering: Visual Reasoning Over Data Visualizations
Vision-Language │ Visual Reasoning │ Document & Chart Understanding
Answering questions about a chart or infographic, such as "what was the peak value in 2019?" or "which
category has the second-highest bar?", requires a different kind of grounding than natural-image VQA:
the model must read numeric and text elements (axis labels, legends) and reason about their spatial and
quantitative relationships. This capstone is a more specialised, harder variant of VQA, building directly
on project 13's fusion approach with an OCR-augmented input representation.
Suggested Steps

1. Subsample ChartQA; run an OCR tool over each chart to extract axis labels, legend text, and data-
point annotations.

2. Encode the chart image, the OCR-extracted text, and the question, then combine into a joint
representation using frozen encoders.
3. Scope the answer space explicitly: either extractive answers from OCR text or a discretised/binned
classification over numeric ranges, and justify the choice in the report.
4. Train a fusion and answer-selection head on top of the joint representation.
5. Evaluate using ChartQA's relaxed-accuracy metric.
6. Separate error analysis into OCR-extraction failures versus reasoning failures.
Dataset ChartQA (available on HuggingFace Datasets / Kaggle mirrors): chart images paired with
questions and answers requiring visual and numerical reasoning.
Expected Outcome A trained chart-QA model with relaxed accuracy on the held-out set, an explicit
breakdown of OCR-extraction failures versus reasoning failures, 10 annotated example predictions, and
a report contrasting chart QA against natural-image VQA challenges.
Masry, A., Long, D. X., Tan, J. Q., Joty, S., & Hoque, E. (2022). ChartQA: A benchmark for question answering about
charts with visual and logical reasoning. Findings of ACL 2022.
Methani, N., Ganguly, P., Khapra, M. M., & Kumar, P. (2020). PlotQA: Reasoning over scientific plots. Proceedings
of WACV 2020.

A Few Points to Note While Working on the Capstone Project
● Though this is a group project, faculty/mentors can access individual performance/contributions
and may award different marks to each individual of the same group.
● Attendance is the minimum criterion for getting marks during the mentoring sessions
scheduled on every Sunday morning, 9 AM to 12 PM.
● Project Proposal Format: a document not more than 7-8 pages, covering the title, problem
statement, literature review, methodologies, possible outcomes in stages, and applicability in
the real world. The first page should contain the title and group number and team members'
names.

● Final Report/PPT Format: a Word Document and a PPT, covering the title, problem statement,
literature review, methodologies, outcomes in stages, final outcomes, challenges, and
applicability in the real world. The first page should contain the title, group number, and team
members' names.
● Demonstration of the deployed model is a must, along with the final presentation and report.
There will be four checkpoints with the following expectations:
● Check Point 1: Preliminary model and training (on a subset of data).
● Check Point 2: Model building and training for the complete dataset.
● Check Point 3: Deployment testing (on a subset of data).
● Check Point 4: Fine-tuning and final deployment.