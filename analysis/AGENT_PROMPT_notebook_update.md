# Task: update `analysis/findings_walkthrough.ipynb` so it reproduces every number in the thesis

## Context

This repository holds a Master's thesis comparing four RAG architectures on FinanceBench
(vector RAG, vectorless RAG via PageIndex, a structure-guided hybrid, and long context).
`analysis/findings_walkthrough.ipynb` is the evidence notebook: every number reported in the
thesis must be computed in it from the saved result files in `experiments/results/`.

Section 13 of the notebook ("Additional figures reported in the thesis", the cells after the
markdown heading `## 13. Additional figures reported in the thesis`) already exists but has
**never been executed**. Some thesis numbers are not computed anywhere yet. Your job is to add
the missing cells, run the notebook, and check every output against the expected values below.

`analysis/THESIS_NUMBERS_TODO.md` lists all thesis numbers with their location in the thesis.
Use it as the checklist.

## Rules

- **No API calls.** The notebook only reads files in `experiments/results/` and `data/`.
- **Do not modify, delete or regenerate any file in `experiments/results/` or `data/`.**
- Reuse the helpers that already exist in the notebook instead of writing new versions:
  `R`, `N`, `questions`, `gold_pages`, `load_jsonl`, `latest_by_id`, `SCORING`, `_chunks`,
  `rouge2_recall`, `node_map`, and from Section 13's first code cell: `VR_SEL`, `VR_RR`,
  `VR_TOP20`, `HY_POOL`, `HY_RR`, `FI_RR`, `NAV_EXP`, `NAV_ORIG`, `chunk_pages`, `nav_pages`,
  `has_gold`, `lab` (a DataFrame of verdicts, one column per run: "Vector RAG", "Vectorless RAG",
  "PageIndex hybrid", "Long context", "Vector RAG without selection agent", "Hybrid filter variant"),
  `CONTEXT` and `exposure` (a DataFrame of ROUGE-2 recall per question, one column per chunk-based run:
  "Vector RAG (after selection agent)", "Vector RAG without selection agent", "Structure-guided hybrid",
  "Hybrid filter variant"). Use `exposure` instead of recomputing ROUGE-2. From Section 9: `COST_STAGES`, `per_query_cost`,
  `LATENCY`, `latency`, `results`. From `evaluation/cost_tracker.py`: `clean_stage_costs`,
  `dedupe_by_last`, `recompute_cost_usd`.
- Put every new cell **inside Section 13**, in the subsection named below, each with a short
  comment line saying which thesis section or table it supports (e.g. `# Thesis 4.2: ...`).
- Keep each new cell small and readable; print plain numbers with labels.
- **Do not commit.** Save the notebook with its outputs and stop, so the result can be reviewed.

## Step 1: small fixes to existing cells

1. In Section 13.5, the cell that starts with `# Sections 4.3.1 and 4.3.2` defines
   `correct = lambda a: ...`. This shadows the DataFrame `correct` defined in Section 3 and
   breaks earlier cells if they are re-run. Rename it to `is_correct` and update its uses in
   that cell.
2. In Section 8, the markdown cell that starts with
   `**Why the chunk-based pipelines struggle here: my reading of the evidence table and the examples**`,
   point 4 says that for Adobe's free-cash-flow conversion "both pipelines had the capex line ...
   and both refused". That is inaccurate. Replace that sentence with:
   "For Adobe's free-cash-flow conversion, both chunk pipelines refused, saying capital
   expenditure wasn't provided, and they were right: the cash flow statement on page 57 is split
   into two chunks, and both retrieved only the first (operating activities). The capex row
   ('Purchases of property and equipment (442) (348)') is in the second chunk,
   `ADOBE_2022_10K__p0056_c01`, which neither retrieved. The phrase 'purchases of property and
   equipment' appears in their context only in a narrative paragraph on page 47, without figures."
   Leave the rest of the cell unchanged.

## Step 2: add the missing cells

### In 13.4 (Answer quality)

**A. Variation between runs (thesis 4.2).** The filter variant and "vector RAG without selection
agent" (`plain_rerank_direct_*`) share the generation prompt, model and scoring. Find the
questions where `FI_RR[i]["chunk_ids"] == VR_RR[i]["chunk_ids"]` (identical input to the
generator) and count how many of them differ in correct / not correct between the two runs.
Expected: **27 questions with identical input, 3 differ.** (26 of these are questions where the
filter kept all 10 chunks; the 27th is a question whose reranked top 10 had fewer than 10
chunks.) Print both counts.

**B. Hybrid against the unfiltered top 10 (thesis 4.2.2).** For "PageIndex hybrid" and
"Vector RAG without selection agent", print Correct / Incorrect / Failure to Answer counts, and
the number correct per question type.
Expected: hybrid 111 / 25 / 14, unfiltered 105 / 31 / 14; correct by type (metrics-generated,
domain-relevant, novel-generated) hybrid 44 / 34 / 33, unfiltered 42 / 30 / 33.

### In 13.5 (Where the architectures fail)

**C. Accuracy by evidence exposure (thesis Table "Share of correct answers by evidence
exposure").** For the four chunk-based runs, take the ROUGE-2 recall from `exposure` (it already
uses the chunks each run passes to its generator). Bin each question into `< 0.5`, `0.5 to < 0.9` and `>= 0.9`, and
print per run and pooled over all four runs: number of questions and share correct.
Expected (share correct, n):

| Run | < 0.5 | 0.5 to 0.9 | >= 0.9 |
|---|---|---|---|
| Vector RAG | 59% (66) | 76% (50) | 85% (34) |
| Vector RAG without selection agent | 39% (28) | 79% (71) | 75% (51) |
| PageIndex hybrid | 56% (25) | 74% (62) | 81% (63) |
| Hybrid filter variant | 41% (37) | 74% (66) | 74% (47) |
| All four pooled | 51% (156) | 76% (249) | 79% (195) |

Also print: vectorless RAG correct among questions where navigation selected a gold page:
expected **114 of 141**.

**D. What the hybrid gains and loses (thesis 4.3.2).** Questions correct in the hybrid but not
in "Vector RAG without selection agent" (gained) and the reverse (lost). For each group print:
the size, the other run's verdicts, and the median ROUGE-2 recall with the hybrid's chunks and
with the unfiltered chunks.
Expected: **gained 14** (previously 9 Incorrect, 5 Failure to Answer), median ROUGE-2
**0.65** hybrid vs **0.46** unfiltered; **lost 8**, median **0.77** hybrid vs **0.90** unfiltered.

### New subsection 13.6 "Latency and cost (thesis 4.4 to 4.6)"

Add a markdown heading `### 13.6 Latency and cost (thesis Sections 4.4 to 4.6)`, then:

**E. 90th percentile latency.** Using the same per-run totals as `latency()` in Section 9
(totals plus the borrowed stage medians from `LATENCY`), print median and 90th percentile per
architecture. Expected (seconds): vector RAG 5.1 / 6.3; vectorless 3.6 / 4.1; hybrid 4.6 / 5.2;
long context 30.3 / 119.3.

**F. Median time per step.** From each latency file, print the median of every key in
`timings_sec` except `total`, plus the borrowed values. Expected (s): vector RAG expansion 1.30,
embedding 0.04, hybrid search 0.01, rerank 0.66, selection 1.22, generation 1.56; vectorless
navigation 0.46, generation 0.68, borrowed expansion 2.46; hybrid search 0.04, rerank 0.36,
generation 1.26, borrowed navigation 0.46, expansion 2.46, embedding 0.05; long context
generation 30.29. Note in a comment: vector RAG's own structured run measured expansion at
1.30 s, the borrowed value (from `vector_rag_latency.jsonl`) is 2.46 s.

**G. Tree-building time.** Read `experiments/results/vectorless_rag_indexing_latency.jsonl`
(keys `doc_name`, `method`, `elapsed_sec`, `n_pages`). Keep the **last** record per
`doc_name`, then print the number of filings, median, max and total `elapsed_sec`.
Expected: 84 filings, median ~188 s, max ~2,512 s, total ~671 minutes (about 11 hours).
Add a comment that the LLM-fallback rebuilds were not timed.

**H. Cost per step.** For each architecture in `COST_STAGES`, print the cost per 1,000 queries
of each stage separately (same cleaning as `per_query_cost`: `dedupe_by_last` when the output
path is `None`, otherwise `clean_stage_costs`, then `recompute_cost_usd`, then mean over the
kept records times 1000). Expected ($ per 1k): vector RAG expansion 0.10, selection 1.29,
generation 0.51 (total 1.91); vectorless expansion 0.09, navigation 3.03, generation 9.68
(12.80); hybrid generation 1.33, borrowed expansions 0.09 + 0.10, navigation 3.03 (4.54);
long context generation 19.96.

**I. Variant costs.** Generation cost per 1,000 queries and median generation input tokens for
`plain_rerank_direct_costs.jsonl` (match against `plain_rerank_direct_stage_generation.jsonl`)
and `rerank_then_filter_costs.jsonl` (match against
`rerank_then_filter_stage_generation.jsonl`), using `clean_stage_costs` + `recompute_cost_usd`.
Expected: without selection 1.36 $/1k, median 4,527 tokens, total with vector RAG's expansion
1.46 $/1k; filter 1.01 $/1k, median 3,100 tokens, total with both expansions and navigation
about 4.23 $/1k.

**J. Mean cost per query and totals.** From `results.cost_per_1k_usd`, print the mean cost per
query (divide by 1000) and the total for all 150 queries. Expected: 0.0019 / 0.0045 / 0.0128 /
0.0200 USD per query and 0.29 / 0.68 / 1.92 / 2.99 USD for 150 (vector RAG, hybrid,
vectorless, long context).

## Step 3: run and check

1. Restart the kernel and run all cells from the top (`Run All`). It takes a few minutes.
2. Confirm there are no errors.
3. Check the outputs of **all** Section 13 cells (old and new) against the expected values above
   and against `analysis/THESIS_NUMBERS_TODO.md`. Known values the existing Section 13 cells
   must print include: 35 questions with more than one gold page; 84 trees, median 30 nodes
   (range 1 to 386); **36** fallback trees; 18 trees with uncovered pages and 3 affected
   questions; 86 / 19 / 5 for all-four / none / exactly-one correct; the "gold page reached the
   generator" tables (vector RAG 104 with 79 / 13 / 12; hybrid 129 with 99 / 21 / 9; vectorless
   141 with 114 / 25 / 2; long context 150 with 118 / 27 / 5).
4. Save the notebook with outputs. **Do not commit.**

## Step 4: report back

Write a short summary listing:
- every cell you added or changed;
- every output that does **not** match an expected value above or in `THESIS_NUMBERS_TODO.md`,
  with the value the notebook printed (small rounding differences in the last digit are fine;
  list them anyway);
- anything you could not compute and why.
