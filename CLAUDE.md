# FinanceBench RAG Thesis — Project Brief for Claude Code

## What This Project Is

Master's thesis at Philipps-Universität Marburg, supervised by Prof. Dr. Daniel Braun.

**Title:** Vector RAG vs. Vectorless RAG vs. Long-Context LLMs on FinanceBench: An Accuracy–Latency–Cost Frontier Study

**Core goal:** Run a controlled three-way comparison of three retrieval architectures on the FinanceBench benchmark, measuring answer quality, retrieval quality, latency, and cost. The final output is an accuracy–latency–cost Pareto frontier that tells practitioners which architecture to use depending on their priorities.

**Deadline:** Experiments done by end of September 2026. Paper writing starts October 2026.

---

## The Three Pipelines

### 1. Vector RAG
- Based on Kim et al. (arXiv 2503.15191), ICLR 2025 Financial AI Workshop
- Repo: https://github.com/seohyunwoo-0407/GAR
- Key configuration:
  - Documents chunked into 512-token passages, restructured to markdown
  - Query expansion via LLM at query time
  - Embedding: fine-tuned Stella en 1.5B v5
  - Hybrid retrieval: dense + sparse, alpha = 0.85 (dense weighted higher)
  - Reranking: voyage-rerank-2, top 20 down to top 10
  - Selection agent filters final chunks before generation

### 2. Vectorless RAG (PageIndex)
- Based on Lumer et al. (arXiv 2511.18177)
- Repo: https://github.com/VectifyAI/PageIndex
- Key configuration:
  - No embeddings, no chunking
  - Document parsed into hierarchical tree (table of contents structure)
  - LLM agent navigates tree to find relevant nodes
  - Only raw text of selected nodes passed to generator
  - Failure mode: incorrect navigation to wrong node

### 3. Long-Context LLM
- No retrieval. Entire PDF placed in context window.
- Feasible because both generation models have 1M token context windows
- Expected: highest per-query cost and latency, no preprocessing cost

---

## Generation Models (used across all three pipelines)
- **Gemini 3.5 Flash** (Google) — originally Gemini 2.5 Flash per the brief, but that model was deprecated for new API keys/projects by the time experiments started (Aug 2026). Substituted with a pinned current version rather than `gemini-flash-latest`, to keep results reproducible/citable.
- **DeepSeek V4** (DeepSeek)

Both chosen for: 1M token context windows, manageable API costs (self-funded project).

---

## Dataset

**FinanceBench** by Islam et al. (arXiv 2311.11944)
- Repo: https://github.com/patronus-ai/financebench
- 150 questions from real SEC filings (10-K and 10-Q reports)
- Human-verified gold answers
- Gold source page annotations (used for retrieval evaluation)
- Two JSONLs in /data/, PDFs in /pdfs/
- Key fields: question, answer, evidence (with evidence_page_num), question_reasoning, doc_name

Load with:
```python
import pandas as pd
df_questions = pd.read_json("data/financebench_open_source.jsonl", lines=True)
df_meta = pd.read_json("data/financebench_document_information.jsonl", lines=True)
df_full = pd.merge(df_questions, df_meta, on="doc_name")
```

---

## Evaluation Metrics

### Answer Quality
- Three categories: Correct / Incorrect / Failure to Answer (Islam et al. scheme)
- Numerical questions: deterministic matching first (handle "1.2 billion" vs "1,200 million", small rounding)
- If deterministic fails or question is descriptive: LLM-as-a-judge
- Judge model: openai/gpt-oss-120b via Groq (originally Claude per the brief, but Claude's API requires billing — switched to Groq's free tier. Llama 3.3 70B was the first Groq pick but Groq deprecated it entirely before this got used; gpt-oss-120b is what's actually live on the free tier. Still satisfies the "different family from Gemini and DeepSeek" requirement that reduces self-enhancement bias; DeepSeek R1 was considered and rejected since it's the same vendor as the DeepSeek V4 generation model)
- Spot-check 10-15% of judge outputs manually

### Retrieval Quality (vector RAG and vectorless RAG only)
- MRR@k and Recall@k against gold evidence_page_num from FinanceBench
- Not applicable to long-context (acknowledged in paper, not treated as missing data)

### Latency
- End-to-end, query submission to answer, in seconds
- Run each query multiple times, report median
- Indexing/preprocessing latency reported separately (one-time per document)

### Cost
- Token consumption × published API prices
- Per-query cost and preprocessing cost reported separately
- Both generation model tokens and auxiliary model tokens (embedding, reranking, judge)

---

## Research Questions

- RQ1: How do the three architectures compare on answer quality on FinanceBench?
- RQ2: How do they compare on retrieval quality, latency, and cost?
- RQ3: Does any architecture dominate all axes, or does the optimal choice depend on priorities?
- RQ4: Can results be expressed as an accuracy–latency–cost Pareto frontier?

---

## Project Structure (to be created)

```
financebench-thesis/
├── CLAUDE.md                  # This file
├── data/                      # FinanceBench JSONLs
├── pdfs/                      # SEC filing PDFs from FinanceBench repo
├── pipelines/
│   ├── vector_rag/            # Adapted from GAR repo
│   ├── vectorless_rag/        # Adapted from PageIndex repo
│   └── long_context/          # Simple full-document pipeline
├── evaluation/
│   ├── answer_scorer.py       # Deterministic matcher + LLM judge
│   ├── retrieval_metrics.py   # MRR, Recall@k
│   └── cost_tracker.py        # Token logging and cost computation
├── experiments/
│   └── results/               # Raw outputs, one file per pipeline-model combo
├── analysis/
│   └── pareto.py              # Pareto frontier plots
└── requirements.txt
```

---

## Timeline

| Weeks | Dates | Goal |
|-------|-------|------|
| 1-2 | Late June | Clone repos, read code, install dependencies, understand structure |
| 3-4 | July 7-20 | Get each pipeline running end-to-end on one question |
| 5 | July 21-27 | Run all three on 10-15 question subset, manual review of outputs |
| 6-7 | July 28 - Aug 10 | Build evaluation framework (scorer, retrieval metrics, cost logger) |
| 8-9 | Aug 11-24 | Full experiment: all 150 questions, both models, 900 total runs |
| 10-11 | Aug 25 - Sep 7 | Aggregate results, compute metrics, build Pareto plots, analyze failures |
| 12-14 | Sep 8-30 | Write thesis draft (results + discussion; intro and related work already drafted) |

---

## How to Work With Me (the human)

- I am learning while building. Explain what you are doing and why, not just how.
- You can handle mundane setup tasks autonomously: creating folders, initializing repos, writing config files, installing dependencies, writing boilerplate code.
- For anything that involves key design decisions (how to structure the evaluation, how to adapt the GAR pipeline, prompt design for the judge), explain the options and let me decide.
- Always log costs and token counts from the start. Do not add this later.
- Keep the three pipelines as independent as possible so results are comparable.
- When in doubt, prioritize reproducibility over cleverness.

---

## Current Status

Week 1. No code written yet. Repos identified, dataset understood. Starting setup.
