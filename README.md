# FinanceBench RAG Thesis

Master's thesis (Philipps-Universität Marburg, supervisor: Prof. Dr. Daniel Braun):
**Vector RAG vs. Vectorless RAG vs. Long-Context LLMs on FinanceBench: An
Accuracy–Latency–Cost Frontier Study**. See [CLAUDE.md](CLAUDE.md) for the full
project brief, architecture details, and timeline.

## Setup

```bash
uv sync
cp .env.example .env   # fill in your API keys
```

## Layout

- `data/`, `pdfs/` — FinanceBench questions/metadata and the 84 source PDFs referenced
  by the 150 open-source questions.
- `external/` — read-only reference clones of the upstream GAR and PageIndex repos
  (not tracked in git; re-fetch with `git clone` if missing).
- `pipelines/{vector_rag,vectorless_rag,long_context}/` — the three architectures
  under comparison, kept independent of each other.
- `evaluation/` — answer scoring, retrieval metrics, cost/token logging.
- `experiments/results/` — raw per-pipeline, per-model run outputs.
- `analysis/` — Pareto frontier plots and aggregate analysis.
