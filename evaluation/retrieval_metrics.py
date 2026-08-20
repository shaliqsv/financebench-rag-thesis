"""Retrieval-quality metrics: MRR@k and Recall@k against FinanceBench gold evidence pages.

Applies to vector RAG and vectorless RAG only (not long-context, which has no
retrieval step to evaluate) — see CLAUDE.md's Evaluation Metrics section.

Metrics are computed at the *page* level, not the chunk level: a hit is any
retrieved chunk whose page_num matches a gold evidence_page_num, since
FinanceBench's gold annotations are pages, not chunks. Both are 0-indexed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

K_VALUES: tuple[int, ...] = (5, 10, 15)


def recall_at_k(ranked_pages: Sequence[int], gold_pages: Iterable[int], k: int) -> float:
    """Fraction of gold pages present anywhere in the top-k ranked pages."""
    gold_set = set(gold_pages)
    if not gold_set:
        raise ValueError("gold_pages must be non-empty")
    retrieved = set(ranked_pages[:k])
    return len(retrieved & gold_set) / len(gold_set)


def reciprocal_rank_at_k(ranked_pages: Sequence[int], gold_pages: Iterable[int], k: int) -> float:
    """1/rank of the first gold-page hit within the top-k ranked pages, else 0."""
    gold_set = set(gold_pages)
    for rank, page in enumerate(ranked_pages[:k], start=1):
        if page in gold_set:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalMetrics:
    financebench_id: str
    doc_name: str
    stage: str  # e.g. "hybrid_top20" or "rerank_top10" — which pipeline stage this measures
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr_at_k: dict[int, float] = field(default_factory=dict)


def compute_retrieval_metrics(
    ranked_pages: Sequence[int],
    gold_pages: Iterable[int],
    financebench_id: str,
    doc_name: str,
    stage: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> RetrievalMetrics:
    gold_set = set(gold_pages)
    return RetrievalMetrics(
        financebench_id=financebench_id,
        doc_name=doc_name,
        stage=stage,
        recall_at_k={k: recall_at_k(ranked_pages, gold_set, k) for k in k_values},
        mrr_at_k={k: reciprocal_rank_at_k(ranked_pages, gold_set, k) for k in k_values},
    )


def aggregate_retrieval_metrics(
    records: Sequence[RetrievalMetrics], k_values: tuple[int, ...] = K_VALUES
) -> dict:
    """Mean Recall@k / MRR@k across a set of per-question RetrievalMetrics."""
    n = len(records)
    if n == 0:
        raise ValueError("records must be non-empty")
    return {
        "n_questions": n,
        "recall_at_k": {k: sum(r.recall_at_k[k] for r in records) / n for k in k_values},
        "mrr_at_k": {k: sum(r.mrr_at_k[k] for r in records) / n for k in k_values},
    }
