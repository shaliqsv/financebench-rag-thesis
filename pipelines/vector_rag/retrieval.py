"""Hybrid retrieval (dense + BM25) and Voyage reranking.

Extracted from vector_rag_pipeline_local.ipynb Stages 5-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines.vector_rag.api_utils import with_retry
from pipelines.vector_rag.indexing import bm25_tokenize

ALPHA_DENSE = 0.85  # dense weight, per Kim et al.

# Kim et al. specify top_k=20 for the hybrid stage (reranked down to 10), but the
# Voyage account has no payment method yet, so it's capped at the free tier's
# ~10K-tokens/minute limit -- a 20-chunk candidate set routinely exceeds that and
# 429s every call. Defaulting to 10 here keeps runs free; bump back to 20 (matching
# the paper) once billing is added on Voyage, and note the change either way in the
# thesis methodology section.


def hybrid_retrieve(
    chunks_df: pd.DataFrame,
    dense_embeddings: np.ndarray,
    bm25_index,
    query_vec: np.ndarray,
    expanded_query: str,
    top_k: int = 10,
    alpha: float = ALPHA_DENSE,
) -> pd.DataFrame:
    """hybrid = alpha * dense_cosine + (1 - alpha) * bm25_minmax_normalized.

    BM25's raw scores are unbounded term-frequency sums (can be in the tens
    to hundreds) while dense cosine similarity is already in [0, 1] for
    normalized embeddings — averaging them directly would let dense dominate
    regardless of alpha, so BM25 is min-max normalized over the candidate
    set first.
    """
    dense_scores = dense_embeddings @ query_vec
    bm25_scores_raw = bm25_index.get_scores(bm25_tokenize(expanded_query))

    bm25_min, bm25_max = bm25_scores_raw.min(), bm25_scores_raw.max()
    bm25_scores_norm = (bm25_scores_raw - bm25_min) / (bm25_max - bm25_min + 1e-9)

    hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_scores_norm

    top_idx = np.argsort(-hybrid_scores)[:top_k]
    top = chunks_df.iloc[top_idx].copy()
    top["hybrid_score"] = hybrid_scores[top_idx]
    top["dense_score"] = dense_scores[top_idx]
    top["bm25_score"] = bm25_scores_raw[top_idx]
    return top


@with_retry()
def _voyage_rerank(voyage_client, query: str, documents: list[str], model: str, top_k: int):
    return voyage_client.rerank(query=query, documents=documents, model=model, top_k=top_k)


def rerank(
    voyage_client,
    query: str,
    candidates_df: pd.DataFrame,
    top_k: int = 10,
    model: str = "rerank-2",
) -> tuple[pd.DataFrame, int]:
    """Re-score candidates with Voyage rerank-2, return top_k in ranked order.

    Returns (reranked_df, total_tokens) — total_tokens is for cost logging.
    """
    result = _voyage_rerank(voyage_client, query, candidates_df["text"].tolist(), model, top_k)

    order = [r.index for r in result.results]
    scores = [r.relevance_score for r in result.results]

    top = candidates_df.iloc[order].copy()
    top["rerank_score"] = scores
    return top, result.total_tokens
