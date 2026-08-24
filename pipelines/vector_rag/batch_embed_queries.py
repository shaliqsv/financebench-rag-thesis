"""Batch query expansion + embedding for all FinanceBench questions.

Runs stage 4 (query expansion, a Gemini call) and the Stella embedding of the
expanded query for every question in df, resumable and fault-tolerant like
indexing.index_all_documents. Needs a GPU-backed embed_model (Stella), so
this is meant to run on the same Colab-backed kernel as indexing.

Saves two files per question into queries_dir:
  {financebench_id}__expanded.npy   - the query embedding
  {financebench_id}__expanded.json  - {financebench_id, doc_name, query_text}
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pipelines.vector_rag.generation import expand_query
from pipelines.vector_rag.indexing import format_query


def query_paths(queries_dir: Path, financebench_id: str) -> tuple[Path, Path]:
    return (
        queries_dir / f"{financebench_id}__expanded.npy",
        queries_dir / f"{financebench_id}__expanded.json",
    )


def is_query_embedded(queries_dir: Path, financebench_id: str) -> bool:
    vec_path, meta_path = query_paths(queries_dir, financebench_id)
    return vec_path.exists() and meta_path.exists()


def embed_all_queries(
    df: pd.DataFrame,
    genai_client,
    embed_model,
    expansion_model: str,
    queries_dir: Path,
    cost_tracker,
    sync_every: int | None = None,
    sync_fn: Callable[[], None] | None = None,
) -> dict:
    """Resumable batch driver: expand + embed every question's query.

    df must have columns: financebench_id, doc_name, question. One failed
    question (e.g. a malformed expansion response) is logged and skipped
    rather than aborting the whole batch — check
    queries_dir/embedding_errors.log afterward for a nonzero failure count.

    If sync_fn is given, it's called after every sync_every newly-embedded
    queries (and once more at the end, if anything new happened since the
    last call) — see indexing.index_all_documents for the same pattern.
    """
    queries_dir.mkdir(parents=True, exist_ok=True)
    errors_log = queries_dir / "embedding_errors.log"
    results = {"embedded": [], "skipped": [], "failed": []}
    synced_through = 0

    for i, row in enumerate(df.itertuples(), start=1):
        fb_id = row.financebench_id
        if is_query_embedded(queries_dir, fb_id):
            results["skipped"].append(fb_id)
            continue

        print(f"[{i}/{len(df)}] {fb_id}: expanding + embedding ...")
        t0 = time.time()
        try:
            expanded_query, usage = expand_query(genai_client, row.question, expansion_model)
            cost_tracker.log(
                pipeline="vector_rag", stage="query_expansion", model=expansion_model,
                input_tokens=usage.prompt_token_count, output_tokens=usage.candidates_token_count,
                doc_name=row.doc_name, financebench_id=fb_id,
            )

            query_vec = embed_model.encode(format_query(expanded_query), normalize_embeddings=True)

            vec_path, meta_path = query_paths(queries_dir, fb_id)
            np.save(vec_path, query_vec)
            meta_path.write_text(json.dumps({
                "financebench_id": fb_id,
                "doc_name": row.doc_name,
                "tag": "expanded",
                "query_text": expanded_query,
            }, indent=2))

            print(f"    done in {time.time() - t0:.1f}s")
            results["embedded"].append(fb_id)
        except Exception as e:
            print(f"    FAILED: {e}")
            with errors_log.open("a") as f:
                f.write(f"{fb_id}\t{e}\n{traceback.format_exc()}\n---\n")
            results["failed"].append(fb_id)

        if sync_fn is not None and sync_every and len(results["embedded"]) - synced_through >= sync_every:
            print(f"    -- syncing after {len(results['embedded'])} newly embedded queries --")
            sync_fn()
            synced_through = len(results["embedded"])

    if sync_fn is not None and len(results["embedded"]) > synced_through:
        print(f"    -- final sync ({len(results['embedded'])} newly embedded queries total) --")
        sync_fn()

    print(
        f"\nQuery embedding summary: {len(results['embedded'])} embedded, "
        f"{len(results['skipped'])} already done, {len(results['failed'])} failed"
    )
    if results["failed"]:
        print(f"Failed questions (see {errors_log}): {results['failed']}")

    return results
