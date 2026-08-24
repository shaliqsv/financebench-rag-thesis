"""Stages 5-7 (hybrid retrieve -> rerank -> select -> generate -> score) for
every FinanceBench question, resumable and fault-tolerant.

Does NOT need Stella/GPU: indexing (indexing.index_all_documents) and query
embedding (batch_embed_queries.embed_all_queries) must have already run and
saved their outputs to index_dir / queries_dir. This function only does
numpy/BM25 math plus API calls (Voyage, Gemini, Groq judge), so it's safe to
run on a plain CPU kernel if that's ever more convenient than the Colab one.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.answer_scorer import score_answer
from evaluation.retrieval_metrics import compute_retrieval_metrics
from pipelines.vector_rag.api_utils import with_retry
from pipelines.vector_rag.batch_embed_queries import is_query_embedded, query_paths
from pipelines.vector_rag.generation import generate_answer, select_chunks
from pipelines.vector_rag.indexing import build_bm25, is_indexed, load_index
from pipelines.vector_rag.retrieval import hybrid_retrieve, rerank

_score_answer = with_retry()(score_answer)


def _load_completed_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["financebench_id"])
    return ids


def run_all_questions(
    df: pd.DataFrame,
    index_dir: Path,
    queries_dir: Path,
    out_path: Path,
    genai_client,
    voyage_client,
    judge_client,
    generation_model: str,
    judge_model: str,
    cost_tracker,
    top_k_hybrid: int = 10,
    top_k_rerank: int = 10,
    sync_every: int | None = None,
    sync_fn=None,
) -> dict:
    """df must have: financebench_id, doc_name, question, answer, evidence.

    Appends one JSON line per question to out_path as soon as it finishes, so
    an interrupted run loses at most the one in-flight question, and re-running
    this function skips every financebench_id already present in out_path.

    If sync_fn is given, it's called after every sync_every newly-completed
    questions (and once more at the end, if anything new happened since the
    last call) — see indexing.index_all_documents for the same pattern.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_log = out_path.parent / "run_errors.log"
    completed = _load_completed_ids(out_path)
    synced_through = 0

    doc_cache: dict[str, tuple] = {}  # doc_name -> (chunks_df, dense_embeddings, bm25_index)
    results = {"done": [], "skipped": [], "failed": [], "missing_index": [], "missing_query": []}

    for i, row in enumerate(df.itertuples(), start=1):
        fb_id = row.financebench_id
        if fb_id in completed:
            results["skipped"].append(fb_id)
            continue

        if not is_indexed(index_dir, row.doc_name):
            print(f"[{i}/{len(df)}] {fb_id}: SKIPPED — {row.doc_name} not indexed yet")
            results["missing_index"].append(fb_id)
            continue
        if not is_query_embedded(queries_dir, fb_id):
            print(f"[{i}/{len(df)}] {fb_id}: SKIPPED — query not embedded yet")
            results["missing_query"].append(fb_id)
            continue

        print(f"[{i}/{len(df)}] {fb_id}: running ...")
        t0 = time.time()
        try:
            if row.doc_name not in doc_cache:
                chunks_df, dense_embeddings = load_index(index_dir, row.doc_name)
                bm25_index = build_bm25(chunks_df)
                doc_cache[row.doc_name] = (chunks_df, dense_embeddings, bm25_index)
            chunks_df, dense_embeddings, bm25_index = doc_cache[row.doc_name]

            vec_path, meta_path = query_paths(queries_dir, fb_id)
            query_vec = np.load(vec_path)
            expanded_query = json.loads(meta_path.read_text())["query_text"]

            gold_pages = [e["evidence_page_num"] for e in row.evidence]

            top_hybrid = hybrid_retrieve(chunks_df, dense_embeddings, bm25_index, query_vec, expanded_query, top_k=top_k_hybrid)
            hybrid_metrics = compute_retrieval_metrics(
                ranked_pages=top_hybrid["page_num"].tolist(), gold_pages=gold_pages,
                financebench_id=fb_id, doc_name=row.doc_name, stage=f"hybrid_top{top_k_hybrid}",
            )

            top_rerank, rerank_tokens = rerank(voyage_client, expanded_query, top_hybrid, top_k=top_k_rerank)
            cost_tracker.log(
                pipeline="vector_rag", stage="rerank", model="voyage-rerank-2",
                input_tokens=rerank_tokens, output_tokens=0, doc_name=row.doc_name, financebench_id=fb_id,
            )
            rerank_metrics = compute_retrieval_metrics(
                ranked_pages=top_rerank["page_num"].tolist(), gold_pages=gold_pages,
                financebench_id=fb_id, doc_name=row.doc_name, stage=f"rerank_top{top_k_rerank}",
            )

            selected_ids, sel_usage = select_chunks(genai_client, row.question, top_rerank, generation_model)
            cost_tracker.log(
                pipeline="vector_rag", stage="selection", model=generation_model,
                input_tokens=sel_usage.prompt_token_count, output_tokens=sel_usage.candidates_token_count,
                doc_name=row.doc_name, financebench_id=fb_id,
            )
            selected_df = top_rerank[top_rerank.chunk_id.isin(selected_ids)]

            answer, gen_usage = generate_answer(genai_client, row.question, selected_df, generation_model)
            cost_tracker.log(
                pipeline="vector_rag", stage="generation", model=generation_model,
                input_tokens=gen_usage.prompt_token_count, output_tokens=gen_usage.candidates_token_count,
                doc_name=row.doc_name, financebench_id=fb_id,
            )

            score_result, judge_usage = _score_answer(row.question, row.answer, answer, judge_client=judge_client, judge_model=judge_model)
            if judge_usage is not None:
                cost_tracker.log(
                    pipeline="vector_rag", stage="judge", model=judge_model,
                    input_tokens=judge_usage["input_tokens"], output_tokens=judge_usage["output_tokens"],
                    doc_name=row.doc_name, financebench_id=fb_id,
                )

            latency = time.time() - t0
            record = {
                "financebench_id": fb_id,
                "doc_name": row.doc_name,
                "question": row.question,
                "gold_answer": row.answer,
                "model_answer": answer,
                "score_label": score_result.label,
                "score_method": score_result.method,
                "generation_model": generation_model,
                "n_selected_chunks": len(selected_df),
                "n_rerank_chunks": len(top_rerank),
                "retrieval_hybrid_recall_at_k": hybrid_metrics.recall_at_k,
                "retrieval_hybrid_mrr_at_k": hybrid_metrics.mrr_at_k,
                "retrieval_rerank_recall_at_k": rerank_metrics.recall_at_k,
                "retrieval_rerank_mrr_at_k": rerank_metrics.mrr_at_k,
                "latency_seconds": latency,
                "timestamp": time.time(),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

            print(f"    done in {latency:.1f}s — {score_result.label}")
            results["done"].append(fb_id)
        except Exception as e:
            print(f"    FAILED: {e}")
            with errors_log.open("a") as f:
                f.write(f"{fb_id}\t{e}\n{traceback.format_exc()}\n---\n")
            results["failed"].append(fb_id)

        if sync_fn is not None and sync_every and len(results["done"]) - synced_through >= sync_every:
            print(f"    -- syncing after {len(results['done'])} newly completed questions --")
            sync_fn()
            synced_through = len(results["done"])

    if sync_fn is not None and len(results["done"]) > synced_through:
        print(f"    -- final sync ({len(results['done'])} newly completed questions total) --")
        sync_fn()

    print(
        f"\nRun summary: {len(results['done'])} done, {len(results['skipped'])} already done, "
        f"{len(results['failed'])} failed, {len(results['missing_index'])} missing index, "
        f"{len(results['missing_query'])} missing query embedding"
    )
    if results["failed"]:
        print(f"Failed questions (see {errors_log}): {results['failed']}")

    return results
