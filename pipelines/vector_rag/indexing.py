"""Dense (Stella) + sparse (BM25) indexing, and the per-document batch driver.

Stella 1.5B needs a GPU-backed kernel to load in reasonable time (confirmed:
it fails to even load on a local machine's RAM, let alone embed on CPU) — see
vector_rag_pipeline.ipynb Stage 3. This module is written to be imported from
that notebook running against a Colab-backed kernel; it does not import
sentence-transformers/torch at module load time so the rest of the pipeline
(retrieval math, generation, scoring) can still be imported without them.
"""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from pipelines.vector_rag.chunking import parse_and_chunk_document

_BM25_TOKEN_RE = re.compile(r"[a-z0-9$][a-z0-9.,%$-]*")


def bm25_tokenize(text: str) -> list[str]:
    return _BM25_TOKEN_RE.findall(text.lower())


def build_bm25(chunks_df: pd.DataFrame) -> BM25Okapi:
    return BM25Okapi([bm25_tokenize(t) for t in chunks_df["text"]])


def format_query(query: str) -> str:
    return f"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {query}"


def index_paths(index_dir: Path, doc_name: str) -> tuple[Path, Path]:
    return index_dir / f"{doc_name}_chunks.parquet", index_dir / f"{doc_name}_dense.npy"


def is_indexed(index_dir: Path, doc_name: str) -> bool:
    chunks_path, dense_path = index_paths(index_dir, doc_name)
    return chunks_path.exists() and dense_path.exists()


def load_index(index_dir: Path, doc_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    chunks_path, dense_path = index_paths(index_dir, doc_name)
    chunks_df = pd.read_parquet(chunks_path)
    dense_embeddings = np.load(dense_path)
    assert len(chunks_df) == dense_embeddings.shape[0], f"{doc_name}: chunks/embeddings length mismatch"
    return chunks_df, dense_embeddings


def index_document(
    pdf_path: Path,
    doc_name: str,
    embed_model,
    count_tokens: Callable[[str], int],
    index_dir: Path,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse + chunk + embed one document, save to index_dir, return the result.

    Caller is responsible for skip-if-already-indexed (see index_all_documents)
    so this function always does the full (slow) work when called directly.
    """
    chunks_df = parse_and_chunk_document(pdf_path, doc_name, count_tokens)

    dense_embeddings = embed_model.encode(
        chunks_df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path, dense_path = index_paths(index_dir, doc_name)
    np.save(dense_path, dense_embeddings)
    chunks_df.to_parquet(chunks_path)

    return chunks_df, dense_embeddings


def index_all_documents(
    doc_names: list[str],
    pdf_dir: Path,
    embed_model,
    count_tokens: Callable[[str], int],
    index_dir: Path,
    batch_size: int = 8,
) -> dict:
    """Resumable batch driver: index every doc in doc_names not already saved.

    A single bad PDF must not abort a 7+ hour unattended run, so failures are
    caught, logged to index_dir/indexing_errors.log, and skipped — check that
    file after the run finishes to see whether the failure count is nonzero.
    Safe to re-run after an interruption: already-indexed docs are skipped.
    """
    errors_log = index_dir / "indexing_errors.log"
    results = {"indexed": [], "skipped": [], "failed": []}

    for i, doc_name in enumerate(doc_names, start=1):
        if is_indexed(index_dir, doc_name):
            results["skipped"].append(doc_name)
            print(f"[{i}/{len(doc_names)}] {doc_name}: already indexed, skipping")
            continue

        pdf_path = pdf_dir / f"{doc_name}.pdf"
        print(f"[{i}/{len(doc_names)}] {doc_name}: indexing ...")
        t0 = time.time()
        try:
            chunks_df, _ = index_document(pdf_path, doc_name, embed_model, count_tokens, index_dir, batch_size)
            print(f"    done in {time.time() - t0:.1f}s — {len(chunks_df)} chunks")
            results["indexed"].append(doc_name)
        except Exception as e:
            print(f"    FAILED: {e}")
            with errors_log.open("a") as f:
                f.write(f"{doc_name}\t{e}\n{traceback.format_exc()}\n---\n")
            results["failed"].append(doc_name)

    print(
        f"\nIndexing summary: {len(results['indexed'])} indexed, "
        f"{len(results['skipped'])} already done, {len(results['failed'])} failed"
    )
    if results["failed"]:
        print(f"Failed docs (see {errors_log}): {results['failed']}")

    return results
