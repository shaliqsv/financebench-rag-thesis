"""PDF -> Markdown -> 512-token page-bounded chunks.

Extracted from vector_rag_pipeline.ipynb Stage 2 so it can be called in a
loop over all 84 FinanceBench documents instead of once, interactively.

Chunks never span a page boundary: FinanceBench's gold evidence is a page
number (evidence_page_num), so every chunk needs an unambiguous page_num to
evaluate retrieval against it. See CLAUDE.md's Evaluation Metrics section.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import pymupdf4llm

_SENT_SPLIT_RE = re.compile(r"(?<!\d)\.[ \n]+(?=[A-Z])")


def sent_tokenize(text: str) -> list[str]:
    sentences = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(
    page_text: str,
    page_num: int,
    doc_name: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Split one page's markdown into <=max_tokens passages, page-bounded.

    Algorithm: greedily accumulate sentences until the token budget would be
    exceeded, flush, then seed the next chunk with the last ~overlap_tokens
    of the flushed chunk so boundary sentences appear in both chunks.
    """
    if not page_text.strip():
        return []

    sentences = sent_tokenize(page_text)
    chunks: list[dict] = []
    curr_sents: list[str] = []
    curr_tokens = 0

    for sent in sentences:
        sent_toks = count_tokens(sent)

        if curr_tokens + sent_toks > max_tokens and curr_sents:
            chunks.append({
                "doc_name": doc_name,
                "page_num": page_num,
                "chunk_idx": len(chunks),
                "text": " ".join(curr_sents),
                "token_count": curr_tokens,
            })
            overlap_sents, overlap_toks = [], 0
            for s in reversed(curr_sents):
                s_toks = count_tokens(s)
                if overlap_toks + s_toks <= overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_toks += s_toks
                else:
                    break
            curr_sents = overlap_sents + [sent]
            curr_tokens = overlap_toks + sent_toks
        else:
            curr_sents.append(sent)
            curr_tokens += sent_toks

    if curr_sents:
        chunks.append({
            "doc_name": doc_name,
            "page_num": page_num,
            "chunk_idx": len(chunks),
            "text": " ".join(curr_sents),
            "token_count": curr_tokens,
        })

    return chunks


def parse_and_chunk_document(
    pdf_path: Path,
    doc_name: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full Stage 2 for one document: PDF -> per-page markdown -> chunks_df.

    pymupdf4llm's page_number is 1-indexed; we store page_num 0-indexed so
    it matches FinanceBench's evidence_page_num convention exactly.
    """
    t0 = time.time()
    md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    if verbose:
        print(f"  parsed {len(md_pages)} pages in {time.time() - t0:.1f}s")

    all_chunks: list[dict] = []
    for page_dict in md_pages:
        pg_num = page_dict["metadata"]["page_number"] - 1  # -> 0-indexed
        all_chunks.extend(
            chunk_page(page_dict["text"], pg_num, doc_name, count_tokens, max_tokens, overlap_tokens)
        )

    chunks_df = pd.DataFrame(all_chunks).reset_index(drop=True)
    if chunks_df.empty:
        raise ValueError(f"{doc_name}: no chunks produced (empty/unparseable PDF?)")

    chunks_df["text"] = chunks_df["text"].str.strip()
    chunks_df["chunk_id"] = chunks_df.apply(
        lambda r: f"{r.doc_name}__p{r.page_num:04d}_c{r.chunk_idx:02d}", axis=1
    )
    return chunks_df
