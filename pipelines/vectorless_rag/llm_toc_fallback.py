"""LLM-based table-of-contents fallback for documents where PageIndex Flash's
layout-heuristic parser silently drops pages (see the notebook's
`tree_coverage_gaps` check -- e.g. 3M_2018_10K, whose first detected heading
was on page 85 because Flash's font/layout heuristic never recognized any
heading in pages 1-134, even though that text has normal, readable section
headings).

This is NOT a replacement for the PageIndex pipeline. It's used only for the
specific documents that fail the coverage check, so the thesis's "vectorless
RAG via PageIndex" pipeline stays untouched for every document where Flash
actually works.

Every function here takes `model` and `llm_completion` as arguments rather
than importing/depending on them, since the notebook wraps `llm_completion`
with its own rate limiter and cost tracker -- this module has no opinion on
either and just calls whatever it's handed.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from pageindex.utils import write_node_id

LLMCompletion = Callable[..., str]


TOC_PROMPT = """You are extracting a table of contents from a financial filing (10-K/10-Q). \
Below is the full document, with each page's text marked by a "--- PAGE n ---" header (n is \
1-indexed).

Identify every top-level section (e.g. Item 1. Business, Item 7. MD&A, Item 8. Financial \
Statements, individual Notes to the Financial Statements, exhibits) and the page range it \
spans. The sections must tile the ENTIRE document with no gaps and no overlaps: the first \
section starts at page 1 and the last section ends at page {n_pages} (the document's last \
page). If a stretch of pages has no clear heading (e.g. a cover page or a table of contents \
page), attach it to whichever adjacent section it introduces rather than leaving it out.

Return ONLY a JSON array, ordered by page, in this exact format:
[{{"title": "<section title>", "start_page": <int>, "end_page": <int>}}, ...]

Document:
{document_text}"""


def _build_page_tagged_text(page_texts: list[str]) -> str:
    return "\n\n".join(f"--- PAGE {i + 1} ---\n{text}" for i, text in enumerate(page_texts))


def _patch_gaps(sections: list[dict], n_pages: int) -> list[dict]:
    """The prompt asks the model for gapless coverage, but nothing forces
    that -- this is the actual guarantee. Any hole the model leaves anyway
    (or any it never had a chance to fill, like a missing page 1) gets an
    "Unlabeled" filler section, since the property this fallback exists for
    is coverage, not heading quality."""
    patched = []
    cursor = 1
    for s in sections:
        if s["start_page"] > cursor:
            patched.append({"title": "Unlabeled", "start_page": cursor, "end_page": s["start_page"] - 1})
        patched.append(s)
        cursor = max(cursor, s["end_page"] + 1)
    if cursor <= n_pages:
        patched.append({"title": "Unlabeled", "start_page": cursor, "end_page": n_pages})
    return patched


def detect_toc_llm(page_texts: list[str], model: str, llm_completion: LLMCompletion) -> list[dict]:
    """One LLM call over the whole document -- a filing-sized PDF's text fits
    comfortably in the project's 1M-token generation models, so no chunking
    needed. Returns a flat, gap-patched list of {"title", "start_page",
    "end_page"} dicts covering pages 1..n_pages."""
    n_pages = len(page_texts)
    prompt = TOC_PROMPT.format(n_pages=n_pages, document_text=_build_page_tagged_text(page_texts))
    raw_response = llm_completion(model, prompt)

    match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    sections = json.loads(match.group(0)) if match else []
    sections = [s for s in sections if s.get("start_page") and s.get("end_page")]
    sections.sort(key=lambda s: s["start_page"])

    return _patch_gaps(sections, n_pages)


SUMMARY_PROMPT = """Summarize the following section of a financial filing in 2-3 sentences, \
focused on the specific figures, line items, or topics it covers -- detailed enough that \
someone deciding whether this section answers a specific question could tell from the summary \
alone.

Section title: {title}

Section text:
{text}"""

# Below this many characters, use the raw text as the "summary" instead of
# spending an LLM call on it -- matches PageIndex's own small-node convention.
SUMMARY_RAW_TEXT_CHARS = 800


def summarize_section_llm(title: str, text: str, model: str, llm_completion: LLMCompletion) -> str:
    if len(text) <= SUMMARY_RAW_TEXT_CHARS:
        return text.strip()
    return llm_completion(model, SUMMARY_PROMPT.format(title=title, text=text)).strip()


def build_tree_llm_fallback(doc_name: str, page_texts: list[str], model: str, llm_completion: LLMCompletion) -> dict:
    """Drop-in replacement for the notebook's build_tree_flash return shape --
    same {"doc_name", "structure", "page_texts", "n_pages"} -- so it saves and
    loads through the exact same tree_path/load_tree/create_node_mapping
    helpers the rest of the notebook already uses. Flat structure (no nested
    sub-nodes): the point of this fallback is guaranteed page coverage, not
    matching Flash's hierarchy depth."""
    n_pages = len(page_texts)
    sections = detect_toc_llm(page_texts, model, llm_completion)

    structure = []
    for s in sections:
        start, end = s["start_page"], s["end_page"]
        section_text = "\n".join(page_texts[start - 1:end])
        structure.append({
            "title": s["title"],
            "start_index": start,
            "end_index": end,
            "summary": summarize_section_llm(s["title"], section_text, model, llm_completion),
            "nodes": [],
        })

    write_node_id(structure)
    return {"doc_name": doc_name, "structure": structure, "page_texts": page_texts, "n_pages": n_pages}
