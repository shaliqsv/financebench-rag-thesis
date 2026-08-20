"""Query expansion, chunk selection, and answer generation (Gemini calls).

Extracted from vector_rag_pipeline_local.ipynb Stages 4 and 7. Only Gemini is
wired up for now (DeepSeek deferred — see CLAUDE.md); model is always passed
explicitly rather than hardcoded so swapping generation models later is a
config change here, not a rewrite.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from pipelines.vector_rag.api_utils import with_retry

EXPANSION_PROMPT = """You are helping a retrieval system find the right passage in a financial \
filing (10-K/10-Q). Rewrite the question below into a short expanded search query: include likely \
synonyms, exact financial line-item names, and related terms that would appear near the answer in \
the filing. Return ONLY the expanded query text - no preamble, no explanation, no quotes.

Question: {question}"""

SELECTION_PROMPT = """You are filtering retrieved passages from a financial filing before they are \
passed to an answer-generation model. Given the question and candidate passages below, return ONLY \
the passage IDs that are actually needed to answer the question, as a JSON array of strings \
(e.g. ["id1", "id2"]). Drop passages that are irrelevant or redundant. Keep at least one passage. \
Return nothing except the JSON array.

Question: {question}

Candidate passages:
{passages}"""

GENERATION_PROMPT = """You are a financial analyst answering a question using only the excerpts \
below from a company's SEC filing. Answer concisely and precisely, matching the format the question \
expects (a number, a yes/no with brief reasoning, etc). If the excerpts don't contain enough \
information to answer, say so explicitly rather than guessing.

Question: {question}

Excerpts:
{passages}

Answer:"""


def format_passages(df: pd.DataFrame) -> str:
    return "\n\n".join(f"[{r.chunk_id}] (page {r.page_num})\n{r.text}" for _, r in df.iterrows())


@with_retry()
def _generate(genai_client, model: str, prompt: str):
    return genai_client.models.generate_content(model=model, contents=prompt)


def expand_query(genai_client, question: str, model: str):
    resp = _generate(genai_client, model, EXPANSION_PROMPT.format(question=question))
    return resp.text.strip(), resp.usage_metadata


def select_chunks(genai_client, question: str, candidates_df: pd.DataFrame, model: str):
    resp = _generate(
        genai_client, model, SELECTION_PROMPT.format(question=question, passages=format_passages(candidates_df))
    )
    raw = resp.text.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    ids = json.loads(match.group(0) if match else raw)
    if not ids:
        # Selection prompt asks the model to keep at least one, but don't trust
        # that blindly — an empty selection would silently generate from zero
        # context, producing a spurious "Failure to Answer" that looks like a
        # retrieval miss when it's actually a selection-stage bug.
        ids = candidates_df["chunk_id"].tolist()
    return ids, resp.usage_metadata


def generate_answer(genai_client, question: str, context_df: pd.DataFrame, model: str):
    resp = _generate(genai_client, model, GENERATION_PROMPT.format(question=question, passages=format_passages(context_df)))
    return resp.text.strip(), resp.usage_metadata
