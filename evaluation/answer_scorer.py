"""Answer-quality scoring: deterministic numeric matching + LLM-as-judge fallback.

Classifies every answer into one of Islam et al.'s three FinanceBench categories:
Correct / Incorrect / Failure to Answer (see CLAUDE.md's Evaluation Metrics section).

Deterministic matching runs first and only fires when both the gold answer and
the model answer contain an extractable number — it handles unit differences
("1.2 billion" vs "1,200 million") and small rounding via a relative tolerance.
Everything else (descriptive questions, unparseable answers, explicit refusals)
falls through to the LLM judge, since guessing "wrong" vs "failed to answer"
from missing-number heuristics alone is unreliable.

Judge model is OpenAI's gpt-oss-120b via Groq's free tier — originally spec'd as
Claude, switched because Claude requires billing and Groq's free tier doesn't.
(Llama 3.3 70B was the first pick here, but Groq had deprecated it entirely by
the time this ran — gpt-oss-120b is what's actually live on the free tier now.)
Still a different model family from both generation models (Gemini, DeepSeek),
which is what actually matters for avoiding self-enhancement bias — see CLAUDE.md.
Spot-check 10-15% of judge outputs by hand before trusting them for the real
experiment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

Label = Literal["Correct", "Incorrect", "Failure to Answer"]

_MAGNITUDE_WORDS = {
    "trillion": 1e12,
    "tn": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "thousand": 1e3,
}

# (?<![A-Za-z0-9]) / (?![A-Za-z0-9]) guard against matching digits glued to
# letters — without them "3M" yields a spurious "3" and "FY2018" yields a
# spurious "2018", both common in these questions/answers. A guard on just the
# single preceding/following character (excluding letters only) isn't enough:
# finditer can still start a fresh match *inside* a blocked digit run, since
# e.g. the "0" in "FY2018" is preceded by "2" (a digit, not a letter), so a
# naive letter-only lookbehind would still let "018" match as a spurious "18".
# Excluding digits too forces the whole contiguous run to match-or-skip as one.
#
# The two num alternatives matter: `\d{1,3}(?:,\d{3})+` only matches numbers that
# actually contain comma grouping (e.g. "1,577"); `\d+` handles plain digit runs
# with no commas (e.g. "1577.00"). An earlier version used `(?:,\d{3})*` (zero or
# more), which let the first alternative match just the first 1-3 digits of a
# longer comma-less number and silently truncate it (e.g. "1577.00" -> "157").
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<num>-?\$?\s*(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))"
    r"\s*(?P<mag>trillion|billion|million|thousand|bn|mm|tn|%)?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

REL_TOLERANCE = 0.01  # 1% — covers "small rounding" per the project brief
ABS_TOLERANCE = 0.005  # for gold values near zero, where relative tolerance breaks down

_QUESTION_UNIT_RE = re.compile(r"in\s+(?:usd\s+)?(thousand|million|billion|trillion)s?\b", re.IGNORECASE)


def infer_question_unit_multiplier(question: str) -> float:
    """FinanceBench gold answers are bare numbers already expressed in whatever unit the
    question asks for (e.g. "(in USD millions)"), with no unit word repeated in the answer
    itself. A model's free-text answer, by contrast, usually spells the unit back out
    ("$1,577 million"). Without accounting for this, a bare gold "1577" and a model's
    explicit "1,577 million" get scaled inconsistently (1,577 vs 1,577,000,000) and
    never match. This scans the question for that unit phrase so both sides can be
    normalized to the same base scale. Defaults to 1.0 (no scaling) if no unit phrase
    is found, e.g. for percentage or already-absolute-dollar questions."""
    m = _QUESTION_UNIT_RE.search(question)
    return _MAGNITUDE_WORDS[m.group(1).lower()] if m else 1.0


def extract_numbers(text: str, default_multiplier: float = 1.0) -> list[float]:
    """All plausible numeric values in text. A number with an explicit magnitude word
    attached ("1.2 billion") is scaled by that word, since it's an unambiguous statement
    of scale. A bare number with no attached magnitude word is scaled by
    `default_multiplier` instead — pass in infer_question_unit_multiplier(question) so
    bare numbers are interpreted in the unit the question actually asked for. A trailing
    "%" is always treated as multiplier 1.0 (explicit unit, never scaled)."""
    values = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num").replace("$", "").replace(",", "").strip()
        if not raw or raw in ("-", "."):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        mag = m.group("mag")
        if mag == "%":
            multiplier = 1.0
        elif mag:
            multiplier = _MAGNITUDE_WORDS[mag.lower()]
        else:
            multiplier = default_multiplier
        values.append(value * multiplier)
    return values


def numbers_match(a: float, b: float) -> bool:
    if abs(a - b) <= ABS_TOLERANCE:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= REL_TOLERANCE


@dataclass
class ScoreResult:
    label: Label
    method: Literal["deterministic", "llm_judge"]
    gold_value: float | None = None
    matched_value: float | None = None
    reasoning: str | None = None


def score_deterministic(question: str, gold_answer: str, model_answer: str) -> ScoreResult | None:
    """Returns a ScoreResult if both answers yield extractable numbers, else None (defer to judge)."""
    unit_multiplier = infer_question_unit_multiplier(question)
    gold_numbers = extract_numbers(gold_answer, default_multiplier=unit_multiplier)
    model_numbers = extract_numbers(model_answer, default_multiplier=unit_multiplier)
    if not gold_numbers or not model_numbers:
        return None

    gold_value = gold_numbers[0]  # FinanceBench gold answers are short; first number is the answer
    for candidate in model_numbers:
        if numbers_match(gold_value, candidate):
            return ScoreResult(
                label="Correct", method="deterministic", gold_value=gold_value, matched_value=candidate
            )
    return ScoreResult(
        label="Incorrect", method="deterministic", gold_value=gold_value, matched_value=model_numbers[0]
    )


JUDGE_PROMPT = """You are grading an AI system's answer to a question about a company's SEC filing, \
against a human-verified gold answer. Classify the model's answer into exactly one of three categories:

- "Correct": the model's answer matches the gold answer in substance (numbers may be phrased \
differently — e.g. rounding, units — but must be materially the same value or conclusion).
- "Incorrect": the model gave a definite answer, but it's wrong.
- "Failure to Answer": the model declined to answer, said the information wasn't available/sufficient, \
or gave a non-answer, rather than committing to a (possibly wrong) answer.

Question: {question}

Gold answer: {gold_answer}

Model answer: {model_answer}

Respond with ONLY a JSON object, no other text: {{"label": "<one of the three categories above>", "reasoning": "<one sentence>"}}"""


def score_with_judge(question: str, gold_answer: str, model_answer: str, client, model: str = "openai/gpt-oss-120b"):
    """client is a groq.Groq instance (OpenAI-compatible chat completions API). Judge model is
    gpt-oss-120b via Groq's free tier — a different model family from both generation models
    (Gemini, DeepSeek), which is what actually matters for avoiding self-enhancement bias, not the
    specific vendor. Returns (ScoreResult, usage) where usage has input/output token counts."""
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, gold_answer=gold_answer, model_answer=model_answer
                ),
            }
        ],
    )
    raw = resp.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    parsed = json.loads(match.group(0) if match else raw)

    label = parsed["label"]
    if label not in ("Correct", "Incorrect", "Failure to Answer"):
        raise ValueError(f"Judge returned an unrecognized label: {label!r}")

    result = ScoreResult(label=label, method="llm_judge", reasoning=parsed.get("reasoning"))
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    return result, usage


def score_answer(question: str, gold_answer: str, model_answer: str, judge_client=None, judge_model: str = "openai/gpt-oss-120b"):
    """Deterministic matching first; falls back to the LLM judge if numbers aren't extractable
    from both answers. Returns (ScoreResult, usage | None) — usage is only populated when the
    judge was actually called, so cost_tracker.log(...) has real token counts to record."""
    result = score_deterministic(question, gold_answer, model_answer)
    if result is not None:
        return result, None

    if judge_client is None:
        raise ValueError(
            "Deterministic matching failed (no extractable number in gold and/or model answer) "
            "and no judge_client was provided to fall back to the LLM judge."
        )
    return score_with_judge(question, gold_answer, model_answer, judge_client, judge_model)
