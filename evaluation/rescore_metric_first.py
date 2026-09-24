"""Re-score an existing pipeline's saved answers with score_answer_metric_first, so all
four architectures are graded by the same rule.

Vector RAG and long context were originally scored with score_answer (deterministic
matching on every question, a deterministic verdict always final); vectorless and the
PageIndex hybrid already use the metric-first rule. See score_answer_metric_first's
docstring for why that rule is the better one.

Writes <pipeline>_stage_scoring_metric_first.jsonl next to the original scoring file,
which stays untouched. Resumable: re-running skips questions already re-scored. Judge
calls are logged to the pipeline's own cost log under stage "judge_rescore" (Groq free
tier, so $0, but logged anyway).

Usage (from the repo root):
    python -m evaluation.rescore_metric_first vector_rag long_context
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from evaluation.answer_scorer import score_answer_metric_first
from evaluation.cost_tracker import CostTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
JUDGE_MODEL = "openai/gpt-oss-120b"
MIN_SECONDS_BETWEEN_CALLS = 2.5   # Groq free tier: 30 requests/minute for this model


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _latest_by_id(path: Path) -> dict:
    records = sorted(_load_jsonl(path), key=lambda r: r.get("timestamp", 0))
    return {r["financebench_id"]: r for r in records}


def _judge_with_retry(fn, max_retries=6):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"    retrying in {wait}s after: {str(e)[:120]}")
            time.sleep(wait)


def rescore(pipeline: str, judge_client) -> None:
    questions = {r["financebench_id"]: r for r in _load_jsonl(REPO_ROOT / "data" / "financebench_open_source.jsonl")}
    generation = _latest_by_id(RESULTS_DIR / f"{pipeline}_stage_generation.jsonl")
    out_path = RESULTS_DIR / f"{pipeline}_stage_scoring_metric_first.jsonl"
    done = {r["financebench_id"] for r in _load_jsonl(out_path)}
    cost_tracker = CostTracker(RESULTS_DIR / f"{pipeline}_costs.jsonl")

    todo = [fb_id for fb_id in generation if fb_id not in done]
    print(f"{pipeline}: {len(generation)} answers, {len(done)} already re-scored, {len(todo)} to go")
    last_call = 0.0
    for i, fb_id in enumerate(todo, start=1):
        q, answer = questions[fb_id], generation[fb_id]["model_answer"]

        def call():
            nonlocal last_call
            time.sleep(max(0.0, MIN_SECONDS_BETWEEN_CALLS - (time.time() - last_call)))
            last_call = time.time()
            return score_answer_metric_first(q["question"], q["question_type"], q["answer"], answer,
                                             judge_client, JUDGE_MODEL)

        t0 = time.time()
        final, det, judge, usage = _judge_with_retry(call)
        if usage is not None:
            cost_tracker.log(pipeline=pipeline, stage="judge_rescore", model=JUDGE_MODEL,
                             input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
                             doc_name=q["doc_name"], financebench_id=fb_id, latency_sec=time.time() - t0)
        record = {
            "financebench_id": fb_id, "doc_name": q["doc_name"], "question_type": q["question_type"],
            "question": q["question"], "gold_answer": q["answer"], "model_answer": answer,
            "deterministic_label": det.label if det else ("Inconclusive" if q["question_type"] == "metrics-generated" else "NA"),
            "deterministic_gold_value": det.gold_value if det else None,
            "deterministic_matched_value": det.matched_value if det else None,
            "judge_label": judge.label if judge else None,
            "judge_reasoning": judge.reasoning if judge else None,
            "label": final.label, "method": final.method,
            "timestamp": time.time(),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [{i}/{len(todo)}] {fb_id}: {final.label} ({final.method})")


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env", override=True)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for pipeline in sys.argv[1:] or ["vector_rag", "long_context"]:
        rescore(pipeline, client)
