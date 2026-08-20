"""Aggregate a finished (or partially finished) run_all_questions results.jsonl
into the headline numbers: answer-quality breakdown, retrieval Recall@k/MRR@k,
latency, and token totals per stage.

Cost in USD is not computed here since PRICING_PER_MILLION_TOKENS in
evaluation/cost_tracker.py is still unfilled — see that file's docstring.
Fill in the published rates, then cost_usd will be populated in the
per-record log and can be summed the same way as tokens below.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.retrieval_metrics import K_VALUES


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_results(results_path: Path, cost_log_path: Path | None = None) -> dict:
    records = _load_jsonl(results_path)
    if not records:
        print(f"No results yet in {results_path}")
        return {}

    n = len(records)
    label_counts = Counter(r["score_label"] for r in records)

    def _mean_metric(field: str, k: int) -> float:
        # json.dumps stringifies int dict keys, so k must be looked up as str(k) here.
        return sum(r[field][str(k)] for r in records) / n

    retrieval_summary = {}
    for stage_prefix in ("hybrid", "rerank"):
        retrieval_summary[stage_prefix] = {
            "recall_at_k": {k: _mean_metric(f"retrieval_{stage_prefix}_recall_at_k", k) for k in K_VALUES},
            "mrr_at_k": {k: _mean_metric(f"retrieval_{stage_prefix}_mrr_at_k", k) for k in K_VALUES},
        }

    latencies = [r["latency_seconds"] for r in records]

    summary = {
        "n_questions": n,
        "answer_quality": {
            label: {"count": count, "pct": round(100 * count / n, 1)}
            for label, count in label_counts.items()
        },
        "retrieval": retrieval_summary,
        "latency_seconds": {
            "median": statistics.median(latencies),
            "mean": statistics.mean(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
    }

    if cost_log_path is not None:
        cost_records = _load_jsonl(cost_log_path)
        by_stage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        for r in cost_records:
            s = by_stage[r["stage"]]
            s["input_tokens"] += r["input_tokens"]
            s["output_tokens"] += r["output_tokens"]
            s["calls"] += 1
        summary["tokens_by_stage"] = dict(by_stage)

    return summary


def print_summary(summary: dict) -> None:
    if not summary:
        return
    print(f"Questions scored: {summary['n_questions']}")
    print("\nAnswer quality:")
    for label, stats in summary["answer_quality"].items():
        print(f"  {label:<20} {stats['count']:>4}  ({stats['pct']}%)")

    print("\nRetrieval quality (mean across questions):")
    for stage, m in summary["retrieval"].items():
        print(f"  {stage}:")
        print(f"    Recall@k: {m['recall_at_k']}")
        print(f"    MRR@k:    {m['mrr_at_k']}")

    lat = summary["latency_seconds"]
    print(f"\nLatency (stages 5-7, seconds): median={lat['median']:.2f}  mean={lat['mean']:.2f}  "
          f"min={lat['min']:.2f}  max={lat['max']:.2f}")

    if "tokens_by_stage" in summary:
        print("\nTokens by stage:")
        for stage, t in summary["tokens_by_stage"].items():
            print(f"  {stage:<18} calls={t['calls']:<5} in={t['input_tokens']:<10} out={t['output_tokens']}")
