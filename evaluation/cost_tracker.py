"""Token and cost logging shared by all three pipelines.

Pricing is per 1M tokens and must be filled in with the currently published
rates before running real experiments — left as None so a missing price
fails loudly instead of silently logging zero cost.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float | None]] = {
    "gemini-3.5-flash": {"input": None, "output": None},
    # standard (non-batch) paid tier, confirmed against ai.google.dev/gemini-api/docs/pricing
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "deepseek-v4": {"input": None, "output": None},
    "voyage-rerank-2": {"input": None, "output": None},
    "stella-en-1.5b-v5": {"input": None, "output": None},
    # judge, via Groq's free tier -- genuinely $0 (no card, rate-limited not billed).
    # Groq's paid on-demand rate for the same model, for reference if usage ever
    # exceeds the free tier: $0.15 input / $0.60 output per 1M tokens.
    "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},
}


@dataclass
class UsageRecord:
    timestamp: float
    pipeline: str  # "vector_rag" | "vectorless_rag" | "long_context"
    stage: str  # e.g. "generation", "embedding", "rerank", "judge", "indexing"
    model: str
    doc_name: str | None
    financebench_id: int | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_sec: float | None = None  # wall-clock time of this one call, when the caller timed it


class CostTracker:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        pipeline: str,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        doc_name: str | None = None,
        financebench_id: int | None = None,
        latency_sec: float | None = None,
    ) -> UsageRecord:
        prices = PRICING_PER_MILLION_TOKENS.get(model)
        cost_usd = None
        if prices and prices["input"] is not None and prices["output"] is not None:
            cost_usd = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

        record = UsageRecord(
            timestamp=time.time(),
            pipeline=pipeline,
            stage=stage,
            model=model,
            doc_name=doc_name,
            financebench_id=financebench_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_sec=latency_sec,
        )
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        return record


# --- Cleaning up an append-only cost log --------------------------------
#
# CostTracker.log() has no dedup: every real API call appends a record, with
# no check for "have I already logged this question's answer". That's fine
# for the resumable run_*_all loops (each question triggers exactly one real
# call per stage, so one log line per call = one log line per question) --
# but every notebook also has non-resumable cells that call the same
# functions directly: smoke tests, a "bottom-up walkthrough" section that
# demos each stage function on one example question, and any cell re-run
# while debugging. None of those check "already done" before calling, so
# every re-run appends more cost records for the same question -- with
# nothing in the log itself to tell a real, paid-for retry (after a
# transient error) apart from harmless demo noise fired hours later.
#
# The fix: a stage's own *output* file already solves this problem for
# answers -- it's resumable (one line per financebench_id, the line that
# survived) and each line already carries a `timestamp`. So instead of
# guessing which cost record is "real", match each output record to the
# cost record for that (financebench_id, stage) whose timestamp sits
# closest to -- and at or before -- the output's own timestamp. That's the
# call that produced the saved result; everything else for that id (earlier
# failed attempts already retried away, or demo/rerun noise from any other
# time) is dropped.

def clean_stage_costs(
    cost_records: list[dict],
    stage_name: str,
    output_records: dict[str, dict],
    id_field: str = "financebench_id",
    output_timestamp_field: str = "timestamp",
    grace_sec: float = 300.0,
) -> tuple[list[dict], dict]:
    """De-duplicates `cost_records` for one stage down to (at most) one record per id,
    by matching against `output_records` (id -> that stage's saved output row, keyed the
    same way `_load_stage_records` keys a resumable output file).

    For each id: among cost records for (id, stage_name), picks the one whose timestamp is
    <= output timestamp + grace_sec and closest to it. `grace_sec` absorbs the gap between
    "API call returned" and "line written to disk" plus any clock skew -- it is not meant to
    reach back across a whole separate rerun.

    Returns (kept_records, report). report has:
      n_input, n_kept, n_dropped_as_noise,
      missing_cost: ids with a saved answer but no matching cost record at all,
      late_only: ids where every cost record for them came in *after* the grace window --
        used anyway (closest one kept), but flagged since the ordering is unexpected and
        worth a manual look,
      no_output_timestamp: ids whose output row predates this timestamp field being added
        (an old, already-saved answer) -- can't be timestamp-matched at all. If there's
        exactly one cost candidate for that id, it's unambiguous and used anyway; with
        more than one, the most recent is used as a labeled guess (same heuristic as
        clean_doc_level_costs) and the id is listed here so it can be spot-checked or the
        stage re-run for a precise number instead of a guess.
    """
    by_id: dict[str, list[dict]] = {}
    for r in cost_records:
        if r.get("stage") == stage_name:
            by_id.setdefault(r.get(id_field), []).append(r)

    kept, missing_cost, late_only, no_output_timestamp = [], [], [], []
    for oid, out_rec in output_records.items():
        candidates = by_id.get(oid, [])
        if not candidates:
            missing_cost.append(oid)
            continue
        out_ts = out_rec.get(output_timestamp_field)
        if out_ts is None:
            no_output_timestamp.append(oid)
            best = candidates[0] if len(candidates) == 1 else max(candidates, key=lambda c: c["timestamp"])
            kept.append(best)
            continue
        on_time = [c for c in candidates if c["timestamp"] <= out_ts + grace_sec]
        pool = on_time or candidates
        if not on_time:
            late_only.append(oid)
        best = min(pool, key=lambda c: abs(c["timestamp"] - out_ts))
        kept.append(best)

    report = {
        "n_input": sum(len(v) for v in by_id.values()),
        "n_kept": len(kept),
        "n_dropped_as_noise": sum(len(v) for v in by_id.values()) - len(kept),
        "missing_cost": missing_cost,
        "no_output_timestamp": no_output_timestamp,
        "late_only": late_only,
    }
    return kept, report


def recompute_cost_usd(
    records: list[dict], pricing: dict[str, dict[str, float | None]] = PRICING_PER_MILLION_TOKENS
) -> list[dict]:
    """Replaces each record's stored `cost_usd` with one computed fresh from its
    (model, input_tokens, output_tokens) against the pricing table passed in.

    Necessary, not just tidy: `cost_usd` is computed once, at log time, against
    whatever `PRICING_PER_MILLION_TOKENS` looked like *then* -- if a price was wrong,
    missing, or added later, every already-logged record keeps the old (wrong) value
    forever, since nothing re-visits it. Concretely: vector_rag_pipeline.ipynb keeps
    its own local copy of this pricing table (not imported from this module) that
    marks `gemini-3.1-flash-lite` as $0.0/$0.0 "free tier" -- it isn't; this module's
    table has the real, confirmed rate. Every record logged against that local copy
    has `cost_usd: 0.0` baked in, understating real spend. Always recompute from raw
    token counts against the current table rather than trusting a stored `cost_usd`."""
    out = []
    for r in records:
        r = dict(r)
        prices = pricing.get(r["model"])
        r["cost_usd"] = (
            (r["input_tokens"] * prices["input"] + r["output_tokens"] * prices["output"]) / 1_000_000
            if prices and prices["input"] is not None and prices["output"] is not None
            else None
        )
        out.append(r)
    return out


def clean_doc_level_costs(cost_records: list[dict], stage_name: str, keep: str = "last") -> tuple[list[dict], dict]:
    """For a stage keyed by doc_name, not financebench_id (e.g. tree/index building) --
    there's no per-question output file to join against, so this can't be as precise as
    clean_stage_costs. Keeps one record per doc_name: the most recent by timestamp
    (`keep="last"`), on the assumption a later record means a rebuild that replaced the
    earlier one (e.g. the LLM-fallback tree rebuild for flagged documents) -- not a
    certainty, so `ambiguous_docs` (docs that had >1 record) is returned for a manual
    spot-check rather than silently trusted."""
    by_doc: dict[str, list[dict]] = {}
    for r in cost_records:
        if r.get("stage") == stage_name:
            by_doc.setdefault(r.get("doc_name"), []).append(r)

    pick = max if keep == "last" else min
    kept = [pick(v, key=lambda r: r["timestamp"]) for v in by_doc.values()]
    report = {
        "n_input": sum(len(v) for v in by_doc.values()),
        "n_kept": len(kept),
        "n_dropped_as_noise": sum(len(v) for v in by_doc.values()) - len(kept),
        "ambiguous_docs": [d for d, v in by_doc.items() if len(v) > 1],
    }
    return kept, report
