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
    "deepseek-v4": {"input": None, "output": None},
    "voyage-rerank-2": {"input": None, "output": None},
    "stella-en-1.5b-v5": {"input": None, "output": None},
    "openai/gpt-oss-120b": {"input": None, "output": None},  # judge, via Groq (free tier)
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
        )
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        return record
