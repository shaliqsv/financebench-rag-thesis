"""Shared retry/backoff for the batch loops.

Every external API used in this pipeline (Gemini, Voyage, Groq) has a free
or low-tier rate limit that a 150-question unattended loop will hit
regularly (e.g. Voyage rerank's ~10K-tokens/minute cap noted in the original
notebook). A single 429 shouldn't kill a multi-hour run, so every API call
in the batch scripts goes through this decorator instead of being called
directly.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")

RATE_LIMIT_MARKERS = ("429", "rate limit", "resource_exhausted", "quota")


def _looks_like_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in RATE_LIMIT_MARKERS)


def with_retry(
    max_retries: int = 6,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential backoff, longer waits for rate-limit-shaped errors.

    Rate-limit errors back off starting at base_delay and double each retry
    (capped at max_delay); any other exception gets one short retry (network
    blip) before propagating, so real bugs still fail fast and loud instead
    of being silently retried into a 6x-slower version of the same crash.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = base_delay
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
                    last_exc = e
                    if _looks_like_rate_limit(e):
                        print(f"    rate limited ({e!s:.100s}), backing off {delay:.0f}s ...")
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                    elif attempt == 0:
                        print(f"    transient error ({e!s:.100s}), retrying once ...")
                        time.sleep(2.0)
                    else:
                        raise
            raise RuntimeError(f"{fn.__name__} failed after {max_retries} retries") from last_exc

        return wrapper

    return decorator
