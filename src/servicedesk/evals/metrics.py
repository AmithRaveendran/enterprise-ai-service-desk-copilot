"""The arithmetic of the evaluation: small pure functions, tested on their own.

Retrieval is scored by the rank of the first relevant result: the first hit whose document
is one of the case's expected documents (rank 1 = the top result, None = not in the top k).
- hit@k: the share of questions with a relevant result in the top k.
- MRR (mean reciprocal rank): the average of 1/rank (0 when nothing relevant was found).
  1.0 means the relevant document is always first; 0.5 means on average second.
"""

import math
from collections import Counter
from collections.abc import Iterable


def first_relevant_rank(doc_ids: list[str], expected: Iterable[str]) -> int | None:
    """1-based position of the first result from an expected document, or None."""
    wanted = set(expected)
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in wanted:
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float | None:
    return rate(rank is not None and rank <= k for rank in ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float | None:
    return mean(1 / rank if rank else 0.0 for rank in ranks)


def rate(flags: Iterable[bool]) -> float | None:
    """The share of True values; None when there are none to count (shown as "n/a":
    "0% of nothing" would look like a failure)."""
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def distribution(labels: Iterable[str]) -> dict[str, int]:
    """How many times each label occurs, most common first."""
    return dict(Counter(labels).most_common())


def percentile(values: Iterable[float], p: float) -> float | None:
    """The p-th percentile (0-100) by the nearest-rank method, e.g. p=95 for "p95"."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(p / 100 * len(ordered)))  # the smallest value with p% at or below
    return ordered[rank - 1]
