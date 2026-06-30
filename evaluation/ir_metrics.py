"""
Standard information-retrieval ranking metrics, computed over a ranked list
of binary relevance judgments (1 = relevant, 0 = not relevant), in the
order chunks were actually returned to the LLM (i.e. AFTER reranking, since
that's the order that matters for what the generator actually sees first).

These are intentionally plain, dependency-free functions so they can be
unit-tested in isolation from the rest of the pipeline (no embedding model
or API key required to verify the math is correct).

Definitions used here (standard IR formulas):
  - Reciprocal Rank = 1 / (rank of the first relevant item), 0 if none found
  - Average Precision = mean of Precision@i computed at each rank i where
    a relevant item appears
  - DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)   [graded form;
    reduces to a simple binary form when rel_i is 0/1]
  - NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal
    (best-possible) ordering of the same relevance values
"""

from __future__ import annotations
import math


def reciprocal_rank(relevances: list[int]) -> float:
    """1/rank of the first relevant (1) item; 0.0 if none are relevant."""
    for i, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def average_precision(relevances: list[int]) -> float:
    """Mean of precision@i, computed only at ranks where a relevant item
    appears. Rewards finding ALL relevant items, not just the first."""
    hits = 0
    precisions_at_hits = []
    for i, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            precisions_at_hits.append(hits / i)
    return sum(precisions_at_hits) / hits if hits > 0 else 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(i + 1)
        for i, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """DCG@k normalized by the IDEAL DCG@k (same relevance values, sorted
    best-first) - so a perfect ordering of whatever relevant items exist
    always scores 1.0, regardless of how many relevant items there are."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_order = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_reciprocal_rank(list_of_relevance_lists: list[list[int]]) -> float:
    """MRR averaged across multiple queries."""
    scores = [reciprocal_rank(r) for r in list_of_relevance_lists]
    return sum(scores) / len(scores) if scores else 0.0


def mean_average_precision(list_of_relevance_lists: list[list[int]]) -> float:
    """MAP averaged across multiple queries."""
    scores = [average_precision(r) for r in list_of_relevance_lists]
    return sum(scores) / len(scores) if scores else 0.0


def mean_ndcg_at_k(list_of_relevance_lists: list[list[int]], k: int) -> float:
    """NDCG@k averaged across multiple queries."""
    scores = [ndcg_at_k(r, k) for r in list_of_relevance_lists]
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    # Sanity checks against hand-calculable cases (mirrors the worked
    # examples in rag_interview_prep.md Q76, Q77, Q80, so the numbers here
    # should match those by construction).

    # Q76: relevant at positions 1, 3, 5 out of 5 -> MAP@5 should be ~0.756
    case_q76 = [1, 0, 1, 0, 1]
    print(f"Q76 check - AP: {average_precision(case_q76):.3f} (expected ~0.756)")

    # Q77: all relevant chunks at the very bottom -> MRR, MAP, NDCG all low
    case_q77 = [0, 0, 0, 0, 1]
    print(f"Q77 check - RR: {reciprocal_rank(case_q77):.3f} (expected 0.200)")
    print(f"Q77 check - AP: {average_precision(case_q77):.3f} (expected 0.200)")
    print(f"Q77 check - NDCG@5: {ndcg_at_k(case_q77, 5):.3f} (expected low, <0.5)")

    # Q80: all relevant retrieved but in REVERSE (worst-first) order
    # vs. the same relevant items in ideal (best-first) order
    case_reverse = [0, 0, 1, 1, 1]   # relevant items pushed to the bottom
    case_ideal = [1, 1, 1, 0, 0]     # same 3 relevant items, ranked first
    print(f"Q80 check - NDCG@5 reverse-order: {ndcg_at_k(case_reverse, 5):.3f}")
    print(f"Q80 check - NDCG@5 ideal-order:   {ndcg_at_k(case_ideal, 5):.3f} (expected 1.000)")
