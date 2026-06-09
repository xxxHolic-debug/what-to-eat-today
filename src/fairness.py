from __future__ import annotations

from collections import Counter
from typing import Iterable


def cuisine_distribution(items: Iterable[dict]) -> dict[str, float]:
    items = list(items)
    if not items:
        return {}
    counts = Counter(item["cuisine"] for item in items)
    total = len(items)
    return {cuisine: count / total for cuisine, count in sorted(counts.items())}


def total_variation_distance(
    observed: dict[str, float], expected: dict[str, float]
) -> float:
    cuisines = set(observed) | set(expected)
    return 0.5 * sum(abs(observed.get(cuisine, 0.0) - expected.get(cuisine, 0.0)) for cuisine in cuisines)


def diversity_distance(items: Iterable[dict]) -> float:
    """Cuisine concentration distance inside the current recommendation list."""
    items = list(items)
    observed = cuisine_distribution(items)
    expected = uniform_expected_distribution(observed.keys())
    return total_variation_distance(observed, expected)


def uniform_expected_distribution(cuisines: Iterable[str]) -> dict[str, float]:
    unique = sorted(set(cuisines))
    if not unique:
        return {}
    share = 1.0 / len(unique)
    return {cuisine: share for cuisine in unique}


def rerank_for_diversity(
    scored_items: list[dict],
    k: int,
    tau: float = 0.35,
    strength: float = 0.50,
) -> list[dict]:
    """Greedy reranking that keeps high scores while reducing cuisine concentration."""
    if k <= 0 or not scored_items:
        return []

    pool = [dict(item) for item in scored_items]
    selected: list[dict] = []
    while pool and len(selected) < k:
        best_choice = None
        best_objective = None

        for item in pool:
            trial = selected + [item]
            distance = diversity_distance(trial)
            objective = item["final_score"] - 0.30 * strength * distance
            if best_objective is None or objective > best_objective:
                best_choice = item
                best_objective = objective

        assert best_choice is not None
        best_choice["fairness_distance_after_pick"] = diversity_distance(selected + [best_choice])
        selected.append(best_choice)
        pool.remove(best_choice)

    distance = diversity_distance(selected)
    if distance > tau and pool:
        selected = _repair_distribution(selected, pool, tau)

    return selected


def _repair_distribution(
    selected: list[dict], pool: list[dict], tau: float
) -> list[dict]:
    current = selected[:]
    original_score = {item["id"]: item["final_score"] for item in selected + pool}

    for _ in range(len(current)):
        observed = cuisine_distribution(current)
        distance = diversity_distance(current)
        if distance <= tau:
            break

        expected = uniform_expected_distribution(observed.keys())
        overrepresented = max(observed, key=lambda cuisine: observed[cuisine] - expected.get(cuisine, 0.0))
        pool_cuisines = sorted({item["cuisine"] for item in pool} - set(observed))
        underrepresented = pool_cuisines[0] if pool_cuisines else min(
            expected, key=lambda cuisine: observed.get(cuisine, 0.0) - expected[cuisine]
        )

        replacement_candidates = [item for item in pool if item["cuisine"] == underrepresented]
        removable_candidates = [item for item in current if item["cuisine"] == overrepresented]
        if not replacement_candidates or not removable_candidates:
            break

        replacement = max(replacement_candidates, key=lambda item: item["final_score"])
        removable = min(removable_candidates, key=lambda item: item["final_score"])

        current.remove(removable)
        replacement["fairness_adjusted"] = True
        current.append(replacement)
        pool.remove(replacement)
        pool.append(removable)

    return sorted(current, key=lambda item: original_score.get(item["id"], item["final_score"]), reverse=True)
