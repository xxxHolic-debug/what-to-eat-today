from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .fairness import (
    cuisine_distribution,
    diversity_distance,
    rerank_for_diversity,
)
from .explainer import explain_recommendation


@dataclass(frozen=True)
class UserPreferences:
    vegetarian_mode: str = "any"
    max_spice: int = 5
    cuisines: tuple[str, ...] = ()
    max_price: int = 80
    max_calories: int = 900
    allergens: tuple[str, ...] = ()
    taste_weight: float = 0.4
    health_weight: float = 0.35
    novelty_weight: float = 0.25
    top_k: int = 5
    diversity_strength: float = 0.35


def normalize_weights(preferences: UserPreferences) -> dict[str, float]:
    raw = {
        "taste": max(0.0, preferences.taste_weight),
        "health": max(0.0, preferences.health_weight),
        "novelty": max(0.0, preferences.novelty_weight),
    }
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}


def recommend(
    dishes: pd.DataFrame,
    preferences: UserPreferences,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history = history or []
    filtered = _filter_dishes(dishes, preferences)
    if filtered.empty:
        return {
            "recommendations": [],
            "baseline": [],
            "message": "当前条件下没有可推荐的菜品。可以适当放宽价格、热量、辣度、菜系或过敏原设置。",
            "allergy_excluded_count": _count_allergy_exclusions(dishes, preferences.allergens),
        }

    weights = normalize_weights(preferences)
    recent_cuisines = [entry.get("cuisine") for entry in history[-7:] if entry.get("cuisine")]
    recommend_counts = _recommend_counts(history)
    scored = _score_dishes(filtered, dishes, weights, recent_cuisines, recommend_counts, history)

    baseline = sorted(scored, key=lambda item: item["final_score"], reverse=True)[: preferences.top_k]
    fairness_tau = _diversity_strength_to_tau(preferences.diversity_strength)
    recommendations = rerank_for_diversity(
        scored,
        preferences.top_k,
        fairness_tau,
        preferences.diversity_strength,
    )
    baseline_ids = {int(item["id"]) for item in baseline}
    for item in recommendations:
        item["fairness_adjusted"] = int(item["id"]) not in baseline_ids
        item["explanation"] = explain_recommendation(item, weights)

    baseline_distance = diversity_distance(baseline)
    final_distance = diversity_distance(recommendations)

    return {
        "recommendations": recommendations,
        "baseline": baseline,
        "weights": weights,
        "baseline_distribution": cuisine_distribution(baseline),
        "final_distribution": cuisine_distribution(recommendations),
        "baseline_distance": baseline_distance,
        "final_distance": final_distance,
        "fairness_tau": fairness_tau,
        "diversity_strength": preferences.diversity_strength,
        "message": "",
        "allergy_excluded_count": _count_allergy_exclusions(dishes, preferences.allergens),
        "selected_allergens": list(preferences.allergens),
    }


def _filter_dishes(dishes: pd.DataFrame, preferences: UserPreferences) -> pd.DataFrame:
    result = dishes.copy()
    if preferences.vegetarian_mode == "vegetarian":
        result = result[result["vegetarian"]]
    elif preferences.vegetarian_mode == "meat":
        result = result[~result["vegetarian"]]

    result = result[result["spice_level"] <= preferences.max_spice]
    result = result[result["price_yuan"] <= preferences.max_price]
    result = result[result["calories"] <= preferences.max_calories]

    if preferences.allergens:
        result = result[
            ~result["allergens"].apply(
                lambda value: bool(_parse_allergens(value) & set(preferences.allergens))
            )
        ]

    if preferences.cuisines:
        result = result[result["cuisine"].isin(preferences.cuisines)]
    return result


def _score_dishes(
    filtered: pd.DataFrame,
    all_dishes: pd.DataFrame,
    weights: dict[str, float],
    recent_cuisines: list[str],
    recommend_counts: dict[int, int],
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    history = history or []
    calorie_min, calorie_max = all_dishes["calories"].min(), all_dishes["calories"].max()
    protein_min, protein_max = all_dishes["protein_g"].min(), all_dishes["protein_g"].max()
    fat_min, fat_max = all_dishes["fat_g"].min(), all_dishes["fat_g"].max()
    rating_min, rating_max = all_dishes["rating"].min(), all_dishes["rating"].max()
    total_recommendations = max(2, sum(recommend_counts.values()) + 1)
    user_dish_ratings = _average_ratings_by_key(history, "id")
    user_cuisine_ratings = _average_ratings_by_key(history, "cuisine")

    scored: list[dict[str, Any]] = []
    for row in filtered.to_dict("records"):
        public_taste = _minmax(row["rating"], rating_min, rating_max)
        dish_taste = _star_to_score(user_dish_ratings.get(int(row["id"])))
        cuisine_taste = _star_to_score(user_cuisine_ratings.get(row["cuisine"]))
        taste = 0.65 * public_taste + 0.25 * dish_taste + 0.10 * cuisine_taste
        low_calorie = 1.0 - _minmax(row["calories"], calorie_min, calorie_max)
        protein = _minmax(row["protein_g"], protein_min, protein_max)
        low_fat = 1.0 - _minmax(row["fat_g"], fat_min, fat_max)
        health = 0.55 * low_calorie + 0.25 * protein + 0.20 * low_fat

        recent_cuisine = row["cuisine"] in recent_cuisines
        cuisine_novelty = 0.35 if recent_cuisine else 0.9
        n_i = recommend_counts.get(int(row["id"]), 0) + 1
        ucb_bonus = math.sqrt(2.0 * math.log(total_recommendations) / n_i)
        normalized_ucb = min(1.0, ucb_bonus / 2.0)
        novelty = 0.62 * cuisine_novelty + 0.38 * normalized_ucb

        trend_penalty = 0.035 if row["popular_tag"] else 0.0
        final_score = (
            weights["taste"] * taste
            + weights["health"] * health
            + weights["novelty"] * novelty
            - trend_penalty
        )

        row.update(
            {
                "taste_score": taste,
                "health_score": health,
                "novelty_score": novelty,
                "ucb_bonus": normalized_ucb,
                "recent_cuisine": recent_cuisine,
                "trend_penalty": trend_penalty,
                "final_score": final_score,
                "allergen_list": sorted(_parse_allergens(row.get("allergens", ""))),
            }
        )
        scored.append(row)
    return sorted(scored, key=lambda item: item["final_score"], reverse=True)


def _recommend_counts(history: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for entry in history:
        dish_id = entry.get("id")
        if dish_id is None:
            continue
        counts[int(dish_id)] = counts.get(int(dish_id), 0) + 1
    return counts


def _average_ratings_by_key(history: list[dict[str, Any]], key: str) -> dict[Any, float]:
    totals: dict[Any, float] = {}
    counts: dict[Any, int] = {}
    for entry in history:
        if key not in entry or "rating" not in entry:
            continue
        value = entry[key]
        if value is None:
            continue
        if key == "id":
            value = int(value)
        totals[value] = totals.get(value, 0.0) + float(entry["rating"])
        counts[value] = counts.get(value, 0) + 1
    return {key_value: totals[key_value] / counts[key_value] for key_value in totals}


def _star_to_score(stars: float | None) -> float:
    if stars is None:
        return 0.5
    return max(0.0, min(1.0, (stars - 1.0) / 4.0))


def _parse_allergens(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    allergens = {part.strip() for part in str(value).split(";") if part.strip()}
    allergens.discard("无")
    return allergens


def _count_allergy_exclusions(dishes: pd.DataFrame, allergens: tuple[str, ...]) -> int:
    if not allergens or "allergens" not in dishes.columns:
        return 0
    selected = set(allergens)
    return int(dishes["allergens"].apply(lambda value: bool(_parse_allergens(value) & selected)).sum())


def _diversity_strength_to_tau(strength: float) -> float:
    """Higher UI diversity means a lower allowed concentration distance."""
    strength = max(0.0, min(1.0, strength))
    return 0.70 - 0.60 * strength


def _minmax(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.5
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
