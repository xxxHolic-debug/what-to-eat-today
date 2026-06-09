from __future__ import annotations


def explain_recommendation(item: dict, weights: dict[str, float]) -> str:
    reasons: list[str] = []

    strongest_value = max(weights, key=weights.get)
    value_labels = {
        "taste": "美味",
        "health": "健康",
        "novelty": "新奇",
    }
    reasons.append(f"它比较符合你当前更重视“{value_labels[strongest_value]}”的设置")

    if item.get("fairness_adjusted"):
        reasons.append("它能改善推荐列表中的菜系多样性")
    if item.get("recent_cuisine"):
        reasons.append("它虽然和最近吃过的菜系相近，但综合得分仍然较高")
    else:
        reasons.append(f"它增加了一个近期较少出现的“{item['cuisine']}”选择")
    if item.get("popular_tag"):
        reasons.append("它带有网红标签，系统已把这种流行度影响明确展示出来")

    score = item.get("final_score", 0.0)
    return f"推荐“{item['name']}”，因为{'；'.join(reasons)}。最终综合分：{score:.3f}。"


def objective_function_text(
    weights: dict[str, float],
    fairness_tau: float,
    diversity_strength: float | None = None,
) -> str:
    diversity_note = ""
    if diversity_strength is not None:
        diversity_note = f"当前多样性强度为 {diversity_strength:.2f}，"
    return (
        "S = "
        f"{weights['taste']:.2f} * 美味得分 + "
        f"{weights['health']:.2f} * 健康得分 + "
        f"{weights['novelty']:.2f} * 新奇得分 + UCB探索奖励 - 网红轻微惩罚；"
        f"随后进行菜系多样性重排，{diversity_note}内部总变差距离阈值为 {fairness_tau:.2f}。"
    )
