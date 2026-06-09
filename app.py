from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.explainer import objective_function_text
from src.privacy import differentially_private_counts
from src.recommender import UserPreferences, recommend


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "dishes.csv"
STATE_PATH = ROOT / ".eatright_state.json"


@st.cache_data
def load_dishes(data_version: float) -> pd.DataFrame:
    _ = data_version
    dishes = pd.read_csv(DATA_PATH)
    bool_map = {"true": True, "false": False, True: True, False: False}
    dishes["vegetarian"] = dishes["vegetarian"].map(bool_map).astype(bool)
    dishes["popular_tag"] = dishes["popular_tag"].map(bool_map).astype(bool)
    return dishes


def allergen_options(dishes: pd.DataFrame) -> list[str]:
    allergens: set[str] = set()
    for value in dishes["allergens"].fillna("无"):
        for part in str(value).split(";"):
            allergen = part.strip()
            if allergen and allergen != "无":
                allergens.add(allergen)
    return sorted(allergens)


def display_dishes(dishes: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "id": "编号",
        "name": "菜品",
        "cuisine": "菜系",
        "vegetarian": "是否素食",
        "spice_level": "辣度",
        "calories": "热量(kcal)",
        "protein_g": "蛋白质(g)",
        "fat_g": "脂肪(g)",
        "rating": "大众评分",
        "popular_tag": "网红标签",
        "price_yuan": "价格(元)",
        "prep_minutes": "准备时间(分钟)",
        "allergens": "可能过敏原",
    }
    result = dishes.rename(columns=columns).copy()
    if "是否素食" in result:
        result["是否素食"] = result["是否素食"].map({True: "是", False: "否"})
    if "网红标签" in result:
        result["网红标签"] = result["网红标签"].map({True: "是", False: "否"})
    return result


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"history": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"history": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_history_item(item: dict[str, Any], rating: int) -> None:
    state = load_state()
    state.setdefault("history", []).append(
        {
            "id": int(item["id"]),
            "name": item["name"],
            "cuisine": item["cuisine"],
            "rating": int(rating),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_state(state)


def sidebar_preferences(dishes: pd.DataFrame) -> UserPreferences:
    st.sidebar.header("价值观控制器")
    taste_weight = st.sidebar.slider(
        "美味权重（可调）",
        0.0,
        1.0,
        0.40,
        0.05,
        help="数值越高，系统越重视你的星级反馈和大众口味评分。",
    )
    health_weight = st.sidebar.slider("健康权重", 0.0, 1.0, 0.35, 0.05)
    novelty_weight = st.sidebar.slider("新奇权重", 0.0, 1.0, 0.25, 0.05)

    st.sidebar.header("饮食约束")
    vegetarian_mode = st.sidebar.segmented_control(
        "荤素偏好",
        options=["any", "vegetarian", "meat"],
        format_func=lambda value: {"any": "不限", "vegetarian": "素食", "meat": "荤食"}[value],
        default="any",
    )
    max_spice = st.sidebar.slider("最高辣度", 0, 5, 5)
    max_price = st.sidebar.slider("最高价格（元）", 10, 80, 45)
    max_calories = st.sidebar.slider("最高热量（kcal）", 150, 900, 650, 25)
    cuisines = st.sidebar.multiselect(
        "允许的菜系",
        options=sorted(dishes["cuisine"].unique()),
        default=[],
        help="留空表示不限制菜系。",
    )
    allergens = st.sidebar.multiselect(
        "过敏提醒",
        options=allergen_options(dishes),
        default=[],
        help="选择后，含有这些过敏原的菜品会被排除。这里只做温和提醒，不能替代专业医疗建议。",
    )
    if allergens:
        st.sidebar.warning("已为你避开所选过敏原。外出用餐时仍建议向商家确认配料。")

    st.sidebar.header("推荐设置")
    top_k = st.sidebar.slider("推荐数量", 3, 8, 5)
    diversity_strength = st.sidebar.slider(
        "多样性(越低，同类别中菜品越集中)",
        0.10,
        0.60,
        0.35,
        0.05,
        help="数值越低，越允许同一菜系集中出现；数值越高，越鼓励推荐列表包含不同菜系。",
    )

    return UserPreferences(
        vegetarian_mode=vegetarian_mode,
        max_spice=max_spice,
        cuisines=tuple(cuisines),
        max_price=max_price,
        max_calories=max_calories,
        allergens=tuple(allergens),
        taste_weight=taste_weight,
        health_weight=health_weight,
        novelty_weight=novelty_weight,
        top_k=top_k,
        diversity_strength=diversity_strength,
    )


def recommendation_card(item: dict[str, Any], index: int) -> None:
    with st.container(border=True):
        cols = st.columns([3, 1, 1, 1])
        cols[0].subheader(f"{index}. {item['name']}")
        cols[1].metric("综合分", f"{item['final_score']:.3f}")
        cols[2].metric("热量", f"{int(item['calories'])} kcal")
        cols[3].metric("价格", f"{int(item['price_yuan'])} 元")

        st.caption(
            f"菜系：{item['cuisine']} | 辣度：{int(item['spice_level'])}/5 | "
            f"蛋白质：{item['protein_g']}g | 网红标签：{'是' if item['popular_tag'] else '否'} | "
            f"可能过敏原：{item.get('allergens', '无')}"
        )
        st.write(item["explanation"])

        score_cols = st.columns([2, 1])
        rating_label = score_cols[0].select_slider(
            "用餐后美味评分",
            options=["1星", "2星", "3星", "4星", "5星"],
            value="4星",
            key=f"rating_{item['id']}_{index}",
            help="你的 1-5 星评分会影响下一次推荐中的“美味得分”。",
        )
        if score_cols[1].button("吃这个", key=f"eat_{item['id']}_{index}", use_container_width=True):
            rating = int(rating_label[0])
            add_history_item(item, rating)
            st.success("已记录到本地历史。下一次推荐会参考你的美味星级、新奇度和探索项。")
            st.rerun()


def transparency_panel(result: dict[str, Any], preferences: UserPreferences) -> None:
    st.subheader("透明度面板")
    st.code(
        objective_function_text(
            result["weights"],
            result["fairness_tau"],
            preferences.diversity_strength,
        ),
        language="text",
    )

    cols = st.columns(2)
    cols[0].metric("未加公平约束的总变差距离", f"{result['baseline_distance']:.3f}")
    cols[1].metric("最终列表的总变差距离", f"{result['final_distance']:.3f}")

    distribution = pd.DataFrame(
        {
            "未加多样性约束": result["baseline_distribution"],
            "最终推荐": result["final_distribution"],
        }
    ).fillna(0.0)
    if not distribution.empty:
        st.bar_chart(distribution)

    st.write("未加公平约束的前列结果")
    baseline_rows = [
        {
            "菜品": item["name"],
            "菜系": item["cuisine"],
            "综合分": round(item["final_score"], 3),
        }
        for item in result["baseline"]
    ]
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)


def feedback_panel(dishes: pd.DataFrame, history: list[dict[str, Any]]) -> None:
    st.subheader("本地反馈与差分隐私演示")
    st.info("原型只在本地保存选择过的菜品、菜系、评分和时间，不保存疾病、禁忌等敏感画像。")

    if not history:
        st.write("暂无历史反馈。点击推荐卡片中的“吃这个”后，这里会显示聚合结果。")
        return

    history_df = pd.DataFrame(history)
    name_lookup = dishes.set_index("id")["name"].to_dict()
    cuisine_lookup = dishes.set_index("id")["cuisine"].to_dict()
    if "id" in history_df:
        history_df["name"] = history_df["id"].map(name_lookup).fillna(history_df.get("name", ""))
        history_df["cuisine"] = history_df["id"].map(cuisine_lookup).fillna(history_df.get("cuisine", ""))
    history_df = history_df.rename(
        columns={
            "id": "编号",
            "name": "菜品",
            "cuisine": "菜系",
            "rating": "美味评分",
            "timestamp": "记录时间",
        }
    )
    st.dataframe(history_df.tail(10), use_container_width=True, hide_index=True)

    epsilon = st.slider("差分隐私 epsilon", 0.2, 3.0, 1.0, 0.1)
    noisy_counts = differentially_private_counts([int(item["id"]) for item in history], epsilon=epsilon)
    noisy_df = pd.DataFrame(
        [
            {
                "菜品": name_lookup.get(dish_id, dish_id),
                "加入噪声后的次数": round(noisy_count, 2),
            }
            for dish_id, noisy_count in noisy_counts.items()
        ]
    ).sort_values("noisy_count", ascending=False)
    st.dataframe(noisy_df, use_container_width=True, hide_index=True)

    if st.button("清空本地历史"):
        save_state({"history": []})
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="EatRight 餐食推荐助手", layout="wide")
    st.title("EatRight: 价值对齐的餐食推荐助手")
    st.caption("一个展示用户自主、推荐透明、饮食多样性和隐私保护的 AI 伦理课程项目原型。")

    dishes = load_dishes(DATA_PATH.stat().st_mtime)
    preferences = sidebar_preferences(dishes)
    state = load_state()
    history = state.get("history", [])
    result = recommend(dishes, preferences, history)

    if preferences.allergens:
        st.warning(
            "过敏提醒：已根据你的选择排除可能含有 "
            f"{'、'.join(preferences.allergens)} 的菜品。"
            "该功能仅作友好提醒，实际用餐前仍请确认配料。"
        )
        excluded_count = result.get("allergy_excluded_count", 0)
        if excluded_count:
            st.caption(f"本次共有 {excluded_count} 道菜因过敏提醒被排除。")

    if result["message"]:
        st.warning(result["message"])
        return

    tabs = st.tabs(["今日推荐", "透明度", "反馈统计", "菜品数据"])
    with tabs[0]:
        st.subheader("今日推荐")
        for index, item in enumerate(result["recommendations"], start=1):
            recommendation_card(item, index)

    with tabs[1]:
        transparency_panel(result, preferences)

    with tabs[2]:
        feedback_panel(dishes, history)

    with tabs[3]:
        st.dataframe(display_dishes(dishes), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
