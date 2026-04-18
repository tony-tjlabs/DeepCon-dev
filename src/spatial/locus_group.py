"""
Locus Group — 계층적 공간 그룹 매핑
====================================
213개 개별 locus를 건물_층 그룹으로 매핑하여
계층적 예측/분석을 지원한다.

핵심 아이디어:
  - 개별 locus 예측 (213개, Top-1 ~27%) 는 현장에서 의미 제한적
  - 건물_층 그룹 예측 (~30개, Top-1 60-70%+) 이 현장 의사결정에 실질적
  - "FAB 5층에 있을 것이다"가 "GW-123에 있을 것이다"보다 유용
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

import pandas as pd
import streamlit as st

from core.cache.policy import SPATIAL_MODEL

logger = logging.getLogger(__name__)

# ─── 건물 추론 규칙 (locus_name에서 building 미지정 시) ─────
_BUILDING_PATTERNS = [
    (r"fab", "FAB"),
    (r"cub", "CUB"),
    (r"wwt", "WWT"),
    (r"154kv|154kV", "154kV"),
    (r"본진", "본진"),
    (r"bl_4|bl4", "bl_4"),
    (r"bl_3|bl3|3bl", "지원_3BL"),
    (r"bl_2|bl2|2bl", "지원_2BL"),
    (r"bl_1|bl1|1bl", "지원_1BL"),
    (r"bl_|bl", "bl"),
    (r"지원", "지원"),
    (r"안전교육|6gate|gate", "야외"),
]


def _infer_building(locus_name: str) -> str:
    """locus_name에서 건물명 추론."""
    name_lower = locus_name.lower()
    for pattern, building in _BUILDING_PATTERNS:
        if re.search(pattern, name_lower):
            return building
    return "기타"


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner=False)
def build_locus_group_map(sector_id: str) -> dict[str, str]:
    """
    locus_id → 건물_층 그룹 매핑 생성.

    Returns:
        {"GW-351": "FAB_1F", "GW-236": "본진_GL", ...}
    """
    from config import SPATIAL_DIR

    locus_path = SPATIAL_DIR / sector_id / "locus" / "locus_v2.csv"
    if not locus_path.exists():
        logger.warning(f"locus_v2.csv 없음: {locus_path}")
        return {}

    df = pd.read_csv(locus_path)
    group_map = {}

    for _, row in df.iterrows():
        lid = str(row.get("locus_id", ""))
        building = row.get("building", "")
        floor = row.get("floor", "")
        name = str(row.get("locus_name", ""))

        # 건물 미지정 → 이름에서 추론
        if pd.isna(building) or building == "":
            building = _infer_building(name)

        # 층 미지정 → GL (Ground Level)
        if pd.isna(floor) or floor == "":
            floor = "GL"

        group_map[lid] = f"{building}_{floor}"

    logger.info(f"Locus group map: {len(group_map)} loci → {len(set(group_map.values()))} groups")
    return group_map


def get_group(locus_id: str, group_map: dict[str, str]) -> str:
    """locus_id의 그룹 반환. 매핑 없으면 'UNKNOWN'."""
    return group_map.get(str(locus_id), "UNKNOWN")


def group_predictions(
    predictions: dict[str, list[tuple[str, float]]],
    current_loci: dict[str, str],
    group_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    예측/현재 위치를 그룹 단위로 변환.

    Returns:
        (current_groups, predicted_groups):
            {user_no: group_name}
    """
    current_groups = {}
    predicted_groups = {}

    for user_no, cur_locus in current_loci.items():
        current_groups[user_no] = get_group(cur_locus, group_map)

    for user_no, preds in predictions.items():
        if preds:
            predicted_groups[user_no] = get_group(preds[0][0], group_map)
        elif user_no in current_loci:
            predicted_groups[user_no] = current_groups.get(user_no, "UNKNOWN")

    return current_groups, predicted_groups


def evaluate_group_accuracy(
    predictions: dict[str, dict],
    actual_sequences: dict[str, list[str]],
    group_map: dict[str, str],
) -> dict:
    """
    그룹 단위 예측 정확도 평가.

    Args:
        predictions: Journal 형식 {user_no: {"predicted": ..., "top3": [...]}}
        actual_sequences: {user_no: [locus1, locus2, ...]}
        group_map: locus_id → group 매핑

    Returns:
        {"top1": accuracy, "top3": accuracy, "n": count}
    """
    total = 0
    top1_correct = 0
    top3_correct = 0

    for user_no, pred_info in predictions.items():
        actual_seq = actual_sequences.get(user_no, [])
        if not actual_seq:
            continue

        actual_group = get_group(actual_seq[0], group_map)
        total += 1

        # Top-1 그룹 정확도
        pred_group = get_group(pred_info["predicted"], group_map)
        if pred_group == actual_group:
            top1_correct += 1

        # Top-3 그룹 정확도
        top3_loci = [loc for loc, _ in pred_info.get("top3", [(pred_info["predicted"], 0)])]
        top3_groups = {get_group(loc, group_map) for loc in top3_loci}
        if actual_group in top3_groups:
            top3_correct += 1

    if total == 0:
        return {"top1": 0.0, "top3": 0.0, "n": 0}

    return {
        "top1": round(top1_correct / total, 4),
        "top3": round(top3_correct / total, 4),
        "n": total,
    }


def get_all_groups(sector_id: str) -> list[str]:
    """등록된 모든 그룹 목록 (정렬)."""
    group_map = build_locus_group_map(sector_id)
    return sorted(set(group_map.values()))
