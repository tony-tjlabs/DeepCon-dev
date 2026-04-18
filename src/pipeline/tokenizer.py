"""
Journey Tokenizer — 이동 데이터 의미 단위 토큰화 모듈
====================================================
1분 단위 raw Journey 데이터를 의미 있는 블록(Dwell, Transit)으로 변환.

핵심 개념:
  - Dwell Block: 동일 Locus에 3분 이상 연속 체류
  - Transit: Dwell 블록 사이의 짧은 이동 구간
  - Block Type: GATE_IN, WORK, REST, TRANSIT, GATE_OUT

사용법:
    from src.pipeline.tokenizer import tokenize_journey, add_journey_blocks

    # 단일 작업자 토큰화
    blocks = tokenize_journey(journey_df, user_no, spatial_graph)

    # 전체 DataFrame에 블록 정보 추가
    journey_df = add_journey_blocks(journey_df, spatial_graph)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.spatial.graph import SpatialGraph

logger = logging.getLogger(__name__)

# ─── 상수 ────────────────────────────────────────────────────────────────

# Dwell Block 최소 지속 시간 (분)
MIN_DWELL_DURATION = 3

# Block Type 판정용 토큰 매핑
# v1: 하드코딩된 영문 토큰
# v2: locus_name 기반 키워드 매칭 (locus_v2.csv의 locus_type/function 참조)
GATE_TOKENS = {"timeclock", "main_gate", "sub_gate"}
WORK_TOKENS = {"work_zone", "outdoor_work", "mechanical_room", "confined_space", "high_voltage", "transit"}
REST_TOKENS = {"breakroom", "smoking_area", "dining_hall", "restroom"}
ADMIN_TOKENS = {"office", "facility"}

# v2 키워드 기반 분류 (locus_token이 한글 gateway_name인 경우)
_V2_GATE_KEYWORDS = ["타각기", "출입", "입구", "출구", "정문"]
_V2_REST_KEYWORDS = ["휴게", "흡연", "식당", "화장실", "휴계"]
_V2_ADMIN_KEYWORDS = ["사무실", "관리"]
_V2_VERTICAL_KEYWORDS = ["호이스트", "클라이머", "엘리베이터"]

BlockType = Literal["GATE_IN", "GATE_OUT", "WORK", "REST", "TRANSIT", "ADMIN", "UNKNOWN"]


@dataclass
class JourneyBlock:
    """Journey 의미 블록."""

    block_type: BlockType
    locus_id: str
    locus_token: str
    start_time: datetime
    end_time: datetime
    duration_min: int
    activity_level: str = "medium"  # "high" | "medium" | "low"
    is_valid_transition: bool = True  # Adjacency 기반 유효 이동 여부
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "block_type": self.block_type,
            "locus_id": self.locus_id,
            "locus_token": self.locus_token,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_min": self.duration_min,
            "activity_level": self.activity_level,
            "is_valid_transition": self.is_valid_transition,
            **self.metadata,
        }


# ─── 블록 타입 판정 ───────────────────────────────────────────────────────


def _determine_block_type(
    locus_token: str,
    is_first: bool = False,
    is_last: bool = False,
) -> BlockType:
    """
    Locus 토큰 기반 블록 타입 판정.

    v1: 영문 토큰 집합 매칭 (work_zone, breakroom 등)
    v2: 한글 gateway_name → 키워드 기반 분류

    Args:
        locus_token: Locus의 token 값 (v1: 영문, v2: gateway_name)
        is_first: 첫 번째 블록 여부 (GATE_IN 후보)
        is_last: 마지막 블록 여부 (GATE_OUT 후보)
    """
    token_lower = (locus_token or "").lower()

    # ── v1 영문 토큰 매칭 (하위 호환) ──
    if token_lower in GATE_TOKENS:
        if is_first:
            return "GATE_IN"
        if is_last:
            return "GATE_OUT"
        return "TRANSIT"

    if token_lower in WORK_TOKENS:
        return "WORK"

    if token_lower in REST_TOKENS:
        return "REST"

    if token_lower in ADMIN_TOKENS:
        return "ADMIN"

    # ── v2 키워드 기반 분류 (한글 gateway_name) ──
    # 휴식 공간 (Gate 키워드보다 먼저 체크 — "타각기_앞_흡연실" 등 처리)
    if any(kw in token_lower for kw in _V2_REST_KEYWORDS):
        return "REST"

    # 관리/시설
    if any(kw in token_lower for kw in _V2_ADMIN_KEYWORDS):
        return "ADMIN"

    # Gate/타각기 판정 (휴식/관리 확인 후)
    if any(kw in token_lower for kw in _V2_GATE_KEYWORDS):
        if is_first:
            return "GATE_IN"
        if is_last:
            return "GATE_OUT"
        return "TRANSIT"

    # 수직 이동 (호이스트/클라이머)
    if any(kw in token_lower for kw in _V2_VERTICAL_KEYWORDS):
        return "WORK"

    # v2 토큰 중 위 키워드에 해당하지 않으면 WORK (기본값)
    # GW-XXX 형태의 locus_id가 들어올 경우 또는 한글 공간명
    if token_lower and token_lower not in ("unknown", "unmapped"):
        return "WORK"

    return "UNKNOWN"


def _determine_activity_level(avg_active_ratio: float) -> str:
    """활성도 수준 판정."""
    if avg_active_ratio >= 0.6:
        return "high"
    if avg_active_ratio >= 0.2:
        return "medium"
    return "low"


# ─── 단일 작업자 토큰화 ───────────────────────────────────────────────────


def tokenize_journey(
    journey_df: pd.DataFrame,
    user_no: str,
    spatial_graph: "SpatialGraph | None" = None,
) -> list[JourneyBlock]:
    """
    작업자의 1분 단위 Journey를 의미 블록으로 변환.

    알고리즘:
      1. 시간순 정렬
      2. 연속 동일 Locus 구간 그룹화 (Run-Length Encoding)
      3. MIN_DWELL_DURATION 이상 → Dwell Block
      4. 그 외 → Transit (이전 블록에 병합 또는 독립)
      5. Block Type 판정
      6. Adjacency 유효성 검증

    Args:
        journey_df: 전체 Journey DataFrame (user_no 필터링 전)
        user_no: 대상 작업자 ID
        spatial_graph: SpatialGraph (없으면 Adjacency 검증 생략)

    Returns:
        JourneyBlock 리스트 (시간순)
    """
    # 해당 작업자 데이터 필터링
    user_df = journey_df[journey_df["user_no"] == user_no].copy()
    if user_df.empty:
        return []

    # 시간순 정렬
    user_df = user_df.sort_values("timestamp").reset_index(drop=True)

    # 연속 동일 Locus 구간 식별
    user_df["_locus_changed"] = user_df["locus_id"].ne(user_df["locus_id"].shift())
    user_df["_run_id"] = user_df["_locus_changed"].cumsum()

    # Run별 집계
    run_agg = user_df.groupby("_run_id").agg(
        locus_id=("locus_id", "first"),
        locus_token=("locus_token", "first"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        duration_min=("_run_id", "count"),
        avg_active_ratio=("active_ratio", "mean"),
    ).reset_index()

    # 블록 생성
    blocks: list[JourneyBlock] = []
    n_runs = len(run_agg)

    for i, row in run_agg.iterrows():
        is_first = (i == 0)
        is_last = (i == n_runs - 1)

        block_type = _determine_block_type(
            row["locus_token"],
            is_first=is_first,
            is_last=is_last,
        )

        # 짧은 구간(MIN_DWELL_DURATION 미만)은 TRANSIT으로 처리
        if row["duration_min"] < MIN_DWELL_DURATION and block_type not in ("GATE_IN", "GATE_OUT"):
            block_type = "TRANSIT"

        activity_level = _determine_activity_level(row["avg_active_ratio"])

        # Adjacency 유효성 검증
        is_valid = True
        if spatial_graph and len(blocks) > 0:
            prev_locus = blocks[-1].locus_id
            curr_locus = row["locus_id"]
            if prev_locus != curr_locus:
                is_valid = spatial_graph.is_adjacent(prev_locus, curr_locus)

        block = JourneyBlock(
            block_type=block_type,
            locus_id=row["locus_id"],
            locus_token=row["locus_token"] or "unknown",
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration_min=int(row["duration_min"]),
            activity_level=activity_level,
            is_valid_transition=is_valid,
        )
        blocks.append(block)

    return blocks


# ─── 전체 DataFrame 토큰화 ────────────────────────────────────────────────


def add_journey_blocks(
    journey_df: pd.DataFrame,
    spatial_graph: "SpatialGraph | None" = None,
) -> pd.DataFrame:
    """
    전체 Journey DataFrame에 블록 정보 컬럼 추가.

    추가 컬럼:
      - block_id: 연속 체류 블록 ID
      - block_type: GATE_IN/WORK/REST/TRANSIT/GATE_OUT
      - block_duration_min: 블록 지속 시간
      - is_valid_transition: Adjacency 기반 유효 이동 여부

    Note:
      벡터화 연산으로 구현 (for 루프 최소화)
    """
    if journey_df.empty:
        return journey_df

    df = journey_df.copy()

    # 필수 컬럼 확인
    required = {"user_no", "timestamp", "locus_id", "locus_token"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(f"필수 컬럼 누락: {missing}")
        return df

    # 정렬
    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # 블록 ID 생성 (user_no 경계 또는 locus_id 변경 시 새 블록)
    user_changed = df["user_no"].ne(df["user_no"].shift())
    locus_changed = df["locus_id"].ne(df["locus_id"].shift())
    df["block_id"] = (user_changed | locus_changed).cumsum()

    # 블록별 집계
    block_info = df.groupby("block_id").agg(
        user_no=("user_no", "first"),
        locus_id=("locus_id", "first"),
        locus_token=("locus_token", "first"),
        block_duration_min=("block_id", "count"),
        block_start_idx=("block_id", lambda x: x.index.min()),
    ).reset_index()

    # 블록 타입 판정 (작업자별 첫/마지막 블록 고려)
    # 작업자별 첫/마지막 블록 ID
    first_blocks = df.groupby("user_no")["block_id"].min().values
    last_blocks = df.groupby("user_no")["block_id"].max().values
    first_set = set(first_blocks)
    last_set = set(last_blocks)

    def _get_block_type(row):
        is_first = row["block_id"] in first_set
        is_last = row["block_id"] in last_set
        duration = row["block_duration_min"]
        token = row["locus_token"]

        block_type = _determine_block_type(token, is_first, is_last)

        # 짧은 구간 처리
        if duration < MIN_DWELL_DURATION and block_type not in ("GATE_IN", "GATE_OUT"):
            block_type = "TRANSIT"

        return block_type

    block_info["block_type"] = block_info.apply(_get_block_type, axis=1)

    # Adjacency 유효성 검증
    if spatial_graph:
        block_info["prev_locus"] = block_info["locus_id"].shift()
        block_info["prev_user"] = block_info["user_no"].shift()

        def _check_valid(row):
            if pd.isna(row["prev_locus"]):
                return True
            if row["user_no"] != row["prev_user"]:
                return True  # 작업자 경계
            if row["locus_id"] == row["prev_locus"]:
                return True
            return spatial_graph.is_adjacent(row["prev_locus"], row["locus_id"])

        block_info["is_valid_transition"] = block_info.apply(_check_valid, axis=1)
        block_info.drop(columns=["prev_locus", "prev_user"], inplace=True)
    else:
        block_info["is_valid_transition"] = True

    # 원본 DataFrame에 병합
    merge_cols = ["block_id", "block_type", "block_duration_min", "is_valid_transition"]
    df = df.merge(
        block_info[merge_cols],
        on="block_id",
        how="left",
    )

    return df


# ─── Journey 요약 ─────────────────────────────────────────────────────────


def summarize_journey(blocks: list[JourneyBlock]) -> dict:
    """
    Journey를 핵심 3-5개 구간으로 요약.

    Returns:
        {
            "total_duration_min": int,
            "work_duration_min": int,
            "rest_duration_min": int,
            "transit_duration_min": int,
            "main_work_location": str,  # 가장 오래 머문 작업 공간
            "invalid_transitions": int,  # 비정상 이동 수
            "summary_text": str,  # 한 줄 요약
        }
    """
    if not blocks:
        return {
            "total_duration_min": 0,
            "work_duration_min": 0,
            "rest_duration_min": 0,
            "transit_duration_min": 0,
            "main_work_location": "",
            "invalid_transitions": 0,
            "summary_text": "데이터 없음",
        }

    total_min = sum(b.duration_min for b in blocks)
    work_min = sum(b.duration_min for b in blocks if b.block_type == "WORK")
    rest_min = sum(b.duration_min for b in blocks if b.block_type == "REST")
    transit_min = sum(b.duration_min for b in blocks if b.block_type == "TRANSIT")
    invalid_count = sum(1 for b in blocks if not b.is_valid_transition)

    # 가장 오래 머문 작업 공간
    work_blocks = [b for b in blocks if b.block_type == "WORK"]
    main_work = ""
    if work_blocks:
        main_work = max(work_blocks, key=lambda b: b.duration_min).locus_id

    # 요약 텍스트
    summary_parts = []
    if work_min > 0:
        summary_parts.append(f"작업 {work_min}분")
    if rest_min > 0:
        summary_parts.append(f"휴식 {rest_min}분")
    if transit_min > 0:
        summary_parts.append(f"이동 {transit_min}분")
    summary_text = ", ".join(summary_parts) if summary_parts else "활동 없음"

    return {
        "total_duration_min": total_min,
        "work_duration_min": work_min,
        "rest_duration_min": rest_min,
        "transit_duration_min": transit_min,
        "main_work_location": main_work,
        "invalid_transitions": invalid_count,
        "summary_text": summary_text,
    }


def journey_to_tokens(blocks: list[JourneyBlock]) -> list[str]:
    """
    ML 학습용 토큰 시퀀스 생성.

    형식: ["GATE_IN", "TRANSIT_3", "WORK_FAB1F_120", "REST_15", ...]
    """
    tokens = []
    for b in blocks:
        if b.block_type in ("GATE_IN", "GATE_OUT"):
            tokens.append(b.block_type)
        elif b.block_type == "TRANSIT":
            tokens.append(f"TRANSIT_{b.duration_min}")
        else:
            # WORK, REST, ADMIN 등
            # v1: "L-Y1-020" → "020", v2: "GW-351" → "351"
            locus_short = b.locus_id
            for prefix in ("L-Y1-", "GW-"):
                locus_short = locus_short.replace(prefix, "")
            tokens.append(f"{b.block_type}_{locus_short}_{b.duration_min}")
    return tokens


# ─── 통계 ─────────────────────────────────────────────────────────────────


def get_tokenization_stats(journey_df: pd.DataFrame) -> dict:
    """토큰화 결과 통계."""
    if "block_type" not in journey_df.columns:
        return {"error": "block_type 컬럼 없음 (토큰화 미적용)"}

    stats = {
        "total_records": len(journey_df),
        "block_type_distribution": journey_df["block_type"].value_counts().to_dict(),
        "invalid_transition_count": int((~journey_df["is_valid_transition"]).sum())
        if "is_valid_transition" in journey_df.columns
        else 0,
    }

    if "block_duration_min" in journey_df.columns:
        # 블록별 통계 (중복 제거)
        block_stats = journey_df.drop_duplicates("block_id").groupby("block_type")["block_duration_min"]
        stats["avg_block_duration"] = block_stats.mean().to_dict()

    return stats
