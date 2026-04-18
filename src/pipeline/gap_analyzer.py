"""
Gap Analyzer — T-Ward 음영 구간 탐지 · 분류 · 채우기 + 활성 레벨 분류
=======================================================================

건설현장 BLE 환경 특성상 T-Ward 작업자의 journey에는 음영 구간(GAP)이 빈번하다.
이 모듈은 GAP을 탐지·분류하고 합리적인 추정값으로 채워 transit 분석의
정확도를 높인다.

[v3 공간 그래프 보간] 2026-04-16
  - _assign_locus_sequence: 단순 50/50 split → Dijkstra 경로 보간 우선
  - Dijkstra 실패 시 Word2Vec co-occurrence fallback
  - sector_id를 fill_gaps / analyze_gaps에 전달하면 자동 활성화
  - SpatialGraph / W2V 모두 lru_cache 싱글턴 — 호출당 재로드 없음

[v2 최적화] 2026-04-11
  - detect_gaps: Python 이중 for 루프 → pandas shift 벡터화 (10~20x 속도 개선)
  - fill_gaps: gap 별 5M row 전체 스캔 → user_no 기준 사전 그룹화 (O(1) 조회)
  - _generate_fill_records: 작업자 base 정보를 dict 캐시로 처리

주요 함수:
  detect_gaps(journey_df)                   → gap_df (GAP 목록 + 분류)
  fill_gaps(journey_df, gap_df, sector_id)  → gap-filled journey_df
  classify_activity(journey_df)             → activity_level 컬럼 추가
  analyze_gaps(journey_df, sector_id)       → 위 3단계 통합 실행 + stats 반환
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 임계값 상수 ───────────────────────────────────────────────────────────────
GAP_MIN_THRESHOLD       = 1.5   # 이 분(min) 초과 = GAP으로 판정
GAP_SHORT_MAX           = 5.0   # 짧은 GAP 상한 (분)
GAP_MEDIUM_MAX          = 20.0  # 중간 GAP 상한 (분)
GAP_LONG_MAX            = 60.0  # 긴 GAP 상한 (분)  — 초과 시 미채움
LOW_CONFIDENCE_SIGNAL   = 2     # signal_count ≤ 이 값이면 low_confidence

# 활성 레벨 임계값
AR_HIGH_ACTIVE          = 0.75
AR_ACTIVE               = 0.50
AR_INACTIVE             = 0.20
SC_HIGH_ACTIVE_MIN      = 3
SC_ACTIVE_MIN           = 2

# GAP 유형 레이블
GAP_TYPE_SHORT_SAME     = "shadow_same_short"
GAP_TYPE_SHORT_DIFF     = "shadow_diff_short"
GAP_TYPE_MEDIUM_SAME    = "shadow_same_medium"
GAP_TYPE_MEDIUM_DIFF    = "shadow_diff_medium"
GAP_TYPE_LONG           = "shadow_long"
GAP_TYPE_VERY_LONG      = "shadow_very_long"

# 채우기 신뢰도
CONF_HIGH   = "high"
CONF_MEDIUM = "medium"
CONF_LOW    = "low"
CONF_NONE   = "none"

# gap_method 레이블 (신규 — 어떻게 채웠는지 추적)
METHOD_SAME          = "same_locus"
METHOD_GRAPH_PATH    = "graph_path"
METHOD_W2V_MIDPOINT  = "w2v_midpoint"
METHOD_RATIO_SPLIT   = "ratio_split"


class GapStats(TypedDict):
    total_gaps: int
    filled_gaps: int
    skipped_gaps: int
    filled_records: int
    gap_workers: int
    avg_gap_min: float
    gap_type_dist: dict[str, int]
    low_confidence_records: int


# ── 공간 그래프 / W2V 캐시 로더 ────────────────────────────────────────────────
# pipeline 컨텍스트(비 Streamlit)에서도 작동하도록 try/except 보호

@lru_cache(maxsize=4)
def _get_gap_spatial_graph(sector_id: str):
    """SpatialGraph lru_cache 싱글턴. 실패 시 None 반환."""
    try:
        from src.spatial.graph import get_spatial_graph
        graph = get_spatial_graph(sector_id)
        if graph and graph.node_count > 0:
            logger.info(
                "SpatialGraph 로드 완료: %d 노드, %d 엣지 (sector=%s)",
                graph.node_count, graph.edge_count, sector_id,
            )
        return graph
    except Exception as exc:
        logger.warning("SpatialGraph 로드 실패 (%s): %s — 기존 보간으로 fallback", sector_id, exc)
        return None


@lru_cache(maxsize=4)
def _get_gap_w2v_wv(sector_id: str):
    """Word2Vec KeyedVectors lru_cache 싱글턴. 실패 시 None 반환."""
    try:
        from src.intelligence.journey_embedding import JourneyEmbedder
        embedder = JourneyEmbedder(sector_id)
        if not embedder.is_available():
            logger.debug("W2V 모델 없음 (sector=%s)", sector_id)
            return None
        embedder.load()
        wv = embedder.w2v_model.wv if embedder.w2v_model else None
        if wv:
            logger.info("W2V 모델 로드 완료: vocab %d (sector=%s)", len(wv), sector_id)
        return wv
    except Exception as exc:
        logger.warning("W2V 모델 로드 실패 (%s): %s — W2V fallback 비활성", sector_id, exc)
        return None


# ── 1. GAP 탐지 (벡터화) ──────────────────────────────────────────────────────

def detect_gaps(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    T-Ward 작업자 journey에서 GAP 구간을 탐지하고 분류한다.

    [v2] pandas shift 기반 완전 벡터화 — Python for 루프 없음.

    Args:
        journey_df: journey.parquet 형식 DataFrame
            필수 컬럼: user_no, timestamp, locus_id, has_tward

    Returns:
        gap_df: 각 GAP을 한 행으로 표현한 DataFrame
    """
    if journey_df.empty:
        return pd.DataFrame()

    # T-Ward 착용자만 대상
    if "has_tward" in journey_df.columns:
        df = journey_df[journey_df["has_tward"] == True].copy()
    else:
        df = journey_df.copy()

    if df.empty:
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── 벡터화 shift 연산 ─────────────────────────────────────────────────
    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("user_no", sort=False)
    prev_ts    = grp["timestamp"].shift(1)
    prev_locus = grp["locus_id"].shift(1)

    gap_min = (df["timestamp"] - prev_ts).dt.total_seconds() / 60

    # GAP 조건: 같은 작업자(prev_ts가 NaT 아님) + 임계값 초과
    mask = (prev_ts.notna()) & (gap_min > GAP_MIN_THRESHOLD)
    gaps = df[mask].copy()

    if gaps.empty:
        return pd.DataFrame()

    gaps["gap_min"]      = gap_min[mask].round(1).values
    gaps["gap_start"]    = prev_ts[mask].values
    gaps["gap_end"]      = gaps["timestamp"].values
    gaps["locus_before"] = prev_locus[mask].values
    gaps["locus_after"]  = gaps["locus_id"].values
    gaps["same_locus"]   = gaps["locus_before"] == gaps["locus_after"]

    # ── 벡터화 분류 ───────────────────────────────────────────────────────
    gm   = gaps["gap_min"]
    same = gaps["same_locus"]

    gap_type = np.select(
        [
            (gm <= GAP_SHORT_MAX)  &  same,
            (gm <= GAP_SHORT_MAX)  & ~same,
            (gm <= GAP_MEDIUM_MAX) &  same,
            (gm <= GAP_MEDIUM_MAX) & ~same,
            (gm <= GAP_LONG_MAX),
        ],
        [
            GAP_TYPE_SHORT_SAME,
            GAP_TYPE_SHORT_DIFF,
            GAP_TYPE_MEDIUM_SAME,
            GAP_TYPE_MEDIUM_DIFF,
            GAP_TYPE_LONG,
        ],
        default=GAP_TYPE_VERY_LONG,
    )

    confidence = np.select(
        [
            (gm <= GAP_SHORT_MAX)  &  same,
            (gm <= GAP_SHORT_MAX)  & ~same,
            (gm <= GAP_MEDIUM_MAX) &  same,
            (gm <= GAP_MEDIUM_MAX) & ~same,
            (gm <= GAP_LONG_MAX),
        ],
        [CONF_HIGH, CONF_MEDIUM, CONF_MEDIUM, CONF_LOW, CONF_LOW],
        default=CONF_NONE,
    )

    gaps["gap_type"]   = gap_type
    gaps["confidence"] = confidence

    gap_df = gaps[
        ["user_no", "gap_start", "gap_end", "gap_min",
         "locus_before", "locus_after", "same_locus",
         "gap_type", "confidence"]
    ].reset_index(drop=True)

    logger.info(
        "detect_gaps: %d gaps detected across %d workers",
        len(gap_df),
        gap_df["user_no"].nunique(),
    )
    return gap_df


# ── 2. GAP 채우기 (최적화 + 공간 그래프) ─────────────────────────────────────

def fill_gaps(
    journey_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    sector_id: str | None = None,
) -> pd.DataFrame:
    """
    탐지된 GAP을 분 단위 레코드로 채운다.

    [v3] sector_id 전달 시 SpatialGraph Dijkstra 보간 + W2V fallback 활성화.
    [v2] user_no 기준 사전 그룹화로 O(1) 조회 — iterrows × 전체 스캔 제거.

    Args:
        journey_df: 원본 journey DataFrame
        gap_df: detect_gaps() 결과
        sector_id: 공간 그래프 로드에 사용. None이면 기존 비율 split만 사용.
    """
    _empty_fill = _make_empty_fill(journey_df)

    if gap_df.empty:
        return _empty_fill

    fillable = gap_df[gap_df["confidence"] != CONF_NONE]
    if fillable.empty:
        return _empty_fill

    # ── 공간 그래프 + W2V 사전 로드 (sector_id 있을 때만) ────────────────
    spatial_graph = None
    w2v_wv        = None
    if sector_id:
        spatial_graph = _get_gap_spatial_graph(sector_id)
        w2v_wv        = _get_gap_w2v_wv(sector_id)

    # ── 작업자 기준 사전 캐시 (O(1) 조회) ────────────────────────────────
    base_cols = [
        "user_no", "user_name", "company_name", "company_code",
        "in_datetime", "out_datetime", "has_tward", "twardid",
        "is_work_hour", "missing_exit", "shift_type", "exit_source",
        "work_minutes", "locus_token", "building_name", "floor_name",
    ]
    avail_cols = ["user_no"] + [c for c in base_cols[1:] if c in journey_df.columns]

    user_base: dict[str, dict] = (
        journey_df[avail_cols]
        .groupby("user_no", sort=False)
        .first()
        .to_dict("index")
    )
    user_avg_ar: dict[str, float] = (
        journey_df.groupby("user_no", sort=False)["active_ratio"]
        .mean()
        .to_dict()
    )

    # ── Dijkstra 경로 캐시 (동일 locus 쌍 반복 호출 최적화) ──────────────
    _path_cache: dict[tuple[str, str], list[str]] = {}

    # ── GAP 순회 + 레코드 생성 ────────────────────────────────────────────
    synthetic_rows: list[dict] = []
    for _, gap in fillable.iterrows():
        rows = _generate_fill_records(
            gap, user_base, user_avg_ar,
            spatial_graph=spatial_graph,
            w2v_wv=w2v_wv,
            path_cache=_path_cache,
        )
        synthetic_rows.extend(rows)

    if not synthetic_rows:
        return _empty_fill

    orig = journey_df.copy()
    orig["is_gap_filled"]  = False
    orig["gap_confidence"] = CONF_NONE

    synth_df = pd.DataFrame(synthetic_rows)
    result   = pd.concat([orig, synth_df], ignore_index=True)
    result   = result.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
    result["is_low_confidence"] = _flag_low_confidence(result)

    # 통계 로그
    method_counts: dict[str, int] = {}
    if "gap_method" in synth_df.columns:
        method_counts = synth_df["gap_method"].value_counts().to_dict()

    logger.info(
        "fill_gaps: %d synthetic records added for %d gaps | methods: %s",
        len(synthetic_rows),
        len(fillable),
        method_counts or "legacy",
    )
    return result


def _make_empty_fill(journey_df: pd.DataFrame) -> pd.DataFrame:
    """gap 채울 것이 없을 때 반환할 기본 DataFrame."""
    df = journey_df.copy()
    df["is_gap_filled"]     = False
    df["gap_confidence"]    = CONF_NONE
    df["is_low_confidence"] = _flag_low_confidence(df)
    return df


def _generate_fill_records(
    gap: pd.Series,
    user_base: dict[str, dict],
    user_avg_ar: dict[str, float],
    spatial_graph=None,
    w2v_wv=None,
    path_cache: "dict | None" = None,
) -> list[dict]:
    """단일 GAP에 대해 분 단위 채우기 레코드 목록을 생성한다."""
    gap_type   = gap["gap_type"]
    locus_b    = gap["locus_before"]
    locus_a    = gap["locus_after"]
    confidence = gap["confidence"]
    user_no    = gap["user_no"]

    start_ts = gap["gap_start"] + pd.Timedelta(minutes=1)
    end_ts   = gap["gap_end"]   - pd.Timedelta(minutes=1)

    # start_ts > end_ts 일 때만 skip (== 허용: 2분 gap의 1분 interior 채움)
    if start_ts > end_ts:
        return []

    minutes = pd.date_range(start=start_ts, end=end_ts, freq="1min")
    n = len(minutes)
    if n == 0:
        return []

    locus_seq, method, upgraded_conf = _assign_locus_sequence(
        gap_type, locus_b, locus_a, n,
        spatial_graph=spatial_graph,
        w2v_wv=w2v_wv,
        path_cache=path_cache,
    )
    # graph_path로 채운 경우 신뢰도 업그레이드 (low → medium)
    final_conf = upgraded_conf if upgraded_conf else confidence

    base    = user_base.get(user_no, {})
    avg_ar  = user_avg_ar.get(user_no, 0.5)

    records = []
    for i, ts in enumerate(minutes):
        locus = locus_seq[i]
        row = {
            "user_no":       user_no,
            "timestamp":     ts,
            "user_name":     base.get("user_name", ""),
            "company_name":  base.get("company_name", ""),
            "company_code":  base.get("company_code", ""),
            "locus_id":      locus,
            "locus_token":   locus,
            "building_name": base.get("building_name", ""),
            "floor_name":    base.get("floor_name", ""),
            "spot_name":     "",
            "x":             np.nan,
            "y":             np.nan,
            "signal_count":  0,
            "active_count":  0,
            "active_ratio":  avg_ar,
            "in_datetime":   base.get("in_datetime"),
            "out_datetime":  base.get("out_datetime"),
            "has_tward":     True,
            "twardid":       base.get("twardid"),
            "is_work_hour":  base.get("is_work_hour", True),
            "missing_exit":  base.get("missing_exit", False),
            "shift_type":    base.get("shift_type", ""),
            "exit_source":   base.get("exit_source", ""),
            "work_minutes":  base.get("work_minutes", np.nan),
            "is_gap_filled": True,
            "gap_confidence": final_conf,
            "gap_method":    method,
        }
        records.append(row)

    return records


# ── 공간 보간 헬퍼 ─────────────────────────────────────────────────────────────

def _path_to_locus_sequence(
    path: list[str],
    n: int,
    graph,
) -> list[str]:
    """
    Dijkstra 경로 → n개 locus 시퀀스.

    각 엣지의 transition_cost_min 비율로 fill 분을 배분한다.

    세그먼트 i (path[i] → path[i+1]):
      - i < 마지막: 목적지 path[i+1]에 체류 (진입 중)
      - i == 마지막: path[-2] 유지 (path[-1]=locus_a는 이미 journey에 존재)
    """
    if len(path) < 2 or n == 0:
        return [path[0]] * n if path else []

    # 엣지 가중치 수집
    edge_weights: list[float] = []
    for i in range(len(path) - 1):
        ed = graph.G.get_edge_data(path[i], path[i + 1])
        edge_weights.append(float(ed.get("weight", 1.0)) if ed else 1.0)

    total_cost = sum(edge_weights) or 1.0

    # 각 세그먼트의 "체류" locus
    target_loci: list[str] = []
    for i in range(len(path) - 1):
        if i == len(path) - 2:   # 마지막 세그먼트
            target_loci.append(path[-2])
        else:
            target_loci.append(path[i + 1])

    # n분 비율 배분 (각 세그먼트 최소 1분)
    mins: list[int] = [max(1, round(n * w / total_cost)) for w in edge_weights]

    # 반올림 오차 보정 — 가장 큰 세그먼트에서 조정
    diff = n - sum(mins)
    if diff != 0:
        idx = mins.index(max(mins))
        mins[idx] = max(1, mins[idx] + diff)

    result: list[str] = []
    for locus, m in zip(target_loci, mins):
        result.extend([locus] * m)

    # 길이 불일치 방어 (반올림 엣지 케이스)
    fallback = target_loci[-1] if target_loci else path[-2]
    while len(result) < n:
        result.append(fallback)
    return result[:n]


def _w2v_midpoint_locus(
    locus_b: str,
    locus_a: str,
    wv,
    topn: int = 10,
) -> str | None:
    """
    Word2Vec co-occurrence로 locus_b와 locus_a 모두와 유사한 중간 경유 locus 추정.

    두 locus의 most_similar 교집합에서 벡터 중점에 가장 가까운 locus를 반환.
    교집합이 없거나 vocab에 없으면 None.
    """
    try:
        if locus_b not in wv or locus_a not in wv:
            return None

        sim_b = {locus for locus, _ in wv.most_similar(locus_b, topn=topn)}
        sim_a = {locus for locus, _ in wv.most_similar(locus_a, topn=topn)}

        common = (sim_b & sim_a) - {locus_b, locus_a}
        if not common:
            return None

        # 두 벡터 중점에 가장 가까운 locus 선택
        mid_vec = (wv[locus_b] + wv[locus_a]) / 2.0
        best = max(common, key=lambda lid: float(wv.cosine_similarities(mid_vec, [wv[lid]])[0]))
        return best
    except Exception as exc:
        logger.debug("W2V midpoint 실패 (%s→%s): %s", locus_b, locus_a, exc)
        return None


def _assign_locus_sequence(
    gap_type: str,
    locus_b: str,
    locus_a: str,
    n: int,
    spatial_graph=None,
    w2v_wv=None,
    path_cache: "dict | None" = None,
) -> tuple[list[str], str, str | None]:
    """
    GAP 유형 + 공간 그래프에 따라 n개 분의 locus 시퀀스를 반환한다.

    Returns:
        (locus_seq, method, upgraded_confidence)
        - method: METHOD_* 상수
        - upgraded_confidence: 기존 confidence 대비 업그레이드가 필요하면 새 값, 아니면 None
    """
    # ── same locus: 공간 추론 불필요 ────────────────────────────────────
    if gap_type in (GAP_TYPE_SHORT_SAME, GAP_TYPE_MEDIUM_SAME):
        return [locus_b] * n, METHOD_SAME, None

    # ── very_long: 방어적 처리 (fillable에서 이미 제외되나 안전장치) ───
    if gap_type == GAP_TYPE_VERY_LONG:
        return [locus_b] * n, METHOD_RATIO_SPLIT, None

    # ── diff: Step 1 — Dijkstra 경로 보간 ────────────────────────────
    if spatial_graph is not None:
        cache_key = (locus_b, locus_a)
        if path_cache is not None and cache_key in path_cache:
            path = path_cache[cache_key]
        else:
            path = spatial_graph.shortest_path(locus_b, locus_a)
            if path_cache is not None:
                path_cache[cache_key] = path

        if path and len(path) > 2:
            # 중간 노드가 있는 경로 → 비율 배분 보간
            seq = _path_to_locus_sequence(path, n, spatial_graph)
            upgraded = CONF_MEDIUM  # low였던 gap도 경로가 명확하면 medium으로 업그레이드
            return seq, METHOD_GRAPH_PATH, upgraded

        if path and len(path) == 2:
            # 직접 인접 — 중간 노드 없음, 단순 split으로도 충분
            # (별도 경로 보간 없이 fallback으로 넘어감)
            pass

    # ── diff: Step 2 — W2V 중간 locus 추정 ───────────────────────────
    if w2v_wv is not None and n >= 3:
        mid = _w2v_midpoint_locus(locus_b, locus_a, w2v_wv)
        if mid:
            third = n // 3
            seq = [locus_b] * third + [mid] * third + [locus_a] * (n - 2 * third)
            return seq, METHOD_W2V_MIDPOINT, None

    # ── fallback: 기존 비율 split ─────────────────────────────────────
    if gap_type == GAP_TYPE_SHORT_DIFF:
        split = n // 2
        seq = [locus_b] * split + [locus_a] * (n - split)
    elif gap_type == GAP_TYPE_MEDIUM_DIFF:
        split = int(n * 0.6)
        seq = [locus_b] * split + [locus_a] * (n - split)
    else:  # GAP_TYPE_LONG
        split = n // 2
        seq = [locus_b] * split + [locus_a] * (n - split)

    return seq, METHOD_RATIO_SPLIT, None


def _flag_low_confidence(df: pd.DataFrame) -> pd.Series:
    if "signal_count" not in df.columns:
        return pd.Series(False, index=df.index)
    is_orig = ~df.get("is_gap_filled", pd.Series(False, index=df.index))
    low_sig = df["signal_count"].fillna(0) <= LOW_CONFIDENCE_SIGNAL
    return is_orig & low_sig


# ── 3. 활성 레벨 분류 ─────────────────────────────────────────────────────────

def classify_activity(journey_df: pd.DataFrame) -> pd.DataFrame:
    """각 레코드에 activity_level 컬럼을 추가한다."""
    df = journey_df.copy()

    ar = df["active_ratio"].fillna(0.0)
    sc = df["signal_count"].fillna(0)
    is_filled = df.get("is_gap_filled", pd.Series(False, index=df.index))

    conditions = [
        is_filled,
        (ar >= AR_HIGH_ACTIVE) & (sc >= SC_HIGH_ACTIVE_MIN),
        (ar >= AR_ACTIVE)      & (sc >= SC_ACTIVE_MIN),
        ar >= AR_INACTIVE,
    ]
    choices = ["ESTIMATED", "HIGH_ACTIVE", "ACTIVE", "INACTIVE"]
    df["activity_level"] = np.select(conditions, choices, default="DEEP_INACTIVE")

    return df


# ── 4. 작업자별 활성 집계 ─────────────────────────────────────────────────────

def aggregate_activity_by_worker(journey_df: pd.DataFrame) -> pd.DataFrame:
    """작업자별 activity_level 집계 지표를 계산한다."""
    if "activity_level" not in journey_df.columns:
        journey_df = classify_activity(journey_df)

    rows = []
    for user_no, grp in journey_df.groupby("user_no"):
        level_counts = grp["activity_level"].value_counts()
        total_min    = len(grp)

        high_active_min   = int(level_counts.get("HIGH_ACTIVE",   0))
        active_min        = int(level_counts.get("ACTIVE",        0))
        inactive_min      = int(level_counts.get("INACTIVE",      0))
        deep_inactive_min = int(level_counts.get("DEEP_INACTIVE", 0))
        estimated_min     = int(level_counts.get("ESTIMATED",     0))

        gap_ratio = round(estimated_min / total_min * 100, 1) if total_min > 0 else 0.0
        real_counts = level_counts.drop("ESTIMATED", errors="ignore")
        dominant    = real_counts.idxmax() if len(real_counts) > 0 else "UNKNOWN"

        rows.append({
            "user_no":           user_no,
            "high_active_min":   high_active_min,
            "active_min":        active_min,
            "inactive_min":      inactive_min,
            "deep_inactive_min": deep_inactive_min,
            "estimated_min":     estimated_min,
            "gap_ratio_pct":     gap_ratio,
            "dominant_activity": dominant,
        })

    return pd.DataFrame(rows)


# ── 5. Locus별 활성 집계 ─────────────────────────────────────────────────────

def aggregate_activity_by_locus(journey_df: pd.DataFrame) -> pd.DataFrame:
    """Locus별 activity_level 분포 및 평균 active_ratio 집계."""
    if "activity_level" not in journey_df.columns:
        journey_df = classify_activity(journey_df)

    real = journey_df[~journey_df.get("is_gap_filled", pd.Series(False, index=journey_df.index))]

    rows = []
    for locus_id, grp in real.groupby("locus_id"):
        total = len(grp)
        if total == 0:
            continue
        counts = grp["activity_level"].value_counts()
        rows.append({
            "locus_id":          locus_id,
            "total_min":         total,
            "high_active_pct":   round(counts.get("HIGH_ACTIVE",   0) / total * 100, 1),
            "active_pct":        round(counts.get("ACTIVE",        0) / total * 100, 1),
            "inactive_pct":      round(counts.get("INACTIVE",      0) / total * 100, 1),
            "deep_inactive_pct": round(counts.get("DEEP_INACTIVE", 0) / total * 100, 1),
            "avg_active_ratio":  round(grp["active_ratio"].mean(), 3),
        })

    return pd.DataFrame(rows)


# ── 6. 통합 실행 ─────────────────────────────────────────────────────────────

def analyze_gaps(
    journey_df: pd.DataFrame,
    sector_id: str | None = None,
) -> tuple[pd.DataFrame, GapStats]:
    """
    GAP 탐지 → 채우기 → 활성 분류 3단계를 통합 실행한다.

    Args:
        journey_df: 원본 journey DataFrame
        sector_id: 공간 그래프 보간 활성화 키. None이면 기존 비율 split 사용.
    """
    gap_df    = detect_gaps(journey_df)
    filled_df = fill_gaps(journey_df, gap_df, sector_id=sector_id)
    filled_df = classify_activity(filled_df)
    stats     = _build_stats(gap_df, filled_df)

    logger.info(
        "analyze_gaps complete: %d gaps → %d filled records added, "
        "%d low_confidence original records",
        stats["total_gaps"],
        stats["filled_records"],
        stats["low_confidence_records"],
    )
    return filled_df, stats


def _build_stats(gap_df: pd.DataFrame, filled_df: pd.DataFrame) -> GapStats:
    if gap_df.empty:
        return GapStats(
            total_gaps=0, filled_gaps=0, skipped_gaps=0, filled_records=0,
            gap_workers=0, avg_gap_min=0.0, gap_type_dist={},
            low_confidence_records=0,
        )

    fillable      = gap_df[gap_df["confidence"] != CONF_NONE]
    skipped       = gap_df[gap_df["confidence"] == CONF_NONE]
    n_filled_rec  = int(filled_df.get("is_gap_filled",     pd.Series(False)).sum())
    n_low_conf    = int(filled_df.get("is_low_confidence", pd.Series(False)).sum())

    return GapStats(
        total_gaps=len(gap_df),
        filled_gaps=len(fillable),
        skipped_gaps=len(skipped),
        filled_records=n_filled_rec,
        gap_workers=int(gap_df["user_no"].nunique()),
        avg_gap_min=round(float(gap_df["gap_min"].mean()), 1),
        gap_type_dist=gap_df["gap_type"].value_counts().to_dict(),
        low_confidence_records=n_low_conf,
    )
