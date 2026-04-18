"""
Journey Corrector — BLE 이동 노이즈 보정 모듈
===============================================

BLE 신호 특성상 발생하는 순간적인 위치 튐(노이즈)을 제거하고
작업자의 실제 동선(Journey)을 추정한다.

[v2 공간 그래프 + 층/건물 우선 보정] 2026-04-16
  - Phase 1 업그레이드: 인접 여부 + 층/건물 관계로 플리커 임계값 3단계 분기
      · 인접 (어느 층이든)           : MAX_FLICKER_RUN_ADJ=2 (보수적, 실제 이동 가능)
      · 비인접, 같은 층              : MAX_FLICKER_RUN_SAME_FLOOR=3 (왕복 최소 4분)
      · 비인접, 다른 층 (같은 건물)   : MAX_FLICKER_RUN_CROSS_FLOOR=4 (층 이동 불가)
      · 비인접, 다른 건물             : MAX_FLICKER_RUN_CROSS_BLDG=5 (건물 이동 불가)
  - Phase 2 신규: Ghost Locus 제거
      · run_len=1이고 앞뒤 locus 모두 비인접 → 명백한 BLE 신호 누설
      · 교정 후 locus_token도 동기화

[v1 원본] Run-Length 기반 플리커 보정 (전체 데이터 단일 패스, 완전 벡터화)
    - 연속 동일 locus 구간(Run) 식별 (user_no 경계에서 강제 분리)
    - Run 길이 ≤ MAX_FLICKER_RUN 이고 앞뒤 locus가 동일하면 플리커 → 앞 locus로 교체
    - 앵커 토큰(timeclock/breakroom/smoking_area)은 보호
    - 작업자별 for 루프 없이 groupby + map으로 처리

설계 원칙 (DeepCon_SOIF v6.1 Multi-Pass 철학):
  "앵커 공간 보호 → Run-Level 플리커 → Ghost 제거 → 보수적 보정"
  - 실제 이동(A→B→C)은 건드리지 않음
  - 오직 공간적으로 불가능하거나 매우 짧은 비인접 노이즈만 제거
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 보정 상수 ────────────────────────────────────────────────────────────
# ★ [v2] 인접 여부 + 층/건물 관계로 3단계 분기
MAX_FLICKER_RUN_ADJ:         int = 2  # 인접: 보수적 (실제 이동일 수 있음)
MAX_FLICKER_RUN_SAME_FLOOR:  int = 3  # 비인접, 같은 층: 왕복 최소 4분 필요
MAX_FLICKER_RUN_CROSS_FLOOR: int = 4  # 비인접, 다른 층 (같은 건물): 층 이동 왕복 불가
MAX_FLICKER_RUN_CROSS_BLDG:  int = 5  # 비인접, 다른 건물: 건물 이동 왕복 불가

# [v1 호환] sector_id 없을 때 사용되는 기존 단일 임계값
MAX_FLICKER_RUN: int = 2

# 앵커 locus_token — 짧아도 보정하지 않는 실제 이벤트 공간
ANCHOR_TOKENS: frozenset[str] = frozenset({
    "timeclock",    # 타각기/게이트 — 출입 확인 이벤트
    "breakroom",    # 휴게실
    "smoking_area", # 흡연 구역
})


# ─── 공간 그래프 캐시 ──────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _get_corrector_spatial_graph(sector_id: str):
    """Corrector 전용 SpatialGraph lru_cache 싱글턴. 실패 시 None 반환."""
    try:
        from src.spatial.graph import get_spatial_graph
        graph = get_spatial_graph(sector_id)
        if graph and graph.node_count > 0:
            logger.info(
                "Corrector SpatialGraph 로드 완료: %d 노드, %d 엣지 (sector=%s)",
                graph.node_count, graph.edge_count, sector_id,
            )
        return graph
    except Exception as exc:
        logger.warning("Corrector SpatialGraph 로드 실패 (%s): %s — 기존 보정으로 fallback", sector_id, exc)
        return None


def _build_adj_pairs(graph) -> frozenset[str]:
    """양방향 인접 쌍을 'GW-A|GW-B' 포맷의 frozenset으로 구축 (O(1) 조회용)."""
    pairs: set[str] = set()
    for a, b in graph.G.edges():
        pairs.add(f"{a}|{b}")
        pairs.add(f"{b}|{a}")
    return frozenset(pairs)


@lru_cache(maxsize=4)
def _get_locus_location_map(sector_id: str) -> dict:
    """
    locus_id → (building_no, floor_no) 정수 튜플 매핑.
    NaN 또는 미발견 locus는 (-1, -1)로 표현한다.

    locus_v2.csv의 building_no / floor_no 컬럼을 읽어 캐싱.
    프로세스당 1회만 로드 (lru_cache).

    Returns:
        dict[str, tuple[int, int]]  — 예: {"GW-348": (2, 27), "GW-001": (-1, -1)}
    """
    try:
        import config as cfg

        paths = cfg.get_sector_paths(sector_id)
        csv_path = paths.get("locus_v2_csv")
        if not csv_path or not csv_path.exists():
            logger.warning(
                "locus_v2.csv 없음 (%s) — 층/건물 보정 비활성화", sector_id
            )
            return {}

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df["locus_id"] = df["locus_id"].astype(str).str.strip()

        for col in ("building_no", "floor_no"):
            if col not in df.columns:
                df[col] = float("nan")

        df["_b"] = (
            pd.to_numeric(df["building_no"], errors="coerce").fillna(-1).astype(int)
        )
        df["_f"] = (
            pd.to_numeric(df["floor_no"], errors="coerce").fillna(-1).astype(int)
        )
        loc_map: dict = dict(zip(df["locus_id"], zip(df["_b"], df["_f"])))
        logger.info(
            "loc_map 로드 완료: %d locus (sector=%s)", len(loc_map), sector_id
        )
        return loc_map
    except Exception as exc:
        logger.warning(
            "loc_map 로드 실패 (%s): %s — 층/건물 보정 비활성화", sector_id, exc
        )
        return {}


# ─── 공개 진입점 ──────────────────────────────────────────────────────────
def correct_journeys(
    journey_df: pd.DataFrame,
    sector_id: str | None = None,
    max_passes: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    모든 작업자에 대해 Journey 보정 수행 (수렴형 멀티패스 + Ghost 제거).

    [v2] sector_id 전달 시 SpatialGraph를 활용해:
      - Phase 1: 비인접 플리커 임계값 확장 (2→5분)
      - Phase 2: Ghost locus (앞뒤 모두 비인접 고립 1분 레코드) 제거

    Args:
        journey_df: apply_locus_mapping + build_worker_journeys 이후 DataFrame
                    필수 컬럼: user_no, timestamp, locus_id, locus_token
        sector_id:  공간 그래프 활성화 키. None이면 기존 run-length만 사용.
        max_passes: 최대 보정 반복 횟수 (기본 3, 변경 0이면 조기 종료)

    Returns:
        (corrected_df, stats)
    """
    if journey_df.empty:
        return journey_df, _empty_stats()

    # 정렬 보장 (user_no + timestamp)
    df = journey_df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
    original_loci = df["locus_id"].copy()

    # ── 공간 그래프 + 인접 pair set + 층/건물 맵 사전 구축 ─────────────────
    adj_pairs: frozenset[str] | None = None
    loc_map:   dict | None           = None
    if sector_id:
        graph = _get_corrector_spatial_graph(sector_id)
        if graph:
            adj_pairs = _build_adj_pairs(graph)
        _lm = _get_locus_location_map(sector_id)
        if _lm:
            loc_map = _lm

    # ── Phase 1: Run-Length 기반 플리커 보정 (수렴형 멀티패스) ──────────────
    passes_used = 0
    for pass_num in range(max_passes):
        pre_pass_loci = df["locus_id"].copy()
        df = _run_flicker_global(df, adj_pairs=adj_pairs, loc_map=loc_map)
        changed = int((df["locus_id"].values != pre_pass_loci.values).sum())
        passes_used = pass_num + 1
        logger.info("보정 Phase1 Pass %d: %d건 변경", passes_used, changed)
        if changed == 0:
            break

    flicker_changed = int((df["locus_id"].values != original_loci.values).sum())

    # ── Phase 2: Ghost Locus 제거 (단일 패스, sector_id 있을 때만) ──────────
    ghost_changed = 0
    if adj_pairs is not None:
        df, ghost_changed = _remove_ghost_loci(df, adj_pairs)
        logger.info("보정 Phase2 Ghost: %d건 제거", ghost_changed)

    # ── 통계 집계 ────────────────────────────────────────────────────────────
    changed_mask   = df["locus_id"].values != original_loci.values
    total_changed  = int(changed_mask.sum())
    total_records  = len(df)
    workers_changed = (
        int(df.loc[changed_mask, "user_no"].nunique())
        if total_changed > 0 else 0
    )
    correction_ratio = round(total_changed / total_records * 100, 2) if total_records > 0 else 0.0

    stats = {
        "corrected_records":  total_changed,
        "correction_ratio":   correction_ratio,
        "workers_corrected":  workers_changed,
        "passes_used":        passes_used,
        "flicker_corrected":  flicker_changed,
        "ghost_corrected":    ghost_changed,
    }
    logger.info(
        "Journey 보정 완료: %d건 (%.2f%%) / 작업자 %d명 / %d패스 "
        "[플리커:%d / Ghost:%d]",
        total_changed, correction_ratio, workers_changed, passes_used,
        flicker_changed, ghost_changed,
    )
    return df, stats


# ─── Phase 1: Run-Length 기반 플리커 보정 (전체 데이터 단일 패스) ─────────
def _run_flicker_global(
    df: pd.DataFrame,
    adj_pairs: "frozenset[str] | None" = None,
    loc_map:   "dict | None"           = None,
) -> pd.DataFrame:
    """
    전체 데이터프레임에 대해 단일 패스로 Run-Length 플리커 보정 수행.

    [v2] adj_pairs + loc_map 제공 시 4단계 임계값 분기:
      ① 인접 (adjacency 엣지 존재)      → MAX_FLICKER_RUN_ADJ=2
      ② 비인접, 같은 층 (같은 건물)      → MAX_FLICKER_RUN_SAME_FLOOR=3
      ③ 비인접, 다른 층 (같은 건물)      → MAX_FLICKER_RUN_CROSS_FLOOR=4
      ④ 비인접, 다른 건물 / 층 미지      → MAX_FLICKER_RUN_CROSS_BLDG=5

    [v2 adj_pairs만] 인접/비인접 2단계 (loc_map 없을 때):
      ① 인접  → MAX_FLICKER_RUN_ADJ=2
      ② 비인접 → MAX_FLICKER_RUN_CROSS_BLDG=5 (보수 fallback)

    [v1 호환] adj_pairs=None → MAX_FLICKER_RUN=2 단일 임계값

    핵심 원리:
      A, A, A, [B, B], A, A  →  B 구간(Run 길이 ≤ threshold)의 앞뒤가 A로 동일 → 플리커 제거

    구현 (for 루프 없이 완전 벡터화):
      1. user_no 경계 또는 locus_id 변경 시 새 Run 시작 → run_id 생성
      2. run별 agg: locus_id, run_len, is_anchor (첫 토큰 기준)
      3. 플리커 조건 판별 (pandas 벡터 연산)
      4. 플리커 run_id에 해당하는 행의 locus_id를 이전 run locus로 교체
         → map() 사용으로 루프 없이 처리
    """
    result = df.copy()
    loci      = result["locus_id"].values
    user_nos  = result["user_no"].values

    # ── 1. Run ID 생성 ─────────────────────────────────────────────────
    is_change = np.concatenate([[True], (
        (loci[1:] != loci[:-1]) | (user_nos[1:] != user_nos[:-1])
    )])
    run_id_arr = np.cumsum(is_change).astype(np.int32)
    result["_run_id"] = run_id_arr

    # ── 2. Run별 집계 ──────────────────────────────────────────────────
    agg_cols = {
        "locus_id":  ("locus_id",  "first"),
        "run_len":   ("_run_id",   "count"),
        "user_no":   ("user_no",   "first"),
    }
    if "locus_token" in result.columns:
        agg_cols["first_token"] = ("locus_token", "first")

    run_info = result.groupby("_run_id", sort=False).agg(**agg_cols).reset_index()

    # 앵커 여부
    if "first_token" in run_info.columns:
        run_info["is_anchor"] = run_info["first_token"].isin(ANCHOR_TOKENS)
    else:
        run_info["is_anchor"] = False

    # ── 3. 이전/다음 Run 정보 ──────────────────────────────────────────
    run_info["prev_locus"]   = run_info["locus_id"].shift(1)
    run_info["next_locus"]   = run_info["locus_id"].shift(-1)
    run_info["prev_user_no"] = run_info["user_no"].shift(1)
    run_info["next_user_no"] = run_info["user_no"].shift(-1)

    # ── 4. 플리커 조건 판별 ────────────────────────────────────────────
    same_user_both = (
        (run_info["prev_user_no"] == run_info["user_no"])
        & (run_info["next_user_no"] == run_info["user_no"])
    )
    sandwich_base = (
        run_info["prev_locus"].notna()
        & run_info["next_locus"].notna()
        & (run_info["prev_locus"] == run_info["next_locus"])   # A-[B]-A 패턴
        & (run_info["prev_locus"] != run_info["locus_id"])     # 현재와 다름
        & ~run_info["is_anchor"]
        & same_user_both
    )

    if adj_pairs is not None:
        # 인접 여부 (prev_locus → current locus 방향)
        adj_key = run_info["prev_locus"].fillna("") + "|" + run_info["locus_id"]
        is_adj  = adj_key.isin(adj_pairs)

        if loc_map:
            # ── [v2 full] 4단계 임계값: 인접 / 같은층 / 다른층 / 다른건물 ──
            locus_bldg  = {lid: b for lid, (b, _f) in loc_map.items()}
            locus_floor = {lid: f for lid, (_b, f) in loc_map.items()}

            run_info["bldg_curr"]  = run_info["locus_id"].map(locus_bldg).fillna(-1).astype(int)
            run_info["bldg_prev"]  = run_info["prev_locus"].map(locus_bldg).fillna(-1).astype(int)
            run_info["floor_curr"] = run_info["locus_id"].map(locus_floor).fillna(-1).astype(int)
            run_info["floor_prev"] = run_info["prev_locus"].map(locus_floor).fillna(-1).astype(int)

            # -1 = 미지 → 다른 건물로 간주 (보수적)
            has_bldg_both  = (run_info["bldg_curr"]  != -1) & (run_info["bldg_prev"]  != -1)
            has_floor_both = (run_info["floor_curr"] != -1) & (run_info["floor_prev"] != -1)
            is_same_bldg   = has_bldg_both  & (run_info["bldg_curr"]  == run_info["bldg_prev"])
            is_same_floor  = (
                is_same_bldg & has_floor_both
                & (run_info["floor_curr"] == run_info["floor_prev"])
            )

            # 비인접 threshold: 같은층 > 다른층 > 다른건물 순으로 엄격해짐
            max_run_nonadj = np.where(
                is_same_floor.values,
                MAX_FLICKER_RUN_SAME_FLOOR,
                np.where(
                    is_same_bldg.values,
                    MAX_FLICKER_RUN_CROSS_FLOOR,
                    MAX_FLICKER_RUN_CROSS_BLDG,
                ),
            )
            max_run_arr = np.where(
                is_adj.values, MAX_FLICKER_RUN_ADJ, max_run_nonadj
            )
        else:
            # ── [v2 lite] adj_pairs만, loc_map 없음 → 2단계 ──
            max_run_arr = np.where(
                is_adj.values, MAX_FLICKER_RUN_ADJ, MAX_FLICKER_RUN_CROSS_BLDG
            )

        flicker_mask = sandwich_base & (run_info["run_len"].values <= max_run_arr)
    else:
        # [v1 호환] 기존 단일 임계값
        flicker_mask = sandwich_base & (run_info["run_len"] <= MAX_FLICKER_RUN)

    flicker_rows = run_info[flicker_mask]
    if flicker_rows.empty:
        result.drop(columns=["_run_id"], inplace=True)
        return result

    # ── 5. 플리커 run → 이전 locus로 교체 (map, 루프 없음) ───────────
    replace_map = dict(zip(flicker_rows["_run_id"], flicker_rows["prev_locus"]))
    is_flicker_row = result["_run_id"].isin(replace_map)

    # locus_id 교체
    result.loc[is_flicker_row, "locus_id"] = (
        result.loc[is_flicker_row, "_run_id"].map(replace_map).values
    )
    # locus_token 동기화 (컬럼 존재 시)
    _sync_locus_token(result, is_flicker_row)

    result.drop(columns=["_run_id"], inplace=True)
    return result


# ─── Phase 2: Ghost Locus 제거 ────────────────────────────────────────────
def _remove_ghost_loci(
    df: pd.DataFrame,
    adj_pairs: frozenset[str],
) -> tuple[pd.DataFrame, int]:
    """
    앞뒤 locus 모두와 비인접한 고립 단일 레코드(Ghost)를 제거한다.

    조건:
      - run_len = 1
      - 해당 locus가 prev_locus와도 비인접 (신호 누설)
      - 해당 locus가 next_locus와도 비인접 (경유 지점이 아님)
      - 앵커 토큰 아님
      - 작업자 경계 없음 (앞뒤 같은 작업자)

    교체: prev_locus로 대체 (해당 분은 아직 이전 locus에 있던 것으로 추정)

    Returns:
        (corrected_df, n_ghost_removed)
    """
    result = df.copy()
    loci     = result["locus_id"].values
    user_nos = result["user_no"].values

    # ── Run ID 재계산 (Phase 1 이후 locus_id 변경됨) ───────────────────
    is_change = np.concatenate([[True], (
        (loci[1:] != loci[:-1]) | (user_nos[1:] != user_nos[:-1])
    )])
    run_id_arr = np.cumsum(is_change).astype(np.int32)
    result["_run_id"] = run_id_arr

    # ── Run별 집계 ──────────────────────────────────────────────────────
    agg_cols: dict = {
        "locus_id":  ("locus_id", "first"),
        "run_len":   ("_run_id",  "count"),
        "user_no":   ("user_no",  "first"),
    }
    if "locus_token" in result.columns:
        agg_cols["first_token"] = ("locus_token", "first")

    run_info = result.groupby("_run_id", sort=False).agg(**agg_cols).reset_index()

    if "first_token" in run_info.columns:
        run_info["is_anchor"] = run_info["first_token"].isin(ANCHOR_TOKENS)
    else:
        run_info["is_anchor"] = False

    run_info["prev_locus"]   = run_info["locus_id"].shift(1)
    run_info["next_locus"]   = run_info["locus_id"].shift(-1)
    run_info["prev_user_no"] = run_info["user_no"].shift(1)
    run_info["next_user_no"] = run_info["user_no"].shift(-1)

    # ── Ghost 조건 ──────────────────────────────────────────────────────
    # 인접 여부 (양방향)
    adj_prev_key = run_info["prev_locus"].fillna("") + "|" + run_info["locus_id"]
    adj_next_key = run_info["locus_id"] + "|" + run_info["next_locus"].fillna("")
    is_adj_prev  = adj_prev_key.isin(adj_pairs)
    is_adj_next  = adj_next_key.isin(adj_pairs)

    same_user_both = (
        (run_info["prev_user_no"] == run_info["user_no"])
        & (run_info["next_user_no"] == run_info["user_no"])
    )

    ghost_mask = (
        (run_info["run_len"] == 1)          # 단일 레코드
        & ~is_adj_prev                       # prev와 비인접
        & ~is_adj_next                       # next와 비인접
        & run_info["prev_locus"].notna()
        & run_info["next_locus"].notna()
        & ~run_info["is_anchor"]             # 앵커 보호
        & same_user_both
    )

    ghost_rows = run_info[ghost_mask]
    n_ghost = len(ghost_rows)

    if ghost_rows.empty:
        result.drop(columns=["_run_id"], inplace=True)
        return result, 0

    # ── Ghost run → prev_locus로 교체 ────────────────────────────────
    replace_map = dict(zip(ghost_rows["_run_id"], ghost_rows["prev_locus"]))
    is_ghost_row = result["_run_id"].isin(replace_map)

    result.loc[is_ghost_row, "locus_id"] = (
        result.loc[is_ghost_row, "_run_id"].map(replace_map).values
    )
    _sync_locus_token(result, is_ghost_row)

    result.drop(columns=["_run_id"], inplace=True)
    return result, n_ghost


# ─── 헬퍼 ────────────────────────────────────────────────────────────────
def _sync_locus_token(df: pd.DataFrame, mask: pd.Series) -> None:
    """
    locus_id 교체 후 locus_token을 동기화한다 (in-place).

    교체된 locus_id에 대응하는 locus_token을 DataFrame 내 다른 행에서 조회하여 갱신.
    locus_token이 없는 locus_id는 locus_id 그대로 사용 (fallback).
    """
    if "locus_token" not in df.columns:
        return
    if not mask.any():
        return

    # locus_id → locus_token 역방향 맵 (교체 전 원본 데이터에서 구축)
    lid_to_token = (
        df[~mask][["locus_id", "locus_token"]]
        .drop_duplicates("locus_id")
        .set_index("locus_id")["locus_token"]
        .to_dict()
    )
    new_tokens = df.loc[mask, "locus_id"].map(lid_to_token)
    # 조회 실패 시 locus_id 그대로 사용 (token = id)
    df.loc[mask, "locus_token"] = new_tokens.fillna(df.loc[mask, "locus_id"]).values


def _empty_stats() -> dict:
    return {
        "corrected_records":  0,
        "correction_ratio":   0.0,
        "workers_corrected":  0,
        "passes_used":        0,
        "flicker_corrected":  0,
        "ghost_corrected":    0,
    }
