"""DeepCon Metrics — EWI, CRE, SII (v1.2)
========================================
DeepCon 데이터 구조에 맞춘 생산성·안전 지표 계산 모듈.
DeepCon_SOIF v6.9 알고리즘을 DeepCon locus 기반으로 이식.

v1.2 (2026-04-06): v2 Locus (GW-XXX, 213개) 호환
  - 토큰 기반 분류 → locus CSV 메타데이터 기반 분류로 전환
  - v1 하드코딩 토큰은 fallback으로 유지
v1.1 (2026-03-28): enriched locus 데이터 통합
  - hazard_level/hazard_grade 기반 static_risk 계산
  - locus_dict 파라미터 추가 (하위 호환 유지)

DeepCon_SOIF vs DeepCon 주요 차이:
  DeepCon_SOIF                  DeepCon
  ─────────────────────────── ──────────────────────────────
  PLACE_TYPE / SPACE_FUNCTION  locus_token (공간 역할 토큰)
  accel_state (moving/static)  active_ratio (BLE 신호 활성도)
  detect_work_shift()          AccessLog in_datetime/out_datetime
  IS_TRANSIT_EPISODE           locus_token in TRANSIT_TOKENS
  corrected_place              locus_id (spot_name → locus_id 매핑)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── 활성도 임계값 ──────────────────────────────────────────────────
HIGH_ACTIVE_THRESHOLD = 0.90   # 고활성: BLE 활성비율 90%+ (실제 집중 작업)
LOW_ACTIVE_THRESHOLD  = 0.40   # 저활성 하한: 이 미만은 대기로 분류

# ─── EWI 가중치 ──────────────────────────────────────────────────────
HIGH_WORK_WEIGHT = 1.0    # 고활성 작업
LOW_WORK_WEIGHT  = 0.5    # 저활성 작업
STANDBY_WEIGHT   = 0.2    # 대기 (크레인 대기·감독 등 실질 작업 포함)

# 음영지역 비율이 이 이상이면 recorded_min을 분모로 대체
EWI_GAP_RELIABLE_THRESHOLD = 0.20

# ─── 공간 토큰 분류 ──────────────────────────────────────────────────
# v1 fallback 토큰 (LOCUS_VERSION="v1" 또는 locus CSV 로드 실패 시)
_V1_WORK_TOKENS = {
    "work_zone", "outdoor_work", "mechanical_room",
    "confined_space", "high_voltage", "transit",
}
_V1_TRANSIT_TOKENS = {"timeclock", "main_gate", "sub_gate"}
_V1_REST_TOKENS = {"breakroom", "smoking_area", "dining_hall", "restroom", "parking_lot"}
_V1_ADMIN_TOKENS = {"office", "facility"}

# v2 호환: locus CSV에서 토큰 분류 집합을 동적으로 로드
# sector_id별 캐시 — 섹터 변경 시 자동 업데이트
_cached_token_sets_by_sector: dict[str, dict[str, set[str]]] = {}


def _get_token_sets(sector_id: str | None = None) -> dict[str, set[str]]:
    """
    LOCUS_VERSION에 따라 토큰 분류 집합 반환.

    v1: 하드코딩된 영문 토큰 집합
    v2: locus_v2.csv의 locus_type/function/locus_name 기반 동적 분류

    Args:
        sector_id: 섹터 ID (None이면 cfg.SECTOR_ID 사용)

    Returns:
        {"work": set, "transit": set, "rest": set, "admin": set}
    """
    try:
        import config as cfg
    except ImportError:
        return _v1_token_sets()

    sid = sector_id or cfg.SECTOR_ID

    if sid in _cached_token_sets_by_sector:
        return _cached_token_sets_by_sector[sid]

    if cfg.LOCUS_VERSION != "v2":
        result = _v1_token_sets()
        _cached_token_sets_by_sector[sid] = result
        return result

    # v2: locus CSV 기반 동적 분류
    # ★ locus_token = locus_id (GW-XXX) 이므로, 토큰 집합도 locus_id 기준으로 구축
    try:
        paths = cfg.get_sector_paths(sid)
        csv_path = paths.get("locus_v2_csv")
        if not csv_path or not csv_path.exists():
            logger.warning("locus_v2.csv 없음, v1 fallback")
            result = _v1_token_sets()
            _cached_token_sets_by_sector[sid] = result
            return result

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # ★ locus_id 존재 확인 (v2 필수 컬럼)
        token_col = "locus_id" if "locus_id" in df.columns else "locus_name"
        name = df["locus_name"].fillna("").str.lower()
        ltype = df.get("locus_type", pd.Series(dtype=str)).fillna("").str.upper()
        func = df.get("function", pd.Series(dtype=str)).fillna("").str.upper()

        # ★ locus_v2.csv의 locus_type/function 기반 직접 분류
        # locus_type: WORK_AREA, GATE, REST_AREA, TRANSIT, VERTICAL
        # function:   WORK, ACCESS, REST, MOVE, VERTICAL

        # 1. REST: REST_AREA 또는 REST function
        rest_mask = (ltype == "REST_AREA") | (func == "REST")
        rest_tokens = set(df.loc[rest_mask, token_col].dropna())

        # 2. GATE/Transit: GATE + TRANSIT 타입 (출입구 + 이동공간)
        transit_mask = (ltype == "GATE") | (func == "ACCESS") | (ltype == "TRANSIT") | (func == "MOVE")
        transit_tokens = set(df.loc[transit_mask, token_col].dropna()) - rest_tokens

        # 3. ADMIN: 별도 지정 없으면 빈 집합 (향후 확장)
        admin_tokens = set()

        # 4. WORK: WORK_AREA/VERTICAL 중 REST/TRANSIT 아닌 것
        work_mask = ((ltype == "WORK_AREA") | (func == "WORK") | (ltype == "VERTICAL")) & ~rest_mask & ~transit_mask
        work_tokens = set(df.loc[work_mask, token_col].dropna())

        result = {
            "work": work_tokens,
            "transit": transit_tokens,
            "rest": rest_tokens,
            "admin": admin_tokens,
        }
        _cached_token_sets_by_sector[sid] = result
    except Exception as e:
        logger.warning("v2 토큰 분류 로드 실패: %s — v1 fallback", e)
        result = _v1_token_sets()
        _cached_token_sets_by_sector[sid] = result

    return _cached_token_sets_by_sector[sid]


def _v1_token_sets() -> dict[str, set[str]]:
    return {
        "work": _V1_WORK_TOKENS,
        "transit": _V1_TRANSIT_TOKENS,
        "rest": _V1_REST_TOKENS,
        "admin": _V1_ADMIN_TOKENS,
    }


# 동적 접근을 위한 래퍼 (기존 코드에서 WORK_TOKENS 등을 직접 참조하므로)
def _init_module_token_sets():
    ts = _get_token_sets()
    return ts["work"], ts["transit"], ts["rest"], ts["admin"]

# ★ 단일화 감사 L1: 모듈 레벨 상수는 최초 로드 시의 sector로 고정되는 "스냅샷"이다.
# 멀티 섹터(Y1 + M15X) 전환 시 계산이 틀어질 위험이 있으므로, EWI/CRE 계산 함수 내부에서는
# _resolve_token_sets(sector_id)를 호출해 동적으로 사용한다.
# 이 모듈 레벨 상수는 하위 호환 목적으로만 유지(외부에서 import하는 경우).
WORK_TOKENS, TRANSIT_TOKENS, REST_TOKENS, ADMIN_TOKENS = _init_module_token_sets()
NON_WORK_TOKENS = TRANSIT_TOKENS | REST_TOKENS | ADMIN_TOKENS


def _resolve_token_sets(sector_id: str | None = None):
    """
    sector별 동적 토큰 집합 반환.
    sector_id=None이면 현재 cfg.SECTOR_ID 기준으로 조회 (세션에서 sector 전환 후에도 안전).

    Returns: (work, transit, rest, admin, non_work) 5-tuple
    """
    ts = _get_token_sets(sector_id)
    work    = ts["work"]
    transit = ts["transit"]
    rest    = ts["rest"]
    admin   = ts["admin"]
    non_work = transit | rest | admin
    return work, transit, rest, admin, non_work

# ─── 공간별 정적 위험도 (Static Risk) ────────────────────────────────
# v1 영문 토큰 기반 (v2에서는 locus_dict 경로 우선 사용)
STATIC_RISK_BY_TOKEN: dict[str, float] = {
    # 최고 위험 (hazard_5)
    "confined_space": 2.0,
    "high_voltage":   2.0,
    # 고위험 (hazard_4)
    "mechanical_room": 1.8,
    "transit":         1.5,   # 호이스트·클라이머 (수직 이동 설비)
    # 중위험 (hazard_3)
    "outdoor_work": 1.3,
    "work_zone":    1.2,
    # 저위험 / 게이트
    "timeclock":  0.5,
    "main_gate":  0.3,
    "sub_gate":   0.3,
    # 비작업
    "breakroom":    0.2,
    "smoking_area": 0.2,
    "dining_hall":  0.2,
    "restroom":     0.2,
    "parking_lot":  0.2,
    "office":       0.3,
    "facility":     0.3,
    # ★ unmapped/unknown — 보수적 중간값
    "unknown":      0.5,
    "unmapped":     0.5,
}
_STATIC_RISK_MIN = 0.2
_STATIC_RISK_MAX = 2.0

# ─── CRE 가중치 ──────────────────────────────────────────────────────
CRE_W_PERSONAL = 0.45
CRE_W_STATIC   = 0.40
CRE_W_DYNAMIC  = 0.15

_DENSITY_SCALE  = 30.0   # 정규화 기준 (30명/슬롯 → dynamic_norm = 1.0)
_DENSITY_CAP    = 1.0    # 상한

# ─── Enriched Locus 기반 Static Risk 헬퍼 (2026-03-28 추가) ──────────

def _calc_locus_static_risk(
    locus_id: str,
    token: str,
    locus_dict: dict | None,
) -> float:
    """
    Locus별 static_risk 계산.

    우선순위:
    1. locus_dict에 hazard_level/hazard_grade 있으면 사용
    2. Fallback: token 기반 STATIC_RISK_BY_TOKEN

    Args:
        locus_id: Locus ID
        token: locus_token
        locus_dict: enriched locus 정보 (없으면 None)

    Returns:
        float: 0.2 ~ 2.0 범위의 static_risk
    """
    # Fallback: token 기반
    if locus_dict is None or not locus_id:
        return STATIC_RISK_BY_TOKEN.get(token, 1.0)

    info = locus_dict.get(locus_id, {})
    hazard_level = info.get("hazard_level")
    hazard_grade = info.get("hazard_grade")

    # hazard_level이 없으면 token fallback
    if not hazard_level:
        return STATIC_RISK_BY_TOKEN.get(token, 1.0)

    # hazard_level 문자열 정규화
    level_str = str(hazard_level).lower().strip()

    # hazard_grade 파싱
    try:
        grade = float(hazard_grade) if hazard_grade is not None else 2.0
    except (ValueError, TypeError):
        grade = 2.0

    # hazard_level 기반 기본값
    level_base = {
        "critical": 1.8,
        "high": 1.4,
        "medium": 1.0,
        "low": 0.6,
    }.get(level_str, 1.0)

    # grade 가중 (1~5 → x0.7~x1.1)
    grade_mult = 0.7 + (grade / 5) * 0.4

    return min(max(level_base * grade_mult, _STATIC_RISK_MIN), _STATIC_RISK_MAX)


# ─── EWI 계산 ────────────────────────────────────────────────────────

def calc_ewi_all_workers(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    locus_dict: dict | None = None,
    sector_id: str | None = None,
) -> pd.DataFrame:
    """
    전체 작업자 EWI (Effective Work Intensity) 벡터화 계산.

    EWI = (고활성×1.0 + 저활성×0.5 + 대기×0.2) / 근무시간(분)

    Args:
        journey_df : is_work_hour, locus_token, active_ratio, user_no 포함
        worker_df  : user_no, work_minutes 포함
        locus_dict : enriched locus 정보 (hazard_level/hazard_grade 기반 static_risk)
                    None이면 token 기반 fallback
        sector_id  : 섹터 ID (None이면 cfg.SECTOR_ID). 토큰 분류를 sector별로 정확히 적용

    Returns:
        DataFrame with columns:
            user_no, high_active_min, low_active_min, standby_min,
            rest_min, transit_min, recorded_work_min,
            ewi, ewi_reliable, gap_ratio, gap_min, static_risk
    """
    wdf = journey_df[journey_df["is_work_hour"]].copy()
    if wdf.empty:
        return _empty_ewi_df(worker_df)

    # ★ L1: 모듈 레벨 상수 대신 동적 lookup (멀티 섹터 전환 안전)
    _WORK, _TRANSIT, _REST, _ADMIN, _NON_WORK = _resolve_token_sets(sector_id)

    ratio      = wdf["active_ratio"].fillna(0)
    token      = wdf["locus_token"].fillna("work_zone")
    is_non_work = token.isin(_NON_WORK)
    is_transit  = token.isin(_TRANSIT)
    is_rest     = token.isin(_REST | _ADMIN)
    is_work     = ~is_non_work

    wdf["_high"]        = is_work & (ratio >= HIGH_ACTIVE_THRESHOLD)
    wdf["_low"]         = is_work & (ratio >= LOW_ACTIVE_THRESHOLD) & (ratio < HIGH_ACTIVE_THRESHOLD)
    wdf["_standby"]     = is_work & (ratio < LOW_ACTIVE_THRESHOLD)
    wdf["_rest"]        = is_rest
    wdf["_transit"]     = is_transit

    # ★ static_risk: enriched locus 기반 (locus_dict 있으면) 또는 token 기반
    if locus_dict and "locus_id" in wdf.columns:
        wdf["_static_risk"] = wdf.apply(
            lambda r: _calc_locus_static_risk(
                r.get("locus_id", ""),
                r.get("locus_token", "unknown"),
                locus_dict,
            ),
            axis=1,
        )
    else:
        wdf["_static_risk"] = token.map(STATIC_RISK_BY_TOKEN).fillna(1.0)

    agg = wdf.groupby("user_no").agg(
        high_active_min  = ("_high",        "sum"),
        low_active_min   = ("_low",         "sum"),
        standby_min      = ("_standby",     "sum"),
        rest_min         = ("_rest",        "sum"),
        transit_min      = ("_transit",     "sum"),
        recorded_work_min = ("user_no",     "count"),
        mean_static_risk = ("_static_risk", "mean"),
        p90_static_risk  = ("_static_risk", lambda x: float(np.percentile(x, 90))),
    ).reset_index()

    # work_minutes 병합 (AccessLog 기반 실제 근무 시간)
    agg = agg.merge(
        worker_df[["user_no", "work_minutes"]].dropna(subset=["work_minutes"]),
        on="user_no", how="left",
    )
    agg["work_minutes"] = agg["work_minutes"].fillna(agg["recorded_work_min"])

    # EWI 분자
    agg["_ewi_num"] = (
        agg["high_active_min"] * HIGH_WORK_WEIGHT
        + agg["low_active_min"] * LOW_WORK_WEIGHT
        + agg["standby_min"]   * STANDBY_WEIGHT
    )

    # 음영지역 (BLE 미수집 구간) 계산
    agg["gap_min"]  = (agg["work_minutes"] - agg["recorded_work_min"]).clip(lower=0)
    # ★ work_minutes=0인 작업자: gap_ratio=NaN (측정 불가) → 1.0 ("미측정")
    # fillna(0)이었으면 "커버리지 100%"로 오해, 1.0이면 "미측정" 등급 부여
    agg["gap_ratio"] = (
        agg["gap_min"] / agg["work_minutes"].replace(0, np.nan)
    ).fillna(1.0)
    # ★ v3.0.1 (2026-04-18) EWI 분모 정책 변경:
    #   생산성 = 생산 활동 분 / **공식 체류시간(work_minutes, 타각기 기준)**
    #   이전 로직(ewi_reliable=False 시 recorded_work_min으로 나눔)은 sparse
    #   BLE 데이터에서 분모 붕괴로 EWI 값이 비정상 부풀려지는 버그가 있었음.
    #   (예: 739분 체류 · BLE 2분 · 고활성 2분 → EWI 2/2 = 1.000 ❌)
    #   음영지역은 "생산 활동 확인 안 됨"으로 간주 — 음영 자체를 줄이는 방향으로
    #   해소해야 하며, 분모 트릭으로 보정하지 않는다.
    #   ewi_reliable 플래그는 UI 힌트(참고용 여부 표시)로만 유지.
    agg["ewi_reliable"] = agg["gap_ratio"] <= EWI_GAP_RELIABLE_THRESHOLD
    agg["ewi_denom"]    = agg["work_minutes"]

    # ★ EWI 계산 — work_minutes=0인 작업자는 측정 불가로 처리
    agg["ewi_calculable"] = agg["work_minutes"] > 0
    agg["ewi"] = (agg["_ewi_num"] / agg["ewi_denom"].replace(0, np.nan)).fillna(0).clip(0, 1).round(4)
    agg.loc[~agg["ewi_calculable"], "ewi"] = 0.0
    agg.loc[~agg["ewi_calculable"], "ewi_reliable"] = False

    # Static risk: 평균 70% + 90th percentile 30% (위험 공간 희석 방지)
    agg["static_risk"] = (0.7 * agg["mean_static_risk"] + 0.3 * agg["p90_static_risk"]).fillna(1.0)

    return agg[[
        "user_no", "high_active_min", "low_active_min", "standby_min",
        "rest_min", "transit_min", "recorded_work_min",
        "ewi", "ewi_reliable", "ewi_calculable", "gap_ratio", "gap_min", "static_risk",
    ]]


def _empty_ewi_df(worker_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["user_no", "high_active_min", "low_active_min", "standby_min",
            "rest_min", "transit_min", "recorded_work_min",
            "ewi", "ewi_reliable", "ewi_calculable", "gap_ratio", "gap_min", "static_risk"]
    df = pd.DataFrame(columns=cols)
    df["user_no"] = worker_df["user_no"]
    df = df.fillna(0)
    df["ewi_calculable"] = False
    return df


# ─── CRE 계산 ────────────────────────────────────────────────────────

def _calc_fatigue_scores(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별 피로도 점수 계산 (연속 활성 작업 기반).

    피로도 = 최대 연속 활성 시간(분) × 0.7 + 평균 연속 활성 시간(분) × 0.3
    → 연속 4시간+ 작업 시 fatigue_score ≥ 0.5 수준으로 설계

    Returns: DataFrame(user_no, fatigue_score, max_active_streak, avg_active_streak)
    """
    wdf = journey_df[journey_df["is_work_hour"]].copy()
    if wdf.empty:
        return pd.DataFrame(columns=["user_no", "fatigue_score", "max_active_streak", "avg_active_streak"])

    wdf = wdf.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
    wdf["_is_active"] = (wdf["active_ratio"].fillna(0) >= 0.30)

    # 연속 run 감지 (user 경계 또는 활성 상태 변화 시 새 run)
    prev_user    = wdf["user_no"].shift()
    prev_active  = wdf["_is_active"].shift()
    run_break    = (wdf["user_no"] != prev_user) | (wdf["_is_active"] != prev_active)
    wdf["_run_id"] = run_break.cumsum()

    run_info = wdf.groupby(["user_no", "_run_id"]).agg(
        run_len        = ("_run_id",    "count"),
        is_active_run  = ("_is_active", "first"),
    ).reset_index()

    active_runs = run_info[run_info["is_active_run"]]
    if active_runs.empty:
        return pd.DataFrame({
            "user_no":           wdf["user_no"].unique(),
            "fatigue_score":     0.0,
            "max_active_streak": 0,
            "avg_active_streak": 0.0,
        })

    streak_agg = active_runs.groupby("user_no").agg(
        max_active_streak = ("run_len", "max"),
        avg_active_streak = ("run_len", "mean"),
    ).reset_index()

    # 피로도 정규화: 4시간(240분) 연속 = fatigue_score 1.0 기준
    _FATIGUE_NORM = 240.0
    streak_agg["fatigue_score"] = (
        (0.7 * streak_agg["max_active_streak"] + 0.3 * streak_agg["avg_active_streak"])
        / _FATIGUE_NORM
    ).clip(0, 1).round(4)

    return streak_agg[["user_no", "fatigue_score", "max_active_streak", "avg_active_streak"]]


def _calc_dynamic_pressure(
    journey_df: pd.DataFrame,
    sector_id: str | None = None,
) -> pd.DataFrame:
    """
    작업자별 동적 밀집 압력 계산.

    동적 압력 = 작업 시간 중 해당 작업자가 머문 locus의 평균 동시 작업자 수 / DENSITY_SCALE

    Args:
        sector_id: 섹터 ID (None이면 cfg.SECTOR_ID). L1 멀티 섹터 안전 조치.

    Returns: DataFrame(user_no, dynamic_norm)
    """
    # ★ L1: 동적 토큰 lookup
    _WORK, *_ = _resolve_token_sets(sector_id)
    wdf = journey_df[journey_df["is_work_hour"] & journey_df["locus_token"].isin(_WORK)].copy()
    if wdf.empty or "locus_id" not in wdf.columns:
        return pd.DataFrame(columns=["user_no", "dynamic_norm"])

    # ★ Performance: transform → merge 제거 (메모리 + 속도 개선)
    wdf["_cnt"] = wdf.groupby(["timestamp", "locus_id"])["user_no"].transform("nunique")

    worker_density = (
        wdf.groupby("user_no")["_cnt"]
        .mean()
        .reset_index()
        .rename(columns={"_cnt": "_mean_density"})
    )
    worker_density["dynamic_norm"] = (
        worker_density["_mean_density"] / _DENSITY_SCALE
    ).clip(0, _DENSITY_CAP).round(4)

    return worker_density[["user_no", "dynamic_norm"]]


def calc_cre_all_workers(
    journey_df: pd.DataFrame,
    ewi_df: pd.DataFrame,
    sector_id: str | None = None,
) -> pd.DataFrame:
    """
    전체 작업자 CRE (Combined Risk Exposure) 계산.

    CRE = 0.45 × personal_norm + 0.40 × static_norm + 0.15 × dynamic_norm

    Personal  = 피로도 × 0.5 + 단독작업 근사 × 0.5
    Static    = locus_token 기반 고유 공간 위험도
    Dynamic   = 해당 작업 구역 동시 밀집 수준

    Args:
        journey_df : is_work_hour, locus_token, active_ratio, user_no, timestamp 포함
        ewi_df     : calc_ewi_all_workers() 결과 (static_risk 포함)
        sector_id  : 섹터 ID (L1 멀티 섹터 안전 조치)

    Returns:
        DataFrame(user_no, cre, personal_norm, static_norm, dynamic_norm,
                  fatigue_score, max_active_streak)
    """
    # ── 피로도 ────────────────────────────────────────────────────────
    fatigue_df = _calc_fatigue_scores(journey_df)

    # ── 단독 작업 근사 ────────────────────────────────────────────────
    # 고위험 구역(confined/high_voltage)에서 작업자 혼자인 시간 비율로 근사
    wdf_high = journey_df[
        journey_df["is_work_hour"]
        & journey_df["locus_token"].isin({"confined_space", "high_voltage", "mechanical_room"})
    ].copy()
    if not wdf_high.empty and "locus_id" in wdf_high.columns:
        coloc_hi = (
            wdf_high.groupby(["timestamp", "locus_id"])["user_no"]
            .nunique()
            .reset_index(name="_cnt")
        )
        merged_hi = wdf_high.merge(coloc_hi, on=["timestamp", "locus_id"], how="left")
        merged_hi["_alone"] = (merged_hi["_cnt"].fillna(1) <= 1).astype(int)
        alone_df = (
            merged_hi.groupby("user_no")["_alone"]
            .mean()
            .reset_index()
            .rename(columns={"_alone": "alone_ratio"})
        )
    else:
        alone_df = pd.DataFrame(columns=["user_no", "alone_ratio"])

    # ── 동적 압력 (sector_id 전파) ────────────────────────────────────
    dynamic_df = _calc_dynamic_pressure(journey_df, sector_id=sector_id)

    # ── 합산 ──────────────────────────────────────────────────────────
    cre_df = ewi_df[["user_no", "static_risk"]].copy()
    cre_df = cre_df.merge(fatigue_df, on="user_no", how="left")
    cre_df = cre_df.merge(alone_df,   on="user_no", how="left")
    cre_df = cre_df.merge(dynamic_df, on="user_no", how="left")

    cre_df["fatigue_score"] = cre_df["fatigue_score"].fillna(0)
    cre_df["alone_ratio"]   = cre_df["alone_ratio"].fillna(0)
    cre_df["dynamic_norm"]  = cre_df["dynamic_norm"].fillna(0)
    cre_df["max_active_streak"] = cre_df.get("max_active_streak", pd.Series(0, index=cre_df.index)).fillna(0)

    # Personal risk (0~1)
    cre_df["personal_norm"] = (
        0.5 * cre_df["fatigue_score"] + 0.5 * cre_df["alone_ratio"]
    ).clip(0, 1)

    # Static risk → normalize to 0~1
    cre_df["static_norm"] = (
        (cre_df["static_risk"] - _STATIC_RISK_MIN)
        / (_STATIC_RISK_MAX - _STATIC_RISK_MIN)
    ).clip(0, 1)

    # CRE
    cre_df["cre"] = (
        CRE_W_PERSONAL * cre_df["personal_norm"]
        + CRE_W_STATIC  * cre_df["static_norm"]
        + CRE_W_DYNAMIC * cre_df["dynamic_norm"]
    ).clip(0, 1).round(4)

    return cre_df[[
        "user_no", "cre", "personal_norm", "static_norm", "dynamic_norm",
        "fatigue_score", "max_active_streak", "alone_ratio",
    ]]


# ─── 전체 지표 통합 ───────────────────────────────────────────────────

def add_metrics_to_worker(
    journey_df: pd.DataFrame,
    worker_df:  pd.DataFrame,
    locus_dict: dict | None = None,
    sector_id:  str | None = None,
) -> pd.DataFrame:
    """
    worker_df에 EWI / CRE / SII 컬럼을 추가하여 반환.

    Args:
        journey_df: Journey 데이터
        worker_df: 작업자 데이터
        locus_dict: enriched locus 정보 (hazard_level/hazard_grade 기반 static_risk)
                   None이면 token 기반 fallback (하위 호환)
        sector_id: 섹터 ID — L1 멀티 섹터 안전 조치.
                   전파 경로: add_metrics_to_worker → calc_ewi / calc_cre → _resolve_token_sets

    신규 컬럼:
        high_active_min, low_active_min, standby_min,
        rest_min, transit_min, recorded_work_min,
        ewi, ewi_reliable, gap_ratio, gap_min,
        cre, personal_norm, static_norm, dynamic_norm,
        fatigue_score, max_active_streak, alone_ratio,
        sii  (= ewi × static_norm, 고강도+고위험 작업자 탐지)
    """
    ewi_df = calc_ewi_all_workers(journey_df, worker_df, locus_dict, sector_id=sector_id)
    cre_df = calc_cre_all_workers(journey_df, ewi_df, sector_id=sector_id)

    result = worker_df.merge(ewi_df, on="user_no", how="left")
    result = result.merge(
        cre_df.drop(columns=["static_norm"], errors="ignore"),
        on="user_no", how="left",
    )

    # SII: EWI × static_norm (고강도 작업 + 위험 공간 동시 해당 작업자 탐지)
    static_norm_col = ewi_df.merge(
        cre_df[["user_no", "static_norm"]], on="user_no", how="left"
    )
    result = result.merge(
        static_norm_col[["user_no", "static_norm"]], on="user_no", how="left"
    )
    result["sii"] = (result["ewi"] * result["static_norm"]).clip(0, 1).round(4)

    # 기본값 채우기
    for col in ["ewi", "cre", "sii", "fatigue_score", "gap_ratio"]:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    # ★ BLE 커버리지 등급 — 데이터 품질 투명성
    # gap_ratio: 0=완벽 커버리지, 1=완전 음영
    gr = result["gap_ratio"].fillna(1.0)
    result["ble_coverage"] = np.where(
        gr <= 0.2, "정상",
        np.where(gr <= 0.5, "부분음영",
                 np.where(gr <= 0.8, "음영", "미측정"))
    )
    result["ble_coverage_pct"] = ((1 - gr) * 100).clip(0, 100).round(1)

    return result


# ─── 지표 등급 판정 ───────────────────────────────────────────────────

def ewi_grade(v: float) -> str:
    if v >= 0.6: return "고강도"
    if v >= 0.2: return "보통"
    return "저강도"

def cre_grade(v: float) -> str:
    if v >= 0.6: return "고위험"
    if v >= 0.3: return "주의"
    return "정상"

def sii_grade(v: float) -> str:
    if v >= 0.5: return "집중관리"
    if v >= 0.25: return "주의"
    return "정상"
