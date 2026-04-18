"""
Spatial Loader — 공간 정보 로딩 모듈
========================================
Locus, SSMP, spot_name→locus_id 매핑을 로드하여
파이프라인 전반에서 공간 정보를 제공한다.

모든 @st.cache_data 함수는 sector_id를 파라미터로 받아
Sector 전환 시 캐시가 올바르게 분리되도록 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config as cfg
from core.cache.policy import SPATIAL_MODEL, SUMMARY_INDEX


# ─── 경로 헬퍼 ────────────────────────────────────────────────────
def _paths(sector_id: str | None) -> dict:
    """sector_id → 공간 모델 경로 딕셔너리."""
    sid = sector_id if sector_id else cfg.SECTOR_ID
    return cfg.get_sector_paths(sid)


# ─── Locus 정보 ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_locus_df(sector_id: str | None = None) -> pd.DataFrame:
    """
    locus.csv 로드 -> Locus 정보 DataFrame 반환.

    LOCUS_VERSION에 따라 v1(locus.csv) 또는 v2(locus_v2.csv)를 로드한다.
    sector_id가 다르면 별도 캐시 엔트리 사용.
    """
    paths = _paths(sector_id)

    # LOCUS_VERSION에 따라 파일 선택
    if cfg.LOCUS_VERSION == "v2":
        v2_path = paths.get("locus_v2_csv")
        if v2_path and v2_path.exists():
            path = v2_path
        else:
            st.warning(f"locus_v2.csv 없음, v1으로 fallback: {v2_path}")
            path = paths["locus_csv"]
    else:
        path = paths["locus_csv"]
    if not path.exists():
        st.warning(f"locus.csv 없음: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["locus_id"] = df["locus_id"].astype(str).str.strip()

    # ★ hazard_level: tags에서 추출 (없으면 기본 3)
    if "hazard_level" not in df.columns and "tags" in df.columns:
        df["hazard_level"] = df["tags"].apply(_extract_hazard_level)

    # ★ lone_work_prohibited: tags에서 #no_lone 추출
    if "lone_work_prohibited" not in df.columns and "tags" in df.columns:
        df["lone_work_prohibited"] = df["tags"].fillna("").astype(str).str.contains("#no_lone", na=False)

    # ★ 신규 컬럼 디폴트값 처리 (하위 호환)
    defaults = {
        # 기존 컬럼
        "function": "unknown",
        "hazard_level": "medium",
        "capacity": None,
        "requires_supervision": False,
        "lone_work_prohibited": False,
        "max_continuous_min": None,
        "ventilation_type": "unknown",
        "building": "unknown",
        "floor": "unknown",
        # ★ 신규 14개 enriched 컬럼 (2026-03-28 추가)
        "dwell_category": "UNKNOWN",
        "avg_dwell_minutes": 0.0,
        "median_dwell_minutes": 0.0,
        "max_dwell_minutes": 0.0,
        "std_dwell_minutes": 0.0,
        "total_visits": 0,
        "avg_daily_visitors": 0.0,
        "total_unique_users": 0,
        "avg_visits_per_person": 0.0,
        "visit_frequency_rank": 0,
        "peak_hour": 0,
        "temporal_pattern": "uniform",
        "avg_concurrent_occupancy": 0.0,
        "max_concurrent_occupancy": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _extract_hazard_level(tags: str) -> int:
    """tags 문자열에서 hazard_level 추출."""
    import pandas as pd
    if pd.isna(tags) or not isinstance(tags, str):
        return 3  # 기본값
    if "#hazard_5" in tags:
        return 5
    if "#hazard_4" in tags:
        return 4
    if "#hazard_3" in tags:
        return 3
    if "#hazard_2" in tags:
        return 2
    return 1


@st.cache_data(show_spinner=False)
def load_locus_dict(sector_id: str | None = None) -> dict[str, dict]:
    """locus_id → dict(token, locus_name, locus_type, ...) 조회용."""
    df = load_locus_df(sector_id)
    if df.empty:
        return {}
    # ★ NaN → None 변환 (JSON 직렬화 호환 + UI "nan" 표시 방지)
    result = df.set_index("locus_id").to_dict("index")
    import math
    for lid, d in result.items():
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
    return result


# ─── spot_name → locus_id 매핑 ────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_spot_name_map(sector_id: str | None = None) -> dict[str, dict]:
    """
    spot_name_map.json 로드.
    반환: { spot_name: {locus_id, locus_token, note} }
    """
    path = _paths(sector_id)["spot_name_map"]
    if not path.exists():
        st.warning(f"spot_name_map.json 없음: {path}")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"spot_name_map.json 파싱 실패: {e}")
        return {}


def resolve_locus_id(
    spot_name: str,
    sector_id: str | None = None,
) -> tuple[str | None, str | None]:
    """
    spot_name → (locus_id, locus_token) 반환.
    매핑 없으면 (None, None).
    """
    if spot_name is None:
        return None, None
    spot_map = load_spot_name_map(sector_id)
    # ★ 정규화: strip 후 정확 매칭 → 실패 시 strip+lower 재시도
    cleaned = spot_name.strip()
    entry = spot_map.get(cleaned)
    if entry is None:
        # case-insensitive fallback
        lower = cleaned.lower()
        for key, val in spot_map.items():
            if key.strip().lower() == lower:
                entry = val
                break
    if entry is None:
        return None, None
    return entry.get("locus_id"), entry.get("locus_token")


# ─── Gateway (v2) 정보 ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_gateway_df(sector_id: str | None = None) -> pd.DataFrame:
    """
    GatewayPoint CSV 로드 (v2 S-Ward 좌표 데이터).

    Returns:
        213개 Gateway의 좌표 + 메타데이터 DataFrame.
        컬럼: gateway_no, gateway_name, building_no, building_name,
              floor_no, floor_name, location_x, location_y
    """
    paths = _paths(sector_id)
    gw_path = paths.get("gateway_csv")
    if gw_path is None or not gw_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(gw_path, encoding="euc-kr")
        return df
    except Exception as e:
        st.warning(f"Gateway CSV 로드 실패: {e}")
        return pd.DataFrame()


# ─── SSMP S-Ward 정보 ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_swards_df(sector_id: str | None = None) -> pd.DataFrame:
    """ssmp_swards.csv 로드 (active만)."""
    path = _paths(sector_id)["ssmp_dir"] / "ssmp_swards.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[df["status"] == "active"].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_adjacency_df(sector_id: str | None = None) -> pd.DataFrame:
    """locus_adjacency.csv 로드."""
    path = _paths(sector_id)["adjacency"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


# ─── 공간 요약 정보 ────────────────────────────────────────────────
@st.cache_data(ttl=SUMMARY_INDEX, show_spinner=False)
def get_spatial_summary(sector_id: str | None = None) -> dict:
    """공간 구조 요약 — 대시보드 사이드바 표시용. ★ 10분 캐시."""
    locus_df = load_locus_df(sector_id)
    sward_df = load_swards_df(sector_id)
    adj_df   = load_adjacency_df(sector_id)
    spot_map = load_spot_name_map(sector_id)

    # ★ dwell_category 분포 추가
    dwell_cats = {}
    if not locus_df.empty and "dwell_category" in locus_df.columns:
        dwell_cats = locus_df["dwell_category"].value_counts().to_dict()

    return {
        "locus_count":     len(locus_df),
        "sward_count":     len(sward_df),
        "adjacency_count": len(adj_df),
        "spot_map_count":  len(spot_map),
        "locus_types":     (locus_df["locus_type"].value_counts().to_dict()
                            if not locus_df.empty else {}),
        "dwell_categories": dwell_cats,
    }


# ─── Enriched Locus 헬퍼 함수 (2026-03-28 추가) ─────────────────────
def get_locus_enriched(locus_id: str, sector_id: str | None = None) -> dict:
    """
    단일 Locus의 enriched 정보 조회.

    Returns:
        dict with keys: locus_id, locus_name, token, dwell_category, hazard_level,
                       avg_dwell_minutes, max_concurrent_occupancy, peak_hour, etc.
    """
    locus_dict = load_locus_dict(sector_id)
    return locus_dict.get(locus_id, {})


def get_hazard_risk_score(locus_id: str, sector_id: str | None = None) -> float:
    """
    hazard_level + hazard_grade 기반 위험도 점수 (0~2 범위).
    metrics.py에서 CRE 계산 시 사용.

    hazard_level 매핑:
        critical → 1.8~2.0
        high     → 1.3~1.6
        medium   → 0.8~1.1
        low      → 0.4~0.6
    """
    info = get_locus_enriched(locus_id, sector_id)
    hazard_level = str(info.get("hazard_level", "medium")).lower()
    hazard_grade = info.get("hazard_grade")

    # hazard_grade 파싱 (None/NaN 처리)
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
    }.get(hazard_level, 1.0)

    # grade 가중 (1~5 → x0.7~x1.1)
    grade_mult = 0.7 + (grade / 5) * 0.4

    return round(min(level_base * grade_mult, 2.0), 2)


def get_dwell_category_stats(sector_id: str | None = None) -> dict:
    """
    dwell_category별 Locus 통계 반환.

    Returns:
        {
            "TRANSIT": {"count": 10, "loci": ["GW-351", ...], "avg_dwell": 5.2},
            "LONG_STAY": {...},
            ...
        }
    """
    locus_df = load_locus_df(sector_id)
    if locus_df.empty:
        return {}

    result = {}
    for cat in locus_df["dwell_category"].unique():
        cat_df = locus_df[locus_df["dwell_category"] == cat]
        result[cat] = {
            "count": len(cat_df),
            "loci": cat_df["locus_id"].tolist(),
            "avg_dwell": cat_df["avg_dwell_minutes"].mean() if "avg_dwell_minutes" in cat_df.columns else 0,
        }
    return result
