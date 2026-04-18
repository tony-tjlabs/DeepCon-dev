"""
DeepCon 일별 집계 엔진 — 단일 진실 소스 (Single Source of Truth).
================================================================
worker.parquet / space.parquet / company.parquet / summary_index.json 에서
탭이 화면에 표시할 모든 일별 집계치를 계산하는 유일한 모듈.

설계 원칙
  1. 탭은 이 모듈의 함수만 호출한다. 탭 내부에서 sum/mean/groupby 금지.
  2. 모든 함수는 @st.cache_data로 캐싱되어 여러 탭에서 중복 호출해도 한 번만 계산.
  3. 반환값은 dataclass 또는 TypedDict — 탭에서 키 오타로 실패하지 않도록.
  4. 모든 지표는 "근거 단위(분/명/건)"와 "표시 단위(%)"를 함께 반환.
  5. 정규화가 필요한 경우(합계 100%) 반드시 `*_display` suffix 컬럼으로 명시.

대표 API
  - get_time_breakdown(sector, date)   → TimeBreakdown
  - get_company_ranking(sector, date)  → DataFrame (업체별 랭킹)
  - get_risk_summary(sector, date)     → RiskSummary
  - get_daily_kpi(sector, date)        → DailyKPI (카드용 요약)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config as cfg
from core.cache.policy import DAILY_PARQUET
from src.metrics.constants import (
    TIME_BREAKDOWN_CATEGORIES, TIME_BREAKDOWN_LABELS,
    RISK_THRESHOLDS, TIME_TARGETS,
)
from src.metrics.filters import apply_default, apply_tward_reliable, safe_sum, safe_mean


# ═══════════════════════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TimeBreakdown:
    """작업시간 분류 — 일별 전체 합계.

    raw ratio (work_minutes 분모) + display ratio (100% 정규화) 두 가지를 함께 제공.
    - 카드·KPI·로직 판정: raw_ratio 사용
    - 도넛·스택바처럼 합 100% 가정 시각화: display_ratio 사용
    """
    date_str:      str
    n_workers:     int
    work_minutes:  float                          # 총 체류 분 (분모)
    minutes:       dict[str, float] = field(default_factory=dict)   # 카테고리별 분
    raw_ratio:     dict[str, float] = field(default_factory=dict)   # work_minutes 분모 %
    display_ratio: dict[str, float] = field(default_factory=dict)   # 합=100 정규화 %
    sum_raw_pct:   float = 0.0                    # raw_ratio 총합 (>100이면 정의 중첩)
    consistency_warning: str = ""

    def minutes_for(self, category: str) -> float:
        return self.minutes.get(category, 0.0)

    def raw_pct_for(self, category: str) -> float:
        return self.raw_ratio.get(category, 0.0)

    def display_pct_for(self, category: str) -> float:
        return self.display_ratio.get(category, 0.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskSummary:
    """일별 위험 요약."""
    date_str:          str
    n_workers:         int
    n_has_tward:       int
    high_sii_count:    int
    high_cre_count:    int
    high_risk_count:   int            # sii_high OR cre_high (union)
    high_risk_rate:    float          # high_risk_count / n_has_tward
    confined_count:    int
    hv_count:          int
    ewi_reliable_count: int


@dataclass(frozen=True)
class DailyKPI:
    """현장 개요용 핵심 KPI (카드 한 줄)."""
    date_str:              str
    n_workers_access:      int        # access_log 기반 출입 인원
    n_workers_tward:       int        # T-Ward 착용자
    tward_wear_pct:        float      # = n_tward / n_access * 100
    avg_work_minutes:      float
    avg_ewi:               float
    avg_cre:               float
    avg_sii:               float
    time_breakdown:        TimeBreakdown


# ═══════════════════════════════════════════════════════════════════
# 내부 로더 (캐싱 가능)
# ═══════════════════════════════════════════════════════════════════

def _worker_path(sector_id: str, date_str: str) -> str:
    return str(cfg.PROCESSED_DIR / sector_id / date_str / "worker.parquet")


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _load_worker(sector_id: str, date_str: str, path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


# ═══════════════════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def get_time_breakdown(sector_id: str, date_str: str) -> TimeBreakdown:
    """일별 작업시간 분류 집계 — 전 탭 공용.

    주의: worker.parquet 정의상 work_zone + transit + rest + gap > work_minutes 가능.
    따라서 raw_ratio 총합이 100%를 초과할 수 있음 → display_ratio로 정규화 제공.
    """
    wdf = _load_worker(sector_id, date_str, _worker_path(sector_id, date_str))
    wdf = apply_default(wdf)

    if wdf.empty:
        return TimeBreakdown(date_str=date_str, n_workers=0, work_minutes=0.0)

    total_work = safe_sum(wdf, "work_minutes")
    if total_work <= 0:
        return TimeBreakdown(date_str=date_str, n_workers=len(wdf), work_minutes=0.0)

    minutes = {
        "work_zone": safe_sum(wdf, "work_zone_minutes"),
        "transit":   safe_sum(wdf, "transit_min"),
        "rest":      safe_sum(wdf, "rest_min"),
        "gap":       safe_sum(wdf, "gap_min"),
    }
    raw_ratio = {k: (v / total_work * 100.0) for k, v in minutes.items()}
    sum_raw   = sum(raw_ratio.values())

    # display_ratio: 합=100으로 정규화 (도넛·스택바용)
    if sum_raw > 0:
        display_ratio = {k: v / sum_raw * 100.0 for k, v in raw_ratio.items()}
    else:
        display_ratio = {k: 0.0 for k in raw_ratio}

    warn = ""
    if sum_raw > 100.5:
        warn = (f"정의 중첩: work_zone+transit+rest+gap = {sum_raw:.1f}%. "
                "gap_min 과대추정으로 총합이 100%를 초과 — display_ratio(정규화)로 도넛 표시 권장.")

    return TimeBreakdown(
        date_str=date_str,
        n_workers=len(wdf),
        work_minutes=total_work,
        minutes=minutes,
        raw_ratio=raw_ratio,
        display_ratio=display_ratio,
        sum_raw_pct=sum_raw,
        consistency_warning=warn,
    )


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def get_risk_summary(sector_id: str, date_str: str) -> RiskSummary:
    """일별 위험 요약 — safety_tab 및 overview에서 공용."""
    wdf = _load_worker(sector_id, date_str, _worker_path(sector_id, date_str))
    wdf = apply_default(wdf)
    if wdf.empty:
        return RiskSummary(date_str=date_str, n_workers=0, n_has_tward=0,
                           high_sii_count=0, high_cre_count=0,
                           high_risk_count=0, high_risk_rate=0.0,
                           confined_count=0, hv_count=0, ewi_reliable_count=0)

    n_workers   = len(wdf)
    n_tward     = int(wdf["has_tward"].sum()) if "has_tward" in wdf.columns else n_workers

    sii = wdf.get("sii", pd.Series(dtype=float)).fillna(0)
    cre = wdf.get("cre", pd.Series(dtype=float)).fillna(0)
    conf = wdf.get("confined_minutes", pd.Series(dtype=float)).fillna(0)
    hv   = wdf.get("high_voltage_minutes", pd.Series(dtype=float)).fillna(0)

    high_sii = int((sii >= RISK_THRESHOLDS["sii_high"]).sum())
    high_cre = int((cre >= RISK_THRESHOLDS["cre_high"]).sum())

    # union: sii_high OR cre_high
    if "has_tward" in wdf.columns:
        mask = (sii >= RISK_THRESHOLDS["sii_high"]) | (cre >= RISK_THRESHOLDS["cre_high"])
        high_risk = int(mask.sum())
    else:
        high_risk = max(high_sii, high_cre)

    denom_risk = n_tward if n_tward > 0 else max(n_workers, 1)
    high_risk_rate = high_risk / denom_risk * 100.0

    confined_count = int((conf >= TIME_TARGETS["confined_minutes_min"]).sum())
    hv_count       = int((hv   >= TIME_TARGETS["hv_minutes_min"]).sum())
    ewi_rel = int(wdf["ewi_reliable"].sum()) if "ewi_reliable" in wdf.columns else n_workers

    return RiskSummary(
        date_str=date_str, n_workers=n_workers, n_has_tward=n_tward,
        high_sii_count=high_sii, high_cre_count=high_cre,
        high_risk_count=high_risk, high_risk_rate=high_risk_rate,
        confined_count=confined_count, hv_count=hv_count,
        ewi_reliable_count=ewi_rel,
    )


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def get_company_ranking(
    sector_id: str,
    date_str: str,
    metric: str = "ewi",
    reliable_only: bool = True,
) -> pd.DataFrame:
    """
    업체별 평균 지표 랭킹 — productivity·overview 공용.

    Args:
        metric: "ewi" / "cre" / "sii" / "work_zone_minutes"
        reliable_only: True면 ewi_reliable=True + has_tward=True 필터

    반환 컬럼: company_name, metric_value, n_workers
    """
    wdf = _load_worker(sector_id, date_str, _worker_path(sector_id, date_str))
    if wdf.empty or metric not in wdf.columns:
        return pd.DataFrame(columns=["company_name", "metric_value", "n_workers"])

    wdf = apply_tward_reliable(wdf) if reliable_only else apply_default(wdf)
    if wdf.empty:
        return pd.DataFrame(columns=["company_name", "metric_value", "n_workers"])

    grp = (wdf.groupby("company_name")
             .agg(metric_value=(metric, "mean"),
                  n_workers=("user_no", "nunique"))
             .reset_index()
             .sort_values("metric_value", ascending=False))
    return grp


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def get_daily_kpi(sector_id: str, date_str: str) -> DailyKPI:
    """현장 개요용 한 줄 KPI."""
    wdf = _load_worker(sector_id, date_str, _worker_path(sector_id, date_str))
    wdf_def = apply_default(wdf)

    n_access = int(wdf_def["user_no"].nunique()) if "user_no" in wdf_def.columns else 0
    n_tward  = int(wdf_def["has_tward"].sum())    if "has_tward" in wdf_def.columns else 0
    wear_pct = (n_tward / n_access * 100.0) if n_access else 0.0

    tdf = apply_tward_reliable(wdf)   # 평균 지표는 신뢰 가능한 표본
    avg_work = safe_mean(tdf, "work_minutes")
    avg_ewi  = safe_mean(tdf, "ewi")
    avg_cre  = safe_mean(tdf, "cre")
    avg_sii  = safe_mean(tdf, "sii")

    tb = get_time_breakdown(sector_id, date_str)

    return DailyKPI(
        date_str=date_str,
        n_workers_access=n_access, n_workers_tward=n_tward, tward_wear_pct=wear_pct,
        avg_work_minutes=avg_work, avg_ewi=avg_ewi, avg_cre=avg_cre, avg_sii=avg_sii,
        time_breakdown=tb,
    )


__all__ = [
    "TimeBreakdown", "RiskSummary", "DailyKPI",
    "get_time_breakdown", "get_risk_summary",
    "get_company_ranking", "get_daily_kpi",
]
