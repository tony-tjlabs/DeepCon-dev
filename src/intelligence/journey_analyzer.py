"""
이동 패턴 분석기 — Journey Analyzer
=====================================
journey.parquet의 이동 데이터에서 의미 있는 패턴과 이상을 추출.

분석 항목:
  1. 위험 구역 장시간 체류
  2. 이동 효율 (transit vs work 비율)
  3. 공간 사용 집중도 (소수 공간만 반복 방문)
"""
from __future__ import annotations

import logging

import pandas as pd

from src.intelligence.models import Insight, Severity

logger = logging.getLogger(__name__)

# 임계값
DWELL_HAZARD_MIN    = 120   # 위험 구역 연속 체류 기준 (분)
TRANSIT_RATIO_HIGH  = 0.30  # 이동 시간 비율 경고 (30%)
LOW_LOCI_THRESHOLD  = 2     # 방문 공간이 극히 적은 기준


def analyze_journeys(results: dict) -> list[Insight]:
    """마스터 이동 패턴 분석."""
    worker_df  = results.get("worker", pd.DataFrame())
    journey_df = results.get("journey", pd.DataFrame())
    insights = []

    if not worker_df.empty:
        insights.extend(_detect_transit_inefficiency(worker_df))
        insights.extend(_detect_low_mobility(worker_df))

    if not journey_df.empty:
        insights.extend(_detect_hazard_dwell(journey_df))

    return insights


def _detect_transit_inefficiency(worker_df: pd.DataFrame) -> list[Insight]:
    """이동 시간이 전체 근무 대비 과도한 작업자 탐지."""
    required = {"transit_min", "work_minutes"}
    if not required.issubset(worker_df.columns):
        return []

    df = worker_df[worker_df["work_minutes"] > 60].copy()  # 1시간 이상 근무자만
    if df.empty:
        return []

    df["transit_ratio"] = (df["transit_min"] / df["work_minutes"].clip(lower=1)).clip(upper=1.0)
    high_transit = df[df["transit_ratio"] >= TRANSIT_RATIO_HIGH]
    n = len(high_transit)

    if n < 5:  # 소수는 개인 이슈
        return []

    avg_ratio = high_transit["transit_ratio"].mean()
    top_companies = high_transit["company_name"].value_counts().head(3)
    company_str = ", ".join(f"{c}({n}명)" for c, n in top_companies.items())

    return [Insight(
        category="movement",
        severity=Severity.MEDIUM,
        title=f"과다 이동 작업자 {n}명 (이동 비율 ≥{TRANSIT_RATIO_HIGH*100:.0f}%)",
        description=f"전체 근무 시간 중 이동 시간이 {avg_ratio*100:.0f}% 이상인 작업자가 {n}명입니다. "
                    f"주요 업체: {company_str}. 비효율적 동선이나 작업 배치 문제일 수 있습니다.",
        evidence={"count": n, "avg_transit_ratio": round(avg_ratio, 3),
                  "top_companies": dict(top_companies)},
        affected=high_transit["user_no"].tolist()[:10],
        recommendation="해당 업체 작업 구역과 자재 동선 최적화 검토",
        source="journey_analyzer",
    )]


def _detect_low_mobility(worker_df: pd.DataFrame) -> list[Insight]:
    """극히 적은 공간만 방문 (고정 작업 vs 고립 의심)."""
    if "unique_loci" not in worker_df.columns or "work_minutes" not in worker_df.columns:
        return []

    long_work = worker_df[worker_df["work_minutes"] >= 360]  # 6시간 이상
    if long_work.empty:
        return []

    fixed = long_work[long_work["unique_loci"] <= LOW_LOCI_THRESHOLD]
    n = len(fixed)
    if n < 10:
        return []

    # 고위험 공간에 고정된 작업자만 의미 있음
    if "cre" in fixed.columns:
        high_risk_fixed = fixed[fixed["cre"] >= 0.4]
        if len(high_risk_fixed) >= 3:
            return [Insight(
                category="movement",
                severity=Severity.MEDIUM,
                title=f"고정 작업 + 높은 위험도 작업자 {len(high_risk_fixed)}명",
                description=f"6시간 이상 근무하면서 {LOW_LOCI_THRESHOLD}개 이하 공간만 방문한 작업자 중 "
                            f"{len(high_risk_fixed)}명이 CRE ≥ 0.4입니다. "
                            f"위험 구역에 장시간 고정 배치된 것일 수 있습니다.",
                evidence={"total_fixed": n, "high_risk_fixed": len(high_risk_fixed)},
                affected=high_risk_fixed["user_no"].tolist()[:10],
                recommendation="해당 작업자 교대 순환 배치 또는 작업 환경 점검",
                source="journey_analyzer",
            )]

    return []


def _detect_hazard_dwell(journey_df: pd.DataFrame) -> list[Insight]:
    """위험 구역(is_confined / is_high_voltage) 장시간 연속 체류."""
    # journey_df에서 밀폐공간 판단이 어려우면 locus_token 기반
    if "locus_token" not in journey_df.columns or "user_no" not in journey_df.columns:
        return []

    # ★ 밀폐공간 토큰 — metrics.py STATIC_RISK_BY_TOKEN 정의와 일치
    confined_tokens = {"confined_space"}
    journey_df_work = journey_df[journey_df.get("is_work_hour", pd.Series(True))]
    if journey_df_work.empty:
        return []

    # locus_token이 밀폐공간인 레코드
    mask = journey_df_work["locus_token"].str.lower().isin(confined_tokens)
    confined_records = journey_df_work[mask]
    if confined_records.empty:
        return []

    # 작업자별 연속 체류 시간 (1분 단위이므로 레코드 수 = 분)
    per_worker = confined_records.groupby("user_no").size().reset_index(name="confined_min")
    long_dwell = per_worker[per_worker["confined_min"] >= DWELL_HAZARD_MIN]
    n = len(long_dwell)

    if n == 0:
        return []

    max_row = long_dwell.nlargest(1, "confined_min").iloc[0]

    return [Insight(
        category="safety",
        severity=Severity.HIGH if int(max_row["confined_min"]) >= 180 else Severity.MEDIUM,
        title=f"밀폐공간 연속 체류 {n}명 (최대 {int(max_row['confined_min'])}분)",
        description=f"밀폐공간에 {DWELL_HAZARD_MIN}분 이상 연속 체류한 작업자가 {n}명입니다. "
                    f"장시간 밀폐공간 작업은 산소 부족·유해가스 노출 위험이 급격히 증가합니다.",
        evidence={"count": n, "max_minutes": int(max_row["confined_min"])},
        affected=long_dwell["user_no"].tolist()[:10],
        recommendation="밀폐공간 환기 상태 점검 및 교대 작업 시행",
        source="journey_analyzer",
    )]
