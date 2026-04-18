"""
인사이트 통합기 — Insight Aggregator
=====================================
모든 탐지기(anomaly, journey, trend)의 결과를 통합하여
심각도 순 정렬 + 최대 N개 핵심만 추출.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st

from src.intelligence.models import Insight, InsightReport, Severity
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH

logger = logging.getLogger(__name__)

# 카테고리별 우선순위 (같은 severity 내 정렬)
_CATEGORY_PRIORITY = {
    "safety": 0,
    "compliance": 1,
    "productivity": 2,
    "movement": 3,
    "space": 4,
    "trend": 5,
}

MAX_INSIGHTS = 10


def run_insight_pipeline(
    results: dict,
    date_str: str,
    sector_id: str,
    processed_dates: list[str] | None = None,
) -> InsightReport:
    """
    전체 인사이트 파이프라인 실행.

    1. baseline 계산 (과거 데이터 있으면)
    2. 이상 탐지
    3. 이동 패턴 분석 (구현 시)
    4. 트렌드 예측 (구현 시)
    5. 통합·순위화

    Args:
        results: load_daily_results() 반환값
        date_str: 분석 날짜
        sector_id: Sector ID
        processed_dates: 처리 완료 날짜 목록 (baseline용)
    """
    all_insights: list[Insight] = []

    # ── 1. Baseline 계산 ─────────────────────────────────────────
    baseline = None
    if processed_dates and len(processed_dates) >= 3:
        try:
            from src.intelligence.anomaly_detector import compute_baseline
            baseline = compute_baseline(processed_dates, date_str, sector_id)
        except Exception as e:
            logger.warning(f"Baseline 계산 실패: {e}")

    # ── 2. 이상 탐지 ─────────────────────────────────────────────
    try:
        from src.intelligence.anomaly_detector import detect_anomalies
        anomalies = detect_anomalies(results, baseline)
        all_insights.extend(anomalies)
        logger.info(f"이상 탐지: {len(anomalies)}건")
    except Exception as e:
        logger.error(f"이상 탐지 실패: {e}")

    # ── 3. 이동 패턴 분석 (Sprint 3에서 구현) ────────────────────
    try:
        from src.intelligence.journey_analyzer import analyze_journeys
        journey_insights = analyze_journeys(results)
        all_insights.extend(journey_insights)
        logger.info(f"이동 패턴 분석: {len(journey_insights)}건")
    except ImportError:
        pass  # 아직 미구현
    except Exception as e:
        logger.warning(f"이동 패턴 분석 실패: {e}")

    # ── 4. 트렌드 예측 (Sprint 3에서 구현) ───────────────────────
    try:
        from src.intelligence.trend_forecaster import forecast_trends
        trend_insights = forecast_trends(processed_dates, date_str, sector_id)
        all_insights.extend(trend_insights)
        logger.info(f"트렌드 예측: {len(trend_insights)}건")
    except ImportError:
        pass  # 아직 미구현
    except Exception as e:
        logger.warning(f"트렌드 예측 실패: {e}")

    # ── 5. Agentic AI 인사이트 (Deep Space 모델 있을 때) ────────
    try:
        from src.intelligence.spatial_predictor import agentic_to_insights
        agentic_report = results.get("agentic_report")
        if agentic_report is not None:
            agentic_insights = agentic_to_insights(agentic_report)
            all_insights.extend(agentic_insights)
            logger.info(f"Agentic AI 인사이트: {len(agentic_insights)}건")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Agentic 인사이트 변환 실패: {e}")

    # ── 6. 통합·순위화 ──────────────────────────────────────────
    ranked = _rank_insights(all_insights)

    report = InsightReport(
        date=date_str,
        sector=sector_id or "",
        insights=ranked[:MAX_INSIGHTS],
    )
    logger.info(
        f"인사이트 리포트: {len(report.insights)}건 "
        f"(🔴{report.critical_count} 🟠{report.high_count})"
    )
    return report


def _rank_insights(insights: list[Insight]) -> list[Insight]:
    """심각도 → 카테고리 우선순위 순 정렬."""
    return sorted(
        insights,
        key=lambda i: (-i.severity, _CATEGORY_PRIORITY.get(i.category, 99)),
    )


def build_kpi_summary(results: dict) -> str:
    """인사이트 LLM 프롬프트용 KPI 요약 텍스트."""
    meta = results.get("meta", {})
    worker_df = results.get("worker", pd.DataFrame())

    lines = []
    lines.append(f"- 총 출입인원: {meta.get('total_workers_access', 0):,}명")
    lines.append(f"- T-Ward 착용 작업자: {meta.get('total_workers_move', 0):,}명")

    if not worker_df.empty:
        if "ewi" in worker_df.columns:
            avg_ewi = worker_df["ewi"].mean()
            high_ewi = len(worker_df[worker_df["ewi"].fillna(0) >= EWI_HIGH])
            lines.append(f"- 평균 EWI(생산성): {avg_ewi:.3f} / 고강도(≥{EWI_HIGH:.1f}): {high_ewi}명")

        if "cre" in worker_df.columns:
            avg_cre = worker_df["cre"].mean()
            high_cre = len(worker_df[worker_df["cre"].fillna(0) >= CRE_HIGH])
            lines.append(f"- 평균 CRE(위험노출): {avg_cre:.3f} / 고위험(≥{CRE_HIGH:.1f}): {high_cre}명")

        if "confined_minutes" in worker_df.columns:
            confined = len(worker_df[worker_df["confined_minutes"] > 0])
            lines.append(f"- 밀폐공간 진입: {confined}명")

        if "fatigue_score" in worker_df.columns:
            high_fatigue = len(worker_df[worker_df["fatigue_score"] >= 0.6])
            lines.append(f"- 고피로(≥0.6): {high_fatigue}명")

    lines.append(f"- 참여 업체: {meta.get('companies', 0)}개")
    return "\n".join(lines)
