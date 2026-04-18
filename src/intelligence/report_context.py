"""
PDF 리포트 맥락 생성기 - Report Context Generator v1
=====================================================
리포트의 각 섹션에 대해 3단 구조 맥락 텍스트를 자동 생성:
  1. 데이터 코멘트: 수치 요약 (1-2줄)
  2. 맥락 해석: 의미 해석 (2-3줄)
  3. 인사이트: 실행 권고 (1-2줄)

LLM 호출 시 anonymizer를 통해 데이터 익명화 적용.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH

logger = logging.getLogger(__name__)


# ─── 정적 맥락 텍스트 (LLM 미사용 시 기본값) ─────────────────────────────────

STATIC_CONTEXT = {
    "ewi_distribution": {
        "data": "이 차트는 작업자별 작업 집중도(EWI) 분포를 보여줍니다.",
        "context": "EWI는 0.4~0.7이 정상 범위입니다. 0.6 이상은 고강도 작업, 0.2 미만은 저강도로 분류됩니다.",
        "insight": None,  # 동적 생성
    },
    "cre_distribution": {
        "data": "이 차트는 위험 노출도(CRE) 분포를 나타냅니다.",
        "context": "CRE 0.6 이상 작업자는 즉시 관리가 필요합니다. 0.3 이상은 주의 대상입니다.",
        "insight": None,
    },
    "fatigue_vs_cre": {
        "data": "이 산점도는 피로도와 위험 노출의 복합 위험을 보여줍니다.",
        "context": "우측 상단(피로도/CRE 모두 높음)은 복합 고위험군으로, 즉각적 관심이 필요합니다.",
        "insight": None,
    },
    "daily_trend": {
        "data": "일별 출입인원 추이를 표시합니다.",
        "context": "T-Ward 착용률은 BLE 추적 커버리지를 나타냅니다. 70% 이상이 권장됩니다.",
        "insight": None,
    },
    "ewi_cre_trend": {
        "data": "일별 평균 EWI/CRE 추이입니다.",
        "context": "빨간 점선(0.6)은 고위험 임계선입니다. 이 선을 초과하는 날은 주의가 필요합니다.",
        "insight": None,
    },
    "company_risk": {
        "data": "업체별 평균 CRE를 비교합니다.",
        "context": "5명 이상 작업자가 있는 업체만 표시됩니다. 빨간색은 고위험, 초록색은 정상입니다.",
        "insight": None,
    },
    "high_risk_table": {
        "data": "CRE 0.6 이상 고위험 작업자 목록입니다.",
        "context": "밀폐공간 체류, 장시간 근무, 피로 누적이 주요 위험 요인입니다.",
        "insight": None,
    },
}


@dataclass
class SectionContext:
    """단일 섹션의 3단 맥락."""
    data_comment: str = ""      # 수치 요약 (1-2줄)
    context_text: str = ""      # 의미 해석 (2-3줄)
    insight_text: str = ""      # 인사이트 (1-2줄)
    severity: str = "normal"    # normal, warning, danger, success


@dataclass
class ReportContext:
    """
    리포트 전체의 맥락 데이터.

    각 섹션별로 SectionContext를 보유하며,
    LLM 또는 정적 텍스트로 초기화됨.
    """
    ewi_distribution: SectionContext = field(default_factory=SectionContext)
    cre_distribution: SectionContext = field(default_factory=SectionContext)
    fatigue_vs_cre: SectionContext = field(default_factory=SectionContext)
    daily_trend: SectionContext = field(default_factory=SectionContext)
    ewi_cre_trend: SectionContext = field(default_factory=SectionContext)
    company_risk: SectionContext = field(default_factory=SectionContext)
    high_risk_table: SectionContext = field(default_factory=SectionContext)
    summary: SectionContext = field(default_factory=SectionContext)


# ─── 맥락 생성 함수 ──────────────────────────────────────────────────────────


def _determine_severity(value: float, thresholds: tuple[float, float] = (0.3, 0.6)) -> str:
    """값에 따른 심각도 판정."""
    if value >= thresholds[1]:
        return "danger"
    elif value >= thresholds[0]:
        return "warning"
    return "success"


def _generate_ewi_context(
    avg_ewi: float,
    high_ewi_count: int,
    low_ewi_count: int,
    total_workers: int,
) -> SectionContext:
    """EWI 분포 맥락 생성."""
    ctx = SectionContext()

    # 데이터 코멘트
    high_pct = high_ewi_count / total_workers * 100 if total_workers > 0 else 0
    low_pct = low_ewi_count / total_workers * 100 if total_workers > 0 else 0
    ctx.data_comment = (
        f"평균 EWI {avg_ewi:.3f}, "
        f"고강도 {high_ewi_count}명({high_pct:.1f}%), "
        f"저강도 {low_ewi_count}명({low_pct:.1f}%)"
    )

    # 맥락 해석
    if avg_ewi >= 0.6:
        ctx.context_text = (
            "전반적으로 작업 강도가 높은 편입니다. "
            "이는 현장 생산성이 높거나 과부하 상태일 수 있습니다."
        )
        ctx.severity = "warning"
    elif avg_ewi >= 0.4:
        ctx.context_text = (
            "작업 강도가 정상 범위 내에 있습니다. "
            "균형 잡힌 작업 분배가 이루어지고 있습니다."
        )
        ctx.severity = "success"
    else:
        ctx.context_text = (
            "작업 강도가 낮은 편입니다. "
            "대기 시간이 많거나 작업 효율성 점검이 필요할 수 있습니다."
        )
        ctx.severity = "normal"

    # 인사이트
    if high_pct > 20:
        ctx.insight_text = f"고강도 작업자 비율({high_pct:.0f}%)이 높습니다. 피로 누적에 주의하세요."
        ctx.severity = "warning"
    elif low_pct > 30:
        ctx.insight_text = f"저강도 작업자 비율({low_pct:.0f}%)이 높습니다. 작업 배분을 확인하세요."
    else:
        ctx.insight_text = "작업 강도 분포가 양호합니다."

    return ctx


def _generate_cre_context(
    avg_cre: float,
    high_cre_count: int,
    total_workers: int,
) -> SectionContext:
    """CRE 분포 맥락 생성."""
    ctx = SectionContext()

    high_pct = high_cre_count / total_workers * 100 if total_workers > 0 else 0
    ctx.data_comment = f"평균 CRE {avg_cre:.3f}, 고위험(>={CRE_HIGH:.1f}) {high_cre_count}명({high_pct:.1f}%)"

    if avg_cre >= CRE_HIGH:
        ctx.context_text = (
            "전반적인 위험 노출 수준이 높습니다. "
            "밀폐공간, 고압전 구역 진입자 또는 장시간 근무자가 많을 수 있습니다."
        )
        ctx.severity = "danger"
    elif avg_cre >= 0.3:
        ctx.context_text = (
            "위험 노출 수준이 주의 구간에 있습니다. "
            "고위험 작업자에 대한 모니터링이 필요합니다."
        )
        ctx.severity = "warning"
    else:
        ctx.context_text = (
            "위험 노출 수준이 정상 범위 내에 있습니다. "
            "현재 안전 관리가 양호합니다."
        )
        ctx.severity = "success"

    if high_cre_count > 0:
        ctx.insight_text = f"고위험 작업자 {high_cre_count}명에 대한 즉각적 관리가 필요합니다."
        if ctx.severity == "success":
            ctx.severity = "warning"
    else:
        ctx.insight_text = "고위험 작업자가 없습니다. 양호한 상태입니다."

    return ctx


def _generate_fatigue_cre_context(
    both_high_count: int,
    high_fatigue_count: int,
    high_cre_count: int,
    total_workers: int,
) -> SectionContext:
    """피로도 vs CRE 산점도 맥락 생성."""
    ctx = SectionContext()

    ctx.data_comment = (
        f"복합 고위험(피로+CRE 모두 높음): {both_high_count}명, "
        f"고피로: {high_fatigue_count}명, 고CRE: {high_cre_count}명"
    )

    if both_high_count > 0:
        ctx.context_text = (
            f"{both_high_count}명이 피로도와 위험 노출 모두 높은 복합 위험 상태입니다. "
            "이들은 장시간 고위험 구역에서 근무했을 가능성이 높습니다."
        )
        ctx.severity = "danger"
        ctx.insight_text = f"복합 고위험 {both_high_count}명에 대한 즉각적 휴식 조치를 권고합니다."
    elif high_fatigue_count > 0 or high_cre_count > 0:
        ctx.context_text = (
            "피로도 또는 위험 노출이 높은 작업자가 있습니다. "
            "개별 요인에 대한 관리가 필요합니다."
        )
        ctx.severity = "warning"
        ctx.insight_text = "고피로/고위험 작업자에 대한 모니터링을 강화하세요."
    else:
        ctx.context_text = (
            "피로도와 위험 노출 모두 양호한 상태입니다. "
            "현재 작업 환경이 적절히 관리되고 있습니다."
        )
        ctx.severity = "success"
        ctx.insight_text = "작업 환경이 양호합니다."

    return ctx


def _generate_trend_context(
    avg_ewi_trend: list[float],
    avg_cre_trend: list[float],
    dates: list[str],
) -> SectionContext:
    """일별 EWI/CRE 트렌드 맥락 생성."""
    ctx = SectionContext()

    if not avg_ewi_trend or not avg_cre_trend:
        ctx.data_comment = "트렌드 데이터가 부족합니다."
        return ctx

    # 최근 값
    last_ewi = avg_ewi_trend[-1]
    last_cre = avg_cre_trend[-1]

    # 평균 대비 변화
    avg_ewi = sum(avg_ewi_trend) / len(avg_ewi_trend)
    avg_cre = sum(avg_cre_trend) / len(avg_cre_trend)

    ewi_change = (last_ewi - avg_ewi) / avg_ewi * 100 if avg_ewi > 0 else 0
    cre_change = (last_cre - avg_cre) / avg_cre * 100 if avg_cre > 0 else 0

    ctx.data_comment = (
        f"최근 EWI {last_ewi:.3f} (기간평균 대비 {ewi_change:+.1f}%), "
        f"CRE {last_cre:.3f} (기간평균 대비 {cre_change:+.1f}%)"
    )

    # 추세 판단
    if len(avg_cre_trend) >= 3:
        recent_cre = sum(avg_cre_trend[-3:]) / 3
        earlier_cre = sum(avg_cre_trend[:3]) / 3 if len(avg_cre_trend) >= 6 else avg_cre

        if recent_cre > earlier_cre * 1.1:
            ctx.context_text = "최근 CRE가 상승 추세입니다. 위험 요인이 증가하고 있을 수 있습니다."
            ctx.severity = "warning"
        elif recent_cre < earlier_cre * 0.9:
            ctx.context_text = "최근 CRE가 하락 추세입니다. 안전 관리가 개선되고 있습니다."
            ctx.severity = "success"
        else:
            ctx.context_text = "CRE가 안정적으로 유지되고 있습니다."
            ctx.severity = "normal"
    else:
        ctx.context_text = "충분한 트렌드 데이터가 없습니다."
        ctx.severity = "normal"

    # 임계 초과 일수 — CRE_HIGH 기준
    cre_danger_days = sum(1 for c in avg_cre_trend if c >= CRE_HIGH)
    if cre_danger_days > 0:
        ctx.insight_text = f"분석 기간 중 {cre_danger_days}일이 고위험 임계({CRE_HIGH:.1f})를 초과했습니다."
        if ctx.severity == "normal":
            ctx.severity = "warning"
    else:
        ctx.insight_text = "모든 날짜가 정상 범위 내에 있습니다."

    return ctx


def _generate_company_context(
    top_company: str,
    top_cre: float,
    high_risk_companies: int,
    total_companies: int,
) -> SectionContext:
    """업체별 위험 분석 맥락 생성."""
    ctx = SectionContext()

    ctx.data_comment = (
        f"분석 대상 {total_companies}개 업체 중 고위험(CRE>=0.5) {high_risk_companies}개"
    )

    if high_risk_companies > 0:
        ctx.context_text = (
            f"가장 높은 CRE를 보인 업체의 평균 CRE는 {top_cre:.3f}입니다. "
            f"해당 업체의 작업 환경 점검이 필요합니다."
        )
        ctx.severity = "warning" if top_cre >= 0.5 else "normal"
        ctx.insight_text = f"고위험 업체 {high_risk_companies}개에 대한 집중 관리를 권고합니다."
    else:
        ctx.context_text = "모든 업체의 평균 CRE가 양호한 범위 내에 있습니다."
        ctx.severity = "success"
        ctx.insight_text = "업체별 안전 현황이 양호합니다."

    return ctx


def _generate_summary_context(
    kpi: dict,
    high_cre_count: int,
    confined_count: int,
    total_workers: int,
) -> SectionContext:
    """전체 요약 맥락 생성."""
    ctx = SectionContext()

    avg_ewi = kpi.get("avg_ewi", 0)
    avg_cre = kpi.get("avg_cre", 0)
    tward_rate = kpi.get("tward_rate", 0)

    # 종합 판정
    issues = []
    if avg_cre >= 0.5:
        issues.append("평균 CRE 높음")
    if high_cre_count > total_workers * 0.1:
        issues.append(f"고위험 작업자 {high_cre_count}명")
    if confined_count > 0:
        issues.append(f"밀폐공간 진입 {confined_count}명")
    if tward_rate < 50:
        issues.append(f"T-Ward 착용률 {tward_rate:.0f}%")

    if issues:
        ctx.data_comment = "주요 이슈: " + ", ".join(issues)
        ctx.context_text = (
            f"이 리포트 기간 동안 {len(issues)}개의 주의 사항이 감지되었습니다. "
            "아래 상세 분석을 참고하여 개선 조치를 검토하세요."
        )
        ctx.severity = "warning" if len(issues) <= 2 else "danger"
        ctx.insight_text = "위 항목들에 대한 즉각적 검토를 권고합니다."
    else:
        ctx.data_comment = "주요 이슈 없음"
        ctx.context_text = (
            "이 리포트 기간 동안 특별한 이슈가 감지되지 않았습니다. "
            "현재 현장 관리 상태가 양호합니다."
        )
        ctx.severity = "success"
        ctx.insight_text = "현재 관리 상태를 유지하세요."

    return ctx


# ─── LLM 기반 맥락 생성 (선택적) ─────────────────────────────────────────────


def _call_llm_for_context(
    section_name: str,
    data_summary: str,
    static_context: str,
    *,
    sector_id: str = "Y1_SKHynix",
) -> str:
    """
    PDF 리포트 섹션용 맥락 텍스트 생성.

    ✅ V-10 해소 (M2-B T-21, 2026-04-18)
       과거 `_call_claude` 직접 호출 = AnonymizationPipeline 우회 경로였다.
       이제 core.ai.LLMGateway 단일 진입점으로 이관 →
         - AnonymizationPipeline K=20 강제 통과 (우회 불가)
         - 감사 로그 자동 기록 (data/audit/{sector}/YYYY-MM.jsonl)
         - 응답 PII 재검증
    """
    try:
        from core.ai import (
            CommentaryRequest,
            build_report_section_context,
            get_gateway,
        )

        gw = get_gateway()
        if not gw.is_available():
            return static_context

        ctx = build_report_section_context(
            sector_id=sector_id,
            section_name=section_name,
            data_summary=data_summary,
            static_context=static_context,
        )
        # productivity_analyst role 재사용 — 짧은 수치 해설에 적합.
        req = CommentaryRequest(
            role="productivity_analyst",
            sector_id=sector_id,
            date_str=None,
            context=ctx,
            user_role="administrator",
            tab="report_context",
            stream=False,
            max_tokens=350,
        )
        resp = gw.analyze(req)
        text = (resp.text or "").strip()
        if text and not text.startswith("["):
            return text
        return static_context

    except Exception as e:
        logger.warning(f"LLM 맥락 생성 실패 ({section_name}): {e}")
        return static_context


# ─── 메인 생성 함수 ──────────────────────────────────────────────────────────


def build_report_context(
    worker_df: pd.DataFrame,
    kpi: dict,
    dates: list[str] | None = None,
    use_llm: bool = False,
) -> ReportContext:
    """
    리포트 전체의 맥락 데이터 빌드.

    Args:
        worker_df: 작업자별 지표 DataFrame
        kpi: KPI 딕셔너리
        dates: 분석 날짜 목록
        use_llm: LLM 사용 여부 (기본: False)

    Returns:
        ReportContext 인스턴스
    """
    ctx = ReportContext()

    if worker_df is None or worker_df.empty:
        logger.warning("worker_df가 비어있어 맥락 생성을 건너뜁니다.")
        return ctx

    total_workers = len(worker_df)

    # EWI 분포 — constants.py의 EWI_HIGH 기준
    if "ewi" in worker_df.columns:
        avg_ewi = worker_df["ewi"].mean()
        high_ewi = int((worker_df["ewi"].fillna(0) >= EWI_HIGH).sum())
        low_ewi = int((worker_df["ewi"].fillna(1) < 0.2).sum())
        ctx.ewi_distribution = _generate_ewi_context(avg_ewi, high_ewi, low_ewi, total_workers)

    # CRE 분포 — constants.py의 CRE_HIGH 기준
    if "cre" in worker_df.columns:
        avg_cre = worker_df["cre"].mean()
        high_cre = int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum())
        ctx.cre_distribution = _generate_cre_context(avg_cre, high_cre, total_workers)

    # 피로도 vs CRE (fatigue 기준은 0.6 생리학 상수로 별도 유지)
    if "fatigue_score" in worker_df.columns and "cre" in worker_df.columns:
        high_fatigue = int((worker_df["fatigue_score"].fillna(0) >= 0.6).sum())
        high_cre = int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum())
        both_high = int(
            ((worker_df["fatigue_score"].fillna(0) >= 0.6) & (worker_df["cre"].fillna(0) >= CRE_HIGH)).sum()
        )
        ctx.fatigue_vs_cre = _generate_fatigue_cre_context(
            both_high, high_fatigue, high_cre, total_workers
        )

    # 트렌드 (날짜별)
    if dates and "date" in worker_df.columns:
        ewi_trend = []
        cre_trend = []
        for d in dates:
            day_df = worker_df[worker_df["date"] == d]
            if not day_df.empty:
                if "ewi" in day_df.columns:
                    ewi_trend.append(day_df["ewi"].mean())
                if "cre" in day_df.columns:
                    cre_trend.append(day_df["cre"].mean())
        if ewi_trend and cre_trend:
            ctx.ewi_cre_trend = _generate_trend_context(ewi_trend, cre_trend, dates)

    # 업체별 분석
    if "company_name" in worker_df.columns and "cre" in worker_df.columns:
        comp_agg = (
            worker_df.groupby("company_name")
            .agg(avg_cre=("cre", "mean"), count=("cre", "count"))
            .reset_index()
        )
        comp_agg = comp_agg[
            (comp_agg["count"] >= 5) & (comp_agg["company_name"] != "미확인")
        ]
        if not comp_agg.empty:
            top_row = comp_agg.nlargest(1, "avg_cre").iloc[0]
            high_risk_co = int((comp_agg["avg_cre"] >= 0.5).sum())
            ctx.company_risk = _generate_company_context(
                top_row["company_name"],
                top_row["avg_cre"],
                high_risk_co,
                len(comp_agg),
            )

    # 전체 요약
    high_cre_count = kpi.get("high_cre", 0)
    confined_count = 0
    if "confined_minutes" in worker_df.columns:
        confined_count = int((worker_df["confined_minutes"] > 0).sum())
    ctx.summary = _generate_summary_context(kpi, high_cre_count, confined_count, total_workers)

    return ctx


def get_context_label(ctx: SectionContext) -> str:
    """맥락 레이블 포맷팅 (PDF용)."""
    parts = []
    if ctx.data_comment:
        parts.append(ctx.data_comment)
    if ctx.context_text:
        parts.append(ctx.context_text)
    return " ".join(parts)


def get_insight_box(ctx: SectionContext) -> tuple[str, str]:
    """
    인사이트 박스 데이터 반환.

    Returns:
        (텍스트, severity) 튜플
    """
    return (ctx.insight_text or "", ctx.severity)
