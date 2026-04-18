"""
건설현장 도메인 규칙 — 알림/판정/분류 기준
=========================================
src/pipeline/metrics.py, src/pipeline/lone_work.py, src/dashboard/components.py에서
실제 사용하는 임계값과 판정 로직을 규칙으로 분리.

이 모듈은 핵심 알고리즘 로직을 포함하지 않으며,
오직 임계값과 판정 기준만 정의한다.

사용법:
    from domain_packs.construction.rules import (
        ALERT_RULES,
        EWI_GRADES,
        CRE_GRADES,
        evaluate_worker_alerts,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ─── 심각도 열거형 ──────────────────────────────────────────────────────────

class Severity(str, Enum):
    """알림 심각도 등급."""
    CRITICAL = "critical"  # 즉시 조치 필요
    HIGH = "high"          # 높은 우선순위
    MEDIUM = "medium"      # 주의 필요
    LOW = "low"            # 참고 사항


# ─── 알림 규칙 정의 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AlertRule:
    """알림 규칙 정의."""
    name: str
    condition: str          # 자연어 설명
    metric: str             # 평가 대상 메트릭
    threshold: float        # 임계값
    severity: Severity
    message_template: str

    def evaluate(self, value: float) -> bool:
        """주어진 값이 임계값을 초과하는지 평가."""
        return value >= self.threshold


# 출처: src/pipeline/metrics.py (ewi_grade, cre_grade, sii_grade)
# 출처: src/dashboard/components.py (build_space_alerts)

ALERT_RULES: list[AlertRule] = [
    # CRE 알림
    AlertRule(
        name="critical_cre",
        condition="CRE가 0.8 이상 (즉시 조치 필요)",
        metric="cre",
        threshold=0.8,
        severity=Severity.CRITICAL,
        message_template="CRE {value:.3f} - 즉시 조치 필요",
    ),
    AlertRule(
        name="high_cre",
        condition="CRE가 0.6 이상 (고위험)",
        metric="cre",
        threshold=0.6,
        severity=Severity.HIGH,
        message_template="CRE {value:.3f} - 고위험 상태",
    ),
    AlertRule(
        name="warning_cre",
        condition="CRE가 0.3 이상 (주의)",
        metric="cre",
        threshold=0.3,
        severity=Severity.MEDIUM,
        message_template="CRE {value:.3f} - 주의 필요",
    ),

    # EWI 알림
    AlertRule(
        name="high_ewi",
        condition="EWI가 0.8 이상 (과로 위험)",
        metric="ewi",
        threshold=0.8,
        severity=Severity.HIGH,
        message_template="EWI {value:.3f} - 과로 위험",
    ),
    AlertRule(
        name="warning_ewi",
        condition="EWI가 0.6 이상 (고강도)",
        metric="ewi",
        threshold=0.6,
        severity=Severity.MEDIUM,
        message_template="EWI {value:.3f} - 고강도 작업",
    ),

    # SII 알림
    AlertRule(
        name="critical_sii",
        condition="SII가 0.5 이상 (집중관리)",
        metric="sii",
        threshold=0.5,
        severity=Severity.CRITICAL,
        message_template="SII {value:.3f} - 집중관리 대상",
    ),
    AlertRule(
        name="warning_sii",
        condition="SII가 0.25 이상 (주의)",
        metric="sii",
        threshold=0.25,
        severity=Severity.MEDIUM,
        message_template="SII {value:.3f} - 주의 필요",
    ),

    # 피로도 알림
    AlertRule(
        name="high_fatigue",
        condition="피로도가 0.8 이상 (과로)",
        metric="fatigue_score",
        threshold=0.8,
        severity=Severity.HIGH,
        message_template="피로도 {value:.3f} - 과로 위험",
    ),
    AlertRule(
        name="warning_fatigue",
        condition="피로도가 0.5 이상 (주의)",
        metric="fatigue_score",
        threshold=0.5,
        severity=Severity.MEDIUM,
        message_template="피로도 {value:.3f} - 주의 필요",
    ),
]


# ─── 등급 판정 규칙 ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GradeRule:
    """등급 판정 규칙."""
    grade: str
    threshold: float    # 이 값 이상이면 해당 등급
    color: str          # UI 표시 색상
    label_ko: str       # 한국어 라벨


# EWI 등급 (출처: metrics.py ewi_grade lines 498-501)
EWI_GRADES: list[GradeRule] = [
    GradeRule(grade="high", threshold=0.6, color="#FF4C4C", label_ko="고강도"),
    GradeRule(grade="normal", threshold=0.2, color="#FFB300", label_ko="보통"),
    GradeRule(grade="low", threshold=0.0, color="#00C897", label_ko="저강도"),
]


# CRE 등급 (출처: metrics.py cre_grade lines 503-506)
CRE_GRADES: list[GradeRule] = [
    GradeRule(grade="high", threshold=0.6, color="#FF4C4C", label_ko="고위험"),
    GradeRule(grade="warning", threshold=0.3, color="#FFB300", label_ko="주의"),
    GradeRule(grade="normal", threshold=0.0, color="#00C897", label_ko="정상"),
]


# SII 등급 (출처: metrics.py sii_grade lines 508-511)
SII_GRADES: list[GradeRule] = [
    GradeRule(grade="critical", threshold=0.5, color="#FF4C4C", label_ko="집중관리"),
    GradeRule(grade="warning", threshold=0.25, color="#FFB300", label_ko="주의"),
    GradeRule(grade="normal", threshold=0.0, color="#00C897", label_ko="정상"),
]


def get_grade(value: float, grades: list[GradeRule]) -> GradeRule:
    """
    값에 해당하는 등급 반환.

    Args:
        value: 평가할 값
        grades: 등급 규칙 목록 (threshold 내림차순으로 정렬되어 있어야 함)

    Returns:
        해당 등급 규칙
    """
    for grade in grades:
        if value >= grade.threshold:
            return grade
    return grades[-1]


# ─── 단독작업 판정 규칙 ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class LoneWorkRule:
    """단독작업 판정 규칙."""
    status: str
    min_duration_minutes: int
    severity: Severity
    color: str
    label_ko: str


# 출처: lone_work.py _status lines 105-110
LONE_WORK_RULES: list[LoneWorkRule] = [
    LoneWorkRule(
        status="critical",
        min_duration_minutes=30,
        severity=Severity.CRITICAL,
        color="#FF4C4C",
        label_ko="즉시 조치",
    ),
    LoneWorkRule(
        status="warning",
        min_duration_minutes=10,
        severity=Severity.HIGH,
        color="#FF8C00",
        label_ko="주의",
    ),
    LoneWorkRule(
        status="normal",
        min_duration_minutes=0,
        severity=Severity.LOW,
        color="#00C897",
        label_ko="정상",
    ),
]


def get_lone_work_status(duration_minutes: int) -> LoneWorkRule:
    """
    단독작업 시간에 따른 상태 판정.

    Args:
        duration_minutes: 단독작업 지속 시간 (분)

    Returns:
        단독작업 판정 규칙
    """
    for rule in LONE_WORK_RULES:
        if duration_minutes >= rule.min_duration_minutes:
            return rule
    return LONE_WORK_RULES[-1]


# ─── 공간 혼잡도 규칙 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CongestionRule:
    """공간 혼잡도 판정 규칙."""
    status: str
    threshold_pct: float    # 최대 수용 인원 대비 비율 (%)
    severity: Severity
    color: str
    label_ko: str


# 출처: components.py build_space_alerts lines 705-726
CONGESTION_RULES: list[CongestionRule] = [
    CongestionRule(
        status="overcrowded",
        threshold_pct=90.0,
        severity=Severity.HIGH,
        color="#FF4C4C",
        label_ko="과밀",
    ),
    CongestionRule(
        status="crowded",
        threshold_pct=70.0,
        severity=Severity.MEDIUM,
        color="#FF6B35",
        label_ko="혼잡",
    ),
    CongestionRule(
        status="normal",
        threshold_pct=0.0,
        severity=Severity.LOW,
        color="#00C897",
        label_ko="정상",
    ),
]


def get_congestion_status(occupancy_pct: float) -> CongestionRule:
    """
    공간 점유율에 따른 혼잡도 상태 판정.

    Args:
        occupancy_pct: 점유율 (%)

    Returns:
        혼잡도 판정 규칙
    """
    for rule in CONGESTION_RULES:
        if occupancy_pct >= rule.threshold_pct:
            return rule
    return CONGESTION_RULES[-1]


# ─── 작업자 알림 평가 함수 ──────────────────────────────────────────────────

def evaluate_worker_alerts(worker_metrics: dict[str, Any]) -> list[dict]:
    """
    작업자 메트릭에 대해 모든 알림 규칙 평가.

    Args:
        worker_metrics: 작업자 메트릭 딕셔너리
            - cre: float
            - ewi: float
            - sii: float (optional)
            - fatigue_score: float (optional)

    Returns:
        발생한 알림 목록
        [
            {
                "rule_name": str,
                "severity": str,
                "message": str,
                "metric": str,
                "value": float,
            },
            ...
        ]
    """
    alerts = []

    for rule in ALERT_RULES:
        value = worker_metrics.get(rule.metric)
        if value is None:
            continue

        if rule.evaluate(value):
            alerts.append({
                "rule_name": rule.name,
                "severity": rule.severity.value,
                "message": rule.message_template.format(value=value),
                "metric": rule.metric,
                "value": value,
            })

    # 심각도별 정렬 (critical > high > medium > low)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(alerts, key=lambda x: severity_order.get(x["severity"], 3))


# ─── BLE 커버리지 등급 ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoverageRule:
    """BLE 커버리지 등급 규칙."""
    status: str
    max_gap_ratio: float    # gap_ratio 상한 (이 값 이하)
    label_ko: str


# 출처: metrics.py add_metrics_to_worker lines 485-490
COVERAGE_RULES: list[CoverageRule] = [
    CoverageRule(status="normal", max_gap_ratio=0.2, label_ko="정상"),
    CoverageRule(status="partial", max_gap_ratio=0.5, label_ko="부분음영"),
    CoverageRule(status="shadow", max_gap_ratio=0.8, label_ko="음영"),
    CoverageRule(status="unmeasured", max_gap_ratio=1.0, label_ko="미측정"),
]


def get_coverage_status(gap_ratio: float) -> CoverageRule:
    """
    gap_ratio에 따른 BLE 커버리지 상태 판정.

    Args:
        gap_ratio: 음영 구간 비율 (0.0 ~ 1.0)

    Returns:
        커버리지 판정 규칙
    """
    for rule in COVERAGE_RULES:
        if gap_ratio <= rule.max_gap_ratio:
            return rule
    return COVERAGE_RULES[-1]
