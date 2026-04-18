"""
Insight 데이터 모델
====================
모든 인사이트 엔진의 공통 출력 형식.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Severity(IntEnum):
    """인사이트 심각도 (높을수록 긴급)."""
    LOW      = 1   # 정보성 — 참고할 만한 패턴
    MEDIUM   = 2   # 주의 — 모니터링 필요
    HIGH     = 3   # 경고 — 당일 조치 필요
    CRITICAL = 4   # 긴급 — 즉각 조치 필요 (안전)


# 심각도별 UI 표시
SEVERITY_BADGE = {
    Severity.CRITICAL: ("🔴", "긴급"),
    Severity.HIGH:     ("🟠", "경고"),
    Severity.MEDIUM:   ("🟡", "주의"),
    Severity.LOW:      ("🔵", "참고"),
}


CATEGORY_LABEL = {
    "safety":       ("🛡️", "안전"),
    "compliance":   ("📋", "규정"),
    "productivity": ("⚡", "생산성"),
    "movement":     ("🚶", "이동"),
    "space":        ("🏗️", "공간"),
    "trend":        ("📈", "트렌드"),
}


@dataclass
class Insight:
    """단일 인사이트."""
    category: str               # safety / compliance / productivity / movement / space / trend
    severity: Severity
    title: str                  # 1줄 한글 요약
    description: str            # 2-3문장 설명
    evidence: dict = field(default_factory=dict)       # 근거 수치
    affected: list = field(default_factory=list)        # 관련 작업자/공간/업체
    recommendation: str = ""    # 구체적 권고
    source: str = ""            # 생성 모듈 (anomaly / journey / trend)

    @property
    def badge(self) -> tuple[str, str]:
        return SEVERITY_BADGE.get(self.severity, ("⚪", "?"))

    @property
    def category_label(self) -> tuple[str, str]:
        return CATEGORY_LABEL.get(self.category, ("❓", self.category))


@dataclass
class InsightReport:
    """하루치 인사이트 집합."""
    date: str
    sector: str
    insights: list[Insight] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.insights if i.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.insights if i.severity == Severity.HIGH)

    def top(self, n: int = 5) -> list[Insight]:
        """심각도 순 상위 N개."""
        return sorted(self.insights, key=lambda i: -i.severity)[:n]

    def by_category(self, cat: str) -> list[Insight]:
        return [i for i in self.insights if i.category == cat]

    def summary_text(self, max_items: int = 5) -> str:
        """LLM 프롬프트용 텍스트 요약."""
        lines = []
        for i, ins in enumerate(self.top(max_items), 1):
            badge_icon, badge_label = ins.badge
            lines.append(
                f"{i}. [{badge_label}] {ins.title}\n"
                f"   {ins.description}\n"
                f"   권고: {ins.recommendation}"
            )
        return "\n\n".join(lines) if lines else "(특이사항 없음)"
