"""
src.dashboard.integrity — 데이터 정합성 탭 패키지
=================================================
`integrity_tab.py` (4,668 LOC) 를 M3-A 리팩터링으로 분해한 모듈 집합.

공개 API:
    render_integrity_tab(sector_id)

구성:
    overview.py          — 메인 진입점 (AI 감사 + 5탭 라우팅)
    worker_review.py     — 🔎 작업자 상세 검토 (최대 서브탭)
    schema_check.py      — 📊 현장 보정 통계
    sanity_check.py      — ⚠️ 이상 패턴 + 🚨 비상식 패턴
    gap_analysis.py      — 📉 BLE 커버리지 이상
    physical_validator.py — 🚶 물리적 이동 검증 (worker_review 내부 위임)
    context.py           — SubTabContext + 캐시 로더
    helpers.py           — 상수 · 컬러 · 순수 헬퍼
"""
from src.dashboard.integrity.overview import render_integrity_tab

__all__ = ["render_integrity_tab"]
