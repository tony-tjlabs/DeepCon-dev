"""
DeepCon 메트릭 상수 — 전체 대시보드에서 단일 소스.
==================================================
모든 임계값·기준값을 여기 모아 **한 곳**에서 튜닝.
탭 코드에는 절대 매직넘버를 두지 않는다.
"""
from __future__ import annotations


# ─── 작업시간 목표값 ──────────────────────────────────────────────
TIME_TARGETS = {
    "mat_target_min":       30,     # 최초 도착시간(Movement Arrival Time) 30분 이내 목표
    "eod_target_min":       30,     # 마지막 이탈시간(End of Day) 30분 이내 목표
    "confined_minutes_min": 60,     # 이 값 이상 머물면 "좁은구역 장시간 체류"로 플래그
    "hv_minutes_min":       60,     # 고전압 구역 장시간 체류 플래그
}


# ─── 위험 판정 임계값 (전사 통일) ─────────────────────────────────
# 이 값들이 바뀌면 safety_tab·productivity_tab·경고 로직이 동시에 변경됨.
# ★ 반드시 이 상수를 import 하여 사용. 숫자 하드코딩 금지.
RISK_THRESHOLDS = {
    "sii_high":     0.5,     # SII ≥ 0.5 → 고위험
    "sii_medium":   0.3,
    "cre_critical": 0.5,     # CRE ≥ 0.5 → 매우 심각 (Alert Center Critical)
    "cre_high":     0.5,     # CRE ≥ 0.5 → 고위험
    "cre_medium":   0.3,
    "ewi_low":      0.3,     # EWI < 0.3 → 저활성 (주의)
    "ewi_high":     0.6,     # EWI ≥ 0.6 → 고강도 (daily/productivity의 고강도 라인)
}

# 편의 상수 (하드코딩 방지용 짧은 별칭)
CRE_HIGH     = RISK_THRESHOLDS["cre_high"]      # 0.5
CRE_CRITICAL = RISK_THRESHOLDS["cre_critical"]  # 0.5
CRE_MEDIUM   = RISK_THRESHOLDS["cre_medium"]    # 0.3
SII_HIGH     = RISK_THRESHOLDS["sii_high"]      # 0.5
EWI_HIGH     = RISK_THRESHOLDS["ewi_high"]      # 0.6
EWI_LOW      = RISK_THRESHOLDS["ewi_low"]       # 0.3


# ─── 업체별 집계 최소 인원 (단일 진실 공급원) ─────────────────────
# 업체 목록을 표시하기 위한 최소 인원 기준. 용도별 3 프로파일.
COMPANY_MIN_WORKERS = {
    "general":     5,   # 개요·전체 요약 — 적당한 노출 (너무 작은 업체 제외)
    "statistical": 10,  # 생산성/안전성 업체 비교 — 평균값의 통계적 신뢰성 확보
    "small":       3,   # 작업시간 분포·일별 차트 — 소규모 업체도 포함
}


# ─── 혼잡도 등급 (config.CONGESTION_GRADE_THRESHOLDS와 동일 값으로 유지) ──
# config.py의 것을 그대로 재노출 (import 순환 방지용)
CONGESTION_GRADES = {
    "과밀": 1.0,
    "혼잡": 0.8,
    "보통": 0.6,
    # 그 외 = "여유"
}


# ─── 작업시간 분류 카테고리 순서 + 색상 키 ──────────────────────
TIME_BREAKDOWN_CATEGORIES = ["work_zone", "transit", "rest", "gap"]
TIME_BREAKDOWN_LABELS = {
    "work_zone": "작업공간",
    "transit":   "이동",
    "rest":      "휴게",
    "gap":       "BLE 음영",
}


# ─── 정합성 자체 검증 허용치 ───────────────────────────────────
# 같은 지표 두 곳에서 읽었을 때 허용 오차. 0.01 = 1%p.
CONSISTENCY_TOLERANCE = {
    "ratio_pct":       0.01,   # 백분율 표기 차이 허용 (0.01 %p)
    "ratio_fraction":  0.0001, # 0~1 분수 표기 허용
    "minutes":         0.5,    # 분 단위 반올림 허용
    "count":           0,      # 정수는 정확해야 함
}


# ─── 공개 API 딕셔너리 (탭에서 import) ──────────────────────────
__all__ = [
    "TIME_TARGETS",
    "RISK_THRESHOLDS",
    "CRE_HIGH", "CRE_CRITICAL", "CRE_MEDIUM",
    "SII_HIGH", "EWI_HIGH", "EWI_LOW",
    "COMPANY_MIN_WORKERS",
    "CONGESTION_GRADES",
    "TIME_BREAKDOWN_CATEGORIES",
    "TIME_BREAKDOWN_LABELS",
    "CONSISTENCY_TOLERANCE",
]
