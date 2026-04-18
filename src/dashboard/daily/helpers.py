"""
Daily Tab Helpers
=================
등급 판정 함수 + 공용 로더.
"""
from __future__ import annotations


def ewi_grade(v: float) -> str:
    """EWI 등급 판정."""
    if v >= 0.6:
        return "고강도"
    if v >= 0.2:
        return "보통"
    return "저강도"


def cre_grade(v: float) -> str:
    """CRE 등급 판정."""
    if v >= 0.6:
        return "고위험"
    if v >= 0.3:
        return "주의"
    return "정상"


def sii_grade(v: float) -> str:
    """SII 등급 판정."""
    if v >= 0.5:
        return "집중관리"
    if v >= 0.25:
        return "주의"
    return "정상"
