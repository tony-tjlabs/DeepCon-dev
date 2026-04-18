"""
DeepCon 표준 필터 — 작업자 DataFrame을 탭 간 동일한 방식으로 필터링.
================================================================
여러 탭이 각자 다른 조건으로 필터를 걸면 숫자가 어긋난다.
모든 필터는 이 모듈에서만 정의하고, 탭은 호출만 한다.

예시:
    from src.metrics.filters import apply_default, keep_has_tward

    wdf = apply_default(worker_df)            # 표준 집계용
    wdf = keep_has_tward(wdf)                 # T-Ward 착용자만
"""
from __future__ import annotations

import pandas as pd


# ─── 단일 필터 (모두 원본을 수정하지 않고 새 DataFrame 반환) ─────

def keep_has_tward(wdf: pd.DataFrame) -> pd.DataFrame:
    """T-Ward 착용자만 (has_tward=True)."""
    if "has_tward" not in wdf.columns:
        return wdf
    return wdf[wdf["has_tward"] == True].copy()


def keep_ewi_reliable(wdf: pd.DataFrame) -> pd.DataFrame:
    """EWI 신뢰 가능한 작업자만.

    ewi_reliable 컬럼이 없으면 모두 신뢰한다고 간주.
    """
    if "ewi_reliable" not in wdf.columns:
        return wdf.copy()
    return wdf[wdf["ewi_reliable"] == True].copy()


def keep_positive_work_minutes(wdf: pd.DataFrame) -> pd.DataFrame:
    """work_minutes > 0 (실제 체류 기록이 있는 작업자)."""
    if "work_minutes" not in wdf.columns:
        return wdf.copy()
    return wdf[wdf["work_minutes"].fillna(0) > 0].copy()


def keep_not_missing_exit(wdf: pd.DataFrame) -> pd.DataFrame:
    """퇴장 기록 이상치 제외 (missing_exit=False)."""
    if "missing_exit" not in wdf.columns:
        return wdf.copy()
    return wdf[wdf["missing_exit"] != True].copy()


def exclude_unidentified(wdf: pd.DataFrame) -> pd.DataFrame:
    """미확인 업체 제외."""
    if "company_name" not in wdf.columns:
        return wdf.copy()
    return wdf[~wdf["company_name"].isin(["미확인", "", None])].copy()


# ─── 표준 프로파일 (탭에서 가장 자주 쓰는 조합) ────────────────

def apply_default(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    표준 집계용 기본 필터.
    - work_minutes > 0
    - missing_exit 제외

    의도: "실제로 현장에 있었던 작업자"만 포함.
    가장 관대한 필터 — 모든 탭의 기본 분모.
    """
    return keep_not_missing_exit(keep_positive_work_minutes(wdf))


def apply_tward_reliable(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    T-Ward 착용 + EWI 신뢰 + 표준 필터.
    생산성·안전성 랭킹 등 '정확한 지표가 필요한 장면'에서 사용.
    """
    return keep_ewi_reliable(keep_has_tward(apply_default(wdf)))


# ─── 컬럼 안전 획득 ───────────────────────────────────────────

def safe_sum(wdf: pd.DataFrame, col: str) -> float:
    """컬럼이 없으면 0, 있으면 NaN을 0으로 처리해 합산."""
    if col not in wdf.columns:
        return 0.0
    return float(wdf[col].fillna(0).sum())


def safe_mean(wdf: pd.DataFrame, col: str) -> float:
    if col not in wdf.columns:
        return 0.0
    s = wdf[col].dropna()
    return float(s.mean()) if len(s) else 0.0


__all__ = [
    "keep_has_tward", "keep_ewi_reliable", "keep_positive_work_minutes",
    "keep_not_missing_exit", "exclude_unidentified",
    "apply_default", "apply_tward_reliable",
    "safe_sum", "safe_mean",
]
