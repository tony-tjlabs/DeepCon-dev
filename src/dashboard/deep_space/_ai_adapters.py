"""
Deep Space AI 어댑터 — M2-B T-20
==================================
기존 `cached_spatial_insight` / `cached_prediction_insight` / `cached_anomaly_insight`
(dashboard/llm_deepcon.py) 를 `core.ai.LLMGateway` 로 이관한 얇은 래퍼.

- 기존 인터페이스(문자열 인자, 문자열 반환) 는 유지 → 호출처 변경 최소화.
- 내부적으로 LLMGateway.analyze() 사용 (익명화 강제, 감사 로그 기록).
- session_state 기반 세션 캐시 (탭 재실행 시 토큰 재소비 방지).
- API 키 미설정 시 빈 문자열 반환 — 기존 동작과 동일.
"""
from __future__ import annotations

import hashlib
import logging

import streamlit as st

from core.ai import (
    CommentaryRequest,
    build_anomaly_context,
    build_deep_space_context,
    build_prediction_context,
    get_gateway,
)
from src.dashboard.auth import get_current_user

log = logging.getLogger(__name__)


def _cache_key(prefix: str, *parts: str) -> str:
    joined = "|".join(str(p or "") for p in parts)
    h = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    return f"ai_ds_{prefix}_{h}"


def _run(role: str, sector_id: str, tab: str, context: dict, *, max_tokens: int = 400) -> str:
    """LLMGateway 동기 호출 — 실패/키 없음 시 빈 문자열 반환."""
    gw = get_gateway()
    if not gw.is_available():
        return ""
    user_role = get_current_user().get("role", "unknown")
    req = CommentaryRequest(
        role=role,
        sector_id=sector_id,
        date_str=None,
        context=context,
        user_role=user_role,
        tab=tab,
        stream=False,
        max_tokens=max_tokens,
    )
    try:
        resp = gw.analyze(req)
        return resp.text or ""
    except Exception as e:  # pragma: no cover
        log.warning("[deep_space AI] %s failed: %s", role, e)
        return ""


# ─── Public API (기존 이름 유지) ────────────────────────────

def cached_spatial_insight(
    summary: str,
    congested_spaces: str,
    locus_context: str,
    *,
    sector_id: str = "Y1_SKHynix",
    tab: str = "deep_space",
) -> str:
    """
    현장 시뮬레이션 결과 해석 — 기존 shim.
    role="deep_space_agent" 로 LLMGateway 위임.
    """
    key = _cache_key("spatial", sector_id, summary, congested_spaces, locus_context)
    if key in st.session_state:
        return st.session_state[key]

    ctx = build_deep_space_context(
        sector_id=sector_id,
        summary=summary,
        congested_spaces=congested_spaces,
        locus_context=locus_context,
    )
    text = _run("deep_space_agent", sector_id, tab, ctx, max_tokens=400)
    st.session_state[key] = text
    return text


def cached_prediction_insight(
    current_locus: str,
    predictions: str,
    locus_context: str,
    *,
    sector_id: str = "Y1_SKHynix",
    tab: str = "deep_space",
) -> str:
    """
    이동 예측 결과 해석 — 기존 shim.
    role="prediction_explainer" 로 LLMGateway 위임.
    """
    key = _cache_key("prediction", sector_id, current_locus, predictions, locus_context)
    if key in st.session_state:
        return st.session_state[key]

    ctx = build_prediction_context(
        sector_id=sector_id,
        current_locus=current_locus,
        predictions=predictions,
        locus_context=locus_context,
    )
    text = _run("prediction_explainer", sector_id, tab, ctx, max_tokens=500)
    st.session_state[key] = text
    return text


def cached_anomaly_insight(
    anomaly_description: str,
    perplexity: str,
    locus_context: str,
    *,
    sector_id: str = "Y1_SKHynix",
    tab: str = "deep_space",
) -> str:
    """
    이상 이동 해석 — 기존 shim.
    role="anomaly_reporter" 로 LLMGateway 위임.
    """
    key = _cache_key("anomaly", sector_id, anomaly_description, perplexity, locus_context)
    if key in st.session_state:
        return st.session_state[key]

    ctx = build_anomaly_context(
        sector_id=sector_id,
        anomaly_description=anomaly_description,
        perplexity=perplexity,
        locus_context=locus_context,
    )
    text = _run("anomaly_reporter", sector_id, tab, ctx, max_tokens=450)
    st.session_state[key] = text
    return text


__all__ = [
    "cached_spatial_insight",
    "cached_prediction_insight",
    "cached_anomaly_insight",
]
