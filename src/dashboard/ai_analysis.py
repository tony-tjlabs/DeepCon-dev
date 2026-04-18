"""
AI Analysis — 탭별 AI 분석 렌더링 컴포넌트 (Y1 전용)
=====================================================
각 탭에서 호출하는 공통 AI 분석 UI 컴포넌트.
LLMGateway + DataPackager + ContextBuilder 통합.

사용법:
    from src.dashboard.ai_analysis import render_ai_analysis
    render_ai_analysis("daily", worker_df=worker_df, journey_df=journey_df, date_str=date_str)

탭 ID:
    - "daily"       : 일별 작업자 분석 (EWI/CRE/SII)
    - "congestion"  : 공간 혼잡도 분석
    - "weekly"      : 주간 트렌드 분석
    - "deep_space"  : Deep Space 예측 해석

보안 원칙:
    - 업체명 → Company_A/B 세션 코드 치환 (AnonymizationPipeline)
    - 작업자명/ID 완전 차단
    - 날짜 → "분석 N일차(요일)" 상대화
    - k-Anonymity K=20 (Y1 기준)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ─── 결과 파싱 + 렌더링 ──────────────────────────────────────────────────────

def _parse_result(result_text: str) -> dict[str, str]:
    """
    [WHAT]/[WHY]/[NOTE] 3단 구조 파싱.

    Returns: {"what": ..., "why": ..., "note": ...}
    """
    sections: dict[str, str] = {"what": "", "why": "", "note": ""}
    current: str | None = None
    buf: list[str] = []

    for line in result_text.splitlines():
        stripped = line.strip()
        lower    = stripped.lower()
        if lower.startswith("[what]"):
            if current and buf:
                sections[current] = " ".join(buf).strip()
            current = "what"
            buf     = [stripped[6:].strip()] if len(stripped) > 6 else []
        elif lower.startswith("[why]"):
            if current and buf:
                sections[current] = " ".join(buf).strip()
            current = "why"
            buf     = [stripped[5:].strip()] if len(stripped) > 5 else []
        elif lower.startswith("[note]"):
            if current and buf:
                sections[current] = " ".join(buf).strip()
            current = "note"
            buf     = [stripped[6:].strip()] if len(stripped) > 6 else []
        elif current and stripped:
            buf.append(stripped)

    if current and buf:
        sections[current] = " ".join(buf).strip()

    return sections


def _render_analysis_result(result_text: str) -> None:
    """
    AI 분석 결과를 [WHAT]/[WHY]/[NOTE] 3단 카드로 렌더링.

    - [WHAT]: 파란 배경 카드 (주요 현상)
    - [WHY]:  분석 섹션 (맥락 해석)
    - [NOTE]: 노란/주황 border 경고 카드 (해석 한계)
    """
    parsed = _parse_result(result_text)

    # 파싱 실패 시 원문 그대로
    if not any(parsed.values()):
        st.markdown(
            f"<div style='background:#0D1520; border-left:3px solid #1E4A6A; "
            f"border-radius:6px; padding:12px 16px; margin:8px 0; "
            f"color:#D5E5FF; font-size:0.88rem; line-height:1.55;'>"
            f"{result_text.replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True,
        )
        return

    # [WHAT] — 파란 배경 카드
    if parsed["what"]:
        st.markdown(
            f"<div style='background:#112030; border-left:4px solid #00AEEF; "
            f"border-radius:8px; padding:14px 16px; margin:6px 0;'>"
            f"<span style='color:#00AEEF; font-size:0.75rem; font-weight:700; "
            f"letter-spacing:1px;'>WHAT</span>"
            f"<div style='color:#D5E5FF; font-size:0.88rem; line-height:1.55; "
            f"margin-top:6px;'>{parsed['what']}</div></div>",
            unsafe_allow_html=True,
        )

    # [WHY] — 분석 섹션
    if parsed["why"]:
        st.markdown(
            f"<div style='background:#0D1B2A; border-left:4px solid #4A7A9B; "
            f"border-radius:8px; padding:14px 16px; margin:6px 0;'>"
            f"<span style='color:#4A7A9B; font-size:0.75rem; font-weight:700; "
            f"letter-spacing:1px;'>WHY</span>"
            f"<div style='color:#B5C8D4; font-size:0.85rem; line-height:1.55; "
            f"margin-top:6px;'>{parsed['why']}</div></div>",
            unsafe_allow_html=True,
        )

    # [NOTE] — 노란 border 주의 카드
    if parsed["note"]:
        st.markdown(
            f"<div style='background:#1A1500; border-left:4px solid #FFB300; "
            f"border-radius:8px; padding:12px 16px; margin:6px 0;'>"
            f"<span style='color:#FFB300; font-size:0.75rem; font-weight:700; "
            f"letter-spacing:1px;'>NOTE</span>"
            f"<div style='color:#D5C87A; font-size:0.82rem; line-height:1.50; "
            f"margin-top:6px;'>{parsed['note']}</div></div>",
            unsafe_allow_html=True,
        )


# ─── 메인 진입점 ─────────────────────────────────────────────────────────────

def render_ai_analysis(
    tab_name: str,
    *,
    # daily
    worker_df: pd.DataFrame | None = None,
    journey_df: pd.DataFrame | None = None,
    date_str: str | None = None,
    shift_filter: str | None = None,
    # congestion
    space_df: pd.DataFrame | None = None,
    # weekly
    worker_df_list: list[pd.DataFrame] | None = None,
    dates: list[str] | None = None,
    # deep_space
    prediction_df: pd.DataFrame | None = None,
    accuracy_metrics: dict[str, Any] | None = None,
    # 공통
    sector_id: str | None = None,
    max_tokens: int = 900,
) -> None:
    """
    탭별 AI 분석 렌더링 메인 진입점.

    Args:
        tab_name:         "daily" | "congestion" | "weekly" | "deep_space"
        worker_df:        일별/주간 작업자 DataFrame
        journey_df:       일별 journey DataFrame (혼잡도 시간대 분석용)
        date_str:         분석 날짜 (YYYYMMDD)
        shift_filter:     교대 필터 ("주간"|"야간"|None=전체) — daily 전용
        space_df:         공간별 집계 DataFrame — congestion 전용
        worker_df_list:   주간 DataFrame 목록 — weekly 전용
        dates:            주간 날짜 목록 — weekly 전용
        prediction_df:    예측 결과 DataFrame — deep_space 전용
        accuracy_metrics: Prediction Journal 최근 정확도 dict — deep_space 전용
        sector_id:        Sector ID (None이면 config.SECTOR_ID)
        max_tokens:       LLM 최대 응답 토큰
    """
    from src.intelligence.llm_gateway import LLMGateway

    # LLM 가용성 확인
    if not LLMGateway.is_available():
        st.info(
            "AI 분석을 사용하려면 `.env` 파일에 `ANTHROPIC_API_KEY`를 설정하세요. "
            "AWS Bedrock 사용 시 `LLM_BACKEND=bedrock`으로 설정하세요."
        )
        return

    # sector_id 결정
    _sector_id = sector_id
    if not _sector_id:
        try:
            import config as cfg
            _sector_id = cfg.SECTOR_ID
        except Exception:
            _sector_id = "Y1_SKHynix"

    # 캐시 키: tab_name + date_str 조합 (탭/날짜별 독립 캐시)
    cache_date = date_str or (dates[-1] if dates else "unknown")
    shift_key  = f"_{shift_filter}" if shift_filter else ""
    cache_key  = f"ai_result_{tab_name}_{cache_date}{shift_key}"

    # 안내 문구
    st.markdown(
        "<div style='font-size:0.85rem; color:#6A8FA6; margin-bottom:10px;'>"
        "Claude AI가 익명화된 핵심 통계를 분석합니다. "
        "데이터 해석 보조 역할만 수행하며, 원인 단정이나 조치 권고는 하지 않습니다."
        "</div>",
        unsafe_allow_html=True,
    )

    # 캐시 결과 표시
    if cache_key in st.session_state and st.session_state[cache_key]:
        _render_analysis_result(st.session_state[cache_key])
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("다시 분석", key=f"re_{cache_key}"):
                del st.session_state[cache_key]
                st.rerun()
        return

    # 분석 실행 버튼
    if not st.button("AI 분석 실행", key=f"btn_{cache_key}", type="primary"):
        return

    with st.spinner("Claude AI 분석 중 (익명화 처리 포함)..."):
        try:
            packed_text, company_names, date_list = _build_packed_text(
                tab_name=tab_name,
                worker_df=worker_df,
                journey_df=journey_df,
                date_str=date_str,
                shift_filter=shift_filter,
                space_df=space_df,
                worker_df_list=worker_df_list,
                dates=dates,
                prediction_df=prediction_df,
                accuracy_metrics=accuracy_metrics,
                sector_id=_sector_id,
            )

            if not packed_text:
                st.warning("분석할 데이터가 없습니다.")
                return

            result = LLMGateway.analyze(
                tab_id=tab_name,
                packed_text=packed_text,
                company_names=company_names,
                date_list=date_list,
                max_tokens=max_tokens,
            )

            if result:
                st.session_state[cache_key] = result
                _render_analysis_result(result)
            else:
                st.warning(
                    "AI 분석 결과를 가져오지 못했습니다. "
                    "API 키와 네트워크 상태를 확인하세요."
                )

        except Exception as e:
            logger.warning("AI 분석 실행 중 오류: %s", e)
            st.error(f"AI 분석 중 오류가 발생했습니다: {e}")


# ─── 내부: 탭별 packed_text 빌드 ─────────────────────────────────────────────

def _build_packed_text(
    tab_name: str,
    *,
    worker_df: pd.DataFrame | None,
    journey_df: pd.DataFrame | None,
    date_str: str | None,
    shift_filter: str | None,
    space_df: pd.DataFrame | None,
    worker_df_list: list[pd.DataFrame] | None,
    dates: list[str] | None,
    prediction_df: pd.DataFrame | None,
    accuracy_metrics: dict[str, Any] | None,
    sector_id: str,
) -> tuple[str, list[str], list[str]]:
    """
    탭별 DataPackager + ContextBuilder 실행.

    Returns: (packed_text, company_names, date_list)
    """
    from src.intelligence.data_packager import DataPackager
    from src.intelligence.context_builder import ContextBuilder

    packed_text   = ""
    company_names: list[str] = []
    date_list:     list[str] = []

    if tab_name == "daily":
        if worker_df is None or worker_df.empty:
            return "", [], []
        packed_text = DataPackager.pack_daily(
            worker_df=worker_df,
            journey_df=journey_df,
            date_str=date_str or "",
            shift_filter=shift_filter,
        )
        # 비교 컨텍스트 추가
        if date_str:
            try:
                ctx = ContextBuilder.build_daily(date_str, sector_id)
                if ctx:
                    packed_text = packed_text + "\n" + ctx
            except Exception as e:
                logger.debug("daily context build failed: %s", e)

        # 익명화 메타
        if "company_name" in worker_df.columns:
            company_names = worker_df["company_name"].dropna().unique().tolist()
        date_list = [date_str] if date_str else []

    elif tab_name == "congestion":
        if space_df is None or space_df.empty:
            return "", [], []
        packed_text = DataPackager.pack_congestion(
            space_df=space_df,
            journey_df=journey_df,
            date_str=date_str or "",
        )
        # 공간 비교 컨텍스트 추가
        if date_str:
            try:
                ctx = ContextBuilder.build_space(sector_id, date_str)
                if ctx:
                    packed_text = packed_text + "\n" + ctx
            except Exception as e:
                logger.debug("space context build failed: %s", e)

        date_list = [date_str] if date_str else []

    elif tab_name == "weekly":
        if not worker_df_list or not dates:
            return "", [], []
        packed_text = DataPackager.pack_weekly(
            worker_df_list=worker_df_list,
            dates=dates,
        )
        # 최신 날짜 기준 비교 컨텍스트
        anchor = max(dates)
        try:
            ctx = ContextBuilder.build_daily(anchor, sector_id)
            if ctx:
                packed_text = packed_text + "\n" + ctx
        except Exception as e:
            logger.debug("weekly context build failed: %s", e)

        # 전체 기간 업체명 수집
        for df in worker_df_list:
            if df is not None and "company_name" in df.columns:
                company_names.extend(df["company_name"].dropna().unique().tolist())
        company_names = list(set(company_names))
        date_list     = list(dates)

    elif tab_name == "deep_space":
        packed_text = DataPackager.pack_deep_space(
            prediction_df=prediction_df,
            accuracy_metrics=accuracy_metrics,
        )
        # Deep Space 정확도 트렌드 컨텍스트 추가
        if date_str:
            try:
                ctx = ContextBuilder.build_deep_space(sector_id, date_str)
                if ctx:
                    packed_text = packed_text + "\n" + ctx
            except Exception as e:
                logger.debug("deep_space context build failed: %s", e)

        date_list = [date_str] if date_str else []

    else:
        logger.warning("알 수 없는 tab_name: %s", tab_name)
        return "", [], []

    return packed_text, company_names, date_list


# ─── 탭별 편의 래퍼 (선택적 사용) ────────────────────────────────────────────

def render_daily_ai(
    worker_df: pd.DataFrame,
    journey_df: pd.DataFrame | None = None,
    date_str: str | None = None,
    shift_filter: str | None = None,
    sector_id: str | None = None,
) -> None:
    """일별 작업자 분석 AI 렌더링 편의 래퍼."""
    render_ai_analysis(
        "daily",
        worker_df=worker_df,
        journey_df=journey_df,
        date_str=date_str,
        shift_filter=shift_filter,
        sector_id=sector_id,
    )


def render_congestion_ai(
    space_df: pd.DataFrame,
    journey_df: pd.DataFrame | None = None,
    date_str: str | None = None,
    sector_id: str | None = None,
) -> None:
    """공간 혼잡도 AI 렌더링 편의 래퍼."""
    render_ai_analysis(
        "congestion",
        space_df=space_df,
        journey_df=journey_df,
        date_str=date_str,
        sector_id=sector_id,
    )


def render_weekly_ai(
    worker_df_list: list[pd.DataFrame],
    dates: list[str],
    sector_id: str | None = None,
) -> None:
    """주간 트렌드 AI 렌더링 편의 래퍼."""
    render_ai_analysis(
        "weekly",
        worker_df_list=worker_df_list,
        dates=dates,
        sector_id=sector_id,
    )


def render_deep_space_ai(
    prediction_df: pd.DataFrame | None = None,
    accuracy_metrics: dict[str, Any] | None = None,
    date_str: str | None = None,
    sector_id: str | None = None,
) -> None:
    """Deep Space 예측 해석 AI 렌더링 편의 래퍼."""
    render_ai_analysis(
        "deep_space",
        prediction_df=prediction_df,
        accuracy_metrics=accuracy_metrics,
        date_str=date_str,
        sector_id=sector_id,
    )
