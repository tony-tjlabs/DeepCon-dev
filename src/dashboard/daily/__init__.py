"""
Daily Tab Package
=================
일별 분석 탭 패키지 라우터.

5개 탭 + 패턴 탭을 서브모듈로 분리하여 구성.
render_daily_tab()을 re-export하여 기존 import 경로 유지.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as cfg
from src.pipeline.cache_manager import detect_processed_dates, load_daily_results, load_journey
from src.pipeline.summary_index import load_summary_index
from src.spatial.loader import load_locus_dict
from src.utils.weather import date_label
from src.dashboard.date_utils import get_available_dates, get_date_selector
from src.dashboard.ai_analysis import render_daily_ai
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, CRE_MEDIUM
# ★ 색상 단일 소스 — src/dashboard/styles.py
from src.dashboard.styles import CHART_COLORS

# 서브모듈 임포트
from .summary import render_summary
from .productivity import render_productivity, render_space_analysis_section
from .safety import render_safety
from .individual import render_individual
from .company import render_company
from .patterns import render_journey_patterns

__all__ = ["render_daily_tab"]


def render_daily_tab(sector_id: str | None = None):
    """일별 분석 탭 진입점."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)
    if not processed:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    # ── 날짜 선택 + KPI 프리뷰 ────────────────────────────────────
    summary_idx = load_summary_index(sid)
    summary_dates = summary_idx.get("dates", {})

    col_sel, col_prev = st.columns([1, 3])
    with col_sel:
        # date_utils 기반 날짜 선택기 (요일+날씨 포함)
        dates_asc = list(reversed(processed))
        date_str = get_date_selector(
            dates_asc,
            key=f"daily_date_{sid}",
            default_index=0,
            label="분석 날짜",
            show_label=True,
        ) or processed[-1]

    # 날짜 변경 시 journey 캐시 무효화
    _prev_date_key = f"daily_prev_date_{sid}"
    if st.session_state.get(_prev_date_key) != date_str:
        _old = st.session_state.get(_prev_date_key)
        if _old:
            st.session_state.pop(f"journey_{sid}_{_old}", None)
        st.session_state[_prev_date_key] = date_str

    # Summary Index 기반 즉시 KPI 프리뷰 (Parquet 로드 전)
    with col_prev:
        _render_kpi_preview(summary_dates, date_str)

    try:
        data = load_daily_results(date_str, sid)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    meta = data.get("meta", {})
    worker_df = data.get("worker", pd.DataFrame())
    space_df = data.get("space", pd.DataFrame())
    company_df = data.get("company", pd.DataFrame())
    locus_dict = load_locus_dict(sid)

    if worker_df.empty:
        st.warning("작업자 데이터 없음")
        return

    has_ewi = "ewi" in worker_df.columns
    has_cre = "cre" in worker_df.columns

    # BLE 커버리지 등급 동적 계산 (기존 parquet 호환)
    if "ble_coverage" not in worker_df.columns and "gap_ratio" in worker_df.columns:
        import numpy as np
        gr = worker_df["gap_ratio"].fillna(1.0)
        worker_df["ble_coverage"] = np.where(
            gr <= 0.2, "정상",
            np.where(gr <= 0.5, "부분음영",
                     np.where(gr <= 0.8, "음영", "미측정"))
        )
        worker_df["ble_coverage_pct"] = ((1 - gr) * 100).clip(0, 100).round(1)

    # ── 3교대 필터 ────────────────────────────────────────────────
    _shift_options: dict[str, str | None] = {
        "전체": None,
        "주간": "day",
        "야간": "night",
        "연장야간": "extended_night",
    }
    selected_shift_label = st.radio(
        "교대 필터",
        list(_shift_options.keys()),
        horizontal=True,
        key=f"daily_shift_filter_{sid}",
    )
    selected_shift = _shift_options[selected_shift_label]

    # shift_type 필터 적용 (컬럼 없는 날짜 방어 처리)
    filtered_worker_df = worker_df.copy()
    if selected_shift is not None:
        if "shift_type" in filtered_worker_df.columns:
            filtered_worker_df = filtered_worker_df[
                filtered_worker_df["shift_type"] == selected_shift
            ]
        else:
            st.caption("shift_type 컬럼이 없어 전체 데이터를 표시합니다.")

    # ── 탭 구성 ────────────────────────────────────────────────────
    tabs = st.tabs(["📊 요약", "⚡ 생산성", "🛡️ 안전", "👷 개인별", "🏗️ 업체별", "🧬 패턴"])

    with tabs[0]:
        render_summary(meta, filtered_worker_df, space_df, locus_dict, date_str, sid, has_ewi, has_cre)

    with tabs[1]:
        render_productivity(filtered_worker_df, space_df, locus_dict, has_ewi)

    with tabs[2]:
        # journey는 session_state에서 재사용 (탭 전환 시 중복 로드 방지)
        # ★ CLOUD_MODE: journey_slim 1일 = ~470MB → OOM 방지를 위해 스킵
        import config as _cfg
        _cache_key = f"journey_{sid}_{date_str}"
        journey_safety = st.session_state.get(_cache_key)
        if journey_safety is None and not _cfg.CLOUD_MODE:
            with st.spinner("이동 데이터 로드 중..."):
                journey_safety = load_journey(date_str, sid)
            st.session_state[_cache_key] = journey_safety
        if journey_safety is None:
            journey_safety = pd.DataFrame()
        render_safety(filtered_worker_df, space_df, locus_dict, date_str, sid, has_cre, journey_safety)

    with tabs[3]:
        render_individual(filtered_worker_df, locus_dict, has_ewi, has_cre)

    with tabs[4]:
        render_company(company_df, filtered_worker_df, has_ewi, has_cre)

    with tabs[5]:
        # ★ CLOUD_MODE: journey_slim 1일 = ~470MB → OOM 방지를 위해 스킵
        _cache_key = f"journey_{sid}_{date_str}"
        journey_df = st.session_state.get(_cache_key)
        if journey_df is None and not _cfg.CLOUD_MODE:
            with st.spinner("이동 데이터 로드 중..."):
                journey_df = load_journey(date_str, sid)
            st.session_state[_cache_key] = journey_df
        if journey_df is None:
            journey_df = pd.DataFrame()
        render_journey_patterns(filtered_worker_df, journey_df, sid)

    # ── AI 분석 섹션 ─────────────────────────────────────────────
    # 메인 차트/테이블 아래, 탭 렌더링 완료 후 배치
    st.divider()
    with st.expander("Claude AI 분석", expanded=False):
        # journey는 session_state 캐시에서 재사용
        _ai_cache_key = f"journey_{sid}_{date_str}"
        _journey_for_ai = st.session_state.get(_ai_cache_key)
        render_daily_ai(
            worker_df=filtered_worker_df,
            journey_df=_journey_for_ai,
            date_str=date_str,
            shift_filter=selected_shift,
            sector_id=sid,
        )


def _render_kpi_preview(summary_dates: dict, date_str: str):
    """Summary Index 기반 KPI 프리뷰 칩 렌더링."""
    entry = summary_dates.get(date_str)
    if not entry:
        return

    avg_cre = entry.get("avg_cre")
    avg_ewi = entry.get("avg_ewi")
    high_cre = entry.get("high_cre_count", 0)
    top_title = entry.get("top_insight_title", "")

    chips = []
    chips.append(
        f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
        f"border-radius:4px; padding:2px 8px; color:#C8D6E8;'>"
        f"📡 {entry.get('total_workers_access', 0):,}명</span>"
    )
    if avg_ewi is not None:
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
            f"border-radius:4px; padding:2px 8px; color:#00AEEF;'>"
            f"EWI {avg_ewi:.3f}</span>"
        )
    if avg_cre is not None:
        cre_col = CHART_COLORS["critical"] if avg_cre >= CRE_HIGH else CHART_COLORS["medium"] if avg_cre >= CRE_MEDIUM else CHART_COLORS["rest"]
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid #2A3A4A; "
            f"border-radius:4px; padding:2px 8px; color:{cre_col};'>"
            f"CRE {avg_cre:.3f}</span>"
        )
    if high_cre:
        chips.append(
            f"<span style='background:#2A1A1A; border:1px solid #FF4C4C44; "
            f"border-radius:4px; padding:2px 8px; color:#FF4C4C;'>"
            f"고위험 {high_cre}명</span>"
        )
    if top_title:
        # 심각도 색상 — CHART_COLORS 단일 소스 (4=critical, 3=high, 2=medium, 1=low)
        sev_colors = {
            4: CHART_COLORS["critical"],
            3: CHART_COLORS["high"],
            2: CHART_COLORS["medium"],
            1: CHART_COLORS["low"],
        }
        top_color = sev_colors.get(entry.get("top_insight_severity", 0), "#7A8FA6")
        chips.append(
            f"<span style='background:#1A2A3A; border:1px solid {top_color}44; "
            f"border-radius:4px; padding:2px 8px; color:{top_color};'>"
            f"💡 {top_title}</span>"
        )

    st.markdown(
        "<div style='display:flex; gap:6px; flex-wrap:wrap; padding:6px 0;'>"
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )
