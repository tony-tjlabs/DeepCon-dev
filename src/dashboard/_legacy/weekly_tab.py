"""
Weekly Tab - 주간 리포트 탭 (라우터)
=====================================
weekly/ 패키지의 서브모듈들을 조합하여 주간 리포트 탭을 렌더링합니다.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.styles import section_header
from src.dashboard.llm_deepcon import cached_weekly_trend_analysis, render_data_comment, is_llm_available
from src.pipeline.cache_manager import detect_processed_dates, load_multi_day_results
from src.utils.weather import date_label
from src.dashboard.date_utils import format_date_label, DAY_NAMES_KR
from src.dashboard.ai_analysis import render_weekly_ai

# weekly 패키지에서 서브모듈 import (_legacy 위치)
from src.dashboard._legacy.weekly import (
    render_weekly_kpi,
    render_weekly_site_status,
    render_daily_trend,
    render_ewi_cre_trend,
    render_weekly_space,
    render_weekly_company,
    render_weekly_safety,
    render_day_of_week_analysis,
    render_shift_trend,
)
from src.dashboard._legacy.weekly.report import render_report_download

logger = logging.getLogger(__name__)


def render_weekly_tab(sector_id: str | None = None):
    """주간 리포트 탭 (메인 진입점)."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)
    if len(processed) < 1:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    # -- 날짜 범위 선택 --
    st.markdown(section_header("분석 기간 설정"), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    date_list_dt = [datetime.strptime(d, "%Y%m%d") for d in processed]

    with col_a:
        start_dt = st.date_input(
            "시작일",
            value=date_list_dt[0].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
        )
    with col_b:
        end_dt = st.date_input(
            "종료일",
            value=date_list_dt[-1].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
        )

    if start_dt > end_dt:
        st.error("시작일이 종료일보다 늦습니다.")
        return

    # 선택 범위 내 처리된 날짜
    selected_dates = [
        d for d in processed
        if start_dt <= datetime.strptime(d, "%Y%m%d").date() <= end_dt
    ]

    if not selected_dates:
        st.warning("선택 기간에 처리된 데이터가 없습니다.")
        return

    # 요일 표시 포함 날짜 라벨
    _start_label = date_label(selected_dates[0])
    _end_label   = date_label(selected_dates[-1])
    _start_dow   = DAY_NAMES_KR[datetime.strptime(selected_dates[0], "%Y%m%d").weekday()]
    _end_dow     = DAY_NAMES_KR[datetime.strptime(selected_dates[-1], "%Y%m%d").weekday()]
    st.caption(
        f"분석 기간: **{_start_label}({_start_dow})** ~ "
        f"**{_end_label}({_end_dow})** ({len(selected_dates)}일)"
    )
    st.divider()

    # -- 데이터 로드 --
    with st.spinner("주간 데이터 로딩 중..."):
        multi = load_multi_day_results(tuple(selected_dates), sid, skip_journey=True)

    worker_df = multi.get("worker")
    space_df = multi.get("space")
    company_df = multi.get("company")
    metas = multi.get("metas", [])

    if worker_df is None or worker_df.empty:
        st.warning("데이터 없음")
        return

    # -- 주간 KPI --
    has_ewi, has_cre = render_weekly_kpi(worker_df, metas, selected_dates)

    st.divider()

    # -- 건설현장 주간 핵심 현황 --
    render_weekly_site_status(worker_df, metas, selected_dates, has_ewi, has_cre)

    st.divider()

    # -- 탭 구성 (5탭) --
    t1, t2, t3, t4, t5 = st.tabs([
        "트렌드", "요일/패턴", "안전/업체", "상세", "리포트",
    ])

    with t1:
        render_daily_trend(worker_df, selected_dates, metas)
        st.divider()
        render_ewi_cre_trend(worker_df, selected_dates, has_ewi, has_cre)

    with t2:
        render_day_of_week_analysis(worker_df, metas, selected_dates, has_ewi, has_cre)

    with t3:
        render_weekly_safety(worker_df, selected_dates)
        st.divider()
        render_weekly_company(company_df)

    with t4:
        with st.expander("주간/야간 비교", expanded=True):
            render_shift_trend(worker_df, selected_dates, metas, has_ewi, has_cre)
        with st.expander("공간별 현황", expanded=False):
            render_weekly_space(space_df, sid)

    with t5:
        render_report_download(
            selected_dates, sid, metas, worker_df, company_df,
            has_ewi, has_cre,
        )

    # -- AI 분석 섹션 (주간 트렌드) --
    st.divider()
    with st.expander("Claude AI 분석", expanded=False):
        # worker_df_list: 날짜별로 분할된 DataFrame 목록
        if "date" in worker_df.columns:
            _worker_df_list = [
                worker_df[worker_df["date"] == d].copy()
                for d in selected_dates
                if not worker_df[worker_df["date"] == d].empty
            ]
        else:
            _worker_df_list = [worker_df]
        render_weekly_ai(
            worker_df_list=_worker_df_list,
            dates=selected_dates,
            sector_id=sid,
        )
