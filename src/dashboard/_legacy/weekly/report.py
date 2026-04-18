"""
Weekly Report 모듈
==================
PDF 리포트 생성 및 다운로드.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.styles import section_header
from src.dashboard.llm_deepcon import is_llm_available
from src.utils.weather import date_label
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH
# ★ 색상 단일 소스
from src.dashboard.styles import CHART_COLORS

logger = logging.getLogger(__name__)


def _render_report_preview(kpi, worker_stats, insight_report, llm_text):
    """리포트 미리보기 카드."""
    st.markdown("##### 리포트 미리보기")

    # KPI 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_ewi = kpi.get("avg_ewi", 0)
        ewi_color = CHART_COLORS["critical"] if avg_ewi >= EWI_HIGH else CHART_COLORS["medium"] if avg_ewi >= 0.2 else CHART_COLORS["rest"]
        st.markdown(
            f"<div style='background:#0D1B2A; border-radius:8px; padding:12px; text-align:center;'>"
            f"<div style='color:#7A8FA6; font-size:0.75rem;'>평균 EWI</div>"
            f"<div style='color:{ewi_color}; font-size:1.5rem; font-weight:bold;'>{avg_ewi:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        avg_cre = kpi.get("avg_cre", 0)
        cre_color = CHART_COLORS["critical"] if avg_cre >= CRE_HIGH else CHART_COLORS["medium"] if avg_cre >= 0.3 else CHART_COLORS["rest"]
        st.markdown(
            f"<div style='background:#0D1B2A; border-radius:8px; padding:12px; text-align:center;'>"
            f"<div style='color:#7A8FA6; font-size:0.75rem;'>평균 CRE</div>"
            f"<div style='color:{cre_color}; font-size:1.5rem; font-weight:bold;'>{avg_cre:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        high_cre = kpi.get("high_cre", 0)
        risk_color = CHART_COLORS["critical"] if high_cre > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div style='background:#0D1B2A; border-radius:8px; padding:12px; text-align:center;'>"
            f"<div style='color:#7A8FA6; font-size:0.75rem;'>고위험 작업자</div>"
            f"<div style='color:{risk_color}; font-size:1.5rem; font-weight:bold;'>{high_cre}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        insight_count = len(insight_report.insights) if insight_report else 0
        critical_count = insight_report.critical_count if insight_report else 0
        insight_color = CHART_COLORS["critical"] if critical_count > 0 else CHART_COLORS["medium"] if insight_count > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div style='background:#0D1B2A; border-radius:8px; padding:12px; text-align:center;'>"
            f"<div style='color:#7A8FA6; font-size:0.75rem;'>인사이트</div>"
            f"<div style='color:{insight_color}; font-size:1.5rem; font-weight:bold;'>{insight_count}건</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 상위 인사이트 미리보기
    if insight_report and insight_report.insights:
        with st.expander("상위 인사이트 미리보기", expanded=False):
            for i, ins in enumerate(insight_report.top(3)):
                sev_icon = {"4": "(critical)", "3": "(high)", "2": "(medium)", "1": "(low)"}.get(str(ins.severity), "")
                desc = ins.description[:100] + "..." if len(ins.description) > 100 else ins.description
                st.markdown(f"{sev_icon} **{ins.title}**: {desc}")

    st.divider()


def render_report_download(
    selected_dates,
    sid,
    metas,
    worker_df,
    company_df,
    has_ewi,
    has_cre,
):
    """PDF 리포트 생성 및 다운로드 버튼."""
    st.markdown(section_header("PDF 리포트 다운로드"), unsafe_allow_html=True)

    date_range = f"{date_label(selected_dates[0])} ~ {date_label(selected_dates[-1])}"

    sec_info = cfg.SECTOR_REGISTRY.get(sid, {})
    sector_label = sec_info.get("label", sid or "Y1")

    # KPI 집계
    kpi = _build_kpi(metas, worker_df, selected_dates, has_ewi, has_cre)
    worker_stats = _build_worker_stats(worker_df)
    trend_data = _build_trend_data(metas, worker_df)

    # 인사이트
    insight_report = st.session_state.get("_insights")

    # LLM 내러티브
    llm_text = ""
    if is_llm_available() and insight_report:
        llm_text = _get_llm_narrative(selected_dates, sid, insight_report)

    # 리포트 미리보기
    _render_report_preview(kpi, worker_stats, insight_report, llm_text)

    st.markdown(
        f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; "
        f"border-radius:10px; padding:16px 20px; margin-bottom:16px;'>"
        f"<div style='color:#C8D6E8; font-size:0.92rem;'>기간: <b>{date_range}</b></div>"
        f"<div style='color:#7A8FA6; font-size:0.8rem; margin-top:4px;'>"
        f"인사이트 {len(insight_report.insights) if insight_report else 0}건 포함 / "
        f"AI 브리핑 {'포함' if llm_text else '미포함'} / "
        f"3단 구조 맥락 해석 포함</div></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Executive Summary (3-5p)", use_container_width=True, key="dl_exec"):
            _generate_and_download_pdf(
                "executive", date_range, sector_label, sid, insight_report,
                kpi, llm_text, worker_stats, worker_df, metas, selected_dates,
                None, None,
            )

    with col2:
        if st.button("Detailed Report (8-12p)", use_container_width=True, key="dl_detail"):
            _generate_and_download_pdf(
                "detailed", date_range, sector_label, sid, insight_report,
                kpi, llm_text, worker_stats, worker_df, metas, selected_dates,
                company_df, trend_data,
            )

    # 다운로드 버튼 (생성 후 표시)
    if st.session_state.get("_pdf_exec"):
        st.download_button(
            "Download Executive Summary",
            data=st.session_state["_pdf_exec"],
            file_name=st.session_state.get("_pdf_exec_name", "report.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key="download_exec",
        )

    if st.session_state.get("_pdf_detail"):
        st.download_button(
            "Download Detailed Report",
            data=st.session_state["_pdf_detail"],
            file_name=st.session_state.get("_pdf_detail_name", "report.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key="download_detail",
        )


def _build_kpi(metas, worker_df, selected_dates, has_ewi, has_cre):
    """KPI 데이터 생성."""
    # 선택 기간 필터링
    if selected_dates:
        date_set = set(selected_dates)
        if "date" in worker_df.columns:
            worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", "") in date_set]

    kpi = {}
    if metas:
        access_vals = [m.get("total_workers_access", 0) for m in metas]
        tward_vals = [m.get("total_workers_move", 0) for m in metas]

        kpi["cum_access"] = int(sum(access_vals))
        kpi["cum_tward"] = int(sum(tward_vals))
        kpi["avg_daily_access"] = int(sum(access_vals) / len(access_vals))
        kpi["avg_daily_tward"] = int(sum(tward_vals) / len(tward_vals))

        # 주중/주말 분리
        wd_access, we_access = [], []
        wd_tward, we_tward = [], []
        for m, d in zip(metas, selected_dates):
            try:
                dow = datetime.strptime(d, "%Y%m%d").weekday()
            except Exception:
                dow = 0
            a = m.get("total_workers_access", 0)
            t = m.get("total_workers_move", 0)
            if dow < 5:
                wd_access.append(a)
                wd_tward.append(t)
            else:
                we_access.append(a)
                we_tward.append(t)
        kpi["weekday_avg_access"] = int(sum(wd_access) / len(wd_access)) if wd_access else 0
        kpi["weekend_avg_access"] = int(sum(we_access) / len(we_access)) if we_access else 0
        kpi["weekday_days"] = len(wd_access)
        kpi["weekend_days"] = len(we_access)

        if "user_no" in worker_df.columns:
            kpi["unique_workers"] = int(worker_df["user_no"].nunique())
        elif "user_name" in worker_df.columns:
            kpi["unique_workers"] = int(worker_df["user_name"].nunique())
        else:
            kpi["unique_workers"] = 0

        kpi["total_access"] = kpi["cum_access"]
        kpi["total_tward"] = kpi["cum_tward"]
        kpi["tward_rate"] = kpi["cum_tward"] / kpi["cum_access"] * 100 if kpi["cum_access"] > 0 else 0
        kpi["companies"] = metas[-1].get("companies", 0)

    if has_ewi and "ewi" in worker_df.columns:
        kpi["avg_ewi"] = round(worker_df["ewi"].mean(), 3)
    if has_cre and "cre" in worker_df.columns:
        kpi["avg_cre"] = round(worker_df["cre"].mean(), 3)
        kpi["high_cre"] = int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum())

    # 주중/주말 EWI/CRE 분리
    if "date" in worker_df.columns:
        def _is_weekday(d):
            try:
                return datetime.strptime(str(d), "%Y%m%d").weekday() < 5
            except Exception:
                return True
        wd_mask = worker_df["date"].apply(_is_weekday)
        if has_ewi and "ewi" in worker_df.columns:
            kpi["weekday_ewi"] = round(worker_df.loc[wd_mask, "ewi"].mean(), 3) if wd_mask.any() else 0
            kpi["weekend_ewi"] = round(worker_df.loc[~wd_mask, "ewi"].mean(), 3) if (~wd_mask).any() else 0
        if has_cre and "cre" in worker_df.columns:
            kpi["weekday_cre"] = round(worker_df.loc[wd_mask, "cre"].mean(), 3) if wd_mask.any() else 0
            kpi["weekend_cre"] = round(worker_df.loc[~wd_mask, "cre"].mean(), 3) if (~wd_mask).any() else 0

    return kpi


def _build_worker_stats(worker_df):
    """작업자 통계 생성."""
    worker_stats = {}
    if "fatigue_score" in worker_df.columns:
        worker_stats["high_fatigue"] = int((worker_df["fatigue_score"] >= 0.6).sum())
    if "confined_minutes" in worker_df.columns:
        worker_stats["confined"] = int((worker_df["confined_minutes"] > 0).sum())
    if "alone_ratio" in worker_df.columns:
        worker_stats["alone_risk"] = int((worker_df["alone_ratio"] >= 0.5).sum())
    return worker_stats


def _build_trend_data(metas, worker_df):
    """트렌드 데이터 생성.

    Note: metas/worker_df는 호출 전에 이미 selected_dates로 필터링된 상태.
    """
    trend_data = []
    if metas:
        for m in metas:
            d = m.get("date_str", "")
            trend_data.append({
                "date": d,
                "total_access": m.get("total_workers_access", 0),
                "total_tward": m.get("total_workers_move", 0),
                "avg_ewi": 0, "avg_cre": 0, "high_cre": 0,
            })
        if "date" in worker_df.columns:
            for td in trend_data:
                day_df = worker_df[worker_df["date"] == td["date"]]
                if not day_df.empty:
                    if "ewi" in day_df.columns:
                        td["avg_ewi"] = round(day_df["ewi"].mean(), 3)
                    if "cre" in day_df.columns:
                        td["avg_cre"] = round(day_df["cre"].mean(), 3)
                        td["high_cre"] = int((day_df["cre"].fillna(0) >= CRE_HIGH).sum())
    return trend_data


def _get_llm_narrative(selected_dates, sid, insight_report):
    """LLM 내러티브 생성."""
    try:
        from src.intelligence.insight_aggregator import build_kpi_summary
        from src.dashboard.llm_deepcon import cached_insight_narrative

        insight_data = st.session_state.get("_insight_data", {})
        kpi_text = build_kpi_summary(insight_data)
        llm_text = cached_insight_narrative(
            date_str=selected_dates[-1], sector_id=sid,
            insights_summary=insight_report.summary_text(5),
            kpi_summary=kpi_text,
        )
        return llm_text
    except Exception:
        return ""


def _generate_and_download_pdf(
    report_type, date_range, sector_label, sid, insight_report,
    kpi, llm_text, worker_stats, worker_df, metas, selected_dates,
    company_df, trend_data,
):
    """PDF 생성 및 세션 상태 저장."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("1/4 데이터 준비 중...")
        progress_bar.progress(10)

        from src.intelligence.report_generator import generate_report
        from src.intelligence.report_context import build_report_context

        status_text.text("2/4 맥락 데이터 생성 중...")
        progress_bar.progress(30)

        report_ctx = build_report_context(
            worker_df=worker_df,
            kpi=kpi,
            dates=selected_dates,
            use_llm=False,
        )

        status_text.text("3/4 PDF 렌더링 중...")
        progress_bar.progress(60)

        if report_type == "executive":
            pdf_bytes = generate_report(
                report_type="executive",
                date_range=date_range,
                sector_label=sector_label,
                insights=insight_report,
                kpi_data=kpi,
                llm_narrative=llm_text,
                worker_stats=worker_stats,
                worker_df=worker_df,
                metas=metas,
                dates=selected_dates,
            )
            st.session_state["_pdf_exec"] = pdf_bytes
            st.session_state["_pdf_exec_name"] = f"DeepCon_{sid}_{selected_dates[0]}-{selected_dates[-1]}_Executive.pdf"
        else:
            pdf_bytes = generate_report(
                report_type="detailed",
                date_range=date_range,
                sector_label=sector_label,
                insights=insight_report,
                kpi_data=kpi,
                llm_narrative=llm_text,
                worker_stats=worker_stats,
                company_data=company_df,
                trend_data=trend_data,
                worker_df=worker_df,
                metas=metas,
                dates=selected_dates,
            )
            st.session_state["_pdf_detail"] = pdf_bytes
            st.session_state["_pdf_detail_name"] = f"DeepCon_{sid}_{selected_dates[0]}-{selected_dates[-1]}_Detailed.pdf"

        status_text.text("4/4 완료!")
        progress_bar.progress(100)

        status_text.empty()
        progress_bar.empty()
        st.success(f"{report_type.title()} Report 생성 완료 ({len(pdf_bytes):,} bytes)")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"PDF 생성 실패: {e}")
