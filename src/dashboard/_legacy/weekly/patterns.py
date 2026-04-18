"""
Weekly Patterns 모듈
====================
요일별 패턴 + EWI/CRE 분석 + 주간/야간 비교.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import section_header, PLOTLY_DARK
from src.utils.anonymizer import mask_names_in_df
from src.utils.weather import date_label_short
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

logger = logging.getLogger(__name__)

_DARK = PLOTLY_DARK
_DAY_NAMES = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


def render_day_of_week_analysis(
    worker_df: pd.DataFrame,
    metas: list[dict],
    date_list: list[str],
    has_ewi: bool,
    has_cre: bool,
):
    """요일별 패턴 분석."""
    st.markdown(section_header("요일별 패턴 분석"), unsafe_allow_html=True)
    st.caption("같은 요일의 데이터를 비교하여 요일별 특성 도출")

    if "date" not in worker_df.columns:
        st.info("날짜 정보 없음")
        return

    # 선택 기간 필터링
    if date_list:
        date_set = set(date_list)
        worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", "") in date_set]

    # meta 기반 요일별 출입인원
    if metas:
        meta_rows = []
        for m in metas:
            d = m.get("date_str", "")
            if len(d) == 8:
                dt = datetime.strptime(d, "%Y%m%d")
                meta_rows.append({
                    "date": d,
                    "day_of_week": dt.weekday(),
                    "day_name": _DAY_NAMES.get(dt.weekday(), ""),
                    "access": m.get("total_workers_access", 0),
                    "tward": m.get("total_workers_move", 0),
                })
        if meta_rows:
            mdf = pd.DataFrame(meta_rows)

            # 요일별 평균 출입인원
            dow_avg = mdf.groupby(["day_of_week", "day_name"]).agg(
                avg_access=("access", "mean"),
                avg_tward=("tward", "mean"),
                count=("date", "count"),
            ).reset_index().sort_values("day_of_week")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dow_avg["day_name"], y=dow_avg["avg_access"],
                name="평균 출입인원", marker_color="#4A90D9",
                text=dow_avg["avg_access"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside", textfont=dict(color="#D5E5FF"),
            ))
            fig.add_trace(go.Bar(
                x=dow_avg["day_name"], y=dow_avg["avg_tward"],
                name="평균 T-Ward 작업자", marker_color="#00C897",
                text=dow_avg["avg_tward"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside", textfont=dict(color="#D5E5FF"),
            ))
            fig.update_layout(
                title="요일별 평균 출입인원 / T-Ward 작업자",
                barmode="group",
                legend=dict(font=dict(color="#C8D6E8")),
                **_DARK, height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # 요일별 EWI/CRE 패턴
    wdf = worker_df.copy()
    wdf["dt"] = pd.to_datetime(wdf["date"], format="%Y%m%d")
    wdf["day_of_week"] = wdf["dt"].dt.dayofweek
    wdf["day_name"] = wdf["day_of_week"].map(_DAY_NAMES)

    agg_cols = {}
    if has_ewi:
        agg_cols["avg_ewi"] = ("ewi", "mean")
    if has_cre:
        agg_cols["avg_cre"] = ("cre", "mean")
    if "work_minutes" in wdf.columns:
        agg_cols["avg_work_h"] = ("work_minutes", lambda x: x.mean() / 60)
        agg_cols["overtime_count"] = ("work_minutes", lambda x: (x >= 720).sum())

    if agg_cols:
        dow_metrics = wdf.groupby(["day_of_week", "day_name"]).agg(**agg_cols).reset_index()
        dow_metrics = dow_metrics.sort_values("day_of_week")

        col1, col2 = st.columns(2)

        if has_ewi and "avg_ewi" in dow_metrics.columns:
            with col1:
                fig2 = go.Figure(go.Bar(
                    x=dow_metrics["day_name"],
                    y=dow_metrics["avg_ewi"].round(3),
                    marker_color=["#00C897" if v < 0.2 else "#FFB300" if v < 0.6 else "#FF4C4C"
                                  for v in dow_metrics["avg_ewi"]],
                    text=dow_metrics["avg_ewi"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside", textfont=dict(color="#D5E5FF"),
                ))
                fig2.update_layout(
                    title="요일별 평균 EWI (생산성)",
                    yaxis=dict(range=[0, max(0.8, dow_metrics["avg_ewi"].max() * 1.3)]),
                    **_DARK, height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)

        if has_cre and "avg_cre" in dow_metrics.columns:
            with col2:
                fig3 = go.Figure(go.Bar(
                    x=dow_metrics["day_name"],
                    y=dow_metrics["avg_cre"].round(3),
                    marker_color=["#00C897" if v < 0.3 else "#FFB300" if v < 0.6 else "#FF4C4C"
                                  for v in dow_metrics["avg_cre"]],
                    text=dow_metrics["avg_cre"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside", textfont=dict(color="#D5E5FF"),
                ))
                fig3.update_layout(
                    title="요일별 평균 CRE (위험노출)",
                    yaxis=dict(range=[0, max(0.8, dow_metrics["avg_cre"].max() * 1.3)]),
                    **_DARK, height=300,
                )
                st.plotly_chart(fig3, use_container_width=True)

        # 요일별 장시간 근무 + 근무시간
        if "overtime_count" in dow_metrics.columns and "avg_work_h" in dow_metrics.columns:
            col3, col4 = st.columns(2)
            with col3:
                fig4 = go.Figure(go.Bar(
                    x=dow_metrics["day_name"],
                    y=dow_metrics["overtime_count"],
                    marker_color=["#FF6B35" if v > 0 else "#2A3A4A" for v in dow_metrics["overtime_count"]],
                    text=dow_metrics["overtime_count"].apply(lambda v: f"{int(v)}명"),
                    textposition="outside", textfont=dict(color="#D5E5FF"),
                ))
                fig4.update_layout(
                    title="요일별 장시간 근무자 수 (12h+)",
                    **_DARK, height=300,
                )
                st.plotly_chart(fig4, use_container_width=True)

            with col4:
                fig5 = go.Figure(go.Bar(
                    x=dow_metrics["day_name"],
                    y=dow_metrics["avg_work_h"].round(1),
                    marker_color="#00AEEF",
                    text=dow_metrics["avg_work_h"].apply(lambda v: f"{v:.1f}h"),
                    textposition="outside", textfont=dict(color="#D5E5FF"),
                ))
                fig5.update_layout(
                    title="요일별 평균 근무시간",
                    **_DARK, height=300,
                )
                st.plotly_chart(fig5, use_container_width=True)

    # 요일별 인사이트 자동 도출
    if metas and len(metas) >= 3:
        _render_dow_insights(metas, wdf, has_ewi, has_cre)


def _render_dow_insights(metas, wdf, has_ewi, has_cre):
    """요일별 데이터에서 자동 인사이트 도출."""
    insights = []

    # 출입인원 가장 많은/적은 요일
    meta_rows = []
    for m in metas:
        d = m.get("date_str", "")
        if len(d) == 8:
            dt = datetime.strptime(d, "%Y%m%d")
            meta_rows.append({"dow": dt.weekday(), "access": m.get("total_workers_access", 0)})
    if meta_rows:
        mdf = pd.DataFrame(meta_rows)
        dow_access = mdf.groupby("dow")["access"].mean()
        if len(dow_access) >= 2:
            busiest = _DAY_NAMES.get(int(dow_access.idxmax()), "")
            quietest = _DAY_NAMES.get(int(dow_access.idxmin()), "")
            if busiest != quietest:
                diff_pct = round(
                    (dow_access.max() - dow_access.min()) / dow_access.max() * 100
                )
                insights.append(
                    f"출입인원: **{busiest}요일** 최다 vs **{quietest}요일** 최소 ({diff_pct}% 차이)"
                )

    # EWI 가장 높은 요일
    if has_ewi and "ewi" in wdf.columns:
        dow_ewi = wdf.groupby("day_of_week")["ewi"].mean()
        if len(dow_ewi) >= 2:
            max_ewi_day = _DAY_NAMES.get(int(dow_ewi.idxmax()), "")
            insights.append(
                f"생산성(EWI): **{max_ewi_day}요일**이 가장 높음 ({dow_ewi.max():.3f})"
            )

    if insights:
        st.markdown("#### 요일별 핵심 패턴")
        for ins in insights:
            st.markdown(
                f"<div style='background:#111820; border-left:3px solid #00AEEF; "
                f"padding:8px 12px; margin:4px 0; border-radius:6px; "
                f"font-size:0.88rem; color:#D5E5FF;'>{ins}</div>",
                unsafe_allow_html=True,
            )


def render_shift_trend(
    worker_df: pd.DataFrame,
    date_list: list[str],
    metas: list[dict],
    has_ewi: bool,
    has_cre: bool,
):
    """주간 / 야간 근무 일별 트렌드 비교."""
    if "shift_type" not in worker_df.columns or "date" not in worker_df.columns:
        st.info("Shift 데이터 없음 - 파이프라인을 재실행하면 주간/야간 분류가 적용됩니다.")
        return

    # 선택 기간 필터링
    if date_list:
        date_set = set(date_list)
        worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", "") in date_set]

    st.markdown("### 주간 / 야간 근무 비교 분석")

    # 일별 Shift 인원 트렌드
    shift_daily = (
        worker_df.groupby(["date", "shift_type"])["user_no"]
        .nunique()
        .reset_index(name="workers")
    )
    shift_daily["date_label"] = shift_daily["date"].apply(
        lambda d: date_label_short(d) if len(d) == 8 else d
    )
    shift_daily["shift_label"] = shift_daily["shift_type"].map(
        {"day": "주간", "night": "야간", "unknown": "미분류"}
    ).fillna("미분류")

    colors = {"주간": "#00AEEF", "야간": "#F5A623", "미분류": "#888888"}

    fig1 = px.line(
        shift_daily[shift_daily["shift_type"].isin(["day", "night"])],
        x="date_label", y="workers", color="shift_label",
        color_discrete_map=colors,
        markers=True,
        title="일별 주간 / 야간 작업자 수",
        labels={"date_label": "날짜", "workers": "작업자 수", "shift_label": "교대"},
    )
    fig1.update_layout(
        paper_bgcolor="#1A2A3A", plot_bgcolor="#111820", font_color="#C8D6E8",
        legend=dict(font=dict(color="#C8D6E8")),
        height=300, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # meta 기반 주간/야간 인원 (상세)
    if metas:
        meta_rows = []
        for m in metas:
            d = m.get("date_str", "")
            meta_rows.append({
                "날짜": date_label_short(d),
                "주간": m.get("day_workers", "-"),
                "야간": m.get("night_workers", "-"),
                "퇴근미기록": m.get("missing_exit_workers", "-"),
                "야간 BLE 보완(건)": m.get("night_supplement_records", 0),
            })
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

    st.divider()

    # EWI / CRE 주간 vs 야간 일별 비교 차트
    if (has_ewi or has_cre) and "shift_type" in worker_df.columns:
        agg_cols = {}
        if has_ewi and "ewi" in worker_df.columns:
            agg_cols["avg_ewi"] = ("ewi", "mean")
        if has_cre and "cre" in worker_df.columns:
            agg_cols["avg_cre"] = ("cre", "mean")
        if has_cre and "fatigue_score" in worker_df.columns:
            agg_cols["avg_fatigue"] = ("fatigue_score", "mean")

        if agg_cols:
            shift_metrics = (
                worker_df[worker_df["shift_type"].isin(["day", "night"])]
                .groupby(["date", "shift_type"])
                .agg(**agg_cols)
                .reset_index()
            )
            shift_metrics["date_label"] = shift_metrics["date"].apply(
                lambda d: date_label_short(d) if len(d) == 8 else d
            )
            shift_metrics["shift_label"] = shift_metrics["shift_type"].map(
                {"day": "주간", "night": "야간"}
            )

            metric_tabs_labels = []
            if "avg_ewi" in shift_metrics.columns:
                metric_tabs_labels.append("EWI")
            if "avg_cre" in shift_metrics.columns:
                metric_tabs_labels.append("CRE")
            if "avg_fatigue" in shift_metrics.columns:
                metric_tabs_labels.append("피로도")

            if metric_tabs_labels:
                mt = st.tabs([f"{m}" for m in metric_tabs_labels])
                col_map = {"EWI": "avg_ewi", "CRE": "avg_cre", "피로도": "avg_fatigue"}
                title_map = {
                    "EWI": "주간 vs 야간 - 평균 EWI (생산성)",
                    "CRE": "주간 vs 야간 - 평균 CRE (위험노출)",
                    "피로도": "주간 vs 야간 - 평균 피로도",
                }
                for tab, metric_name in zip(mt, metric_tabs_labels):
                    with tab:
                        ycol = col_map[metric_name]
                        if ycol not in shift_metrics.columns:
                            continue
                        fig = px.bar(
                            shift_metrics, x="date_label", y=ycol,
                            color="shift_label", barmode="group",
                            color_discrete_map=colors,
                            title=title_map[metric_name],
                            labels={"date_label": "날짜", ycol: metric_name, "shift_label": "교대"},
                            text=shift_metrics[ycol].round(3),
                        )
                        fig.update_traces(textposition="outside",
                                          textfont=dict(color="#C8D6E8", size=10))
                        fig.update_layout(
                            paper_bgcolor="#1A2A3A", plot_bgcolor="#111820",
                            font_color="#C8D6E8",
                            legend=dict(font=dict(color="#C8D6E8")),
                            yaxis=dict(range=[0, 1.15]),
                            height=320, margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # 야간 고위험 누적 현황 — constants.py의 CRE_HIGH 기준
    if has_cre and "cre" in worker_df.columns:
        st.divider()
        st.markdown(f"#### 야간 고위험 작업자 현황 (CRE>={CRE_HIGH:.1f})")
        night_high = worker_df[
            (worker_df["shift_type"] == "night") & (worker_df["cre"].fillna(0) >= CRE_HIGH)
        ].sort_values("cre", ascending=False)

        if night_high.empty:
            st.success("야간 고위험 작업자 없음")
        else:
            st.warning(f"총 {len(night_high)}건 (중복 포함) - 야간 작업자는 피로/조명 부재로 위험 가중")
            show = ["date", "user_name", "company_name", "cre", "fatigue_score",
                    "confined_minutes", "high_voltage_minutes"]
            show = [c for c in show if c in night_high.columns]
            st.dataframe(
                mask_names_in_df(night_high[show].head(50), "user_name").rename(columns={
                    "date": "날짜", "user_name": "작업자", "company_name": "업체",
                    "cre": "CRE", "fatigue_score": "피로도",
                    "confined_minutes": "밀폐(분)", "high_voltage_minutes": "고압전(분)",
                }),
                use_container_width=True, hide_index=True,
            )
