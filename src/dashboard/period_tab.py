"""
기간 분석 탭 — 기간 선택 기반 종합 분석
========================================
기존 weekly_tab 확장형. 기간 선택 슬라이더 + 종합 대시보드.

섹션:
  1. 기간 선택 슬라이더 + 주요 KPI
  2. 3대 지표 통합 트렌드 (작업공간비율 / EWI / CRE)
  3. 요일별 패턴 (월~일 평균)
  4. 업체별 변화 추이 (선택 업체)
  5. 주간 리포트 (기존 weekly 재활용)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from core.cache.policy import MULTI_DAY_AGG
from src.dashboard.styles import (
    COLORS, CHART_COLORS, PLOTLY_DARK, PLOTLY_LEGEND,
    metric_card, metric_card_sm,
    section_header, sub_header,
)
from src.dashboard.date_utils import format_date_label
from src.pipeline.cache_manager import detect_processed_dates
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

logger = logging.getLogger(__name__)

DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _period_work_ratio_one(wp) -> float | None:
    """단일 날짜 work_ratio 계산 — 병렬 worker용 (picklable)."""
    if not wp.exists():
        return None
    try:
        wdf = pd.read_parquet(wp, columns=["work_minutes", "work_zone_minutes"])
        tw = wdf["work_minutes"].fillna(0).sum()
        tz = wdf["work_zone_minutes"].fillna(0).sum()
        return (tz / tw * 100) if tw > 0 else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=MULTI_DAY_AGG)
def _aggregate_period(sector_id: str, dates: tuple) -> pd.DataFrame:
    """기간 내 일자별 3대 지표 + 작업자 수 집계.

    ★ 성능: 작업공간 비율 계산을 ThreadPoolExecutor로 병렬 처리.
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.pipeline.summary_index import load_summary_index
    idx = load_summary_index(sector_id)
    dates_idx = idx.get("dates", {})

    paths = cfg.get_sector_paths(sector_id)
    paths_by_d = [(d, paths["processed_dir"] / d / "worker.parquet") for d in dates]

    # worker.parquet 병렬 집계
    max_w = min(len(paths_by_d), 8) if paths_by_d else 1
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        ratios = list(pool.map(lambda x: _period_work_ratio_one(x[1]), paths_by_d))

    rows = []
    for (d, _), work_ratio in zip(paths_by_d, ratios):
        entry = dates_idx.get(d, {})
        rows.append({
            "date":         d,
            "avg_ewi":      entry.get("avg_ewi", 0),
            "avg_cre":      entry.get("avg_cre", 0),
            "avg_sii":      entry.get("avg_sii", 0),
            "workers":      entry.get("tward_holders", 0),
            "work_ratio":   work_ratio,
        })

    return pd.DataFrame(rows)


def render_period_tab(sector_id: str) -> None:
    """기간 분석 탭 메인."""
    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    dates_asc = list(processed)
    n = len(dates_asc)

    # ── 기간 선택 슬라이더 ──────────────────────────────────────────
    st.markdown(section_header("분석 기간 선택"), unsafe_allow_html=True)
    s_idx, e_idx = st.select_slider(
        "기간 범위",
        options=list(range(n)),
        value=(max(0, n - 14), n - 1),
        format_func=lambda i: dates_asc[i],
        key="period_range",
    )
    selected = dates_asc[s_idx:e_idx + 1]
    st.caption(f"선택 기간: **{selected[0]} ~ {selected[-1]}**  ({len(selected)}일)")

    # ── M4-T34: 스키마 버전 검증 (선택 기간의 마지막 일자 기준) ──
    try:
        from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
        from src.dashboard.components import handle_schema_mismatch
        validate_schema(selected[-1], sector_id, strict_legacy=False)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, selected[-1])
        return
    except FileNotFoundError:
        pass

    # ── 데이터 집계 ────────────────────────────────────────────────
    with st.spinner("기간 데이터 집계 중..."):
        df = _aggregate_period(sector_id, tuple(selected))

    if df.empty:
        st.warning("집계 데이터 없음.")
        return

    df["date_label"]  = df["date"].apply(format_date_label)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["dow"]         = df["date_parsed"].dt.dayofweek

    # ── KPI 행 ──────────────────────────────────────────────────────
    avg_ewi_p  = df["avg_ewi"].mean()
    avg_cre_p  = df["avg_cre"].mean()
    avg_wr_p   = df["work_ratio"].dropna().mean() if df["work_ratio"].notna().any() else 0
    avg_wk_p   = df["workers"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("기간 평균 EWI", f"{avg_ewi_p:.3f}",
                                color=COLORS["accent"]),
                    unsafe_allow_html=True)
    with col2:
        cre_color = (COLORS["danger"] if avg_cre_p >= CRE_HIGH
                     else COLORS["warning"] if avg_cre_p >= 0.25
                     else COLORS["success"])
        st.markdown(metric_card("기간 평균 CRE", f"{avg_cre_p:.3f}",
                                color=cre_color),
                    unsafe_allow_html=True)
    with col3:
        wr_color = (COLORS["success"] if avg_wr_p >= 60
                    else COLORS["warning"])
        st.markdown(metric_card("평균 작업공간 비율", f"{avg_wr_p:.1f}%",
                                color=wr_color),
                    unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("일 평균 T-Ward", f"{avg_wk_p:.0f}명"),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── 3대 지표 통합 트렌드 ────────────────────────────────────────
    st.markdown(section_header("3대 지표 통합 트렌드"), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["work_ratio"],
        name="작업공간 비율 (%)", mode="lines+markers",
        line=dict(color=CHART_COLORS["rest"], width=2),   # 녹색 = 작업공간 비율 KPI
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_ewi"],
        name="EWI", mode="lines+markers",
        line=dict(color=CHART_COLORS["ewi"], width=2),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_cre"],
        name="CRE", mode="lines+markers",
        line=dict(color=CHART_COLORS["cre"], width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        **PLOTLY_DARK, height=360,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="작업공간 비율 (%)", range=[0, 110],
                   gridcolor=COLORS["grid"], ticksuffix="%"),
        yaxis2=dict(title="EWI / CRE", range=[0, 1],
                    overlaying="y", side="right", showgrid=False),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 요일별 패턴 ────────────────────────────────────────────────
    st.markdown(section_header("요일별 평균 패턴"), unsafe_allow_html=True)
    dow_agg = df.groupby("dow").agg(
        avg_ewi=("avg_ewi", "mean"),
        avg_cre=("avg_cre", "mean"),
        avg_wr=("work_ratio", "mean"),
        avg_workers=("workers", "mean"),
    ).reset_index()
    dow_agg["dow_name"] = dow_agg["dow"].map(lambda i: DOW_KR[i])

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(sub_header("요일별 EWI · CRE"), unsafe_allow_html=True)
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=dow_agg["dow_name"], y=dow_agg["avg_ewi"],
            name="EWI", marker_color=CHART_COLORS["ewi"],
        ))
        fig_dow.add_trace(go.Bar(
            x=dow_agg["dow_name"], y=dow_agg["avg_cre"],
            name="CRE", marker_color=CHART_COLORS["cre"],
        ))
        fig_dow.update_layout(
            **PLOTLY_DARK, height=280, barmode="group",
            yaxis=dict(range=[0, 1], gridcolor=COLORS["grid"]),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1},
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_r:
        st.markdown(sub_header("요일별 일 평균 작업자"), unsafe_allow_html=True)
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=dow_agg["dow_name"], y=dow_agg["avg_workers"],
            marker_color=COLORS["success"],
            text=[f"{v:.0f}" for v in dow_agg["avg_workers"]],
            textposition="outside",
        ))
        fig_wk.update_layout(
            **PLOTLY_DARK, height=280,
            yaxis=dict(title="명", gridcolor=COLORS["grid"]),
            showlegend=False,
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    # ── 주중 / 주말 비교 ────────────────────────────────────────────
    st.markdown(section_header("주중 / 주말 비교"), unsafe_allow_html=True)
    df["is_weekend"] = df["dow"].isin([5, 6])
    wk_agg = df.groupby("is_weekend").agg(
        avg_ewi=("avg_ewi", "mean"),
        avg_cre=("avg_cre", "mean"),
        avg_wr=("work_ratio", "mean"),
        avg_workers=("workers", "mean"),
        n_days=("date", "count"),
    ).reset_index()
    wk_agg["label"] = wk_agg["is_weekend"].map({True: "주말", False: "주중"})

    cols = st.columns(len(wk_agg))
    for i, (_, r) in enumerate(wk_agg.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div style='background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]};
                            border-radius:10px; padding:16px;'>
                    <div style='font-size:1.2rem; font-weight:700; color:{COLORS["accent"]};'>
                        {r["label"]} ({r["n_days"]}일)
                    </div>
                    <div style='margin-top:8px; font-size:0.85rem; color:{COLORS["text_muted"]};'>
                        EWI <b style='color:{COLORS["text"]}'>{r["avg_ewi"]:.3f}</b> ·
                        CRE <b style='color:{COLORS["warning"]}'>{r["avg_cre"]:.3f}</b><br>
                        작업공간 <b style='color:{COLORS["success"]}'>{r["avg_wr"]:.1f}%</b> ·
                        평균 작업자 <b style='color:{COLORS["text"]}'>{r["avg_workers"]:.0f}명</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 기존 주간 리포트 (expanded optional, _legacy) ─────────────
    with st.expander("📋 상세 주간 리포트 (legacy)"):
        try:
            from src.dashboard._legacy.weekly_tab import render_weekly_tab
            render_weekly_tab(sector_id)
        except Exception as e:
            st.error(f"주간 리포트 로드 실패: {e}")
            logger.exception("weekly_tab 로드 실패")
