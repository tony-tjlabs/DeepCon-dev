"""
integrity/schema_check.py — 현장 보정 통계 서브탭 (일자별 추이)
==============================================================
tab2: `📊 현장 보정 통계`. 40일 전체를 훑어 보정률·저신뢰 비율·
신호 품질 평균을 추이 차트로 그린다. 실제 집계는 context.py 의
`_compute_daily_integrity_stats` 가 담당.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    COLORS,
    PLOTLY_DARK,
    PLOTLY_LEGEND,
    metric_card,
    section_header,
    sub_header,
)
from src.dashboard.date_utils import format_date_label
from src.dashboard.integrity.helpers import GAP_COLORS
from src.dashboard.integrity.context import _compute_daily_integrity_stats

logger = logging.getLogger(__name__)


def _render_daily_stats(sector_id: str) -> None:
    """전체 기간 보정 통계 추이."""
    st.markdown(section_header("전체 기간 보정 / 신호 품질 추이"),
                unsafe_allow_html=True)

    with st.spinner("전체 기간 통계 집계 중..."):
        stats = _compute_daily_integrity_stats(sector_id)

    if stats.empty:
        st.info("집계할 데이터가 없습니다.")
        return

    # 날짜 라벨
    stats["date_label"] = stats["date"].apply(format_date_label)

    # ── KPI Row ─────────────────────────────────────────────────────
    avg_gap    = stats["gap_filled_rate"].mean()
    avg_lowc   = stats["low_conf_rate"].mean()
    avg_invtr  = stats["invalid_tr_rate"].mean()
    avg_signal = stats["avg_signal"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gc = (COLORS["danger"] if avg_gap >= 30
              else COLORS["warning"] if avg_gap >= 15
              else COLORS["success"])
        st.markdown(metric_card("평균 Gap-fill 비율", f"{avg_gap:.1f}%", color=gc),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("평균 저신뢰 비율", f"{avg_lowc:.1f}%"),
                    unsafe_allow_html=True)
    with col3:
        ic = COLORS["danger"] if avg_invtr >= 1 else COLORS["success"]
        st.markdown(metric_card("평균 비유효 전이", f"{avg_invtr:.2f}%", color=ic),
                    unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("평균 signal/분", f"{avg_signal:.1f}"),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 일자별 보정률 추이 + 유형 분포 ──────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(sub_header("일자별 보정 / 저신뢰 비율 추이"),
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats["date_label"], y=stats["gap_filled_rate"],
            name="Gap-fill 비율", mode="lines+markers",
            line=dict(color=COLORS["warning"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=stats["date_label"], y=stats["low_conf_rate"],
            name="저신뢰 비율", mode="lines+markers",
            line=dict(color=COLORS["text_muted"], width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=stats["date_label"], y=stats["invalid_tr_rate"],
            name="비유효 전이", mode="lines+markers",
            line=dict(color=COLORS["danger"], width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            **PLOTLY_DARK, height=320,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(title="비율 (%)", gridcolor="#2A3A4A", ticksuffix="%"),
            yaxis2=dict(title="비유효 (%)", overlaying="y", side="right",
                        showgrid=False, ticksuffix="%"),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, key="plotly_13")

    with col_r:
        st.markdown(sub_header("일자별 Gap 신뢰도 분포 (Stacked)"),
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=stats["date_label"], y=stats["gap_high"],
            name="high", marker_color=GAP_COLORS["high"],
        ))
        fig2.add_trace(go.Bar(
            x=stats["date_label"], y=stats["gap_medium"],
            name="medium", marker_color=GAP_COLORS["medium"],
        ))
        fig2.add_trace(go.Bar(
            x=stats["date_label"], y=stats["gap_low"],
            name="low", marker_color=GAP_COLORS["low"],
        ))
        fig2.update_layout(
            **PLOTLY_DARK, height=320, barmode="stack",
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(title="보정 행 수 (건)", gridcolor="#2A3A4A"),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True, key="plotly_14")

    # ── 원본 테이블 ──────────────────────────────────────────────
    with st.expander("📋 일자별 보정 통계 원본 데이터"):
        st.dataframe(
            stats[[
                "date", "total_rows", "gap_filled_rate", "low_conf_rate",
                "invalid_tr_rate", "zero_signal_rate", "avg_signal",
                "gap_high", "gap_medium", "gap_low",
            ]].round(2),
            use_container_width=True,
        )
