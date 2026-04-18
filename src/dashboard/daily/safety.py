"""
Daily Tab - Safety Section
==========================
안전 탭 관련 함수들.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    section_header, sub_header,
    PLOTLY_DARK, PLOTLY_LEGEND,
)
from .helpers import cre_grade, sii_grade
from src.utils.anonymizer import mask_name, mask_names_in_df
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, SII_HIGH, CRE_MEDIUM
# ★ 색상 단일 소스
from src.dashboard.styles import CHART_COLORS

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def _render_lone_work_section(journey_df: pd.DataFrame, sid: str):
    """
    단독 작업 현황 섹션.

    #no_lone 태그가 있는 구역에서 혼자 작업하는 작업자를 탐지하여 표시.
    건설현장 안전 규칙: 밀폐공간, 고압전 등 위험 구역에서 단독 작업 금지.
    """
    from src.pipeline.lone_work import detect_lone_work_realtime
    from src.spatial.loader import load_locus_df

    locus_df = load_locus_df(sid)
    if locus_df.empty:
        return

    lone_df = detect_lone_work_realtime(journey_df, locus_df)

    st.markdown(sub_header("단독 작업 현황 (#no_lone 구역)"), unsafe_allow_html=True)

    if lone_df.empty:
        st.markdown(
            "<div style='background:#0D2A1A; border:1px solid #00C897; "
            "border-radius:8px; padding:12px; margin:12px 0;'>"
            "<span style='color:#00C897; font-weight:600;'>OK</span> "
            "<span style='color:#7A8FA6;'>단독 작업 감지 없음</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Critical/Warning 카운트
    critical = lone_df[lone_df["status"] == "critical"]
    warning = lone_df[lone_df["status"] == "warning"]

    col1, col2 = st.columns(2)
    with col1:
        if not critical.empty:
            st.markdown(
                f"<div style='background:#2A1A1A; border:1px solid #FF4C4C44; "
                f"border-radius:8px; padding:10px; margin:6px 0;'>"
                f"<span style='color:#FF4C4C; font-weight:600;'>Critical</span> "
                f"<span style='color:#C8D6E8;'>30분+ 단독 작업: {len(critical)}명</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#0D2A1A; border:1px solid #00C89744; "
                "border-radius:8px; padding:10px; margin:6px 0;'>"
                "<span style='color:#00C897;'>OK</span> "
                "<span style='color:#7A8FA6;'>30분+ 단독 없음</span></div>",
                unsafe_allow_html=True,
            )

    with col2:
        if not warning.empty:
            st.markdown(
                f"<div style='background:#2A2A1A; border:1px solid #FFB30044; "
                f"border-radius:8px; padding:10px; margin:6px 0;'>"
                f"<span style='color:#FFB300; font-weight:600;'>Warning</span> "
                f"<span style='color:#C8D6E8;'>10분+ 단독 작업: {len(warning)}명</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#0D2A1A; border:1px solid #00C89744; "
                "border-radius:8px; padding:10px; margin:6px 0;'>"
                "<span style='color:#00C897;'>OK</span> "
                "<span style='color:#7A8FA6;'>10분+ 단독 없음</span></div>",
                unsafe_allow_html=True,
            )

    # 테이블 표시 (critical 또는 warning이 있을 때만)
    critical_warning = lone_df[lone_df["status"].isin(["critical", "warning"])]
    if not critical_warning.empty:
        display_df = critical_warning[["user_name", "locus_name", "lone_minutes", "status"]].copy()
        display_df["user_name"] = display_df["user_name"].apply(mask_name)
        display_df.columns = ["작업자", "구역", "단독 시간(분)", "상태"]
        display_df["상태"] = display_df["상태"].map({"critical": "Critical", "warning": "Warning"})

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()


def _render_basic_safety(worker_df):
    """EWI/CRE 없어도 표시 가능한 기본 안전 데이터."""
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**밀폐공간 장시간 체류 (30분 이상)**")
        if "confined_minutes" in worker_df.columns:
            danger = worker_df[worker_df["confined_minutes"] >= 30].sort_values("confined_minutes", ascending=False)
            if danger.empty:
                st.success("해당 없음")
            else:
                st.dataframe(mask_names_in_df(danger[["user_name", "company_name", "confined_minutes"]], "user_name")
                             .rename(columns={"user_name": "작업자", "company_name": "업체",
                                              "confined_minutes": "밀폐체류(분)"}),
                             use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**고압전 구역 체류 작업자**")
        if "high_voltage_minutes" in worker_df.columns:
            hv = worker_df[worker_df["high_voltage_minutes"] > 0].sort_values("high_voltage_minutes", ascending=False)
            if hv.empty:
                st.success("해당 없음")
            else:
                st.dataframe(mask_names_in_df(hv[["user_name", "company_name", "high_voltage_minutes"]], "user_name")
                             .rename(columns={"user_name": "작업자", "company_name": "업체",
                                              "high_voltage_minutes": "고압전체류(분)"}),
                             use_container_width=True, hide_index=True)


def render_safety(worker_df, space_df, locus_dict, date_str, sid, has_cre, journey_df=None):
    """안전 탭 전체 렌더링."""
    st.markdown(section_header("안전 분석"), unsafe_allow_html=True)

    # 단독 작업 현황 섹션 (최상단)
    if journey_df is not None and not journey_df.empty:
        _render_lone_work_section(journey_df, sid)

    # 지표 설명 토글 — 임계값은 constants.py 상수 동적 참조 (단일 진실 공급원)
    with st.expander("CRE / SII 지표 설명 및 계산식", expanded=False):
        st.markdown(f"""
**CRE (Combined Risk Exposure) - 복합 위험 노출도**

$$CRE = 0.45 \\times P_{{norm}} + 0.40 \\times S_{{norm}} + 0.15 \\times D_{{norm}}$$

| 요소 | 설명 | 계산 |
|------|------|------|
| **Personal (P)** | 피로도 + 단독작업 위험 | 연속 활성 시간 + 고위험 구역 혼자 비율 |
| **Static (S)** | 공간 고유 위험도 | locus_token 기반 위험 가중치 평균 |
| **Dynamic (D)** | 동적 밀집도 압력 | 동시 작업자 수 / 기준치(30명) |

**공간 위험도 가중치**: 밀폐공간/고압전(2.0) -> 기계실(1.8) -> 수직이동설비(1.5) -> 야외작업(1.3) -> 작업구역(1.2)
**CRE 등급**: 고위험(>={CRE_HIGH:.1f}) | 주의({CRE_MEDIUM:.1f}~{CRE_HIGH:.1f}) | 정상(<{CRE_MEDIUM:.1f})

---
**SII (Site Intensity-Risk Index) - 현장 강도/위험 복합 지수**

$$SII = EWI \\times S_{{norm}}$$

> 열심히 일하면서 위험한 공간에 있는 작업자 탐지
> SII>={SII_HIGH:.1f} = 집중 관리 대상
        """)

    # 밀폐공간 + 고압전 현황
    col1, col2, col3 = st.columns(3)
    with col1:
        confined = int((worker_df.get("confined_minutes", pd.Series(0)) > 0).sum())
        color = CHART_COLORS["critical"] if confined > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{confined}명</div>"
            f"<div class='metric-label'>밀폐공간 진입 작업자</div></div>", unsafe_allow_html=True)
    with col2:
        hv = int((worker_df.get("high_voltage_minutes", pd.Series(0)) > 0).sum())
        color = CHART_COLORS["high"] if hv > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{hv}명</div>"
            f"<div class='metric-label'>고압전 구역 작업자</div></div>", unsafe_allow_html=True)
    with col3:
        long_work = int((worker_df.get("work_minutes", pd.Series(0)) >= 720).sum())
        color = CHART_COLORS["medium"] if long_work > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{long_work}명</div>"
            f"<div class='metric-label'>장시간 근무 (12h+)</div></div>", unsafe_allow_html=True)

    st.divider()

    if not has_cre:
        # CRE 없어도 기본 안전 데이터 표시
        _render_basic_safety(worker_df)
        return

    cre_ser = worker_df["cre"].dropna()

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            worker_df, x="cre", nbins=40,
            color_discrete_sequence=[CHART_COLORS["critical"]],
            title="CRE 분포 (전체 작업자)",
            labels={"cre": "CRE", "count": "작업자 수"},
        )
        fig.add_vline(x=CRE_HIGH, line_dash="dash", line_color=CHART_COLORS["critical"],
                      annotation_text="고위험", annotation_font_color=CHART_COLORS["critical"])
        fig.add_vline(x=CRE_MEDIUM, line_dash="dash", line_color=CHART_COLORS["medium"],
                      annotation_text="주의", annotation_font_color=CHART_COLORS["medium"])
        fig.update_layout(**_DARK, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        grade_counts = cre_ser.apply(cre_grade).value_counts()
        grade_colors = {
            "고위험": CHART_COLORS["critical"],
            "주의":   CHART_COLORS["medium"],
            "정상":   CHART_COLORS["rest"],
        }
        fig2 = go.Figure(go.Pie(
            labels=grade_counts.index,
            values=grade_counts.values,
            marker_colors=[grade_colors.get(g, "#888") for g in grade_counts.index],
            hole=0.45,
        ))
        fig2.update_layout(title="CRE 등급 분포", legend=_LEG, **_DARK, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # SII 산점도 (EWI vs CRE)
    if "ewi" in worker_df.columns:
        st.markdown("#### EWI vs CRE 산점도 (크기 = SII)")
        plot_df = mask_names_in_df(worker_df[["user_name", "company_name", "ewi", "cre", "sii"]].dropna(), "user_name")
        plot_df["sii_size"] = (plot_df["sii"] * 30 + 5).clip(5, 40)
        plot_df["등급"] = plot_df["sii"].apply(sii_grade)

        fig3 = px.scatter(
            plot_df, x="ewi", y="cre",
            size="sii_size", color="등급",
            color_discrete_map={
                "집중관리": CHART_COLORS["critical"],
                "주의":     CHART_COLORS["medium"],
                "정상":     CHART_COLORS["rest"],
            },
            hover_data={"user_name": True, "company_name": True,
                        "ewi": ":.3f", "cre": ":.3f", "sii_size": False},
            title="EWI x CRE 위험 매트릭스 (점 크기 = SII)",
            labels={"ewi": "EWI (생산성)", "cre": "CRE (위험노출)"},
        )
        fig3.add_hline(y=CRE_HIGH, line_dash="dash", line_color=CHART_COLORS["critical"])
        fig3.add_vline(x=CRE_HIGH, line_dash="dash", line_color=CHART_COLORS["ewi"])
        fig3.update_layout(**_DARK, height=400, legend=_LEG)
        st.plotly_chart(fig3, use_container_width=True)

    # 고위험 작업자 목록 — constants.py의 CRE_HIGH 기준
    high_risk = worker_df[worker_df["cre"].fillna(0) >= CRE_HIGH].sort_values("cre", ascending=False)
    if not high_risk.empty:
        st.markdown(f"#### 고위험 작업자 (CRE >= {CRE_HIGH:.1f})")
        display_cols = ["user_name", "company_name", "cre", "ewi", "fatigue_score",
                        "confined_minutes", "high_voltage_minutes"]
        display_cols = [c for c in display_cols if c in high_risk.columns]
        rename = {
            "user_name": "작업자", "company_name": "업체",
            "cre": "CRE", "ewi": "EWI", "fatigue_score": "피로도",
            "confined_minutes": "밀폐(분)", "high_voltage_minutes": "고압전(분)",
        }
        st.dataframe(
            mask_names_in_df(high_risk[display_cols], "user_name").rename(columns=rename).head(50),
            use_container_width=True, hide_index=True,
        )

        # 고위험 통계 요약 (데이터 기반)
        n_high = len(high_risk)
        avg_cre_high = round(high_risk["cre"].mean(), 3) if "cre" in high_risk.columns else 0
        st.caption(
            f"고위험 작업자 {n_high}명 | 평균 CRE {avg_cre_high} | "
            f"최대 CRE {high_risk['cre'].max():.3f}" if "cre" in high_risk.columns else ""
        )
    else:
        st.success(f"OK 고위험 작업자(CRE >= {CRE_HIGH:.1f}) 없음")

    _render_basic_safety(worker_df)
