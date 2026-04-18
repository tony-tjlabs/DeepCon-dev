"""
Daily Tab - Productivity Section
================================
생산성 탭 관련 함수들.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    section_header, PLOTLY_DARK, PLOTLY_LEGEND,
    CHART_COLORS,
)
from src.dashboard.components import (
    render_space_card,
    DWELL_CATEGORY_STYLES,
)
from src.dashboard.metric_constants import (
    HIGH_ACTIVE_THRESHOLD,
    LOW_ACTIVE_THRESHOLD,
    HIGH_WORK_WEIGHT,
    LOW_WORK_WEIGHT,
    STANDBY_WEIGHT,
    EWI_GAP_RELIABLE_THRESHOLD,
)
from .helpers import ewi_grade

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def render_productivity(worker_df, space_df, locus_dict, has_ewi):
    """생산성 탭 전체 렌더링."""
    st.markdown(section_header("생산성 분석"), unsafe_allow_html=True)

    # 지표 설명 토글 — 임계값은 pipeline/metrics.py 상수를 동적 참조 (단일 진실 공급원)
    with st.expander("EWI 지표 설명 및 계산식", expanded=False):
        st.markdown(f"""
**EWI (Effective Work Intensity) - 유효 작업 집중도**

$$EWI = \\frac{{고활성 \\times {HIGH_WORK_WEIGHT} + 저활성 \\times {LOW_WORK_WEIGHT} + 대기 \\times {STANDBY_WEIGHT}}}{{실제\\ 근무시간(분)}}$$

| 구분 | 기준 | 가중치 |
|------|------|--------|
| 고활성 (High Active) | active_ratio >= {HIGH_ACTIVE_THRESHOLD} | {HIGH_WORK_WEIGHT} |
| 저활성 (Low Active) | {LOW_ACTIVE_THRESHOLD} <= active_ratio < {HIGH_ACTIVE_THRESHOLD} | {LOW_WORK_WEIGHT} |
| 대기 (Standby) | active_ratio < {LOW_ACTIVE_THRESHOLD} | {STANDBY_WEIGHT} |
| 휴식/이동 | 비작업 공간 | 0 |

**active_ratio**: BLE T-Ward의 활성 신호수 / 전체 신호수
(1에 가까울수록 활발한 신체 이동, 0에 가까울수록 정지 상태)

**EWI 등급**: 고강도(>=0.6) | 보통(0.2~0.6) | 저강도(<0.2)
**음영 신뢰도**: BLE 미수집 구간(gap) > {int(EWI_GAP_RELIABLE_THRESHOLD * 100)}% 이면 기록 분 기준으로 계산
        """)

    if not has_ewi:
        st.info("EWI 데이터가 없습니다. 파이프라인에서 데이터를 재처리하세요.")
        return

    ewi_ser = worker_df["ewi"].dropna()

    # EWI 분포 히스토그램
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            worker_df, x="ewi", nbins=40,
            color_discrete_sequence=[CHART_COLORS["ewi"]],
            title="EWI 분포 (전체 작업자)",
            labels={"ewi": "EWI", "count": "작업자 수"},
        )
        fig.add_vline(x=0.6, line_dash="dash", line_color=CHART_COLORS["critical"],
                      annotation_text="고강도 기준", annotation_font_color=CHART_COLORS["critical"])
        fig.add_vline(x=0.2, line_dash="dash", line_color=CHART_COLORS["medium"],
                      annotation_text="보통 기준", annotation_font_color=CHART_COLORS["medium"])
        fig.update_layout(**_DARK, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        grade_counts = ewi_ser.apply(ewi_grade).value_counts()
        # EWI 등급 색상 — CHART_COLORS 단일 소스
        grade_colors = {
            "고강도": CHART_COLORS["critical"],
            "보통":   CHART_COLORS["medium"],
            "저강도": CHART_COLORS["rest"],
        }
        fig2 = go.Figure(go.Pie(
            labels=grade_counts.index,
            values=grade_counts.values,
            marker_colors=[grade_colors.get(g, "#888") for g in grade_counts.index],
            hole=0.45,
        ))
        fig2.update_layout(title="EWI 등급 분포", legend=_LEG, **_DARK, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # 시간 구성 (고활성/저활성/대기/휴식/이동) — 컬럼 결측 방어
    if "high_active_min" in worker_df.columns:
        st.markdown("#### 평균 시간 구성 (분)")
        def _col_mean(col: str) -> float:
            return float(worker_df[col].mean()) if col in worker_df.columns else 0.0
        avg_time = {
            "고활성 (x1.0)": _col_mean("high_active_min"),
            "저활성 (x0.5)": _col_mean("low_active_min"),
            "대기 (x0.2)":   _col_mean("standby_min"),
            "휴식":          _col_mean("rest_min"),
            "게이트/이동":   _col_mean("transit_min"),
        }
        # 시간 구성 색상 — CHART_COLORS 단일 소스
        color_map = {
            "고활성 (x1.0)": CHART_COLORS["high_active"],   # danger 계열 (고강도)
            "저활성 (x0.5)": CHART_COLORS["low_active"],    # low_active
            "대기 (x0.2)":   CHART_COLORS["medium"],        # 대기 = warning
            "휴식":          CHART_COLORS["rest"],          # rest = success
            "게이트/이동":   CHART_COLORS["transit"],       # transit
        }
        fig3 = go.Figure(go.Bar(
            x=list(avg_time.keys()),
            y=[round(v, 1) for v in avg_time.values()],
            marker_color=[color_map[k] for k in avg_time],
            text=[f"{v:.0f}분" for v in avg_time.values()],
            textposition="outside",
        ))
        fig3.update_layout(title="작업자 평균 시간 구성", **_DARK, height=320,
                           yaxis_title="평균 (분)", legend=_LEG)
        st.plotly_chart(fig3, use_container_width=True)

    # 업체별 평균 EWI
    if "company_name" in worker_df.columns:
        company_ewi = (
            worker_df.groupby("company_name")["ewi"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_ewi", "count": "workers"})
        )
        company_ewi = company_ewi[company_ewi["workers"] >= 3].nlargest(20, "avg_ewi")
        fig4 = px.bar(
            company_ewi, x="avg_ewi", y="company_name", orientation="h",
            color="avg_ewi", color_continuous_scale=CHART_COLORS["scale_risk"],
            range_color=[0, 0.8],
            title="업체별 평균 EWI (3명 이상 업체, 상위 20개)",
            labels={"avg_ewi": "평균 EWI", "company_name": "업체"},
            text=company_ewi["avg_ewi"].apply(lambda v: f"{v:.3f}"),
        )
        fig4.update_layout(yaxis=dict(autorange="reversed"), height=500, **_DARK,
                           coloraxis_colorbar=dict(tickfont=dict(color="#C8D6E8")))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 공간 분석 섹션 - Enriched Locus 정보 활용
# ══════════════════════════════════════════════════════════════════

def render_space_analysis_section(
    space_df: pd.DataFrame,
    locus_dict: dict,
):
    """
    공간별 분석 섹션 렌더링.

    Enriched locus 정보(dwell_category, peak_hour, max_concurrent_occupancy 등)를
    활용하여 공간 카드 그리드를 표시.
    """
    st.markdown(section_header("공간별 분석"), unsafe_allow_html=True)
    st.caption("공간 유형, 혼잡도, 체류 특성을 한눈에 파악")

    if space_df.empty or not locus_dict:
        st.info("공간 데이터가 없습니다.")
        return

    # 카테고리별 필터
    categories = ["전체"]
    cat_counts = {}
    for lid, info in locus_dict.items():
        cat = info.get("dwell_category", "UNKNOWN")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat in ["LONG_STAY", "SHORT_STAY", "TRANSIT", "HAZARD_ZONE", "ADMIN"]:
        if cat in cat_counts:
            categories.append(cat)

    col_filter, col_summary = st.columns([1, 3])

    with col_filter:
        selected_cat = st.selectbox(
            "유형 필터",
            categories,
            key="space_cat_filter",
            format_func=lambda x: f"{x} ({cat_counts.get(x, 0)})" if x != "전체" else "전체"
        )

    # 카테고리별 요약 카드
    with col_summary:
        cat_chips = []
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            style = DWELL_CATEGORY_STYLES.get(cat, DWELL_CATEGORY_STYLES["UNKNOWN"])
            cat_chips.append(
                f"<span style='background:{style['bg']}; color:{style['color']}; "
                f"padding:3px 10px; border-radius:4px; margin-right:6px; font-size:0.8rem;'>"
                f"{style['icon']} {style['label']} {cnt}</span>"
            )
        st.markdown(
            f"<div style='display:flex; flex-wrap:wrap; gap:4px;'>{''.join(cat_chips)}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # 공간 목록 생성
    space_list = []
    for _, row in space_df.iterrows():
        locus_id = row.get("locus_id", "")
        if not locus_id:
            continue
        info = locus_dict.get(locus_id, {})
        dwell_cat = info.get("dwell_category", "UNKNOWN")

        if selected_cat != "전체" and dwell_cat != selected_cat:
            continue

        space_list.append({
            "locus_id": locus_id,
            "locus_name": info.get("locus_name", locus_id),
            "dwell_category": dwell_cat,
            "hazard_level": info.get("hazard_level", "medium"),
            "avg_dwell_minutes": info.get("avg_dwell_minutes") or 0,
            "peak_hour": info.get("peak_hour") or 0,
            "max_concurrent_occupancy": info.get("max_concurrent_occupancy") or 50,
            "current_workers": row.get("total_workers") or row.get("unique_workers") or 0,
            "avg_ewi": row.get("avg_ewi") or 0,
            "temporal_pattern": info.get("temporal_pattern", "uniform"),
        })

    if not space_list:
        st.info("해당 유형의 공간이 없습니다.")
        return

    # 혼잡률 순 정렬
    for s in space_list:
        max_occ = s.get("max_concurrent_occupancy") or 50
        s["occupancy_pct"] = min(100, s["current_workers"] / max_occ * 100) if max_occ > 0 else 0

    space_list = sorted(space_list, key=lambda x: -x["occupancy_pct"])

    # 3열 그리드로 공간 카드 표시
    st.markdown(f"**{len(space_list)}개 공간** (혼잡률 순)")
    cols = st.columns(3)
    for i, space in enumerate(space_list[:12]):  # 최대 12개
        with cols[i % 3]:
            st.markdown(
                render_space_card(space["locus_id"], locus_dict, space["current_workers"]),
                unsafe_allow_html=True,
            )

    # 상세 테이블
    with st.expander("전체 공간 상세 테이블", expanded=False):
        df_display = pd.DataFrame(space_list)
        if not df_display.empty:
            df_display = df_display.rename(columns={
                "locus_name": "공간",
                "dwell_category": "유형",
                "current_workers": "현재 인원",
                "max_concurrent_occupancy": "최대 수용",
                "occupancy_pct": "혼잡률(%)",
                "avg_dwell_minutes": "평균 체류(분)",
                "peak_hour": "피크 시간",
                "temporal_pattern": "패턴",
            })
            cols_to_show = ["공간", "유형", "현재 인원", "최대 수용", "혼잡률(%)", "평균 체류(분)", "피크 시간"]
            cols_available = [c for c in cols_to_show if c in df_display.columns]
            st.dataframe(
                df_display[cols_available].round(1),
                use_container_width=True,
                hide_index=True,
            )
