"""
Daily Tab - Company Section
===========================
업체별 분석 탭 관련 함수들.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    section_header, metric_card,
    PLOTLY_DARK, PLOTLY_LEGEND,
)
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, COMPANY_MIN_WORKERS
# ★ 색상 단일 소스
from src.dashboard.styles import CHART_COLORS

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def _render_shift_comparison(worker_df: pd.DataFrame, meta: dict, has_ewi: bool, has_cre: bool):
    """주간(day) / 야간(night) 근무 비교 섹션."""
    day_cnt = meta.get("day_workers", 0)
    night_cnt = meta.get("night_workers", 0)
    missing = meta.get("missing_exit_workers", 0)
    supplement = meta.get("night_supplement_records", 0)

    if day_cnt == 0 and night_cnt == 0:
        return

    st.markdown("---")
    st.markdown("### sun/moon 주간 / 야간 근무 현황")

    # 인원 요약 배너
    sup_note = f" | D+1 BLE 보완: {supplement:,}건" if supplement > 0 else ""
    missing_color = "#F5A623" if missing > 0 else "#00C897"
    st.markdown(
        f"<div style='background:#0D1F2D; border:1px solid #1E3A5F; border-radius:8px; "
        f"padding:10px 16px; margin-bottom:12px; font-size:0.85rem; color:#C8D6E8;'>"
        f"sun <b>주간</b>: {day_cnt:,}명 &nbsp;|&nbsp; "
        f"moon <b>야간</b>: {night_cnt:,}명 &nbsp;|&nbsp; "
        f"<span style='color:{missing_color}'>? 퇴근미기록: {missing:,}명</span>"
        f"{sup_note}</div>",
        unsafe_allow_html=True,
    )

    # worker_df에 shift_type 있어야 비교 가능
    if "shift_type" not in worker_df.columns:
        return

    day_df = worker_df[worker_df["shift_type"] == "day"]
    night_df = worker_df[worker_df["shift_type"] == "night"]

    if day_df.empty or night_df.empty:
        return

    # KPI 비교 테이블
    def _mean(df, col, default=0.0):
        return round(df[col].mean(), 3) if col in df.columns and not df.empty else default

    metrics = {
        "근무시간(평균, 분)": ("work_minutes", "{:.0f}"),
        "유효BLE(평균, 분)": ("valid_ble_minutes", "{:.0f}"),
        "방문 구역 수(평균)": ("unique_loci", "{:.1f}"),
        "이동 횟수(평균)": ("transition_count", "{:.1f}"),
        "밀폐공간 진입(명)": ("confined_minutes", None),  # special
        "고압전 구역(명)": ("high_voltage_minutes", None),
    }
    if has_ewi:
        metrics["평균 EWI"] = ("ewi", "{:.3f}")
    if has_cre:
        metrics["평균 CRE"] = ("cre", "{:.3f}")
        metrics["평균 피로도"] = ("fatigue_score", "{:.3f}")

    rows = []
    for label, (col, fmt) in metrics.items():
        if col not in worker_df.columns:
            continue
        if label in ("밀폐공간 진입(명)", "고압전 구역(명)"):
            dv = int((day_df[col] > 0).sum())
            nv = int((night_df[col] > 0).sum())
            diff = nv - dv
            rows.append({
                "지표": label,
                "sun 주간": f"{dv}명",
                "moon 야간": f"{nv}명",
                "야간-주간": f"+{diff}명" if diff > 0 else str(diff) + "명",
                "_dv": dv, "_nv": nv,
            })
        else:
            dv = _mean(day_df, col)
            nv = _mean(night_df, col)
            diff = round(nv - dv, 3)
            sign = "+" if diff > 0 else ""
            rows.append({
                "지표": label,
                "sun 주간": fmt.format(dv) if fmt else str(dv),
                "moon 야간": fmt.format(nv) if fmt else str(nv),
                "야간-주간": f"{sign}{fmt.format(diff)}" if fmt else f"{sign}{diff}",
                "_dv": dv, "_nv": nv,
            })

    cmp_df = pd.DataFrame(rows)

    # 시각화: EWI/CRE 야간 vs 주간 bar 비교
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**지표 비교**")
        disp = cmp_df[["지표", "sun 주간", "moon 야간", "야간-주간"]]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with col_b:
        # EWI / CRE / 피로도 비교 차트
        chart_metrics = {}
        if has_ewi and "ewi" in worker_df.columns:
            chart_metrics["EWI"] = (_mean(day_df, "ewi"), _mean(night_df, "ewi"))
        if has_cre and "cre" in worker_df.columns:
            chart_metrics["CRE"] = (_mean(day_df, "cre"), _mean(night_df, "cre"))
        if has_cre and "fatigue_score" in worker_df.columns:
            chart_metrics["피로도"] = (_mean(day_df, "fatigue_score"), _mean(night_df, "fatigue_score"))

        if chart_metrics:
            labels_list = list(chart_metrics.keys())
            day_vals = [chart_metrics[k][0] for k in labels_list]
            night_vals = [chart_metrics[k][1] for k in labels_list]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="sun 주간", x=labels_list, y=day_vals,
                marker_color=CHART_COLORS["work_zone"],   # 주간 = accent
                text=[f"{v:.3f}" for v in day_vals],
                textposition="outside", textfont=dict(color="#C8D6E8", size=11),
            ))
            fig.add_trace(go.Bar(
                name="moon 야간", x=labels_list, y=night_vals,
                marker_color=CHART_COLORS["sii"],         # 야간 = orange
                text=[f"{v:.3f}" for v in night_vals],
                textposition="outside", textfont=dict(color="#C8D6E8", size=11),
            ))
            fig.update_layout(
                barmode="group",
                title="주간 vs 야간 - EWI / CRE / 피로도",
                paper_bgcolor="#1A2A3A", plot_bgcolor="#111820",
                font_color="#C8D6E8",
                legend=dict(font=dict(color="#C8D6E8")),
                height=280, margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # 야간 고위험 경보 — constants.py의 CRE_HIGH 기준
    if has_cre and "cre" in night_df.columns:
        night_high = night_df[night_df["cre"].fillna(0) >= CRE_HIGH]
        if not night_high.empty:
            st.warning(
                f"moon 야간 고위험(CRE>={CRE_HIGH:.1f}) 작업자 **{len(night_high)}명** - "
                f"야간 작업은 피로/조명/감독 부재로 위험 가중. 즉시 점검 필요."
            )


def render_company(company_df, worker_df, has_ewi, has_cre):
    """업체별 분석 탭 전체 렌더링."""
    st.markdown(section_header("업체별 분석"), unsafe_allow_html=True)

    if company_df is None or company_df.empty:
        st.info("업체 데이터 없음")
        return

    # 업체별 작업자 수 + EWI
    top20 = company_df.nlargest(20, "worker_count")

    color_col = "avg_ewi" if (has_ewi and "avg_ewi" in top20.columns) else "avg_work_zone_minutes"
    color_label = "평균 EWI" if color_col == "avg_ewi" else "평균 작업시간(분)"

    fig = px.bar(
        top20, x="worker_count", y="company_name", orientation="h",
        color=color_col,
        color_continuous_scale=["#00C897", "#F5A623", "#FF4C4C"] if color_col == "avg_ewi" else "Blues",
        title=f"업체별 작업자 수 (색상 = {color_label})",
        labels={"worker_count": "작업자 수", "company_name": "업체"},
        text=top20["worker_count"].apply(lambda v: f"{v}명"),
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500, **_DARK,
                      coloraxis_colorbar=dict(tickfont=dict(color="#C8D6E8")))
    st.plotly_chart(fig, use_container_width=True)

    # EWI vs CRE 업체 비교
    if has_ewi and has_cre and "avg_ewi" in company_df.columns and "avg_cre" in company_df.columns:
        comp_plot = company_df[company_df["worker_count"] >= COMPANY_MIN_WORKERS["small"]].copy()
        fig2 = px.scatter(
            comp_plot, x="avg_ewi", y="avg_cre",
            size="worker_count", hover_data={"company_name": True, "worker_count": True},
            color="worker_count", color_continuous_scale="Blues",
            title="업체별 평균 EWI vs CRE (크기 = 작업자 수, 3명 이상)",
            labels={"avg_ewi": "평균 EWI", "avg_cre": "평균 CRE"},
        )
        fig2.add_hline(y=0.3, line_dash="dash", line_color="#F5A623")
        fig2.add_vline(x=0.2, line_dash="dash", line_color="#F5A623")
        fig2.update_layout(**_DARK, height=400,
                           coloraxis_colorbar=dict(tickfont=dict(color="#C8D6E8")))
        st.plotly_chart(fig2, use_container_width=True)

    # 밀폐공간 진입 업체
    if "confined_workers" in company_df.columns:
        confined_co = company_df[company_df["confined_workers"] > 0].sort_values(
            "total_confined_minutes", ascending=False)
        if not confined_co.empty:
            st.markdown("**warning 밀폐공간 진입 업체**")
            disp_cols = ["company_name", "worker_count", "confined_workers", "total_confined_minutes"]
            st.dataframe(
                confined_co[disp_cols].rename(columns={
                    "company_name": "업체", "worker_count": "전체작업자",
                    "confined_workers": "밀폐진입", "total_confined_minutes": "밀폐총시간(분)"
                }),
                use_container_width=True, hide_index=True,
            )

    # 전체 업체 테이블
    with st.expander("전체 업체 데이터 보기"):
        show_cols = [c for c in [
            "company_name", "worker_count", "total_person_minutes",
            "avg_work_zone_minutes", "avg_ewi", "avg_cre",
            "confined_workers", "total_confined_minutes",
        ] if c in company_df.columns]
        rename = {
            "company_name": "업체", "worker_count": "작업자",
            "total_person_minutes": "총체류(인분)", "avg_work_zone_minutes": "평균작업(분)",
            "avg_ewi": "평균EWI", "avg_cre": "평균CRE",
            "confined_workers": "밀폐진입", "total_confined_minutes": "밀폐시간(분)",
        }
        st.dataframe(company_df[show_cols].rename(columns=rename), use_container_width=True, hide_index=True)
