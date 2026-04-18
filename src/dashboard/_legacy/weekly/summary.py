"""
Weekly Summary 모듈
===================
주간 KPI 카드 + 출입 트렌드 + 요약 섹션.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import metric_card, section_header, PLOTLY_DARK
from src.utils.weather import date_label_short
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH
# ★ 색상 단일 소스
from src.dashboard.styles import CHART_COLORS

logger = logging.getLogger(__name__)

_DARK = PLOTLY_DARK


def render_weekly_kpi(
    worker_df: pd.DataFrame,
    metas: list[dict],
    selected_dates: list[str],
) -> tuple[bool, bool]:
    """주간 요약 KPI 렌더링.

    Returns:
        (has_ewi, has_cre) 튜플
    """
    st.markdown(section_header("주간 요약 KPI"), unsafe_allow_html=True)

    # 선택 기간 필터링
    if selected_dates:
        date_set = set(selected_dates)
        if "date" in worker_df.columns:
            worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", m.get("date", "")) in date_set]

    # 생체인식 출입인원 (참값, meta 기반)
    if metas:
        access_vals = [m.get("total_workers_access", 0) for m in metas]
        # ★ T-Ward 보유자 수 (tward_holders) 사용 — BLE 활성(total_workers_move)은 헬멧 방치 포함
        tward_vals = [m.get("tward_holders", m.get("total_workers_move", 0)) for m in metas]
        avg_access = sum(access_vals) / len(access_vals)
        max_access = max(access_vals)
        min_access = min(access_vals)
        avg_tward = sum(tward_vals) / len(tward_vals)
        max_tward = max(tward_vals)
        min_tward = min(tward_vals)
        # ★ 착용률 = T-Ward 보유자 / 타각기 출입인원
        avg_tward_rate = avg_tward / avg_access * 100 if avg_access > 0 else 0
    else:
        # fallback: worker_df 기반
        daily_counts = (
            worker_df.groupby("date")["user_no"].nunique()
            if "date" in worker_df.columns
            else pd.Series([len(worker_df)])
        )
        avg_access = avg_tward = daily_counts.mean()
        max_access = max_tward = daily_counts.max()
        min_access = min_tward = daily_counts.min()
        avg_tward_rate = 100.0

    # 평균 근무시간
    avg_wm_tward = worker_df["work_minutes"].mean() / 60 if "work_minutes" in worker_df.columns else 0

    confined_events = int((worker_df["confined_minutes"] > 0).sum()) if "confined_minutes" in worker_df.columns else 0

    # EWI/CRE 주간 평균
    has_ewi = "ewi" in worker_df.columns
    has_cre = "cre" in worker_df.columns

    # 카드 행 1: 출입인원
    st.markdown("##### 생체인식 출입인원 (참값)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("일평균 출입인원", f"{avg_access:,.0f}명"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("최다 출입 (Max)", f"{max_access:,}명"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("최소 출입 (Min)", f"{min_access:,}명"), unsafe_allow_html=True)

    # 카드 행 2: T-Ward 보유자
    st.markdown("##### T-Ward 보유자 (헬멧 지급)")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(metric_card("일평균 T-Ward 보유자", f"{avg_tward:,.0f}명"), unsafe_allow_html=True)
    with d2:
        st.markdown(metric_card("최다 보유자 (Max)", f"{max_tward:,}명"), unsafe_allow_html=True)
    with d3:
        st.markdown(metric_card("최소 보유자 (Min)", f"{min_tward:,}명"), unsafe_allow_html=True)
    with d4:
        rate_color = CHART_COLORS["critical"] if avg_tward_rate < 50 else (CHART_COLORS["medium"] if avg_tward_rate < 70 else CHART_COLORS["rest"])
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{rate_color}'>"
            f"{avg_tward_rate:.1f}%</div><div class='metric-label'>평균 T-Ward 착용률</div></div>",
            unsafe_allow_html=True,
        )

    # 카드 행 3: 지표 요약
    st.markdown("##### 지표 요약")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(metric_card("참여 업체", f"{worker_df['company_name'].nunique()}개"), unsafe_allow_html=True)
    with e2:
        st.markdown(metric_card("T-Ward 평균 근무", f"{avg_wm_tward:.1f}h"), unsafe_allow_html=True)
    with e3:
        if has_ewi:
            avg_ewi = round(worker_df["ewi"].mean(), 3)
            color = CHART_COLORS["critical"] if avg_ewi >= EWI_HIGH else (CHART_COLORS["medium"] if avg_ewi >= 0.2 else CHART_COLORS["rest"])
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
                f"{avg_ewi:.3f}</div><div class='metric-label'>기간 평균 EWI</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(metric_card("평균 EWI", "미처리"), unsafe_allow_html=True)
    with e4:
        color = CHART_COLORS["critical"] if confined_events > 0 else CHART_COLORS["rest"]
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
            f"{confined_events}명</div><div class='metric-label'>밀폐공간 진입 (연인원)</div></div>",
            unsafe_allow_html=True,
        )

    return has_ewi, has_cre


def render_weekly_site_status(
    worker_df: pd.DataFrame,
    metas: list[dict],
    date_list: list[str],
    has_ewi: bool,
    has_cre: bool,
):
    """주간 건설현장 핵심 현황 요약."""
    # 선택 기간 필터링
    if date_list:
        date_set = set(date_list)
        if "date" in worker_df.columns:
            worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", m.get("date", "")) in date_set]

    alerts = []

    # 1) 장시간 근무 트렌드
    if "work_minutes" in worker_df.columns and "date" in worker_df.columns:
        overtime_daily = worker_df[worker_df["work_minutes"] >= 720].groupby("date").size()
        total_overtime = overtime_daily.sum() if not overtime_daily.empty else 0
        if total_overtime > 0:
            max_day = overtime_daily.idxmax() if not overtime_daily.empty else ""
            max_day_fmt = f"{max_day[4:6]}/{max_day[6:]}" if max_day else ""
            alerts.append((
                "clock", f"장시간 근무 총 {total_overtime}명/일 (12h+)",
                f"최다일: {max_day_fmt} ({overtime_daily.max()}명)",
                "#FF6B35",
            ))

    # 2) 밀폐공간 진입 추세
    if "confined_minutes" in worker_df.columns and "date" in worker_df.columns:
        confined_daily = worker_df[worker_df["confined_minutes"] > 0].groupby("date").size()
        total_confined = confined_daily.sum() if not confined_daily.empty else 0
        if total_confined > 0:
            trend_str = "증가" if len(confined_daily) > 1 and confined_daily.iloc[-1] > confined_daily.iloc[0] else "안정"
            alerts.append((
                "lock", f"밀폐공간 진입 총 {total_confined}명/일",
                f"추세: {trend_str}",
                "#FF4C4C",
            ))

    # 3) 고위험(CRE>=CRE_HIGH) 누적 — constants.py 기준
    if has_cre and "date" in worker_df.columns:
        high_cre_daily = worker_df[worker_df["cre"].fillna(0) >= CRE_HIGH].groupby("date").size()
        total_high = high_cre_daily.sum() if not high_cre_daily.empty else 0
        if total_high > 0:
            alerts.append((
                "warning", f"고위험 작업자 총 {total_high}명/일 (CRE>={CRE_HIGH:.1f})",
                f"일평균 {total_high / len(date_list):.1f}명",
                "#FF4C4C",
            ))

    # 4) 출입인원 변동
    if metas:
        access_vals = [m.get("total_workers_access", 0) for m in metas]
        if len(access_vals) >= 2 and max(access_vals) > 0:
            variation = (max(access_vals) - min(access_vals)) / max(access_vals) * 100
            if variation > 30:
                alerts.append((
                    "chart", f"출입인원 변동폭 {variation:.0f}%",
                    f"최소 {min(access_vals):,}명 ~ 최대 {max(access_vals):,}명",
                    "#FFB300",
                ))

    if not alerts:
        st.markdown(
            "<div style='background:#0D2A1A; border:1px solid #00C897; "
            "border-radius:10px; padding:12px 16px;'>"
            "<span style='color:#00C897; font-size:0.92rem; font-weight:600;'>"
            "기간 내 특이사항 없음</span></div>",
            unsafe_allow_html=True,
        )
        return

    cols = st.columns(min(len(alerts), 4))
    for i, (icon_type, title, detail, color) in enumerate(alerts):
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#111820; border-left:4px solid {color}; "
                f"border-radius:8px; padding:10px 14px;'>"
                f"<div style='color:{color}; font-size:0.85rem; font-weight:600;'>"
                f"{title}</div>"
                f"<div style='color:#9AB5D4; font-size:0.78rem; margin-top:3px;'>"
                f"{detail}</div></div>",
                unsafe_allow_html=True,
            )


def render_daily_trend(
    worker_df: pd.DataFrame,
    date_list: list[str],
    metas: list[dict],
):
    """일별 작업자 수 트렌드."""
    if "date" not in worker_df.columns:
        st.info("날짜 정보 없음")
        return

    # 선택 기간 필터링
    if date_list:
        date_set = set(date_list)
        worker_df = worker_df[worker_df["date"].isin(date_set)]
        metas = [m for m in metas if m.get("date_str", m.get("date", "")) in date_set]

    # meta 기반 일별 출입 통계
    meta_rows = []
    for m in metas:
        d = m.get("date_str", m.get("date", ""))
        meta_rows.append({
            "date": d,
            "date_fmt": date_label_short(d),
            "access_workers": m.get("total_workers_access", 0),
            "tward_move_workers": m.get("total_workers_move", 0),
            "tward_holders": m.get("tward_holders", 0),
        })
    meta_df = pd.DataFrame(meta_rows).sort_values("date").reset_index(drop=True)

    if not meta_df.empty:
        # ★ T-Ward 착용률 = T-Ward 보유자 / 타각기 출입인원
        # (기존: BLE 활성 작업자 / 출입인원 → 헬멧 방치로 100% 초과 가능했음)
        meta_df["tward_rate"] = meta_df.apply(
            lambda r: round(r["tward_holders"] / r["access_workers"] * 100, 1)
            if r["access_workers"] > 0 else 0.0,
            axis=1,
        )
        # T-Ward 활성률 = BLE 활성 작업자 / T-Ward 보유자
        meta_df["tward_active_rate"] = meta_df.apply(
            lambda r: round(r["tward_move_workers"] / r["tward_holders"] * 100, 1)
            if r["tward_holders"] > 0 else 0.0,
            axis=1,
        )

        # 출입인원 + T-Ward 이동 작업자 막대 차트
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=meta_df["date_fmt"], y=meta_df["access_workers"],
            name="생체인식 출입인원", marker_color=CHART_COLORS["access"],
            hovertemplate="날짜: %{x}<br>출입인원(생체인식): %{y:,}명<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=meta_df["date_fmt"], y=meta_df["tward_move_workers"],
            name="T-Ward 이동 작업자", marker_color=CHART_COLORS["tward"],
            hovertemplate="날짜: %{x}<br>T-Ward 이동: %{y:,}명<extra></extra>",
        ))
        fig.update_layout(
            title="일별 출입인원 / T-Ward 이동 작업자",
            xaxis_title="날짜", yaxis_title="인원 수",
            barmode="group",
            legend=dict(font=dict(color="#C8D6E8")),
            **_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

        # T-Ward 착용률 + 활성률 라인 차트
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=meta_df["date_fmt"], y=meta_df["tward_rate"],
            mode="lines+markers+text",
            name="T-Ward 착용률 (보유/출입)",
            line=dict(color=CHART_COLORS["medium"], width=2),
            marker=dict(size=8),
            text=meta_df["tward_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            textfont=dict(color=CHART_COLORS["medium"], size=10),
            hovertemplate="날짜: %{x}<br>착용률: %{y:.1f}%<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=meta_df["date_fmt"], y=meta_df["tward_active_rate"].clip(upper=100),
            mode="lines+markers",
            name="T-Ward 활성률 (BLE활성/보유)",
            line=dict(color=CHART_COLORS["tward"], width=1.5, dash="dot"),
            marker=dict(size=6),
            hovertemplate="날짜: %{x}<br>활성률: %{y:.1f}%<extra></extra>",
        ))
        fig2.update_layout(
            title="일별 T-Ward 착용률 (T-Ward 보유자 / 타각기 출입인원)",
            xaxis_title="날짜", yaxis_title="비율 (%)",
            yaxis=dict(range=[0, 105]),
            legend=dict(font=dict(color="#C8D6E8"), orientation="h", y=-0.15),
            **_DARK,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 밀폐공간 진입 트렌드
    daily = worker_df.groupby("date").agg(
        confined_count=("confined_minutes", lambda x: (x > 0).sum()),
    ).reset_index()
    daily["date_fmt"] = daily["date"].apply(date_label_short)

    if daily["confined_count"].sum() > 0:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=daily["date_fmt"], y=daily["confined_count"],
            name="밀폐공간 진입 작업자", marker_color=CHART_COLORS["confined_space"],
            hovertemplate="날짜: %{x}<br>진입 작업자: %{y}명<extra></extra>",
        ))
        fig3.update_layout(
            title="일별 밀폐공간 진입 작업자 수",
            legend=dict(font=dict(color="#C8D6E8")),
            **_DARK,
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_ewi_cre_trend(
    worker_df: pd.DataFrame,
    date_list: list[str],
    has_ewi: bool,
    has_cre: bool,
):
    """일별 EWI / CRE 트렌드."""
    if not has_ewi and not has_cre:
        st.info("EWI/CRE 데이터가 없습니다. 파이프라인에서 데이터를 재처리하세요.")
        return

    # 선택 기간 필터링
    if "date" in worker_df.columns and date_list:
        date_set = set(date_list)
        worker_df = worker_df[worker_df["date"].isin(date_set)]

    # 지표 설명 토글 — constants.py 상수 동적 참조
    from src.metrics.constants import SII_HIGH
    with st.expander("EWI / CRE / SII 지표 설명", expanded=False):
        st.markdown(f"""
| 지표 | 의미 | 범위 | 고위험 기준 |
|------|------|------|------------|
| **EWI** | Effective Work Intensity | 0~1 | >= {EWI_HIGH:.1f} |
| **CRE** | Combined Risk Exposure | 0~1 | >= {CRE_HIGH:.1f} |
| **SII** | Site Intensity-Risk Index | 0~1 | >= {SII_HIGH:.1f} |

**EWI** = (고활성*1.0 + 저활성*0.5 + 대기*0.2) / 근무시간
**CRE** = 0.45*개인위험 + 0.40*공간위험 + 0.15*밀집도
**SII** = EWI * 공간위험 정규화값
        """)

    if "date" not in worker_df.columns:
        st.info("날짜 정보 없음")
        return

    # 일별 EWI/CRE 평균 계산
    agg_cols = {}
    if has_ewi:
        agg_cols["avg_ewi"] = ("ewi", "mean")
        agg_cols["high_ewi_count"] = ("ewi", lambda x: (x.fillna(0) >= EWI_HIGH).sum())
    if has_cre:
        agg_cols["avg_cre"] = ("cre", "mean")
        agg_cols["high_cre_count"] = ("cre", lambda x: (x.fillna(0) >= CRE_HIGH).sum())
    if "sii" in worker_df.columns:
        agg_cols["avg_sii"] = ("sii", "mean")

    daily = worker_df.groupby("date").agg(**agg_cols).reset_index()
    daily["date_fmt"] = daily["date"].apply(date_label_short)

    # EWI/CRE/SII 라인 차트 — CHART_COLORS 단일 소스
    fig = go.Figure()
    if has_ewi:
        fig.add_trace(go.Scatter(
            x=daily["date_fmt"], y=daily["avg_ewi"].round(3),
            mode="lines+markers+text", name="평균 EWI",
            line=dict(color=CHART_COLORS["ewi"], width=2), marker=dict(size=8),
            text=daily["avg_ewi"].apply(lambda v: f"{v:.3f}"),
            textposition="top center", textfont=dict(color=CHART_COLORS["ewi"]),
        ))
    if has_cre:
        fig.add_trace(go.Scatter(
            x=daily["date_fmt"], y=daily["avg_cre"].round(3),
            mode="lines+markers+text", name="평균 CRE",
            line=dict(color=CHART_COLORS["cre"], width=2), marker=dict(size=8),
            text=daily["avg_cre"].apply(lambda v: f"{v:.3f}"),
            textposition="bottom center", textfont=dict(color=CHART_COLORS["cre"]),
        ))
    if "avg_sii" in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["date_fmt"], y=daily["avg_sii"].round(3),
            mode="lines+markers", name="평균 SII",
            line=dict(color=CHART_COLORS["sii"], width=2, dash="dot"), marker=dict(size=6),
        ))
    fig.add_hline(y=EWI_HIGH, line_dash="dash", line_color=CHART_COLORS["low"], line_width=1,
                  annotation_text=f"고위험 기준({EWI_HIGH:.1f})",
                  annotation_font_color=CHART_COLORS["low"])
    fig.update_layout(
        title="일별 평균 EWI / CRE / SII 트렌드",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(font=dict(color="#C8D6E8")),
        **_DARK, height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 고위험 작업자 수 트렌드
    fig2 = go.Figure()
    if "high_ewi_count" in daily.columns:
        fig2.add_trace(go.Bar(
            x=daily["date_fmt"], y=daily["high_ewi_count"],
            name=f"고강도 작업자 (EWI>={EWI_HIGH:.1f})", marker_color=CHART_COLORS["ewi"],
        ))
    if "high_cre_count" in daily.columns:
        fig2.add_trace(go.Bar(
            x=daily["date_fmt"], y=daily["high_cre_count"],
            name=f"고위험 작업자 (CRE>={CRE_HIGH:.1f})", marker_color=CHART_COLORS["critical"],
        ))
    fig2.update_layout(
        title="일별 고강도/고위험 작업자 수",
        barmode="group",
        legend=dict(font=dict(color="#C8D6E8")),
        **_DARK, height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)
