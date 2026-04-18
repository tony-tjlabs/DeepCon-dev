"""
Agentic AI 탭 — 공간 지능 예측
===============================
Intelligence Engine (EWI/CRE/SII) + Deep Space (Transformer) 통합 분석.

5개 서브탭:
  1. 혼잡도 예측 — 공간별 미래 인원 분포
  2. 병목 예측 — 유입/유출 불균형 공간
  3. 안전 위험 예측 — HAZARD 구역 진입 위험 작업자
  4. 생산성 예측 — 피로/이동 패턴 기반 생산성 하락 예측
  5. 예측 정확도 — 예측 vs 실제 비교 추적
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, RADIUS, SPACING, PLOTLY_DARK
from src.dashboard.deep_space.helpers import (
    extract_worker_sequences, render_card,
)
from src.dashboard.deep_space.model_loader import (
    load_journey_data, load_worker_data, load_locus_info,
)
from src.utils.anonymizer import mask_name
from src.utils.weather import date_label

logger = logging.getLogger(__name__)


# ─── 심각도 색상 / 아이콘 ─────────────────────────────────────

SEVERITY_STYLE = {
    "critical": {"icon": "🔴", "color": "#FF4C4C", "bg": "#3A1A1A", "label": "긴급"},
    "high":     {"icon": "🟠", "color": "#FFB300", "bg": "#3A2A1A", "label": "경고"},
    "medium":   {"icon": "🟡", "color": "#FFE066", "bg": "#3A3A1A", "label": "주의"},
    "low":      {"icon": "🔵", "color": "#00AEEF", "bg": "#1A2A3A", "label": "참고"},
}


def _severity_badge(severity: str) -> str:
    s = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["low"])
    return (
        f"<span style='background:{s['bg']}; color:{s['color']}; "
        f"font-size:0.72rem; font-weight:700; padding:2px 8px; "
        f"border-radius:10px;'>{s['icon']} {s['label']}</span>"
    )


def _kpi_card(label: str, value: str, color: str = "") -> str:
    c = color or COLORS["accent"]
    return (
        f"<div style='background:{COLORS['card_bg']}; border:1px solid {COLORS['border']}; "
        f"border-radius:{RADIUS['md']}; padding:12px 16px; text-align:center;'>"
        f"<div style='color:{COLORS['text_muted']}; font-size:0.78rem;'>{label}</div>"
        f"<div style='color:{c}; font-size:1.5rem; font-weight:700; margin-top:2px;'>{value}</div>"
        f"</div>"
    )


# ─── 메인 렌더링 ──────────────────────────────────────────────

def render_agentic(model, tokenizer, sector_id: str, dates: list[str]):
    """Agentic AI 탭 메인."""
    if not dates:
        st.warning("데이터가 없습니다.")
        return

    selected_date = st.selectbox(
        "분석 날짜", dates, index=len(dates) - 1,
        key="ds_agentic_date", format_func=date_label,
    )

    journey_df = load_journey_data(sector_id, selected_date)
    worker_df = load_worker_data(sector_id, selected_date)
    locus_info = load_locus_info(sector_id)

    if journey_df is None:
        st.warning("Journey 데이터를 로드할 수 없습니다. (로컬 전용)")
        return

    # 이전 날짜 worker_df 로드 (CRE 추세용)
    prev_worker_df = None
    if len(dates) >= 2:
        date_idx = dates.index(selected_date) if selected_date in dates else len(dates) - 1
        if date_idx > 0:
            prev_worker_df = load_worker_data(sector_id, dates[date_idx - 1])

    # 캐시 키 (sector_id 포함)
    cache_key = f"_agentic_report_{sector_id}_{selected_date}"

    # 날짜 변경 시 이전 캐시 정리
    prev_date_key = st.session_state.get("_agentic_prev_date")
    if prev_date_key and prev_date_key != selected_date:
        old_key = f"_agentic_report_{sector_id}_{prev_date_key}"
        st.session_state.pop(old_key, None)
    st.session_state["_agentic_prev_date"] = selected_date

    # 갱신 버튼
    if st.button("🔄 분석 갱신", key="ds_agentic_refresh"):
        st.session_state.pop(cache_key, None)

    if cache_key in st.session_state:
        report = st.session_state[cache_key]
    else:
        with st.spinner("Agentic AI 분석 중... (혼잡도 / 병목 / 안전 / 생산성)"):
            from src.intelligence.spatial_predictor import SpatialPredictor
            sequences = extract_worker_sequences(journey_df)
            predictor = SpatialPredictor(
                model, tokenizer, worker_df, journey_df, locus_info, sequences,
                prev_worker_df=prev_worker_df,
            )
            report = predictor.generate_report(date_str=selected_date, sector_id=sector_id)

            # M4: Prediction Journal — 예측 저장
            try:
                from src.intelligence.prediction_journal import save_predictions
                pred_data = predictor._batch_predict(top_k=3)
                save_predictions(
                    sector_id=sector_id,
                    date_str=selected_date,
                    predictions=pred_data["predictions"],
                    current_loci=pred_data["current_loci"],
                    congestion_alerts=report.congestion_alerts,
                )
            except Exception as e:
                logger.debug(f"Prediction journal 저장 건너뜀: {e}")

        st.session_state[cache_key] = report

    # ─── M4: 미평가 예측 자동 정확도 계산 ──────────────────
    _auto_evaluate_pending(sector_id, dates, journey_df, selected_date)

    # ─── 상단 KPI ──────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(_kpi_card("분석 대상", f"{report.total_workers:,}명"), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi_card("긴급 경고", f"{report.critical_count}", COLORS["danger"]), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi_card("경고", f"{report.high_count}", COLORS["warning"]), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi_card("혼잡 예측", f"{len(report.congestion_alerts)}건", COLORS["confined"]), unsafe_allow_html=True)
    with c5:
        st.markdown(_kpi_card("안전 위험", f"{len(report.safety_alerts)}건", COLORS["danger"]), unsafe_allow_html=True)
    with c6:
        st.markdown(_kpi_card("생산성 경고", f"{len(report.productivity_alerts)}건", COLORS["warning"]), unsafe_allow_html=True)

    st.markdown("---")

    # ─── 5개 서브섹션 ─────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"혼잡도 예측 ({len(report.congestion_alerts)})",
        f"병목 예측 ({len(report.bottleneck_alerts)})",
        f"안전 위험 ({len(report.safety_alerts)})",
        f"생산성 예측 ({len(report.productivity_alerts)})",
        "예측 정확도",
    ])

    with tab1:
        _render_congestion(report)
    with tab2:
        _render_bottlenecks(report)
    with tab3:
        _render_safety(report, worker_df)
    with tab4:
        _render_productivity(report, worker_df)
    with tab5:
        _render_accuracy(sector_id)

    # ─── AI 분석 섹션 ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### AI 분석")

    # Prediction Journal 최근 정확도를 accuracy_metrics로 구성
    _accuracy_metrics: dict | None = None
    try:
        from src.intelligence.prediction_journal import get_accuracy_history
        history = get_accuracy_history(sector_id)
        if history:
            _accuracy_metrics = history[-1]
    except Exception as e:
        logger.debug("accuracy_metrics 로드 건너뜀: %s", e)

    from src.dashboard.ai_analysis import render_deep_space_ai
    render_deep_space_ai(
        prediction_df=None,
        accuracy_metrics=_accuracy_metrics,
        date_str=selected_date,
        sector_id=sector_id,
    )


# ─── 1. 혼잡도 예측 ──────────────────────────────────────────

def _render_congestion(report):
    """혼잡도 예측 서브탭."""
    alerts = report.congestion_alerts
    if not alerts:
        st.info("혼잡 경고 공간이 없습니다. (용량 대비 70% 이하)")
        return

    st.markdown("#### 혼잡 예상 공간")
    st.caption("현재 인원 + 예측 이동 = 예측 혼잡도. 용량 대비 70% 이상 공간만 표시.")

    # 바 차트
    df = pd.DataFrame([{
        "공간": a.locus_name,
        "현재": a.current_count,
        "예측": a.predicted_count,
        "수용력": a.capacity,
        "혼잡도": a.congestion_pct,
        "심각도": a.severity,
        "트렌드": a.trend,
    } for a in alerts[:15]])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["공간"], y=df["현재"],
        name="현재 인원",
        marker_color="#4A90D9",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=df["공간"], y=df["예측"],
        name="예측 인원",
        marker_color=[
            SEVERITY_STYLE.get(s, SEVERITY_STYLE["low"])["color"]
            for s in df["심각도"]
        ],
    ))
    fig.add_trace(go.Scatter(
        x=df["공간"], y=df["수용력"],
        name="수용력",
        mode="markers+lines",
        marker=dict(color=COLORS["danger"], size=8, symbol="line-ew-open"),
        line=dict(dash="dot", color=COLORS["danger"], width=2),
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        barmode="group",
        height=380,
        yaxis_title="인원 수",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 권고사항 (critical/high만)
    recs = [a for a in alerts if a.recommendation and a.severity in ("critical", "high")]
    if recs:
        with st.expander(f"권고사항 ({len(recs)}건)", expanded=True):
            for a in recs[:5]:
                sty = SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])
                st.markdown(
                    f"<div style='background:{sty['bg']}; border-left:3px solid {sty['color']}; "
                    f"padding:6px 12px; border-radius:4px; margin-bottom:6px; font-size:0.85rem;'>"
                    f"{sty['icon']} <b>{a.locus_name}</b>: {a.recommendation}</div>",
                    unsafe_allow_html=True,
                )

    # 상세 테이블
    with st.expander("상세 데이터"):
        table_data = []
        for a in alerts:
            table_data.append({
                "공간": a.locus_name,
                "현재": a.current_count,
                "예측": a.predicted_count,
                "유입": a.inflow,
                "유출": a.outflow,
                "수용력": a.capacity,
                "혼잡도": f"{a.congestion_pct:.0%}",
                "트렌드": a.trend,
                "심각도": a.severity,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ─── 2. 병목 예측 ────────────────────────────────────────────

def _render_bottlenecks(report):
    """병목 예측 서브탭."""
    alerts = report.bottleneck_alerts
    if not alerts:
        st.info("병목 경고 공간이 없습니다. (순유입 3명 미만)")
        return

    st.markdown("#### 병목 예상 공간")
    st.caption("유입 > 유출인 공간 = 사람이 쌓이는 곳. 순유입 3명 이상만 표시.")

    # Funnel 차트
    df = pd.DataFrame([{
        "공간": a.locus_name,
        "유입": a.inflow,
        "유출": a.outflow,
        "순유입": a.net_accumulation,
        "병목점수": a.bottleneck_score,
        "심각도": a.severity,
    } for a in alerts[:10]])

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["공간"], y=df["유입"],
            name="유입",
            marker_color=COLORS["success"],
        ))
        fig.add_trace(go.Bar(
            x=df["공간"], y=[-v for v in df["유출"]],
            name="유출",
            marker_color=COLORS["danger"],
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            barmode="relative",
            height=350,
            yaxis_title="인원 (유입+/유출-)",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 병목 공간 상세")
        for a in alerts[:5]:
            sty = SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])
            st.markdown(
                f"<div style='background:{sty['bg']}; border-left:3px solid {sty['color']}; "
                f"padding:8px 12px; border-radius:6px; margin-bottom:8px;'>"
                f"<div style='color:{sty['color']}; font-weight:700;'>{sty['icon']} {a.locus_name}</div>"
                f"<div style='color:{COLORS['text_muted']}; font-size:0.82rem;'>"
                f"유입 {a.inflow} / 유출 {a.outflow} → <b>순유입 +{a.net_accumulation}명</b></div>"
                f"<div style='color:{COLORS['text_dim']}; font-size:0.75rem; margin-top:4px;'>"
                f"주요 유입원: {', '.join(s for s, _ in a.upstream_sources[:3])}</div>"
                + (f"<div style='color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:3px; "
                   f"font-style:italic;'>💡 {a.recommendation}</div>" if a.recommendation else "")
                + f"</div>",
                unsafe_allow_html=True,
            )


# ─── 3. 안전 위험 예측 ───────────────────────────────────────

def _render_safety(report, worker_df):
    """안전 위험 예측 서브탭."""
    alerts = report.safety_alerts
    if not alerts:
        st.success("안전 위험 예측 경고가 없습니다.")
        st.info("ℹ️ Locus 메타데이터에 위험등급(hazard_grade ≥ 3) 구역이 없으면 안전 경고가 생성되지 않습니다. "
                "Locus 속성 편집 후 다시 확인하세요.")
        return

    st.markdown("#### 안전 위험 예측")
    st.caption("Deep Space 이동 예측 + 작업자 CRE/피로도 → 위험 구역 진입 가능성이 높은 작업자")

    # 경고 카드
    for a in alerts[:10]:
        sty = SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])
        masked = mask_name(a.user_name) if a.user_name else a.user_no

        st.markdown(
            f"<div style='background:{sty['bg']}; border-left:4px solid {sty['color']}; "
            f"padding:12px 16px; border-radius:8px; margin-bottom:10px;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div>"
            f"<span style='color:{sty['color']}; font-weight:700; font-size:1rem;'>"
            f"{sty['icon']} {masked}</span>"
            f"<span style='color:{COLORS['text_muted']}; font-size:0.82rem; margin-left:8px;'>"
            f"{a.company_name}</span>"
            f"</div>"
            f"<div>{_severity_badge(a.severity)}</div>"
            f"</div>"
            f"<div style='color:{COLORS['text']}; font-size:0.88rem; margin-top:6px;'>"
            f"{a.reason}</div>"
            f"<div style='color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:4px;'>"
            f"예측 위치: <b>{a.predicted_locus_name}</b> (확률 {a.prediction_prob:.1%}) "
            f"| CRE: {a.cre:.2f}"
            + (f" <span style='color:{COLORS['danger']};'>(+{a.cre_delta:.2f}↑)</span>" if a.cre_delta >= 0.1
               else f" ({a.cre_delta:+.2f})" if a.cre_delta != 0 else "")
            + f" | 피로도: {a.fatigue_score:.2f} "
            f"| 위험등급: {a.hazard_grade} | risk: {a.risk_score:.3f}</div>"
            + (f"<div style='color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:3px; "
               f"font-style:italic;'>💡 {a.recommendation}</div>" if a.recommendation else "")
            + f"</div>",
            unsafe_allow_html=True,
        )

    # scatter: CRE vs Fatigue (경고 작업자 표시)
    if len(alerts) >= 3:
        st.markdown("##### CRE vs 피로도 (위험 작업자)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[a.cre for a in alerts],
            y=[a.fatigue_score for a in alerts],
            mode="markers+text",
            text=[mask_name(a.user_name) if a.user_name else a.user_no for a in alerts],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["text_muted"]),
            marker=dict(
                size=[max(8, a.risk_score * 40) for a in alerts],
                color=[SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])["color"] for a in alerts],
                opacity=0.8,
                line=dict(width=1, color=COLORS["border"]),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CRE: %{x:.2f}<br>"
                "피로도: %{y:.2f}<br>"
                "예측 위치: %{customdata}<br>"
                "<extra></extra>"
            ),
            customdata=[a.predicted_locus_name for a in alerts],
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["warning"], opacity=0.5,
                      annotation_text="피로 경계", annotation_position="top left")
        fig.add_vline(x=0.4, line_dash="dash", line_color=COLORS["danger"], opacity=0.5,
                      annotation_text="CRE 경계", annotation_position="top right")
        fig.update_layout(
            **PLOTLY_DARK,
            height=380,
            xaxis_title="CRE (누적 위험도)",
            yaxis_title="피로도",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── 4. 생산성 예측 ──────────────────────────────────────────

def _render_productivity(report, worker_df):
    """생산성 예측 서브탭."""
    alerts = report.productivity_alerts
    if not alerts:
        st.success("생산성 하락 예측 경고가 없습니다.")
        return

    st.markdown("#### 생산성 하락 예측")
    st.caption("피로도, 이동 빈도, 작업공간 비율 기반 EWI 하락 예측")

    # 유형별 분류
    type_labels = {
        "fatigue_decline": ("피로 누적", COLORS["danger"]),
        "high_movement": ("과다 이동", COLORS["warning"]),
        "low_utilization": ("저활용", COLORS["accent"]),
    }

    # 유형별 카운트 KPI
    type_counts = {}
    for a in alerts:
        type_counts[a.risk_type] = type_counts.get(a.risk_type, 0) + 1

    cols = st.columns(len(type_counts) or 1)
    for i, (rtype, count) in enumerate(type_counts.items()):
        label, color = type_labels.get(rtype, (rtype, COLORS["text"]))
        with cols[i % len(cols)]:
            st.markdown(_kpi_card(label, f"{count}명", color), unsafe_allow_html=True)

    st.markdown("")

    # 업체 단위 요약
    company_summaries = report.company_summaries if hasattr(report, "company_summaries") else []
    if company_summaries:
        st.markdown("##### 업체별 생산성 경고")
        for cs in company_summaries[:10]:
            sty = SEVERITY_STYLE.get(cs.severity, SEVERITY_STYLE["low"])
            risk_label, _ = type_labels.get(cs.dominant_risk, (cs.dominant_risk, COLORS["text"]))
            st.markdown(
                f"<div style='background:{sty['bg']}; border-left:3px solid {sty['color']}; "
                f"padding:8px 12px; border-radius:6px; margin-bottom:6px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='color:{sty['color']}; font-weight:700;'>{sty['icon']} {cs.company_name}</span>"
                f"<span style='color:{COLORS['text_muted']}; font-size:0.78rem;'>"
                f"{cs.alert_count}/{cs.worker_count}명 ({cs.alert_pct:.0%})</span></div>"
                f"<div style='color:{COLORS['text_muted']}; font-size:0.82rem; margin-top:4px;'>"
                f"평균 EWI {cs.avg_ewi:.2f} | 평균 피로도 {cs.avg_fatigue:.2f} | 주요 유형: {risk_label}</div>"
                + (f"<div style='color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:3px; "
                   f"font-style:italic;'>💡 {cs.recommendation}</div>" if cs.recommendation else "")
                + f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

    # 작업자 테이블
    table_data = []
    for a in alerts[:30]:
        masked = mask_name(a.user_name) if a.user_name else a.user_no
        label, _ = type_labels.get(a.risk_type, (a.risk_type, ""))
        table_data.append({
            "작업자": masked,
            "업체": a.company_name,
            "유형": label,
            "현재 EWI": f"{a.current_ewi:.2f}",
            "예측 EWI": f"{a.predicted_ewi:.2f}",
            "EWI 추세": a.ewi_trend,
            "피로도": f"{a.fatigue_score:.2f}",
            "작업공간": f"{a.work_zone_pct:.0%}",
            "이동빈도": f"{a.transition_rate:.3f}/분",
            "심각도": a.severity,
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # 권고사항
    recs = [a for a in alerts if a.recommendation and a.severity in ("critical", "high")]
    if recs:
        with st.expander(f"작업자별 권고사항 ({len(recs)}건)"):
            for a in recs[:10]:
                masked = mask_name(a.user_name) if a.user_name else a.user_no
                sty = SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])
                st.markdown(
                    f"<div style='background:{sty['bg']}; border-left:3px solid {sty['color']}; "
                    f"padding:5px 10px; border-radius:4px; margin-bottom:4px; font-size:0.82rem;'>"
                    f"{sty['icon']} <b>{masked}</b>: {a.recommendation}</div>",
                    unsafe_allow_html=True,
                )

    # EWI 변화 차트
    if len(alerts) >= 3:
        st.markdown("##### 현재 EWI vs 예측 EWI")
        fig = go.Figure()

        names = [mask_name(a.user_name) if a.user_name else a.user_no for a in alerts[:15]]
        current_ewi = [a.current_ewi for a in alerts[:15]]
        pred_ewi = [a.predicted_ewi for a in alerts[:15]]

        fig.add_trace(go.Bar(
            x=names, y=current_ewi,
            name="현재 EWI",
            marker_color=COLORS["accent"],
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=names, y=pred_ewi,
            name="예측 EWI",
            marker_color=[
                SEVERITY_STYLE.get(a.severity, SEVERITY_STYLE["low"])["color"]
                for a in alerts[:15]
            ],
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            barmode="group",
            height=350,
            yaxis_title="EWI",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── M4: 자동 정확도 평가 + 정확도 탭 ──────────────────────


def _auto_evaluate_pending(sector_id: str, dates: list[str], journey_df, selected_date: str):
    """미평가 예측 자동 정확도 계산 (백그라운드, 1회)."""
    eval_done_key = f"_agentic_eval_done_{sector_id}"
    if st.session_state.get(eval_done_key):
        return

    try:
        from src.intelligence.prediction_journal import (
            get_pending_evaluations,
            evaluate_accuracy,
        )

        pending = get_pending_evaluations(sector_id, dates)
        if not pending:
            st.session_state[eval_done_key] = True
            return

        for pred_date in pending[:3]:  # 최대 3건만 (성능)
            # 다음 날 journey 로드
            from datetime import datetime as dt, timedelta
            d = dt.strptime(pred_date, "%Y%m%d")
            next_date = (d + timedelta(days=1)).strftime("%Y%m%d")

            next_journey = load_journey_data(sector_id, next_date)
            if next_journey is None:
                continue

            actual_seqs = extract_worker_sequences(next_journey)
            if actual_seqs:
                evaluate_accuracy(sector_id, pred_date, actual_seqs)

        st.session_state[eval_done_key] = True

    except ImportError:
        st.session_state[eval_done_key] = True
    except Exception as e:
        logger.debug(f"자동 정확도 평가 건너뜀: {e}")
        st.session_state[eval_done_key] = True


def _render_accuracy(sector_id: str):
    """예측 정확도 추적 서브탭."""
    st.markdown("#### 예측 정확도 추적")

    try:
        from src.intelligence.prediction_journal import get_accuracy_history
        history = get_accuracy_history(sector_id)
    except ImportError:
        st.info("Prediction Journal 모듈이 없습니다.")
        return

    # ── 시간 범위별 정확도 비교 (핵심 차트) ──────────────────
    st.markdown("##### 예측 범위별 정확도 비교")
    st.caption(
        "같은 모델로 예측 범위만 달라지면 정확도가 크게 달라집니다. "
        "30분 후 예측은 당일 문맥이 있어 높고, 다음 날은 문맥 없이 예측하므로 낮습니다."
    )

    horizon_data = [
        {"예측 범위": "30분 후", "Top-1": 0.271, "Top-3": 0.439, "비고": "같은 날 단기"},
        {"예측 범위": "1시간 후", "Top-1": 0.228, "Top-3": 0.370, "비고": "같은 날 단기"},
        {"예측 범위": "2시간 후", "Top-1": 0.171, "Top-3": 0.290, "비고": "같은 날 단기"},
        {"예측 범위": "다음 날", "Top-1": 0.141, "Top-3": 0.319, "비고": "익일 예측"},
        {"예측 범위": "MLM (학습)", "Top-1": 0.601, "Top-3": 0.825, "비고": "빈칸 채우기 (참고)"},
    ]
    horizon_df = pd.DataFrame(horizon_data)

    # KPI: 주요 3개
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _kpi_card("30분 후 Top-1", "27.1%", COLORS["success"]),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card("1시간 후 Top-1", "22.8%", COLORS["accent"]),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card("다음 날 Top-1", "14.1%", COLORS["warning"]),
            unsafe_allow_html=True,
        )

    # 범위별 바 차트
    bar_labels = ["30분 후", "1시간 후", "2시간 후", "다음 날"]
    bar_top1 = [27.1, 22.8, 17.1, 14.1]
    bar_top3 = [43.9, 37.0, 29.0, 31.9]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_labels, y=bar_top1, name="Top-1",
        marker_color=COLORS["accent"], text=[f"{v:.1f}%" for v in bar_top1],
        textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        x=bar_labels, y=bar_top3, name="Top-3",
        marker_color=COLORS["success"], text=[f"{v:.1f}%" for v in bar_top3],
        textposition="outside",
    ))
    fig_bar.add_hline(
        y=0.47, line_dash="dot", line_color="#FF4C4C",
        annotation_text="랜덤 기준 (0.47%)",
    )
    fig_bar.update_layout(
        **PLOTLY_DARK,
        height=350,
        barmode="group",
        yaxis_title="정확도 (%)",
        yaxis_range=[0, 55],
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "213개 공간 중 정확한 위치를 맞추는 Task입니다. "
        "랜덤 기준 0.47%이므로, 30분 후 Top-1 27.1%는 **랜덤 대비 58배** 성능입니다."
    )

    # ── 건물_층 그룹 단위 정확도 ──────────────────────────────
    st.divider()
    st.markdown("##### 🏢 건물_층 그룹 단위 정확도")
    st.caption(
        "213개 개별 공간 대신 ~30개 건물_층 그룹으로 집계하면 정확도가 **대폭 향상**됩니다. "
        '"FAB 4층에 있을 것이다"가 "GW-234에 있을 것이다"보다 현장에서 훨씬 유용합니다.'
    )

    # 그룹 정확도 데이터 (locus 정확도와 나란히 비교)
    group_horizon_data = [
        {"예측 범위": "30분 후", "Locus Top-1": "27.1%", "Group Top-1": "—", "Group Top-3": "—",
         "비고": "그룹 정확도는 평가 데이터 축적 후 표시"},
    ]

    # history에서 그룹 정확도 확인
    has_group = any("group_top1" in h for h in history) if history else False
    if has_group:
        group_entries = [h for h in history if "group_top1" in h]
        if group_entries:
            avg_g1 = sum(h["group_top1"] for h in group_entries) / len(group_entries)
            avg_g3 = sum(h["group_top3"] for h in group_entries) / len(group_entries)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    _kpi_card("그룹 수", "~30개", COLORS["text_muted"]),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    _kpi_card("그룹 Top-1 (다음날)", f"{avg_g1:.1%}", COLORS["accent"]),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    _kpi_card("그룹 Top-3 (다음날)", f"{avg_g3:.1%}", COLORS["success"]),
                    unsafe_allow_html=True,
                )

            # Locus vs Group 비교 바 차트
            avg_l1 = sum(h["top1_accuracy"] for h in group_entries) / len(group_entries)
            avg_l3 = sum(h["top3_accuracy"] for h in group_entries) / len(group_entries)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=["Top-1", "Top-3"], y=[avg_l1 * 100, avg_l3 * 100],
                name="개별 Locus (213개)", marker_color="#4A90D9",
                text=[f"{avg_l1:.1%}", f"{avg_l3:.1%}"], textposition="outside",
            ))
            fig_comp.add_trace(go.Bar(
                x=["Top-1", "Top-3"], y=[avg_g1 * 100, avg_g3 * 100],
                name="건물_층 그룹 (~30개)", marker_color="#FF6B35",
                text=[f"{avg_g1:.1%}", f"{avg_g3:.1%}"], textposition="outside",
            ))
            fig_comp.update_layout(
                **PLOTLY_DARK,
                height=300,
                barmode="group",
                yaxis_title="정확도 (%)",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

            st.caption(
                f"개별 Locus Top-1 **{avg_l1:.1%}** → 건물_층 그룹 Top-1 **{avg_g1:.1%}** "
                f"({avg_g1/avg_l1:.1f}배 향상)" if avg_l1 > 0 else ""
            )
    else:
        st.info(
            "건물_층 그룹 정확도는 Prediction Journal 평가 시 자동 계산됩니다. "
            "다음 날 데이터가 있으면 Agentic AI 탭에서 예측 저장 → 자동 평가됩니다."
        )

    # ── 다음 날 예측 일별 추이 ────────────────────────────────
    if not history:
        return

    st.divider()
    st.markdown("##### 다음 날 예측 — 일별 추이")

    latest = history[-1]
    avg_top1 = sum(h["top1_accuracy"] for h in history) / len(history)
    avg_top3 = sum(h["top3_accuracy"] for h in history) / len(history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _kpi_card("평가 일수", f"{len(history)}일"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card("평균 Top-1", f"{avg_top1:.1%}", COLORS["accent"]),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card("평균 Top-3", f"{avg_top3:.1%}", COLORS["success"]),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi_card("최근 Top-1", f"{latest['top1_accuracy']:.1%}", COLORS["accent"]),
            unsafe_allow_html=True,
        )

    # 추이 차트
    if len(history) >= 2:
        hist_df = pd.DataFrame(history)
        hist_df["date_label"] = pd.to_datetime(hist_df["date"], format="%Y%m%d").dt.strftime("%m/%d")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_df["date_label"], y=hist_df["top1_accuracy"],
            mode="lines+markers", name="Top-1 정확도",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["date_label"], y=hist_df["top3_accuracy"],
            mode="lines+markers", name="Top-3 정확도",
            line=dict(color=COLORS["success"], width=2),
            marker=dict(size=6),
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            height=300,
            yaxis_title="정확도",
            yaxis_tickformat=".0%",
            xaxis_title="예측 날짜",
            xaxis_type="category",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 상세 테이블
    with st.expander("일별 정확도 상세"):
        table_data = []
        for h in reversed(history):
            row = {
                "예측 날짜": h["date"],
                "평가 대상": h.get("total_evaluated", 0),
                "Locus Top-1": f"{h['top1_accuracy']:.1%}",
                "Locus Top-3": f"{h['top3_accuracy']:.1%}",
            }
            if "group_top1" in h:
                row["Group Top-1"] = f"{h['group_top1']:.1%}"
                row["Group Top-3"] = f"{h['group_top3']:.1%}"
            if "congestion_mae" in h:
                row["혼잡도 MAE"] = f"{h['congestion_mae']:.1f}명"
            table_data.append(row)
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
