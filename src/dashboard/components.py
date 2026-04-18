"""
Dashboard Components — 재사용 가능한 대시보드 컴포넌트
======================================================
DeepCon 대시보드 전반에서 사용되는 공통 UI 컴포넌트.

컴포넌트:
  - alert_center: 사이드바 알림 센터
  - metric_card_enhanced: 향상된 메트릭 카드 (스파크라인 + 컨텍스트)
  - journey_timeline: Journey 타임라인 Plotly Figure
  - drill_panel: 드릴다운 상세 패널
  - validation_badge: 데이터 품질 뱃지
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, CHART_COLORS, PLOTLY_DARK
from src.utils.anonymizer import mask_name
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, CRE_MEDIUM, EWI_HIGH

if TYPE_CHECKING:
    pass


# ─── 알림 센터 ─────────────────────────────────────────────────────────────


def alert_center(alerts: list[dict]) -> None:
    """
    사이드바 알림 센터.

    Args:
        alerts: [
            {"severity": "critical", "title": "...", "count": 2, "detail": "..."},
            ...
        ]
        severity: critical > high > medium > low
    """
    severity_order = ["critical", "high", "medium", "low"]
    severity_colors = {
        "critical": CHART_COLORS["critical"],
        "high":     COLORS["danger_soft"],
        "medium":   CHART_COLORS["medium"],
        "low":      CHART_COLORS["info"],
    }
    severity_icons = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "⚡",
        "low": "💡",
    }

    if not alerts:
        st.markdown(
            f"<div style='color:{COLORS['text_muted2']}; font-size:0.85rem;'>알림 없음</div>",
            unsafe_allow_html=True,
        )
        return

    # 심각도별 카운트 뱃지
    counts = {s: 0 for s in severity_order}
    for a in alerts:
        sev = a.get("severity", "low")
        counts[sev] += a.get("count", 1)

    badges = []
    for sev in severity_order:
        if counts[sev] > 0:
            color = severity_colors[sev]
            badges.append(
                f"<span style='background:{color}22; color:{color}; "
                f"padding:2px 10px; border-radius:12px; font-size:0.82rem; "
                f"font-weight:600;'>{counts[sev]}</span>"
            )

    st.markdown(
        f"<div style='display:flex; gap:8px; margin-bottom:8px;'>{''.join(badges)}</div>",
        unsafe_allow_html=True,
    )

    # 알림 목록 (우선순위 정렬, 상위 5개)
    sorted_alerts = sorted(
        alerts,
        key=lambda x: severity_order.index(x.get("severity", "low")),
    )[:5]

    with st.expander("알림 상세", expanded=False):
        for alert in sorted_alerts:
            sev = alert.get("severity", "low")
            color = severity_colors.get(sev, CHART_COLORS["info"])
            icon = severity_icons.get(sev, "💡")
            title = alert.get("title", "")
            detail = alert.get("detail", "")

            st.markdown(
                f"""
                <div style='border-left:3px solid {color}; padding:6px 10px;
                            margin:6px 0; background:{COLORS['bg_card_sub']}; border-radius:4px;'>
                    <div style='font-size:0.85rem; color:{color}; font-weight:600;'>
                        {icon} {title}
                    </div>
                    <div style='font-size:0.78rem; color:{COLORS['text_muted']}; margin-top:4px;'>
                        {detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── 향상된 메트릭 카드 ─────────────────────────────────────────────────────


def metric_card_enhanced(
    label: str,
    value: str,
    delta: str = "",
    delta_up: bool = True,
    sparkline_data: list = None,
    context: str = "",
    color: str = None,
) -> str:
    """
    향상된 메트릭 카드 (스파크라인 + 컨텍스트).

    Args:
        label: 라벨
        value: 주 값
        delta: 변화량 (예: "+5%")
        delta_up: 증가가 긍정적인지 (True면 초록, False면 빨강)
        sparkline_data: 최근 7일 등 추이 데이터 리스트
        context: 추가 컨텍스트 (예: "지난주 평균 대비")
        color: 값 색상 오버라이드

    Returns:
        HTML 문자열
    """
    # 변화량 색상
    delta_color = COLORS["success"] if delta_up else COLORS["danger"]
    delta_html = (
        f'<div style="font-size:0.85rem; color:{delta_color}; margin-top:4px;">{delta}</div>'
        if delta
        else ""
    )

    # 값 색상
    value_color = color or COLORS["accent"]

    # 스파크라인 (미니 SVG)
    sparkline_html = ""
    if sparkline_data and len(sparkline_data) >= 2:
        sparkline_html = _render_sparkline_svg(sparkline_data)

    # 컨텍스트
    context_html = (
        f'<div style="font-size:0.72rem; color:{COLORS["text_dim"]}; margin-top:4px;">{context}</div>'
        if context
        else ""
    )

    return f"""
    <div class="metric-card" style="min-height:110px;">
        <div class="metric-value" style="color:{value_color};">{value}</div>
        {delta_html}
        {sparkline_html}
        <div class="metric-label">{label}</div>
        {context_html}
    </div>"""


def _render_sparkline_svg(
    data: list,
    width: int = 80,
    height: int = 20,
    color: str | None = None,
) -> str:
    if color is None:
        color = COLORS["accent"]
    """미니 스파크라인 SVG."""
    if not data or len(data) < 2:
        return ""

    # NaN 제거
    clean_data = [v for v in data if v is not None and not (isinstance(v, float) and v != v)]
    if len(clean_data) < 2:
        return ""

    min_v, max_v = min(clean_data), max(clean_data)
    range_v = max_v - min_v if max_v != min_v else 1

    points = []
    for i, v in enumerate(clean_data):
        x = i * width / (len(clean_data) - 1)
        y = height - ((v - min_v) / range_v) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    return f"""
    <svg width="{width}" height="{height}" style="margin:6px auto; display:block; opacity:0.7;">
        <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
    </svg>"""


# ─── Journey 타임라인 ──────────────────────────────────────────────────────


def journey_timeline(
    journey_df: pd.DataFrame,
    user_no: str,
    locus_dict: dict,
    height: int = 100,
) -> go.Figure:
    """
    작업자 Journey 타임라인 Plotly Figure.

    Args:
        journey_df: Journey 데이터 (block_type 포함 시 사용)
        user_no: 작업자 ID
        locus_dict: locus_id → 정보 딕셔너리
        height: 차트 높이

    Returns:
        Plotly Figure
    """
    # 해당 작업자 데이터 필터링
    user_journey = journey_df[journey_df["user_no"] == user_no].copy()
    if user_journey.empty:
        fig = go.Figure()
        fig.update_layout(**{
            **PLOTLY_DARK,
            "height": height,
            "margin": dict(l=0, r=0, t=0, b=0),
            "annotations": [
                dict(text="데이터 없음", x=0.5, y=0.5, showarrow=False)
            ],
        })
        return fig

    user_journey = user_journey.sort_values("timestamp")

    # 블록 추출
    if "block_id" in user_journey.columns:
        # 토큰화된 데이터 사용
        blocks = _extract_blocks_from_tokenized(user_journey, locus_dict)
    else:
        # 기본 방식: 연속 동일 locus 그룹화
        blocks = _extract_blocks_basic(user_journey, locus_dict)

    if not blocks:
        fig = go.Figure()
        fig.update_layout(height=height, **PLOTLY_DARK)
        return fig

    # Plotly 타임라인 구성
    fig = go.Figure()

    # 블록 타입별 색상 (CHART_COLORS / COLORS 토큰 기반)
    color_map = {
        "GATE_IN":  COLORS["block_gate"],
        "GATE_OUT": COLORS["block_gate"],
        "WORK":     COLORS["block_work"],
        "REST":     COLORS["block_rest"],
        "TRANSIT":  COLORS["block_transit"],
        "ADMIN":    COLORS["block_admin"],
        "UNKNOWN":  COLORS["block_unknown"],
    }

    # 토큰별 색상 (블록 타입 없을 때)
    # v1 영문 토큰 + v2에서는 block_type 기반 color_map이 우선 적용됨
    token_color_map = {
        "work_zone":     COLORS["block_work"],
        "outdoor_work":  "#60A5FA",  # work 의 밝은 변형 (intent 전용)
        "breakroom":     COLORS["block_rest"],
        "smoking_area":  "#34D399",  # rest 의 밝은 변형
        "timeclock":     COLORS["block_gate"],
        "main_gate":     COLORS["block_transit"],
        "confined_space": CHART_COLORS["confined_space"],
        "high_voltage":  CHART_COLORS["high_voltage"],
        # v2: locus_type 기반 (token 컬럼 값)
        "work_area":     COLORS["block_work"],
        "gate":          COLORS["block_gate"],
        "vertical":      "#F59E0B",  # 호이스트/수직이동 (amber)
    }

    # 시작 시간 기준
    start_time = blocks[0]["start"]

    for block in blocks:
        block_type = block.get("block_type", "UNKNOWN")
        token = block.get("token", "unknown")
        color = color_map.get(block_type, token_color_map.get(token, COLORS["block_unknown"]))

        # 상대 시간 (분)
        start_min = (block["start"] - start_time).total_seconds() / 60
        duration = block["duration"]
        name = block.get("name", block.get("locus_id", ""))

        fig.add_trace(go.Bar(
            x=[duration],
            y=["Journey"],
            orientation="h",
            base=start_min,
            marker_color=color,
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"유형: {block_type}<br>"
                f"시간: {block['start'].strftime('%H:%M')} - {block['end'].strftime('%H:%M')}<br>"
                f"체류: {duration:.0f}분<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(**{
        **PLOTLY_DARK,
        "barmode": "stack",
        "showlegend": False,
        "height": height,
        "margin": dict(l=0, r=0, t=10, b=20),
        "xaxis": dict(
            title="시간 (분)",
            showgrid=True,
            gridcolor=COLORS["grid"],
        ),
        "yaxis": dict(showticklabels=False),
    })

    return fig


def _extract_blocks_from_tokenized(
    user_journey: pd.DataFrame,
    locus_dict: dict,
) -> list[dict]:
    """토큰화된 데이터에서 블록 추출."""
    blocks = []
    for block_id in user_journey["block_id"].unique():
        block_data = user_journey[user_journey["block_id"] == block_id]
        if block_data.empty:
            continue

        locus_id = block_data["locus_id"].iloc[0]
        locus_info = locus_dict.get(locus_id, {})

        blocks.append({
            "locus_id": locus_id,
            "name": locus_info.get("locus_name", locus_id),
            "token": locus_info.get("token", "unknown"),
            "block_type": block_data["block_type"].iloc[0] if "block_type" in block_data.columns else "UNKNOWN",
            "start": block_data["timestamp"].min(),
            "end": block_data["timestamp"].max(),
            "duration": len(block_data),
        })

    return blocks


def _extract_blocks_basic(
    user_journey: pd.DataFrame,
    locus_dict: dict,
) -> list[dict]:
    """기본 방식으로 블록 추출 (연속 동일 locus 그룹화)."""
    blocks = []
    current_locus = None
    block_start = None
    block_rows = []

    for _, row in user_journey.iterrows():
        if row["locus_id"] != current_locus:
            if current_locus is not None and block_rows:
                locus_info = locus_dict.get(current_locus, {})
                blocks.append({
                    "locus_id": current_locus,
                    "name": locus_info.get("locus_name", current_locus),
                    "token": locus_info.get("token", "unknown"),
                    "block_type": "UNKNOWN",
                    "start": block_start,
                    "end": block_rows[-1]["timestamp"],
                    "duration": len(block_rows),
                })
            current_locus = row["locus_id"]
            block_start = row["timestamp"]
            block_rows = [row]
        else:
            block_rows.append(row)

    # 마지막 블록
    if current_locus is not None and block_rows:
        locus_info = locus_dict.get(current_locus, {})
        blocks.append({
            "locus_id": current_locus,
            "name": locus_info.get("locus_name", current_locus),
            "token": locus_info.get("token", "unknown"),
            "block_type": "UNKNOWN",
            "start": block_start,
            "end": block_rows[-1]["timestamp"] if isinstance(block_rows[-1], dict) else block_rows[-1]["timestamp"],
            "duration": len(block_rows),
        })

    return blocks


# ─── 데이터 품질 뱃지 ──────────────────────────────────────────────────────


def validation_badge(level: str, score: float = None) -> str:
    """
    데이터 품질 뱃지 HTML.

    Args:
        level: "pass" | "warning" | "fail"
        score: 0~1 점수 (선택)

    Returns:
        HTML 문자열
    """
    colors = {
        "pass": COLORS["success"],
        "warning": COLORS["warning"],
        "fail": COLORS["danger"],
    }
    icons = {
        "pass": "✓",
        "warning": "⚠",
        "fail": "✗",
    }
    labels = {
        "pass": "정상",
        "warning": "주의",
        "fail": "문제",
    }

    color = colors.get(level, COLORS["text_muted"])
    icon = icons.get(level, "?")
    label = labels.get(level, level)

    score_text = f" ({score * 100:.0f}%)" if score is not None else ""

    return (
        f'<span style="background:{color}22; color:{color}; '
        f'padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:600;">'
        f'{icon} {label}{score_text}</span>'
    )


# ─── 드릴다운 패널 ─────────────────────────────────────────────────────────


def render_worker_detail_panel(
    user_no: str,
    worker_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    locus_dict: dict,
) -> None:
    """작업자 상세 패널 렌더링."""
    # 작업자 정보 조회
    worker_row = worker_df[worker_df["user_no"] == user_no]
    if worker_row.empty:
        st.info("작업자 정보 없음")
        return

    row = worker_row.iloc[0]

    # 헤더
    st.markdown(
        f"""
        <div style='background:{COLORS['card_bg']}; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.1rem; font-weight:600; color:{COLORS['text']};'>
                {mask_name(row.get('user_name', user_no))}
            </div>
            <div style='font-size:0.82rem; color:{COLORS['text_dimmer']};'>
                {row.get('company_name', '소속 미확인')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI 3개
    col1, col2, col3 = st.columns(3)
    with col1:
        ewi = row.get("ewi", 0)
        ewi_color = (
            CHART_COLORS["critical"] if ewi >= 0.8
            else CHART_COLORS["medium"] if ewi >= EWI_HIGH
            else CHART_COLORS["rest"]
        )
        st.markdown(
            metric_card_enhanced("EWI", f"{ewi:.3f}", color=ewi_color),
            unsafe_allow_html=True,
        )
    with col2:
        cre = row.get("cre", 0)
        cre_color = (
            CHART_COLORS["critical"] if cre >= CRE_HIGH
            else CHART_COLORS["medium"] if cre >= CRE_MEDIUM
            else CHART_COLORS["rest"]
        )
        st.markdown(
            metric_card_enhanced("CRE", f"{cre:.3f}", color=cre_color),
            unsafe_allow_html=True,
        )
    with col3:
        work_min = row.get("work_minutes", 0)
        st.markdown(
            metric_card_enhanced("근무 시간", f"{work_min:.0f}분"),
            unsafe_allow_html=True,
        )

    # Journey 타임라인
    st.markdown("### Journey")
    if not journey_df.empty:
        fig = journey_timeline(journey_df, user_no, locus_dict)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Journey 데이터 없음")

    # 상세 지표
    with st.expander("상세 지표", expanded=False):
        detail_cols = [
            ("고활성 시간", "high_active_min", "분"),
            ("저활성 시간", "low_active_min", "분"),
            ("대기 시간", "standby_min", "분"),
            ("휴식 시간", "rest_min", "분"),
            ("이동 시간", "transit_min", "분"),
            ("방문 공간 수", "unique_loci", "개"),
            ("피로도", "fatigue_score", ""),
            ("단독작업 비율", "alone_ratio", ""),
        ]

        detail_data = []
        for label, col, unit in detail_cols:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    if isinstance(val, float):
                        detail_data.append({"지표": label, "값": f"{val:.2f}{unit}"})
                    else:
                        detail_data.append({"지표": label, "값": f"{val}{unit}"})

        if detail_data:
            st.table(pd.DataFrame(detail_data))


def render_space_detail_panel(
    locus_id: str,
    space_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    locus_dict: dict,
) -> None:
    """공간 상세 패널 렌더링."""
    locus_info = locus_dict.get(locus_id, {})

    # 헤더
    st.markdown(
        f"""
        <div style='background:{COLORS['card_bg']}; border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.1rem; font-weight:600; color:{COLORS['text']};'>
                {locus_info.get('locus_name', locus_id)}
            </div>
            <div style='font-size:0.82rem; color:{COLORS['text_dimmer']};'>
                {locus_info.get('token', '')} · {locus_info.get('locus_type', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 공간 지표 (space_df에서)
    if not space_df.empty and "locus_id" in space_df.columns:
        space_row = space_df[space_df["locus_id"] == locus_id]
        if not space_row.empty:
            row = space_row.iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("체류 인원", f"{row.get('total_workers', 0):,}명")
            with col2:
                st.metric("평균 체류", f"{row.get('avg_dwell_min', 0):.0f}분")
            with col3:
                st.metric("평균 EWI", f"{row.get('avg_ewi', 0):.3f}")

    # 시간대별 체류 현황
    if not journey_df.empty and "locus_id" in journey_df.columns:
        space_journey = journey_df[journey_df["locus_id"] == locus_id]
        if not space_journey.empty and "timestamp" in space_journey.columns:
            # 시간대별 집계
            space_journey["hour"] = pd.to_datetime(space_journey["timestamp"]).dt.hour
            hourly = space_journey.groupby("hour")["user_no"].nunique().reset_index()
            hourly.columns = ["시간", "인원"]

            st.markdown("### 시간대별 체류 인원")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly["시간"],
                y=hourly["인원"],
                marker_color=COLORS["accent"],
            ))
            fig.update_layout(**{
                **PLOTLY_DARK,
                "height": 200,
                "margin": dict(l=0, r=0, t=10, b=30),
                "xaxis_title": "시간",
                "yaxis_title": "인원",
            })
            st.plotly_chart(fig, use_container_width=True)


# ─── Enriched Locus 기반 공간 알림 (2026-03-28 추가) ─────────────────────

# dwell_category 컬러 및 라벨
DWELL_CATEGORY_STYLES = {
    "TRANSIT": {"color": "#6B7280", "bg": "#6B728022", "label": "통과형", "icon": "🚶"},
    "SHORT_STAY": {"color": "#10B981", "bg": "#10B98122", "label": "단기", "icon": "☕"},
    "LONG_STAY": {"color": "#3B82F6", "bg": "#3B82F622", "label": "장기", "icon": "🔧"},
    "HAZARD_ZONE": {"color": "#EF4444", "bg": "#EF444422", "label": "고위험", "icon": "⚠️"},
    "ADMIN": {"color": "#8B5CF6", "bg": "#8B5CF622", "label": "관리", "icon": "📋"},
    "UNKNOWN": {"color": "#4B5563", "bg": "#4B556322", "label": "미분류", "icon": "❓"},
}


def get_dwell_category_style(category: str) -> dict:
    """dwell_category에 해당하는 스타일 반환."""
    return DWELL_CATEGORY_STYLES.get(category, DWELL_CATEGORY_STYLES["UNKNOWN"])


def build_space_alerts(
    journey_df: pd.DataFrame,
    locus_dict: dict,
    space_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Enriched locus 정보 기반 공간 관련 알림 생성.

    알림 유형:
    1. HAZARD_ZONE 장시간 체류 (avg_dwell의 2배 초과) → critical
    2. TRANSIT 이상 체류 (30분 초과) → medium
    3. 공간 과밀 (max_concurrent_occupancy의 90% 초과) → high
    4. 공간 혼잡 (70% 초과) → medium

    Args:
        journey_df: 현재 journey 데이터
        locus_dict: enriched locus 정보
        space_df: 공간별 집계 데이터 (optional)

    Returns:
        list[dict]: 각 dict는 {severity, title, detail, count} 포함
    """
    alerts = []

    if journey_df.empty or not locus_dict:
        return alerts

    # 최근 30분 데이터 추출
    if "timestamp" in journey_df.columns:
        try:
            latest = pd.to_datetime(journey_df["timestamp"]).max()
            recent = journey_df[pd.to_datetime(journey_df["timestamp"]) >= latest - pd.Timedelta(minutes=30)]
        except Exception:
            recent = journey_df.tail(1000)
    else:
        recent = journey_df.tail(1000)

    if recent.empty or "locus_id" not in recent.columns:
        return alerts

    # Locus별 분석
    seen_alerts = set()

    for locus_id in recent["locus_id"].dropna().unique():
        info = locus_dict.get(locus_id, {})
        if not info:
            continue

        cat = info.get("dwell_category", "UNKNOWN")
        avg_dwell = info.get("avg_dwell_minutes") or 30
        max_occ = info.get("max_concurrent_occupancy") or 50
        locus_name = info.get("locus_name", locus_id)

        locus_journey = recent[recent["locus_id"] == locus_id]
        current_workers = locus_journey["user_no"].nunique()

        # 1. HAZARD_ZONE 장시간 체류 체크
        if cat == "HAZARD_ZONE":
            for user_no in locus_journey["user_no"].unique():
                user_data = locus_journey[locus_journey["user_no"] == user_no]
                dwell_min = len(user_data)  # 분 단위 근사
                if dwell_min > avg_dwell * 2:
                    alert_key = f"hazard_dwell_{locus_id}_{user_no}"
                    if alert_key not in seen_alerts:
                        seen_alerts.add(alert_key)
                        alerts.append({
                            "severity": "critical",
                            "title": f"고위험 구역 장시간 체류: {locus_name}",
                            "detail": f"작업자 {user_no}: {dwell_min}분 체류 (평균 {avg_dwell:.0f}분의 {dwell_min/avg_dwell:.1f}배)",
                            "count": 1,
                        })

        # 2. TRANSIT 이상 체류 체크
        if cat == "TRANSIT":
            for user_no in locus_journey["user_no"].unique():
                user_data = locus_journey[locus_journey["user_no"] == user_no]
                dwell_min = len(user_data)
                if dwell_min > 30:
                    alert_key = f"transit_dwell_{locus_id}_{user_no}"
                    if alert_key not in seen_alerts:
                        seen_alerts.add(alert_key)
                        alerts.append({
                            "severity": "medium",
                            "title": f"통과 구역 이상 체류: {locus_name}",
                            "detail": f"작업자 {user_no}: {dwell_min}분 체류 (통과형 구역)",
                            "count": 1,
                        })

        # 3. 공간 과밀/혼잡 체크
        if max_occ > 0:
            occ_pct = current_workers / max_occ * 100
            if occ_pct >= 90:
                alert_key = f"overcrowd_{locus_id}"
                if alert_key not in seen_alerts:
                    seen_alerts.add(alert_key)
                    alerts.append({
                        "severity": "high",
                        "title": f"공간 과밀 경고: {locus_name}",
                        "detail": f"현재 {current_workers}명 / 최대 {max_occ}명 ({occ_pct:.0f}%)",
                        "count": 1,
                    })
            elif occ_pct >= 70:
                alert_key = f"crowded_{locus_id}"
                if alert_key not in seen_alerts:
                    seen_alerts.add(alert_key)
                    alerts.append({
                        "severity": "medium",
                        "title": f"공간 혼잡 주의: {locus_name}",
                        "detail": f"현재 {current_workers}명 / 최대 {max_occ}명 ({occ_pct:.0f}%)",
                        "count": 1,
                    })

    # 심각도별 정렬
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(alerts, key=lambda x: severity_order.get(x.get("severity", "low"), 3))


def render_space_card(
    locus_id: str,
    locus_dict: dict,
    current_workers: int = 0,
) -> str:
    """
    공간 카드 HTML 생성.

    Args:
        locus_id: Locus ID
        locus_dict: enriched locus 정보
        current_workers: 현재 체류 인원

    Returns:
        HTML 문자열
    """
    info = locus_dict.get(locus_id, {})
    cat = info.get("dwell_category", "UNKNOWN")
    style = get_dwell_category_style(cat)

    locus_name = info.get("locus_name", locus_id)
    avg_dwell = info.get("avg_dwell_minutes") or 0
    peak_hour = info.get("peak_hour") or 0
    max_occ = info.get("max_concurrent_occupancy") or 50
    hazard_level = str(info.get("hazard_level", "")).lower()

    # 혼잡률 계산
    occ_pct = min(100, (current_workers / max_occ * 100)) if max_occ > 0 else 0

    # 혼잡도 색상 (CHART_COLORS 토큰)
    if occ_pct >= 90:
        occ_color = CHART_COLORS["critical"]
    elif occ_pct >= 70:
        occ_color = CHART_COLORS["high"]
    elif occ_pct >= 40:
        occ_color = CHART_COLORS["medium"]
    else:
        occ_color = CHART_COLORS["rest"]

    # 고위험 뱃지
    hazard_badge = ""
    if hazard_level in ("high", "critical"):
        hazard_badge = (
            f"<span style='color:{CHART_COLORS['critical']};font-size:0.78rem;'>"
            f"⚡ 고위험</span>"
        )

    return f"""
    <div style='background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]}; border-radius:8px;
                padding:12px 14px; margin:6px 0;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <span style='background:{style["bg"]}; color:{style["color"]}; padding:2px 6px;
                             border-radius:4px; font-size:0.72rem;'>{style["icon"]} {style["label"]}</span>
                <span style='color:{COLORS["text"]}; font-weight:600; margin-left:6px;'>
                    {locus_name}
                </span>
            </div>
            {hazard_badge}
        </div>
        <div style='color:{COLORS["text_muted"]}; font-size:0.82rem; margin-top:8px;'>
            👥 {current_workers}명 / {max_occ}명 ({occ_pct:.1f}%)
        </div>
        <div style='color:{COLORS["text_dimmer"]}; font-size:0.78rem; margin-top:4px;'>
            평균 {avg_dwell:.0f}분 · 피크 {peak_hour}시
        </div>
        <div style='background:{COLORS["border"]}; height:5px; border-radius:3px; margin-top:8px; overflow:hidden;'>
            <div style='background:{occ_color}; width:{occ_pct:.1f}%; height:100%;'></div>
        </div>
    </div>
    """


def render_dwell_category_badge(category: str) -> str:
    """dwell_category 뱃지 HTML 생성."""
    style = get_dwell_category_style(category)
    return (
        f"<span style='background:{style['bg']}; color:{style['color']}; "
        f"padding:2px 8px; border-radius:4px; font-size:0.75rem;'>"
        f"{style['icon']} {style['label']}</span>"
    )


# ═══════════════════════════════════════════════════════════════════════
# AI Commentary Box — M2-A (T-13) · 5개 탭 공통 AI 코멘터리 컴포넌트
# ───────────────────────────────────────────────────────────────────────
# M2-B 에서 각 탭이 역할(role)과 context 를 주입하여 재사용할 예정.
# 현재(M2-A) 는 프레임워크만 확정 — 탭 삽입 위치는 다음 티켓에서 결정.
# ═══════════════════════════════════════════════════════════════════════

def _compute_context_hash(context: dict | None) -> str:
    """Context dict 의 해시 (session_state 캐시 키 생성용)."""
    import hashlib
    import json

    try:
        payload = json.dumps(context or {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = str(context)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _ai_error_box(message: str, *, retry_key: str | None = None) -> None:
    """AI 에러 표시 박스 — styles.py 토큰 사용 (hex 하드코딩 금지)."""
    st.markdown(
        f"<div style='background:{COLORS['card_bg']}; "
        f"border-left:3px solid {COLORS['danger']}; "
        f"border-radius:6px; padding:12px 16px; margin:8px 0; "
        f"color:{COLORS['text_muted']}; font-size:0.85rem;'>"
        f"{message}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if retry_key:
        if st.button("다시 시도", key=retry_key, type="secondary"):
            for k in list(st.session_state.keys()):
                if k.startswith(retry_key.rsplit("_retry", 1)[0]):
                    st.session_state.pop(k, None)
            st.rerun()


def _ai_header(title: str, *, role_label: str = "") -> None:
    """AI 코멘터리 박스 헤더 (배지 스타일)."""
    suffix = f" · {role_label}" if role_label else ""
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:8px; "
        f"margin-bottom:10px;'>"
        f"<span style='background:{COLORS['primary']}; color:{COLORS['accent']}; "
        f"padding:3px 10px; border-radius:12px; font-size:0.75rem; "
        f"font-weight:700; letter-spacing:0.3px;'>AI</span>"
        f"<span style='color:{COLORS['text']}; font-size:0.9rem; "
        f"font-weight:600;'>{title}{suffix}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _ai_meta_footer(
    *, latency_ms: int, tokens_in: int, tokens_out: int, cache_hit: int
) -> None:
    """응답 메타 정보 (tokens/latency) 하단 표시."""
    secs = latency_ms / 1000.0 if latency_ms else 0.0
    st.markdown(
        f"<div style='color:{COLORS['text_dim']}; font-size:0.72rem; "
        f"margin-top:8px; text-align:right;'>"
        f"응답 시간 {secs:.1f}s · in {tokens_in:,} → out {tokens_out:,} tokens "
        f"· cached {cache_hit}"
        f"</div>",
        unsafe_allow_html=True,
    )


def ai_commentary_box(
    role: str,
    context: dict,
    sector_id: str,
    date_str: str | None = None,
    *,
    title: str = "AI 분석 코멘트",
    spinner_text: str = "AI가 분석 중...",
    cached_key: str | None = None,
    show_meta: bool = False,
    button_label: str = "AI 분석 실행",
    streaming: bool = True,
    user_role: str = "unknown",
    tab: str | None = None,
) -> None:
    """
    탭에 삽입하는 AI 코멘터리 박스 (M2-A T-13).

    사용 예 (M2-B 에서 각 탭에 삽입):
        from src.dashboard.components import ai_commentary_box

        ai_commentary_box(
            role="overview_commentator",
            context={
                "today_kpi": {"total_workers": 8432, "avg_ewi": 0.52, ...},
                "recent_7days": [...],
                "shift_distribution": {...},
            },
            sector_id="Y1_SKHynix",
            date_str="20260409",
            user_role=st.session_state.get("user_role", "unknown"),
            tab="overview",
        )

    Args:
        role:          core.ai.role_prompts 의 Role (overview_commentator 등)
        context:       파이프라인 산출물 요약 (LLM 전송 전 내부에서 anonymize)
        sector_id:     "Y1_SKHynix" / "M15X_SKHynix"
        date_str:      "20260409" (옵션)
        title:         박스 헤더 제목
        spinner_text:  로딩 스피너 텍스트
        cached_key:    session_state 캐시 키 (None 이면 자동 생성)
        show_meta:     tokens/latency 하단 표시 여부
        button_label:  실행 버튼 레이블
        streaming:     True 면 스트리밍, False 면 동기 호출
        user_role:     감사 로그용 사용자 역할 ("administrator"|sector id)
        tab:           감사 로그용 탭 식별자 (None 이면 role 에서 파생)

    동작:
        1) 배지 헤더 + 실행 버튼 표시.
        2) 버튼 클릭 시 LLMGateway.analyze_stream() (streaming=True) 또는 analyze().
        3) 같은 컨텍스트 해시로 이미 실행된 결과가 session_state 에 있으면 그대로 표시.
        4) API 키 미설정/오류 시 친절한 안내 + 재시도 버튼.
    """
    # 1) core.ai import (실패해도 import error 를 앱 전체에 전파시키지 않음)
    try:
        from core.ai import (
            CommentaryRequest,
            ROLE_LABELS,
            get_gateway,
        )
    except Exception as e:
        _ai_header(title)
        _ai_error_box(
            f"AI 모듈 로드 실패: {e}. 관리자에게 문의하세요.",
            retry_key=None,
        )
        return

    role_label = ROLE_LABELS.get(role, role)

    # 2) 캐시 키 생성
    ctx_hash = _compute_context_hash(context)
    key = cached_key or f"ai_{role}_{sector_id}_{date_str or 'na'}_{ctx_hash}"
    resp_key = f"{key}_response"
    meta_key = f"{key}_meta"
    retry_key = f"{key}_retry"

    with st.container(border=True):
        _ai_header(title, role_label=role_label)

        # 3) 캐시된 응답이 있으면 즉시 표시
        cached_text = st.session_state.get(resp_key)
        cached_meta = st.session_state.get(meta_key)

        if cached_text:
            st.markdown(cached_text)
            if show_meta and cached_meta:
                _ai_meta_footer(**cached_meta)
            # 재실행 버튼
            if st.button(f"다시 분석", key=f"{key}_rerun", type="secondary"):
                st.session_state.pop(resp_key, None)
                st.session_state.pop(meta_key, None)
                st.rerun()
            return

        # 4) 실행 버튼 — 사용자가 누를 때만 LLM 호출 (비용 제어)
        clicked = st.button(button_label, key=f"{key}_run", type="primary")
        if not clicked:
            st.caption("버튼을 눌러 AI 해설을 생성하세요.")
            return

        # 5) Gateway 준비 + API 키 체크
        gw = get_gateway()
        if not gw.is_available():
            _ai_error_box(
                "AI 기능을 사용하려면 `.env` 에 `ANTHROPIC_API_KEY` 를 설정해야 합니다. "
                "관리자에게 문의하세요.",
                retry_key=retry_key,
            )
            return

        # 6) 요청 조립
        req = CommentaryRequest(
            role=role,
            sector_id=sector_id,
            date_str=date_str,
            context=context,
            user_role=user_role,
            tab=tab or role.split("_")[0],
            stream=streaming,
        )

        # 7) 호출
        try:
            if streaming:
                with st.spinner(spinner_text):
                    placeholder = st.empty()
                    collected: list[str] = []
                    for chunk in gw.analyze_stream(req):
                        if chunk:
                            collected.append(chunk)
                            placeholder.markdown("".join(collected) + "▌")
                    final_text = "".join(collected)
                    placeholder.markdown(final_text)
                # 스트리밍은 usage 를 별도 호출로 얻기 어려워 meta 는 표시 안 함
                st.session_state[resp_key] = final_text
                if show_meta:
                    # 감사 로그에는 기록되지만 UI 에는 최소 표시
                    st.session_state[meta_key] = {
                        "latency_ms": 0,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "cache_hit": 0,
                    }
            else:
                with st.spinner(spinner_text):
                    resp = gw.analyze(req)
                st.markdown(resp.text)
                st.session_state[resp_key] = resp.text
                if show_meta:
                    st.session_state[meta_key] = {
                        "latency_ms": resp.latency_ms,
                        "tokens_in": resp.tokens_in,
                        "tokens_out": resp.tokens_out,
                        "cache_hit": resp.cache_hit_tokens,
                    }
                    _ai_meta_footer(**st.session_state[meta_key])
        except Exception as e:
            _ai_error_box(
                f"AI 호출 중 오류가 발생했습니다: {e}",
                retry_key=retry_key,
            )


# ═══════════════════════════════════════════════════════════════════════
# Schema version mismatch 공통 핸들러 (M4-T34)
# ═══════════════════════════════════════════════════════════════════════

def handle_schema_mismatch(exc, sector_id: str, date_str: str) -> None:
    """
    SchemaVersionMismatch 를 사용자 친화 메시지로 안내하고 st.stop().

    각 탭의 _load_daily 진입부에서 try/except 로 감싸 호출한다.
    관리자 역할이면 CLI 재처리 명령까지, 고객사 역할이면 담당자 연락 안내만.

    Args:
        exc: SchemaVersionMismatch 인스턴스 (속성: file, expected, found)
        sector_id: 섹터 ID
        date_str: 문제의 날짜

    Side Effect:
        st.error() + st.stop() — 탭 렌더링 중단.
    """
    # 관리자 여부 확인 (auth 모듈 느슨 결합 — 없으면 고객사 모드로 가정)
    is_admin = False
    try:
        from src.dashboard.auth import get_current_user
        role = (get_current_user() or {}).get("role", "")
        is_admin = (role == "administrator")
    except Exception:
        pass

    file = getattr(exc, "file", "(unknown)")
    expected = getattr(exc, "expected", "?")
    found = getattr(exc, "found", None)
    found_label = f"v{found}" if found is not None else "legacy (버전 정보 없음)"

    if is_admin:
        st.error(
            f"⚠ 데이터 스키마 버전 불일치 — **{date_str}** 재처리가 필요합니다.\n\n"
            f"- 섹터: `{sector_id}`\n"
            f"- 문제 파일: `{file}`\n"
            f"- 기대 버전: **v{expected}** / 현재 버전: **{found_label}**\n\n"
            f"재처리 명령 (관리자):\n"
            f"```bash\n"
            f"cd SandBox/DeepCon\n"
            f"python -m src.pipeline.cli reprocess --sector {sector_id} --date {date_str}\n"
            f"```\n"
            f"또는 ⚙️ 시스템 관리 → 파이프라인 탭에서 재처리를 실행하세요."
        )
    else:
        st.error(
            f"⚠ **{date_str}** 데이터가 최신 분석 버전과 호환되지 않습니다.\n\n"
            f"TJLABS 담당자에게 재처리를 요청해주세요.\n"
            f"- 섹터: `{sector_id}`\n"
            f"- 파일: `{file}`\n"
            f"- 기대 버전: v{expected} / 현재: {found_label}"
        )
    st.stop()
