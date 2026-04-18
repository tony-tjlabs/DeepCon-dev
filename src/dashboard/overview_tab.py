"""
현장 개요 탭 — Y1 SK하이닉스 건설현장 전체 기간 요약
=====================================================
전체 기간 핵심 KPI + T-Ward 착용률 + 일별 추이 + 작업시간 분류 + 업체 랭킹.

데이터 소스:
  - summary_index.json  → 전체 기간 KPI / 일별 추이 (빠른 로드)
  - worker.parquet(최신) → 작업시간 분류 / 업체 랭킹 / 3교대 분포

섹션:
  0. T-Ward 착용률 하이라이트
  1. 전체 기간 KPI (출입 인원 기준 / T-Ward 기준)
  2. 일별 추이 (출입 vs T-Ward 착용 인원 + 착용률 + EWI)
  3. 최신일 작업시간 분류 도넛 (T-Ward 기준)
  4. 업체별 EWI 랭킹 Top 10 + 3교대 분포
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from core.cache.policy import DAILY_PARQUET, SUMMARY_INDEX
from src.dashboard.styles import (
    COLORS,
    CHART_COLORS,
    PLOTLY_DARK,
    PLOTLY_LEGEND,
    metric_card,
    metric_card_sm,
    section_header,
    sub_header,
)
from src.dashboard.date_utils import format_date_label
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH, COMPANY_MIN_WORKERS

logger = logging.getLogger(__name__)

# ─── 색상 상수 (styles.CHART_COLORS 단일 소스 참조) ───────────────────
COLOR_WORK    = CHART_COLORS["work_zone"]
COLOR_TWARD   = CHART_COLORS["tward"]
COLOR_ACCESS  = CHART_COLORS["access"]
COLOR_TRANSIT = CHART_COLORS["transit"]
COLOR_REST    = CHART_COLORS["rest"]
COLOR_GAP     = CHART_COLORS["gap"]

SHIFT_COLORS = {
    "day":            CHART_COLORS["work_zone"],     # 주간 = accent
    "night":          CHART_COLORS["gate"],           # 야간 = purple
    "extended_night": CHART_COLORS["sii"],            # 연장야간 = orange
    "unknown":        CHART_COLORS["gap"],            # 미분류 = 회색
}
SHIFT_LABELS = {
    "day":            "주간",
    "night":          "야간",
    "extended_night": "심야",
    "unknown":        "미분류",
}


# ─── 캐시 로더 ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=SUMMARY_INDEX)
def _load_summary_index(sector_id: str) -> dict:
    """summary_index.json 로드 (빠른 KPI 집계용)."""
    try:
        from src.pipeline.summary_index import load_summary_index
        return load_summary_index(sector_id)
    except Exception as e:
        logger.warning(f"summary_index 로드 실패: {e}")
    return {"dates": {}}


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _load_latest_worker(sector_id: str, date_str: str) -> pd.DataFrame:
    """최신 날짜 worker.parquet 로드 (핵심 컬럼만, 존재 컬럼만 필터)."""
    try:
        paths = cfg.get_sector_paths(sector_id)
        p = paths["processed_dir"] / date_str / "worker.parquet"
        if p.exists():
            desired = [
                "user_no", "company_name", "ewi", "ewi_reliable",
                "work_zone_minutes", "work_minutes",
                "transit_min", "rest_min", "gap_min",
                # fallback columns (EWI 미계산 parquet 호환)
                "rest_minutes", "transit_count",
                "shift_type", "cre",
            ]
            # 스키마 확인 후 존재 컬럼만 로드 (EWI 실패한 구 parquet 호환)
            try:
                import pyarrow.parquet as pq
                avail = set(pq.read_schema(p).names)
                cols = [c for c in desired if c in avail]
                return pd.read_parquet(p, columns=cols) if cols else pd.read_parquet(p)
            except Exception:
                return pd.read_parquet(p)
    except Exception as e:
        logger.warning(f"worker.parquet 로드 실패 [{date_str}]: {e}")
    return pd.DataFrame()


# ─── 집계 헬퍼 ───────────────────────────────────────────────────────

def _build_summary_df(idx_dates: dict) -> pd.DataFrame:
    """summary_index dates dict → DataFrame 변환."""
    rows = []
    for date_str, v in sorted(idx_dates.items()):
        access   = v.get("total_workers_access", 0)
        tward    = v.get("tward_holders", 0)
        wear_pct = (tward / access * 100) if access > 0 else 0.0
        rows.append({
            "date":           date_str,
            "workers_access": access,
            "workers_move":   v.get("total_workers_move", 0),
            "tward_holders":  tward,
            "wear_rate":      wear_pct,
            "companies":      v.get("companies", 0),
            "avg_ewi":        v.get("avg_ewi", 0.0),
            "avg_cre":        v.get("avg_cre", 0.0),
            "avg_fatigue":    v.get("avg_fatigue", 0.0),
            "avg_sii":        v.get("avg_sii", 0.0),
            "high_ewi_count": v.get("high_ewi_count", 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _safe_mean(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.mean()) if len(v) > 0 else 0.0


# ─── 섹션 렌더러 ─────────────────────────────────────────────────────

def _render_tward_banner(df: pd.DataFrame) -> None:
    """
    T-Ward 착용률 하이라이트 배너.
    출입 인원 → T-Ward 착용 인원 → 착용률을 한눈에 표시.
    """
    total_access = int(df["workers_access"].sum())
    total_tward  = int(df["tward_holders"].sum())
    avg_rate     = _safe_mean(df["wear_rate"])

    # 착용률 색상
    rate_color = (
        COLORS["success"] if avg_rate >= 80
        else COLORS["warning"] if avg_rate >= 60
        else COLORS["danger"]
    )
    # 프로그레스 바 너비 클램프
    bar_w = min(100, max(0, avg_rate))

    st.markdown(
        f"""
        <div style='background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]}; border-radius:12px;
                    padding:20px 24px; margin-bottom:20px;'>
            <div style='font-size:0.82rem; color:{COLORS["text_dimmer"]}; margin-bottom:10px; font-weight:600;
                        letter-spacing:0.5px; text-transform:uppercase;'>
                T-Ward 착용률 (전체 기간 평균)
            </div>
            <div style='display:flex; align-items:center; gap:32px;'>
                <!-- 착용률 큰 숫자 -->
                <div style='min-width:120px;'>
                    <span style='font-size:2.8rem; font-weight:800; color:{rate_color};
                                 line-height:1;'>{avg_rate:.1f}%</span>
                </div>
                <!-- 출입 → 착용 흐름 -->
                <div style='flex:1;'>
                    <div style='display:flex; gap:20px; margin-bottom:10px;'>
                        <div>
                            <div style='font-size:0.75rem; color:{COLORS["text_dimmer"]};'>타각기 출입 (연인원)</div>
                            <div style='font-size:1.4rem; font-weight:700; color:{COLORS["text"]};'>{total_access:,}명</div>
                        </div>
                        <div style='color:{COLORS["gap_dark"]}; font-size:1.4rem; align-self:center;'>→</div>
                        <div>
                            <div style='font-size:0.75rem; color:{COLORS["text_dimmer"]};'>T-Ward 착용 (연인원)</div>
                            <div style='font-size:1.4rem; font-weight:700; color:{rate_color};'>{total_tward:,}명</div>
                        </div>
                        <div style='color:{COLORS["gap_dark"]}; font-size:1.4rem; align-self:center;'>→</div>
                        <div>
                            <div style='font-size:0.75rem; color:{COLORS["text_dimmer"]};'>미착용 (연인원)</div>
                            <div style='font-size:1.4rem; font-weight:700;
                                        color:{COLORS["text_muted"]};'>{total_access - total_tward:,}명</div>
                        </div>
                    </div>
                    <!-- 프로그레스 바 -->
                    <div style='background:{COLORS["bg"]}; border-radius:6px; height:10px; overflow:hidden;'>
                        <div style='width:{bar_w:.1f}%; height:100%;
                                    background:linear-gradient(90deg, {rate_color} 0%, {rate_color}AA 100%);
                                    border-radius:6px;'></div>
                    </div>
                    <div style='font-size:0.72rem; color:{COLORS["gap_dark"]}; margin-top:4px;'>
                        ※ EWI · CRE · 피로도 등 이하 모든 수치는 <b style='color:{COLORS["text_muted"]}'>T-Ward 착용 작업자 기준</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_weekday_weekend(df: pd.DataFrame):
    """date 컬럼(YYYYMMDD str) 기준 주중/주말 분리."""
    dates = pd.to_datetime(df["date"], format="%Y%m%d")
    is_weekend = dates.dt.dayofweek >= 5          # 5=토, 6=일
    return df[~is_weekend].copy(), df[is_weekend].copy()


def _render_kpi_section(df: pd.DataFrame) -> None:
    """
    주중 / 주말 분리 KPI 표시.
    Row 1: 타각기 출입 기준 (주중 | 주말)
    Row 2: T-Ward 착용 기준 (주중 | 주말)
    """
    df_wd, df_we = _split_weekday_weekend(df)
    n_wd = len(df_wd)
    n_we = len(df_we)

    # ── 헬퍼: 컬럼 색상 ──────────────────────────────────────────────
    def _ewi_color(v: float) -> str:
        return COLORS["success"] if v >= 0.35 else COLORS["warning"] if v >= 0.15 else COLORS["danger"]
    def _cre_color(v: float) -> str:
        return COLORS["danger"] if v >= CRE_HIGH else COLORS["warning"] if v >= 0.25 else COLORS["success"]
    def _fat_color(v: float) -> str:
        return COLORS["danger"] if v >= 0.4 else COLORS["warning"] if v >= 0.2 else COLORS["success"]

    # ── 헬퍼: 구간 헤더 ──────────────────────────────────────────────
    def _sub_badge(label: str, n_days: int, color: str) -> str:
        return (
            f"<div style='font-size:0.78rem; color:{color}; font-weight:700; "
            f"margin-bottom:4px;'>{label}"
            f"<span style='font-size:0.68rem; color:{COLORS['text_muted2']}; font-weight:400; "
            f"margin-left:6px;'>({n_days}일)</span></div>"
        )

    # ════════════════════════════════════════════════════════════════
    # Row 1 — 타각기 출입 기준
    # ════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='font-size:0.78rem; color:{COLORS['text_dimmer']}; margin-bottom:8px; "
        f"font-weight:600;'>📋 타각기 출입 기준</div>",
        unsafe_allow_html=True,
    )

    col_wd, col_div, col_we = st.columns([10, 1, 10])

    with col_wd:
        st.markdown(_sub_badge("📅 주중 (평일)", n_wd, "#6EB5FF"), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("일 평균 출입",
                                    f"{_safe_mean(df_wd['workers_access']):.0f}명"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("일 최대 출입",
                                    f"{int(df_wd['workers_access'].max()) if n_wd else 0:,}명"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("연인원 누적",
                                    f"{int(df_wd['workers_access'].sum()):,}명"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("평균 참여 업체",
                                    f"{_safe_mean(df_wd['companies']):.0f}개"),
                        unsafe_allow_html=True)

    with col_div:
        st.markdown(
            f"<div style='border-left:1px solid {COLORS['border']}; height:80px; margin:0 auto;'></div>",
            unsafe_allow_html=True,
        )

    with col_we:
        st.markdown(_sub_badge("🏖️ 주말", n_we, "#FFB86C"), unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(metric_card("일 평균 출입",
                                    f"{_safe_mean(df_we['workers_access']):.0f}명"),
                        unsafe_allow_html=True)
        with c6:
            st.markdown(metric_card("일 최대 출입",
                                    f"{int(df_we['workers_access'].max()) if n_we else 0:,}명"),
                        unsafe_allow_html=True)
        with c7:
            st.markdown(metric_card("연인원 누적",
                                    f"{int(df_we['workers_access'].sum()):,}명"),
                        unsafe_allow_html=True)
        with c8:
            st.markdown(metric_card("평균 참여 업체",
                                    f"{_safe_mean(df_we['companies']):.0f}개"),
                        unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # Row 2 — T-Ward 착용 기준
    # ════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='font-size:0.78rem; color:{COLORS['success']}; margin-bottom:8px; "
        f"font-weight:600;'>📡 T-Ward 착용 기준</div>",
        unsafe_allow_html=True,
    )

    col_wd2, col_div2, col_we2 = st.columns([10, 1, 10])

    # 주중 EWI/CRE/피로도
    wd_ewi = _safe_mean(df_wd["avg_ewi"])
    wd_cre = _safe_mean(df_wd["avg_cre"])
    wd_fat = _safe_mean(df_wd["avg_fatigue"])
    # 주말 EWI/CRE/피로도
    we_ewi = _safe_mean(df_we["avg_ewi"])
    we_cre = _safe_mean(df_we["avg_cre"])
    we_fat = _safe_mean(df_we["avg_fatigue"])

    with col_wd2:
        st.markdown(_sub_badge("📅 주중 (평일)", n_wd, "#6EB5FF"), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("일 평균 T-Ward",
                                    f"{_safe_mean(df_wd['tward_holders']):.0f}명",
                                    color=COLORS["success"]),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("평균 EWI", f"{wd_ewi:.3f}",
                                    color=_ewi_color(wd_ewi)),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("평균 CRE", f"{wd_cre:.3f}",
                                    color=_cre_color(wd_cre)),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("평균 피로도", f"{wd_fat:.3f}",
                                    color=_fat_color(wd_fat)),
                        unsafe_allow_html=True)

    with col_div2:
        st.markdown(
            "<div style='border-left:1px solid #2A3A4A; height:80px; margin:0 auto;'></div>",
            unsafe_allow_html=True,
        )

    with col_we2:
        st.markdown(_sub_badge("🏖️ 주말", n_we, "#FFB86C"), unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(metric_card("일 평균 T-Ward",
                                    f"{_safe_mean(df_we['tward_holders']):.0f}명",
                                    color=COLORS["success"]),
                        unsafe_allow_html=True)
        with c6:
            st.markdown(metric_card("평균 EWI", f"{we_ewi:.3f}",
                                    color=_ewi_color(we_ewi)),
                        unsafe_allow_html=True)
        with c7:
            st.markdown(metric_card("평균 CRE", f"{we_cre:.3f}",
                                    color=_cre_color(we_cre)),
                        unsafe_allow_html=True)
        with c8:
            st.markdown(metric_card("평균 피로도", f"{we_fat:.3f}",
                                    color=_fat_color(we_fat)),
                        unsafe_allow_html=True)


def _render_trend_section(df: pd.DataFrame) -> None:
    """
    일별 추이 차트 2개:
      Left : 출입 인원 vs T-Ward 착용 인원 (Grouped Bar)
      Right: T-Ward 착용률(%) + EWI (이중 Y축 라인)
    """
    st.markdown(section_header("일별 추이"), unsafe_allow_html=True)

    df = df.copy()
    df["date_label"] = df["date"].apply(format_date_label)

    col1, col2 = st.columns(2)

    # ── Left: 출입 vs 착용 인원 ──────────────────────────────────────
    with col1:
        st.markdown(sub_header("출입 인원 vs T-Ward 착용 인원"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["date_label"], y=df["workers_access"],
            name="출입 인원", marker_color=COLOR_ACCESS,
            opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=df["date_label"], y=df["tward_holders"],
            name="T-Ward 착용", marker_color=COLOR_TWARD,
            opacity=0.9,
        ))
        fig.update_layout(
            **PLOTLY_DARK,
            height=260,
            barmode="overlay",
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(title="명", gridcolor=COLORS["grid"]),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Right: 착용률(%) + EWI 추이 ──────────────────────────────────
    with col2:
        st.markdown(sub_header("T-Ward 착용률 (%) · EWI 추이"), unsafe_allow_html=True)
        fig2 = go.Figure()

        # 착용률 — 막대 (주 Y축 %, 짙은 녹색 강조)
        fig2.add_trace(go.Bar(
            x=df["date_label"], y=df["wear_rate"],
            name="착용률 (%)",
            marker_color="#1A5C3A",  # brand-specific deep-green bar; 토큰 없음
            opacity=0.6,
            yaxis="y",
        ))
        # 착용률 라인
        fig2.add_trace(go.Scatter(
            x=df["date_label"], y=df["wear_rate"],
            name="착용률 라인",
            mode="lines+markers",
            line=dict(color=COLOR_TWARD, width=2),
            marker=dict(size=4),
            yaxis="y",
            showlegend=False,
        ))
        # EWI 라인 — 보조 Y축 (0~1)
        fig2.add_trace(go.Scatter(
            x=df["date_label"], y=df["avg_ewi"],
            name="EWI",
            mode="lines+markers",
            line=dict(color=COLOR_WORK, width=2, dash="dot"),
            marker=dict(size=4),
            yaxis="y2",
        ))

        fig2.update_layout(
            **PLOTLY_DARK,
            height=260,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(
                title="착용률 (%)", range=[0, 110],
                gridcolor=COLORS["grid"], ticksuffix="%",
            ),
            yaxis2=dict(
                title="EWI", range=[0, 1],
                overlaying="y", side="right",
                showgrid=False,
            ),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_zone_time_section(sector_id: str, date_str: str, date_label: str) -> None:
    """
    작업시간 분류 도넛 차트 — **메트릭 엔진(src.metrics) 단일 소스** 사용.

    - 카드 수치: raw_ratio (work_minutes 분모) → 정확한 업무상 비율
    - 도넛 내부 라벨: raw_ratio 그대로 표기 (탭 간 수치 동일 보장)
    - 도넛 조각 크기: display_ratio (합=100 정규화) — Plotly 재정규화 왜곡 방지
    """
    from src.metrics import get_time_breakdown, TIME_BREAKDOWN_LABELS

    st.markdown(
        section_header(f"작업시간 분류 — {date_label} (T-Ward 기준)"),
        unsafe_allow_html=True,
    )

    tb = get_time_breakdown(sector_id, date_str)
    if tb.work_minutes <= 0:
        st.info("작업시간 데이터가 없습니다.")
        return

    if tb.consistency_warning:
        logger.debug("time_breakdown consistency: %s", tb.consistency_warning)

    n_workers = max(1, tb.n_workers)

    # 카테고리 순서 고정 — 엔진이 보장하는 순서
    cats = ["work_zone", "transit", "rest", "gap"]
    labels_ko = [TIME_BREAKDOWN_LABELS[c] for c in cats]
    raw_pct   = [tb.raw_pct_for(c)    for c in cats]
    disp_pct  = [tb.display_pct_for(c) for c in cats]
    minutes   = [tb.minutes_for(c)    for c in cats]
    colors_   = [COLOR_WORK, COLOR_TRANSIT, COLOR_REST, COLOR_GAP]

    # 평균 분 (참고 카드용)
    avg_minutes = {c: m / n_workers for c, m in zip(cats, minutes)}

    col1, col2 = st.columns([1, 1])

    with col1:
        # 도넛: 크기는 정규화된 display_pct, 라벨은 raw_pct (카드와 동일 값)
        text_labels = [f"{lbl} {p:.1f}%" for lbl, p in zip(labels_ko, raw_pct)]
        # hover 텍스트는 사전 포맷 (Plotly Pie customdata 호환성 이슈 회피)
        hover_texts = [
            f"<b>{lbl}</b><br>근무시간 대비: {r:.1f}%<br>누적: {int(m):,}분"
            for lbl, r, m in zip(labels_ko, raw_pct, minutes)
        ]
        fig = go.Figure(go.Pie(
            labels=labels_ko, values=disp_pct,
            hole=0.55,
            marker_colors=colors_,
            text=text_labels,
            textinfo="text",
            textfont_size=11,
            insidetextfont=dict(color=COLORS["text"]),
            hovertext=hover_texts,
            hoverinfo="text",
            sort=False,
        ))
        fig.update_layout(
            **{**PLOTLY_DARK, "margin": dict(l=10, r=10, t=30, b=10)},
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        work_ratio = tb.raw_pct_for("work_zone")
        ratio_color = COLORS["accent"] if work_ratio >= 60 else COLORS["warning"]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(metric_card_sm("작업공간 비율", f"{work_ratio:.1f}%",
                                   color=ratio_color),
                    unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card_sm("인당 작업공간",
                                       f"{avg_minutes['work_zone']:.0f}분"),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown(metric_card_sm("인당 이동",
                                       f"{avg_minutes['transit']:.0f}분"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_sm("인당 휴게",
                                       f"{avg_minutes['rest']:.0f}분"),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown(metric_card_sm("BLE 음영",
                                       f"{avg_minutes['gap']:.0f}분"),
                        unsafe_allow_html=True)


def _render_company_ranking(worker_df: pd.DataFrame) -> None:
    """업체별 EWI Top 10 수평 막대 차트 (T-Ward ewi_reliable 기준)."""
    st.markdown(section_header("업체별 EWI 랭킹 Top 10 (T-Ward 기준)"),
                unsafe_allow_html=True)

    wdf = worker_df.copy()
    if "ewi_reliable" in wdf.columns:
        wdf = wdf[wdf["ewi_reliable"] == True]  # noqa: E712
    if wdf.empty or "ewi" not in wdf.columns:
        st.info("EWI 데이터가 없습니다.")
        return

    agg = (
        wdf.groupby("company_name")
        .agg(avg_ewi=("ewi", "mean"), worker_count=("user_no", "nunique"))
        .reset_index()
        .query(f"worker_count >= {COMPANY_MIN_WORKERS['general']}")
        .sort_values("avg_ewi", ascending=False)
        .head(10)
    )

    if agg.empty:
        st.info("집계 가능한 업체가 없습니다 (5명 이상 기준).")
        return

    fig = go.Figure(go.Bar(
        x=agg["avg_ewi"],
        y=agg["company_name"],
        orientation="h",
        marker_color=COLOR_WORK,
        text=[f"{v:.3f}" for v in agg["avg_ewi"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        customdata=agg["worker_count"],
        hovertemplate="<b>%{y}</b><br>EWI: %{x:.3f}<br>T-Ward 작업자: %{customdata}명<extra></extra>",
    ))
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=160, r=60, t=30, b=20)},
        height=max(280, len(agg) * 30 + 60),
        xaxis=dict(
            range=[0, min(1, agg["avg_ewi"].max() * 1.3)],
            title="평균 EWI", gridcolor=COLORS["grid"],
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_shift_section(worker_df: pd.DataFrame) -> None:
    """3교대 분포 파이 차트 (T-Ward 착용 작업자 기준)."""
    if "shift_type" not in worker_df.columns:
        return

    shift_cnt = (
        worker_df["shift_type"]
        .fillna("unknown")
        .value_counts()
        .reset_index()
    )
    shift_cnt.columns = ["shift_type", "count"]

    if shift_cnt.empty:
        return

    labels  = [SHIFT_LABELS.get(s, s) for s in shift_cnt["shift_type"]]
    values  = shift_cnt["count"].tolist()
    colors_ = [SHIFT_COLORS.get(s, COLORS["gap_dark"]) for s in shift_cnt["shift_type"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors_,
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=10, r=10, t=30, b=10)},
        height=240,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── 메인 진입점 ─────────────────────────────────────────────────────

def render_overview_tab(sector_id: str) -> None:
    """현장 개요 탭 메인 렌더링."""
    # ── summary_index 로드 ──────────────────────────────────────────
    idx = _load_summary_index(sector_id)
    idx_dates = idx.get("dates", {})

    if not idx_dates:
        st.info("처리된 데이터가 없습니다. 파이프라인 탭에서 전처리를 실행하세요.")
        return

    df = _build_summary_df(idx_dates)
    if df.empty:
        st.warning("요약 데이터를 구성할 수 없습니다.")
        return

    dates       = sorted(df["date"].tolist())
    start_date  = dates[0]
    end_date    = dates[-1]
    n_days      = len(dates)
    latest_date = end_date

    # ── M4-T34: 최신일 스키마 버전 검증 ──────────────────────────
    # (40일 legacy meta 가 섞여 있어 strict_legacy=False 로 관용 처리)
    try:
        from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
        from src.dashboard.components import handle_schema_mismatch
        validate_schema(latest_date, sector_id, strict_legacy=False)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, latest_date)
        return
    except FileNotFoundError:
        # meta.json 자체가 없으면 다음 블록에서 자연스럽게 경고
        pass

    # ── 헤더 ────────────────────────────────────────────────────────
    try:
        from src.dashboard.date_utils import format_date_full
        end_label = format_date_full(latest_date)
    except Exception:
        end_label = latest_date

    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:16px; margin-bottom:6px;'>
            <div style='font-size:1.4rem; font-weight:700; color:{COLORS["text"]};'>
                현장 전체 기간 요약
            </div>
            <div style='background:{COLORS["primary"]}; border-radius:6px; padding:3px 10px;
                        font-size:0.82rem; color:{COLORS["accent"]}; font-weight:600;'>
                {n_days}일
            </div>
        </div>
        <div style='font-size:0.88rem; color:{COLORS["text_dimmer"]}; margin-bottom:16px;'>
            {start_date} &nbsp;→&nbsp; {end_date}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Section 0: T-Ward 착용률 하이라이트 배너 ─────────────────────
    _render_tward_banner(df)

    # ── Section 0.5: AI 코멘터리 (T-15) ──────────────────────────────
    try:
        from src.dashboard.components import ai_commentary_box
        from core.ai import build_overview_context
        from src.dashboard.auth import get_current_user

        worker_df_for_ai = _load_latest_worker(sector_id, latest_date)
        ai_ctx = build_overview_context(
            sector_id=sector_id,
            latest_date=str(latest_date),
            summary_df=df,
            worker_df=worker_df_for_ai if not worker_df_for_ai.empty else None,
            top_n_companies=3,
        )
        ai_commentary_box(
            role="overview_commentator",
            context=ai_ctx,
            sector_id=sector_id,
            date_str=str(latest_date),
            title="오늘의 현장 코멘트",
            spinner_text="현장 상황을 분석 중...",
            button_label="AI 분석 실행 (Haiku)",
            user_role=get_current_user().get("role", "unknown"),
            tab="overview",
            show_meta=False,
        )
    except Exception as e:
        logger.warning(f"AI 코멘터리 렌더 실패 (overview): {e}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Section 1: 전체 기간 KPI ─────────────────────────────────────
    _render_kpi_section(df)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Section 2: 일별 추이 ─────────────────────────────────────────
    _render_trend_section(df)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 최신일 worker.parquet 로드 ───────────────────────────────────
    worker_df = _load_latest_worker(sector_id, latest_date)

    if not worker_df.empty:
        # ── Section 3: 작업시간 분류 도넛 (메트릭 엔진 사용) ───────
        latest_date_str = str(latest_date).replace("-", "")
        _render_zone_time_section(sector_id, latest_date_str, end_label)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Section 4 + 5: 업체 랭킹 & 3교대 분포 ───────────────────
        col_left, col_right = st.columns([2, 1])
        with col_left:
            _render_company_ranking(worker_df)
        with col_right:
            st.markdown(section_header("3교대 분포 (최신일, T-Ward 기준)"),
                        unsafe_allow_html=True)
            _render_shift_section(worker_df)
