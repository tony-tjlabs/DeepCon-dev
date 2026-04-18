"""
생산성 분석 탭 — EWI · CRE 중심의 생산성 지표 다각도 분석
==========================================================
5개 서브탭:
  1. 📊 일별 — 선택 날짜의 생산성 상세 (기존 daily/productivity 재활용)
  2. 🏗️ 업체별 — 업체 랭킹, 장기 추이, 우수/주의 업체 식별
  3. 🗺️ 공간별 — Locus별 평균 EWI, 병목 공간 식별
  4. 📅 기간별 — 전체 기간 EWI/CRE 추이 + 요일 패턴
  5. 👤 개인별 — 작업자 랭킹 (EWI/피로도/고활성 등), 위험 배지, 타임라인 상세

데이터 소스:
  - worker.parquet (일별)
  - company.parquet (업체별)
  - space.parquet (공간별)
  - summary_index.json (기간별)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from core.cache.policy import DAILY_PARQUET, MULTI_DAY_AGG
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
from src.dashboard.date_utils import get_date_selector, format_date_label
from src.pipeline.cache_manager import detect_processed_dates
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import COMPANY_MIN_WORKERS

logger = logging.getLogger(__name__)

# ─── 색상 상수 (styles.CHART_COLORS 단일 소스 참조) ───────────────────
COLOR_EWI = CHART_COLORS["ewi"]
COLOR_CRE = CHART_COLORS["cre"]
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


# ═══════════════════════════════════════════════════════════════════════
# 공통 로더
# ═══════════════════════════════════════════════════════════════════════

# ★ 성능: 각 parquet에서 사용 컬럼만 명시 로드 (전체 로드 대비 ~30% 절감)
# 주의: processor.py의 "*_minutes"(분 단위 기본 집계) + metrics.py의 "*_min"(EWI 분류) 모두 사용됨
_WORKER_COLS = [
    "user_no", "user_name", "company_name",
    # identity / shift
    "in_datetime", "out_datetime", "work_minutes", "shift_type", "exit_source",
    # basic aggregates (processor.calc_basic_metrics)
    "recorded_minutes", "active_minutes", "valid_ble_minutes", "unique_loci",
    "locus_sequence", "confined_minutes", "high_voltage_minutes",
    "transition_count", "work_zone_minutes", "rest_minutes",
    "helmet_abandoned",
    # EWI & CRE & SII (metrics.add_metrics_to_worker)
    "ewi", "cre", "sii", "ewi_reliable", "ewi_calculable",
    "high_active_min", "low_active_min", "standby_min",
    "rest_min", "transit_min", "recorded_work_min",
    "gap_ratio", "gap_min", "static_risk",
    "personal_norm", "static_norm", "dynamic_norm",
    "fatigue_score", "max_active_streak", "alone_ratio",
    "ble_coverage", "ble_coverage_pct",
]
_SPACE_COLS = [
    "locus_id", "locus_token", "locus_name",
    "total_person_minutes", "unique_workers", "unique_companies",
    "avg_signal_count", "avg_active_ratio",
    "is_confined", "is_high_voltage",
    # space에 EWI가 있을 수 있음 (도메인 확장용)
    "avg_ewi", "avg_cre",
]
_COMPANY_COLS = [
    "company_name", "worker_count", "total_person_minutes",
    "avg_work_zone_minutes", "avg_rest_minutes",
    "confined_workers", "total_confined_minutes",
    "avg_ewi", "avg_cre", "total_high_active_min",
]


def _read_parquet_projected(path: "Path", desired: list[str]) -> pd.DataFrame:
    """존재하는 컬럼만 projection하여 읽기 (pyarrow 스키마 확인)."""
    try:
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(path).names)
        cols = [c for c in desired if c in avail]
        if not cols:
            return pd.read_parquet(path)
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _load_daily(sector_id: str, date_str: str) -> dict:
    """특정 날짜의 worker/space/company parquet 로드 (컬럼 프루닝 적용)."""
    paths = cfg.get_sector_paths(sector_id)
    date_dir = paths["processed_dir"] / date_str
    out = {}
    col_map = {"worker": _WORKER_COLS, "space": _SPACE_COLS, "company": _COMPANY_COLS}
    for name in ["worker", "space", "company"]:
        p = date_dir / f"{name}.parquet"
        if p.exists():
            out[name] = _read_parquet_projected(p, col_map[name])
        else:
            out[name] = pd.DataFrame()
    return out


_PROD_WORKER_MULTI_COLS = [
    "user_no", "company_name", "ewi", "cre", "sii",
    "ewi_reliable", "shift_type", "work_minutes",
    "work_zone_minutes",
]


def _prod_load_worker_one(args: tuple):
    """단일 날짜 worker.parquet 로드 — 병렬 worker용."""
    d, p = args
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=_PROD_WORKER_MULTI_COLS)
        df["date"] = d
        return df
    except Exception as e:
        logger.warning(f"[{d}] worker 로드 실패: {e}")
        return None


def _prod_load_space_one(args: tuple):
    """단일 날짜 space.parquet 로드 — 병렬 worker용."""
    d, p = args
    if not p.exists():
        return None
    try:
        df = _read_parquet_projected(p, _SPACE_COLS)
        df["date"] = d
        return df
    except Exception as e:
        logger.warning(f"[{d}] space 로드 실패: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=MULTI_DAY_AGG)
def _load_multi_day_workers(sector_id: str, dates: tuple) -> pd.DataFrame:
    """여러 날짜의 worker.parquet 병렬 집계.

    ★ 성능: ThreadPoolExecutor로 I/O 병렬화 (40일 순차 → 8 worker 병렬).
    """
    from concurrent.futures import ThreadPoolExecutor
    paths = cfg.get_sector_paths(sector_id)
    tasks = [(d, paths["processed_dir"] / d / "worker.parquet") for d in dates]
    if not tasks:
        return pd.DataFrame()
    max_w = min(len(tasks), 8)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        frames = [f for f in pool.map(_prod_load_worker_one, tasks) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=MULTI_DAY_AGG)
def _load_multi_day_spaces(sector_id: str, dates: tuple) -> pd.DataFrame:
    """여러 날짜의 space.parquet 병렬 집계 (컬럼 프루닝 적용)."""
    from concurrent.futures import ThreadPoolExecutor
    paths = cfg.get_sector_paths(sector_id)
    tasks = [(d, paths["processed_dir"] / d / "space.parquet") for d in dates]
    if not tasks:
        return pd.DataFrame()
    max_w = min(len(tasks), 8)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        frames = [f for f in pool.map(_prod_load_space_one, tasks) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 1: 일별
# ═══════════════════════════════════════════════════════════════════════

def _render_daily_subtab(sector_id: str, date_str: str, shift_filter: str | None) -> None:
    """
    선택 날짜의 생산성 상세.
    기존 src.dashboard.daily.productivity.render_productivity() 재활용.
    """
    from src.dashboard.daily.productivity import render_productivity
    from src.spatial.loader import load_locus_dict
    from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
    from src.dashboard.components import handle_schema_mismatch

    # ── M4-T34: 스키마 버전 검증 — 불일치 시 재처리 CTA 표시 후 종료 ──
    try:
        validate_schema(date_str, sector_id, strict_legacy=False)
        data = _load_daily(sector_id, date_str)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, date_str)
        return
    worker_df = data["worker"]
    space_df  = data["space"]

    if worker_df.empty:
        st.warning("해당 날짜의 worker 데이터가 없습니다.")
        return

    # 교대 필터 적용
    if shift_filter and "shift_type" in worker_df.columns:
        worker_df = worker_df[worker_df["shift_type"] == shift_filter]

    # ── AI 코멘터리 (T-17) ──────────────────────────────────────────
    try:
        from src.dashboard.components import ai_commentary_box
        from core.ai import build_productivity_context
        from src.dashboard.auth import get_current_user
        import logging as _logging

        ai_ctx = build_productivity_context(
            sector_id=sector_id,
            date_str=str(date_str),
            worker_df=worker_df,
            space_df=space_df,
        )
        ai_commentary_box(
            role="productivity_analyst",
            context=ai_ctx,
            sector_id=sector_id,
            date_str=str(date_str),
            title="층별 생산성 AI 분석",
            spinner_text="층별·업체별 생산성 분석 중...",
            button_label="AI 분석 실행 (Haiku)",
            user_role=get_current_user().get("role", "unknown"),
            tab="productivity",
            show_meta=False,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    except Exception as e:
        _logging.getLogger(__name__).warning(f"AI 코멘터리 렌더 실패 (productivity): {e}")

    has_ewi = "ewi" in worker_df.columns
    locus_dict = load_locus_dict(sector_id)
    render_productivity(worker_df, space_df, locus_dict, has_ewi)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 2: 업체별
# ═══════════════════════════════════════════════════════════════════════

def _render_company_subtab(sector_id: str, dates: list[str]) -> None:
    """
    업체별 생산성 — 선택 기간 내 업체별 EWI/CRE 랭킹 + 추이.
    """
    st.markdown(section_header("업체별 생산성 분석"), unsafe_allow_html=True)

    # 기간 슬라이더
    col_range, col_info = st.columns([2, 2])
    with col_range:
        if len(dates) == 1:
            selected_dates = dates
            st.caption(f"기간: {dates[0]}")
        else:
            s_idx, e_idx = st.select_slider(
                "분석 기간",
                options=list(range(len(dates))),
                value=(max(0, len(dates) - 14), len(dates) - 1),
                format_func=lambda i: dates[i],
                key="prod_company_range",
            )
            selected_dates = dates[s_idx:e_idx + 1]

    with col_info:
        st.caption(f"선택된 기간: {selected_dates[0]} ~ {selected_dates[-1]} "
                   f"({len(selected_dates)}일)")

    # 데이터 로드
    wdf = _load_multi_day_workers(sector_id, tuple(selected_dates))
    if wdf.empty:
        st.warning("데이터 없음.")
        return

    # ewi_reliable 필터
    if "ewi_reliable" in wdf.columns:
        wdf = wdf[wdf["ewi_reliable"] == True]  # noqa: E712

    # 업체별 집계
    agg = (
        wdf.groupby("company_name")
        .agg(
            avg_ewi=("ewi", "mean"),
            avg_cre=("cre", "mean"),
            avg_sii=("sii", "mean"),
            worker_count=("user_no", "nunique"),
            total_days=("date", "nunique"),
        )
        .reset_index()
        .query(f"worker_count >= {COMPANY_MIN_WORKERS['statistical']}")   # 통계 신뢰 기준
        .sort_values("avg_ewi", ascending=False)
    )

    if agg.empty:
        st.info("집계 가능한 업체가 없습니다 (10명 이상 기준).")
        return

    # ── Top 15 업체 바 차트 ─────────────────────────────────────────
    st.markdown(sub_header(f"EWI 상위 15 업체 (기간 평균, {len(selected_dates)}일)"),
                unsafe_allow_html=True)
    top15 = agg.head(15)
    fig = go.Figure(go.Bar(
        x=top15["avg_ewi"],
        y=top15["company_name"],
        orientation="h",
        marker_color=COLOR_EWI,
        text=[f"{v:.3f}" for v in top15["avg_ewi"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        customdata=np.stack([top15["worker_count"], top15["avg_cre"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "EWI: %{x:.3f}<br>"
            "CRE: %{customdata[1]:.3f}<br>"
            "작업자: %{customdata[0]}명<extra></extra>"
        ),
    ))
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=200, r=60, t=20, b=20)},
        height=max(380, len(top15) * 28 + 50),
        xaxis=dict(range=[0, min(1, top15["avg_ewi"].max() * 1.2)],
                   title="평균 EWI", gridcolor=COLORS["grid"]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 우수 / 주의 업체 ──────────────────────────────────────────
    col_good, col_warn = st.columns(2)
    with col_good:
        st.markdown(sub_header("✨ 우수 업체 Top 5"), unsafe_allow_html=True)
        top5 = agg.head(5)[["company_name", "avg_ewi", "avg_cre", "worker_count"]]
        top5.columns = ["업체", "평균 EWI", "평균 CRE", "작업자"]
        st.dataframe(top5.round(3), use_container_width=True, hide_index=True)

    with col_warn:
        st.markdown(sub_header("⚠ 주의 업체 Bottom 5"), unsafe_allow_html=True)
        bot5 = agg.tail(5)[["company_name", "avg_ewi", "avg_cre", "worker_count"]]
        bot5.columns = ["업체", "평균 EWI", "평균 CRE", "작업자"]
        st.dataframe(bot5.round(3), use_container_width=True, hide_index=True)

    # ── 업체별 EWI 추이 (선택 업체만) ─────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(sub_header("업체별 일자별 EWI 추이"), unsafe_allow_html=True)

    compare_names = st.multiselect(
        "비교할 업체 선택 (최대 8개)",
        options=agg["company_name"].tolist(),
        default=agg["company_name"].head(5).tolist(),
        max_selections=8,
        key="prod_company_compare",
    )

    if compare_names:
        import pandas as _pd
        trend = (
            wdf[wdf["company_name"].isin(compare_names)]
            .groupby(["date", "company_name"])
            .agg(avg_ewi=("ewi", "mean"))
            .reset_index()
        )
        # YYYYMMDD 문자열 → datetime (Plotly 숫자 포맷 방지)
        trend["date"] = _pd.to_datetime(trend["date"].astype(str), format="%Y%m%d", errors="coerce")
        fig2 = px.line(
            trend, x="date", y="avg_ewi", color="company_name",
            markers=True,
        )
        fig2.update_layout(
            **PLOTLY_DARK, height=360,
            xaxis=dict(tickangle=-45, gridcolor=COLORS["grid"],
                       tickformat="%m/%d"),
            yaxis=dict(title="평균 EWI", range=[0, 1], gridcolor=COLORS["grid"]),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": -0.3,
                    "xanchor": "center", "x": 0.5},
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 3: 공간별
# ═══════════════════════════════════════════════════════════════════════

def _render_space_subtab(sector_id: str, dates: list[str]) -> None:
    """공간별 생산성 — 층(building+floor) 단위 집계.

    space.parquet(locus_id 기준)을 locus_v2.csv와 join하여
    building+floor 단위로 재집계. WORK_AREA locus만 포함.

    지표:
      - 활성 비율 (wavg_active_ratio): person_minutes 가중평균 — T-Ward 신호 기반 활동성
      - 총 체류 인분 (total_person_minutes): 해당 층에서 발생한 전체 체류
      - 연인원 (unique_workers): 해당 층을 방문한 고유 작업자 수 (일 합산)
    """
    st.markdown(section_header("층별 생산성 분석"), unsafe_allow_html=True)
    st.caption(
        "작업구역(WORK_AREA) locus를 건물·층 단위로 묶어 집계합니다. "
        "활성 비율은 T-Ward 신호 기반 가중평균(체류시간 기준)입니다."
    )

    # ── 기간 선택 ─────────────────────────────────────────────────────
    col_range, col_info = st.columns([2, 2])
    with col_range:
        if len(dates) == 1:
            selected_dates = dates
        else:
            s_idx, e_idx = st.select_slider(
                "분석 기간",
                options=list(range(len(dates))),
                value=(max(0, len(dates) - 7), len(dates) - 1),
                format_func=lambda i: dates[i],
                key="prod_space_range",
            )
            selected_dates = dates[s_idx:e_idx + 1]
    with col_info:
        st.caption(f"선택 기간: {selected_dates[0]} ~ {selected_dates[-1]} "
                   f"({len(selected_dates)}일)")

    sdf = _load_multi_day_spaces(sector_id, tuple(selected_dates))
    if sdf.empty or "locus_id" not in sdf.columns:
        st.warning("공간 집계 데이터가 없습니다.")
        return

    # ── locus_v2.csv join ────────────────────────────────────────────
    try:
        import config as cfg
        import pandas as _pd
        paths = cfg.get_sector_paths(sector_id)
        ldf = _pd.read_csv(paths["locus_v2_csv"], encoding="utf-8-sig")
    except Exception as e:
        st.error(f"locus_v2.csv 로드 실패: {e}")
        return

    # WORK_AREA만 필터
    work_loci = ldf[ldf["locus_type"] == "WORK_AREA"][
        ["locus_id", "locus_name", "building", "floor", "floor_no", "building_no"]
    ]
    merged = sdf.merge(work_loci, on="locus_id", how="inner")
    if merged.empty:
        st.warning("WORK_AREA locus와 매칭되는 데이터가 없습니다.")
        return

    # ── 층별 집계 ─────────────────────────────────────────────────────
    def _floor_agg(g):
        pm = g["total_person_minutes"].sum()
        wavg = (
            (g["avg_active_ratio"] * g["total_person_minutes"]).sum() / pm
            if pm > 0 else 0.0
        )
        return pd.Series({
            "total_person_minutes": pm,
            "unique_workers":       g["unique_workers"].sum(),
            "unique_companies":     int(g["unique_companies"].max()),
            "locus_count":          len(g),
            "wavg_active_ratio":    round(wavg, 4),
        })

    floor_agg = (
        merged.groupby(["building", "floor", "building_no", "floor_no"])
        .apply(_floor_agg)
        .reset_index()
        .sort_values(["building_no", "floor_no"])
    )
    floor_agg["floor_label"] = floor_agg["building"] + " " + floor_agg["floor"]

    # ── 건물 필터 ──────────────────────────────────────────────────
    buildings = sorted(floor_agg["building"].unique().tolist())
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        sel_bld = st.multiselect(
            "건물 필터",
            options=buildings,
            default=[],
            placeholder="전체 (미선택 시 전체 표시)",
            key="prod_space_bld_filter",
        )
    if sel_bld:
        floor_agg = floor_agg[floor_agg["building"].isin(sel_bld)]

    if floor_agg.empty:
        st.info("선택된 건물에 해당하는 층 데이터가 없습니다.")
        return

    # ── KPI 카드 (전체 요약) ──────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            metric_card("분석 층 수", f"{len(floor_agg)}개층",
                        color=CHART_COLORS["info"]),
            unsafe_allow_html=True,
        )
    with k2:
        best = floor_agg.loc[floor_agg["wavg_active_ratio"].idxmax(), "floor_label"]
        best_v = floor_agg["wavg_active_ratio"].max()
        st.markdown(
            metric_card("최고 활성층", f"{best}  {best_v:.3f}",
                        color=CHART_COLORS["rest"]),
            unsafe_allow_html=True,
        )
    with k3:
        worst = floor_agg.loc[floor_agg["wavg_active_ratio"].idxmin(), "floor_label"]
        worst_v = floor_agg["wavg_active_ratio"].min()
        st.markdown(
            metric_card("최저 활성층", f"{worst}  {worst_v:.3f}",
                        color=CHART_COLORS["critical"]),
            unsafe_allow_html=True,
        )
    with k4:
        overall = (
            (floor_agg["wavg_active_ratio"] * floor_agg["total_person_minutes"]).sum()
            / floor_agg["total_person_minutes"].sum()
        )
        st.markdown(
            metric_card("전체 평균 활성", f"{overall:.3f}",
                        color=CHART_COLORS["ewi"]),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 차트 1: 층별 활성 비율 (가로 막대) ──────────────────────────
    # 활성 비율 내림차순 정렬 (높을수록 상단)
    plot_df = floor_agg.sort_values("wavg_active_ratio", ascending=True)

    # 건물별 색상 구분
    bld_palette = [
        CHART_COLORS.get("work_zone", "#4FC3F7"),
        CHART_COLORS.get("rest",      "#00C897"),
        CHART_COLORS.get("transit",   "#FFB86C"),
        CHART_COLORS.get("critical",  "#FF6B6B"),
        CHART_COLORS.get("medium",    "#FFA726"),
        "#9C77E0", "#5CE8FF", "#FF8A65",
    ]
    bld_list = sorted(plot_df["building"].unique())
    bld_color = {b: bld_palette[i % len(bld_palette)] for i, b in enumerate(bld_list)}
    bar_colors = [bld_color[b] for b in plot_df["building"]]

    # 활성 비율 기준선 (전체 평균)
    avg_line = float(
        (floor_agg["wavg_active_ratio"] * floor_agg["total_person_minutes"]).sum()
        / floor_agg["total_person_minutes"].sum()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["wavg_active_ratio"],
        y=plot_df["floor_label"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in plot_df["wavg_active_ratio"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        customdata=plot_df[["total_person_minutes", "unique_workers",
                             "unique_companies", "locus_count"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "활성 비율: %{x:.3f}<br>"
            "총 체류: %{customdata[0]:,.0f}인분<br>"
            "연인원: %{customdata[1]:,.0f}명<br>"
            "참여 업체: %{customdata[2]}개<br>"
            "게이트웨이: %{customdata[3]}개<extra></extra>"
        ),
    ))
    # 전체 평균선
    fig.add_vline(
        x=avg_line, line_dash="dash", line_color="#FFD700", line_width=1.5,
        annotation_text=f"평균 {avg_line:.3f}",
        annotation_font_color="#FFD700",
        annotation_position="top right",
    )
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=110, r=60, t=30, b=20)},
        title="층별 T-Ward 활성 비율 (체류시간 가중평균, WORK_AREA 기준)",
        height=max(300, len(plot_df) * 38 + 60),
        xaxis=dict(title="활성 비율 (0~1)", range=[0, 1.05], gridcolor=COLORS["grid"]),
        yaxis=dict(autorange=True),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 차트 2: 층별 총 체류 인분 (누적 — 건물별 색상) ──────────────
    plot_df2 = floor_agg.sort_values(["building_no", "floor_no"])
    fig2 = go.Figure()
    for bld in bld_list:
        sub = plot_df2[plot_df2["building"] == bld]
        fig2.add_trace(go.Bar(
            name=bld,
            x=sub["floor_label"],
            y=sub["total_person_minutes"],
            marker_color=bld_color[bld],
            text=[f"{v/1000:.1f}K" for v in sub["total_person_minutes"]],
            textposition="outside",
            textfont=dict(size=9, color=COLORS["text"]),
        ))
    fig2.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=20, r=20, t=40, b=60)},
        title="층별 총 체류 인분 (WORK_AREA, 기간 합산)",
        barmode="group",
        height=360,
        xaxis=dict(title="층", tickangle=-30, gridcolor=COLORS["grid"]),
        yaxis=dict(title="체류 인분", gridcolor=COLORS["grid"]),
        legend=PLOTLY_LEGEND,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 상세 테이블 ──────────────────────────────────────────────────
    with st.expander("📋 층별 집계 상세 테이블", expanded=False):
        disp = floor_agg[[
            "floor_label", "wavg_active_ratio", "total_person_minutes",
            "unique_workers", "unique_companies", "locus_count"
        ]].rename(columns={
            "floor_label":          "층",
            "wavg_active_ratio":    "활성 비율",
            "total_person_minutes": "총 체류(인분)",
            "unique_workers":       "연인원",
            "unique_companies":     "참여 업체",
            "locus_count":          "게이트웨이 수",
        }).sort_values("활성 비율", ascending=False)
        st.dataframe(disp.round(4), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 4: 기간별 (트렌드 + 요일 패턴)
# ═══════════════════════════════════════════════════════════════════════

def _render_period_subtab(sector_id: str) -> None:
    """전체 기간 생산성 추이 + 요일별 패턴."""
    st.markdown(section_header("전체 기간 생산성 추이"), unsafe_allow_html=True)

    from src.pipeline.summary_index import load_summary_index
    idx = load_summary_index(sector_id)
    dates_idx = idx.get("dates", {})

    if not dates_idx:
        st.info("summary_index 데이터가 없습니다.")
        return

    rows = []
    for d, v in sorted(dates_idx.items()):
        rows.append({
            "date":    d,
            "avg_ewi": v.get("avg_ewi", 0),
            "avg_cre": v.get("avg_cre", 0),
            "avg_sii": v.get("avg_sii", 0),
            "workers": v.get("tward_holders", 0),
        })
    df = pd.DataFrame(rows)
    df["date_label"] = df["date"].apply(format_date_label)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["dow"] = df["date_parsed"].dt.dayofweek

    # ── 추이 차트 ──────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_ewi"],
        name="EWI", mode="lines+markers",
        line=dict(color=COLOR_EWI, width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_cre"],
        name="CRE", mode="lines+markers",
        line=dict(color=COLOR_CRE, width=2, dash="dot"),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_sii"],
        name="SII", mode="lines+markers",
        line=dict(color=COLORS["text_muted"], width=2, dash="dash"),
        marker=dict(size=4),
    ))
    fig.update_layout(
        **PLOTLY_DARK, height=320,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="지수", range=[0, 1], gridcolor=COLORS["grid"]),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 요일별 패턴 ────────────────────────────────────────────────
    st.markdown(sub_header("요일별 평균 지표 (월~일)"), unsafe_allow_html=True)

    dow_agg = df.groupby("dow").agg(
        avg_ewi=("avg_ewi", "mean"),
        avg_cre=("avg_cre", "mean"),
        workers=("workers", "mean"),
    ).reset_index()
    dow_agg["dow_name"] = dow_agg["dow"].map(lambda i: DOW_KR[i])

    col_l, col_r = st.columns(2)
    with col_l:
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=dow_agg["dow_name"], y=dow_agg["avg_ewi"],
            name="평균 EWI", marker_color=COLOR_EWI,
            text=[f"{v:.3f}" for v in dow_agg["avg_ewi"]],
            textposition="outside",
        ))
        fig_dow.update_layout(
            **PLOTLY_DARK, height=280,
            yaxis=dict(title="평균 EWI", range=[0, 1], gridcolor=COLORS["grid"]),
            showlegend=False,
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_r:
        fig_wc = go.Figure()
        fig_wc.add_trace(go.Bar(
            x=dow_agg["dow_name"], y=dow_agg["workers"],
            marker_color=CHART_COLORS["tward"],   # T-Ward 착용률 = 녹색
            text=[f"{v:.0f}" for v in dow_agg["workers"]],
            textposition="outside",
        ))
        fig_wc.update_layout(
            **PLOTLY_DARK, height=280,
            yaxis=dict(title="평균 T-Ward 착용", gridcolor=COLORS["grid"]),
            showlegend=False,
        )
        st.plotly_chart(fig_wc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 5: 👤 개인별 — 작업자 랭킹 + 배지 + 상세
# ═══════════════════════════════════════════════════════════════════════

# 정렬 옵션 (상수 — 매직 문자열 방지)
_SORT_OPTIONS_PROD: dict[str, tuple[str, bool]] = {
    "EWI 높은 순 (생산성↑)":     ("ewi",                False),
    "EWI 낮은 순 (관찰 필요)":    ("ewi",                True),
    "피로도 높은 순":              ("fatigue_score",      False),
    "고활성 시간 긴 순":          ("high_active_min",    False),
    "저활성 시간 긴 순 (대기)":   ("low_active_min",     False),
    "작업공간 비율 높은 순":      ("work_zone_ratio",    False),
    "작업공간 비율 낮은 순":      ("work_zone_ratio",    True),
    "연속 고활성 긴 순 (무리)":   ("max_active_streak",  False),
    "BLE 커버리지 낮은 순":       ("ble_coverage_pct",   True),
    "총 작업시간 긴 순":           ("work_minutes",       False),
}


def _nan_safe_float(v) -> float:
    """pd.NA / NaN / None → 0.0, 그 외는 float 캐스팅."""
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _prod_individual_badges(row: pd.Series) -> str:
    """경고 배지 HTML — 작업자 단위 위험 플래그 시각화."""
    badges: list[str] = []
    streak = _nan_safe_float(row.get("max_active_streak"))
    low_min = _nan_safe_float(row.get("low_active_min"))
    work_min = _nan_safe_float(row.get("work_minutes"))
    cov_pct = _nan_safe_float(row.get("ble_coverage_pct"))

    if streak > 180:
        badges.append(
            f"<span style='background:{CHART_COLORS['fatigue']}22;color:{CHART_COLORS['fatigue']};"
            f"padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:600;"
            f"border:1px solid {CHART_COLORS['fatigue']}66;'>⚠ 피로 누적</span>"
        )
    if work_min > 0 and (low_min / work_min) > 0.7:
        badges.append(
            f"<span style='background:{CHART_COLORS['low_active']}22;color:{CHART_COLORS['low_active']};"
            f"padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:600;"
            f"border:1px solid {CHART_COLORS['low_active']}66;'>🔇 저활성</span>"
        )
    # ★ v3.0.1: 음영지역 경고 (BLE 커버리지 기반) — 3단계 차등
    #   EWI는 이제 항상 work_minutes(타각기)가 분모이므로 커버리지 낮으면
    #   실제 활동보다 낮게 집계될 수 있음 — 해석 시 참고 배지.
    if cov_pct < 30:
        badges.append(
            f"<span style='background:{CHART_COLORS['critical']}22;color:{CHART_COLORS['critical']};"
            f"padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:600;"
            f"border:1px solid {CHART_COLORS['critical']}66;'>🛑 음영심각 ({cov_pct:.0f}%)</span>"
        )
    elif cov_pct < 50:
        badges.append(
            f"<span style='background:{CHART_COLORS['medium']}22;color:{CHART_COLORS['medium']};"
            f"padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:600;"
            f"border:1px solid {CHART_COLORS['medium']}66;'>📉 음영많음 ({cov_pct:.0f}%)</span>"
        )
    elif cov_pct < 70:
        badges.append(
            f"<span style='background:{CHART_COLORS['info']}22;color:{CHART_COLORS['info']};"
            f"padding:2px 8px;border-radius:10px;font-size:0.74rem;font-weight:600;"
            f"border:1px solid {CHART_COLORS['info']}66;'>📡 커버리지 {cov_pct:.0f}%</span>"
        )
    return " ".join(badges)


def _render_individual_productivity_subtab(
    sector_id: str, date_str: str, shift_filter: str | None
) -> None:
    """개인별 생산성 랭킹 + 상세.

    현장 운영 담당자가 "EWI 높은/낮은 사람", "피로도 누적자", "저활성 누적자"를
    즉시 식별할 수 있게 한다. 필터·정렬·배지·상세 모두 pandas vectorize로 처리.
    """
    st.markdown(section_header("개인별 생산성 랭킹"), unsafe_allow_html=True)

    # ── 데이터 로드 ─────────────────────────────────────────────────
    from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
    from src.dashboard.components import handle_schema_mismatch

    try:
        validate_schema(date_str, sector_id, strict_legacy=False)
        data = _load_daily(sector_id, date_str)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, date_str)
        return

    wdf = data.get("worker", pd.DataFrame())
    if wdf.empty:
        st.warning("해당 날짜의 worker 데이터가 없습니다.")
        return

    # 교대 필터 (상단 공통 필터 반영)
    if shift_filter and "shift_type" in wdf.columns:
        wdf = wdf[wdf["shift_type"] == shift_filter]

    if wdf.empty:
        st.info("선택한 교대에 해당하는 작업자가 없습니다.")
        return

    # ── work_zone_ratio 파생 (정렬 키) ─────────────────────────────
    wz = pd.to_numeric(wdf.get("work_zone_minutes", 0), errors="coerce").fillna(0)
    wm = pd.to_numeric(wdf.get("work_minutes", 0), errors="coerce").fillna(0)
    wdf = wdf.copy()
    wdf["work_zone_ratio"] = np.where(wm > 0, wz / wm, 0.0)

    # ── 필터 영역 ───────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{COLORS['bg_chart']}; border:1px solid {COLORS['border']}; "
        f"border-radius:8px; padding:12px 16px; margin-bottom:12px;'>",
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 3])

    with fc1:
        companies = sorted(wdf["company_name"].dropna().unique().tolist())
        selected_companies = st.multiselect(
            "업체 필터",
            options=companies,
            default=[],
            placeholder="전체 (미선택 시 전체)",
            key="prod_indiv_company",
        )

    with fc2:
        shift_options_sub = {"전체": None, "주간": "day",
                             "야간": "night", "연장야간": "extended_night"}
        shift_label_sub = st.selectbox(
            "교대 (서브)",
            list(shift_options_sub.keys()),
            index=0,
            key="prod_indiv_shift",
        )
        shift_sub = shift_options_sub[shift_label_sub]

    with fc3:
        uno_search = st.text_input(
            "User_no 검색",
            value="",
            placeholder="예) 12345",
            key="prod_indiv_uno",
        )

    with fc4:
        min_work = st.slider(
            "최소 체류(분)",
            min_value=0, max_value=480, value=30, step=15,
            key="prod_indiv_minwork",
            help="짧은 체류(점심만 찍힘 등)를 제외합니다.",
        )

    fc5, fc6 = st.columns([2, 2])
    with fc5:
        reliable_only = st.checkbox(
            "EWI 신뢰 가능한 작업자만 (ewi_reliable)",
            value=False,
            key="prod_indiv_reliable",
            help="BLE 음영이 적어 EWI 계산이 신뢰 가능한 작업자만 표시합니다.",
        )
    with fc6:
        sort_key = st.selectbox(
            "정렬 기준",
            options=list(_SORT_OPTIONS_PROD.keys()),
            index=0,
            key="prod_indiv_sort",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 필터 적용 (vectorized) ─────────────────────────────────────
    filt = wdf.copy()
    if selected_companies:
        filt = filt[filt["company_name"].isin(selected_companies)]
    if shift_sub and "shift_type" in filt.columns:
        filt = filt[filt["shift_type"] == shift_sub]
    uno_q = uno_search.strip()
    if uno_q:
        filt = filt[filt["user_no"].astype(str).str.contains(uno_q, na=False)]
    filt = filt[pd.to_numeric(filt["work_minutes"], errors="coerce").fillna(0) >= min_work]
    if reliable_only and "ewi_reliable" in filt.columns:
        filt = filt[filt["ewi_reliable"] == True]  # noqa: E712

    if filt.empty:
        st.warning("조건에 맞는 작업자가 없습니다. 필터를 조정해주세요.")
        return

    # ── 정렬 ────────────────────────────────────────────────────────
    sort_col, ascending = _SORT_OPTIONS_PROD[sort_key]
    if sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=ascending, na_position="last")
    filt = filt.reset_index(drop=True)

    # ── KPI 4개 ────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    n_workers = len(filt)
    ewi_v = pd.to_numeric(filt["ewi"], errors="coerce")
    avg_ewi = ewi_v.mean() if len(ewi_v.dropna()) else 0
    std_ewi = ewi_v.std()  if len(ewi_v.dropna()) else 0
    avg_fat = pd.to_numeric(filt.get("fatigue_score", 0), errors="coerce").mean() or 0
    ha_total = pd.to_numeric(filt.get("high_active_min", 0), errors="coerce").fillna(0).sum()
    wm_total = pd.to_numeric(filt.get("work_minutes", 0), errors="coerce").fillna(0).sum()
    ha_ratio = (ha_total / wm_total * 100) if wm_total > 0 else 0

    with k1:
        st.markdown(metric_card("대상 작업자", f"{n_workers:,}명",
                                color=CHART_COLORS["info"]), unsafe_allow_html=True)
    with k2:
        st.markdown(
            metric_card("평균 EWI", f"{avg_ewi:.3f}",
                        delta=f"±{std_ewi:.3f}", delta_up=True,
                        color=CHART_COLORS["ewi"]),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(metric_card("평균 피로도", f"{avg_fat:.3f}",
                                color=CHART_COLORS["fatigue"]), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card("고활성 비율", f"{ha_ratio:.1f}%",
                                color=CHART_COLORS["high_active"]), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 랭킹 테이블 (Top N + 전체 보기) ────────────────────────────
    show_all = st.toggle(
        f"전체 보기 ({n_workers:,}명)",
        value=False,
        key="prod_indiv_show_all",
        help="끄면 Top 50만 표시. 1만명 데이터에서도 즉각 반응합니다.",
    )
    top_n = n_workers if show_all else min(50, n_workers)
    rank_df = filt.head(top_n).copy()

    # 표시용 컬럼 구성 (vectorized)
    rank_df["순위"] = np.arange(1, len(rank_df) + 1)
    rank_df["작업자"] = rank_df["user_name"].astype(str) + " (" + rank_df["user_no"].astype(str) + ")"
    rank_df["체류(분)"] = pd.to_numeric(rank_df["work_minutes"], errors="coerce").fillna(0).round(0).astype(int)
    rank_df["작업공간(분)"] = pd.to_numeric(rank_df.get("work_zone_minutes", 0), errors="coerce").fillna(0).round(0).astype(int)
    rank_df["작업공간 비율"] = (rank_df["work_zone_ratio"] * 100).round(1)
    rank_df["EWI"] = pd.to_numeric(rank_df["ewi"], errors="coerce").round(3)
    rank_df["피로도"] = pd.to_numeric(rank_df.get("fatigue_score", 0), errors="coerce").round(3)
    rank_df["고활성(분)"] = pd.to_numeric(rank_df.get("high_active_min", 0), errors="coerce").fillna(0).round(0).astype(int)
    rank_df["최대연속고활성"] = pd.to_numeric(rank_df.get("max_active_streak", 0), errors="coerce").fillna(0).astype(int)
    rank_df["BLE 커버리지(%)"] = pd.to_numeric(rank_df.get("ble_coverage_pct", 0), errors="coerce").round(1)

    # 배지 (HTML) — dataframe의 column_config 활용 대신 markdown 테이블
    # 대량 데이터에서도 빠르게 렌더링되도록 st.dataframe + column_config 사용
    disp_cols = [
        "순위", "작업자", "company_name", "shift_type",
        "체류(분)", "작업공간(분)", "작업공간 비율",
        "EWI", "피로도", "고활성(분)", "최대연속고활성",
        "BLE 커버리지(%)",
    ]
    disp = rank_df[disp_cols].rename(columns={
        "company_name": "업체",
        "shift_type":   "교대",
    })

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순위":            st.column_config.NumberColumn(format="#%d", width="small"),
            "EWI":             st.column_config.ProgressColumn(
                                   "EWI", min_value=0.0, max_value=1.0, format="%.3f"),
            "피로도":          st.column_config.ProgressColumn(
                                   "피로도", min_value=0.0, max_value=1.0, format="%.3f"),
            "작업공간 비율":   st.column_config.ProgressColumn(
                                   "작업공간 비율", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "BLE 커버리지(%)": st.column_config.ProgressColumn(
                                   "BLE 커버리지(%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
        },
    )

    # ── 경고 배지 요약 (상위 20명만 노출) ──────────────────────────
    badge_preview = rank_df.head(20).copy()
    badge_preview["배지"] = badge_preview.apply(_prod_individual_badges, axis=1)
    has_badges = badge_preview[badge_preview["배지"].str.strip().astype(bool)]
    if not has_badges.empty:
        st.markdown(sub_header("⚠ 경고 대상 (상위 20명 중)"), unsafe_allow_html=True)
        for _, r in has_badges.iterrows():
            st.markdown(
                f"<div style='padding:6px 10px;margin:3px 0;background:{COLORS['bg_chart']};"
                f"border-left:3px solid {CHART_COLORS['critical']};border-radius:6px;'>"
                f"<span style='color:{COLORS['text']};font-weight:600;'>"
                f"#{int(r['순위'])} {r['작업자']}</span> "
                f"<span style='color:{COLORS['text_muted']};font-size:0.82rem;'>· {r.get('company_name','')}</span>"
                f"&nbsp;&nbsp;{r['배지']}</div>",
                unsafe_allow_html=True,
            )

    # ── 상세 확장 ──────────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.expander("🔍 개인 상세 분석 (작업자 선택)", expanded=False):
        pool = filt.head(200).copy()  # 상세 대상 pool 제한 — UI 응답성 우선
        labels = [
            f"#{i+1}  {row['user_name']} ({row['user_no']}) · {row.get('company_name','')} · "
            f"EWI {_nan_safe_float(row.get('ewi')):.3f}"
            for i, (_, row) in enumerate(pool.iterrows())
        ]
        label_to_uno = dict(zip(labels, pool["user_no"].tolist()))
        if not labels:
            st.info("상세 대상이 없습니다.")
            return

        sel_label = st.selectbox(
            f"작업자 선택 (상위 {len(labels)}명 중)",
            labels,
            key="prod_indiv_detail_select",
        )
        sel_uno = label_to_uno[sel_label]
        row = pool[pool["user_no"] == sel_uno].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(metric_card("EWI", f"{_nan_safe_float(row.get('ewi')):.3f}",
                                    color=CHART_COLORS["ewi"]), unsafe_allow_html=True)
        with d2:
            st.markdown(metric_card("CRE", f"{_nan_safe_float(row.get('cre')):.3f}",
                                    color=CHART_COLORS["cre"]), unsafe_allow_html=True)
        with d3:
            st.markdown(metric_card("피로도", f"{_nan_safe_float(row.get('fatigue_score')):.3f}",
                                    color=CHART_COLORS["fatigue"]), unsafe_allow_html=True)
        with d4:
            st.markdown(metric_card("연속 고활성 최대",
                                    f"{int(_nan_safe_float(row.get('max_active_streak'))):,} 분",
                                    color=CHART_COLORS["high_active"]), unsafe_allow_html=True)

        # 시간 분류 스택 바 (고/대기/저/휴게/이동/음영)
        ha = _nan_safe_float(row.get("high_active_min"))
        sb = _nan_safe_float(row.get("standby_min"))
        la = _nan_safe_float(row.get("low_active_min"))
        rs = _nan_safe_float(row.get("rest_min"))
        tr = _nan_safe_float(row.get("transit_min"))
        gm = _nan_safe_float(row.get("gap_min"))

        cat_names  = ["고활성", "대기", "저활성", "휴게", "이동", "음영"]
        cat_values = [ha, sb, la, rs, tr, gm]
        cat_colors = [
            CHART_COLORS["high_active"], CHART_COLORS["standby"],
            CHART_COLORS["low_active"],  CHART_COLORS["rest"],
            CHART_COLORS["transit"],     CHART_COLORS["gap"],
        ]
        # 각 카테고리를 별도 trace로 쪼개어 범례 표시 (stack)
        fig_stk = go.Figure()
        for lbl, v, c in zip(cat_names, cat_values, cat_colors):
            fig_stk.add_trace(go.Bar(
                name=lbl,
                x=[v], y=["시간 구성"],
                orientation="h",
                marker_color=c,
                text=f"{v:.0f}분" if v > 0 else "",
                textposition="inside",
                textfont=dict(color=COLORS["text"], size=11),
                hovertemplate=f"{lbl}: %{{x:.0f}}분<extra></extra>",
            ))
        fig_stk.update_layout(
            **{**PLOTLY_DARK, "margin": dict(l=40, r=20, t=30, b=20)},
            title="활동 시간 분해 (단위: 분)",
            barmode="stack",
            height=180,
            xaxis=dict(title="분", gridcolor=COLORS["grid"]),
            yaxis=dict(showgrid=False, showticklabels=False),
            legend={**PLOTLY_LEGEND, "orientation": "h",
                    "yanchor": "bottom", "y": -0.4,
                    "xanchor": "center", "x": 0.5},
        )
        st.plotly_chart(fig_stk, use_container_width=True)

        # 보조 정보 테이블
        helmet_raw = row.get("helmet_abandoned")
        try:
            helmet_flag = False if helmet_raw is None or pd.isna(helmet_raw) else bool(helmet_raw)
        except (TypeError, ValueError):
            helmet_flag = bool(helmet_raw) if helmet_raw is not None else False
        info_rows = [
            ("업체",          str(row.get("company_name") or "-")),
            ("교대",          str(row.get("shift_type") or "-")),
            ("총 체류시간",   f"{_nan_safe_float(row.get('work_minutes')):.0f} 분"),
            ("작업공간 비율", f"{(_nan_safe_float(row.get('work_zone_ratio')) * 100):.1f}%"),
            ("BLE 커버리지",  f"{_nan_safe_float(row.get('ble_coverage_pct')):.1f}% "
                              f"({str(row.get('ble_coverage') or '-')})"),
            ("alone_ratio",   f"{_nan_safe_float(row.get('alone_ratio')):.3f}"),
            ("정적 위험",     f"{_nan_safe_float(row.get('static_risk')):.3f}"),
            ("헬멧 방치",     "⚠ 플래그" if helmet_flag else "—"),
        ]
        info_df = pd.DataFrame(info_rows, columns=["항목", "값"])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

        # ★ v3.0.1: EWI 해석 가이드 — 음영지역이 많을 때만 노출
        cov_pct = _nan_safe_float(row.get("ble_coverage_pct"))
        if cov_pct < 70:
            _severity = (
                ("🛑", CHART_COLORS["critical"], "심각")  if cov_pct < 30 else
                ("📉", CHART_COLORS["medium"],   "많음")   if cov_pct < 50 else
                ("📡", CHART_COLORS["info"],     "보통")
            )
            ic, col, lbl = _severity
            st.markdown(
                f"<div style='background:{col}15;border-left:3px solid {col};"
                f"padding:10px 14px;border-radius:4px;margin-top:10px;font-size:0.86rem;"
                f"color:{COLORS['text']};'>"
                f"<b style='color:{col};'>{ic} 음영 지역 {lbl} ({cov_pct:.0f}% 커버리지)</b><br>"
                f"EWI는 <b>공식 체류시간({_nan_safe_float(row.get('work_minutes')):.0f}분)</b> 기준으로 계산됩니다. "
                f"이 작업자는 BLE 신호가 잡힌 구간이 {cov_pct:.0f}%뿐이라, "
                f"실제 생산성이 지표보다 높을 가능성이 있습니다. "
                f"음영 지역을 줄이는 방향으로 현장 수신 체계 점검을 권장합니다."
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# 메인 진입점
# ═══════════════════════════════════════════════════════════════════════

def render_productivity_tab(sector_id: str) -> None:
    """생산성 분석 탭 메인."""
    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    dates_asc = list(processed)  # 오름차순

    # 공통 상단: 날짜 + 교대 필터 (일별 탭용)
    col_date, col_shift = st.columns([1, 2])
    with col_date:
        date_str = get_date_selector(
            list(reversed(processed)),
            key="prod_daily_date",
            default_index=0,
            label="일별 분석 날짜",
            show_label=True,
        ) or processed[-1]

    with col_shift:
        shift_options = {"전체": None, "주간": "day",
                         "야간": "night", "연장야간": "extended_night"}
        shift_label = st.radio(
            "교대 필터 (일별 탭 전용)",
            list(shift_options.keys()),
            horizontal=True,
            key="prod_shift_filter",
        )
        shift_filter = shift_options[shift_label]

    st.divider()

    t1, t2, t3, t4, t5 = st.tabs([
        "📊 일별",
        "🏗️ 업체별",
        "🗺️ 공간별",
        "📅 기간별",
        "👤 개인별",
    ])

    with t1:
        _render_daily_subtab(sector_id, date_str, shift_filter)

    with t2:
        _render_company_subtab(sector_id, dates_asc)

    with t3:
        _render_space_subtab(sector_id, dates_asc)

    with t4:
        _render_period_subtab(sector_id)

    with t5:
        _render_individual_productivity_subtab(sector_id, date_str, shift_filter)
