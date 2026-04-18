"""
Congestion Tab — 공간 혼잡도 분석 탭
======================================
건설현장 공간별 시간대별 혼잡도를 시각화.

구성:
  📊 개요     : 혼잡도 KPI + 피크 공간/시간
  🗺️ 히트맵  : 공간 × 시간대 혼잡도 히트맵
  📈 추이     : 시간대별 혼잡도 곡선 (공간별)
  📅 기간분석 : 날짜별/요일별 혼잡도 패턴
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

import config as cfg
from config import DEFAULT_CAPACITY_BY_TYPE, CONGESTION_GRADE_THRESHOLDS
from src.dashboard.styles import (
    metric_card, section_header, PLOTLY_DARK, PLOTLY_LEGEND,
)
from src.pipeline.cache_manager import detect_processed_dates, _date_dir, load_journey
from src.spatial.loader import load_locus_dict
from src.dashboard.components import DWELL_CATEGORY_STYLES
from src.spatial.locus_group import build_locus_group_map, get_group
from src.utils.weather import date_label
from src.dashboard.ai_analysis import render_congestion_ai

# 전체 Locus 수 (음영지역 비율 계산용)
_TOTAL_LOCUS_COUNT = 213

logger = logging.getLogger(__name__)

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND

# 혼잡도 등급 색상
_CONGESTION_COLORS = {
    "여유": "#00C897",
    "보통": "#FFB300",
    "혼잡": "#FF6B35",
    "과밀": "#FF4C4C",
}

# dwell_category 스타일 (components.py에서 import)
_DWELL_CATEGORY_STYLES = DWELL_CATEGORY_STYLES


def _congestion_grade(count: int) -> str:
    """[DEPRECATED] 절대 인원수 기준 등급 — _congestion_grade_by_max() 사용 권장."""
    import warnings
    warnings.warn(
        "_congestion_grade()는 deprecated. capacity 대비 비율 기반 _congestion_grade_by_max() 사용.",
        DeprecationWarning,
        stacklevel=2,
    )
    if count >= 100:
        return "과밀"
    if count >= 50:
        return "혼잡"
    if count >= 20:
        return "보통"
    return "여유"


def _congestion_grade_by_max(current: int, max_occ: int) -> tuple[str, str]:
    """
    capacity 대비 혼잡률 기반 등급.

    임계값은 config.CONGESTION_GRADE_THRESHOLDS에서 관리:
      과밀 >= 100%, 혼잡 >= 80%, 보통 >= 60%, 여유 < 60%

    Args:
        current: 현재 인원
        max_occ: 최대 수용 인원 (capacity 또는 max_concurrent_occupancy)

    Returns:
        (등급 문자열, 색상 코드)
    """
    if max_occ <= 0:
        return "여유", _CONGESTION_COLORS["여유"]

    ratio = current / max_occ
    thresholds = CONGESTION_GRADE_THRESHOLDS
    if ratio >= thresholds["과밀"]:
        return "과밀", _CONGESTION_COLORS["과밀"]
    elif ratio >= thresholds["혼잡"]:
        return "혼잡", _CONGESTION_COLORS["혼잡"]
    elif ratio >= thresholds["보통"]:
        return "보통", _CONGESTION_COLORS["보통"]
    else:
        return "여유", _CONGESTION_COLORS["여유"]


def render_congestion_tab(sector_id: str | None = None):
    """혼잡도 분석 탭 진입점."""
    sid = sector_id or cfg.SECTOR_ID
    processed = detect_processed_dates(sid)

    if not processed:
        st.info("처리된 데이터가 없습니다. [파이프라인] 탭에서 먼저 전처리를 실행하세요.")
        return

    locus_dict = load_locus_dict(sid)

    st.markdown(section_header("🏢 공간 혼잡도 분석"), unsafe_allow_html=True)
    st.caption(
        "건설현장 구역별 동시 체류 인원 분석 — "
        "시간대별 혼잡도 변화, 피크 공간/시간 탐지, 요일별 패턴"
    )

    # ── 분석 모드 선택 ───────────────────────────────────────────
    mode = st.radio(
        "분석 범위",
        ["📊 일별 분석", "📅 기간 분석"],
        horizontal=True,
        key="congestion_mode",
    )

    if mode == "📊 일별 분석":
        _render_single_day(sid, processed, locus_dict)
    else:
        _render_multi_day(sid, processed, locus_dict)


# ══════════════════════════════════════════════════════════════════
# 일별 혼잡도 분석
# ══════════════════════════════════════════════════════════════════

def _render_single_day(sid: str, processed: list[str], locus_dict: dict):
    """단일 날짜 혼잡도 분석."""
    date_options = {date_label(d): d for d in reversed(processed)}
    selected_label = st.selectbox("분석 날짜", list(date_options.keys()), key="cong_date")
    date_str = date_options[selected_label]

    # journey 로드 (★ 캐시 활용 — 2회차부터 즉시 로드)
    # 혼잡도 분석 + 예측에 필요한 컬럼만 로드 (메모리 절감)
    # Cloud 환경: journey.parquet 없으면 journey_slim.parquet 자동 fallback
    _CONGESTION_COLS = ["timestamp", "user_no", "locus_id", "locus_token", "is_work_hour"]

    from src.pipeline.cache_manager import is_journey_slim
    import config as _cfg
    from pathlib import Path as _Path

    # journey.parquet 없고 slim도 없으면 → Drive 온디맨드 다운로드 시도
    _slim_local = (_cfg.PROCESSED_DIR / sid / date_str / "journey_slim.parquet")
    _full_local = (_cfg.PROCESSED_DIR / sid / date_str / "journey.parquet")
    if not _full_local.exists() and not _slim_local.exists() and _cfg.CLOUD_MODE:
        with st.spinner(
            f"☁️ 혼잡도 데이터 다운로드 중... ({date_str} · 약 10~15초 소요)"
        ):
            try:
                from src.pipeline.drive_storage import init_drive_storage_from_secrets
                _drive = init_drive_storage_from_secrets(sid)
                if _drive:
                    _drive.ensure_journey_slim(sid, date_str, _cfg.PROCESSED_DIR)
            except Exception as _e:
                logger.warning(f"journey_slim 온디맨드 다운로드 실패: {_e}")

    with st.spinner("혼잡도 분석 중..."):
        journey_full = load_journey(date_str, sid, columns=_CONGESTION_COLS)

    if journey_full.empty:
        st.warning(
            "📦 혼잡도 데이터가 없습니다. "
            "Drive에 journey_slim.parquet 업로드 후 이용 가능합니다."
        )
        return

    # Cloud slim 안내
    if is_journey_slim(journey_full):
        st.info(
            "☁️ **Cloud 환경** — 슬림 데이터(6컬럼) 기준으로 혼잡도를 분석합니다. "
            "첫 날짜 접근 시 **약 10~15초** 소요되며, "
            "이후 같은 날짜는 캐시에서 즉시 로드됩니다."
        )

    with st.spinner("혼잡도 데이터 분석 중..."):
        cols = [c for c in ["timestamp", "user_no", "locus_id", "locus_token"] if c in journey_full.columns]
        journey_df = journey_full[cols]
        from src.pipeline.congestion import (
            compute_congestion, compute_hourly_profile,
            compute_congestion_summary, compute_space_ranking,
        )
        congestion_df = compute_congestion(journey_df, time_bin_minutes=30, locus_dict=locus_dict)
        hourly_df = compute_hourly_profile(journey_df, locus_dict)
        summary = compute_congestion_summary(congestion_df)
        ranking_df = compute_space_ranking(congestion_df, top_n=15)

    if congestion_df.empty:
        st.warning("혼잡도 데이터가 없습니다.")
        return

    # ── 음영지역 비율 계산 ──────────────────────────────────────
    active_spaces = summary["total_spaces"]
    blind_count = max(0, _TOTAL_LOCUS_COUNT - active_spaces)
    blind_pct = blind_count / _TOTAL_LOCUS_COUNT * 100 if _TOTAL_LOCUS_COUNT > 0 else 0

    # ── KPI 카드 ──────────────────────────────────────────────────
    st.divider()
    cols = st.columns(7)
    kpis = [
        ("피크 공간", summary["peak_space"], None),
        ("최대 동시 인원", f"{summary['peak_count']}명", "#FF4C4C" if summary["peak_count"] >= 100 else "#FFB300"),
        ("피크 시간", summary["peak_time"], None),
        ("최붐비는 시간", f"{summary['busiest_hour']}시", "#FF6B35"),
        ("가장 한산한 시간", f"{summary['quietest_hour']}시", "#00C897"),
        ("감지 공간", f"{active_spaces}개", None),
        ("음영지역", f"{blind_count}개 ({blind_pct:.0f}%)",
         "#FF4C4C" if blind_pct >= 50 else ("#FFB300" if blind_pct >= 30 else "#00C897")),
    ]
    for col, (label, val, color) in zip(cols, kpis):
        with col:
            if color:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
                    f"{val}</div><div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(metric_card(label, str(val)), unsafe_allow_html=True)

    st.divider()

    # ── 탭 구성 ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🗺️ 히트맵", "📈 시간대 추이", "🏆 공간 랭킹", "🔮 예측"])

    with t1:
        _render_heatmap(hourly_df)

    with t2:
        _render_time_series(congestion_df)

    with t3:
        _render_space_ranking(ranking_df, locus_dict)

    with t4:
        _render_prediction_layer(sid, date_str, journey_full, congestion_df, locus_dict)

    # ── AI 분석 섹션 ─────────────────────────────────────────────
    # 혼잡도 차트 아래, journey 데이터 함께 전달
    st.divider()
    with st.expander("Claude AI 분석", expanded=False):
        _space_df_for_ai = congestion_df  # 공간별 혼잡도 집계
        _journey_cols = [c for c in ["timestamp", "user_no", "locus_id"] if c in journey_full.columns]
        _journey_for_ai = journey_full[_journey_cols] if _journey_cols else None
        render_congestion_ai(
            space_df=_space_df_for_ai,
            journey_df=_journey_for_ai,
            date_str=date_str,
            sector_id=sid,
        )


def _log_scale_colorbar_ticks(z_max: float) -> tuple[list, list]:
    """log1p 스케일용 colorbar tick vals/text 생성.

    원본 값이 1, 5, 10, 50, 100, 500, 1000 등으로 표시되도록
    해당 값의 log1p 변환값을 tickvals로 설정.
    """
    candidates = [0, 1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    vals_raw = [v for v in candidates if v <= z_max]
    if vals_raw[-1] != int(z_max) and z_max > vals_raw[-1] * 1.2:
        vals_raw.append(int(round(z_max)))
    return [float(np.log1p(v)) for v in vals_raw], [str(v) for v in vals_raw]


# ★ 비선형 (log1p) 팔레트 — 낮은 값에서도 색상이 잘 구분되도록
# 검정(0) → 짙은 파랑(작음) → 청록(중간) → 노랑/주황(높음) → 빨강(최대)
# 기존 선형 팔레트는 분포가 skewed한 경우 대부분이 검정에 몰림
_HEATMAP_COLORSCALE_LOG = [
    [0.00, "#0D1B2A"],   # 0 → 배경색
    [0.08, "#1A3A5C"],   # 매우 낮음
    [0.20, "#2E5F8C"],
    [0.35, "#00AEEF"],   # 중간 하한
    [0.55, "#00C897"],   # 녹색 구간 추가
    [0.72, "#FFB300"],   # 경고
    [0.88, "#FF6B35"],   # 혼잡
    [1.00, "#FF4C4C"],   # 과밀
]


def _render_heatmap(hourly_df: pd.DataFrame):
    """공간 × 시간대 혼잡도 히트맵 (log1p 비선형 색상 매핑)."""
    if hourly_df.empty:
        st.info("히트맵 데이터 없음")
        return

    st.markdown("#### 🗺️ 공간 × 시간대 평균 혼잡도")
    st.caption(
        "색이 진할수록 해당 시간대에 해당 공간의 평균 체류 인원이 많음. "
        "★ 특정 공간(타각기 등)에 인원이 편중되어 대부분 공간이 어둡게 보이는 현상을 완화하기 위해 "
        "**로그 스케일(log1p)** 로 색상을 매핑했습니다 — 낮은 인원 구간의 차이도 잘 드러납니다."
    )

    # 작업 구간만 필터 (5~23시)
    df = hourly_df[(hourly_df["hour"] >= 5) & (hourly_df["hour"] <= 23)].copy()

    if df.empty:
        st.info("업무 시간(5~23시) 데이터 없음")
        return

    # 피벗 테이블: 행=공간, 열=시간
    pivot = df.pivot_table(
        index="locus_name",
        columns="hour",
        values="avg_workers",
        aggfunc="mean",
        fill_value=0,
    )

    # 총 체류 인원이 큰 순서로 정렬
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True)
    pivot = pivot.drop(columns=["_total"])

    # 열 라벨 정리
    pivot.columns = [f"{h}시" for h in pivot.columns]

    # ★ 비선형 매핑: z를 log1p 변환 (log(1+x))
    z_raw = pivot.values.astype(float)
    z_log = np.log1p(z_raw)
    z_max_raw = float(np.nanmax(z_raw)) if z_raw.size else 0.0

    # colorbar는 원본 값 기준으로 tick 표시 (사용자 친화적)
    tickvals, ticktext = _log_scale_colorbar_ticks(z_max_raw)

    fig = go.Figure(data=go.Heatmap(
        z=z_log,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        customdata=z_raw,                       # 원본 값 (hover용)
        colorscale=_HEATMAP_COLORSCALE_LOG,
        hovertemplate=(
            "공간: %{y}<br>시간: %{x}<br>"
            "평균 인원: %{customdata:.1f}명<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="평균 인원<br>(로그 스케일)", font=dict(color="#D5E5FF")),
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(color="#D5E5FF"),
        ),
    ))
    fig.update_layout(
        **_DARK,
        height=max(350, len(pivot) * 28 + 100),
        xaxis_title="시간대",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 최대 체류 시간대 히트맵 (동일한 로그 스케일 적용)
    if "max_workers" in hourly_df.columns:
        with st.expander("📊 최대 동시 인원 히트맵 (피크 기준)", expanded=False):
            df_max = hourly_df[(hourly_df["hour"] >= 5) & (hourly_df["hour"] <= 23)].copy()
            pivot_max = df_max.pivot_table(
                index="locus_name", columns="hour", values="max_workers",
                aggfunc="max", fill_value=0,
            )
            pivot_max["_total"] = pivot_max.sum(axis=1)
            pivot_max = pivot_max.sort_values("_total", ascending=True).drop(columns=["_total"])
            pivot_max.columns = [f"{h}시" for h in pivot_max.columns]

            z_raw2 = pivot_max.values.astype(float)
            z_log2 = np.log1p(z_raw2)
            z_max2 = float(np.nanmax(z_raw2)) if z_raw2.size else 0.0
            tickvals2, ticktext2 = _log_scale_colorbar_ticks(z_max2)

            fig2 = go.Figure(data=go.Heatmap(
                z=z_log2,
                x=pivot_max.columns.tolist(),
                y=pivot_max.index.tolist(),
                customdata=z_raw2,
                colorscale=_HEATMAP_COLORSCALE_LOG,
                hovertemplate=(
                    "공간: %{y}<br>시간: %{x}<br>"
                    "최대 인원: %{customdata:.0f}명<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="최대 인원<br>(로그 스케일)",
                               font=dict(color="#D5E5FF")),
                    tickvals=tickvals2,
                    ticktext=ticktext2,
                    tickfont=dict(color="#D5E5FF"),
                ),
            ))
            fig2.update_layout(**_DARK, height=max(350, len(pivot_max) * 28 + 100))
            st.plotly_chart(fig2, use_container_width=True)


def _render_time_series(congestion_df: pd.DataFrame):
    """시간대별 혼잡도 추이 (상위 공간)."""
    if congestion_df.empty:
        st.info("추이 데이터 없음")
        return

    st.markdown("#### 📈 시간대별 혼잡도 추이")

    # 상위 10개 공간 필터
    top_spaces = (
        congestion_df.groupby("locus_name")["worker_count"]
        .max()
        .nlargest(10)
        .index.tolist()
    )

    selected_spaces = st.multiselect(
        "공간 선택 (최대 10개)",
        options=top_spaces,
        default=top_spaces[:5],
        key="cong_space_select",
    )

    if not selected_spaces:
        st.info("공간을 선택하세요.")
        return

    df = congestion_df[congestion_df["locus_name"].isin(selected_spaces)].copy()
    # 작업 시간만
    df = df[(df["hour"] >= 5) & (df["hour"] <= 23)]
    df["time_label"] = df["time_bin"].dt.strftime("%H:%M")

    fig = px.line(
        df,
        x="time_bin",
        y="worker_count",
        color="locus_name",
        title="공간별 동시 체류 인원 추이 (30분 단위)",
        labels={"worker_count": "동시 인원", "time_bin": "시간", "locus_name": "공간"},
        markers=True,
    )
    fig.update_layout(**_DARK, height=420, legend=_LEG)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

    # 전체 현장 총 인원 추이
    total_by_time = congestion_df.groupby("time_bin")["worker_count"].sum().reset_index()
    total_by_time = total_by_time[
        (total_by_time["time_bin"].dt.hour >= 5) & (total_by_time["time_bin"].dt.hour <= 23)
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=total_by_time["time_bin"],
        y=total_by_time["worker_count"],
        fill="tozeroy",
        fillcolor="rgba(0, 174, 239, 0.15)",
        line=dict(color="#00AEEF", width=2),
        name="현장 전체",
        hovertemplate="시간: %{x|%H:%M}<br>전체 인원: %{y}명<extra></extra>",
    ))
    fig2.update_layout(
        title="현장 전체 동시 체류 인원 추이",
        yaxis_title="전체 인원",
        **_DARK, height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_space_ranking(ranking_df: pd.DataFrame, locus_dict: dict | None = None):
    """공간별 혼잡도 랭킹 — max_concurrent_occupancy 대비 혼잡률 포함."""
    if ranking_df.empty:
        st.info("랭킹 데이터 없음")
        return

    st.markdown("#### 🏆 공간별 혼잡도 랭킹")
    st.caption("최대 동시 인원 기준 상위 공간 — 피크 시간대 + 혼잡률 포함")

    # ★ capacity 대비 혼잡률 계산 (enriched locus + DEFAULT_CAPACITY_BY_TYPE fallback)
    ranking_df = ranking_df.copy()
    if locus_dict:
        def _resolve_capacity(lid: str) -> int:
            info = locus_dict.get(lid, {})
            cap = info.get("max_concurrent_occupancy")
            if cap and cap > 0:
                return int(cap)
            ltype = str(info.get("dwell_category", "WORK")).upper()
            return DEFAULT_CAPACITY_BY_TYPE.get(ltype, 50)

        ranking_df["_max_capacity"] = ranking_df["locus_id"].apply(_resolve_capacity)
        ranking_df["_dwell_cat"] = ranking_df["locus_id"].apply(
            lambda lid: locus_dict.get(lid, {}).get("dwell_category", "UNKNOWN")
        )
    else:
        ranking_df["_max_capacity"] = DEFAULT_CAPACITY_BY_TYPE.get("WORK", 100)
        ranking_df["_dwell_cat"] = "UNKNOWN"

    ranking_df["occupancy_pct"] = (
        ranking_df["max_workers"] / ranking_df["_max_capacity"] * 100
    ).clip(0, 150).round(1)

    # 수평 바 차트
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ranking_df["locus_name"],
        x=ranking_df["max_workers"],
        orientation="h",
        name="최대 인원",
        marker_color="#FF6B35",
        text=ranking_df.apply(
            lambda r: f"{int(r['max_workers'])}명 ({r['occupancy_pct']:.0f}%)", axis=1
        ),
        textposition="outside",
        textfont=dict(color="#D5E5FF", size=11),
    ))
    fig.add_trace(go.Bar(
        y=ranking_df["locus_name"],
        x=ranking_df["avg_workers"],
        orientation="h",
        name="평균 인원",
        marker_color="#00AEEF",
        text=ranking_df["avg_workers"].apply(lambda v: f"{v:.1f}명"),
        textposition="outside",
        textfont=dict(color="#D5E5FF", size=10),
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(ranking_df) * 40 + 100),
        legend=_LEG,
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ★ 유형별 혼잡도 비교 차트 추가
    if locus_dict and "_dwell_cat" in ranking_df.columns:
        _render_category_congestion_chart(ranking_df)

    # 상세 테이블
    with st.expander("📋 상세 데이터"):
        display = ranking_df[[
            "locus_name", "_dwell_cat", "max_workers", "_max_capacity",
            "occupancy_pct", "avg_workers", "peak_hour"
        ]].copy()
        display.columns = ["공간", "유형", "최대 인원", "수용 기준", "혼잡률(%)", "평균 인원", "피크 시간"]
        display["피크 시간"] = display["피크 시간"].astype(int).astype(str) + "시"
        # 유형 한글화
        display["유형"] = display["유형"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("label", x)
        )
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_category_congestion_chart(ranking_df: pd.DataFrame):
    """dwell_category별 평균 혼잡률 비교 차트."""
    if "_dwell_cat" not in ranking_df.columns or ranking_df.empty:
        return

    cat_agg = ranking_df.groupby("_dwell_cat").agg(
        avg_occupancy=("occupancy_pct", "mean"),
        count=("locus_id", "count"),
    ).reset_index()

    if cat_agg.empty:
        return

    with st.expander("📊 공간 유형별 평균 혼잡률", expanded=False):
        cat_agg["label"] = cat_agg["_dwell_cat"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("label", x)
        )
        cat_agg["color"] = cat_agg["_dwell_cat"].apply(
            lambda x: _DWELL_CATEGORY_STYLES.get(x, {}).get("color", "#6B7280")
        )

        fig = go.Figure(go.Bar(
            x=cat_agg["label"],
            y=cat_agg["avg_occupancy"],
            marker_color=cat_agg["color"].tolist(),
            text=cat_agg.apply(lambda r: f"{r['avg_occupancy']:.1f}% ({r['count']}개)", axis=1),
            textposition="outside",
            textfont=dict(color="#D5E5FF"),
        ))
        fig.update_layout(
            yaxis_title="평균 혼잡률 (%)",
            **_DARK,
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 예측 레이어 (Deep Space 연동)
# ══════════════════════════════════════════════════════════════════


def _render_prediction_layer(
    sid: str,
    date_str: str,
    journey_full: pd.DataFrame,
    congestion_df: pd.DataFrame,
    locus_dict: dict | None,
):
    """Deep Space 기반 혼잡도 예측 — 시간 cutoff 기준 미래 예측."""
    st.markdown("#### 🔮 Deep Space 혼잡도 예측")
    st.caption(
        "선택한 시각까지의 이동 시퀀스를 기반으로 다음 위치를 예측 → "
        "공간별 예상 인원을 실측과 비교"
    )

    # ── 모델 로드 ─────────────────────────────────────────────
    try:
        from src.dashboard.deep_space.model_loader import load_model
        model, tokenizer = load_model(sid)
    except Exception:
        model, tokenizer = None, None

    if model is None:
        st.info("Deep Space 모델이 없습니다. [Deep Space] 탭에서 모델을 학습하세요.")
        return

    # ── 시간 cutoff 선택 ──────────────────────────────────────
    cutoff_hour = st.slider(
        "기준 시각 (이 시각까지의 데이터로 예측)",
        min_value=7, max_value=17, value=12, step=1,
        format="%d시",
        key="cong_pred_cutoff",
    )

    # ── 시퀀스 추출 (cutoff 시각까지) ──────────────────────────
    df = journey_full.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 작업시간 필터
    if "is_work_hour" in df.columns:
        df = df[df["is_work_hour"]].copy()

    locus_col = "locus_id" if "locus_id" in df.columns else "locus_token"
    df = df.dropna(subset=[locus_col])

    # cutoff 이전 데이터만
    df_before = df[df["timestamp"].dt.hour < cutoff_hour].copy()
    # cutoff 이후 실측 (검증용)
    df_after = df[
        (df["timestamp"].dt.hour >= cutoff_hour)
        & (df["timestamp"].dt.hour < cutoff_hour + 2)
    ].copy()

    if df_before.empty:
        st.warning(f"{cutoff_hour}시 이전 데이터가 없습니다.")
        return

    # 작업자별 시퀀스 추출
    df_before = df_before.sort_values(["user_no", "timestamp"])
    sequences = {}
    current_loci = {}
    for user_no, grp in df_before.groupby("user_no"):
        loci = grp[locus_col].dropna().tolist()
        # 연속 중복 제거
        deduped = []
        for loc in loci:
            if not deduped or deduped[-1] != str(loc):
                deduped.append(str(loc))
        if len(deduped) >= 3:
            sequences[str(user_no)] = deduped
            current_loci[str(user_no)] = deduped[-1]

    if not sequences:
        st.warning("예측 가능한 시퀀스가 부족합니다.")
        return

    # ── 배치 예측 ─────────────────────────────────────────────
    with st.spinner(f"Deep Space 예측 중... ({len(sequences)}명)"):
        from src.dashboard.deep_space.helpers import predict_next_batch
        from collections import Counter

        worker_ids = list(sequences.keys())
        seqs = [sequences[wid] for wid in worker_ids]

        try:
            preds = predict_next_batch(model, tokenizer, seqs, top_k=1, use_cache=False)
        except Exception as e:
            st.error(f"예측 실패: {e}")
            return

        # 현재 위치별 인원 집계
        current_counts = Counter(current_loci.values())

        # 예측 위치별 인원 집계
        predicted_counts = Counter()
        for wid, pred_list in zip(worker_ids, preds):
            if pred_list:
                predicted_counts[pred_list[0][0]] += 1
            else:
                # 예측 불가 → 현재 위치 유지
                predicted_counts[current_loci.get(wid, "")] += 1

        # 실측 (cutoff 이후 2시간) 위치별 인원
        actual_counts = Counter()
        if not df_after.empty:
            # 각 작업자의 cutoff 이후 첫 번째 위치
            df_after_sorted = df_after.sort_values(["user_no", "timestamp"])
            first_after = df_after_sorted.groupby("user_no")[locus_col].first()
            actual_counts = Counter(first_after.values)

    # ── 비교 테이블 생성 ──────────────────────────────────────
    all_loci = sorted(set(current_counts.keys()) | set(predicted_counts.keys()) | set(actual_counts.keys()))

    comparison_rows = []
    for lid in all_loci:
        cur = current_counts.get(lid, 0)
        pred = predicted_counts.get(lid, 0)
        act = actual_counts.get(lid, 0)
        lname = locus_dict.get(lid, {}).get("locus_name", lid) if locus_dict else lid
        diff = pred - cur
        comparison_rows.append({
            "locus_id": lid,
            "공간": lname,
            f"현재 ({cutoff_hour}시 기준)": cur,
            "예측 (다음 위치)": pred,
            "변화": diff,
            f"실측 ({cutoff_hour}~{cutoff_hour+2}시)": act if actual_counts else None,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values("예측 (다음 위치)", ascending=False).reset_index(drop=True)

    # ── KPI 카드 ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("예측 대상 인원", f"{len(sequences)}명"), unsafe_allow_html=True)
    with c2:
        increasing = sum(1 for r in comparison_rows if r["변화"] > 0)
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:#FF6B35'>"
            f"{increasing}개</div><div class='metric-label'>인원 증가 예상 공간</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        top_pred = comp_df.iloc[0] if not comp_df.empty else None
        top_name = top_pred["공간"] if top_pred is not None else "—"
        top_count = top_pred["예측 (다음 위치)"] if top_pred is not None else 0
        st.markdown(metric_card("예측 최다 공간", f"{top_name} ({top_count}명)"), unsafe_allow_html=True)
    with c4:
        if actual_counts:
            # 예측 정확도: 예측 인원과 실측 인원의 MAE
            errors = []
            for lid in all_loci:
                errors.append(abs(predicted_counts.get(lid, 0) - actual_counts.get(lid, 0)))
            mae = sum(errors) / len(errors) if errors else 0
            color = "#00C897" if mae < 3 else ("#FFB300" if mae < 5 else "#FF4C4C")
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
                f"{mae:.1f}명</div><div class='metric-label'>예측 오차 (MAE)</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(metric_card("실측 비교", "데이터 없음"), unsafe_allow_html=True)

    # ── 비교 차트: 현재 vs 예측 (vs 실측) ──────────────────────
    st.markdown(f"##### 공간별 인원 비교 — {cutoff_hour}시 기준")

    top_n = min(15, len(comp_df))
    chart_df = comp_df.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["공간"], y=chart_df[f"현재 ({cutoff_hour}시 기준)"],
        name=f"현재 ({cutoff_hour}시)", marker_color="#4A90D9",
        hovertemplate="공간: %{x}<br>현재: %{y}명<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=chart_df["공간"], y=chart_df["예측 (다음 위치)"],
        name="Deep Space 예측", marker_color="#FF6B35",
        hovertemplate="공간: %{x}<br>예측: %{y}명<extra></extra>",
    ))
    if actual_counts:
        actual_col = f"실측 ({cutoff_hour}~{cutoff_hour+2}시)"
        fig.add_trace(go.Bar(
            x=chart_df["공간"], y=chart_df[actual_col],
            name=f"실측 ({cutoff_hour}~{cutoff_hour+2}시)", marker_color="#00C897",
            hovertemplate="공간: %{x}<br>실측: %{y}명<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="공간", yaxis_title="인원 수",
        legend=dict(font=dict(color="#C8D6E8"), orientation="h", y=-0.2),
        height=450,
        **_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 인원 변화 예측 (증가/감소 Top-10) ─────────────────────
    change_df = comp_df[comp_df["변화"] != 0].copy()
    if not change_df.empty:
        with st.expander("📊 인원 변화 예측 (증가/감소 공간)", expanded=True):
            inc_df = change_df[change_df["변화"] > 0].nlargest(10, "변화")
            dec_df = change_df[change_df["변화"] < 0].nsmallest(10, "변화")

            c_l, c_r = st.columns(2)
            with c_l:
                st.markdown("**인원 증가 예상**")
                if not inc_df.empty:
                    fig_inc = go.Figure(go.Bar(
                        y=inc_df["공간"], x=inc_df["변화"],
                        orientation="h", marker_color="#FF6B35",
                        text=inc_df["변화"].apply(lambda v: f"+{v}명"),
                        textposition="outside",
                        textfont=dict(color="#FF6B35", size=11),
                    ))
                    fig_inc.update_layout(
                        height=max(200, len(inc_df) * 30 + 60),
                        yaxis=dict(autorange="reversed"),
                        **_DARK,
                    )
                    st.plotly_chart(fig_inc, use_container_width=True)
                else:
                    st.info("증가 예상 공간 없음")

            with c_r:
                st.markdown("**인원 감소 예상**")
                if not dec_df.empty:
                    fig_dec = go.Figure(go.Bar(
                        y=dec_df["공간"], x=dec_df["변화"].abs(),
                        orientation="h", marker_color="#00AEEF",
                        text=dec_df["변화"].apply(lambda v: f"{v}명"),
                        textposition="outside",
                        textfont=dict(color="#00AEEF", size=11),
                    ))
                    fig_dec.update_layout(
                        height=max(200, len(dec_df) * 30 + 60),
                        yaxis=dict(autorange="reversed"),
                        **_DARK,
                    )
                    st.plotly_chart(fig_dec, use_container_width=True)
                else:
                    st.info("감소 예상 공간 없음")

    # ── 건물_층 그룹 단위 비교 ───────────────────────────────────
    _render_group_prediction(sid, current_counts, predicted_counts, actual_counts, cutoff_hour)

    # ── 음영지역 상세 ──────────────────────────────────────────
    with st.expander("🌑 음영지역 상세 (BLE 신호 미감지 공간)", expanded=False):
        _render_blind_spot_detail(congestion_df, locus_dict)

    # ── 상세 데이터 테이블 ────────────────────────────────────
    with st.expander("📋 전체 비교 데이터"):
        display_cols = ["공간", f"현재 ({cutoff_hour}시 기준)", "예측 (다음 위치)", "변화"]
        if actual_counts:
            display_cols.append(f"실측 ({cutoff_hour}~{cutoff_hour+2}시)")
        st.dataframe(comp_df[display_cols], use_container_width=True, hide_index=True)


def _render_group_prediction(
    sector_id: str,
    current_counts: dict,
    predicted_counts: dict,
    actual_counts: dict,
    cutoff_hour: int,
):
    """건물_층 그룹 단위로 집계하여 비교 — 개별 locus 대비 정확도 대폭 향상."""
    group_map = build_locus_group_map(sector_id)
    if not group_map:
        return

    from collections import Counter

    def _aggregate_to_groups(counts: dict) -> Counter:
        grp_counts = Counter()
        for lid, cnt in counts.items():
            grp_counts[get_group(str(lid), group_map)] += cnt
        return grp_counts

    cur_grp = _aggregate_to_groups(current_counts)
    pred_grp = _aggregate_to_groups(predicted_counts)
    act_grp = _aggregate_to_groups(actual_counts) if actual_counts else Counter()

    all_groups = sorted(set(cur_grp) | set(pred_grp) | set(act_grp))
    if not all_groups:
        return

    with st.expander("🏢 건물_층 그룹 단위 비교 (213개 → ~30개 그룹)", expanded=True):
        st.caption(
            "개별 공간(213개) 예측은 난이도가 높지만, "
            "건물_층 그룹(~30개)으로 집계하면 현장 의사결정에 실질적인 정확도를 제공합니다."
        )

        rows = []
        for grp in all_groups:
            c = cur_grp.get(grp, 0)
            p = pred_grp.get(grp, 0)
            a = act_grp.get(grp, 0)
            rows.append({
                "그룹": grp,
                f"현재 ({cutoff_hour}시)": c,
                "예측": p,
                "변화": p - c,
            })
            if actual_counts:
                rows[-1][f"실측 ({cutoff_hour}~{cutoff_hour+2}시)"] = a

        grp_df = pd.DataFrame(rows).sort_values("예측", ascending=False).reset_index(drop=True)

        # 그룹 MAE
        if actual_counts:
            errors = [abs(pred_grp.get(g, 0) - act_grp.get(g, 0)) for g in all_groups]
            grp_mae = sum(errors) / len(errors) if errors else 0
            col_color = "#00C897" if grp_mae < 5 else ("#FFB300" if grp_mae < 10 else "#FF4C4C")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    metric_card("그룹 수", f"{len(all_groups)}개"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{col_color}'>"
                    f"{grp_mae:.1f}명</div><div class='metric-label'>그룹 예측 오차 (MAE)</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                # 방향 정확도: 증가/감소 방향이 맞는 비율
                direction_correct = sum(
                    1 for g in all_groups
                    if (pred_grp.get(g, 0) - cur_grp.get(g, 0)) * (act_grp.get(g, 0) - cur_grp.get(g, 0)) > 0
                )
                dir_pct = direction_correct / len(all_groups) * 100 if all_groups else 0
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:#00AEEF'>"
                    f"{dir_pct:.0f}%</div><div class='metric-label'>방향 예측 정확도</div></div>",
                    unsafe_allow_html=True,
                )

        # 그룹 비교 차트
        top_n = min(20, len(grp_df))
        chart_grp = grp_df.head(top_n)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_grp["그룹"], y=chart_grp[f"현재 ({cutoff_hour}시)"],
            name=f"현재 ({cutoff_hour}시)", marker_color="#4A90D9",
        ))
        fig.add_trace(go.Bar(
            x=chart_grp["그룹"], y=chart_grp["예측"],
            name="Deep Space 예측", marker_color="#FF6B35",
        ))
        if actual_counts:
            act_col = f"실측 ({cutoff_hour}~{cutoff_hour+2}시)"
            fig.add_trace(go.Bar(
                x=chart_grp["그룹"], y=chart_grp[act_col],
                name=f"실측", marker_color="#00C897",
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="건물_층 그룹", yaxis_title="인원 수",
            legend=dict(font=dict(color="#C8D6E8"), orientation="h", y=-0.25),
            height=400,
            **_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(grp_df, use_container_width=True, hide_index=True)


def _render_blind_spot_detail(congestion_df: pd.DataFrame, locus_dict: dict | None):
    """음영지역 상세 — 전체 Locus 중 BLE 신호가 감지되지 않은 공간."""
    try:
        locus_v2_path = (
            __import__("config").SPATIAL_DIR
            / __import__("config").SECTOR_ID
            / "locus" / "locus_v2.csv"
        )
        all_locus_df = pd.read_csv(locus_v2_path)
    except Exception:
        st.info("locus_v2.csv를 로드할 수 없습니다.")
        return

    all_locus_ids = set(all_locus_df["locus_id"].astype(str))
    active_locus_ids = set(congestion_df["locus_id"].astype(str)) if not congestion_df.empty else set()
    blind_ids = sorted(all_locus_ids - active_locus_ids)

    if not blind_ids:
        st.success(f"전체 {len(all_locus_ids)}개 공간에서 BLE 신호 감지됨 — 음영지역 없음")
        return

    # 음영지역 분류
    blind_info = all_locus_df[all_locus_df["locus_id"].isin(blind_ids)].copy()
    blind_pct = len(blind_ids) / len(all_locus_ids) * 100

    # 유형별 음영지역
    type_counts = blind_info["locus_type"].value_counts()

    c1, c2, c3 = st.columns(3)
    with c1:
        color = "#FF4C4C" if blind_pct >= 50 else ("#FFB300" if blind_pct >= 30 else "#00C897")
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
            f"{blind_pct:.1f}%</div><div class='metric-label'>"
            f"음영 비율 ({len(blind_ids)}/{len(all_locus_ids)})</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        work_blind = type_counts.get("WORK_AREA", 0)
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:#FF6B35'>"
            f"{work_blind}개</div><div class='metric-label'>작업공간 음영</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        transit_blind = type_counts.get("TRANSIT", 0) + type_counts.get("GATE", 0)
        st.markdown(
            f"<div class='metric-card'><div class='metric-value' style='color:#00AEEF'>"
            f"{transit_blind}개</div><div class='metric-label'>이동/출입구 음영</div></div>",
            unsafe_allow_html=True,
        )

    # 음영지역 유형별 차트
    if not type_counts.empty:
        fig = go.Figure(go.Bar(
            x=type_counts.index.tolist(),
            y=type_counts.values.tolist(),
            marker_color=["#FF4C4C" if t == "WORK_AREA" else "#FFB300" if t == "REST_AREA"
                          else "#00AEEF" for t in type_counts.index],
            text=type_counts.values.tolist(),
            textposition="outside",
            textfont=dict(color="#D5E5FF"),
        ))
        fig.update_layout(
            title="음영지역 유형별 분포",
            yaxis_title="공간 수",
            height=280,
            **_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 음영지역 목록
    display = blind_info[["locus_id", "locus_name", "locus_type", "building", "floor"]].copy()
    display.columns = ["Locus ID", "공간명", "유형", "건물", "층"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=300)


# ══════════════════════════════════════════════════════════════════
# 기간 혼잡도 분석
# ══════════════════════════════════════════════════════════════════

def _render_multi_day(sid: str, processed: list[str], locus_dict: dict):
    """복수 날짜 혼잡도 분석 — 날짜별/요일별 패턴."""
    date_list_dt = [datetime.strptime(d, "%Y%m%d") for d in processed]

    col_a, col_b = st.columns(2)
    with col_a:
        start_dt = st.date_input(
            "시작일",
            value=date_list_dt[0].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
            key="cong_start",
        )
    with col_b:
        end_dt = st.date_input(
            "종료일",
            value=date_list_dt[-1].date(),
            min_value=date_list_dt[0].date(),
            max_value=date_list_dt[-1].date(),
            key="cong_end",
        )

    if start_dt > end_dt:
        st.error("시작일이 종료일보다 늦습니다.")
        return

    selected_dates = [
        d for d in processed
        if start_dt <= datetime.strptime(d, "%Y%m%d").date() <= end_dt
    ]

    if not selected_dates:
        st.warning("선택 기간에 데이터가 없습니다.")
        return

    st.caption(
        f"분석 기간: **{date_label(selected_dates[0])}** ~ "
        f"**{date_label(selected_dates[-1])}** ({len(selected_dates)}일)"
    )

    with st.spinner(f"기간 혼잡도 분석 중... ({len(selected_dates)}일)"):
        from src.pipeline.congestion import (
            compute_multi_day_congestion, compute_day_of_week_pattern,
        )
        multi_df = compute_multi_day_congestion(selected_dates, sid, locus_dict)

    if multi_df.empty:
        st.warning("혼잡도 데이터 없음 (journey.parquet 필요)")
        return

    st.divider()

    # ── 탭 ─────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📊 날짜별 추이", "📅 요일별 패턴", "🏢 공간별 비교"])

    with t1:
        _render_daily_congestion_trend(multi_df)

    with t2:
        dow_df = compute_day_of_week_pattern(multi_df)
        _render_dow_pattern(dow_df)

    with t3:
        _render_space_comparison(multi_df)


def _render_daily_congestion_trend(multi_df: pd.DataFrame):
    """날짜별 전체 혼잡도 추이."""
    st.markdown("#### 📊 날짜별 혼잡도 추이")

    # 날짜별 시간대별 총 인원
    daily_agg = (
        multi_df.groupby(["date", "hour"])["avg_workers"]
        .sum()
        .reset_index()
    )
    from src.utils.weather import date_label_short
    daily_agg["date_fmt"] = daily_agg["date"].apply(date_label_short)

    # 작업 시간만
    daily_agg = daily_agg[(daily_agg["hour"] >= 5) & (daily_agg["hour"] <= 23)]

    fig = px.line(
        daily_agg,
        x="hour",
        y="avg_workers",
        color="date_fmt",
        title="날짜별 시간대 혼잡도 (전체 공간 합산)",
        labels={"avg_workers": "평균 인원 합계", "hour": "시간", "date_fmt": "날짜"},
        markers=True,
    )
    fig.update_layout(**_DARK, height=400, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)

    # 날짜별 피크 인원 카드
    daily_peak = multi_df.groupby("date").agg(
        total_avg=("avg_workers", "sum"),
        max_single=("max_workers", "max"),
    ).reset_index()
    daily_peak["date_fmt"] = daily_peak["date"].apply(date_label_short)

    cols = st.columns(min(len(daily_peak), 5))
    for i, (_, row) in enumerate(daily_peak.iterrows()):
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:#1A2A3A; border:1px solid #2A3A4A; "
                f"border-radius:8px; padding:10px; text-align:center; margin:4px 0;'>"
                f"<div style='color:#9AB5D4; font-size:0.78rem;'>{row['date_fmt']}</div>"
                f"<div style='color:#FF6B35; font-size:1.3rem; font-weight:700;'>"
                f"{int(row['max_single'])}명</div>"
                f"<div style='color:#6A7A95; font-size:0.72rem;'>단일 공간 최대</div></div>",
                unsafe_allow_html=True,
            )


def _render_dow_pattern(dow_df: pd.DataFrame):
    """요일별 혼잡도 패턴."""
    if dow_df.empty:
        st.info("요일별 패턴 데이터 부족 (최소 2일 이상)")
        return

    st.markdown("#### 📅 요일별 시간대 혼잡도 패턴")
    st.caption("같은 요일의 시간대별 평균 혼잡도 — 요일 간 차이를 비교")

    # 작업 시간만
    df = dow_df[(dow_df["hour"] >= 5) & (dow_df["hour"] <= 23)].copy()

    fig = px.line(
        df,
        x="hour",
        y="avg_workers",
        color="day_name",
        title="요일별 시간대 평균 혼잡도 (전체 공간 합산)",
        labels={"avg_workers": "평균 인원", "hour": "시간", "day_name": "요일"},
        markers=True,
        category_orders={"day_name": ["월", "화", "수", "목", "금", "토", "일"]},
    )
    fig.update_layout(**_DARK, height=380, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)

    # 요일별 총 평균 비교 (바 차트)
    dow_total = df.groupby(["day_of_week", "day_name"])["avg_workers"].mean().reset_index()
    dow_total = dow_total.sort_values("day_of_week")

    fig2 = go.Figure(go.Bar(
        x=dow_total["day_name"],
        y=dow_total["avg_workers"],
        marker_color=["#00AEEF" if d < 5 else "#FF6B35" for d in dow_total["day_of_week"]],
        text=dow_total["avg_workers"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        textfont=dict(color="#D5E5FF"),
    ))
    fig2.update_layout(
        title="요일별 평균 혼잡도 비교",
        xaxis_title="요일",
        yaxis_title="평균 인원",
        **_DARK, height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_space_comparison(multi_df: pd.DataFrame):
    """공간별 날짜 간 혼잡도 비교."""
    if multi_df.empty:
        return

    st.markdown("#### 🏢 공간별 날짜 간 혼잡도 비교")

    # 공간별 날짜별 평균
    space_daily = (
        multi_df.groupby(["date", "locus_name"])["avg_workers"]
        .mean()
        .reset_index()
    )
    from src.utils.weather import date_label_short
    space_daily["date_fmt"] = space_daily["date"].apply(date_label_short)

    # 상위 10개 공간
    top_spaces = (
        space_daily.groupby("locus_name")["avg_workers"]
        .mean()
        .nlargest(10)
        .index.tolist()
    )
    df = space_daily[space_daily["locus_name"].isin(top_spaces)]

    fig = px.bar(
        df,
        x="locus_name",
        y="avg_workers",
        color="date_fmt",
        barmode="group",
        title="상위 10개 공간 — 날짜별 평균 혼잡도",
        labels={"avg_workers": "평균 인원", "locus_name": "공간", "date_fmt": "날짜"},
    )
    fig.update_layout(**_DARK, height=420, legend=_LEG)
    st.plotly_chart(fig, use_container_width=True)
