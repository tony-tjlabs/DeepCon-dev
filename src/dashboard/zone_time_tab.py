"""
작업 시간 분석 탭
================
작업공간 vs 비작업공간 시간 분포 분석.
Y1 특성: 현장 이탈 없음, 213 locus, 3교대 운영.

섹션:
  1. 현장 전체 KPI (작업공간 비율 / 평균 이동·휴게 시간 / BLE 음영 비율)
  2. 시간대별 작업공간/비작업공간 인원 분포 (Stacked Bar)
  3. 교대별 비교 (day / night / unknown)
  4. 출근 → 첫 작업공간 도달 시간 (MAT 유사)
  5. 퇴근 전 마지막 작업공간 이탈 시간 (EOD 유사)
  6. 업체별 작업공간 비율 Top / Bottom 10
  7. 개인별 상세 (작업자 선택 → 파이 + 타임라인)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.cache.policy import DAILY_PARQUET
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
from src.dashboard.token_utils import get_token_sets as _get_token_sets
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import COMPANY_MIN_WORKERS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── 색상 상수 (styles.CHART_COLORS 단일 소스 참조) ───────────────────
COLOR_WORK    = CHART_COLORS["work_zone"]
COLOR_TRANSIT = CHART_COLORS["transit"]
COLOR_REST    = CHART_COLORS["rest"]
COLOR_GAP     = CHART_COLORS["gap"]

CATEGORY_COLORS = {
    "WORK":    COLOR_WORK,
    "TRANSIT": COLOR_TRANSIT,
    "REST":    COLOR_REST,
    "OTHER":   COLOR_GAP,
}

SHIFT_LABELS = {
    "day":            "주간 (Day)",
    "night":          "야간 (Night)",
    "extended_night": "심야 (Extended Night)",
    "unknown":        "미분류",
}


# ─── 캐시 집계 함수 ───────────────────────────────────────────────────

# ─── 마스터 journey 로더 (탭 전체에서 공유) ──────────────────────────
# 기존 5개 함수가 각자 journey.parquet을 읽던 것을 단일 로더로 통합.
# 필요한 컬럼의 합집합을 한 번만 로드한 뒤, 각 함수가 필요한 컬럼만 뷰로 슬라이스.
_ZONE_JOURNEY_COLS = [
    "timestamp", "user_no", "locus_token", "locus_name",
    "is_work_hour",
]


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _load_zone_journey(sector_id: str, date_str: str,
                       journey_parquet: str, _mtime: float = 0.0) -> pd.DataFrame:
    """zone_time 탭 전체에서 공유되는 journey 마스터 로더.

    ★ 성능: 기존 5개 함수가 각자 read_parquet 하던 것을 단일 로드로 통합.
       동일 파일을 여러 번 로드하던 비용을 제거.
    """
    try:
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(journey_parquet).names)
        cols = [c for c in _ZONE_JOURNEY_COLS if c in avail]
        return pd.read_parquet(journey_parquet, columns=cols)
    except Exception:
        return pd.read_parquet(journey_parquet)


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _compute_hourly_category(sector_id: str, date_str: str, journey_parquet: str) -> pd.DataFrame:
    """
    시간대(0~23) × 카테고리(WORK/TRANSIT/REST/OTHER) 인원 집계.
    작업시간(is_work_hour) 행만 대상.

    Args:
        sector_id: 섹터 ID (캐시 키)
        date_str: 날짜 문자열 (캐시 키)
        journey_parquet: journey.parquet 절대 경로 (캐시 키)

    Returns:
        DataFrame(hour, category, workers)
    """
    # ★ 마스터 로더 사용 (중복 로드 제거)
    jdf_master = _load_zone_journey(sector_id, date_str, journey_parquet)
    jdf = jdf_master[["timestamp", "user_no", "locus_token", "is_work_hour"]].copy()
    ts = _get_token_sets(sector_id)
    work_set    = ts["work"]
    transit_set = ts["transit"]
    rest_set    = ts["rest"]

    # ★ 벡터화: dict 매핑 대신 Series.map + isin (빠른 구현)
    cat = pd.Series("OTHER", index=jdf.index)
    cat[jdf["locus_token"].isin(work_set)]    = "WORK"
    cat[jdf["locus_token"].isin(transit_set)] = "TRANSIT"
    cat[jdf["locus_token"].isin(rest_set)]    = "REST"
    jdf["category"] = cat
    jdf["hour"]     = jdf["timestamp"].dt.hour

    hourly = (
        jdf[jdf["is_work_hour"]]
        .groupby(["hour", "category"])["user_no"]
        .nunique()
        .reset_index(name="workers")
    )
    return hourly


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _compute_mat_eod(sector_id: str, date_str: str, journey_parquet: str, worker_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    MAT (출근 → 첫 작업공간 도달 시간) 및
    EOD (마지막 작업공간 이탈 → 퇴근 시간) 계산.

    Returns:
        (mat_df, eod_df) — 각각 user_no, mat_min / eod_min 컬럼 포함
    """
    ts = _get_token_sets(sector_id)
    work_set = ts["work"]

    # ★ 마스터 로더 사용 (중복 로드 제거)
    jdf = _load_zone_journey(sector_id, date_str, journey_parquet)
    wdf = pd.read_parquet(
        worker_parquet,
        columns=["user_no", "in_datetime", "out_datetime"],
    )

    j_work = jdf[jdf["is_work_hour"] & jdf["locus_token"].isin(work_set)].copy()

    # MAT: 첫 WORK 도달
    first_work = (
        j_work.sort_values("timestamp")
        .groupby("user_no")["timestamp"]
        .first()
        .reset_index(name="first_work_time")
    )
    mat_df = first_work.merge(wdf[["user_no", "in_datetime"]].dropna(), on="user_no", how="inner")
    mat_df["in_datetime"]     = pd.to_datetime(mat_df["in_datetime"])
    mat_df["first_work_time"] = pd.to_datetime(mat_df["first_work_time"])
    mat_df["mat_min"] = (mat_df["first_work_time"] - mat_df["in_datetime"]).dt.total_seconds() / 60
    mat_df = mat_df[(mat_df["mat_min"] >= 0) & (mat_df["mat_min"] <= 120)].copy()

    # EOD: 마지막 WORK 이탈
    last_work = (
        j_work.sort_values("timestamp")
        .groupby("user_no")["timestamp"]
        .last()
        .reset_index(name="last_work_time")
    )
    eod_df = last_work.merge(wdf[["user_no", "out_datetime"]].dropna(), on="user_no", how="inner")
    eod_df["out_datetime"]   = pd.to_datetime(eod_df["out_datetime"])
    eod_df["last_work_time"] = pd.to_datetime(eod_df["last_work_time"])
    eod_df["eod_min"] = (eod_df["out_datetime"] - eod_df["last_work_time"]).dt.total_seconds() / 60
    eod_df = eod_df[(eod_df["eod_min"] >= 0) & (eod_df["eod_min"] <= 120)].copy()

    return mat_df[["user_no", "mat_min"]], eod_df[["user_no", "eod_min"]]


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _compute_mat_eod_flow(
    sector_id: str,
    date_str: str,
    journey_parquet: str,
    worker_parquet: str,
) -> tuple[dict, dict]:
    """
    MAT/EOD 동선 구간별 평균 소요시간 분해.

    MAT 창(in_datetime → first_work_time), EOD 창(last_work_time → out_datetime) 에서
    각 1분 레코드가 어느 구간(gate / transit / gap)에 해당하는지 집계한 뒤
    전체 작업자 평균을 반환한다.

    Returns:
        (mat_flow, eod_flow)
        각각 {"gate_min", "transit_min", "gap_min", "total_min", "n_workers"}
    """
    import config as cfg

    ts          = _get_token_sets(sector_id)
    work_set    = ts["work"]
    transit_set = ts["transit"]

    # Gate token set — locus_v2.csv에서 locus_type=="GATE" 인 locus_id 집합
    gate_set: set[str] = set()
    try:
        paths    = cfg.get_sector_paths(sector_id)
        csv_path = paths.get("locus_v2_csv")
        if csv_path and csv_path.exists():
            ldf      = pd.read_csv(csv_path, encoding="utf-8-sig")
            gate_set = set(ldf[ldf["locus_type"] == "GATE"]["locus_id"].dropna())
    except Exception as _e:
        logger.debug("gate_set 로드 실패: %s", _e)

    # ── journey 로드 (최소 컬럼) ─────────────────────────────────
    try:
        import pyarrow.parquet as pq
        _avail = set(pq.read_schema(journey_parquet).names)
        _jcols = [c for c in ["timestamp", "user_no", "locus_token"] if c in _avail]
        jdf = pd.read_parquet(journey_parquet, columns=_jcols)
    except Exception:
        jdf = pd.read_parquet(journey_parquet)
    jdf["timestamp"] = pd.to_datetime(jdf["timestamp"])

    # ── worker in/out 로드 ───────────────────────────────────────
    wdf = pd.read_parquet(
        worker_parquet,
        columns=["user_no", "in_datetime", "out_datetime"],
    )
    wdf["in_datetime"]  = pd.to_datetime(wdf["in_datetime"])
    wdf["out_datetime"] = pd.to_datetime(wdf["out_datetime"])
    wdf = wdf.dropna(subset=["in_datetime", "out_datetime"])

    # ── 첫/마지막 WORK 시각 ──────────────────────────────────────
    j_work     = jdf[jdf["locus_token"].isin(work_set)].copy()
    first_work = (
        j_work.groupby("user_no")["timestamp"].min()
        .reset_index(name="first_work_time")
    )
    last_work = (
        j_work.groupby("user_no")["timestamp"].max()
        .reset_index(name="last_work_time")
    )

    # ── MAT 창 ───────────────────────────────────────────────────
    mat_win = wdf.merge(first_work, on="user_no", how="inner")
    mat_win["mat_total"] = (
        (mat_win["first_work_time"] - mat_win["in_datetime"])
        .dt.total_seconds() / 60
    ).clip(0, 120)
    mat_win = mat_win[mat_win["mat_total"] > 0].copy()

    # ── EOD 창 ───────────────────────────────────────────────────
    eod_win = wdf.merge(last_work, on="user_no", how="inner")
    eod_win["eod_total"] = (
        (eod_win["out_datetime"] - eod_win["last_work_time"])
        .dt.total_seconds() / 60
    ).clip(0, 120)
    eod_win = eod_win[eod_win["eod_total"] > 0].copy()

    def _flow_from_window(
        win_df: pd.DataFrame,
        start_col: str,
        end_col: str,
        total_col: str,
    ) -> dict:
        """창 내 분류별 평균 분 계산 (벡터화)."""
        _empty = {"gate_min": 0.0, "transit_min": 0.0, "gap_min": 0.0,
                  "total_min": 0.0, "n_workers": 0}
        if win_df.empty:
            return _empty

        # journey와 창 정보 merge 후 시간 범위 필터
        j_m = jdf.merge(
            win_df[["user_no", start_col, end_col, total_col]],
            on="user_no", how="inner",
        )
        j_win = j_m[
            (j_m["timestamp"] >= j_m[start_col]) &
            (j_m["timestamp"] <= j_m[end_col])
        ].copy()

        # 카테고리 분류 (gate → transit → other 순)
        cat = pd.Series("other", index=j_win.index)
        if gate_set:
            cat[j_win["locus_token"].isin(gate_set)]    = "gate"
        cat[j_win["locus_token"].isin(transit_set)]     = "transit"
        j_win["_cat"] = cat

        # 작업자 × 카테고리 집계
        counts = (
            j_win.groupby(["user_no", "_cat"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["gate", "transit", "other"], fill_value=0)
        )

        # 전체 창 분 수 merge
        totals = win_df.set_index("user_no")[total_col]
        counts = counts.join(totals, how="left").fillna(0)

        # 음영 = 창 전체 - 탐지 분
        detected     = counts[["gate", "transit", "other"]].sum(axis=1)
        counts["gap_min"] = (counts[total_col] - detected).clip(0)

        # journey 미감지 작업자 (창에는 있지만 journey에 없음)
        missing_users = set(win_df["user_no"]) - set(counts.index)
        if missing_users:
            miss_tot = win_df.set_index("user_no").loc[list(missing_users), total_col]
            miss_df  = pd.DataFrame({
                "gate": 0.0, "transit": 0.0, "other": 0.0,
                total_col: miss_tot.values,
                "gap_min": miss_tot.values,
            }, index=list(missing_users))
            counts = pd.concat([counts, miss_df])

        def _col_mean(col: str) -> float:
            return float(counts[col].mean()) if col in counts.columns else 0.0

        return {
            "gate_min":    _col_mean("gate"),
            "transit_min": _col_mean("transit"),
            "gap_min":     _col_mean("gap_min"),
            "total_min":   float(counts[total_col].mean()),
            "n_workers":   len(counts),
        }

    mat_flow = _flow_from_window(mat_win, "in_datetime",    "first_work_time", "mat_total")
    eod_flow = _flow_from_window(eod_win, "last_work_time", "out_datetime",    "eod_total")
    return mat_flow, eod_flow


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _compute_company_ratios(sector_id: str, date_str: str, worker_parquet: str) -> pd.DataFrame:
    """
    업체별 작업공간 비율 집계.
    work_zone_minutes / work_minutes.

    ★ 신뢰도 보정 2단계:
      1) helmet_abandoned=True 작업자 제외 (AccessLog 5분 미만, T-Ward 2시간 이상 — 헬멧 방치 의심)
      2) work_zone_minutes > work_minutes 인 작업자는 work_zone을 work_minutes로 캡핑
         (is_work_hour 필터 미적용 구 parquet 호환 또는 AccessLog 오류 방어)

    Returns:
        DataFrame(company_name, work_zone_minutes_sum, work_minutes_sum, work_ratio,
                  worker_count, excluded_count)
        work_ratio: 0~1
    """
    try:
        import pyarrow.parquet as pq
        _avail = set(pq.read_schema(worker_parquet).names)
        _cols = [c for c in [
            "user_no", "company_name", "work_zone_minutes", "work_minutes", "helmet_abandoned"
        ] if c in _avail]
        wdf = pd.read_parquet(worker_parquet, columns=_cols)
    except Exception:
        wdf = pd.read_parquet(worker_parquet)

    wdf = wdf[wdf["work_minutes"].fillna(0) > 0].copy()

    # 단계 1: helmet_abandoned 제외
    _excluded_count = 0
    if "helmet_abandoned" in wdf.columns:
        _excl = wdf["helmet_abandoned"].fillna(False).astype(bool)
        _excluded_count = int(_excl.sum())
        wdf = wdf[~_excl].copy()

    # 단계 2: work_zone > work_minutes 캡핑 (구 parquet 호환)
    #   work_zone_minutes는 전체 journey 기반, work_minutes는 AccessLog 기반이므로
    #   is_work_hour 필터가 미적용된 구 parquet에서 work_zone > work_minutes 발생 가능.
    wdf["work_zone_minutes"] = wdf[["work_zone_minutes", "work_minutes"]].min(axis=1)

    agg = (
        wdf.groupby("company_name")
        .agg(
            work_zone_sum=("work_zone_minutes", "sum"),
            work_sum=("work_minutes", "sum"),
            worker_count=("user_no", "nunique"),
        )
        .reset_index()
    )
    agg["work_ratio"] = (agg["work_zone_sum"] / agg["work_sum"]).clip(0, 1).round(4)
    agg["excluded_count"] = _excluded_count  # 업체 단위가 아닌 전체 제외 수 (캡션용)
    return agg.sort_values("work_ratio", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=DAILY_PARQUET)
def _compute_individual_timeline(
    sector_id: str,
    date_str: str,
    user_no: int,
    journey_parquet: str,
) -> pd.DataFrame:
    """
    특정 작업자의 시간대별 locus 카테고리 타임라인.

    Returns:
        DataFrame(timestamp, locus_token, category) — 1분 간격
    """
    ts = _get_token_sets(sector_id)
    work_set    = ts["work"]
    transit_set = ts["transit"]
    rest_set    = ts["rest"]

    # ★ 마스터 로더 사용 (중복 로드 제거)
    jdf = _load_zone_journey(sector_id, date_str, journey_parquet)
    user_df = jdf[jdf["user_no"] == user_no].copy()
    user_df = user_df.sort_values("timestamp").reset_index(drop=True)

    # ★ 벡터화 카테고리 할당
    cat = pd.Series("OTHER", index=user_df.index)
    cat[user_df["locus_token"].isin(work_set)]    = "WORK"
    cat[user_df["locus_token"].isin(transit_set)] = "TRANSIT"
    cat[user_df["locus_token"].isin(rest_set)]    = "REST"
    user_df["category"] = cat
    return user_df


# ─── 차트 헬퍼 ───────────────────────────────────────────────────────

def _kpi_row(cols_data: list[tuple[str, str, str]]) -> None:
    """
    KPI 카드 행 렌더링.
    cols_data: [(label, value, color), ...]
    """
    cols = st.columns(len(cols_data))
    for col, (label, value, color) in zip(cols, cols_data):
        with col:
            st.markdown(metric_card(label, value, color=color), unsafe_allow_html=True)


def _plotly_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """공통 Plotly 다크 레이아웃 적용."""
    fig.update_layout(
        **PLOTLY_DARK,
        title=title,
        title_font_color=COLORS["text"],
        height=height,
        legend=PLOTLY_LEGEND,
    )
    return fig


# ─── 섹션별 렌더 함수 ─────────────────────────────────────────────────

def _render_kpi(wdf: pd.DataFrame) -> None:
    """섹션 1: 현장 전체 KPI."""
    st.markdown(section_header("현장 전체 KPI"), unsafe_allow_html=True)

    # 분모가 0인 경우 방어 + 컬럼 결측 방어 (EWI 계산 실패한 parquet 호환)
    def _safe_col(df, col):
        return df[col] if col in df.columns else pd.Series([0] * len(df), index=df.index)

    total_work = _safe_col(wdf, "work_minutes").fillna(0).sum()
    total_work_zone = _safe_col(wdf, "work_zone_minutes").fillna(0).sum()
    total_gap = _safe_col(wdf, "gap_min").fillna(0).sum()
    # metrics.py의 *_min 우선, 없으면 processor.py의 *_minutes fallback
    _transit_col = "transit_min" if "transit_min" in wdf.columns else ("transit_count" if "transit_count" in wdf.columns else None)
    _rest_col    = "rest_min"    if "rest_min"    in wdf.columns else ("rest_minutes"  if "rest_minutes"  in wdf.columns else None)
    avg_transit = _safe_col(wdf, _transit_col).fillna(0).mean() if _transit_col else 0.0
    avg_rest    = _safe_col(wdf, _rest_col).fillna(0).mean()    if _rest_col    else 0.0

    work_ratio = (total_work_zone / total_work * 100) if total_work > 0 else 0.0
    gap_ratio = (total_gap / total_work * 100) if total_work > 0 else 0.0

    # 색상: 작업공간 비율 60% 이상 → accent, 미만 → warning
    ratio_color = COLORS["accent"] if work_ratio >= 60 else COLORS["warning"]
    gap_color = COLORS["text_muted"] if gap_ratio < 30 else COLORS["warning"]

    _kpi_row([
        ("작업공간 비율",   f"{work_ratio:.1f}%",      ratio_color),
        ("평균 이동 시간",  f"{avg_transit:.0f} 분",   COLORS["text_muted"]),
        ("평균 휴게 시간",  f"{avg_rest:.0f} 분",      COLORS["success"]),
        ("BLE 음영 비율",  f"{gap_ratio:.1f}%",       gap_color),
    ])

    with st.expander("KPI 해석 가이드"):
        st.markdown(
            """
- **작업공간 비율**: `work_zone_minutes / work_minutes`. 현장 전체 체류 중 실제 작업구역 체류 비율.
- **BLE 음영 비율**: `gap_min / work_minutes`. 작업공간 깊숙한 음영 지역으로 BLE 미수집 추정 구간.
  음영이 높을수록 작업공간 비율이 실제보다 낮게 측정될 수 있음.
- **이동 시간**: TRANSIT locus (복도·게이트·호이스트) 체류 기준.
- **휴게 시간**: REST locus (휴게실·흡연장·식당) 체류 기준.
"""
        )


def _render_hourly_distribution(hourly_df: pd.DataFrame) -> None:
    """섹션 2: 시간대별 카테고리 인원 분포 Stacked Bar."""
    st.markdown(section_header("시간대별 작업공간 / 비작업공간 인원 분포"), unsafe_allow_html=True)
    st.markdown(
        sub_header("시간대별 WORK / TRANSIT / REST 동시 체류 인원"),
        unsafe_allow_html=True,
    )

    cat_order = ["WORK", "TRANSIT", "REST", "OTHER"]
    cat_labels = {
        "WORK":    "작업공간",
        "TRANSIT": "이동",
        "REST":    "휴게",
        "OTHER":   "기타",
    }

    pivot = (
        hourly_df.pivot(index="hour", columns="category", values="workers")
        .reindex(columns=cat_order, fill_value=0)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    # work_hours 마커 (07~18시)
    fig = go.Figure()
    for cat in cat_order:
        if cat not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot["hour"],
            y=pivot[cat],
            name=cat_labels[cat],
            marker_color=CATEGORY_COLORS[cat],
        ))

    fig.update_layout(
        **PLOTLY_DARK,
        barmode="stack",
        height=380,
        xaxis=dict(
            title="시간 (시)",
            tickvals=list(range(0, 24)),
            ticktext=[f"{h:02d}" for h in range(24)],
            tickfont_color=COLORS["text_muted"],
        ),
        yaxis=dict(title="작업자 수 (명)", tickfont_color=COLORS["text_muted"]),
        legend=PLOTLY_LEGEND,
    )
    # 작업시간대 07~18 강조 배경
    fig.add_vrect(x0=6.5, x1=18.5, fillcolor="rgba(0,174,239,0.04)", line_width=0, layer="below")
    st.plotly_chart(fig, use_container_width=True)


def _render_shift_comparison(wdf: pd.DataFrame) -> None:
    """섹션 3: 교대별 작업공간/비작업공간 비교."""
    st.markdown(section_header("교대별 작업 시간 분포"), unsafe_allow_html=True)

    if "shift_type" not in wdf.columns:
        st.info("shift_type 컬럼이 없습니다.")
        return

    wdf_copy = wdf.copy()
    wdf_copy["shift_label"] = (
        wdf_copy["shift_type"]
        .fillna("unknown")
        .astype(str)
        .map(lambda x: SHIFT_LABELS.get(x, x))
    )
    wdf_copy = wdf_copy[wdf_copy["work_minutes"].fillna(0) > 0]

    # 컬럼 결측 방어 — EWI 계산 실패 parquet 호환
    for _c in ("transit_min", "rest_min", "gap_min"):
        if _c not in wdf_copy.columns:
            wdf_copy[_c] = 0.0

    agg = (
        wdf_copy.groupby("shift_label")
        .agg(
            avg_work_zone=("work_zone_minutes", "mean"),
            avg_transit=("transit_min",         "mean"),
            avg_rest=("rest_min",               "mean"),
            avg_gap=("gap_min",                 "mean"),
            worker_count=("user_no",             "nunique"),
        )
        .reset_index()
    )

    # work_ratio 추가
    agg["avg_work_total"] = agg[["avg_work_zone", "avg_transit", "avg_rest", "avg_gap"]].sum(axis=1)
    agg["work_ratio_pct"] = (
        agg["avg_work_zone"] / agg["avg_work_total"].replace(0, np.nan) * 100
    ).fillna(0).round(1)

    col_table, col_chart = st.columns([2, 3])
    with col_table:
        st.markdown(sub_header("교대별 평균 시간 (분)"), unsafe_allow_html=True)
        disp = agg.rename(columns={
            "shift_label":    "교대",
            "avg_work_zone":  "작업공간",
            "avg_transit":    "이동",
            "avg_rest":       "휴게",
            "avg_gap":        "음영",
            "work_ratio_pct": "작업공간 비율(%)",
            "worker_count":   "작업자 수",
        })
        disp = disp[["교대", "작업공간", "이동", "휴게", "음영", "작업공간 비율(%)", "작업자 수"]].copy()
        for c in ["작업공간", "이동", "휴게", "음영"]:
            disp[c] = disp[c].round(0).astype(int)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown(sub_header("교대별 시간 구성"), unsafe_allow_html=True)
        fig = go.Figure()
        for col, label, color in [
            ("avg_work_zone", "작업공간", COLOR_WORK),
            ("avg_transit",   "이동",    COLOR_TRANSIT),
            ("avg_rest",      "휴게",    COLOR_REST),
            ("avg_gap",       "음영",    COLOR_GAP),
        ]:
            fig.add_trace(go.Bar(
                x=agg["shift_label"],
                y=agg[col].round(1),
                name=label,
                marker_color=color,
            ))
        fig.update_layout(
            **PLOTLY_DARK,
            barmode="stack",
            height=320,
            xaxis=dict(tickfont_color=COLORS["text_muted"]),
            yaxis=dict(title="평균 시간 (분)", tickfont_color=COLORS["text_muted"]),
            legend=PLOTLY_LEGEND,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_mat(mat_df: pd.DataFrame) -> None:
    """섹션 4: 출근 후 첫 작업공간 도달 시간 (MAT)."""
    st.markdown(section_header("출근 후 첫 작업공간 도달 시간 (MAT)"), unsafe_allow_html=True)

    if mat_df.empty:
        st.info("MAT 계산 가능한 데이터가 없습니다.")
        return

    median_mat = mat_df["mat_min"].median()
    within_30 = (mat_df["mat_min"] <= 30).mean() * 100
    over_30   = int((mat_df["mat_min"] > 30).sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            metric_card("중앙값 MAT", f"{median_mat:.0f} 분", color=COLORS["accent"]),
            unsafe_allow_html=True,
        )
    with c2:
        col = COLORS["success"] if within_30 >= 70 else COLORS["warning"]
        st.markdown(
            metric_card("30분 이내 비율", f"{within_30:.1f}%", color=col),
            unsafe_allow_html=True,
        )
    with c3:
        col = COLORS["text_muted"] if over_30 < 500 else COLORS["warning"]
        st.markdown(
            metric_card("30분 초과 인원", f"{over_30:,} 명", color=col),
            unsafe_allow_html=True,
        )

    st.markdown(sub_header("MAT 분포 (0~60분, 5분 단위)"), unsafe_allow_html=True)
    fig = px.histogram(
        mat_df[mat_df["mat_min"] <= 60],
        x="mat_min",
        nbins=12,
        color_discrete_sequence=[COLOR_WORK],
    )
    fig.update_traces(marker_line_color=COLORS["bg"], marker_line_width=1)
    fig.update_layout(
        **PLOTLY_DARK,
        height=300,
        xaxis=dict(title="도달 시간 (분)", tickfont_color=COLORS["text_muted"]),
        yaxis=dict(title="작업자 수 (명)", tickfont_color=COLORS["text_muted"]),
    )
    fig.add_vline(
        x=30, line_dash="dash", line_color=COLORS["warning"],
        annotation_text="30분", annotation_font_color=COLORS["warning"],
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("해석"):
        st.markdown(
            "MAT(Mobile Arrival Time)는 출입 기록 후 첫 WORK locus 감지까지 소요 시간입니다. "
            "이 시간에는 타각기 통과, 안전 장비 착용, 작업구역 이동이 포함됩니다."
        )


def _render_eod(eod_df: pd.DataFrame) -> None:
    """섹션 5: 마지막 작업공간 이탈 시간 (EOD)."""
    st.markdown(section_header("퇴근 전 마지막 작업공간 이탈 시간 (EOD)"), unsafe_allow_html=True)

    if eod_df.empty:
        st.info("EOD 계산 가능한 데이터가 없습니다.")
        return

    median_eod = eod_df["eod_min"].median()
    within_30  = (eod_df["eod_min"] <= 30).mean() * 100
    over_30    = int((eod_df["eod_min"] > 30).sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            metric_card("중앙값 EOD", f"{median_eod:.0f} 분", color=COLORS["accent"]),
            unsafe_allow_html=True,
        )
    with c2:
        col = COLORS["success"] if within_30 >= 70 else COLORS["warning"]
        st.markdown(
            metric_card("30분 이내 비율", f"{within_30:.1f}%", color=col),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("30분 초과 인원", f"{over_30:,} 명", color=COLORS["text_muted"]),
            unsafe_allow_html=True,
        )

    st.markdown(sub_header("EOD 분포 (0~60분, 5분 단위)"), unsafe_allow_html=True)
    fig = px.histogram(
        eod_df[eod_df["eod_min"] <= 60],
        x="eod_min",
        nbins=12,
        color_discrete_sequence=[COLOR_TRANSIT],
    )
    fig.update_traces(marker_line_color=COLORS["bg"], marker_line_width=1)
    fig.update_layout(
        **PLOTLY_DARK,
        height=300,
        xaxis=dict(title="이탈 시간 (분)", tickfont_color=COLORS["text_muted"]),
        yaxis=dict(title="작업자 수 (명)", tickfont_color=COLORS["text_muted"]),
    )
    fig.add_vline(
        x=30, line_dash="dash", line_color=COLORS["warning"],
        annotation_text="30분", annotation_font_color=COLORS["warning"],
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("해석"):
        st.markdown(
            "EOD(End-of-Day departure)는 마지막 WORK locus 감지 후 출입 종료 기록까지 소요 시간입니다. "
            "이 시간에는 공구 정리, 장비 반납, 타각기 통과가 포함됩니다."
        )


def _render_journey_flow(mat_flow: dict, eod_flow: dict) -> None:
    """섹션 4.5: 출퇴근 동선 구간별 평균 소요시간 (스택 가로 막대).

    M15X의 _render_route_segment_chart 와 동일한 컨셉.
    Y1은 세그먼트 컬럼 대신 locus_type 기반으로 gate / transit / gap 으로 분류.
    배치: MAT 분포 · EOD 분포 직후.
    """
    st.markdown(section_header("출퇴근 동선 구간별 소요시간"), unsafe_allow_html=True)
    st.caption(
        "출근(MAT) · 퇴근(EOD) 평균 소요시간을 구간별로 분해합니다. "
        "타각기(게이트) ■ 이동 ■ 음영(BLE 미감지) ⊘"
    )

    def _make_flow_fig(flow: dict, direction_label: str) -> go.Figure:
        """스택 가로 막대 — 각 구간 평균 분."""
        gate_m    = flow.get("gate_min",    0.0)
        transit_m = flow.get("transit_min", 0.0)
        gap_m     = flow.get("gap_min",     0.0)
        total_m   = flow.get("total_min",   0.0)

        fig = go.Figure()

        if gate_m > 0.05:
            fig.add_trace(go.Bar(
                name="타각기",
                x=[gate_m], y=[direction_label],
                orientation="h",
                marker_color=CHART_COLORS["gate"],
                text=[f"{gate_m:.1f}분"], textposition="inside",
                textfont=dict(size=12, color="white"),
            ))

        if transit_m > 0.05:
            fig.add_trace(go.Bar(
                name="이동",
                x=[transit_m], y=[direction_label],
                orientation="h",
                marker_color=COLOR_TRANSIT,
                text=[f"{transit_m:.1f}분"], textposition="inside",
                textfont=dict(size=12, color="white"),
            ))

        if gap_m > 0.3:
            fig.add_trace(go.Bar(
                name="음영(미감지)",
                x=[gap_m], y=[direction_label],
                orientation="h",
                marker=dict(
                    color="rgba(140,140,140,0.25)",
                    pattern=dict(shape="/", fgcolor="rgba(190,190,190,0.5)", size=5),
                    line=dict(color="rgba(180,180,180,0.35)", width=1),
                ),
                text=[f"~{gap_m:.1f}분"], textposition="inside",
                textfont=dict(size=11, color="rgba(210,210,210,0.8)"),
            ))

        _layout = {**PLOTLY_DARK, "margin": dict(l=10, r=20, t=8, b=28)}
        fig.update_layout(
            **_layout,
            barmode="stack",
            height=110,
            xaxis=dict(
                title="소요시간 (분)",
                range=[0, max(total_m * 1.2, 5.0)],
                tickfont_color=COLORS["text_muted"],
            ),
            yaxis=dict(tickfont_color=COLORS["text"]),
            legend=dict(
                orientation="h", y=1.45, x=0,
                font_color=COLORS["text"], bgcolor="rgba(0,0,0,0)",
            ),
        )
        return fig

    def _flow_table(flow: dict, labels: list[tuple[str, str]]) -> pd.DataFrame:
        """구간 분해 요약 테이블. labels = [(label, key), ...]"""
        rows = []
        for lbl, key in labels:
            v = flow.get(key, 0.0)
            prefix = "~" if key == "gap_min" else ""
            rows.append({"구간": lbl, "평균 (분)": f"{prefix}{v:.1f}"})
        total = flow.get("total_min", 0.0)
        rows.append({"구간": "**합계**", "평균 (분)": f"**{total:.1f}**"})
        return pd.DataFrame(rows)

    col_mat, col_eod = st.columns(2)

    with col_mat:
        st.markdown(sub_header("출근 동선 (MAT)"), unsafe_allow_html=True)
        st.caption("타각기 진입 → 이동 → 첫 작업공간 도달")
        n = mat_flow.get("n_workers", 0)
        if n == 0:
            st.info("출근 동선 데이터 없음")
        else:
            st.plotly_chart(_make_flow_fig(mat_flow, "출근 →"), use_container_width=True)
            tbl = _flow_table(mat_flow, [
                ("타각기 체류", "gate_min"),
                ("이동",       "transit_min"),
                ("음영(미감지)", "gap_min"),
            ])
            st.dataframe(tbl, hide_index=True, use_container_width=True, height=178)
            st.caption(f"※ {n:,}명 기준")

    with col_eod:
        st.markdown(sub_header("퇴근 동선 (EOD)"), unsafe_allow_html=True)
        st.caption("마지막 작업공간 이탈 → 이동 → 타각기 퇴장")
        n_e = eod_flow.get("n_workers", 0)
        if n_e == 0:
            st.info("퇴근 동선 데이터 없음")
        else:
            st.plotly_chart(_make_flow_fig(eod_flow, "퇴근 →"), use_container_width=True)
            tbl_e = _flow_table(eod_flow, [
                ("이동",       "transit_min"),
                ("타각기 체류", "gate_min"),
                ("음영(미감지)", "gap_min"),
            ])
            st.dataframe(tbl_e, hide_index=True, use_container_width=True, height=178)
            st.caption(f"※ {n_e:,}명 기준")


def _render_work_zone_definition(sector_id: str) -> None:
    """
    현재 '작업공간'으로 분류된 Locus 목록과 한계를 expander로 표시.
    locus_v2.csv에서 동적으로 로드 → 항상 실제 설정과 일치.
    """
    import config as cfg
    try:
        import pandas as _pd
        paths = cfg.get_sector_paths(sector_id)
        ldf = _pd.read_csv(paths["locus_v2_csv"], encoding="utf-8-sig")

        work_df  = ldf[ldf["locus_type"] == "WORK_AREA"]
        gate_df  = ldf[ldf["locus_type"] == "GATE"]
        rest_df  = ldf[ldf["locus_type"] == "REST_AREA"]
        tran_df  = ldf[ldf["locus_type"] == "TRANSIT"]

        # 건물/층별 작업공간 요약
        grp = (
            work_df.groupby(["building", "floor"])
            .size()
            .reset_index(name="cnt")
        )
        work_lines = []
        for _, row in grp.iterrows():
            work_lines.append(f"**{row['building']} {row['floor']}** ({row['cnt']}개 게이트웨이)")
        work_summary = " · ".join(work_lines)

        n_work  = len(work_df)
        n_gate  = len(gate_df)
        n_rest  = len(rest_df)
        n_tran  = len(tran_df)
    except Exception:
        work_summary = "locus 데이터 로드 실패"
        n_work = n_gate = n_rest = n_tran = 0

    with st.expander("ℹ️ '작업공간' 정의 및 측정 한계 — 수치 해석 전 필독", expanded=False):
        st.markdown(f"""
**현재 시스템에서 '작업공간'으로 집계되는 구역** (총 {n_work}개 게이트웨이, `locus_type = WORK_AREA`)

{work_summary}

---

**작업공간에 포함되지 않는 구역** (비율 계산에서 제외)

| 유형 | 게이트웨이 수 | 설명 |
|------|------------|------|
| 게이트 (GATE) | {n_gate}개 | 타각기, 출입구 |
| 휴게/편의 (REST_AREA) | {n_rest}개 | 휴게실, 흡연실, 화장실 등 |
| 이동/경유 (TRANSIT) | {n_tran}개 | 복도, 승강기, 경유 지점 등 |

---

**⚠️ 수치 해석 시 주의사항**

1. **게이트웨이 미설치 구역**: 작업공간 내 BLE 수신기가 없는 현장 일부는 측정에서 누락됩니다.
   → 실제보다 **작업공간 비율이 낮게** 측정될 수 있습니다.

2. **인접 구역 신호 혼재**: 작업공간과 비작업공간의 경계에서 신호가 섞일 수 있습니다.
   → 실제로는 이동 중인데 작업공간으로 분류되거나, 그 반대가 발생할 수 있습니다.

3. **154kV·비상발전기동 포함**: 고압전 시설 및 비상발전기동도 `WORK_AREA`로 분류됩니다.
   해당 구역은 실제 공사 작업이 이루어지는 장소이지만, 안전 관리 맥락에서는 별도 해석이 필요합니다.

4. **BLE 음영 시간**: 작업공간에 있었더라도 BLE 신호 미수신 시 해당 시간은 '음영'으로 처리됩니다.
   → 음영 비율이 높은 날/업체는 작업공간 비율이 **과소평가**될 수 있습니다.
""")


def _render_company_ranking(company_df: pd.DataFrame, sector_id: str = "Y1_SKHynix") -> None:
    """섹션 6: 업체별 작업공간 비율 Top / Bottom 10."""
    st.markdown(section_header("업체별 작업공간 비율 Top / Bottom 10"), unsafe_allow_html=True)

    if company_df.empty or len(company_df) < 2:
        st.info("업체 데이터가 부족합니다.")
        return

    # 작업공간 정의 및 한계 명시
    _render_work_zone_definition(sector_id)

    # 보정 안내 캡션 (excluded_count 컬럼이 있을 때만)
    _excl = int(company_df["excluded_count"].iloc[0]) if "excluded_count" in company_df.columns else 0
    if _excl > 0:
        st.caption(
            f"⚠️ 헬멧 방치 의심 작업자 {_excl}명 제외 · "
            "work_zone > work_minutes 케이스는 work_minutes로 캡핑 보정. "
            "정확한 수치는 파이프라인 재처리 후 확인하세요."
        )

    # 최소 3명 이상 업체만 표시
    filtered = company_df[company_df["worker_count"] >= COMPANY_MIN_WORKERS["small"]].copy()
    if filtered.empty:
        filtered = company_df.copy()

    # ★ 100% 업체 플래그 (상단 알림)
    perfect_100 = filtered[filtered["work_ratio"] >= 1.0]
    if not perfect_100.empty:
        names = ", ".join(perfect_100["company_name"].tolist()[:5])
        st.warning(
            f"100% 업체 {len(perfect_100)}개 ({names}) — "
            "보정 후에도 100%이면 해당 업체 parquet 재처리를 권장합니다.",
            icon="⚠️",
        )

    mode = st.radio(
        "표시 기준",
        ["Top 10 (높은 업체)", "Bottom 10 (낮은 업체)"],
        horizontal=True,
        key="zone_company_mode",
    )
    if "Top" in mode:
        subset = filtered.head(10)
        bar_color = COLOR_WORK
        title = "작업공간 비율 상위 10개 업체"
    else:
        subset = filtered.tail(10).iloc[::-1].reset_index(drop=True)
        bar_color = COLOR_TRANSIT
        title = "작업공간 비율 하위 10개 업체"

    subset["work_ratio_pct"] = (subset["work_ratio"] * 100).round(1)
    subset["label"] = subset.apply(
        lambda r: f"{r['company_name']} ({int(r['worker_count'])}명)",
        axis=1,
    )
    # 100% 업체는 빨간색으로 강조
    bar_colors = subset["work_ratio"].apply(
        lambda v: CHART_COLORS["critical"] if v >= 1.0 else bar_color
    ).tolist()

    fig = go.Figure(go.Bar(
        x=subset["work_ratio_pct"],
        y=subset["label"],
        orientation="h",
        marker_color=bar_colors,
        text=subset["work_ratio_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=200, r=60, t=50, b=40)},
        title=title,
        height=max(350, len(subset) * 35 + 80),
        xaxis=dict(title="작업공간 비율 (%)", range=[0, 110], tickfont_color=COLORS["text_muted"]),
        yaxis=dict(autorange="reversed", tickfont_color=COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_individual(wdf: pd.DataFrame, sector_id: str, date_str: str, journey_parquet: str) -> None:
    """섹션 7: 개인별 상세."""
    st.markdown(section_header("개인별 상세 분석"), unsafe_allow_html=True)

    # user_no 기반 selectbox (동명이인 버그 방지)
    valid_workers = wdf[wdf["work_minutes"].fillna(0) > 0].copy()
    if valid_workers.empty:
        st.info("분석 가능한 작업자가 없습니다.")
        return

    # ── 필터 & 정렬 패널 ─────────────────────────────────────────────
    # 정렬 기준 정의 (컬럼명, 오름차순 여부, 레이블)
    SORT_OPTIONS = {
        "이름순 (가→나)":       ("user_name",         True),
        "User_no 오름차순":     ("user_no",            True),
        "작업공간 긴 순":       ("work_zone_minutes",  False),
        "이동시간 긴 순":       ("transit_min",        False),
        "휴게시간 긴 순":       ("rest_min",           False),
        "음영시간 긴 순":       ("gap_min",            False),
        "EWI 높은 순":          ("ewi",                False),
        "피로도 높은 순":       ("fatigue_score",      False),
    }

    with st.container():
        st.markdown(
            f"<div style='background:{COLORS['bg_chart']}; border:1px solid {COLORS['border']}; "
            f"border-radius:8px; padding:12px 16px; margin-bottom:12px;'>",
            unsafe_allow_html=True,
        )
        fc1, fc2, fc3 = st.columns([3, 3, 2])

        with fc1:
            # 업체 필터
            companies = sorted(valid_workers["company_name"].dropna().unique().tolist())
            selected_companies = st.multiselect(
                "업체 필터",
                options=companies,
                default=[],
                placeholder="전체 (미선택 시 전체 표시)",
                key="zone_indiv_company_filter",
            )

        with fc2:
            # 정렬 기준
            sort_key = st.selectbox(
                "정렬 기준",
                options=list(SORT_OPTIONS.keys()),
                index=0,
                key="zone_indiv_sort",
            )

        with fc3:
            # User_no 직접 검색
            uno_search = st.text_input(
                "User_no 검색",
                value="",
                placeholder="예) 12345",
                key="zone_indiv_uno_search",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── 필터 적용 ────────────────────────────────────────────────────
    filtered = valid_workers.copy()

    # 1) 업체 필터
    if selected_companies:
        filtered = filtered[filtered["company_name"].isin(selected_companies)]

    # 2) User_no 검색 (부분 매칭)
    uno_search = uno_search.strip()
    if uno_search:
        filtered = filtered[
            filtered["user_no"].astype(str).str.contains(uno_search, na=False)
        ]

    if filtered.empty:
        st.warning("조건에 맞는 작업자가 없습니다. 필터를 조정해주세요.")
        return

    # ── 정렬 적용 ────────────────────────────────────────────────────
    sort_col, ascending = SORT_OPTIONS[sort_key]
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
    else:
        filtered = filtered.sort_values("user_name", ascending=True)

    # ── 필터 결과 요약 ───────────────────────────────────────────────
    total_all = len(valid_workers)
    total_flt = len(filtered)
    filter_note = ""
    if selected_companies or uno_search:
        filter_note = f" (전체 {total_all:,}명 중 {total_flt:,}명 필터됨)"
    st.caption(
        f"👤 {total_flt:,}명{filter_note} — 정렬: **{sort_key}**"
    )

    # ── 작업자 선택 selectbox ───────────────────────────────────────
    # 표시 레이블: rank + 이름 + user_no + 정렬값
    def _make_label(row, rank: int) -> str:
        name = str(row.get("user_name") or "이름없음")
        uno  = str(row["user_no"])
        # 정렬 기준값 부가 표시
        sort_val = ""
        if sort_col == "user_no":
            sort_val = ""
        elif sort_col in ("work_zone_minutes", "transit_min", "rest_min", "gap_min"):
            v = row.get(sort_col, 0) or 0
            sort_val = f"  {v:.0f}분"
        elif sort_col == "ewi":
            v = row.get("ewi", None)
            sort_val = f"  EWI {v:.3f}" if v is not None else ""
        elif sort_col == "fatigue_score":
            v = row.get("fatigue_score", None)
            sort_val = f"  피로 {v:.3f}" if v is not None else ""
        return f"#{rank}  {name} ({uno}){sort_val}"

    filtered = filtered.reset_index(drop=True)
    labels = [_make_label(row, i + 1) for i, row in filtered.iterrows()]
    label_to_uno = dict(zip(labels, filtered["user_no"].tolist()))

    selected_label = st.selectbox(
        f"작업자 선택 ({total_flt:,}명)",
        labels,
        key="zone_individual_select",
    )
    selected_uno = label_to_uno[selected_label]

    row = filtered[filtered["user_no"] == selected_uno].iloc[0]

    # ── 작업자 KPI ──────────────────────────────────────────────────
    work_min = float(row.get("work_minutes") or 0)
    work_zone = float(row.get("work_zone_minutes") or 0)
    transit_m = float(row.get("transit_min") or 0)
    rest_m = float(row.get("rest_min") or 0)
    gap_m = float(row.get("gap_min") or 0)
    ratio = (work_zone / work_min * 100) if work_min > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, color in zip(
        [c1, c2, c3, c4, c5],
        ["총 현장 체류", "작업공간", "이동", "휴게", "음영(미기록)"],
        [f"{work_min:.0f} 분", f"{work_zone:.0f} 분", f"{transit_m:.0f} 분", f"{rest_m:.0f} 분", f"{gap_m:.0f} 분"],
        [COLORS["text"], COLOR_WORK, COLOR_TRANSIT, COLOR_REST, COLOR_GAP],
    ):
        with col:
            st.markdown(metric_card(lbl, val, color=color), unsafe_allow_html=True)

    # ── 파이 차트 ─────────────────────────────────────────────────
    st.markdown(sub_header("시간 구성 비율"), unsafe_allow_html=True)

    pie_values = [
        max(0, work_zone),
        max(0, transit_m),
        max(0, rest_m),
        max(0, gap_m),
    ]
    pie_labels = ["작업공간", "이동", "휴게", "음영"]
    pie_colors = [COLOR_WORK, COLOR_TRANSIT, COLOR_REST, COLOR_GAP]

    fig_pie = go.Figure(go.Pie(
        labels=pie_labels,
        values=pie_values,
        marker_colors=pie_colors,
        hole=0.45,
        textinfo="label+percent",
        textfont_color=COLORS["text"],
    ))
    fig_pie.update_layout(
        **PLOTLY_DARK,
        height=300,
        showlegend=True,
        legend=PLOTLY_LEGEND,
        annotations=[dict(
            text=f"{ratio:.0f}%<br>작업",
            x=0.5, y=0.5,
            font_size=16,
            font_color=COLORS["accent"],
            showarrow=False,
        )],
    )
    col_pie, col_timeline = st.columns([1, 2])
    with col_pie:
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 시간대별 타임라인 ──────────────────────────────────────────
    with col_timeline:
        st.markdown(sub_header("시간대별 locus 카테고리"), unsafe_allow_html=True)
        try:
            tl_df = _compute_individual_timeline(sector_id, date_str, selected_uno, journey_parquet)
            if tl_df.empty:
                st.info("타임라인 데이터 없음")
            else:
                tl_df["hour_min"] = tl_df["timestamp"].dt.strftime("%H:%M")
                # Plotly strip chart — 1줄 타임라인
                cat_order_tl = ["WORK", "TRANSIT", "REST", "OTHER"]
                cat_colors_tl = {
                    "WORK":    COLOR_WORK,
                    "TRANSIT": COLOR_TRANSIT,
                    "REST":    COLOR_REST,
                    "OTHER":   COLOR_GAP,
                }
                cat_num = {"WORK": 3, "TRANSIT": 2, "REST": 1, "OTHER": 0}
                tl_df["cat_num"] = tl_df["category"].map(cat_num).fillna(0)

                fig_tl = go.Figure()
                for cat in cat_order_tl:
                    sub = tl_df[tl_df["category"] == cat]
                    if sub.empty:
                        continue
                    cat_labels_tl = {
                        "WORK": "작업공간", "TRANSIT": "이동",
                        "REST": "휴게",    "OTHER": "기타",
                    }
                    fig_tl.add_trace(go.Scatter(
                        x=sub["timestamp"],
                        y=[cat_labels_tl[cat]] * len(sub),
                        mode="markers",
                        name=cat_labels_tl[cat],
                        marker=dict(
                            color=cat_colors_tl[cat],
                            size=5,
                            opacity=0.7,
                        ),
                        hovertemplate=(
                            "%{x|%H:%M}<br>"
                            + "locus: " + sub["locus_name"].fillna("").astype(str).values[0]
                            + "<extra></extra>"
                        ) if not sub.empty else "<extra></extra>",
                    ))

                fig_tl.update_layout(
                    **PLOTLY_DARK,
                    height=250,
                    xaxis=dict(
                        title="시간",
                        tickfont_color=COLORS["text_muted"],
                        tickformat="%H:%M",
                    ),
                    yaxis=dict(
                        title="",
                        tickfont_color=COLORS["text"],
                        categoryorder="array",
                        categoryarray=["기타", "휴게", "이동", "작업공간"],
                    ),
                    legend=PLOTLY_LEGEND,
                )
                st.plotly_chart(fig_tl, use_container_width=True)
        except Exception as e:
            logger.warning("개인 타임라인 로드 실패: %s", e)
            st.info("타임라인 데이터를 로드할 수 없습니다.")


# ─── Sanity Warning 렌더 ──────────────────────────────────────────────

def render_sanity_warnings(worker_df: pd.DataFrame) -> None:
    """
    비상식 패턴 감지 결과 표시 (작업시간 분석 탭 하단).

    표시 내용:
      - HIGH 심각도 작업자 카운트 + 비율 (빨간 박스)
      - 규칙별 감지 현황 가로 막대차트
      - 상세 테이블 (expander)
    """
    import json

    if "sanity_flag_count" not in worker_df.columns:
        return  # sanity 컬럼 없으면 (구 parquet) 조용히 skip

    # ── 인라인 집계 (sanity 컬럼이 worker.parquet에 이미 저장됨) ──────────
    def _sanity_summary(df: pd.DataFrame) -> dict:
        n = len(df)
        sev = df.get("sanity_severity", pd.Series(dtype=str))
        suspicious = int((df["sanity_flag_count"] > 0).sum())
        rule_counts: dict[str, int] = {}
        for flags_json in df.get("sanity_flags", pd.Series(dtype=str)).dropna():
            try:
                for rule_id in json.loads(flags_json):
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "total": n,
            "suspicious": suspicious,
            "suspicious_pct": round(suspicious / max(n, 1) * 100, 1),
            "high":   int((sev == "HIGH").sum()),
            "medium": int((sev == "MEDIUM").sum()),
            "low":    int((sev == "LOW").sum()),
            "rule_counts": rule_counts,
        }

    summary = _sanity_summary(worker_df)
    if summary["suspicious"] == 0:
        return  # 감지 없으면 표시 안 함

    st.markdown(section_header("비상식 패턴 감지"), unsafe_allow_html=True)
    st.caption(
        "하루 종일 한 공간에서 휴식·이동 없이 작업하는 등 현실적으로 불가능한 패턴을 자동 탐지합니다. "
        "데이터 오류 또는 T-Ward 착용 이상 가능성을 나타냅니다."
    )

    # ── KPI 행 ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = COLORS["danger"] if summary["high"] > 0 else COLORS["text_muted"]
        st.markdown(
            metric_card("HIGH 이상", f"{summary['high']}명", color=color),
            unsafe_allow_html=True,
        )
    with c2:
        color = COLORS["warning"] if summary["medium"] > 0 else COLORS["text_muted"]
        st.markdown(
            metric_card("MEDIUM 이상", f"{summary['medium']}명", color=color),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("전체 이상 감지", f"{summary['suspicious']}명", color=COLORS["warning"]),
            unsafe_allow_html=True,
        )
    with c4:
        pct = summary["suspicious_pct"]
        color = COLORS["danger"] if pct > 5 else (COLORS["warning"] if pct > 2 else COLORS["text_muted"])
        st.markdown(
            metric_card("이상 감지 비율", f"{pct:.1f}%", color=color),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── 규칙별 막대차트 ────────────────────────────────────────────────
    rule_counts = summary["rule_counts"]
    if rule_counts:
        rule_descs: dict = {}  # 상세 설명 없음 — 규칙 ID를 가독성 있는 레이블로 변환
        rule_labels = []
        rule_vals   = []
        rule_colors = []

        for rule_id, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
            meta = rule_descs.get(rule_id, {})
            sev  = meta.get("severity", "LOW")
            # 짧은 레이블 (ID 앞부분 + 건수)
            short = rule_id.replace("_", " ").title()
            rule_labels.append(short)
            rule_vals.append(cnt)
            rule_colors.append(
                CHART_COLORS["critical"] if sev == "HIGH" else
                CHART_COLORS["medium"]   if sev == "MEDIUM" else
                CHART_COLORS["low"]
            )

        fig = go.Figure(go.Bar(
            x=rule_vals,
            y=rule_labels,
            orientation="h",
            marker_color=rule_colors,
            text=rule_vals,
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
        ))
        fig.update_layout(**{
            **PLOTLY_DARK,
            "title": "규칙별 감지 현황",
            "height": max(280, len(rule_labels) * 38 + 80),
            "margin": dict(l=220, r=60, t=50, b=20),
            "xaxis": dict(title="감지 작업자 수 (명)", tickfont_color=COLORS["text_muted"]),
            "yaxis": dict(autorange="reversed", tickfont_color=COLORS["text"]),
        })
        st.plotly_chart(fig, use_container_width=True)

    # ── 상세 테이블 (expander) ─────────────────────────────────────────
    with st.expander(f"상세 작업자 목록 ({summary['suspicious']}명)"):
        suspicious_df = worker_df[worker_df["is_suspicious"]].copy()
        if suspicious_df.empty:
            st.info("상세 데이터 없음")
        else:
            # 플래그 목록 → 가독성 있는 문자열
            def _flags_to_str(flags_json: str) -> str:
                try:
                    flags = json.loads(flags_json)
                    return ", ".join(f.replace("_", " ") for f in flags)
                except Exception:
                    return flags_json or ""

            disp = suspicious_df[[
                "user_name", "company_name",
                "work_minutes", "sanity_severity", "sanity_flag_count", "sanity_flags",
            ]].copy()
            disp["sanity_flags"] = disp["sanity_flags"].apply(_flags_to_str)
            disp = disp.sort_values(
                ["sanity_severity", "sanity_flag_count"],
                key=lambda col: col.map(
                    {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "OK": 0}
                ) if col.name == "sanity_severity" else col,
                ascending=[False, False],
            ).reset_index(drop=True)

            disp = disp.rename(columns={
                "user_name":         "작업자",
                "company_name":      "업체",
                "work_minutes":      "근무시간(분)",
                "sanity_severity":   "심각도",
                "sanity_flag_count": "플래그 수",
                "sanity_flags":      "감지 규칙",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True, height=380)

            csv_bytes = disp.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV 다운로드",
                data=csv_bytes,
                file_name="sanity_warnings.csv",
                mime="text/csv",
            )


# ─── 메인 렌더 함수 ───────────────────────────────────────────────────

def render_zone_time_tab(sector_id: str) -> None:
    """
    작업 시간 분석 탭 진입점.

    Args:
        sector_id: 현재 선택된 Sector ID (예: "Y1_SKHynix")
    """
    import config as cfg
    from src.pipeline.cache_manager import detect_processed_dates

    # ── 날짜 선택 ──────────────────────────────────────────────────
    processed = detect_processed_dates(sector_id)
    if not processed:
        st.warning("처리된 데이터가 없습니다. 먼저 파이프라인을 실행하세요.")
        return

    try:
        from src.dashboard.date_utils import get_date_selector
        selected_date = get_date_selector(processed, key="zone_date_selector")
    except Exception:
        # date_utils 없을 경우 단순 selectbox 사용
        selected_date = st.selectbox(
            "날짜 선택",
            processed,
            index=len(processed) - 1,
            key="zone_date_selector_fallback",
        )

    if not selected_date:
        return

    # ── M4-T34: 스키마 버전 검증 ─────────────────────────────────
    try:
        from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
        from src.dashboard.components import handle_schema_mismatch
        validate_schema(selected_date, sector_id, strict_legacy=False)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, selected_date)
        return
    except FileNotFoundError:
        pass  # 아래 Parquet 경로 확인에서 안내됨

    # ── Parquet 경로 확인 ──────────────────────────────────────────
    paths = cfg.get_sector_paths(sector_id)
    proc_dir = paths["processed_dir"] / selected_date
    worker_parquet  = str(proc_dir / "worker.parquet")
    journey_parquet = str(proc_dir / "journey.parquet")
    slim_parquet    = str(proc_dir / "journey_slim.parquet")

    from pathlib import Path
    if not Path(worker_parquet).exists():
        st.error(f"worker.parquet 없음: {worker_parquet}")
        return

    # journey 파일 결정: full → slim(Drive 온디맨드) 순으로 fallback
    _using_slim = False
    if not Path(journey_parquet).exists():
        # 로컬에 slim이 없으면 Drive에서 온디맨드 다운로드 시도
        if not Path(slim_parquet).exists():
            import config as _cfg
            if _cfg.CLOUD_MODE:
                with st.spinner(
                    f"☁️ 작업시간 데이터 다운로드 중... "
                    f"({selected_date} · 약 10~15초 소요)"
                ):
                    try:
                        from src.pipeline.drive_storage import init_drive_storage_from_secrets
                        _drive = init_drive_storage_from_secrets(sector_id)
                        if _drive:
                            _drive.ensure_journey_slim(
                                sector_id, selected_date, _cfg.PROCESSED_DIR
                            )
                    except Exception as _e:
                        logger.warning(f"journey_slim 온디맨드 다운로드 실패: {_e}")

        if Path(slim_parquet).exists():
            journey_parquet = slim_parquet
            _using_slim = True
        else:
            st.warning(
                "📦 작업시간 분석에 필요한 데이터가 없습니다. "
                "Drive에 journey_slim.parquet 업로드 후 이용 가능합니다."
            )
            return

    # Cloud slim 안내 배너
    if _using_slim:
        st.info(
            "☁️ **Cloud 환경** — 핵심 6개 컬럼 슬림 데이터를 사용합니다. "
            "첫 날짜 접근 시 **약 10~15초** 다운로드 후 캐시됩니다. "
            "이후 같은 날짜는 즉시 로드됩니다. "
            "개인별 상세 타임라인은 로컬 전체 데이터 환경에서만 지원됩니다."
        )

    # ── 데이터 로드 (컬럼 프루닝 적용) ────────────────────────────
    with st.spinner("데이터 로드 중..."):
        # 주의: processor.py "*_minutes" + metrics.py "*_min" 모두 탭에서 사용
        _worker_cols_needed = [
            "user_no", "user_name", "company_name",
            "work_minutes", "work_zone_minutes", "rest_minutes",
            "transit_count", "confined_minutes", "high_voltage_minutes",
            "shift_type", "in_datetime", "out_datetime",
            "active_minutes", "valid_ble_minutes", "gap_min",
            # EWI 분류 (metrics.py — 개인 상세 파이 차트에 필요)
            "rest_min", "transit_min", "high_active_min", "low_active_min",
            "standby_min", "recorded_work_min",
            "ewi", "ewi_reliable",
        ]
        try:
            import pyarrow.parquet as pq
            _avail = set(pq.read_schema(worker_parquet).names)
            _cols = [c for c in _worker_cols_needed if c in _avail]
            wdf = pd.read_parquet(worker_parquet, columns=_cols) if _cols else pd.read_parquet(worker_parquet)
        except Exception:
            wdf = pd.read_parquet(worker_parquet)

    if wdf.empty:
        st.warning("선택한 날짜의 worker 데이터가 없습니다.")
        return

    # ── 섹션 1: KPI ────────────────────────────────────────────────
    _render_kpi(wdf)
    st.divider()

    # ── 섹션 2: 시간대별 분포 ───────────────────────────────────────
    with st.spinner("시간대별 집계 중..."):
        hourly_df = _compute_hourly_category(sector_id, selected_date, journey_parquet)
    _render_hourly_distribution(hourly_df)
    st.divider()

    # ── AI 코멘터리 (T-16) ──────────────────────────────────────────
    try:
        from src.dashboard.components import ai_commentary_box
        from core.ai import build_zone_time_context
        from src.dashboard.auth import get_current_user

        ai_ctx = build_zone_time_context(
            sector_id=sector_id,
            date_str=str(selected_date),
            worker_df=wdf,
            hourly_df=hourly_df,
        )
        ai_commentary_box(
            role="zone_time_analyst",
            context=ai_ctx,
            sector_id=sector_id,
            date_str=str(selected_date),
            title="업체별 작업시간 AI 분석",
            spinner_text="업체 패턴 분석 중...",
            button_label="AI 분석 실행 (Haiku)",
            user_role=get_current_user().get("role", "unknown"),
            tab="zone_time",
            show_meta=False,
        )
    except Exception as e:
        logger.warning(f"AI 코멘터리 렌더 실패 (zone_time): {e}")
    st.divider()

    # ── 섹션 3: 교대별 비교 ─────────────────────────────────────────
    _render_shift_comparison(wdf)
    st.divider()

    # ── 섹션 4 & 5: MAT / EOD 분포 히스토그램 ──────────────────────
    with st.spinner("MAT / EOD 계산 중..."):
        mat_df, eod_df = _compute_mat_eod(sector_id, selected_date, journey_parquet, worker_parquet)

    col_mat, col_eod = st.columns(2)
    with col_mat:
        _render_mat(mat_df)
    with col_eod:
        _render_eod(eod_df)

    # ── 섹션 4.5: 출퇴근 동선 구간별 소요시간 (가로 스택 막대) ────
    with st.spinner("동선 구간 분해 중..."):
        mat_flow, eod_flow = _compute_mat_eod_flow(
            sector_id, selected_date, journey_parquet, worker_parquet,
        )
    _render_journey_flow(mat_flow, eod_flow)
    st.divider()

    # ── 섹션 6: 업체별 랭킹 ─────────────────────────────────────────
    with st.spinner("업체별 집계 중..."):
        company_df = _compute_company_ratios(sector_id, selected_date, worker_parquet)
    _render_company_ranking(company_df, sector_id=sector_id)
    st.divider()

    # ── 섹션 7: 개인별 상세 ─────────────────────────────────────────
    _render_individual(wdf, sector_id, selected_date, journey_parquet)
    st.divider()

    # ── 섹션 8: 비상식 패턴 감지 (Sanity Check) ─────────────────────
    render_sanity_warnings(wdf)
