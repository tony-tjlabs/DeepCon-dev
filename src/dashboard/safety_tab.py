"""
안전성 분석 탭 — SII · CRE · 위험 구역 · 혼잡도 통합
====================================================
5개 서브탭:
  1. 🛡️ 일별 — 선택 날짜의 안전 상세 (기존 daily/safety + lone_work)
  2. 🏗️ 업체별 — 업체별 SII/CRE 위험 작업자 비율
  3. 🗺️ 공간별 — 혼잡도 히트맵 + CRE 위험 공간 (기존 congestion_tab 재활용)
  4. 📅 기간별 — SII/CRE 추이 + 요일별 혼잡 패턴
  5. 👤 개인별 — 작업자 위험 랭킹 (CRE/SII/밀폐/고립/헬멧방치 등) + 조치 권고
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from core.cache.policy import DAILY_PARQUET, MULTI_DAY_AGG
from src.dashboard.styles import (
    COLORS, CHART_COLORS, PLOTLY_DARK, PLOTLY_LEGEND,
    metric_card, metric_card_sm,
    section_header, sub_header,
)
from src.dashboard.date_utils import get_date_selector, format_date_label
from src.pipeline.cache_manager import detect_processed_dates
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, SII_HIGH, COMPANY_MIN_WORKERS

logger = logging.getLogger(__name__)

# ─── 색상 상수 (styles.CHART_COLORS 단일 소스 참조) ───────────────────
COLOR_SII    = CHART_COLORS["sii"]
COLOR_CRE    = CHART_COLORS["cre"]
COLOR_DANGER = CHART_COLORS["critical"]
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


_SAFETY_WORKER_COLS = [
    "user_no", "company_name", "cre", "sii",
    "static_risk", "confined_minutes",
    "high_voltage_minutes", "alone_ratio",
    "shift_type", "work_minutes", "ewi_reliable",
]


def _safety_load_one(args: tuple):
    """단일 날짜 worker.parquet 로드 — 병렬 worker용 (picklable)."""
    d, p = args
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=_SAFETY_WORKER_COLS)
        df["date"] = d
        return df
    except Exception as e:
        logger.warning(f"[{d}] worker 로드 실패: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=MULTI_DAY_AGG)
def _load_multi_day(sector_id: str, dates: tuple) -> pd.DataFrame:
    """여러 날짜의 worker.parquet 병렬 로드.

    ★ 성능: 기간이 길수록 (예: 40일) ThreadPool 병렬 I/O 효과가 크다.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = cfg.get_sector_paths(sector_id)
    tasks = [(d, paths["processed_dir"] / d / "worker.parquet") for d in dates]
    if not tasks:
        return pd.DataFrame()

    max_w = min(len(tasks), 8)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        frames = [f for f in pool.map(_safety_load_one, tasks) if f is not None]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 1: 일별 안전
# ═══════════════════════════════════════════════════════════════════════

def _render_daily_safety(sector_id: str, date_str: str, shift_filter: str | None) -> None:
    """선택 날짜의 안전 상세 — 기존 daily/safety 재활용."""
    from src.dashboard.daily.safety import render_safety
    from src.pipeline.cache_manager import (
        SchemaVersionMismatch,
        load_daily_results,
        load_journey,
        validate_schema,
    )
    from src.spatial.loader import load_locus_dict
    from src.dashboard.components import handle_schema_mismatch

    # ── M4-T34: 스키마 버전 검증 ─────────────────────────────────
    try:
        validate_schema(date_str, sector_id, strict_legacy=False)
        data = load_daily_results(date_str, sector_id)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, date_str)
        return
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    worker_df = data.get("worker", pd.DataFrame())
    space_df  = data.get("space", pd.DataFrame())
    locus_dict = load_locus_dict(sector_id)

    if worker_df.empty:
        st.warning("worker 데이터 없음")
        return

    if shift_filter and "shift_type" in worker_df.columns:
        worker_df = worker_df[worker_df["shift_type"] == shift_filter]

    has_cre = "cre" in worker_df.columns

    # ── AI 코멘터리 (T-18) ──────────────────────────────────────────
    try:
        from src.dashboard.components import ai_commentary_box
        from core.ai import build_safety_context
        from src.dashboard.auth import get_current_user

        ai_ctx = build_safety_context(
            sector_id=sector_id,
            date_str=str(date_str),
            worker_df=worker_df,
            space_df=space_df,
        )
        ai_commentary_box(
            role="safety_analyst",
            context=ai_ctx,
            sector_id=sector_id,
            date_str=str(date_str),
            title="안전성 AI 분석",
            spinner_text="위험도·고립작업 분석 중...",
            button_label="AI 분석 실행 (Haiku)",
            user_role=get_current_user().get("role", "unknown"),
            tab="safety",
            show_meta=False,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"AI 코멘터리 렌더 실패 (safety): {e}")

    # journey 로드 (세션 캐시 활용)
    _cache_key = f"journey_{sector_id}_{date_str}"
    journey_df = st.session_state.get(_cache_key)
    if journey_df is None:
        with st.spinner("journey 로드 중..."):
            journey_df = load_journey(date_str, sector_id)
        st.session_state[_cache_key] = journey_df

    render_safety(worker_df, space_df, locus_dict, date_str,
                  sector_id, has_cre, journey_df)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 2: 업체별
# ═══════════════════════════════════════════════════════════════════════

def _render_company_safety(sector_id: str, dates: list[str]) -> None:
    """업체별 안전 — SII/CRE 위험 작업자 비율."""
    st.markdown(section_header("업체별 안전성 분석"), unsafe_allow_html=True)

    if len(dates) == 1:
        selected_dates = dates
    else:
        s_idx, e_idx = st.select_slider(
            "분석 기간",
            options=list(range(len(dates))),
            value=(max(0, len(dates) - 14), len(dates) - 1),
            format_func=lambda i: dates[i],
            key="safety_company_range",
        )
        selected_dates = dates[s_idx:e_idx + 1]

    st.caption(f"선택 기간: {selected_dates[0]} ~ {selected_dates[-1]} "
               f"({len(selected_dates)}일)")

    wdf = _load_multi_day(sector_id, tuple(selected_dates))
    if wdf.empty:
        st.warning("데이터 없음.")
        return

    # ★ 안전 분석 정책 (감사 H3):
    # 생산성 탭은 `ewi_reliable==True` 필터 후 집계하지만,
    # 안전 분석에서는 신뢰도 낮은 작업자(음영 과다자)도 포함한다.
    # 이유: 고위험 징후를 놓치지 않기 위해 전수 집계. UI에 이 차이를 명시한다.
    total_workers = wdf["user_no"].nunique()
    reliable_workers = (
        wdf[wdf["ewi_reliable"] == True]["user_no"].nunique()
        if "ewi_reliable" in wdf.columns else total_workers
    )
    unreliable_pct = (1 - reliable_workers / total_workers) * 100 if total_workers else 0
    st.caption(
        f"📊 **집계 정책**: 안전 분석은 BLE 신뢰도 낮은 작업자도 포함합니다 "
        f"(전체 {total_workers:,}명 / 신뢰 {reliable_workers:,}명, "
        f"음영 과다 {unreliable_pct:.1f}%). "
        f"생산성 탭과 분모가 다를 수 있습니다."
    )

    # 위험 플래그 — constants.py의 CRE_HIGH, SII_HIGH 기준
    wdf["is_high_sii"] = (wdf["sii"].fillna(0) >= SII_HIGH).astype(int)
    wdf["is_high_cre"] = (wdf["cre"].fillna(0) >= CRE_HIGH).astype(int)
    wdf["is_confined"] = (wdf["confined_minutes"].fillna(0) >= 60).astype(int)
    wdf["is_hv"]       = (wdf["high_voltage_minutes"].fillna(0) >= 60).astype(int)

    agg = (
        wdf.groupby("company_name")
        .agg(
            worker_count=("user_no", "nunique"),
            avg_sii=("sii", "mean"),
            avg_cre=("cre", "mean"),
            n_high_sii=("is_high_sii", "sum"),
            n_high_cre=("is_high_cre", "sum"),
            n_confined=("is_confined", "sum"),
            n_hv=("is_hv", "sum"),
        )
        .reset_index()
        .query(f"worker_count >= {COMPANY_MIN_WORKERS['statistical']}")
    )

    if agg.empty:
        st.info("집계 가능한 업체가 없습니다.")
        return

    # 고위험 작업자 비율
    agg["high_risk_rate"] = (agg["n_high_sii"] + agg["n_high_cre"]) / (2 * agg["worker_count"]) * 100
    agg = agg.sort_values("high_risk_rate", ascending=False)

    # ── 주의 업체 Top 15 ─────────────────────────────────────────
    st.markdown(sub_header("⚠ 안전 주의 업체 Top 15 (고위험 작업자 비율 기준)"),
                unsafe_allow_html=True)
    top15 = agg.head(15)
    fig = go.Figure(go.Bar(
        x=top15["high_risk_rate"],
        y=top15["company_name"],
        orientation="h",
        marker_color=COLOR_DANGER,
        text=[f"{v:.1f}%" for v in top15["high_risk_rate"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        customdata=np.stack([top15["avg_sii"], top15["avg_cre"],
                             top15["worker_count"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "고위험 비율: %{x:.1f}%<br>"
            "SII: %{customdata[0]:.3f}<br>"
            "CRE: %{customdata[1]:.3f}<br>"
            "작업자: %{customdata[2]}명<extra></extra>"
        ),
    ))
    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=200, r=80, t=20, b=20)},
        height=max(420, len(top15) * 28 + 60),
        xaxis=dict(title="고위험 작업자 비율 (%)", gridcolor=COLORS["grid"],
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 원본 테이블 ──────────────────────────────────────────────
    with st.expander("📋 업체별 안전 지표 원본 데이터"):
        st.dataframe(
            agg[["company_name", "worker_count", "avg_sii", "avg_cre",
                 "n_high_sii", "n_high_cre", "n_confined", "n_hv",
                 "high_risk_rate"]].round(2),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 3: 공간별 (혼잡도 재활용)
# ═══════════════════════════════════════════════════════════════════════

def _render_space_safety(sector_id: str) -> None:
    """공간별 안전 — legacy congestion 뷰 재활용."""
    from src.dashboard._legacy.congestion_tab import render_congestion_tab
    render_congestion_tab(sector_id)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 4: 기간별
# ═══════════════════════════════════════════════════════════════════════

def _render_period_safety(sector_id: str) -> None:
    """기간별 안전 지표 추이 + 요일별 패턴."""
    st.markdown(section_header("전체 기간 안전 지표 추이"), unsafe_allow_html=True)

    from src.pipeline.summary_index import load_summary_index
    idx = load_summary_index(sector_id)
    dates_idx = idx.get("dates", {})

    if not dates_idx:
        st.info("summary_index 데이터가 없습니다.")
        return

    rows = []
    for d, v in sorted(dates_idx.items()):
        rows.append({
            "date":         d,
            "avg_cre":      v.get("avg_cre", 0),
            "avg_sii":      v.get("avg_sii", 0),
            "avg_fatigue":  v.get("avg_fatigue", 0),
            "high_cre":     v.get("high_cre_count", 0),
            "high_fatigue": v.get("high_fatigue_count", 0),
        })
    df = pd.DataFrame(rows)
    df["date_label"]  = df["date"].apply(format_date_label)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["dow"]         = df["date_parsed"].dt.dayofweek

    # 추이 (라인)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_cre"],
        name="CRE", mode="lines+markers",
        line=dict(color=COLOR_CRE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_sii"],
        name="SII", mode="lines+markers",
        line=dict(color=COLOR_SII, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["date_label"], y=df["avg_fatigue"],
        name="피로도", mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2, dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_DARK, height=320,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="지수", range=[0, 1], gridcolor=COLORS["grid"]),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 고위험 인원 추이 (바)
    st.markdown(sub_header("일자별 고위험 인원 수 (CRE ≥ 0.5 / 피로도 ≥ 0.5)"),
                unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["date_label"], y=df["high_cre"],
        name="고 CRE", marker_color=COLOR_CRE,
    ))
    fig2.add_trace(go.Bar(
        x=df["date_label"], y=df["high_fatigue"],
        name="고 피로도", marker_color=COLORS["danger"],
    ))
    fig2.update_layout(
        **PLOTLY_DARK, height=280, barmode="group",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="인원 (명)", gridcolor=COLORS["grid"]),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 요일별
    st.markdown(sub_header("요일별 평균 안전 지표"), unsafe_allow_html=True)
    dow_agg = df.groupby("dow").agg(
        avg_cre=("avg_cre", "mean"),
        avg_sii=("avg_sii", "mean"),
    ).reset_index()
    dow_agg["dow_name"] = dow_agg["dow"].map(lambda i: DOW_KR[i])

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=dow_agg["dow_name"], y=dow_agg["avg_cre"],
        name="CRE", marker_color=COLOR_CRE,
    ))
    fig3.add_trace(go.Bar(
        x=dow_agg["dow_name"], y=dow_agg["avg_sii"],
        name="SII", marker_color=COLOR_SII,
    ))
    fig3.update_layout(
        **PLOTLY_DARK, height=260, barmode="group",
        yaxis=dict(title="평균", range=[0, 1], gridcolor=COLORS["grid"]),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# 서브탭 5: 👤 개인별 — 위험 랭킹 + 배지 + 조치 권고
# ═══════════════════════════════════════════════════════════════════════

# 개인별 탭에서 필요한 worker 컬럼 (일별 전용 loader)
_INDIV_SAFETY_COLS = [
    "user_no", "user_name", "company_name", "shift_type",
    "work_minutes", "work_zone_minutes",
    "cre", "sii", "static_risk",
    "confined_minutes", "high_voltage_minutes",
    "alone_ratio", "fatigue_score", "max_active_streak",
    "helmet_abandoned", "ewi", "ewi_reliable",
    "ble_coverage_pct",
]


def _read_parquet_projected_safety(path, desired: list[str]) -> pd.DataFrame:
    """존재하는 컬럼만 projection하여 읽기."""
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
def _load_daily_safety(sector_id: str, date_str: str) -> pd.DataFrame:
    """개인별 안전 탭용 — worker.parquet 일일 로드 (컬럼 프루닝 적용)."""
    paths = cfg.get_sector_paths(sector_id)
    p = paths["processed_dir"] / date_str / "worker.parquet"
    if not p.exists():
        return pd.DataFrame()
    return _read_parquet_projected_safety(p, _INDIV_SAFETY_COLS)


# 정렬 옵션 (상수 — 매직 문자열 방지)
_SORT_OPTIONS_SAFETY: dict[str, tuple[str, bool]] = {
    "CRE 높은 순 (누적 노출)":    ("cre",                  False),
    "SII 높은 순 (안전 이슈)":    ("sii",                  False),
    "밀폐공간 노출 긴 순":        ("confined_minutes",     False),
    "고압 노출 긴 순":             ("high_voltage_minutes", False),
    "고립 작업 비율 높은 순":      ("alone_ratio",          False),
    "피로도 높은 순":              ("fatigue_score",        False),
    "정적 위험 높은 순":           ("static_risk",          False),
    "연속 고활성 긴 순 (과로)":   ("max_active_streak",    False),
    "헬멧 방치 의심 순":           ("helmet_abandoned",     False),
}

# 위험 레벨 라벨
_RISK_LABELS = ["🟢 저위험", "🟡 주의", "🟠 고위험", "🔴 심각"]
_RISK_LEVEL_TO_COLOR = {
    "🟢 저위험": CHART_COLORS["rest"],
    "🟡 주의":   CHART_COLORS["medium"],
    "🟠 고위험": CHART_COLORS["high"],
    "🔴 심각":   CHART_COLORS["critical"],
}


def _compute_risk_level(df: pd.DataFrame) -> pd.Series:
    """작업자별 위험 레벨 분류 (vectorized).

    - 심각(🔴): CRE > 0.7 OR SII > 0.7 OR confined >= 120 OR helmet_abandoned
    - 고위험(🟠): CRE > 0.5 OR SII > 0.5 OR alone_ratio > 0.7
    - 주의(🟡): CRE > 0.3 OR static_risk > 0.5
    - 저위험(🟢): 나머지
    """
    cre = pd.to_numeric(df.get("cre", 0), errors="coerce").fillna(0)
    sii = pd.to_numeric(df.get("sii", 0), errors="coerce").fillna(0)
    conf = pd.to_numeric(df.get("confined_minutes", 0), errors="coerce").fillna(0)
    alone = pd.to_numeric(df.get("alone_ratio", 0), errors="coerce").fillna(0)
    sr = pd.to_numeric(df.get("static_risk", 0), errors="coerce").fillna(0)
    helmet = df.get("helmet_abandoned", False)
    if not isinstance(helmet, pd.Series):
        helmet = pd.Series([False] * len(df), index=df.index)
    helmet = helmet.fillna(False).astype(bool)

    critical = (cre > 0.7) | (sii > 0.7) | (conf >= 120) | helmet
    high     = (cre > 0.5) | (sii > 0.5) | (alone > 0.7)
    medium   = (cre > 0.3) | (sr > 0.5)

    level = pd.Series(["🟢 저위험"] * len(df), index=df.index)
    level = level.where(~medium, "🟡 주의")
    level = level.where(~high, "🟠 고위험")
    level = level.where(~critical, "🔴 심각")
    return level


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


def _safety_individual_badges(row: pd.Series) -> list[str]:
    """작업자 경고 배지 목록 반환."""
    badges: list[str] = []
    helmet_raw = row.get("helmet_abandoned")
    try:
        helmet = False if helmet_raw is None or pd.isna(helmet_raw) else bool(helmet_raw)
    except (TypeError, ValueError):
        helmet = bool(helmet_raw) if helmet_raw is not None else False
    if helmet:
        badges.append("⚠ 헬멧방치")
    cre = _nan_safe_float(row.get("cre"))
    sii = _nan_safe_float(row.get("sii"))
    conf = _nan_safe_float(row.get("confined_minutes"))
    alone = _nan_safe_float(row.get("alone_ratio"))

    if cre > 0.7 or sii > 0.7:
        badges.append("🔴 심각")
    elif cre > 0.5 or sii > 0.5:
        badges.append("🟠 고위험")
    if alone >= 0.7:
        badges.append("🧑 고립잦음")
    if conf >= 120:
        badges.append("🏭 밀폐누적")
    return badges


def _safety_action_recommendations(row: pd.Series) -> list[str]:
    """작업자별 조치 권고 — 규칙 기반."""
    recs: list[str] = []
    conf = _nan_safe_float(row.get("confined_minutes"))
    hv = _nan_safe_float(row.get("high_voltage_minutes"))
    alone = _nan_safe_float(row.get("alone_ratio"))
    work_min = _nan_safe_float(row.get("work_minutes"))
    streak = _nan_safe_float(row.get("max_active_streak"))
    helmet_raw = row.get("helmet_abandoned")
    try:
        helmet = False if helmet_raw is None or pd.isna(helmet_raw) else bool(helmet_raw)
    except (TypeError, ValueError):
        helmet = bool(helmet_raw) if helmet_raw is not None else False

    if conf >= 120:
        recs.append(f"🏭 밀폐 공간 누적 {conf:.0f}분 — 10분 이상 휴식 및 환기 권고.")
    if hv >= 60:
        recs.append(f"⚡ 고압 구역 누적 {hv:.0f}분 — 2인 1조 확인 및 안전장비 점검 필요.")
    if alone >= 0.7 and work_min >= 240:
        recs.append(f"🧑 고립 작업 비율 {alone:.0%} — 2인 1조 배정 재검토 권고.")
    if streak > 180:
        recs.append(f"⚠ 연속 고활성 {streak:.0f}분 — 피로 누적 징후, 휴식 배정 권고.")
    if helmet:
        recs.append("🪖 헬멧 방치 의심 — 운영팀 확인 필요 (작업공간 + 저활성 30분+).")
    if not recs:
        recs.append("✅ 특별 조치 권고 사항 없음.")
    return recs


def _render_individual_safety_subtab(
    sector_id: str, date_str: str, shift_filter: str | None
) -> None:
    """개인별 안전 랭킹 + 상세.

    현장 안전관리자가 "CRE 높은 사람", "밀폐공간 누적자", "헬멧 방치 의심자" 를
    즉시 식별할 수 있게 한다.
    """
    st.markdown(section_header("개인별 안전 위험 랭킹"), unsafe_allow_html=True)

    from src.pipeline.cache_manager import SchemaVersionMismatch, validate_schema
    from src.dashboard.components import handle_schema_mismatch

    try:
        validate_schema(date_str, sector_id, strict_legacy=False)
        wdf = _load_daily_safety(sector_id, date_str)
    except SchemaVersionMismatch as exc:
        handle_schema_mismatch(exc, sector_id, date_str)
        return

    if wdf.empty:
        st.warning("해당 날짜의 worker 데이터가 없습니다.")
        return

    # 교대 상단 필터 반영
    if shift_filter and "shift_type" in wdf.columns:
        wdf = wdf[wdf["shift_type"] == shift_filter]

    if wdf.empty:
        st.info("선택한 교대에 해당하는 작업자가 없습니다.")
        return

    # ── 위험 레벨 분류 (vectorized, 전체 대상) ────────────────────
    wdf = wdf.copy()
    wdf["risk_level"] = _compute_risk_level(wdf)

    # ── 필터 영역 ─────────────────────────────────────────────────
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
            key="safety_indiv_company",
        )

    with fc2:
        shift_options_sub = {"전체": None, "주간": "day",
                             "야간": "night", "연장야간": "extended_night"}
        shift_label_sub = st.selectbox(
            "교대 (서브)",
            list(shift_options_sub.keys()),
            index=0,
            key="safety_indiv_shift",
        )
        shift_sub = shift_options_sub[shift_label_sub]

    with fc3:
        uno_search = st.text_input(
            "User_no 검색",
            value="",
            placeholder="예) 12345",
            key="safety_indiv_uno",
        )

    with fc4:
        min_work = st.slider(
            "최소 체류(분)",
            min_value=0, max_value=480, value=30, step=15,
            key="safety_indiv_minwork",
            help="짧은 체류를 제외합니다.",
        )

    fc5, fc6 = st.columns([3, 3])
    with fc5:
        selected_risks = st.multiselect(
            "위험 레벨 필터",
            options=_RISK_LABELS,
            default=[],
            placeholder="전체 (미선택 시 전체)",
            key="safety_indiv_risk",
        )
    with fc6:
        sort_key = st.selectbox(
            "정렬 기준",
            options=list(_SORT_OPTIONS_SAFETY.keys()),
            index=0,
            key="safety_indiv_sort",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 필터 적용 ───────────────────────────────────────────────────
    filt = wdf.copy()
    if selected_companies:
        filt = filt[filt["company_name"].isin(selected_companies)]
    if shift_sub and "shift_type" in filt.columns:
        filt = filt[filt["shift_type"] == shift_sub]
    uno_q = uno_search.strip()
    if uno_q:
        filt = filt[filt["user_no"].astype(str).str.contains(uno_q, na=False)]
    filt = filt[pd.to_numeric(filt["work_minutes"], errors="coerce").fillna(0) >= min_work]
    if selected_risks:
        filt = filt[filt["risk_level"].isin(selected_risks)]

    if filt.empty:
        st.warning("조건에 맞는 작업자가 없습니다. 필터를 조정해주세요.")
        return

    # ── 정렬 ────────────────────────────────────────────────────────
    sort_col, ascending = _SORT_OPTIONS_SAFETY[sort_key]
    if sort_col in filt.columns:
        sort_series = filt[sort_col]
        # helmet_abandoned 는 bool → 내림차순(True 가 위) 처리
        if sort_col == "helmet_abandoned":
            sort_series = sort_series.fillna(False).astype(int)
            filt = filt.assign(_sort=sort_series).sort_values(
                "_sort", ascending=ascending, na_position="last"
            ).drop(columns=["_sort"])
        else:
            filt = filt.sort_values(sort_col, ascending=ascending, na_position="last")
    filt = filt.reset_index(drop=True)

    # ── KPI 4개 ────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    n_workers = len(filt)
    n_critical = int((filt["risk_level"] == "🔴 심각").sum())
    avg_cre = pd.to_numeric(filt["cre"], errors="coerce").mean() or 0
    avg_sii = pd.to_numeric(filt["sii"], errors="coerce").mean() or 0
    n_helmet = int(filt["helmet_abandoned"].fillna(False).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(metric_card("대상 작업자", f"{n_workers:,}명",
                                color=CHART_COLORS["info"]), unsafe_allow_html=True)
    with k2:
        st.markdown(
            metric_card("🔴 심각 위험", f"{n_critical:,}명",
                        color=CHART_COLORS["critical"]),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            metric_card("평균 CRE / SII",
                        f"{avg_cre:.3f} / {avg_sii:.3f}",
                        color=CHART_COLORS["cre"]),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            metric_card("헬멧 방치 플래그", f"{n_helmet:,}명",
                        color=CHART_COLORS["fatigue"]),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 랭킹 테이블 ─────────────────────────────────────────────────
    show_all = st.toggle(
        f"전체 보기 ({n_workers:,}명)",
        value=False,
        key="safety_indiv_show_all",
        help="끄면 Top 50만 표시. 1만명 데이터에서도 즉각 반응합니다.",
    )
    top_n = n_workers if show_all else min(50, n_workers)
    rank_df = filt.head(top_n).copy()

    rank_df["순위"] = np.arange(1, len(rank_df) + 1)
    rank_df["작업자"] = rank_df["user_name"].astype(str) + " (" + rank_df["user_no"].astype(str) + ")"
    rank_df["CRE"] = pd.to_numeric(rank_df["cre"], errors="coerce").round(3)
    rank_df["SII"] = pd.to_numeric(rank_df["sii"], errors="coerce").round(3)
    rank_df["밀폐(분)"] = pd.to_numeric(rank_df["confined_minutes"], errors="coerce").fillna(0).round(0).astype(int)
    rank_df["고압(분)"] = pd.to_numeric(rank_df["high_voltage_minutes"], errors="coerce").fillna(0).round(0).astype(int)
    rank_df["고립 비율"] = (pd.to_numeric(rank_df["alone_ratio"], errors="coerce").fillna(0) * 100).round(1)
    rank_df["피로도"] = pd.to_numeric(rank_df.get("fatigue_score", 0), errors="coerce").round(3)
    rank_df["헬멧방치"] = rank_df["helmet_abandoned"].fillna(False).map(
        lambda v: "⚠" if bool(v) else "—"
    )

    disp = rank_df[[
        "순위", "작업자", "company_name", "shift_type", "risk_level",
        "CRE", "SII", "밀폐(분)", "고압(분)", "고립 비율", "피로도", "헬멧방치",
    ]].rename(columns={
        "company_name": "업체",
        "shift_type":   "교대",
        "risk_level":   "위험 레벨",
    })

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순위":      st.column_config.NumberColumn(format="#%d", width="small"),
            "CRE":       st.column_config.ProgressColumn(
                             "CRE", min_value=0.0, max_value=1.0, format="%.3f"),
            "SII":       st.column_config.ProgressColumn(
                             "SII", min_value=0.0, max_value=1.0, format="%.3f"),
            "피로도":    st.column_config.ProgressColumn(
                             "피로도", min_value=0.0, max_value=1.0, format="%.3f"),
            "고립 비율": st.column_config.ProgressColumn(
                             "고립 비율", min_value=0.0, max_value=100.0, format="%.1f%%"),
        },
    )

    # ── 상위 20명 경고 배지 요약 ───────────────────────────────────
    preview = rank_df.head(20).copy()
    preview["배지_list"] = preview.apply(_safety_individual_badges, axis=1)
    has_badges = preview[preview["배지_list"].map(len) > 0]
    if not has_badges.empty:
        st.markdown(sub_header("⚠ 경고 대상 (상위 20명 중)"), unsafe_allow_html=True)
        for _, r in has_badges.iterrows():
            badge_html = " ".join([
                f"<span style='background:{CHART_COLORS['critical']}22;"
                f"color:{CHART_COLORS['critical']};padding:2px 8px;border-radius:10px;"
                f"font-size:0.74rem;font-weight:600;"
                f"border:1px solid {CHART_COLORS['critical']}66;'>{b}</span>"
                for b in r["배지_list"]
            ])
            st.markdown(
                f"<div style='padding:6px 10px;margin:3px 0;background:{COLORS['bg_chart']};"
                f"border-left:3px solid {CHART_COLORS['critical']};border-radius:6px;'>"
                f"<span style='color:{COLORS['text']};font-weight:600;'>"
                f"#{int(r['순위'])} {r['작업자']}</span> "
                f"<span style='color:{COLORS['text_muted']};font-size:0.82rem;'>· {r.get('company_name','')}</span>"
                f"&nbsp;&nbsp;{badge_html}</div>",
                unsafe_allow_html=True,
            )

    # ── 상세 확장 ──────────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.expander("🔍 개인 상세 분석 (작업자 선택)", expanded=False):
        pool = filt.head(200).copy()
        labels = [
            f"#{i+1}  {row['user_name']} ({row['user_no']}) · {row.get('company_name','')} · "
            f"CRE {_nan_safe_float(row.get('cre')):.3f} · {row['risk_level']}"
            for i, (_, row) in enumerate(pool.iterrows())
        ]
        label_to_uno = dict(zip(labels, pool["user_no"].tolist()))
        if not labels:
            st.info("상세 대상이 없습니다.")
            return

        sel_label = st.selectbox(
            f"작업자 선택 (상위 {len(labels)}명 중)",
            labels,
            key="safety_indiv_detail_select",
        )
        sel_uno = label_to_uno[sel_label]
        row = pool[pool["user_no"] == sel_uno].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(metric_card("CRE", f"{_nan_safe_float(row.get('cre')):.3f}",
                                    color=CHART_COLORS["cre"]), unsafe_allow_html=True)
        with d2:
            st.markdown(metric_card("SII", f"{_nan_safe_float(row.get('sii')):.3f}",
                                    color=CHART_COLORS["sii"]), unsafe_allow_html=True)
        with d3:
            st.markdown(metric_card("위험 레벨", str(row["risk_level"]),
                                    color=_RISK_LEVEL_TO_COLOR.get(row["risk_level"],
                                                                   CHART_COLORS["info"])),
                        unsafe_allow_html=True)
        with d4:
            st.markdown(metric_card("피로도",
                                    f"{_nan_safe_float(row.get('fatigue_score')):.3f}",
                                    color=CHART_COLORS["fatigue"]), unsafe_allow_html=True)

        # 노출 시간 막대 (밀폐/고압/고립)
        conf_m = _nan_safe_float(row.get("confined_minutes"))
        hv_m   = _nan_safe_float(row.get("high_voltage_minutes"))
        work_m = _nan_safe_float(row.get("work_minutes"))
        alone_min = _nan_safe_float(row.get("alone_ratio")) * work_m

        exposure_names  = ["밀폐공간", "고압구역", "고립작업 (추정)"]
        exposure_values = [conf_m, hv_m, alone_min]
        exposure_colors = [
            CHART_COLORS["confined_space"],
            CHART_COLORS["high_voltage"],
            CHART_COLORS["low_active"],
        ]
        fig_exp = go.Figure()
        for lbl, v, c in zip(exposure_names, exposure_values, exposure_colors):
            fig_exp.add_trace(go.Bar(
                name=lbl,
                x=[v], y=[lbl],
                orientation="h",
                marker_color=c,
                text=f"{v:.0f}분" if v > 0 else "",
                textposition="outside",
                textfont=dict(color=COLORS["text"], size=11),
                hovertemplate=f"{lbl}: %{{x:.0f}}분<extra></extra>",
            ))
        fig_exp.update_layout(
            **{**PLOTLY_DARK, "margin": dict(l=100, r=40, t=30, b=20)},
            title="위험 노출 시간 (단위: 분)",
            height=240,
            xaxis=dict(title="분", gridcolor=COLORS["grid"]),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        # 보조 정보
        helmet_raw = row.get("helmet_abandoned")
        try:
            helmet_flag = False if helmet_raw is None or pd.isna(helmet_raw) else bool(helmet_raw)
        except (TypeError, ValueError):
            helmet_flag = bool(helmet_raw) if helmet_raw is not None else False
        info_rows = [
            ("업체",          str(row.get("company_name") or "-")),
            ("교대",          str(row.get("shift_type") or "-")),
            ("총 체류시간",   f"{work_m:.0f} 분"),
            ("작업공간(분)",  f"{_nan_safe_float(row.get('work_zone_minutes')):.0f}"),
            ("정적 위험",     f"{_nan_safe_float(row.get('static_risk')):.3f}"),
            ("연속 고활성 최대",
                              f"{int(_nan_safe_float(row.get('max_active_streak'))):,} 분"),
            ("BLE 커버리지",  f"{_nan_safe_float(row.get('ble_coverage_pct')):.1f}%"),
            ("헬멧 방치",     "⚠ 플래그" if helmet_flag else "—"),
        ]
        info_df = pd.DataFrame(info_rows, columns=["항목", "값"])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

        # 조치 권고
        st.markdown(sub_header("📋 조치 권고"), unsafe_allow_html=True)
        for rec in _safety_action_recommendations(row):
            st.markdown(
                f"<div style='padding:8px 12px;margin:4px 0;background:{COLORS['bg_chart']};"
                f"border-left:3px solid {CHART_COLORS['info']};border-radius:6px;"
                f"color:{COLORS['text']};font-size:0.9rem;'>{rec}</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════

def render_safety_tab(sector_id: str) -> None:
    """안전성 분석 탭 메인."""
    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    dates_asc = list(processed)

    col_date, col_shift = st.columns([1, 2])
    with col_date:
        date_str = get_date_selector(
            list(reversed(processed)),
            key="safety_daily_date",
            default_index=0,
            label="일별 분석 날짜",
            show_label=True,
        ) or processed[-1]

    with col_shift:
        shift_options = {"전체": None, "주간": "day",
                         "야간": "night", "연장야간": "extended_night"}
        shift_label = st.radio(
            "교대 필터 (일별 탭)",
            list(shift_options.keys()),
            horizontal=True,
            key="safety_shift_filter",
        )
        shift_filter = shift_options[shift_label]

    st.divider()

    t1, t2, t3, t4, t5 = st.tabs([
        "🛡️ 일별",
        "🏗️ 업체별",
        "🗺️ 공간별 (혼잡도)",
        "📅 기간별",
        "👤 개인별",
    ])

    with t1:
        _render_daily_safety(sector_id, date_str, shift_filter)

    with t2:
        _render_company_safety(sector_id, dates_asc)

    with t3:
        _render_space_safety(sector_id)

    with t4:
        _render_period_safety(sector_id)

    with t5:
        _render_individual_safety_subtab(sector_id, date_str, shift_filter)
