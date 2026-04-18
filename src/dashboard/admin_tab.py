"""
시스템 관리 탭 — 파이프라인 + 방법론 + AI 감사 로그 (Admin 전용)
================================================================
3개 서브탭:
  1. ⚙️ 데이터 파이프라인 — 전처리 실행, 캐시 상태 (기존 pipeline_tab)
  2. 📚 방법론 — 알고리즘 설명 (기존 theory_tab)
  3. 🔐 AI 감사 로그 — LLM 호출 내역 (M2-B T-22, V-02 대응)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, CHART_COLORS, PLOTLY_DARK, section_header, sub_header

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# AI 감사 로그 뷰어 (T-22)
# ═══════════════════════════════════════════════════════════════════

def _render_ai_audit_log() -> None:
    """
    AI 감사 로그 뷰어.

    data/audit/{sector_id}/YYYY-MM.jsonl 를 읽어:
      - 월/Sector 필터
      - 호출 목록 (timestamp, role, tab, tokens, latency, PII)
      - PII 누출 감지된 행 빨간색 강조
      - Top 10 비싼 호출 (tokens_in 기준)
      - 일/주 집계 차트 (호출 수, 토큰 합)

    ⚠️ 감사 로그에는 원문이 없음 — SHA256 해시만.
       실제 프롬프트/응답 복원은 불가능 (설계 상 의도).
    """
    from core.ai import list_audit_months, read_audit_log

    st.markdown(section_header("🔐 AI 호출 감사 로그"), unsafe_allow_html=True)
    st.caption(
        "모든 LLM 호출은 `core.ai.LLMGateway` 를 통과하며 "
        "자동 기록됩니다. 원문은 저장되지 않고 SHA256 해시만 보관합니다 "
        "(보안 정책 V-02)."
    )

    # 기본 디렉토리
    base_dir = Path("data/audit")

    months = list_audit_months(base_dir=base_dir)
    if not months:
        st.info(
            "📭 감사 로그가 아직 없습니다.\n\n"
            "AI 기능을 한 번 이상 사용하면 여기에 기록이 쌓입니다.\n"
            f"저장 위치: `{base_dir}/SECTOR_ID/YYYY-MM.jsonl`"
        )
        return

    # 사용 가능한 Sector / YYYY-MM 목록
    sectors = sorted({s for s, _ in months})
    sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])
    with sel_col1:
        sel_sector = st.selectbox("Sector", sectors, key="audit_sel_sector")
    with sel_col2:
        sector_months = sorted({m for s, m in months if s == sel_sector}, reverse=True)
        sel_month = st.selectbox("월 (YYYY-MM)", sector_months, key="audit_sel_month")
    with sel_col3:
        limit_label = st.selectbox(
            "최근 N건",
            ["전체", "100", "500", "1000"],
            key="audit_limit",
        )
        limit = None if limit_label == "전체" else int(limit_label)

    entries = read_audit_log(sel_sector, sel_month, base_dir=base_dir, limit=limit)
    if not entries:
        st.info(f"📭 `{sel_sector}` / `{sel_month}` 로그가 비어있습니다.")
        return

    df = pd.DataFrame(entries)

    # ── 요약 KPI ────────────────────────────────────────────────
    total_calls = len(df)
    total_in = int(df["tokens_in"].sum())
    total_out = int(df["tokens_out"].sum())
    total_cache_read = int(df.get("cache_read_tokens", pd.Series([0])).sum())
    pii_leak_count = int(df["pii_leak_detected"].sum()) if "pii_leak_detected" in df.columns else 0
    blocked_count = int(df["blocked"].sum()) if "blocked" in df.columns else 0
    error_count = int(df["error"].notna().sum()) if "error" in df.columns else 0
    avg_latency = float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 호출", f"{total_calls:,}")
    c2.metric("입력 토큰", f"{total_in:,}")
    c3.metric("출력 토큰", f"{total_out:,}")
    c4.metric("평균 지연", f"{avg_latency/1000:.1f} s")
    pii_color = "inverse" if pii_leak_count > 0 else "off"
    c5.metric("PII 의심", f"{pii_leak_count}", delta=f"차단 {blocked_count}",
              delta_color=pii_color)

    if error_count > 0:
        st.warning(f"⚠️ 에러 기록 {error_count}건 — 아래 목록에서 `error` 컬럼 확인 필요")

    st.divider()

    # ── 일/주 집계 차트 ─────────────────────────────────────────
    if "timestamp" in df.columns:
        df["_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        daily = (
            df.groupby("_date")
            .agg(calls=("request_id", "count"),
                 tokens_in=("tokens_in", "sum"),
                 tokens_out=("tokens_out", "sum"))
            .reset_index()
        )
        if not daily.empty:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(sub_header("일별 호출 수"), unsafe_allow_html=True)
                fig1 = go.Figure(go.Bar(
                    x=[str(d) for d in daily["_date"]],
                    y=daily["calls"],
                    marker_color=CHART_COLORS["work_zone"],
                ))
                fig1.update_layout(**PLOTLY_DARK, height=240,
                                   margin=dict(l=40, r=10, t=20, b=40))
                st.plotly_chart(fig1, use_container_width=True)
            with col_r:
                st.markdown(sub_header("일별 토큰 합계 (in + out)"), unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=[str(d) for d in daily["_date"]],
                    y=daily["tokens_in"],
                    name="입력", marker_color=CHART_COLORS["access"],
                ))
                fig2.add_trace(go.Bar(
                    x=[str(d) for d in daily["_date"]],
                    y=daily["tokens_out"],
                    name="출력", marker_color=CHART_COLORS["tward"],
                ))
                fig2.update_layout(**PLOTLY_DARK, height=240, barmode="stack",
                                   margin=dict(l=40, r=10, t=20, b=40))
                st.plotly_chart(fig2, use_container_width=True)

    # ── Role 분포 ─────────────────────────────────────────────────
    if "role" in df.columns:
        st.markdown(sub_header("Role 별 호출 분포"), unsafe_allow_html=True)
        role_counts = df.groupby("role").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(role_counts, use_container_width=True, hide_index=True)

    # ── Top 10 비싼 호출 ────────────────────────────────────────
    st.markdown(sub_header("💰 Top 10 비싼 호출 (tokens_in 기준)"),
                unsafe_allow_html=True)
    top_cols = ["timestamp", "role", "tab", "tokens_in", "tokens_out",
                "latency_ms", "pii_leak_detected", "blocked", "error"]
    top_cols = [c for c in top_cols if c in df.columns]
    top_df = df.sort_values("tokens_in", ascending=False).head(10)[top_cols]
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # ── 전체 목록 (PII 누출 강조) ───────────────────────────────
    st.markdown(sub_header("📋 전체 호출 목록"), unsafe_allow_html=True)
    show_cols = ["timestamp", "user_role", "role", "tab", "date_str",
                 "tokens_in", "tokens_out", "cache_read_tokens",
                 "latency_ms", "pii_leak_detected", "blocked", "error",
                 "model", "request_id"]
    show_cols = [c for c in show_cols if c in df.columns]
    view_df = df[show_cols].copy()

    # PII 누출 감지 필터
    if "pii_leak_detected" in view_df.columns:
        show_only_pii = st.checkbox(
            "🚨 PII 의심 행만 표시",
            value=False,
            key="audit_only_pii",
        )
        if show_only_pii:
            view_df = view_df[view_df["pii_leak_detected"] == True]  # noqa: E712

    # 스타일링: PII 의심 행 빨간색
    def _style_pii_row(row):
        if row.get("pii_leak_detected", False):
            return ["background-color: rgba(239, 68, 68, 0.18)"] * len(row)
        if row.get("blocked", False):
            return ["background-color: rgba(239, 68, 68, 0.35)"] * len(row)
        if row.get("error"):
            return ["background-color: rgba(255, 179, 0, 0.18)"] * len(row)
        return [""] * len(row)

    try:
        styled = view_df.style.apply(_style_pii_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=420)
    except Exception:
        st.dataframe(view_df, use_container_width=True, hide_index=True, height=420)

    # ── JSONL 다운로드 ───────────────────────────────────────────
    st.caption("원본 JSONL 은 `data/audit/{sector}/YYYY-MM.jsonl` 에서 다운로드 가능합니다.")


# ═══════════════════════════════════════════════════════════════════
# 메인 진입점
# ═══════════════════════════════════════════════════════════════════

def render_admin_tab(sector_id: str) -> None:
    """시스템 관리 탭 메인."""
    t1, t2, t3 = st.tabs([
        "⚙️ 데이터 파이프라인",
        "📚 방법론",
        "🔐 AI 감사 로그",
    ])

    with t1:
        try:
            from src.dashboard.pipeline_tab import render_pipeline_tab
            render_pipeline_tab(sector_id)
        except ImportError:
            st.info("⚙️ 파이프라인 모듈은 로컬 환경에서만 실행 가능합니다.")
        except Exception as e:
            st.error(f"파이프라인 탭 로드 실패: {e}")
            logger.exception("pipeline_tab 에러")

    with t2:
        try:
            from src.dashboard._legacy.theory_tab import render_theory_tab
            render_theory_tab()
        except Exception as e:
            st.error(f"방법론 탭 로드 실패: {e}")
            logger.exception("theory_tab 에러")

    with t3:
        try:
            _render_ai_audit_log()
        except Exception as e:
            st.error(f"AI 감사 로그 뷰어 로드 실패: {e}")
            logger.exception("audit_log 뷰어 에러")
