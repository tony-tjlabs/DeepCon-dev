"""
integrity/gap_analysis.py — BLE 커버리지 이상 서브탭
=====================================================
tab4: `📉 BLE 커버리지 이상`. 타각기 체류 대비 BLE 수신률이
낮은 작업자를 추출해 카드 목록으로 나열. BLE 인프라 사각지대·
T-Ward 비착용·배터리 방전 등의 원인 진단 도구.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    COLORS,
    PLOTLY_DARK,
    metric_card,
    section_header,
    sub_header,
)
from src.dashboard.date_utils import get_date_selector
from src.dashboard.integrity.context import _load_worker

logger = logging.getLogger(__name__)


def _render_ble_coverage_gap(sector_id: str) -> None:
    """BLE 커버리지 이상 — 타각기 체류 대비 BLE 미수신 작업자 목록.

    타각기(AccessLog) 기준 출입 시간은 길지만,
    BLE 커버리지(journey.parquet 기록 / work_minutes)가 낮은 작업자를 추출.
    BLE 인프라 사각지대·T-Ward 비착용·배터리 방전 등 원인 파악에 활용.
    """
    from src.pipeline.cache_manager import detect_processed_dates

    st.markdown(section_header("BLE 커버리지 이상 — 타각기 대비 미수신 작업자"), unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7A8FA6; font-size:0.83rem; margin-bottom:16px;'>"
        "타각기 체류시간 대비 BLE 신호가 충분히 수신되지 않은 작업자를 추출합니다. "
        "BLE 음영 구역·T-Ward 비착용·배터리 방전이 주요 원인입니다.</p>",
        unsafe_allow_html=True,
    )

    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    paths = cfg.get_sector_paths(sector_id)

    # ── 1) 날짜 + 커버리지 임계값 설정 ──────────────────────────────────
    col_date, col_thresh, col_min_stay = st.columns([1, 1, 1])
    with col_date:
        date_str = get_date_selector(
            list(reversed(processed)),
            key="ble_gap_date",
            default_index=0,
            label="검토 날짜",
            show_label=True,
        ) or processed[-1]

    with col_thresh:
        threshold_pct = st.select_slider(
            "BLE 커버리지 임계값 (%)",
            options=[10, 20, 30, 40, 50, 60, 70, 80],
            value=50,
            key="ble_gap_threshold",
            help="이 값 이하인 작업자를 '커버리지 이상'으로 분류합니다.",
        )

    with col_min_stay:
        min_stay_min = st.number_input(
            "최소 타각기 체류 (분)",
            min_value=30, max_value=600, value=120, step=30,
            key="ble_gap_min_stay",
            help="이 시간 이상 체류한 작업자만 대상으로 합니다. (단시간 방문자 제외)",
        )

    # ── 2) worker.parquet 로드 ───────────────────────────────────────────
    worker_path = paths["processed_dir"] / date_str / "worker.parquet"
    if not worker_path.exists():
        st.error("해당 날짜의 worker.parquet가 없습니다.")
        return

    wdf = _load_worker(sector_id, date_str, str(worker_path))
    if wdf.empty:
        st.warning("작업자 데이터가 없습니다.")
        return

    # ble_coverage_pct 컬럼 확인
    if "ble_coverage_pct" not in wdf.columns:
        st.warning("worker.parquet에 ble_coverage_pct 컬럼이 없습니다. (파이프라인 재실행 필요)")
        return

    # ── 3) 필터링 ────────────────────────────────────────────────────────
    wdf["ble_coverage_pct"] = pd.to_numeric(wdf["ble_coverage_pct"], errors="coerce").fillna(0)
    wdf["work_minutes"]     = pd.to_numeric(wdf["work_minutes"],     errors="coerce").fillna(0)

    # 타각기 기준 최소 체류 + 커버리지 임계값 이하
    gap_df = wdf[
        (wdf["work_minutes"] >= min_stay_min) &
        (wdf["ble_coverage_pct"] <= threshold_pct)
    ].copy()

    gap_df = gap_df.sort_values("ble_coverage_pct", ascending=True).reset_index(drop=True)

    # ── 4) KPI 요약 ───────────────────────────────────────────────────────
    n_total  = len(wdf[wdf["work_minutes"] >= min_stay_min])
    n_gap    = len(gap_df)
    gap_rate = n_gap / n_total * 100 if n_total > 0 else 0

    # 커버리지 분포 buckets — pd.cut() 벡터화
    _COV_BINS   = [0, 10, 30, 50, 70, 100]
    _COV_LABELS = ["0~10%", "11~30%", "31~50%", "51~70%", "71~100%"]

    wdf_stay = wdf[wdf["work_minutes"] >= min_stay_min].copy()
    wdf_stay["cov_bucket"] = pd.cut(
        wdf_stay["ble_coverage_pct"],
        bins=_COV_BINS,
        labels=_COV_LABELS,
        include_lowest=True,
    ).astype(str)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("전체 대상", f"{n_total:,}명",
                                delta=f"체류 {min_stay_min}분 이상"),
                    unsafe_allow_html=True)
    with col2:
        gap_color = COLORS["danger"] if gap_rate >= 20 else COLORS["warning"] if gap_rate >= 10 else COLORS["success"]
        st.markdown(metric_card("커버리지 이상", f"{n_gap:,}명",
                                delta=f"{threshold_pct}% 이하",
                                color=gap_color),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("이상 비율", f"{gap_rate:.1f}%",
                                color=gap_color),
                    unsafe_allow_html=True)
    with col4:
        avg_cov = float(gap_df["ble_coverage_pct"].mean()) if not gap_df.empty else 0
        st.markdown(metric_card("이상군 평균 커버리지", f"{avg_cov:.1f}%"),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if gap_df.empty:
        st.success(f"BLE 커버리지 {threshold_pct}% 이하 작업자가 없습니다. 👍")
        return

    # ── 5) 커버리지 분포 차트 ───────────────────────────────────────────
    with st.expander("📊 전체 커버리지 분포 (히스토그램)", expanded=False):
        bucket_order = ["0~10%", "11~30%", "31~50%", "51~70%", "71~100%"]
        bucket_colors = {
            "0~10%":   "#FF4C4C",
            "11~30%":  "#FF8C42",
            "31~50%":  "#FFB300",
            "51~70%":  "#A0C8FF",
            "71~100%": "#00C897",
        }
        bucket_cnt = wdf_stay["cov_bucket"].value_counts().reindex(bucket_order, fill_value=0)
        fig_hist = go.Figure(go.Bar(
            x=bucket_order,
            y=bucket_cnt.values,
            marker_color=[bucket_colors[b] for b in bucket_order],
            text=bucket_cnt.values,
            textposition="outside",
        ))
        fig_hist.update_layout(**{
            **PLOTLY_DARK,
            "height": 300,
            "margin": dict(t=20, b=40, l=40, r=20),
            "xaxis_title": "BLE 커버리지 구간",
            "yaxis_title": "작업자 수 (명)",
        })
        st.plotly_chart(fig_hist, use_container_width=True, key="plotly_15")

    # ── 6) 이상 작업자 목록 ─────────────────────────────────────────────
    st.markdown(sub_header(f"커버리지 {threshold_pct}% 이하 작업자 목록 ({n_gap:,}명)"),
                unsafe_allow_html=True)

    # 업체 필터
    companies_gap = ["전체"] + sorted(gap_df["company_name"].dropna().unique().tolist())
    col_cf, col_sort = st.columns([2, 1])
    with col_cf:
        sel_company = st.selectbox(
            "업체 필터",
            options=companies_gap,
            index=0,
            key="ble_gap_company",
        )
    with col_sort:
        sort_by = st.selectbox(
            "정렬 기준",
            options=["BLE 커버리지 ↑ (낮은순)", "체류시간 ↓ (긴순)", "업체명 ↑"],
            index=0,
            key="ble_gap_sort",
        )

    filtered = gap_df.copy()
    if sel_company != "전체":
        filtered = filtered[filtered["company_name"] == sel_company]

    if sort_by == "BLE 커버리지 ↑ (낮은순)":
        filtered = filtered.sort_values("ble_coverage_pct", ascending=True)
    elif sort_by == "체류시간 ↓ (긴순)":
        filtered = filtered.sort_values("work_minutes", ascending=False)
    elif sort_by == "업체명 ↑":
        filtered = filtered.sort_values("company_name", na_position="last")

    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.info("해당 업체의 이상 작업자가 없습니다.")
        return

    # ── 7) 이상 작업자 카드 목록 ──────────────────────────────────────────
    def _fmt_min(m: float) -> str:
        """분 → 가독성 포맷 (X분 / Xh Ymin)."""
        m = int(round(m))
        if m <= 0:   return "0분"
        if m < 60:   return f"{m}분"
        h, rem = divmod(m, 60)
        return f"{h}h {rem}분" if rem else f"{h}h"

    for seq_num, (_, row) in enumerate(filtered.iterrows(), start=1):
        cov_pct      = float(row.get("ble_coverage_pct", 0) or 0)
        work_min     = float(row.get("work_minutes", 0) or 0)
        gap_min      = float(row.get("gap_min", 0) or 0)
        cov_lbl      = str(row.get("ble_coverage", "미측정") or "미측정")
        uname        = str(row.get("user_name", "") or "")
        user_no      = str(row.get("user_no", "") or "")
        company      = str(row.get("company_name", "") or "")
        shift        = str(row.get("shift_type", "") or "")
        ewi_rel      = bool(row.get("ewi_reliable", False))

        # BLE 수신 시간 추산
        ble_recv_min = work_min * cov_pct / 100.0

        # 커버리지 색상
        if cov_pct <= 10:
            cov_color = "#FF4C4C"
        elif cov_pct <= 30:
            cov_color = "#FF8C42"
        else:
            cov_color = "#FFB300"

        ewi_badge = (
            "<span style='background:#2A241A;color:#FFB300;"
            "padding:2px 5px;border-radius:3px;font-size:0.68rem;margin-left:6px;'>"
            "ⓘ EWI 저신뢰</span>"
            if not ewi_rel else ""
        )

        # 원인 추정
        causes = []
        if cov_pct <= 5:
            causes.append("BLE 인프라 미배치 구역 의심")
        if gap_min > work_min * 0.5:
            causes.append("gap_fill 과다 (음영 보정 50%+)")
        if cov_lbl == "미측정":
            causes.append("BLE 측정 자체 불가 환경")
        cause_txt = " · ".join(causes) if causes else "원인 불명 (직접 확인 필요)"

        # 진행 막대 너비 (최소 0.5% 보장 — 너무 작으면 안 보임)
        bar_w = max(0.5, min(100.0, cov_pct))

        # ── 카드 HTML ────────────────────────────────────────────────────
        card = f"""
<div style='background:#0D1B2A;border:1px solid #1E2E3E;
            border-radius:8px;padding:14px 18px;margin-bottom:8px;'>

  <!-- 헤더: 이름 / 번호 / 업체 / 비율 -->
  <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;'>
    <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
      <span style='color:#7A8FA6;font-size:0.75rem;'>#{seq_num}</span>
      <span style='color:#D5E5FF;font-weight:600;font-size:0.95rem;'>{uname}</span>
      <span style='color:#6A8AA6;font-size:0.8rem;font-family:monospace;'>({user_no})</span>
      <span style='color:#9AB5D4;font-size:0.82rem;'>{company[:30]}</span>
      {ewi_badge}
    </div>
    <span style='color:{cov_color};font-size:1.5rem;font-weight:700;
                 letter-spacing:-0.5px;white-space:nowrap;'>{cov_pct:.1f}%</span>
  </div>

  <!-- 3-박스: 타각기 체류 / BLE 수신 / 교대 -->
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px;'>
    <div style='background:#0A1520;border-radius:6px;padding:10px 12px;text-align:center;'>
      <div style='color:#7A8FA6;font-size:0.68rem;margin-bottom:4px;letter-spacing:0.3px;'>
        ⏱ 타각기 체류</div>
      <div style='color:#D5E5FF;font-size:1.1rem;font-weight:600;'>
        {_fmt_min(work_min)}</div>
      <div style='color:#5A7A9A;font-size:0.68rem;margin-top:2px;'>
        {work_min:.0f}분</div>
    </div>
    <div style='background:#0A1520;border-radius:6px;padding:10px 12px;text-align:center;'>
      <div style='color:#7A8FA6;font-size:0.68rem;margin-bottom:4px;letter-spacing:0.3px;'>
        📡 BLE 수신</div>
      <div style='color:{cov_color};font-size:1.1rem;font-weight:600;'>
        {_fmt_min(ble_recv_min)}</div>
      <div style='color:#5A7A9A;font-size:0.68rem;margin-top:2px;'>
        {ble_recv_min:.0f}분</div>
    </div>
    <div style='background:#0A1520;border-radius:6px;padding:10px 12px;text-align:center;'>
      <div style='color:#7A8FA6;font-size:0.68rem;margin-bottom:4px;letter-spacing:0.3px;'>
        📊 커버리지</div>
      <div style='color:{cov_color};font-size:1.1rem;font-weight:600;'>
        {cov_pct:.1f}%</div>
      <div style='color:#5A7A9A;font-size:0.68rem;margin-top:2px;'>
        {cov_lbl}</div>
    </div>
  </div>

  <!-- 커버리지 비율 진행 막대 -->
  <div style='background:#1A2A3A;border-radius:4px;height:6px;
              margin-bottom:10px;overflow:hidden;'>
    <div style='background:{cov_color};height:6px;width:{bar_w:.1f}%;
                border-radius:4px;transition:width 0.3s;'></div>
  </div>

  <!-- 교대 + 원인 -->
  <div style='display:flex;gap:16px;flex-wrap:wrap;align-items:center;'>
    <span style='color:#7A8FA6;font-size:0.73rem;'>교대&nbsp;
      <span style='color:#9AB5D4;'>{shift}</span></span>
    <span style='color:#7A8FA6;font-size:0.73rem;'>추정 원인&nbsp;
      <span style='color:#FFB300;font-size:0.73rem;'>{cause_txt}</span></span>
  </div>

</div>"""
        st.markdown(card, unsafe_allow_html=True)

    # ── 8) 다운로드 ─────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    dl_cols = [c for c in [
        "user_no", "user_name", "company_name", "shift_type",
        "work_minutes", "ble_coverage", "ble_coverage_pct",
        "gap_ratio", "gap_min", "ewi_reliable",
    ] if c in filtered.columns]
    csv_bytes = filtered[dl_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label=f"⬇ 커버리지 이상 {len(filtered):,}명 CSV 다운로드",
        data=csv_bytes,
        file_name=f"ble_coverage_gap_{date_str}_{threshold_pct}pct.csv",
        mime="text/csv",
        key="ble_gap_download",
    )
