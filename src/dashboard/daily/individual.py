"""
Daily Tab - Individual Section
==============================
개인별 분석 탭 관련 함수들.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import (
    section_header, sub_header,
    PLOTLY_DARK, PLOTLY_LEGEND,
    CHART_COLORS,
)
from src.dashboard.components import journey_timeline
from src.utils.anonymizer import mask_name, mask_names_in_df

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def render_individual(worker_df, locus_dict, has_ewi, has_cre):
    """개인별 분석 탭 전체 렌더링."""
    st.markdown(section_header("개인별 분석"), unsafe_allow_html=True)

    # 검색 + 정렬
    col_search, col_sort = st.columns([3, 2])
    with col_search:
        search = st.text_input("작업자 이름 또는 업체 검색", placeholder="예: 홍길동, A건설")
    with col_sort:
        sort_options = {"BLE 기록 많은 순": "valid_ble_minutes",
                        "EWI 높은 순": "ewi", "CRE 높은 순": "cre",
                        "근무시간 긴 순": "work_minutes", "밀폐공간 긴 순": "confined_minutes"}
        sort_valid = {k: v for k, v in sort_options.items() if v in worker_df.columns}
        if sort_valid:
            sort_by_label = st.selectbox("정렬 기준", list(sort_valid.keys()))
            sort_col = sort_valid[sort_by_label]
        else:
            sort_col = "valid_ble_minutes" if "valid_ble_minutes" in worker_df.columns else "recorded_minutes"

    df = worker_df.copy()
    if search:
        mask = (
            df["user_name"].str.contains(search, case=False, na=False)
            | df["company_name"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        st.info("검색 결과 없음")
        return

    # ★ BLE 기록이 극소량인 작업자는 하단 배치 (의미없는 EWI=1.0 상위 노출 방지)
    MIN_BLE_MINUTES = 30
    ble_col = "valid_ble_minutes" if "valid_ble_minutes" in df.columns else "recorded_minutes"
    df["_sufficient_ble"] = df[ble_col].fillna(0) >= MIN_BLE_MINUTES
    df = df.sort_values(
        ["_sufficient_ble", sort_col],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Shift 필터 (shift_type 컬럼 있을 때)
    if "shift_type" in df.columns:
        shift_filter = st.radio(
            "근무 교대", ["전체", "sun 주간", "moon 야간"],
            horizontal=True, key="individual_shift_filter",
        )
        if shift_filter == "sun 주간":
            df = df[df["shift_type"] == "day"]
        elif shift_filter == "moon 야간":
            df = df[df["shift_type"] == "night"]

        # Shift 배지 컬럼 생성
        df = df.copy()
        df["shift"] = df["shift_type"].map({"day": "sun", "night": "moon", "unknown": "?"}).fillna("?")

        # 헬멧 방치 의심 플래그
        if "helmet_abandoned" in df.columns:
            df["헬멧방치?"] = df["helmet_abandoned"].map({True: "!", False: ""})

    if df.empty:
        st.info("해당 교대의 작업자 없음")
        return

    # BLE 커버리지 등급 컬럼 추가
    if "ble_coverage" in df.columns:
        cov_icons = {"정상": "green", "부분음영": "yellow", "음영": "orange", "미측정": "red"}
        df["BLE"] = df["ble_coverage"].map(cov_icons).fillna("?")

    # 표시 컬럼 구성
    base_cols = ["shift", "user_name", "company_name", "work_minutes",
                 "valid_ble_minutes", "recorded_minutes",
                 "unique_loci", "transition_count", "confined_minutes", "high_voltage_minutes"]
    if "ble_coverage" in df.columns:
        base_cols.insert(3, "BLE")
    if "helmet_abandoned" in df.columns:
        base_cols.append("헬멧방치?")
    metric_cols = []
    if has_ewi:
        metric_cols += ["ewi", "high_active_min", "low_active_min", "standby_min"]
    if has_cre:
        metric_cols += ["cre", "fatigue_score", "sii"]

    show_cols = [c for c in base_cols + metric_cols if c in df.columns]
    rename = {
        "shift": "교대", "user_name": "작업자", "company_name": "업체",
        "work_minutes": "근무(분)", "valid_ble_minutes": "유효BLE(분)",
        "recorded_minutes": "전체BLE(분)",
        "unique_loci": "방문구역", "transition_count": "이동횟수",
        "confined_minutes": "밀폐(분)", "high_voltage_minutes": "고압전(분)",
        "ewi": "EWI", "high_active_min": "고활성(분)", "low_active_min": "저활성(분)",
        "standby_min": "대기(분)", "cre": "CRE", "fatigue_score": "피로도",
        "sii": "SII",
    }

    n_insufficient = int((~df["_sufficient_ble"]).sum())
    ble_note = f" (BLE {MIN_BLE_MINUTES}분 미만: {n_insufficient}명 하단 배치)" if n_insufficient else ""
    st.caption(f"총 {len(df)}명 / 상위 100명 표시{ble_note}")
    display_top = mask_names_in_df(df[show_cols].head(100), "user_name")
    st.dataframe(
        display_top.rename(columns=rename)
        .style.format({
            "EWI": "{:.3f}", "CRE": "{:.3f}", "SII": "{:.3f}", "피로도": "{:.3f}",
            "근무(분)": "{:.0f}", "유효BLE(분)": "{:.0f}", "전체BLE(분)": "{:.0f}",
        }, na_rep="-"),
        use_container_width=True, hide_index=True,
    )

    # 개별 작업자 상세 보기 (그래픽 포함)
    st.divider()
    st.markdown(section_header("작업자 상세 분석"), unsafe_allow_html=True)
    worker_options = df["user_name"].head(50).tolist()
    if not worker_options:
        return
    masked_options = [mask_name(n) for n in worker_options]
    selected_idx = st.selectbox("작업자 선택", range(len(worker_options)),
                                format_func=lambda i: masked_options[i])
    selected_worker = worker_options[selected_idx]
    wrow = df[df["user_name"] == selected_worker].iloc[0]

    # 프로필 헤더
    in_t = str(wrow.get("in_datetime", ""))[:16] if pd.notna(wrow.get("in_datetime")) else "-"
    out_t = str(wrow.get("out_datetime", ""))[:16] if pd.notna(wrow.get("out_datetime")) else "-"
    wm = wrow.get("work_minutes", 0)
    wm_h = f"{wm/60:.1f}h" if wm > 0 else "-"
    shift_icon = {"day": "sun", "night": "moon"}.get(wrow.get("shift_type", ""), "?")

    st.markdown(
        f"<div style='background:#1A2A3A; border:1px solid #2A3A4A; border-radius:10px; "
        f"padding:16px 20px; margin-bottom:16px;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div>"
        f"<span style='font-size:1.3rem; font-weight:700; color:#D5E5FF;'>"
        f"{shift_icon} {mask_name(selected_worker)}</span>"
        f"<span style='color:#9AB5D4; font-size:0.88rem; margin-left:12px;'>"
        f"{wrow.get('company_name', '-')}</span>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<div style='color:#9AB5D4; font-size:0.82rem;'>출근 {in_t} -> 퇴근 {out_t}</div>"
        f"<div style='color:#00AEEF; font-size:1.1rem; font-weight:700;'>{wm_h}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # BLE 커버리지 경고 — CHART_COLORS 단일 소스
    cov_level = wrow.get("ble_coverage", "")
    cov_pct = wrow.get("ble_coverage_pct", 0)
    if cov_level in ("음영", "미측정"):
        cov_color = CHART_COLORS["critical"] if cov_level == "미측정" else CHART_COLORS["confined_space"]
        st.markdown(
            f"<div style='background:#1A1015; border:1px solid {cov_color}44; "
            f"border-left:4px solid {cov_color}; border-radius:8px; "
            f"padding:10px 16px; margin-bottom:12px; font-size:0.85rem; color:#D5E5FF;'>"
            f"signal <b>BLE 커버리지 {cov_pct:.0f}%</b> - "
            f"<span style='color:{cov_color}'>{cov_level}</span> | "
            f"근무 {wm:.0f}분 중 BLE 기록 {wrow.get('recorded_minutes', 0)}분 | "
            f"EWI/CRE 지표를 신뢰할 수 없습니다. "
            f"T-Ward 미착용, 센서 음영, 또는 장비 고장 가능성이 있습니다."
            f"</div>",
            unsafe_allow_html=True,
        )

    # 핵심 지표 4칸 — CHART_COLORS 단일 소스
    kpi_cols = st.columns(4)
    _ewi  = wrow.get("ewi", 0)
    _cre  = wrow.get("cre", 0)
    _fat  = wrow.get("fatigue_score", 0)
    kpis = [
        ("EWI (생산성)", f"{_ewi:.3f}" if has_ewi else "-",
         CHART_COLORS["ewi"] if _ewi < 0.6 else CHART_COLORS["critical"]),
        ("CRE (위험노출)", f"{_cre:.3f}" if has_cre else "-",
         CHART_COLORS["rest"] if _cre < 0.4 else CHART_COLORS["medium"] if _cre < 0.6 else CHART_COLORS["critical"]),
        ("피로도", f"{_fat:.2f}" if "fatigue_score" in wrow else "-",
         CHART_COLORS["rest"] if _fat < 0.5 else CHART_COLORS["critical"]),
        ("방문 구역", f"{wrow.get('unique_loci', 0)}개", "#D5E5FF"),
    ]
    for col, (label, val, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value-secondary' style='color:{color}'>{val}</div>"
                f"<div class='metric-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    # 활동 시간 분배 차트
    col_chart, col_space = st.columns([3, 2])

    with col_chart:
        st.markdown(sub_header("time 활동 시간 분배"), unsafe_allow_html=True)
        activity_data = {
            "고활성 작업": wrow.get("high_active_min", 0),
            "저활성 작업": wrow.get("low_active_min", 0),
            "대기": wrow.get("standby_min", 0),
            "휴게": wrow.get("rest_min", 0),
            "이동": wrow.get("transit_min", 0),
        }
        activity_data = {k: v for k, v in activity_data.items() if v > 0}
        if activity_data:
            # 활동 수준 색상 — CHART_COLORS 단일 소스
            act_colors = {
                "고활성 작업": CHART_COLORS["work_area"],   # #00AEEF
                "저활성 작업": CHART_COLORS["low_active"],  # #0078AA
                "대기":        CHART_COLORS["medium"],      # #FFB300
                "휴게":        CHART_COLORS["rest"],        # #00C897
                "이동":        CHART_COLORS["transit"],     # #9AB5D4
            }
            fig = go.Figure(go.Bar(
                x=list(activity_data.values()),
                y=list(activity_data.keys()),
                orientation="h",
                marker_color=[act_colors.get(k, "#666") for k in activity_data.keys()],
                text=[f"{v:.0f}분" for v in activity_data.values()],
                textposition="auto",
            ))
            fig.update_layout(
                paper_bgcolor="#1A2A3A",
                plot_bgcolor="#111820",
                font_color="#C8D6E8",
                margin=dict(l=80, r=15, t=10, b=30),
                height=220, showlegend=False, xaxis_title="시간(분)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("활동 데이터 없음")

    with col_space:
        st.markdown(sub_header("pin 공간 체류"), unsafe_allow_html=True)
        # 이동공간 시간 = 전체 기록 - 작업 - 휴게
        _rec = wrow.get("recorded_minutes", 0)
        _wz = wrow.get("work_zone_minutes", 0)
        _rest = wrow.get("rest_minutes", 0)
        _transit = max(0, _rec - _wz - _rest)
        space_data = {
            "작업구역": _wz,
            "이동공간": _transit,
            "휴게공간": _rest,
            "밀폐공간": wrow.get("confined_minutes", 0),
            "고압전구역": wrow.get("high_voltage_minutes", 0),
        }
        space_data = {k: v for k, v in space_data.items() if v > 0}
        if space_data:
            # 공간별 색상 — CHART_COLORS 단일 소스 (integrity_tab._SPACE_COLORS와 일치)
            sp_colors = {
                "작업구역":   CHART_COLORS["work_area"],
                "이동공간":   CHART_COLORS["transit"],
                "휴게공간":   CHART_COLORS["rest_area"],
                "밀폐공간":   CHART_COLORS["confined_space"],
                "고압전구역": CHART_COLORS["high_voltage"],
            }
            fig = go.Figure(go.Pie(
                labels=list(space_data.keys()),
                values=list(space_data.values()),
                marker_colors=[sp_colors.get(k, "#666") for k in space_data.keys()],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                paper_bgcolor="#1A2A3A",
                plot_bgcolor="#111820",
                font_color="#C8D6E8",
                margin=dict(l=10, r=10, t=10, b=10),
                height=220, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("공간 체류 데이터 없음")

    # 위험 지표 상세
    if has_cre:
        st.markdown(sub_header("warning 위험 구성 요소"), unsafe_allow_html=True)
        risk_cols = st.columns(4)
        risks = [
            ("공간위험", wrow.get("static_norm", 0), 0.5),
            ("밀집도", wrow.get("dynamic_norm", 0), 0.5),
            ("피로도", wrow.get("fatigue_score", 0), 0.6),
            ("단독작업", wrow.get("alone_ratio", 0), 0.5),
        ]
        for col, (label, val, threshold) in zip(risk_cols, risks):
            with col:
                color = "#FF4C4C" if val >= threshold else "#FFB300" if val >= threshold * 0.7 else "#00C897"
                st.markdown(
                    f"<div style='background:#111820; border-radius:8px; padding:10px; text-align:center;'>"
                    f"<div style='color:{color}; font-size:1.2rem; font-weight:700;'>"
                    f"{val:.2f}</div>"
                    f"<div style='color:#9AB5D4; font-size:0.78rem;'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

    # 이동 경로
    if "locus_sequence" in wrow and pd.notna(wrow.get("locus_sequence")):
        st.markdown(sub_header("map 이동 경로"), unsafe_allow_html=True)

        # Journey 타임라인 시각화
        user_no = wrow.get("user_no", "")
        if user_no:
            try:
                if "current_journey_df" in st.session_state and not st.session_state.current_journey_df.empty:
                    journey_df_for_timeline = st.session_state.current_journey_df
                    fig_timeline = journey_timeline(journey_df_for_timeline, user_no, locus_dict)
                    st.plotly_chart(fig_timeline, use_container_width=True)
            except Exception as e:
                logger.debug(f"journey_timeline 렌더링 실패: {e}")

        tokens = str(wrow["locus_sequence"]).split()
        locus_names = [locus_dict.get(t, {}).get("locus_name", t) for t in tokens]
        # 중복 연속 제거하여 이동 흐름만 표시
        path = []
        for name in locus_names:
            if not path or path[-1] != name:
                path.append(name)
        path_display = " -> ".join(f"`{p}`" for p in path[:25])
        st.markdown(path_display)
        if len(path) > 25:
            st.caption(f"(+{len(path)-25}개 추가 이동)")
        st.caption(f"총 {len(tokens)}분 기록 | {wrow.get('transition_count', 0)}회 이동")
