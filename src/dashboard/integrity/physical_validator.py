"""
integrity/physical_validator.py — 물리적 이동 검증 (worker_review 탭 내부 위임)
==============================================================================
`_render_physical_validation(sector_id, user_jdf)` 는 worker_review 의
7번째 inner 탭에서 호출된다. 실제 연산은 `src/pipeline/physical_validator.py`
에 위임하고, 여기서는 UI 레이어만 담당.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    COLORS,
    PLOTLY_DARK,
    PLOTLY_LEGEND,
    metric_card_sm,
    sub_header,
)

logger = logging.getLogger(__name__)


def _render_physical_validation(sector_id: str, user_jdf: pd.DataFrame) -> None:
    """
    물리적 이동 가능성 검증 서브탭.

    사람은 점프하지 못한다 → 1분 단위 locus 전이를 보행 속도·건물/층 변경 규칙으로 평가.
    ( src/pipeline/physical_validator.py 참조 )
    """
    # ── CLOUD_MODE: 재계산 불가 (pipeline 코드 없음) ────────────────
    if getattr(cfg, "CLOUD_MODE", False):
        st.info(
            "☁️ **클라우드 모드** — 물리적 이동 검증은 파이프라인 실행 환경에서만 지원됩니다.  \n"
            "로컬 환경에서 파이프라인을 실행하면 검증 결과가 journey.parquet에 저장됩니다."
        )
        return

    try:
        from src.pipeline.physical_validator import (
            FEAS_OK, FEAS_WARN_FAST, FEAS_IMPOSSIBLE,
            FEAS_CROSS_FLOOR, FEAS_CROSS_BLDG,
            annotate_journey, get_gateway_geo, summarize_feasibility,
        )
    except ImportError:
        st.warning("physical_validator 모듈을 로드할 수 없습니다.")
        return

    if user_jdf.empty:
        st.info("선택한 작업자의 journey 데이터가 없습니다.")
        return

    paths = cfg.get_sector_paths(sector_id)
    locus_csv = paths.get("locus_v2_csv") or paths.get("locus_csv")
    if not locus_csv or not Path(locus_csv).exists():
        st.warning(
            f"locus 좌표 파일을 찾을 수 없습니다 ({locus_csv}). "
            "검증을 위해서는 locus_v2.csv가 필요합니다."
        )
        return

    geo = get_gateway_geo(sector_id, str(locus_csv))
    df = annotate_journey(user_jdf, geo)
    summary = summarize_feasibility(df)

    # ── 설명 ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style='background:#1A2A3A; padding:12px 14px; border-radius:6px;
                    border-left:3px solid #00AEEF; margin-bottom:12px;
                    color:#C8D6E8; font-size:0.86rem; line-height:1.55;'>
            <b style='color:#00AEEF;'>🚶 물리적 이동 검증</b> —
            사람은 점프할 수 없습니다. 1분 간격 locus 변경을 건물/층/거리 기준으로
            평가합니다.
            <ul style='margin:6px 0 0 18px; font-size:0.82rem; color:#9AB5D4;'>
                <li><b style='color:#00C897;'>OK</b> — 보행 속도 이내 (≤120 m/min)</li>
                <li><b style='color:#00AEEF;'>CROSS_FLOOR / CROSS_BLDG</b> — 층/건물 변경 (정상 범위)</li>
                <li><b style='color:#FFB300;'>WARN_FAST</b> — 달리기 수준 (120~200 m/min, 주의)</li>
                <li><b style='color:#FF4C4C;'>IMPOSSIBLE</b> — 보행·엘리베이터로 불가능 (플리커/오탐 의심)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI ───────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    ok_pct  = summary['ok'] / max(summary['total'], 1) * 100
    with k1:
        st.markdown(metric_card_sm("전이 총계", f"{summary['total']:,}"), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card_sm(f"정상 {ok_pct:.0f}%",
                                   f"{summary['ok']:,}", "#00C897"),
                    unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card_sm("층·건물 변경",
                                   f"{summary['cross_floor'] + summary['cross_bldg']:,}",
                                   "#00AEEF"), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card_sm(f"달리기 {summary['warn_pct']:.1f}%",
                                   f"{summary['warn_fast']:,}", "#FFB300"),
                    unsafe_allow_html=True)
    with k5:
        st.markdown(metric_card_sm(f"물리 불가 {summary['impossible_pct']:.1f}%",
                                   f"{summary['impossible']:,}", "#FF4C4C"),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── 타임라인 마커 ───────────────────────────────────────
    st.markdown(sub_header("시간대별 판정 (문제 구간 강조)"), unsafe_allow_html=True)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 전이가 발생한 (locus_changed) 지점만 우선 시각화
    tr = df[df["locus_id"] != df["locus_id"].shift(1)].copy()

    color_map = {
        FEAS_OK:          "#00C897",
        FEAS_CROSS_FLOOR: "#00AEEF",
        FEAS_CROSS_BLDG:  "#4A8FD6",
        FEAS_WARN_FAST:   "#FFB300",
        FEAS_IMPOSSIBLE:  "#FF4C4C",
    }

    fig = go.Figure()
    for cat, color in color_map.items():
        sub = tr[tr["phys_feasibility"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=[cat] * len(sub),
            mode="markers",
            name=cat,
            marker=dict(size=10, color=color, line=dict(width=0.5, color="#0D1B2A")),
            customdata=np.stack([
                sub["locus_id"].fillna("-").values,
                sub["phys_distance_m"].fillna(-1).values,
                sub["phys_speed"].fillna(-1).values,
                sub["phys_reason"].fillna("").values,
            ], axis=-1),
            hovertemplate=(
                "%{x|%H:%M}<br>"
                "→ locus: <b>%{customdata[0]}</b><br>"
                "거리: %{customdata[1]:.0f} m<br>"
                "속도: %{customdata[2]:.0f} m/분<br>"
                "%{customdata[3]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=140, r=20, t=20, b=40)},
        height=260,
        xaxis=dict(title="시간", tickformat="%H:%M",
                   tickfont_color=COLORS["text_muted"], gridcolor="#2A3A4A"),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=[FEAS_OK, FEAS_CROSS_FLOOR, FEAS_CROSS_BLDG,
                           FEAS_WARN_FAST, FEAS_IMPOSSIBLE],
            tickfont=dict(size=10, color=COLORS["text"]),
        ),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True, key="plotly_12")

    # ── 문제 구간 테이블 ─────────────────────────────────────
    st.markdown(sub_header("문제 전이 상세 (IMPOSSIBLE / WARN_FAST)"), unsafe_allow_html=True)

    bad = df[df["phys_feasibility"].isin([FEAS_IMPOSSIBLE, FEAS_WARN_FAST])].copy()
    if bad.empty:
        st.success("✓ 이 작업자는 물리적 불가 또는 달리기 수준 전이가 없습니다.")
        return

    bad["time"] = bad["timestamp"].dt.strftime("%H:%M")
    bad["prev_locus"] = bad["locus_id"].shift(1)
    bad_display = bad[[
        "time", "phys_feasibility", "locus_id", "locus_name",
        "phys_distance_m", "phys_speed", "phys_reason",
    ]].rename(columns={
        "time":             "시간",
        "phys_feasibility": "판정",
        "locus_id":         "locus",
        "locus_name":       "locus 이름",
        "phys_distance_m":  "거리(m)",
        "phys_speed":       "속도(m/분)",
        "phys_reason":      "사유",
    })

    def _row_style(row):
        color = ("background-color:#3A2020; color:#FF8080"
                 if row["판정"] == FEAS_IMPOSSIBLE
                 else "background-color:#3A3020; color:#FFC860")
        return [color] * len(row)

    st.dataframe(
        bad_display.style.apply(_row_style, axis=1).format({
            "거리(m)":    lambda v: f"{v:.0f}" if pd.notna(v) else "-",
            "속도(m/분)": lambda v: f"{v:.0f}" if pd.notna(v) else "-",
        }),
        use_container_width=True,
        height=min(400, 36 * (len(bad_display) + 1)),
    )

    csv = bad_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ 문제 전이 CSV 다운로드",
        data=csv,
        file_name=f"physical_issues_{sector_id}.csv",
        mime="text/csv",
    )

