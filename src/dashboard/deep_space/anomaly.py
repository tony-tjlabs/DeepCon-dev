"""
Deep Space 이상 이동 탐지 탭
============================
Surprisal 기반 비정상 이동 패턴 감지.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.styles import COLORS, section_header, badge
from src.dashboard.llm_deepcon import is_llm_available, render_data_comment
from src.dashboard.deep_space._ai_adapters import cached_anomaly_insight
from src.utils.anonymizer import mask_name
from src.dashboard.deep_space.model_loader import load_journey_data, load_worker_data
from src.dashboard.deep_space.helpers import (
    extract_worker_sequences, compute_anomaly_score, classify_severity, render_card,
)
from src.dashboard.deep_space.locus_prediction import build_locus_context
from src.utils.weather import date_label

logger = logging.getLogger(__name__)


def render_anomaly(model, tokenizer, sector_id: str, dates: list[str]):
    """이상 이동 탐지 탭 렌더링."""
    if not dates:
        st.warning("데이터가 없습니다.")
        return

    selected_date = st.selectbox("분석 날짜", dates, index=len(dates) - 1, key="ds_anomaly_date", format_func=date_label)
    journey_df = load_journey_data(sector_id, selected_date)
    worker_df = load_worker_data(sector_id, selected_date)

    if journey_df is None:
        st.warning("Journey 데이터를 로드할 수 없습니다.")
        return

    sequences = extract_worker_sequences(journey_df)
    if not sequences:
        st.warning("이동 시퀀스를 추출할 수 없습니다.")
        return

    # 임계값 슬라이더
    sigma_mult = st.slider(
        "이상 판정 임계값 (sigma 배수)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="평균 + N*표준편차 이상을 이상으로 판정합니다. 값이 낮을수록 민감하게 감지합니다.",
        key="ds_anomaly_sigma",
    )

    max_workers = st.slider(
        "분석 작업자 수 (많을수록 느림)",
        10, min(100, len(sequences)), 30,
        key="ds_anomaly_n"
    )

    # 캐시 키: 날짜 + 작업자수 + sigma → 동일 설정이면 재계산 방지
    cache_key = f"_ds_anomaly_{selected_date}_{max_workers}_{sigma_mult}"
    if cache_key in st.session_state:
        results = st.session_state[cache_key]
    else:
        with st.spinner(f"{max_workers}명 이상 이동 분석 중..."):
            results = []
            worker_ids = sorted(sequences.keys())[:max_workers]
            for wid in worker_ids:
                seq = sequences[wid]
                score, anomalies = compute_anomaly_score(model, tokenizer, seq)
                results.append({
                    "user_no": wid,
                    "avg_surprisal": score,
                    "n_anomalies": len(anomalies),
                    "seq_length": len(seq),
                    "anomaly_details": anomalies,
                })
        st.session_state[cache_key] = results

    result_df = pd.DataFrame(results).sort_values("avg_surprisal", ascending=False)

    # 이름 매핑
    if worker_df is not None:
        name_col = "masked_name" if "masked_name" in worker_df.columns else "user_name" if "user_name" in worker_df.columns else None
        if name_col and "user_no" in worker_df.columns:
            name_map = dict(zip(worker_df["user_no"].astype(str), worker_df[name_col].apply(mask_name)))
            result_df["name"] = result_df["user_no"].map(name_map).fillna("")

    # 위험도 분류
    mean_s = result_df["avg_surprisal"].mean()
    std_s = result_df["avg_surprisal"].std()

    result_df["severity_label"] = result_df["avg_surprisal"].apply(
        lambda s: classify_severity(s, mean_s, std_s, sigma_mult)[0]
    )
    result_df["severity_color_key"] = result_df["avg_surprisal"].apply(
        lambda s: classify_severity(s, mean_s, std_s, sigma_mult)[1]
    )

    n_anomalous = len(result_df[result_df["severity_label"].isin(["Critical", "High", "Medium"])])
    anomaly_ratio = n_anomalous / len(result_df) * 100 if len(result_df) > 0 else 0

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("분석 작업자", f"{len(result_df)}명")
    with c2:
        render_card("이상 이동 감지", f"{n_anomalous}명")
    with c3:
        render_card("이상 비율", f"{anomaly_ratio:.1f}%")
    with c4:
        render_card("평균 Surprisal", f"{mean_s:.2f}")

    # 경고
    if anomaly_ratio > 30:
        st.warning(
            f"이상 비율이 {anomaly_ratio:.1f}%로 높습니다. "
            "모델 재학습 또는 임계값 조정을 권고합니다."
        )

    # Surprisal 산점도
    st.markdown(section_header("이상 이동 분포"), unsafe_allow_html=True)

    severity_color_map = {
        "Critical": COLORS["danger"],
        "High": COLORS["confined"],
        "Medium": COLORS["warning"],
        "Low": COLORS["text_muted"],
    }

    scatter_df = result_df[["user_no", "avg_surprisal", "seq_length", "n_anomalies", "severity_label"]].copy()

    fig_scatter = px.scatter(
        scatter_df,
        x="seq_length",
        y="avg_surprisal",
        color="severity_label",
        size="n_anomalies",
        size_max=18,
        hover_data={"user_no": True, "seq_length": True, "avg_surprisal": ":.2f", "n_anomalies": True},
        color_discrete_map=severity_color_map,
        category_orders={"severity_label": ["Critical", "High", "Medium", "Low"]},
    )

    fig_scatter.add_hline(
        y=mean_s + sigma_mult * std_s, line_dash="dash",
        line_color=COLORS["warning"], annotation_text=f"임계값 (mean+{sigma_mult}*std)"
    )
    if std_s > 0:
        fig_scatter.add_hline(
            y=mean_s + 2.0 * std_s, line_dash="dot",
            line_color=COLORS["confined"], annotation_text="High (mean+2*std)"
        )
        fig_scatter.add_hline(
            y=mean_s + 2.5 * std_s, line_dash="dot",
            line_color=COLORS["danger"], annotation_text="Critical (mean+2.5*std)"
        )

    fig_scatter.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title="Surprisal vs 이동 횟수 (크기=이상 구간 수)",
        xaxis_title="이동 횟수 (시퀀스 길이)",
        yaxis_title="평균 Surprisal",
        height=450,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(font=dict(color=COLORS["text"], size=11)),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # 히스토그램
    fig_hist = px.histogram(
        result_df, x="avg_surprisal", nbins=20,
        color_discrete_sequence=[COLORS["accent"]]
    )
    fig_hist.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title="이상 점수(Surprisal) 분포",
        xaxis_title="평균 Surprisal (높을수록 비정상)",
        yaxis_title="작업자 수",
        height=300,
        margin=dict(l=15, r=15, t=50, b=15),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # LLM 인사이트
    if is_llm_available() and n_anomalous > 0 and not result_df.empty:
        top_anomaly = result_df.iloc[0]
        anomaly_details = top_anomaly.get("anomaly_details", [])
        if anomaly_details:
            top_anom = anomaly_details[0]
            from_loc = sequences.get(top_anomaly["user_no"], [""])[-1] if sequences.get(top_anomaly["user_no"]) else ""
            to_loc = top_anom[1] if len(top_anom) > 1 else ""
            locus_context = build_locus_context(from_loc, [to_loc], sector_id)

            insight = cached_anomaly_insight(
                anomaly_description=f"비정상 이동: {to_loc}",
                perplexity=f"{top_anomaly['avg_surprisal']:.2f}",
                locus_context=locus_context,
                sector_id=sector_id,
                tab="deep_space_anomaly",
            )
            if insight:
                with st.expander("AI 이상 이동 해석", expanded=False):
                    render_data_comment("이상 이동 해석", insight)

    # 상위 10명
    st.markdown(section_header("고위험 작업자 Top 10"), unsafe_allow_html=True)
    top10 = result_df.head(10).copy()

    top10["위험도"] = top10.apply(
        lambda r: badge(r["severity_label"], r["severity_color_key"]),
        axis=1,
    )

    display_cols = ["user_no"]
    if "name" in top10.columns:
        display_cols.append("name")
    display_cols += ["avg_surprisal", "n_anomalies", "seq_length", "severity_label"]

    display_top10 = top10[display_cols].rename(columns={
        "user_no": "작업자 ID",
        "name": "이름",
        "avg_surprisal": "이상 점수",
        "n_anomalies": "이상 구간 수",
        "seq_length": "이동 수",
        "severity_label": "위험도",
    })
    st.dataframe(display_top10, use_container_width=True, hide_index=True)

    # 상위 1명 상세
    if not result_df.empty:
        top_worker = result_df.iloc[0]
        anomalies = top_worker["anomaly_details"]
        if anomalies:
            wid = top_worker["user_no"]
            name_label = top_worker.get("name", "")
            sev_label = top_worker["severity_label"]
            st.markdown(
                f"### 상세: {wid} {name_label} {badge(sev_label, top_worker['severity_color_key'])}",
                unsafe_allow_html=True,
            )
            seq = sequences[wid]
            for idx, loc, score in anomalies[:5]:
                before = seq[max(0, idx - 2):idx]
                after = seq[idx + 1:idx + 3]
                st.markdown(
                    f"<div style='background:{COLORS['card_bg']}; border-left:3px solid {COLORS['danger']}; "
                    f"padding:8px 12px; margin:4px 0; border-radius:4px;'>"
                    f"<span style='color:{COLORS['text_muted']};'>{'->'.join(before)} -> </span>"
                    f"<span style='color:{COLORS['danger']}; font-weight:700;'>!! {loc}</span>"
                    f"<span style='color:{COLORS['text_muted']};'> -> {'->'.join(after)}</span>"
                    f"<span style='color:{COLORS['warning']}; float:right;'>surprisal: {score:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
