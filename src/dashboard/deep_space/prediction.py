"""
Deep Space 이동 예측 탭
======================
작업자별 다음 이동 장소 예측 + 예측 vs 실제 비교
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, RADIUS, SPACING, badge
from src.utils.anonymizer import mask_name
from src.dashboard.deep_space.model_loader import (
    load_journey_data, load_worker_data, load_locus_info,
)
from src.dashboard.deep_space.helpers import (
    extract_worker_sequences, predict_next, predict_next_batch,
    classify_confidence, render_card,
)
from src.dashboard.llm_deepcon import is_llm_available, render_data_comment
from src.dashboard.deep_space._ai_adapters import cached_prediction_insight
from src.utils.weather import date_label


def render_prediction(model, tokenizer, sector_id: str, dates: list[str]):
    """이동 예측 분석."""
    if not dates:
        st.warning("데이터가 없습니다.")
        return

    selected_date = st.selectbox("분석 날짜", dates, index=len(dates) - 1, key="ds_pred_date", format_func=date_label)
    journey_df = load_journey_data(sector_id, selected_date)
    worker_df = load_worker_data(sector_id, selected_date)

    if journey_df is None:
        st.warning("Journey 데이터를 로드할 수 없습니다.")
        return

    sequences = extract_worker_sequences(journey_df)
    if not sequences:
        st.warning("이동 시퀀스를 추출할 수 없습니다.")
        return

    st.markdown(f"**{len(sequences)}명**의 작업자 이동 시퀀스 추출 완료")

    # 작업자 선택
    worker_ids = sorted(sequences.keys())

    # worker.parquet에서 이름(마스킹) 매핑
    worker_labels = {}
    if worker_df is not None:
        name_col = None
        for nc in ["masked_name", "user_name"]:
            if nc in worker_df.columns:
                name_col = nc
                break
        if name_col and "user_no" in worker_df.columns:
            for _, row in worker_df.iterrows():
                uid = str(row["user_no"])
                name = row.get(name_col, "")
                worker_labels[uid] = f"{uid} ({mask_name(name)})" if name else uid

    display_labels = [worker_labels.get(w, w) for w in worker_ids]
    selected_idx = st.selectbox("작업자", range(len(worker_ids)),
                                 format_func=lambda i: display_labels[i],
                                 key="ds_pred_worker")
    selected_worker = worker_ids[selected_idx]
    seq = sequences[selected_worker]

    # 시퀀스 표시
    st.markdown(f"**이동 경로** ({len(seq)}개 장소)")
    seq_display = " -> ".join(seq[-20:])  # 최근 20개
    if len(seq) > 20:
        seq_display = "... -> " + seq_display
    st.code(seq_display, language=None)

    # 예측
    predictions = predict_next(model, tokenizer, seq, top_k=5)

    locus_info = load_locus_info(sector_id)
    locus_names = {}
    if locus_info is not None:
        for _, row in locus_info.iterrows():
            lid = row.get("locus_id", "")
            name = row.get("locus_name", row.get("name", ""))
            locus_names[str(lid)] = name

    st.markdown("### 다음 이동 예측")
    cols = st.columns(len(predictions))
    for i, (loc, prob) in enumerate(predictions):
        with cols[i]:
            name = locus_names.get(loc, loc)
            color = COLORS["accent"] if i == 0 else COLORS["text_muted"]
            rank_label = "#1" if i == 0 else "#2" if i == 1 else "#3" if i == 2 else f"#{i+1}"

            # 신뢰도 라벨
            conf_label, conf_kind = classify_confidence(prob)
            conf_badge = badge(conf_label, conf_kind)

            st.markdown(
                f"<div style='background:{COLORS['card_bg']}; border:1px solid {color}; "
                f"border-radius:{RADIUS['md']}; padding:{SPACING['md']}; text-align:center;'>"
                f"<div style='font-size:1.2rem; font-weight:700; color:{COLORS['text_muted']};'>{rank_label}</div>"
                f"<div style='color:{COLORS['text']}; font-size:0.95rem; font-weight:600; margin:4px 0;'>{name or loc}</div>"
                f"<div style='color:{COLORS['text_muted']}; font-size:0.75rem;'>{loc}</div>"
                f"<div style='color:{color}; font-size:1.2rem; font-weight:700; margin-top:4px;'>{prob:.1%}</div>"
                f"<div style='margin-top:4px;'>{conf_badge}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # 예측 확률 바 차트 (신뢰도 색상 반영)
    bar_colors = []
    for _, prob in predictions:
        _, kind = classify_confidence(prob)
        bar_colors.append(COLORS.get(kind, COLORS["text_muted"]))

    fig = go.Figure(go.Bar(
        x=[p[1] for p in predictions],
        y=[locus_names.get(p[0], p[0]) for p in predictions],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{p[1]:.1%}" for p in predictions],
        textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title="예측 확률 분포",
        xaxis_title="확률",
        yaxis=dict(autorange="reversed"),
        height=250,
        margin=dict(l=120, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # LLM 인사이트 (이동 예측 맥락)
    if is_llm_available() and predictions:
        with st.expander("AI 이동 패턴 해석", expanded=True):
            current_locus_name = locus_names.get(seq[-1], seq[-1]) if seq else "알 수 없음"
            pred_summary = ", ".join(
                f"{locus_names.get(loc, loc)}({prob:.0%})"
                for loc, prob in predictions[:3]
            )
            # locus_context: 현재 위치 + 예측 Top-3 위치 속성
            locus_context_parts = []
            if locus_info is not None:
                relevant_ids = {seq[-1]} | {p[0] for p in predictions[:3]}
                for r in locus_info.to_dict("records"):
                    if str(r.get("locus_id", "")) in relevant_ids:
                        locus_context_parts.append(
                            f"{r.get('locus_name', '')}: {r.get('locus_type', '')} / {r.get('description', '')}"
                        )
            locus_context = "\n".join(locus_context_parts)

            with st.spinner("AI 해석 중..."):
                insight = cached_prediction_insight(
                    current_locus=current_locus_name,
                    predictions=pred_summary,
                    locus_context=locus_context,
                    sector_id=sector_id,
                    tab="deep_space_prediction",
                )
            if insight:
                render_data_comment("이동 예측 해석", insight)

    # -- 예측 vs 실제 비교 (전체 경로) --------------------------------------
    st.markdown("### 예측 vs 실제 비교")
    st.caption("각 이동 시점에서 모델의 Top-1 예측과 실제 이동을 비교합니다")

    eval_steps = min(st.slider("평가 구간 수", 5, min(50, len(seq) - 1), min(20, len(seq) - 1), key="ds_eval_n"), len(seq) - 1)

    with st.spinner("예측 vs 실제 비교 분석 중..."):
        start_idx = max(3, len(seq) - eval_steps)  # 최소 3개 컨텍스트 필요

        # 배치 예측: 모든 컨텍스트를 한 번에 처리
        contexts = [seq[:i] for i in range(start_idx, len(seq))]
        actuals = [seq[i] for i in range(start_idx, len(seq))]
        all_preds = predict_next_batch(model, tokenizer, contexts, top_k=5)

        eval_results = []
        for step_i, (context, actual, preds_i) in enumerate(zip(contexts, actuals, all_preds)):
            pred_top1 = preds_i[0][0] if preds_i else ""
            pred_top1_prob = preds_i[0][1] if preds_i else 0.0

            # Top-K에 실제값이 있는지
            pred_loci = [p[0] for p in preds_i]
            hit_top1 = actual == pred_top1
            hit_top3 = actual in pred_loci[:3]
            hit_top5 = actual in pred_loci[:5]

            eval_results.append({
                "step": step_i + 1,
                "context_from": locus_names.get(context[-1], context[-1]) if context else "",
                "actual": actual,
                "actual_name": locus_names.get(actual, actual),
                "predicted": pred_top1,
                "predicted_name": locus_names.get(pred_top1, pred_top1),
                "confidence": pred_top1_prob,
                "hit_top1": hit_top1,
                "hit_top3": hit_top3,
                "hit_top5": hit_top5,
            })

    eval_df = pd.DataFrame(eval_results)

    # KPI
    n_eval = len(eval_df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        acc1 = eval_df["hit_top1"].mean()
        render_card("Top-1 정확도", f"{acc1:.1%}")
    with c2:
        acc3 = eval_df["hit_top3"].mean()
        render_card("Top-3 정확도", f"{acc3:.1%}")
    with c3:
        acc5 = eval_df["hit_top5"].mean()
        render_card("Top-5 정확도", f"{acc5:.1%}")
    with c4:
        avg_conf = eval_df["confidence"].mean()
        render_card("평균 신뢰도", f"{avg_conf:.1%}")

    # 스텝별 적중 차트
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Bar(
        x=eval_df["step"],
        y=[1] * n_eval,
        marker_color=[COLORS["success"] if h else COLORS["danger"] for h in eval_df["hit_top3"]],
        text=eval_df.apply(
            lambda r: f"O {r['actual_name']}" if r["hit_top3"] else f"X 실제:{r['actual_name']}<br>예측:{r['predicted_name']}",
            axis=1,
        ),
        hovertext=eval_df.apply(
            lambda r: f"From: {r['context_from']}<br>실제: {r['actual_name']}<br>예측: {r['predicted_name']} ({r['confidence']:.1%})",
            axis=1,
        ),
        hoverinfo="text",
        showlegend=False,
    ))
    fig_eval.add_trace(go.Scatter(
        x=eval_df["step"],
        y=eval_df["confidence"],
        mode="lines+markers",
        name="예측 신뢰도",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=5),
        yaxis="y2",
    ))
    fig_eval.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title=f"이동 예측 적중률 (최근 {n_eval}회)",
        xaxis_title="이동 순서",
        yaxis=dict(visible=False, range=[0, 1.2]),
        yaxis2=dict(
            title="신뢰도",
            overlaying="y", side="right",
            range=[0, 1], tickformat=".0%",
            gridcolor="#2A3A4A",
        ),
        height=350,
        margin=dict(l=15, r=60, t=50, b=30),
        legend=dict(font=dict(color=COLORS["text"], size=11)),
        bargap=0.3,
    )
    st.plotly_chart(fig_eval, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"<div style='display:flex; gap:16px; margin-top:4px;'>"
        f"<span style='color:{COLORS['success']};'>-- Top-3 적중</span>"
        f"<span style='color:{COLORS['danger']};'>-- 미적중</span>"
        f"<span style='color:{COLORS['accent']};'>-- 예측 신뢰도</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 상세 테이블
    with st.expander("상세 결과 테이블", expanded=False):
        display_eval = eval_df[["step", "context_from", "actual_name", "predicted_name", "confidence", "hit_top1", "hit_top3"]].copy()
        display_eval.columns = ["순서", "출발", "실제", "예측(Top-1)", "신뢰도", "Top-1 적중", "Top-3 적중"]
        display_eval["신뢰도"] = display_eval["신뢰도"].apply(lambda x: f"{x:.1%}")
        display_eval["Top-1 적중"] = display_eval["Top-1 적중"].map({True: "O", False: "X"})
        display_eval["Top-3 적중"] = display_eval["Top-3 적중"].map({True: "O", False: "X"})
        st.dataframe(display_eval, use_container_width=True, hide_index=True)
