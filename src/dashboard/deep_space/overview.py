"""
Deep Space 개요 탭
==================
모델 KPI + 전이 히트맵 + 유입/유출 + 인사이트
"""
from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.cache.policy import SPATIAL_MODEL
from src.dashboard.styles import (
    COLORS, section_header, metric_card, metric_card_sm, badge, insight_card,
)
from src.dashboard.deep_space.model_loader import (
    load_training_history, load_locus_info, build_locus_meta,
)
from src.dashboard.deep_space.helpers import get_inflow_outflow
from src.dashboard.llm_deepcon import is_llm_available, render_data_comment
from src.dashboard.deep_space._ai_adapters import cached_spatial_insight

logger = logging.getLogger(__name__)


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner="전이 확률 계산 중...")
def _compute_transition_matrix_cached(
    _model, _tokenizer, sector_id: str,
) -> tuple[np.ndarray, list[str]] | None:
    """NextLocusPredictor 를 통해 전이 매트릭스를 계산하고 캐싱."""
    try:
        from src.model.downstream.predictor import NextLocusPredictor
        predictor = NextLocusPredictor(_model, _tokenizer)
        matrix, locus_ids = predictor.get_transition_matrix(context_length=1)
        return matrix, locus_ids
    except Exception as e:
        logger.warning(f"전이 매트릭스 계산 실패: {e}")
        return None


def _generate_overview_insights(
    matrix: np.ndarray,
    locus_ids: list[str],
    locus_meta: dict,
    history: dict | None,
) -> list[dict]:
    """데이터 기반 인사이트 자동 생성 (LLM 없음)."""
    insights = []

    if matrix is None or len(locus_ids) == 0:
        return insights

    n = len(locus_ids)

    # 1) 가장 빈번한 이동 경로 (자기자신 제외)
    best_prob = 0.0
    best_from = ""
    best_to = ""
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > best_prob:
                best_prob = matrix[i, j]
                best_from = locus_ids[i]
                best_to = locus_ids[j]

    if best_prob > 0:
        from_name = locus_meta.get(best_from, {}).get("name", best_from)
        to_name = locus_meta.get(best_to, {}).get("name", best_to)
        insights.append({
            "title": f"최빈 이동 경로: {from_name} -> {to_name}",
            "desc": f"전이 확률 {best_prob:.1%} -- 전체 Locus 쌍 중 최고 확률",
            "severity": "info",
        })

    # 2) 가장 혼잡한 병목 공간 (유입 합계 최대)
    inflow_sums = matrix.sum(axis=0)  # 각 열 합 = 유입 합계
    outflow_sums = matrix.sum(axis=1)  # 각 행 합 = 유출 합계 (~1)
    bottleneck_idx = int(np.argmax(inflow_sums))
    bottleneck_locus = locus_ids[bottleneck_idx]
    bottleneck_name = locus_meta.get(bottleneck_locus, {}).get("name", bottleneck_locus)
    inflow_val = float(inflow_sums[bottleneck_idx])
    outflow_val = float(outflow_sums[bottleneck_idx])
    if inflow_val > 0:
        retention = max(0, inflow_val - outflow_val) / inflow_val * 100
        insights.append({
            "title": f"최대 유입 공간: {bottleneck_name}",
            "desc": f"유입 합계 {inflow_val:.2f}, 유출 합계 {outflow_val:.2f}",
            "severity": "medium" if retention > 20 else "info",
        })

    # 3) 전이 커버리지 (0이 아닌 비율)
    nonzero_ratio = float(np.count_nonzero(matrix > 0.01)) / (n * n) * 100
    insights.append({
        "title": f"전이 커버리지: {nonzero_ratio:.1f}%",
        "desc": f"전체 {n}x{n} = {n*n}개 Locus 쌍 중 {int(np.count_nonzero(matrix > 0.01))}개가 유의미한 전이 확률 보유 (>1%)",
        "severity": "success" if nonzero_ratio > 30 else "low",
    })

    return insights


def render_overview(model, tokenizer, sector_id: str, dates: list[str]):
    """개요 탭: 모델 KPI + 전이 히트맵 + 유입/유출 + 인사이트."""

    # -- 모델 KPI 카드 -------------------------------------------------------
    history = load_training_history(sector_id)
    locus_info = load_locus_info(sector_id)
    locus_meta = build_locus_meta(locus_info)

    st.markdown(section_header("모델 개요"), unsafe_allow_html=True)

    top1_val = "N/A"
    top3_val = "N/A"
    top5_val = "N/A"
    vocab_val = str(len(tokenizer.locus_ids))
    train_days = "N/A"
    model_status = "No Data"
    status_kind = "danger"

    if history is not None:
        top1_list = history.get("val_acc_top1", [])
        top3_list = history.get("val_acc_top3", [])
        top5_list = history.get("val_acc_top5", [])
        best_epoch = history.get("best_epoch", 0)
        best_idx = best_epoch - 1 if best_epoch > 0 else -1
        # Best epoch 기준 정확도 (last epoch 아님)
        if top1_list:
            top1_val = f"{top1_list[best_idx]:.1%}"
        if top3_list:
            top3_val = f"{top3_list[best_idx]:.1%}"
        if top5_list:
            top5_val = f"{top5_list[best_idx]:.1%}"
        train_days = str(len(dates)) if dates else "N/A"
        model_status = "Ready · v2"
        status_kind = "success"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card("Top-1 정확도", top1_val), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Top-3 정확도", top3_val), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_sm("Top-5 정확도", top5_val), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_sm("Vocab Size", f"{vocab_val} Loci"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card_sm("학습 데이터", f"{train_days}일"), unsafe_allow_html=True)
    with c6:
        st.markdown(
            metric_card_sm("모델 상태", badge(model_status, status_kind), color=COLORS.get(status_kind, "")),
            unsafe_allow_html=True,
        )

    # -- 전이 확률 히트맵 ---------------------------------------------------
    st.markdown(section_header("전이 확률 히트맵"), unsafe_allow_html=True)

    result = _compute_transition_matrix_cached(model, tokenizer, sector_id)
    if result is None:
        st.warning("전이 확률 매트릭스를 계산할 수 없습니다.")
        return

    matrix, locus_ids = result

    # 상위 20 Locus (행 합 기준)
    row_activity = matrix.sum(axis=1) + matrix.sum(axis=0)
    n_display = min(20, len(locus_ids))
    top_indices = np.argsort(row_activity)[::-1][:n_display]
    top_indices = np.sort(top_indices)  # 원래 순서 유지

    display_matrix = matrix[np.ix_(top_indices, top_indices)]
    display_ids = [locus_ids[i] for i in top_indices]
    display_names = [locus_meta.get(lid, {}).get("name", lid) for lid in display_ids]

    fig_heat = go.Figure(go.Heatmap(
        z=display_matrix,
        x=display_names,
        y=display_names,
        colorscale="Viridis",
        text=np.where(display_matrix > 0.01, np.round(display_matrix, 2), ""),
        texttemplate="%{text}",
        textfont={"size": 7},
        hoverongaps=False,
        colorbar=dict(title="확률"),
    ))
    fig_heat.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title=f"Locus 전이 확률 (상위 {n_display}개)",
        height=600,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
        margin=dict(l=120, r=20, t=50, b=100),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # -- 특정 Locus 유입/유출 -----------------------------------------------
    st.markdown(section_header("Locus 유입/유출 분석"), unsafe_allow_html=True)

    select_options = [(lid, locus_meta.get(lid, {}).get("name", lid)) for lid in locus_ids]
    selected_idx = st.selectbox(
        "Locus 선택",
        range(len(select_options)),
        format_func=lambda i: f"{select_options[i][1]} ({select_options[i][0]})",
        key="ds_overview_locus",
    )
    selected_locus = select_options[selected_idx][0]

    flows = get_inflow_outflow(matrix, locus_ids, selected_locus, top_k=5)

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown(
            f"<div style='font-size:0.95rem; font-weight:600; color:{COLORS['text']}; "
            f"margin-bottom:8px;'>유입 (이 공간으로)</div>",
            unsafe_allow_html=True,
        )
        if not flows["inflow"]:
            st.caption("유입 데이터 없음")
        for rank, (loc, prob) in enumerate(flows["inflow"], 1):
            name = locus_meta.get(loc, {}).get("name", loc)
            bar_w = max(prob * 100 * 2.5, 5)  # 시각적 스케일링
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:3px 0;'>"
                f"<div style='width:16px; color:{COLORS['text_muted']}; font-size:0.8rem;'>#{rank}</div>"
                f"<div style='width:120px; color:{COLORS['text']}; font-size:0.85rem; margin-left:4px;'>{name}</div>"
                f"<div style='flex:1; background:{COLORS['border']}; border-radius:4px; height:18px;'>"
                f"<div style='width:{min(bar_w, 100)}%; background:{COLORS['accent']}; height:100%; "
                f"border-radius:4px; display:flex; align-items:center; padding-left:6px;'>"
                f"<span style='font-size:0.7rem; color:white;'>{prob:.1%}</span>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

    with col_out:
        st.markdown(
            f"<div style='font-size:0.95rem; font-weight:600; color:{COLORS['text']}; "
            f"margin-bottom:8px;'>유출 (여기서 이동)</div>",
            unsafe_allow_html=True,
        )
        if not flows["outflow"]:
            st.caption("유출 데이터 없음")
        for rank, (loc, prob) in enumerate(flows["outflow"], 1):
            name = locus_meta.get(loc, {}).get("name", loc)
            bar_w = max(prob * 100 * 2.5, 5)
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:3px 0;'>"
                f"<div style='width:16px; color:{COLORS['text_muted']}; font-size:0.8rem;'>#{rank}</div>"
                f"<div style='width:120px; color:{COLORS['text']}; font-size:0.85rem; margin-left:4px;'>{name}</div>"
                f"<div style='flex:1; background:{COLORS['border']}; border-radius:4px; height:18px;'>"
                f"<div style='width:{min(bar_w, 100)}%; background:{COLORS['success']}; height:100%; "
                f"border-radius:4px; display:flex; align-items:center; padding-left:6px;'>"
                f"<span style='font-size:0.7rem; color:white;'>{prob:.1%}</span>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

    # -- 데이터 기반 인사이트 -----------------------------------------------
    st.markdown(section_header("데이터 기반 인사이트"), unsafe_allow_html=True)

    insights = _generate_overview_insights(matrix, locus_ids, locus_meta, history)
    if insights:
        for ins in insights:
            st.markdown(
                insight_card(ins["title"], ins["desc"], severity=ins["severity"]),
                unsafe_allow_html=True,
            )
    else:
        st.caption("인사이트를 생성할 데이터가 부족합니다.")

    # LLM 인사이트 (모델 전체 패턴 해석)
    if is_llm_available():
        with st.expander("🤖 AI 공간 패턴 해석", expanded=True):
            # 전이 행렬에서 가장 많이 이동하는 경로 Top-5 추출
            if matrix is not None and locus_meta:
                top_transitions = []
                for i, from_id in enumerate(locus_ids):
                    row_vals = matrix[i]
                    top_idx = row_vals.argsort()[::-1][:2]
                    for idx in top_idx:
                        if row_vals[idx] > 0.1:
                            from_meta = locus_meta.get(from_id, {})
                            to_meta = locus_meta.get(locus_ids[idx], {})
                            from_name = from_meta.get("name", from_id) if isinstance(from_meta, dict) else from_meta
                            to_name = to_meta.get("name", locus_ids[idx]) if isinstance(to_meta, dict) else to_meta
                            top_transitions.append(f"{from_name}->{to_name}({row_vals[idx]:.0%})")
                movement_pattern = ", ".join(top_transitions[:6])
            else:
                movement_pattern = "데이터 없음"

            summary = f"모델 학습 완료 -- 주요 이동 패턴: {movement_pattern}"
            congested_spaces = "N/A (개요 탭)"

            with st.spinner("AI 해석 중..."):
                insight = cached_spatial_insight(
                    summary=summary,
                    congested_spaces=congested_spaces,
                    locus_context=movement_pattern,
                    sector_id=sector_id,
                    tab="deep_space_overview",
                )
            if insight:
                render_data_comment("공간 패턴 해석", insight)
