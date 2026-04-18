"""
Deep Space 공간 예측 탭
=======================
전체 작업자의 다음 이동을 예측하여 Locus별 미래 상태 예측.
"""
from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cache.policy import MULTI_DAY_AGG
from src.dashboard.styles import (
    COLORS, section_header, metric_card_sm, insight_card,
)
from src.dashboard.deep_space.model_loader import load_model, load_journey_data, load_locus_info, DEFAULT_CAPACITY_BY_TYPE
from src.dashboard.deep_space.helpers import extract_worker_sequences, predict_next_batch
from src.dashboard.llm_deepcon import is_llm_available, render_data_comment
from src.dashboard.deep_space._ai_adapters import cached_spatial_insight
from src.utils.weather import date_label

logger = logging.getLogger(__name__)

_DEFAULT_CAPACITY_BY_TYPE = DEFAULT_CAPACITY_BY_TYPE


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def predict_locus_states(
    _model_hash: str,
    _tokenizer_hash: str,
    sequences: dict[str, list[str]],
    locus_info_json: str,
    sector_id: str,
    max_workers: int = 1000,
) -> pd.DataFrame:
    """
    전체 작업자의 다음 이동을 배치 추론으로 예측하여 각 Locus별 미래 상태를 예측.

    Returns:
        DataFrame with columns:
            locus_id, locus_name, locus_type, hazard_level, hazard_grade,
            current_count, predicted_count, change, change_pct,
            congestion_score, bottleneck_risk, risk_score, productivity_score
    """
    from collections import Counter

    model, tokenizer = load_model(sector_id)
    if model is None or tokenizer is None:
        return pd.DataFrame()

    locus_info = pd.read_json(StringIO(locus_info_json)) if locus_info_json else pd.DataFrame()

    # Locus 메타 정보 구축 (vectorized)
    locus_meta = {}
    if not locus_info.empty:
        for r in locus_info.to_dict("records"):
            lid = str(r.get("locus_id", ""))
            if not lid:
                continue
            capacity = 0
            if pd.notna(r.get("capacity")):
                capacity = int(r.get("capacity"))
            elif pd.notna(r.get("max_concurrent_occupancy")):
                capacity = int(float(r.get("max_concurrent_occupancy")) * 1.5)
            locus_meta[lid] = {
                "name": r.get("locus_name", lid),
                "type": str(r.get("locus_type", "WORK")).upper(),
                "hazard_level": r.get("hazard_level", "low"),
                "hazard_grade": float(r.get("hazard_grade", 2.0)) if pd.notna(r.get("hazard_grade")) else 2.0,
                "capacity": capacity,
            }

    # 전체 작업자 대상 (샘플링 제거)
    all_workers = list(sequences.keys())
    all_seqs = [sequences[w] for w in all_workers]

    # 현재 위치 집계
    current_counts: Counter = Counter()
    for seq in all_seqs:
        if seq:
            current_counts[seq[-1]] += 1

    # 배치 추론 (한 번에)
    batch_results = predict_next_batch(model, tokenizer, all_seqs, batch_size=256, top_k=1)

    # 예측 결과 집계
    predicted_counts: Counter = Counter()
    inflow_counts: Counter = Counter()
    outflow_counts: Counter = Counter()

    for i, preds in enumerate(batch_results):
        seq = all_seqs[i]
        if not seq:
            continue
        current_loc = seq[-1]

        if preds:
            next_loc = preds[0][0]
            predicted_counts[next_loc] += 1

            if next_loc != current_loc:
                inflow_counts[next_loc] += 1
                outflow_counts[current_loc] += 1
        else:
            # 예측 불가 시 현 위치 유지
            predicted_counts[current_loc] += 1

    all_loci = set(current_counts.keys()) | set(predicted_counts.keys()) | set(locus_meta.keys())
    total_work_predicted = sum(
        predicted_counts.get(lid, 0)
        for lid, meta in locus_meta.items()
        if meta.get("type", "").upper() == "WORK"
    )

    results = []
    for lid in all_loci:
        meta = locus_meta.get(lid, {})
        locus_name = meta.get("name", lid)
        locus_type = meta.get("type", "WORK")
        hazard_level = meta.get("hazard_level", "low")
        hazard_grade = meta.get("hazard_grade", 2.0)

        capacity = meta.get("capacity", 0)
        if capacity <= 0:
            capacity = _DEFAULT_CAPACITY_BY_TYPE.get(locus_type, 50)

        curr = current_counts.get(lid, 0)
        pred = predicted_counts.get(lid, 0)
        change = pred - curr
        change_pct = (change / max(curr, 1)) * 100 if curr > 0 else (100.0 if pred > 0 else 0.0)

        inflow = inflow_counts.get(lid, 0)
        outflow = outflow_counts.get(lid, 0)

        congestion_score = min(pred / max(capacity, 1), 2.0)
        bottleneck_risk = (inflow - outflow) / max(curr, 1) if curr > 0 else (inflow - outflow)
        risk_score = congestion_score * (hazard_grade / 5.0)

        productivity_score = 0.0
        if locus_type == "WORK" and total_work_predicted > 0:
            productivity_score = pred / total_work_predicted

        results.append({
            "locus_id": lid,
            "locus_name": locus_name,
            "locus_type": locus_type,
            "hazard_level": hazard_level,
            "hazard_grade": hazard_grade,
            "capacity": capacity,
            "current_count": curr,
            "predicted_count": pred,
            "change": change,
            "change_pct": change_pct,
            "inflow": inflow,
            "outflow": outflow,
            "congestion_score": congestion_score,
            "bottleneck_risk": bottleneck_risk,
            "risk_score": risk_score,
            "productivity_score": productivity_score,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("predicted_count", ascending=False)
    return df


def build_locus_context(current_locus_id: str, related_loci: list[str], sector_id: str) -> str:
    """Locus 자연어 컨텍스트 생성 (LLM용)."""
    locus_info = load_locus_info(sector_id)
    if locus_info is None or locus_info.empty:
        return ""

    loci_of_interest = set([current_locus_id] + related_loci)
    context_parts = []

    for r in locus_info.to_dict("records"):
        lid = str(r.get("locus_id", ""))
        if lid in loci_of_interest:
            name = r.get("locus_name", lid)
            ltype = r.get("locus_type", "")
            hazard = r.get("hazard_level", "low")
            desc = r.get("description", "")
            context_parts.append(
                f"- {name} ({lid}): {ltype}, 위험등급={hazard}" +
                (f", {desc}" if desc else "")
            )

    return "\n".join(context_parts)


def build_simulation_locus_context(locus_df: pd.DataFrame, sector_id: str) -> str:
    """시뮬레이션용 공간 컨텍스트 생성."""
    if locus_df.empty:
        return ""

    context_parts = []
    for r in locus_df.head(10).to_dict("records"):
        name = r.get("locus_name", "")
        ltype = r.get("locus_type", "")
        count = r.get("current_count", 0)
        cong = r.get("congestion", 0)
        hazard = r.get("hazard_level", "low")
        context_parts.append(
            f"- {name}: {ltype}, 현재 {count}명, 혼잡도 {cong:.0%}, 위험등급={hazard}"
        )

    return "\n".join(context_parts)


def generate_locus_insights(states_df: pd.DataFrame) -> list[dict]:
    """데이터 기반 인사이트 생성 (LLM 미사용)."""
    insights = []

    if states_df.empty:
        return insights

    # 1) 혼잡 공간
    congested = states_df[states_df["congestion_score"] >= 0.7]
    if not congested.empty:
        top_congested = congested.nlargest(1, "congestion_score").iloc[0]
        insights.append({
            "title": f"혼잡 경고: {top_congested['locus_name']}",
            "desc": f"혼잡도 {top_congested['congestion_score']:.0%} (수용력 대비 예측 인원)",
            "severity": "critical" if top_congested["congestion_score"] >= 1.0 else "high",
        })

    # 2) 병목 공간
    bottleneck = states_df[states_df["bottleneck_risk"] >= 0.3]
    if not bottleneck.empty:
        top_bottle = bottleneck.nlargest(1, "bottleneck_risk").iloc[0]
        sign = "+" if top_bottle["change"] > 0 else ""
        insights.append({
            "title": f"병목 경고: {top_bottle['locus_name']}",
            "desc": f"예상 변화 {sign}{int(top_bottle['change'])}명, 유입-유출 불균형",
            "severity": "medium",
        })

    # 3) 작업 공간 가동률
    work_spaces = states_df[states_df["locus_type"] == "WORK"]
    if not work_spaces.empty:
        total_pred = work_spaces["predicted_count"].sum()
        total_capacity = work_spaces["capacity"].sum()
        if total_capacity > 0:
            utilization = total_pred / total_capacity
            insights.append({
                "title": f"작업 공간 가동률: {utilization:.0%}",
                "desc": f"WORK 타입 공간 총 수용력 {total_capacity}명 중 {total_pred}명 예상",
                "severity": "success" if utilization >= 0.5 else "low",
            })

    return insights


def render_locus_prediction(model, tokenizer, sector_id: str, dates: list[str]):
    """공간 예측 탭 렌더링."""
    if not dates:
        st.warning("데이터가 없습니다.")
        return

    selected_date = st.selectbox("분석 날짜", dates, index=len(dates) - 1, key="ds_locus_pred_date", format_func=date_label)
    journey_df = load_journey_data(sector_id, selected_date)
    if journey_df is None:
        st.warning("Journey 데이터를 로드할 수 없습니다.")
        return

    sequences = extract_worker_sequences(journey_df)
    if not sequences:
        st.warning("이동 시퀀스를 추출할 수 없습니다.")
        return

    locus_info = load_locus_info(sector_id)
    locus_info_json = locus_info.to_json() if locus_info is not None else ""

    model_hash = f"{sector_id}_{id(model)}"
    tokenizer_hash = f"{sector_id}_{id(tokenizer)}"

    n_workers = len(sequences)
    with st.spinner(f"전체 {n_workers}명 배치 추론 중..."):
        states_df = predict_locus_states(
            model_hash, tokenizer_hash,
            sequences, locus_info_json, sector_id
        )

    if states_df.empty:
        st.warning("예측 결과를 생성할 수 없습니다.")
        return

    # KPI 카드
    st.markdown(section_header("공간 상태 예측"), unsafe_allow_html=True)

    n_congested = len(states_df[states_df["congestion_score"] >= 0.7])
    n_bottleneck = len(states_df[states_df["bottleneck_risk"] >= 0.3])
    hazard_types = ("HAZARD", "TRANSPORT")
    n_high_risk = len(states_df[
        (states_df["locus_type"].isin(hazard_types)) &
        (states_df["predicted_count"] > 0) &
        (states_df["risk_score"] >= 0.3)
    ])

    work_spaces = states_df[states_df["locus_type"] == "WORK"]
    utilization = 0.0
    if not work_spaces.empty:
        total_pred = work_spaces["predicted_count"].sum()
        total_capacity = work_spaces["capacity"].sum()
        if total_capacity > 0:
            utilization = total_pred / total_capacity

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = COLORS["danger"] if n_congested >= 3 else COLORS["warning"] if n_congested >= 1 else COLORS["success"]
        st.markdown(metric_card_sm("혼잡 예상 공간", f"{n_congested}개", color=color), unsafe_allow_html=True)
    with c2:
        color = COLORS["warning"] if n_bottleneck >= 2 else COLORS["success"]
        st.markdown(metric_card_sm("병목 위험", f"{n_bottleneck}개", color=color), unsafe_allow_html=True)
    with c3:
        color = COLORS["danger"] if n_high_risk >= 1 else COLORS["success"]
        st.markdown(metric_card_sm("고위험 공간", f"{n_high_risk}개", color=color), unsafe_allow_html=True)
    with c4:
        color = COLORS["success"] if utilization >= 0.5 else COLORS["warning"]
        st.markdown(metric_card_sm("작업 가동률", f"{utilization:.0%}", color=color), unsafe_allow_html=True)

    # LLM 인사이트 (공간 예측 맥락)
    if is_llm_available() and not states_df.empty:
        with st.expander("AI 공간 예측 해석", expanded=True):
            total_pred = int(states_df["predicted_count"].sum())
            congested_list = states_df[states_df["congestion_score"] >= 0.7]["locus_name"].tolist()
            summary = f"전체 예측 인원 {total_pred}명, 혼잡 예상 공간 {n_congested}개, 병목 위험 {n_bottleneck}개"
            congested_str = ", ".join(congested_list[:5]) if congested_list else "없음"

            # locus_context: 혼잡 예상 공간 속성
            locus_context_parts = []
            if locus_info is not None:
                congested_ids = set(
                    states_df[states_df["congestion_score"] >= 0.7]["locus_id"].astype(str).tolist()
                )
                for r in locus_info.to_dict("records"):
                    if str(r.get("locus_id", "")) in congested_ids:
                        locus_context_parts.append(
                            f"{r.get('locus_name', '')}: {r.get('locus_type', '')} / {r.get('description', '')}"
                        )
            locus_context = "\n".join(locus_context_parts)

            with st.spinner("AI 해석 중..."):
                insight = cached_spatial_insight(
                    summary=summary,
                    congested_spaces=congested_str,
                    locus_context=locus_context,
                    sector_id=sector_id,
                    tab="deep_space_locus_prediction",
                )
            if insight:
                render_data_comment("공간 예측 해석", insight)

    # 차트: 상태 변화 + 위험도 버블
    col_bar, col_bubble = st.columns(2)

    with col_bar:
        top_change = states_df.nlargest(10, "change").copy()
        if not top_change.empty:
            colors = [COLORS["danger"] if c > 0 else COLORS["success"] for c in top_change["change"]]
            fig_bar = go.Figure(go.Bar(
                x=top_change["change"],
                y=top_change["locus_name"],
                orientation="h",
                marker_color=colors,
                text=[f"{'+' if c > 0 else ''}{int(c)}명" for c in top_change["change"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                title="인원 변화 예측 (Top 10)",
                xaxis_title="변화량 (명)",
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(l=120, r=30, t=50, b=30),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_bubble:
        bubble_df = states_df[states_df["predicted_count"] > 0].copy()
        if not bubble_df.empty:
            # 크기: 예측 인원 기반, 넓이 비례로 정규화 (15~60px)
            max_count = bubble_df["predicted_count"].max() or 1
            bubble_df["bubble_size"] = 15 + (bubble_df["predicted_count"] / max_count) * 45

            # 위험등급별 색상 (0=안전 ~ 3=위험)
            grade_colors = {0: "#00C897", 1: "#FFB300", 2: "#FF6B35", 3: "#FF4C4C"}
            bubble_df["color"] = bubble_df["hazard_grade"].map(
                lambda g: grade_colors.get(int(g) if not pd.isna(g) else 0, "#888")
            )
            # 상위 위험 공간만 라벨 표시 (risk_score 기준 상위 6개)
            top_risk_ids = bubble_df.nlargest(6, "risk_score").index
            bubble_df["label"] = bubble_df.apply(
                lambda row: (row["locus_name"][:7] + "…" if len(str(row["locus_name"])) > 7 else str(row["locus_name"]))
                if row.name in top_risk_ids else "",
                axis=1,
            )

            fig_bubble = go.Figure(go.Scatter(
                x=bubble_df["congestion_score"],
                y=bubble_df["risk_score"],
                mode="markers+text",
                marker=dict(
                    size=bubble_df["bubble_size"],
                    color=bubble_df["color"],
                    opacity=0.80,
                    line=dict(color="rgba(255,255,255,0.4)", width=1.5),
                    sizemode="diameter",
                ),
                text=bubble_df["label"],
                textposition="top center",
                textfont=dict(size=10, color=COLORS["text"]),
                customdata=bubble_df[["locus_name", "predicted_count", "congestion_score", "risk_score", "hazard_grade"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "예측 인원: %{customdata[1]:.0f}명<br>"
                    "혼잡도: %{customdata[2]:.0%}<br>"
                    "위험도: %{customdata[3]:.2f}<br>"
                    "위험등급: %{customdata[4]:.0f}등급<extra></extra>"
                ),
            ))
            # 사분면 구분선 (데이터 좌표계 기준)
            fig_bubble.add_vline(x=0.7, line=dict(dash="dash", color=COLORS["warning"], width=1.5),
                                 annotation=dict(text="혼잡 임계(70%)",
                                                 font=dict(size=9, color=COLORS["warning"]),
                                                 yanchor="top"))
            fig_bubble.add_hline(y=0.3, line=dict(dash="dash", color=COLORS["danger"], width=1.5),
                                 annotation=dict(text="위험 임계(0.3)",
                                                 font=dict(size=9, color=COLORS["danger"]),
                                                 xanchor="right"))
            # 4개 사분면 배경 레이블 (데이터 좌표)
            _quad_labels = [
                (0.85, 0.9, "⚠️ 위험+혼잡", COLORS["danger"]),
                (0.15, 0.9, "🔴 고위험 여유", "#FF6B35"),
                (0.85, 0.05, "🟡 혼잡 주의", COLORS["warning"]),
                (0.15, 0.05, "🟢 안전", COLORS["success"]),
            ]
            for _qx, _qy, _qt, _qc in _quad_labels:
                fig_bubble.add_annotation(
                    x=_qx, y=_qy, text=_qt, showarrow=False,
                    font=dict(size=10, color=_qc),
                    bgcolor="rgba(0,0,0,0.5)", borderpad=3,
                    xref="x", yref="y",
                )
            fig_bubble.update_layout(
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                title="혼잡도 vs 위험도  ·  크기=예측인원 / 색상=위험등급",
                xaxis=dict(title="혼잡도 스코어 (예측)", tickformat=".0%",
                           range=[-0.05, 1.15], gridcolor="rgba(255,255,255,0.06)",
                           zeroline=False),
                yaxis=dict(title="위험도 지수",
                           range=[-0.05, 1.1], gridcolor="rgba(255,255,255,0.06)",
                           zeroline=False),
                height=420,
                margin=dict(l=60, r=30, t=55, b=40),
            )
            st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "🟢 안전(0등급)  🟡 주의(1등급)  🟠 위험(2등급)  🔴 고위험(3등급)  ·  "
                "버블 크기 = 예측 인원  ·  상위 위험 공간만 이름 표시"
            )

    # 상세 테이블
    with st.expander("상세 데이터 테이블", expanded=False):
        display_cols = [
            "locus_name", "locus_type", "current_count", "predicted_count",
            "change", "congestion_score", "bottleneck_risk", "risk_score"
        ]
        display_df = states_df[display_cols].rename(columns={
            "locus_name": "공간", "locus_type": "유형",
            "current_count": "현재", "predicted_count": "예측",
            "change": "변화", "congestion_score": "혼잡도",
            "bottleneck_risk": "병목위험", "risk_score": "위험도",
        })
        display_df["혼잡도"] = display_df["혼잡도"].apply(lambda x: f"{x:.0%}")
        display_df["병목위험"] = display_df["병목위험"].apply(lambda x: f"{x:.2f}")
        display_df["위험도"] = display_df["위험도"].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 인사이트
    st.markdown(section_header("데이터 기반 인사이트"), unsafe_allow_html=True)
    insights = generate_locus_insights(states_df)
    if insights:
        for ins in insights:
            st.markdown(
                insight_card(ins["title"], ins["desc"], severity=ins["severity"]),
                unsafe_allow_html=True,
            )
    else:
        st.caption("인사이트를 생성할 데이터가 부족합니다.")
