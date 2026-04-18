"""
Deep Space 공간 관계 탭
=======================
Locus 간 관계 시각화 (임베딩 맵 + 유사도 매트릭스).
"""
from __future__ import annotations

import logging

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.cache.policy import SPATIAL_MODEL
from src.dashboard.styles import COLORS, badge
from src.dashboard.deep_space.model_loader import load_locus_info
from src.dashboard.deep_space.helpers import (
    compute_locus_embeddings, compute_similarity_matrix, render_card,
)

logger = logging.getLogger(__name__)


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner="임베딩 계산 중...")
def _compute_embeddings_cached(_model, _tokenizer, sector_id: str):
    """임베딩 계산 캐싱."""
    return compute_locus_embeddings(_model, _tokenizer)


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner="유사도 계산 중...")
def _compute_similarity_cached(_model, _tokenizer, sector_id: str):
    """유사도 매트릭스 계산 캐싱."""
    return compute_similarity_matrix(_model, _tokenizer)


def render_spatial_relations(model, tokenizer, sector_id: str):
    """공간 관계 시각화 탭 렌더링."""
    locus_info = load_locus_info(sector_id)
    if locus_info is None or locus_info.empty:
        st.warning("Locus 정보를 불러올 수 없습니다.")
        return

    # Locus 메타 빌드 (vectorized)
    locus_meta = {
        str(r.get("locus_id", "")): {
            "name": r.get("locus_name", r.get("name", str(r.get("locus_id", "")))),
            "type": r.get("locus_type", r.get("dwell_category", "")),
        }
        for r in locus_info.to_dict("records")
        if r.get("locus_id")
    }

    # 2D 임베딩
    emb_df = _compute_embeddings_cached(model, tokenizer, sector_id)
    if emb_df.empty:
        st.warning("임베딩을 추출할 수 없습니다.")
        return

    # 메타 정보 추가
    emb_df["name"] = emb_df["locus_id"].map(lambda x: locus_meta.get(x, {}).get("name", x))
    emb_df["type"] = emb_df["locus_id"].map(lambda x: locus_meta.get(x, {}).get("type", "기타"))

    var_explained = emb_df["pca_var_explained"].iloc[0] if not emb_df.empty else 0

    c1, c2 = st.columns(2)
    with c1:
        render_card("Locus 수", f"{len(emb_df)}개")
    with c2:
        render_card("PCA 설명력", f"{var_explained:.1%}")

    # PCA 산점도
    type_colors = {
        "GATE": COLORS["accent"],
        "WORK": COLORS["success"],
        "REST": COLORS["warning"],
        "HAZARD": COLORS["danger"],
        "ADMIN": COLORS["text_muted"],
        "FACILITY": "#9B59B6",
    }

    fig = px.scatter(
        emb_df, x="x", y="y",
        color="type",
        hover_name="name",
        hover_data={"locus_id": True, "x": False, "y": False},
        text="name",
        color_discrete_map=type_colors,
    )
    fig.update_traces(textposition="top center", textfont_size=9, marker_size=10)
    fig.update_layout(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor="#111820",
        font_color=COLORS["text"],
        title=f"공간 임베딩 맵 (PCA 2D, 설명력 {var_explained:.0%})",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=550,
        margin=dict(l=15, r=15, t=50, b=15),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("가까이 있는 공간 = 이동 패턴이 유사한 공간 (물리적 거리와 다를 수 있음)")

    # 유사도 매트릭스 히트맵
    with st.expander("공간 간 유사도 매트릭스", expanded=False):
        sim_df, names = _compute_similarity_cached(model, tokenizer, sector_id)
        display_names = [locus_meta.get(n, {}).get("name", n) for n in names]

        fig_heat = go.Figure(go.Heatmap(
            z=sim_df.values,
            x=display_names,
            y=display_names,
            colorscale="RdBu",
            text=[[f"{v:.2f}" if abs(v) > 0.3 else "" for v in row] for row in sim_df.values],
            texttemplate="%{text}",
            textfont={"size": 7},
            colorbar=dict(title="코사인 유사도"),
        ))
        fig_heat.update_layout(
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            title="Locus 간 코사인 유사도",
            xaxis=dict(tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
            height=600,
            margin=dict(l=120, r=20, t=50, b=100),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # 유사도 검색
    st.markdown("### 유사 공간 검색")

    select_options = [(lid, locus_meta.get(lid, {}).get("name", lid)) for lid in emb_df["locus_id"].tolist()]
    if select_options:
        selected_idx = st.selectbox(
            "기준 공간 선택",
            range(len(select_options)),
            format_func=lambda i: f"{select_options[i][1]} ({select_options[i][0]})",
            key="ds_spatial_search",
        )
        selected_locus = select_options[selected_idx][0]

        # 유사도 계산 (캐시된 매트릭스 활용)
        sim_df, names = _compute_similarity_cached(model, tokenizer, sector_id)
        if selected_locus in names:
            idx = names.index(selected_locus)
            similarities = sim_df.iloc[idx, :].copy()
            similarities = similarities.drop(selected_locus, errors="ignore")
            top_similar = similarities.nlargest(5)

            st.markdown(f"**{select_options[selected_idx][1]}**과(와) 가장 유사한 공간:")

            for rank, (lid, sim_val) in enumerate(top_similar.items(), 1):
                name = locus_meta.get(lid, {}).get("name", lid)
                ltype = locus_meta.get(lid, {}).get("type", "")

                # 유형별 badge
                type_badge_kind = {
                    "GATE": "accent",
                    "WORK": "success",
                    "REST": "warning",
                    "HAZARD": "danger",
                }.get(ltype, "text_muted")
                type_badge_html = badge(ltype, type_badge_kind) if ltype else ""

                # 유사도 바
                bar_w = max(sim_val * 100, 5)

                st.markdown(
                    f"<div style='display:flex; align-items:center; margin:4px 0; "
                    f"background:{COLORS['card_bg']}; padding:8px 12px; border-radius:6px;'>"
                    f"<div style='width:20px; color:{COLORS['text_muted']}; font-size:0.85rem;'>#{rank}</div>"
                    f"<div style='width:140px; color:{COLORS['text']}; font-size:0.9rem; margin-left:8px;'>{name}</div>"
                    f"<div style='flex:1; background:{COLORS['border']}; border-radius:4px; height:20px; margin:0 12px;'>"
                    f"<div style='width:{min(bar_w, 100)}%; background:{COLORS['accent']}; height:100%; "
                    f"border-radius:4px; display:flex; align-items:center; padding-left:6px;'>"
                    f"<span style='font-size:0.75rem; color:white;'>{sim_val:.2f}</span>"
                    f"</div></div>"
                    f"<div style='width:80px; text-align:right; margin-left:8px;'>{type_badge_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
