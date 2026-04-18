"""
Deep Space 학습 현황 탭
======================
모델 상태 + 학습 이력 시각화 (v2 업그레이드)
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.styles import COLORS, badge
from src.dashboard.deep_space.model_loader import load_training_history
from src.dashboard.deep_space.helpers import render_card


def _fmt_time(seconds: float) -> str:
    """초 → 'X시간 Y분' 형식 변환."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}시간 {m}분"
    return f"{m}분"


def render_training(sector_id: str):
    """모델 상태 + 학습 이력 (v2)."""
    history = load_training_history(sector_id)
    if history is None:
        st.info("학습된 모델이 없습니다. 파이프라인 탭에서 Deep Space 학습을 먼저 진행하세요.")
        return

    best_epoch   = history.get("best_epoch", 0)
    best_loss    = history.get("best_val_loss", 0.0)
    total_time   = history.get("total_train_time", 0.0)
    total_epochs = len(history.get("train_losses", []))

    top1_list = history.get("val_acc_top1", [])
    top3_list = history.get("val_acc_top3", [])
    top5_list = history.get("val_acc_top5", [])
    epoch_times = history.get("epoch_times", [])

    # Best epoch 기준 정확도
    best_idx  = best_epoch - 1 if best_epoch > 0 else -1
    top1_best = top1_list[best_idx] if top1_list else 0.0
    top3_best = top3_list[best_idx] if top3_list else 0.0
    top5_best = top5_list[best_idx] if top5_list else 0.0

    # ── 모델 배지 ──────────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>"
        f"<span style='font-size:1.1rem; font-weight:700; color:{COLORS['text']};'>"
        f"Deep Space Foundation Model</span>"
        f"{badge('v2', 'success')}"
        f"<span style='font-size:0.78rem; color:{COLORS['text_muted']};'>"
        f"vocab=218 &middot; d_model=256 &middot; 4 layers &middot; 8 heads &middot; 3.4M params"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # ── 6-컬럼 KPI 카드 ────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_card("학습 에포크", f"{total_epochs}")
    with c2:
        render_card("Best Epoch", f"#{best_epoch}")
    with c3:
        render_card("Best Val Loss", f"{best_loss:.4f}")
    with c4:
        render_card("Top-1 정확도", f"{top1_best:.1%}")
    with c5:
        render_card("Top-3 정확도", f"{top3_best:.1%}")
    with c6:
        render_card("Top-5 정확도", f"{top5_best:.1%}")

    st.caption(
        f"총 학습 시간: **{_fmt_time(total_time)}** ({total_time:,.0f}초)"
        f"&nbsp;·&nbsp;데이터: 40일 / 200K 시퀀스"
        f"&nbsp;·&nbsp;MPS GPU (Apple M-series)"
    )

    # ── v1 vs v2 비교 ─────────────────────────────────────────────
    with st.expander("📊 v1 vs v2 모델 성능 비교", expanded=False):
        col_v1, col_v2 = st.columns(2)
        v2_top1_str = f"{top1_best:.1%}"
        v2_top3_str = f"{top3_best:.1%}"
        with col_v1:
            st.markdown(
                f"<div style='background:{COLORS['card_bg']}; border:1px solid {COLORS['border']}; "
                f"border-radius:8px; padding:14px;'>"
                f"<div style='font-size:0.85rem; color:{COLORS['text_muted']}; margin-bottom:8px; font-weight:600;'>"
                f"v1 (이전)</div>"
                f"<div style='font-size:0.82rem; color:{COLORS['text']}; line-height:1.7;'>"
                f"Val Loss: 1.6666<br>"
                f"Top-1: 41.1%&nbsp;&nbsp;Top-3: 75.4%&nbsp;&nbsp;Top-5: N/A<br>"
                f"학습 데이터: 20일 · 50K 시퀀스<br>"
                f"Architecture: d_model=64 · 120K params<br>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with col_v2:
            st.markdown(
                f"<div style='background:{COLORS['card_bg']}; border:1px solid {COLORS['success']}; "
                f"border-radius:8px; padding:14px;'>"
                f"<div style='font-size:0.85rem; color:{COLORS['success']}; margin-bottom:8px; font-weight:600;'>"
                f"v2 (현재) ✓</div>"
                f"<div style='font-size:0.82rem; color:{COLORS['text']}; line-height:1.7;'>"
                f"Val Loss: <b>{best_loss:.4f}</b> "
                f"<span style='color:{COLORS['success']};'>↓ 20%</span><br>"
                f"Top-1: <b>{v2_top1_str}</b>"
                f"<span style='color:{COLORS['success']};'> ↑ +19.1%p</span>"
                f"&nbsp;&nbsp;Top-3: <b>{v2_top3_str}</b>"
                f"<span style='color:{COLORS['success']};'> ↑ +7.1%p</span><br>"
                f"학습 데이터: 40일 · 200K 시퀀스<br>"
                f"Architecture: d_model=256 · 4 layers · 3.4M params<br>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Loss 곡선 + 정확도 곡선 (나란히) ──────────────────────────
    train_losses = history.get("train_losses", [])
    val_losses   = history.get("val_losses", [])

    if train_losses:
        epochs = list(range(1, len(train_losses) + 1))
        col_loss, col_acc = st.columns(2)

        with col_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=train_losses, name="Train Loss",
                line=dict(color=COLORS["accent"], width=2),
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=val_losses, name="Val Loss",
                line=dict(color=COLORS["warning"], width=2),
            ))
            if best_epoch > 0:
                fig_loss.add_vline(
                    x=best_epoch,
                    line=dict(color=COLORS["success"], width=1.5, dash="dash"),
                    annotation_text=f"Best #{best_epoch}",
                    annotation_font=dict(color=COLORS["success"], size=10),
                )
            fig_loss.update_layout(
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                title="Loss 추이",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=310,
                margin=dict(l=15, r=15, t=45, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

        with col_acc:
            fig_acc = go.Figure()
            if top1_list:
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=[v * 100 for v in top1_list], name="Top-1",
                    line=dict(color=COLORS["danger"], width=2),
                ))
            if top3_list:
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=[v * 100 for v in top3_list], name="Top-3",
                    line=dict(color=COLORS["warning"], width=2),
                ))
            if top5_list:
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=[v * 100 for v in top5_list], name="Top-5",
                    line=dict(color=COLORS["success"], width=2),
                ))
            if best_epoch > 0:
                fig_acc.add_vline(
                    x=best_epoch,
                    line=dict(color=COLORS["success"], width=1.5, dash="dash"),
                    annotation_text=f"Best #{best_epoch}",
                    annotation_font=dict(color=COLORS["success"], size=10),
                )
            fig_acc.update_layout(
                paper_bgcolor=COLORS["card_bg"],
                plot_bgcolor="#111820",
                font_color=COLORS["text"],
                title="예측 정확도 추이",
                xaxis_title="Epoch",
                yaxis_title="정확도 (%)",
                height=310,
                margin=dict(l=15, r=15, t=45, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

    # ── 에포크별 학습 시간 바 차트 ─────────────────────────────────
    if epoch_times:
        epochs = list(range(1, len(epoch_times) + 1))
        bar_colors = [
            COLORS["success"] if i + 1 == best_epoch else COLORS["accent"]
            for i in range(len(epoch_times))
        ]
        fig_time = go.Figure(go.Bar(
            x=epochs,
            y=[t / 60 for t in epoch_times],
            marker_color=bar_colors,
            text=[f"{t/60:.0f}m" for t in epoch_times],
            textposition="outside",
            textfont=dict(size=8, color=COLORS["text_muted"]),
        ))
        fig_time.update_layout(
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor="#111820",
            font_color=COLORS["text"],
            title=f"에포크별 학습 시간 — 총 {_fmt_time(total_time)} | 초록 = Best Epoch",
            xaxis_title="Epoch",
            yaxis_title="시간 (분)",
            height=260,
            margin=dict(l=15, r=15, t=45, b=30),
        )
        st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

    # ── 에포크별 상세 테이블 ───────────────────────────────────────
    with st.expander("에포크별 상세 수치", expanded=False):
        import pandas as pd
        rows = []
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            rows.append({
                "Epoch":      f"{'★ ' if i == best_epoch else ''}{i}",
                "Train Loss": f"{tl:.4f}",
                "Val Loss":   f"{vl:.4f}",
                "Top-1":      f"{top1_list[i-1]:.2%}" if top1_list else "-",
                "Top-3":      f"{top3_list[i-1]:.2%}" if top3_list else "-",
                "Top-5":      f"{top5_list[i-1]:.2%}" if top5_list else "-",
                "Time":       _fmt_time(epoch_times[i-1]) if epoch_times else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
