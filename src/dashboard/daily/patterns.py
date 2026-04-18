"""
Daily Tab - Patterns Section
============================
Journey 패턴 분석 탭 관련 함수들.
"""
from __future__ import annotations

import json as _json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    section_header, sub_header,
    PLOTLY_DARK, PLOTLY_LEGEND,
)

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND


def _load_cluster_labels(sid: str) -> dict | None:
    """cluster_labels.json을 직접 로드 (gensim 불필요)."""
    label_path = cfg.MODEL_DIR / sid / "cluster_labels.json"
    if not label_path.exists():
        return None
    try:
        with open(label_path, encoding="utf-8") as f:
            raw = _json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return None


def _render_patterns_summary(cluster_labels: dict):
    """Cloud용: 클러스터 분포 카드만 표시 (gensim 불필요)."""
    st.markdown("##### diamond 학습된 패턴 그룹")

    cols = st.columns(min(len(cluster_labels), 5))
    for cid, info in sorted(cluster_labels.items()):
        cname = info.get("name", f"패턴 {cid}")
        color = info.get("color", "#00AEEF")
        top_loci = info.get("loci", [])
        with cols[cid % 5]:
            st.markdown(
                f"<div style='background:#1A2A3A; border:1px solid {color}44; "
                f"border-top:3px solid {color}; border-radius:10px; "
                f"padding:12px 14px; text-align:center; margin-bottom:8px;'>"
                f"<div style='font-size:0.78rem; color:#C8D6E8; margin-top:4px;'>"
                f"{cname}</div>"
                f"<div style='font-size:0.7rem; color:#5A6A7A; margin-top:4px;'>"
                f"{' / '.join(top_loci[:3]) if top_loci else ''}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption("i 클러스터 할당 및 PCA 산점도는 로컬 환경에서 확인할 수 있습니다.")


def _render_patterns_full(
    worker_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    emb,
):
    """Local용: 전체 클러스터 분석 (할당 + 분포 카드 + PCA + 지표 비교)."""
    if journey_df.empty:
        st.warning("현재 날짜의 Journey 데이터 없음 (journey.parquet 미존재)")
        return

    if "locus_id" not in journey_df.columns:
        st.warning("journey.parquet에 locus_id 컬럼 없음")
        return

    # 클러스터 할당
    try:
        enriched_df = emb.assign_clusters(worker_df, journey_df)
    except Exception as e:
        st.error(f"클러스터 할당 실패: {e}")
        return

    has_cluster = "cluster_id" in enriched_df.columns and enriched_df["cluster_id"].notna().any()
    if not has_cluster:
        st.warning("클러스터 할당 결과 없음")
        return

    # 클러스터 분포 카드
    cluster_counts = (
        enriched_df[enriched_df["cluster_id"] >= 0]
        .groupby(["cluster_id", "cluster_name"])
        .size()
        .reset_index(name="count")
        .sort_values("cluster_id")
    )

    if not cluster_counts.empty:
        cols = st.columns(min(len(cluster_counts), 5))
        for i, row in cluster_counts.iterrows():
            cid = int(row["cluster_id"])
            cname = row["cluster_name"]
            cnt = row["count"]
            color = emb.cluster_labels.get(cid, {}).get("color", "#00AEEF")
            top_loci = emb.cluster_labels.get(cid, {}).get("loci", [])
            with cols[cid % 5]:
                st.markdown(
                    f"<div style='background:#1A2A3A; border:1px solid {color}44; "
                    f"border-top:3px solid {color}; border-radius:10px; "
                    f"padding:12px 14px; text-align:center; margin-bottom:8px;'>"
                    f"<div style='font-size:1.4rem; font-weight:800; color:{color}'>"
                    f"{cnt}명</div>"
                    f"<div style='font-size:0.78rem; color:#C8D6E8; margin-top:4px;'>"
                    f"{cname}</div>"
                    f"<div style='font-size:0.7rem; color:#5A6A7A; margin-top:4px;'>"
                    f"{' / '.join(top_loci[:2]) if top_loci else ''}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2D/3D 산점도 토글
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        # 투영 모드 선택
        proj_mode = st.radio(
            "투영 방식",
            ["2D (PCA)", "3D (PCA)"],
            horizontal=True,
            key="scatter_proj_mode",
        )
        is_3d = proj_mode == "3D (PCA)"

        st.markdown(f"**작업자 Journey 분포 (PCA {'3D' if is_3d else '2D'})**")
        try:
            proj_df = emb.get_2d_projection(journey_df, method="pca", n_components=3 if is_3d else 2)
            if not proj_df.empty:
                proj_df["cluster_name"] = proj_df["cluster_id"].map(
                    lambda c: emb.cluster_labels.get(c, {}).get("name", f"패턴 {c}")
                )
                color_map = {
                    emb.cluster_labels.get(c, {}).get("name", f"패턴 {c}"):
                    emb.cluster_labels.get(c, {}).get("color", "#00AEEF")
                    for c in emb.cluster_labels
                }

                if is_3d and "z" in proj_df.columns:
                    # 3D Scatter Plot
                    fig = px.scatter_3d(
                        proj_df, x="x", y="y", z="z",
                        color="cluster_name",
                        color_discrete_map=color_map,
                        hover_data=["user_no"],
                        labels={"x": "PC1", "y": "PC2", "z": "PC3", "cluster_name": "패턴"},
                        template="plotly_dark",
                    )
                    fig.update_layout(**_DARK, height=450)
                    fig.update_traces(marker=dict(size=4, opacity=0.8))
                else:
                    # 2D Scatter Plot
                    fig = px.scatter(
                        proj_df, x="x", y="y",
                        color="cluster_name",
                        color_discrete_map=color_map,
                        hover_data=["user_no"],
                        labels={"x": "PC1", "y": "PC2", "cluster_name": "패턴"},
                        template="plotly_dark",
                    )
                    fig.update_layout(**_DARK, height=380)
                    fig.update_traces(marker=dict(size=7, opacity=0.8))

                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"산점도 생성 실패: {e}")

    with col_table:
        st.markdown("**패턴별 지표 비교**")
        if "ewi" in enriched_df.columns and "cre" in enriched_df.columns:
            agg = (
                enriched_df[enriched_df["cluster_id"] >= 0]
                .groupby("cluster_name")
                .agg(
                    count=("user_no", "count"),
                    avg_ewi=("ewi", "mean"),
                    avg_cre=("cre", "mean"),
                )
                .round(3)
                .reset_index()
                .rename(columns={"cluster_name": "패턴", "count": "인원수", "avg_ewi": "평균EWI", "avg_cre": "평균CRE"})
            )
            st.dataframe(agg, use_container_width=True, hide_index=True)
        else:
            agg = (
                enriched_df[enriched_df["cluster_id"] >= 0]
                .groupby("cluster_name")
                .size()
                .reset_index(name="인원수")
                .rename(columns={"cluster_name": "패턴"})
            )
            st.dataframe(agg, use_container_width=True, hide_index=True)

    # Deep Space 모델 예측/이상탐지 섹션
    st.divider()
    _render_deep_space_analysis(journey_df, emb.sector_id if hasattr(emb, 'sector_id') else cfg.SECTOR_ID)


def _render_deep_space_analysis(journey_df: pd.DataFrame, sector_id: str):
    """
    Deep Space 모델 기반 분석.

    - 이동 확률 매트릭스 (상위 Locus 전이 확률)
    - 다음 이동 예측 시연
    - 이상 패턴 탐지 결과
    """
    st.markdown("##### rocket Deep Space 모델 분석")

    # 모델 존재 여부 확인
    try:
        from src.model.trainer import get_model_info, load_trained_model
        from src.model.tokenizer import get_tokenizer
        model_info = get_model_info(sector_id)
    except ImportError:
        st.caption("i Deep Space 모델을 사용하려면 PyTorch가 필요합니다.")
        return
    except Exception:
        model_info = {"exists": False}

    if not model_info.get("exists"):
        st.info(
            "signal Deep Space 모델이 없습니다.  \n"
            "**[파이프라인] 탭 -> rocket Deep Space Foundation Model** 섹션에서 학습하세요."
        )
        return

    # 모델 정보 표시
    n_params = model_info.get("n_params", 0)
    best_loss = model_info.get("best_val_loss", 0)
    val_acc_top3 = model_info.get("val_acc_top3", [])
    acc_str = f"{val_acc_top3[-1]:.1%}" if val_acc_top3 else "N/A"

    st.markdown(
        f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; "
        f"border-radius:8px; padding:10px 14px; font-size:0.85rem; color:#C8D6E8;'>"
        f"<b>모델 상태</b>: {n_params:,} params &nbsp;|&nbsp; "
        f"Val Loss {best_loss:.3f} &nbsp;|&nbsp; Top-3 Acc {acc_str}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 서브탭: 예측 시연 / 이상 탐지 / 전이 확률
    sub_tabs = st.tabs(["target 이동 예측", "warning 이상 패턴", "chart 전이 확률"])

    with sub_tabs[0]:
        _render_next_prediction_demo(sector_id)

    with sub_tabs[1]:
        _render_anomaly_detection(journey_df, sector_id)

    with sub_tabs[2]:
        _render_transition_matrix(sector_id)


def _render_next_prediction_demo(sector_id: str):
    """다음 이동 예측 시연 UI."""
    try:
        from src.model.downstream.predictor import get_predictor
        from src.model.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(sector_id)
        if tokenizer is None:
            st.warning("토크나이저 없음")
            return

        locus_ids = tokenizer.locus_ids

        st.markdown("**현재 위치 시퀀스를 선택하면 다음 이동 위치를 예측합니다.**")

        # Locus 선택
        selected_loci = st.multiselect(
            "현재 이동 경로 (순서대로 선택)",
            options=locus_ids,
            default=locus_ids[:2] if len(locus_ids) >= 2 else locus_ids,
            max_selections=10,
            key="deep_space_pred_input",
            format_func=lambda x: f"{x} ({tokenizer.get_locus_info(x).get('locus_name', '')})",
        )

        if not selected_loci:
            st.caption("Locus를 1개 이상 선택하세요.")
            return

        # 예측 버튼
        if st.button("target 다음 위치 예측", key="predict_next_btn"):
            predictor = get_predictor(sector_id)
            if predictor is None:
                st.error("예측기 로드 실패")
                return

            predictions = predictor.predict_next(selected_loci, top_k=5)

            if predictions:
                st.markdown("**예측 결과 (Top-5)**")
                for i, (locus_id, prob) in enumerate(predictions, 1):
                    info = tokenizer.get_locus_info(locus_id)
                    name = info.get("locus_name", locus_id)
                    bar_width = int(prob * 100)
                    st.markdown(
                        f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
                        f"<div style='width:20px; color:#7A8FA6;'>#{i}</div>"
                        f"<div style='flex:1; margin-left:8px;'>"
                        f"<div style='background:#1A3A5C; border-radius:4px; padding:4px 8px;'>"
                        f"<div style='background:#00AEEF; height:6px; width:{bar_width}%; "
                        f"border-radius:3px; margin-bottom:4px;'></div>"
                        f"<span style='font-size:0.85rem; color:#C8D6E8;'>{name}</span>"
                        f"<span style='font-size:0.75rem; color:#5A6A7A; margin-left:8px;'>"
                        f"{prob:.1%}</span></div></div></div>",
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"예측 실패: {e}")


def _render_anomaly_detection(journey_df: pd.DataFrame, sector_id: str):
    """이상 패턴 탐지 결과 표시."""
    st.markdown("**Journey 시퀀스의 이상 점수 (Perplexity 기반)**")

    if journey_df.empty or "locus_id" not in journey_df.columns:
        st.info("Journey 데이터가 없습니다.")
        return

    try:
        from src.model.downstream.anomaly import get_detector

        detector = get_detector(sector_id)
        if detector is None:
            st.warning("이상 탐지기 로드 실패")
            return

        # 작업자별 시퀀스 추출 (상위 100명만)
        sequences = []
        user_ids = []
        for user_no, grp in journey_df.groupby("user_no"):
            locus_seq = grp.sort_values("seq")["locus_id"].dropna().astype(str).tolist()
            if len(locus_seq) >= 5:
                sequences.append(locus_seq)
                user_ids.append(str(user_no))
            if len(sequences) >= 100:
                break

        if not sequences:
            st.info("분석할 시퀀스가 없습니다.")
            return

        # 이상 탐지 실행
        with st.spinner("이상 패턴 분석 중..."):
            results = detector.score_batch(sequences, user_ids)

        # 정렬 (perplexity 높은 순)
        results.sort(key=lambda r: r.perplexity, reverse=True)

        # 통계 표시
        perplexities = [r.perplexity for r in results]
        avg_ppl = sum(perplexities) / len(perplexities)
        anomaly_count = sum(1 for r in results if r.is_anomaly)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("분석 대상", f"{len(results)}명")
        with col2:
            st.metric("평균 Perplexity", f"{avg_ppl:.2f}")
        with col3:
            color = "#FF4C4C" if anomaly_count > 0 else "#00C897"
            st.markdown(
                f"<div style='text-align:center; padding:8px;'>"
                f"<div style='font-size:1.1rem; font-weight:700; color:{color};'>"
                f"{anomaly_count}</div>"
                f"<div style='font-size:0.78rem; color:#7A8FA6;'>이상 탐지</div></div>",
                unsafe_allow_html=True,
            )

        # 이상 작업자 목록 (Top 10)
        if anomaly_count > 0:
            st.markdown("**이상 패턴 작업자 (Top 10)**")
            anomalies = [r for r in results if r.is_anomaly][:10]
            for r in anomalies:
                st.markdown(
                    f"<div style='background:#2A1A1A; border-left:3px solid #FF4C4C; "
                    f"padding:8px 12px; border-radius:4px; margin-bottom:4px;'>"
                    f"<span style='color:#FFB300; font-weight:600;'>{r.user_no}</span> "
                    f"<span style='color:#7A8FA6; font-size:0.85rem;'>"
                    f"Perplexity {r.perplexity:.1f} &nbsp;|&nbsp; "
                    f"시퀀스 {r.sequence_length}개</span></div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"이상 탐지 실패: {e}")


def _render_transition_matrix(sector_id: str):
    """전이 확률 매트릭스 히트맵."""
    st.markdown("**Locus 간 이동 확률 (상위 20개)**")

    try:
        from src.model.downstream.predictor import get_predictor
        from src.model.tokenizer import get_tokenizer

        predictor = get_predictor(sector_id)
        if predictor is None:
            st.warning("예측기 로드 실패")
            return

        tokenizer = get_tokenizer(sector_id)

        # 상위 20개 Locus만 (시각화 위해)
        locus_ids = tokenizer.locus_ids[:20]

        with st.spinner("전이 확률 계산 중..."):
            # 간단한 전이 확률 계산 (각 Locus에서 Top-5)
            n = len(locus_ids)
            matrix = np.zeros((n, n))

            for i, from_locus in enumerate(locus_ids):
                preds = predictor.predict_next([from_locus], top_k=n)
                pred_dict = {loc: prob for loc, prob in preds}
                for j, to_locus in enumerate(locus_ids):
                    matrix[i, j] = pred_dict.get(to_locus, 0.0)

        # 히트맵 생성
        locus_names = [tokenizer.get_locus_info(lid).get("locus_name", lid)[:12] for lid in locus_ids]

        fig = px.imshow(
            matrix,
            x=locus_names,
            y=locus_names,
            color_continuous_scale="Blues",
            labels={"x": "To", "y": "From", "color": "Probability"},
            template="plotly_dark",
        )
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0D1B2A",
            plot_bgcolor="#0D1B2A",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"전이 확률 계산 실패: {e}")


def render_journey_patterns(
    worker_df: pd.DataFrame,
    journey_df: pd.DataFrame,
    sid: str,
):
    """
    Journey 임베딩 클러스터링 결과를 시각화.

    모델이 없으면 학습 안내.
    Cloud: cluster_labels.json만 읽어서 분포 카드 표시 (gensim 불필요).
    Local: 전체 기능 (클러스터 할당 + PCA 산점도).
    """
    st.markdown(section_header("dna Journey 패턴 분석"), unsafe_allow_html=True)
    st.caption(
        "작업자의 하루 이동 시퀀스(Locus 토큰)를 Word2Vec으로 임베딩하고 "
        "K-means로 패턴을 자동 분류합니다."
    )

    # 임베딩 모듈 로드 시도 (로컬)
    emb = None
    try:
        from src.intelligence.journey_embedding import get_or_load_embedder
        emb = get_or_load_embedder(sid)
    except ImportError:
        pass  # Cloud 환경 - gensim 없음

    # 클러스터 라벨 로드 (Cloud/Local 공통)
    cluster_labels = _load_cluster_labels(sid)

    if emb is None and cluster_labels is None:
        st.info(
            "signal Journey 임베딩 모델이 없습니다.  \n"
            "**[파이프라인] 탭 -> dna Journey 임베딩 모델** 섹션에서 학습하세요."
        )
        return

    # Local: 전체 기능 (임베딩 모듈 사용 가능)
    if emb is not None:
        _render_patterns_full(worker_df, journey_df, emb)
    else:
        # Cloud: cluster_labels.json 기반 요약만 표시
        _render_patterns_summary(cluster_labels)
