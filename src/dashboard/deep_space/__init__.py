"""
Deep Space 분석 탭 패키지
========================
Transformer 기반 Spatial AI 분석.

서브모듈 구조:
    model_loader.py     - 모델/데이터 로드 함수
    helpers.py          - 공용 헬퍼 함수 (예측, 이상탐지, 임베딩)
    agentic.py          - ★ Agentic AI 탭 (혼잡/병목/안전/생산성 예측)
    overview.py         - 개요 서브탭
    training.py         - 학습 현황 서브탭
    prediction.py       - 이동 예측 서브탭
    simulation.py       - 현장 시뮬레이션 서브탭
    locus_prediction.py - 공간 예측 서브탭
    anomaly.py          - 이상 이동 탐지 서브탭
    spatial.py          - 공간 관계 서브탭

핵심 엔진:
    src/intelligence/spatial_predictor.py
        - Intelligence Engine (EWI/CRE/SII) + Deep Space (Transformer) 통합
        - SpatialPredictor → AgenticReport (4종 예측)
"""
from src.dashboard.deep_space.model_loader import (
    load_model,
    load_journey_data,
    load_worker_data,
    get_available_dates,
    load_training_history,
    load_locus_info,
    build_locus_meta,
)

from src.dashboard.deep_space.helpers import (
    extract_worker_sequences,
    predict_next,
    predict_next_batch,
    compute_anomaly_score,
    compute_locus_embeddings,
    compute_similarity_matrix,
    classify_confidence,
    classify_severity,
    get_inflow_outflow,
    render_card,
)

from src.dashboard.deep_space.overview import render_overview
from src.dashboard.deep_space.training import render_training
from src.dashboard.deep_space.prediction import render_prediction
from src.dashboard.deep_space.simulation import (
    render_simulation,
    compute_simulation_state,
    _cached_extract_sequences,
    _build_transition_matrix,
    _apply_transition_matrix,
    _build_predicted_locus_df,
)
from src.dashboard.deep_space.locus_prediction import (
    render_locus_prediction,
    predict_locus_states,
    build_locus_context,
    build_simulation_locus_context,
    generate_locus_insights,
)
from src.dashboard.deep_space.anomaly import render_anomaly
from src.dashboard.deep_space.spatial import render_spatial_relations
from src.dashboard.deep_space.agentic import render_agentic

__all__ = [
    # model_loader
    "load_model",
    "load_journey_data",
    "load_worker_data",
    "get_available_dates",
    "load_training_history",
    "load_locus_info",
    "build_locus_meta",
    # helpers
    "extract_worker_sequences",
    "predict_next",
    "predict_next_batch",
    "compute_anomaly_score",
    "compute_locus_embeddings",
    "compute_similarity_matrix",
    "classify_confidence",
    "classify_severity",
    "get_inflow_outflow",
    "render_card",
    # subtabs
    "render_overview",
    "render_training",
    "render_prediction",
    "render_simulation",
    "compute_simulation_state",
    "_cached_extract_sequences",
    "_build_transition_matrix",
    "_apply_transition_matrix",
    "_build_predicted_locus_df",
    "render_locus_prediction",
    "predict_locus_states",
    "build_locus_context",
    "build_simulation_locus_context",
    "generate_locus_insights",
    "render_anomaly",
    "render_spatial_relations",
    "render_agentic",
]
