"""
Deep Space Downstream Tasks
===========================
사전학습된 Deep Space 모델을 활용한 다운스트림 태스크.

모듈:
  - predictor: 다음 Locus 예측
  - anomaly: 이상 탐지 (perplexity 기반)
"""
from src.model.downstream.predictor import NextLocusPredictor
from src.model.downstream.anomaly import AnomalyDetector

__all__ = [
    "NextLocusPredictor",
    "AnomalyDetector",
]
