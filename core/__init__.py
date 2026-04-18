"""
DeepCon Core Layer
=================
도메인 무관 범용 모듈 — Locus 스키마, 레지스트리, 보안, 추론 API.

레이어 구조:
  core/schema/     — Pydantic v2 데이터 모델 (LocusBase, DeepConRecord 등)
  core/registry/   — Locus 조회/관리/자연어 변환
  core/api/        — Deep Space 모델 추론 인터페이스
  core/security/   — LLM 데이터 보안 + 코드 보호
"""

__version__ = "0.1.0"
__author__ = "TJLABS"
