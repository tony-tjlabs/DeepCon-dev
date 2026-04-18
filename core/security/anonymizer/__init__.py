"""
core.security.anonymizer — 단일 PII 마스킹 게이트 (M1/M2 Security)
=====================================================================
모든 LLM 호출은 이 모듈을 통과해야 한다.

구성:
  core.py         — PII 마스킹 규칙 (이름/사번/twardid/자유텍스트)
  llm_pipeline.py — LLM 파이프라인 훅 (AnonymizationPipeline, LLMGateway)
  k_anonymity.py  — K-anonymity 파이프라인 (Y1 전용, K=20)

기존 shim:
  src/utils/anonymizer.py                  — 얇은 shim
  src/pipeline/anonymizer.py               — 얇은 shim + Anonymizer 코드명 치환식 유지
  src/intelligence/anonymization_pipeline.py — KAnonymizationPipeline 위임 shim
"""
from .core import (
    mask_name,
    mask_name_series,
    mask_names_in_df,
    mask_user_no,
    mask_twardid,
    mask_free_text,
    detect_leftover_pii,
)
from .llm_pipeline import (
    AnonymizationPipeline,
    AnonymizationResult,
    PIIWarning,
)
from .k_anonymity import (
    K_ANON_MIN,
    KAnonymizationPipeline,
)

__all__ = [
    "mask_name",
    "mask_name_series",
    "mask_names_in_df",
    "mask_user_no",
    "mask_twardid",
    "mask_free_text",
    "detect_leftover_pii",
    "AnonymizationPipeline",
    "AnonymizationResult",
    "PIIWarning",
    "K_ANON_MIN",
    "KAnonymizationPipeline",
]
