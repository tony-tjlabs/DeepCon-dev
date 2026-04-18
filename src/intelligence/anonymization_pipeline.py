"""
src.intelligence.anonymization_pipeline — DEPRECATED shim (M2-B, 2026-04-18)
=============================================================================
이 모듈은 얇은 호환성 shim 이다. 새 코드는 다음을 import 하라:

    from core.security.anonymizer import (
        KAnonymizationPipeline,
        K_ANON_MIN,
    )

역사:
  - 원본 (M15X anonymization_pipeline.py 기반, Y1 K=20 상향) 은
    `core/security/anonymizer/k_anonymity.py` 로 이관됨.
  - 이 파일은 기존 호출자 (data_packager, llm_gateway) 의 import 경로를
    깨지 않기 위한 shim.

호환성 유지:
  - AnonymizationPipeline  — 이름 그대로 re-export (core 의 KAnonymizationPipeline)
  - K_ANON_MIN             — core 에서 re-export

보존 의미:
  기존 코드가 `from src.intelligence.anonymization_pipeline import AnonymizationPipeline`
  를 호출해도 실제로는 core 의 K-anonymity 파이프라인이 실행된다.
"""
from __future__ import annotations

import warnings

from core.security.anonymizer.k_anonymity import (
    K_ANON_MIN,
    KAnonymizationPipeline as _KAP,
    _LOCUS_GROUP_MAP,  # noqa: F401  (내부 상수, 테스트/디버깅용 유지)
)

# 단 1회 shim import 경고
_warned = False


def _warn_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        "src.intelligence.anonymization_pipeline 는 deprecated 입니다. "
        "`from core.security.anonymizer import KAnonymizationPipeline, K_ANON_MIN` "
        "로 이전하세요. (M3 에서 완전 제거 예정)",
        DeprecationWarning,
        stacklevel=3,
    )


_warn_once()


# ─── 호환 alias (기존 이름 유지) ─────────────────────────────────────────────
# 외부 코드가 `from src.intelligence.anonymization_pipeline import AnonymizationPipeline`
# 을 호출하면 core 의 KAnonymizationPipeline 이 그대로 반환된다.
AnonymizationPipeline = _KAP


__all__ = [
    "AnonymizationPipeline",
    "K_ANON_MIN",
]
