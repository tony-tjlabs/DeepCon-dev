"""
src.utils.anonymizer — DEPRECATED shim (M1 Security 2026-04-18)
================================================================
이 모듈은 얇은 호환성 shim 이다. 새 코드는 다음을 import 하라.

    from core.security.anonymizer import mask_name, mask_names_in_df

진짜 구현은 core.security.anonymizer.core 에 있다.
모든 호출은 core 로 위임된다. 여러 모듈이 이 shim 을 의존하고 있어
즉시 삭제는 하지 않지만, 최초 import 시 DeprecationWarning 을 발생시킨다.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "src.utils.anonymizer 는 deprecated 입니다. "
    "core.security.anonymizer 에서 import 하세요.",
    DeprecationWarning,
    stacklevel=2,
)

# core 모듈로 위임
from core.security.anonymizer.core import (  # noqa: E402,F401
    mask_name,
    mask_names_in_df,
    mask_name_series,
)

__all__ = ["mask_name", "mask_names_in_df", "mask_name_series"]
