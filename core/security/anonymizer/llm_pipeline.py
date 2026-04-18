"""
core.security.anonymizer.llm_pipeline — LLM 단일 게이트
========================================================
모든 LLM 호출의 입력을 마스킹하고, 응답의 PII 유출을 감지한다.

M1 범위 (2026-04-18):
  - AnonymizationPipeline: 마스킹 + verify 로그 WARNING.
  - 실제 호출 차단(M2)은 아직 강제하지 않음.
    대신 verify() 로 "남은 PII 가 있으면 WARNING 로그" 만 남긴다.

설계 참조: upgrade_v3_03_security.md §3.C
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from .core import (
    detect_leftover_pii,
    mask_free_text,
    mask_name,
    mask_twardid,
    mask_user_no,
)

# 숫자형 값도 마스킹해야 하는 키 목록 (int/float 도 PII 판단)
# 예: {"user_no": 12345} → {"user_no": "A-xxxxx"}
_NUMERIC_PII_KEYS: frozenset[str] = frozenset({
    "user_no",
    "worker_no",
    "employee_no",
    "사번",
})

log = logging.getLogger(__name__)


# ─── 결과 타입 ──────────────────────────────────────────────
@dataclass
class PIIWarning:
    kind: str          # 패턴 종류 (kr_name_honorific / user_no_5digit / raw_twardid)
    match: str         # 매칭된 원문
    span: tuple[int, int]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "match": self.match, "span": list(self.span)}


@dataclass
class AnonymizationResult:
    masked_text: str
    mapping: dict[str, str] = field(default_factory=dict)  # original → masked
    warnings: list[PIIWarning] = field(default_factory=list)
    mask_events: int = 0

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ─── 파이프라인 ─────────────────────────────────────────────
class AnonymizationPipeline:
    """
    LLM 전송 payload 를 마스킹하는 단일 게이트.

    사용:
        pipe = AnonymizationPipeline()
        result = pipe.run(text,
                          known_names=["홍길동"],
                          known_user_nos=["32763"])
        # LLM 에는 result.masked_text 만 전달.
        # result.warnings 가 비어있지 않으면 잔여 PII 존재 (M1: 로그만, M2: 차단).

        # 응답 검증
        response_warnings = pipe.verify(response_text)
        if response_warnings:
            log.warning("LLM 응답에 잔여 PII 감지: %s", response_warnings)
    """

    def __init__(self, *, strict: bool = False) -> None:
        """
        Args:
            strict: True 면 verify() 에서 잔여 PII 발견 시 예외.
                    M1 범위에서는 False 가 기본 (로그만).
        """
        self._strict = strict

    # ─── payload 마스킹 ─────────────────────────────────
    def run(
        self,
        payload: Any,
        *,
        known_names: Iterable[str] | None = None,
        known_user_nos: Iterable[str] | None = None,
    ) -> AnonymizationResult:
        """payload 를 마스킹한다.

        - str        : mask_free_text 적용
        - dict/list  : 재귀 마스킹 (str 필드만 마스킹, 그 외 값은 유지)
        - 그 외      : str 로 변환 후 마스킹
        """
        names = list(known_names) if known_names else []
        nos = list(known_user_nos) if known_user_nos else []

        # mapping 수집
        mapping: dict[str, str] = {}
        for n in names:
            if n and isinstance(n, str):
                mapping[n] = mask_name(n)
        for u in nos:
            if u is not None:
                mapping[str(u)] = mask_user_no(u)

        if isinstance(payload, str):
            masked = mask_free_text(
                payload, known_names=names, known_user_nos=nos
            )
            warnings_list = self._collect_warnings(masked)
            return AnonymizationResult(
                masked_text=masked,
                mapping=mapping,
                warnings=warnings_list,
                mask_events=self._count_events(payload, masked),
            )

        if isinstance(payload, (dict, list)):
            masked_struct, events = _mask_nested(payload, names, nos)
            # 구조체는 flatten 후 warning 스캔
            flat = _flatten_to_text(masked_struct)
            warnings_list = self._collect_warnings(flat)
            return AnonymizationResult(
                masked_text=flat,
                mapping=mapping,
                warnings=warnings_list,
                mask_events=events,
            )

        # fallback
        text = str(payload)
        masked = mask_free_text(text, known_names=names, known_user_nos=nos)
        warnings_list = self._collect_warnings(masked)
        return AnonymizationResult(
            masked_text=masked,
            mapping=mapping,
            warnings=warnings_list,
            mask_events=self._count_events(text, masked),
        )

    # ─── 응답 검증 ─────────────────────────────────────
    def verify(self, text: str) -> list[PIIWarning]:
        """
        LLM 응답 / 임의 텍스트에 잔여 PII 패턴이 있는지 검사.

        M1: 로그 WARNING.
        M2 (예정): strict=True 면 예외 발생.
        """
        warnings_list = self._collect_warnings(text)
        if warnings_list:
            log.warning(
                "[anonymizer.verify] %d leftover PII pattern(s) detected: %s",
                len(warnings_list),
                [w.kind for w in warnings_list],
            )
            if self._strict:
                raise ValueError(
                    f"PII leak detected (strict mode): {warnings_list}"
                )
        return warnings_list

    # ─── internal ────────────────────────────────────
    @staticmethod
    def _collect_warnings(text: str) -> list[PIIWarning]:
        return [
            PIIWarning(kind=f["kind"], match=f["match"], span=f["span"])
            for f in detect_leftover_pii(text or "")
        ]

    @staticmethod
    def _count_events(before: str, after: str) -> int:
        # 단순 추정: ** 개수 증가분 + A- alias 개수
        if not before or not after:
            return 0
        return max(0, after.count("**") - before.count("**")) + (
            after.count("A-") - before.count("A-")
        )


# ─── 재귀 마스킹 helper ───────────────────────────────────
def _mask_nested(
    obj: Any, names: list[str], nos: list[str], parent_key: str | None = None
) -> tuple[Any, int]:
    """
    dict/list 구조를 재귀적으로 순회하며 PII 를 마스킹.

    - str         : mask_free_text 적용
    - int/float   : parent_key 가 PII 키(user_no 등) 면 alias 로 치환, 그 외는 유지
    - bool        : 유지 (int 로 취급되지만 PII 아님)
    - dict/list   : 재귀
    - 기타        : 유지
    """
    events = 0
    if isinstance(obj, dict):
        result: dict = {}
        for k, v in obj.items():
            new_v, ev = _mask_nested(v, names, nos, parent_key=str(k))
            result[k] = new_v
            events += ev
        return result, events
    if isinstance(obj, list):
        out_list: list = []
        for item in obj:
            # list 아이템은 parent_key 를 상속 (예: {"user_no": [12345, 67890]})
            new_v, ev = _mask_nested(item, names, nos, parent_key=parent_key)
            out_list.append(new_v)
            events += ev
        return out_list, events
    if isinstance(obj, str):
        masked = mask_free_text(obj, known_names=names, known_user_nos=nos)
        events += abs(len(masked) - len(obj))
        return masked, events
    # bool 은 int 의 서브타입이므로 먼저 걸러낸다.
    if isinstance(obj, bool):
        return obj, 0
    if isinstance(obj, (int, float)):
        # parent_key 가 PII 숫자 키면 alias 로 치환
        if parent_key and parent_key.lower() in _NUMERIC_PII_KEYS:
            alias = mask_user_no(obj)
            if alias:
                return alias, 1
        return obj, 0
    return obj, 0


def _flatten_to_text(obj: Any) -> str:
    """중첩 dict/list 를 검사용 단일 텍스트로 평탄화 (verify 용)."""
    if isinstance(obj, dict):
        return " ".join(
            f"{k}: {_flatten_to_text(v)}" for k, v in obj.items()
        )
    if isinstance(obj, list):
        return " ".join(_flatten_to_text(v) for v in obj)
    return str(obj)
