"""
core.ai.llm_gateway — DeepCon 단일 LLM 게이트웨이 (M2-A T-10, T-11-ext)
========================================================================
모든 LLM 호출은 이 게이트웨이를 통과한다. 다른 경로는 기존 shim만 허용.

책임:
  1) Context 를 AnonymizationPipeline 에 통과 (익명화 강제 — 우회 불가)
  2) role_prompts 에서 system prompt + instruction 로드
  3) Sector Brief 주입 (일별 고정 prefix — prompt cache 대상 2계층)
  4) Anthropic SDK 호출 (cache_control 배선 포함. 현재 prefix 가 짧아 실제 cache_write=0)
  5) 응답의 PII 유출 감지 (AnonymizationPipeline.verify) — strict=차단, warn=로그
  6) AuditLogger 로 영속 감사 (prompt/response 원문 저장 금지, 해시만)

설계 참조:
  - upgrade_v3_02_architecture.md §3.1 ~ §3.5
  - upgrade_v3_03_security.md §3.C-1 ~ §3.C-5
  - upgrade_v3_05_llm_cost_analysis.md (Haiku 유지 결정)

모델:
  - 환경변수 `ANTHROPIC_MODEL` 로 중앙 관리 (기본 claude-haiku-4-5).
  - **코드에 모델명 하드코딩 금지**. shared.utils.claude_utils.get_claude_model() 사용.

Prompt Caching:
  - system[0] COMMON_DOMAIN_PRIMER + role system_base: cache_control=ephemeral.
  - system[1] Sector Brief + instruction: cache_control=ephemeral.
  - Haiku 는 cache_control 이 무시될 수 있으나 Sonnet 전환 시 자동 활성화.
  - usage.cache_creation_input_tokens / cache_read_input_tokens 가 있으면 감사에 반영.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from core.ai.audit_log import AuditLogger
from core.ai.role_prompts import (
    ROLE_PROMPTS,
    Role,
    get_role_spec,
    get_sector_brief,
)

log = logging.getLogger(__name__)

# shared.utils.claude_utils 는 anthropic 패키지가 없어도 import 는 성공해야 함.
try:
    from shared.utils.claude_utils import get_claude_model as _shared_get_model
except Exception:  # pragma: no cover
    _shared_get_model = None  # type: ignore


# ─── 리퀘스트 / 리스폰스 dataclass ───────────────────────────
@dataclass
class CommentaryRequest:
    """LLMGateway.analyze() 입력."""

    role: Role
    sector_id: str
    date_str: str | None
    context: dict  # 파이프라인 산출물 요약 (PII 포함 가능 — 내부에서 anonymize)
    user_query: str | None = None
    user_role: str = "unknown"  # "administrator" | sector id (감사 로그용)
    tab: str = ""  # 탭 식별자 (감사 로그용). 없으면 role에서 파생.
    stream: bool = True
    max_tokens: int | None = None  # None 이면 role 스펙 기본값 사용


@dataclass
class CommentaryResponse:
    """LLMGateway.analyze() 출력."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit_tokens: int = 0      # cache_read_input_tokens
    cache_write_tokens: int = 0    # cache_creation_input_tokens
    latency_ms: int = 0
    model: str = ""
    request_id: str = ""
    blocked: bool = False
    error: str | None = None
    pii_warnings: list[dict] = field(default_factory=list)

    def stream_chunks(self) -> Iterator[str]:
        """비스트리밍 응답을 청크로 변환 (UI 호환)."""
        if self.text:
            yield self.text


# ─── 기본 모델 조회 ─────────────────────────────────────────
def _resolve_model(override: str | None = None) -> str:
    if override:
        return override
    if _shared_get_model is not None:
        try:
            return _shared_get_model()
        except Exception:  # pragma: no cover
            pass
    return os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


# ─── Prompt Injection 방어: user_data 블록 격리 ─────────────
def _escape_user_block(text: str) -> str:
    """user_data 블록 안에 명령문이 있어도 LLM 이 따르지 않도록 fence 격리."""
    if text is None:
        return ""
    # 기존 ``` 을 non-fence로 치환
    safe = str(text).replace("```", "\\u0060\\u0060\\u0060")
    return f"```user_data\n{safe}\n```"


# ─── Context 직렬화 ─────────────────────────────────────────
def _serialize_context(context: dict, user_query: str | None) -> str:
    """dict context 를 LLM 에 전달할 텍스트로 직렬화. JSON 형태로 단순화."""
    import json

    try:
        payload = json.dumps(context, ensure_ascii=False, indent=2, default=str)
    except Exception:
        payload = str(context)
    body = f"[context]\n{payload}"
    if user_query:
        body += f"\n\n[user_query]\n{user_query}"
    return body


# ─── Anonymization 어댑터 ────────────────────────────────────
def _default_anonymizer():
    """
    기본 AnonymizationPipeline (core.security.anonymizer 기반).

    M1 에서 core 로 통합된 AnonymizationPipeline 을 사용.
    """
    from core.security.anonymizer import AnonymizationPipeline as _CoreAnon
    return _CoreAnon()


# ─── 메인 게이트웨이 ────────────────────────────────────────
class LLMGateway:
    """
    DeepCon 의 유일한 LLM 진입점.

    인스턴스화:
        gw = LLMGateway()                             # 기본: Haiku + warn 모드
        gw = LLMGateway(strict_pii=True)              # 응답 PII 감지 시 차단
        gw = LLMGateway(audit_logger=my_logger,
                        anonymization_pipeline=my_anon,
                        model="claude-haiku-4-5")

    호출:
        resp = gw.analyze(CommentaryRequest(role="overview_commentator", ...))
        for chunk in gw.analyze_stream(req): st.write(chunk)
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        anonymization_pipeline: Any | None = None,
        model: str | None = None,
        *,
        strict_pii: bool = False,
        audit_base_dir: Path | str | None = None,
    ):
        self._audit = audit_logger or AuditLogger(
            base_dir=audit_base_dir or Path("data/audit")
        )
        self._anon = anonymization_pipeline or _default_anonymizer()
        self._model = _resolve_model(model)
        self._strict_pii = bool(strict_pii)

    # ─── public API ─────────────────────────────────────
    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """API 키 존재 여부."""
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def analyze(self, req: CommentaryRequest) -> CommentaryResponse:
        """
        Non-streaming 호출. Streaming 이 요청돼도 내부에서 합쳐 반환.
        """
        return self._run(req, stream=False)

    def analyze_stream(self, req: CommentaryRequest) -> Iterator[str]:
        """
        Streaming. 각 청크 yield. 완료 후에도 감사 로그 기록.
        """
        yield from self._run(req, stream=True)

    # ─── 내부 ───────────────────────────────────────────
    def _run(
        self,
        req: CommentaryRequest,
        *,
        stream: bool,
    ) -> Any:
        request_id = str(uuid.uuid4())
        t0 = time.time()
        tab = req.tab or req.role.split("_")[0]

        # 1) role 유효성
        try:
            spec = get_role_spec(req.role)
        except KeyError as e:
            log.warning("[LLMGateway] unknown role: %s", e)
            if stream:
                return iter([f"[LLM 오류: 알 수 없는 역할 {req.role}]"])
            return CommentaryResponse(
                text=f"[LLM 오류: 알 수 없는 역할 {req.role}]",
                request_id=request_id,
                model=self._model,
                error=str(e),
                latency_ms=int((time.time() - t0) * 1000),
            )

        # 2) 시스템 / 사용자 메시지 조립 (익명화 + 프롬프트 설계)
        try:
            system_blocks, user_text, anon_warnings = self._build_messages(req, spec)
        except Exception as e:
            log.exception("[LLMGateway] _build_messages failed: %s", e)
            if stream:
                return iter([f"[LLM 오류: 컨텍스트 구성 실패]"])
            return CommentaryResponse(
                text="[LLM 오류: 컨텍스트 구성 실패]",
                request_id=request_id,
                model=self._model,
                error=str(e),
                latency_ms=int((time.time() - t0) * 1000),
            )

        # 3) Anthropic 호출
        try:
            import anthropic  # type: ignore
        except ImportError:
            msg = "[AI 비활성화] anthropic 패키지 미설치."
            if stream:
                return iter([msg])
            return CommentaryResponse(text=msg, request_id=request_id, error="no_anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            msg = (
                "[AI 비활성화] ANTHROPIC_API_KEY 가 설정되지 않았습니다. "
                ".env 에 API 키를 추가하거나 관리자에게 문의하세요."
            )
            if stream:
                return iter([msg])
            return CommentaryResponse(text=msg, request_id=request_id, error="no_api_key")

        client = anthropic.Anthropic(api_key=api_key)
        max_tokens = req.max_tokens or spec.get("max_tokens") or 600

        if stream:
            return self._call_stream(
                client=client,
                req=req,
                system_blocks=system_blocks,
                user_text=user_text,
                max_tokens=max_tokens,
                request_id=request_id,
                tab=tab,
                anon_warnings=anon_warnings,
                t0=t0,
            )
        return self._call_sync(
            client=client,
            req=req,
            system_blocks=system_blocks,
            user_text=user_text,
            max_tokens=max_tokens,
            request_id=request_id,
            tab=tab,
            anon_warnings=anon_warnings,
            t0=t0,
        )

    # ─── 프롬프트 조립 ─────────────────────────────────
    def _build_messages(
        self,
        req: CommentaryRequest,
        spec: dict,
    ) -> tuple[list[dict], str, list[dict]]:
        """
        Returns:
            system_blocks: Anthropic messages.create 의 system 파라미터용 list
                           (cache_control=ephemeral 배선 포함)
            user_text:     user role 메시지 본문 (fence 격리된 context)
            anon_warnings: 익명화 후 남은 PII 경고 list
        """
        # 1. 익명화 — 우회 불가 게이트
        raw_ctx_text = _serialize_context(req.context, req.user_query)
        anon_result = self._anon.run(raw_ctx_text)
        safe_text = anon_result.masked_text
        anon_warnings = [w.to_dict() for w in getattr(anon_result, "warnings", [])]
        if anon_warnings:
            log.info(
                "[LLMGateway] anonymizer left %d warning(s): %s",
                len(anon_warnings),
                [w["kind"] for w in anon_warnings],
            )

        # 2. system 블록 — prompt cache 배선
        #    Tier 1 : common primer + role system_base (역할별 고정)
        #    Tier 2 : sector brief + instruction + output format (일별 고정)
        system_base = spec["system_base"]
        sector_brief = get_sector_brief(req.sector_id)
        instruction_block = (
            f"{sector_brief}\n\n"
            f"[분석 지시]\n{spec['instruction']}\n\n"
            f"[출력 포맷]\n{spec['output_format']}"
        )

        system_blocks: list[dict] = [
            {
                "type": "text",
                "text": system_base,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": instruction_block,
                "cache_control": {"type": "ephemeral"},
            },
        ]

        # 3. user 메시지 — context 는 fence 격리
        user_text = _escape_user_block(safe_text)

        return system_blocks, user_text, anon_warnings

    # ─── 동기 호출 ─────────────────────────────────────
    def _call_sync(
        self,
        *,
        client: Any,
        req: CommentaryRequest,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        request_id: str,
        tab: str,
        anon_warnings: list[dict],
        t0: float,
    ) -> CommentaryResponse:
        try:
            msg = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_text}],
            )
            text = self._extract_text(msg)
            usage = getattr(msg, "usage", None)
            tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        except Exception as e:
            log.warning("[LLMGateway] sync call failed: %s", e)
            return self._record_failure(
                req=req, request_id=request_id, tab=tab,
                user_text=user_text, error=str(e), t0=t0,
                anon_warnings=anon_warnings,
            )

        # 응답 PII 검증
        resp_warnings = self._verify_response(text)
        blocked = False
        final_text = text
        if resp_warnings and self._strict_pii:
            blocked = True
            final_text = "[응답 차단: PII 유출 감지]"

        resp = CommentaryResponse(
            text=final_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_hit_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=int((time.time() - t0) * 1000),
            model=self._model,
            request_id=request_id,
            blocked=blocked,
            pii_warnings=resp_warnings + anon_warnings,
        )

        self._audit.log(
            user_role=req.user_role,
            sector_id=req.sector_id,
            tab=tab,
            role=req.role,
            date_str=req.date_str,
            model=self._model,
            prompt=user_text,
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=resp.latency_ms,
            pii_warnings=resp.pii_warnings,
            blocked=blocked,
            request_id=request_id,
        )
        return resp

    # ─── 스트리밍 호출 ─────────────────────────────────
    def _call_stream(
        self,
        *,
        client: Any,
        req: CommentaryRequest,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        request_id: str,
        tab: str,
        anon_warnings: list[dict],
        t0: float,
    ) -> Iterator[str]:
        collected: list[str] = []
        tokens_in = tokens_out = cache_read = cache_write = 0
        error: str | None = None
        blocked = False
        resp_warnings: list[dict] = []

        # ── M4-T32: streaming usage 정확 수집 ─────────────────────
        # Anthropic SDK 스트림은 세 가지 경로로 usage 제공:
        #   (A) message_start 이벤트: input_tokens (+ cache_*_tokens) — 시작 시 확정
        #   (B) message_delta 이벤트: output_tokens 누적 — 생성 중 업데이트
        #   (C) get_final_message(): 종료 후 최종 usage (A+B의 최종값)
        # SDK 버전/Anthropic 측 응답 구조에 따라 일부 필드가 비어 있을 수 있어
        # 세 경로 모두에서 수집한 뒤 "가장 큰 값" 을 채택한다 (None/0 fallback 안전).
        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_text}],
            ) as stream:
                for event in stream:
                    # 텍스트 델타
                    ev_type = getattr(event, "type", "")
                    if ev_type == "content_block_delta":
                        delta_obj = getattr(event, "delta", None)
                        if delta_obj is not None:
                            text_piece = getattr(delta_obj, "text", None)
                            if text_piece:
                                collected.append(text_piece)
                                yield text_piece

                    # message_start — input_tokens + cache_*_tokens 확정
                    elif ev_type == "message_start":
                        msg_obj = getattr(event, "message", None)
                        msg_usage = getattr(msg_obj, "usage", None)
                        if msg_usage is not None:
                            tokens_in = max(
                                tokens_in,
                                int(getattr(msg_usage, "input_tokens", 0) or 0),
                            )
                            cache_read = max(
                                cache_read,
                                int(getattr(msg_usage, "cache_read_input_tokens", 0) or 0),
                            )
                            cache_write = max(
                                cache_write,
                                int(getattr(msg_usage, "cache_creation_input_tokens", 0) or 0),
                            )

                    # message_delta — output_tokens 누적 (최신값이 최종값)
                    elif ev_type == "message_delta":
                        delta_usage = getattr(event, "usage", None)
                        if delta_usage is not None:
                            tokens_out = max(
                                tokens_out,
                                int(getattr(delta_usage, "output_tokens", 0) or 0),
                            )

                # 스트림 종료 후 최종 usage 재확인 (fallback — SDK 구버전 대응)
                final_msg = stream.get_final_message()
                usage = getattr(final_msg, "usage", None)
                if usage is not None:
                    tokens_in = max(
                        tokens_in,
                        int(getattr(usage, "input_tokens", 0) or 0),
                    )
                    tokens_out = max(
                        tokens_out,
                        int(getattr(usage, "output_tokens", 0) or 0),
                    )
                    cache_read = max(
                        cache_read,
                        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
                    )
                    cache_write = max(
                        cache_write,
                        int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
                    )
        except Exception as e:
            log.warning("[LLMGateway] stream call failed: %s", e)
            error = str(e)
            err_msg = "\n\n[AI 스트리밍 오류 — 잠시 후 다시 시도해주세요]"
            collected.append(err_msg)
            yield err_msg

        text = "".join(collected)

        # 응답 PII 검증
        if text and not error:
            resp_warnings = self._verify_response(text)
            if resp_warnings and self._strict_pii:
                blocked = True
                warn_tail = "\n\n[경고: PII 유출 가능성 감지 — 응답 검토 필요]"
                collected.append(warn_tail)
                yield warn_tail

        latency_ms = int((time.time() - t0) * 1000)

        # 감사 로그 기록
        self._audit.log(
            user_role=req.user_role,
            sector_id=req.sector_id,
            tab=tab,
            role=req.role,
            date_str=req.date_str,
            model=self._model,
            prompt=user_text,
            response=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_ms=latency_ms,
            pii_warnings=resp_warnings + anon_warnings,
            blocked=blocked,
            error=error,
            request_id=request_id,
        )

    # ─── PII verify ─────────────────────────────────────
    def _verify_response(self, text: str) -> list[dict]:
        try:
            warnings = self._anon.verify(text) or []
        except Exception as e:  # pragma: no cover
            log.warning("[LLMGateway] verify failed: %s", e)
            return []
        out: list[dict] = []
        for w in warnings:
            if hasattr(w, "to_dict"):
                out.append(w.to_dict())
            elif isinstance(w, dict):
                out.append(w)
            else:  # pragma: no cover
                out.append({"kind": str(type(w).__name__), "match": "", "span": [0, 0]})
        return out

    # ─── 실패 기록 ─────────────────────────────────────
    def _record_failure(
        self,
        *,
        req: CommentaryRequest,
        request_id: str,
        tab: str,
        user_text: str,
        error: str,
        t0: float,
        anon_warnings: list[dict],
    ) -> CommentaryResponse:
        latency_ms = int((time.time() - t0) * 1000)
        self._audit.log(
            user_role=req.user_role,
            sector_id=req.sector_id,
            tab=tab,
            role=req.role,
            date_str=req.date_str,
            model=self._model,
            prompt=user_text,
            response="",
            tokens_in=0,
            tokens_out=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            pii_warnings=anon_warnings,
            blocked=False,
            error=error,
            request_id=request_id,
        )
        return CommentaryResponse(
            text=f"[AI 호출 오류 — 잠시 후 다시 시도해주세요]",
            latency_ms=latency_ms,
            model=self._model,
            request_id=request_id,
            error=error,
            pii_warnings=anon_warnings,
        )

    # ─── 응답 텍스트 추출 ──────────────────────────────
    @staticmethod
    def _extract_text(msg: Any) -> str:
        try:
            content = msg.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    # anthropic SDK: TextBlock.text
                    t = getattr(block, "text", None)
                    if t:
                        parts.append(t)
                return "".join(parts)
            if isinstance(content, str):
                return content
        except Exception:  # pragma: no cover
            pass
        return ""


# ─── 싱글턴 accessor (Streamlit 재실행 시 재사용) ─────
_SINGLETON: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    """프로세스 전역 LLMGateway 싱글턴 반환."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LLMGateway()
    return _SINGLETON


__all__ = [
    "CommentaryRequest",
    "CommentaryResponse",
    "LLMGateway",
    "get_gateway",
]
