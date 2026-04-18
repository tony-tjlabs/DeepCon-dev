"""
core.ai — DeepCon AI 단일 게이트웨이 (M2-A)
============================================
모든 LLM 호출은 이 패키지의 LLMGateway 를 통과한다.

공개:
  llm_gateway — LLMGateway, CommentaryRequest, CommentaryResponse, get_gateway
  role_prompts — ROLE_PROMPTS, ROLE_LABELS, SECTOR_BRIEFS, get_role_spec, get_sector_brief
  audit_log   — AuditLogger, read_audit_log, list_audit_months

마이그레이션 대상 (M2-B):
  기존 `src/dashboard/llm_deepcon.*` 및 `src/intelligence/llm_gateway.*` 의 모든
  `call_claude` 호출 지점은 아래 인터페이스로 교체될 예정.
"""
from .audit_log import (
    ANONYMIZATION_VERSION,
    AuditEntry,
    AuditLogger,
    list_audit_months,
    read_audit_log,
)
from .context_builders import (
    build_anomaly_context,
    build_deep_space_context,
    build_integrity_context,
    build_overview_context,
    build_prediction_context,
    build_productivity_context,
    build_report_section_context,
    build_safety_context,
    build_zone_time_context,
)
from .llm_gateway import (
    CommentaryRequest,
    CommentaryResponse,
    LLMGateway,
    get_gateway,
)
from .role_prompts import (
    ROLE_LABELS,
    ROLE_PROMPTS,
    SECTOR_BRIEFS,
    Role,
    RolePromptSpec,
    get_role_spec,
    get_sector_brief,
    validate_catalog,
)

__all__ = [
    # llm_gateway
    "CommentaryRequest",
    "CommentaryResponse",
    "LLMGateway",
    "get_gateway",
    # audit_log
    "ANONYMIZATION_VERSION",
    "AuditEntry",
    "AuditLogger",
    "list_audit_months",
    "read_audit_log",
    # role_prompts
    "Role",
    "RolePromptSpec",
    "ROLE_LABELS",
    "ROLE_PROMPTS",
    "SECTOR_BRIEFS",
    "get_role_spec",
    "get_sector_brief",
    "validate_catalog",
    # context_builders
    "build_anomaly_context",
    "build_deep_space_context",
    "build_integrity_context",
    "build_overview_context",
    "build_prediction_context",
    "build_productivity_context",
    "build_report_section_context",
    "build_safety_context",
    "build_zone_time_context",
]
