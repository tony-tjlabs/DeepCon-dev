"""
DeepCon LLM — 데이터 해석 보조 모듈 (DEPRECATED shim — M2-A, 2026-04-18)
=========================================================================
★ v4 (Session 6): 보안 강화 — 로직 추상화 + 수치 상대화
★ M2-A (Session 39): DeprecationWarning + FIXME — M2-B 에서 LLMGateway 로 완전 이관 예정.

현재 상태:
  - 이 모듈은 기존 Deep Space 탭 + Daily/Weekly 탭 호환을 위해 유지되고 있다.
  - 모든 `_call_claude()` 지점은 M2-B 에서 `core.ai.LLMGateway` 로 교체된다.
  - 새 코드는 이 모듈 대신 `from core.ai import get_gateway, CommentaryRequest` 사용.

마이그레이션 매핑(참고 — M2-B 에서 완수):
  cached_daily_summary       → role="overview_commentator" 또는 "productivity_analyst"
  cached_anomaly_flag        → role="anomaly_reporter"
  cached_weekly_trend_analysis → role="trend_analyst" (신설 예정)
  cached_insight_narrative   → role="productivity_analyst"
  cached_worker_analysis     → role="productivity_analyst" (worker-level)
  cached_prediction_insight  → role="prediction_explainer"
  cached_anomaly_insight     → role="anomaly_reporter"
  cached_spatial_insight     → role="deep_space_agent"
  cached_locus_enrichment    → role="deep_space_agent"

백엔드 설정 (config.py / 환경변수):
  LLM_BACKEND = "anthropic" (기본) | "bedrock"

보안 설정 (config.py / 환경변수):
  ANONYMIZE_LLM = true (기본) — 개인/업체/구역명 마스킹
  ANONYMIZE_LOGIC = true (기본) — 알고리즘 공식 추상화
"""
from __future__ import annotations

import logging
import os
import textwrap
import warnings
from typing import Sequence

import streamlit as st

from core.cache.policy import LLM_CACHE, LLM_INSIGHT

# ─── M2-A: shim 경고 ─────────────────────────────────────────────
warnings.warn(
    "src.dashboard.llm_deepcon 는 M2-B 에서 core.ai.LLMGateway 로 대체됩니다. "
    "새 코드는 `from core.ai import get_gateway, CommentaryRequest` 를 사용하세요.",
    DeprecationWarning,
    stacklevel=2,
)

from core.security.data_guard import DataGuard
from domain_packs.construction.prompts import (
    SYSTEM_CONTEXT as DOMAIN_SYSTEM_CONTEXT,
    SECURE_SYSTEM_CONTEXT,
    SPACE_TYPE_DESCRIPTIONS,
    OUTPUT_FORMAT_INSTRUCTION,
    build_daily_prompt as domain_build_daily_prompt,
    build_weekly_prompt as domain_build_weekly_prompt,
    build_insight_prompt as domain_build_insight_prompt,
    build_anomaly_prompt as domain_build_anomaly_prompt,
    build_worker_prompt as domain_build_worker_prompt,
    build_locus_context_prompt,
    # Deep Space 전용 프롬프트
    build_prediction_context_prompt,
    build_anomaly_context_prompt,
    build_spatial_insight_prompt,
    build_locus_enrichment_prompt,
)

logger = logging.getLogger(__name__)

# ─── DataGuard 싱글턴 ────────────────────────────────────────────
_data_guard = DataGuard(audit_enabled=True)


# ─── 설정 로드 ──────────────────────────────────────────────────────

def _load_config():
    """config.py에서 LLM 관련 설정 로드."""
    try:
        import sys
        from pathlib import Path
        _deepcon_root = str(Path(__file__).resolve().parent.parent.parent)
        if _deepcon_root not in sys.path:
            sys.path.insert(0, _deepcon_root)
        import config as cfg
        return {
            "backend": getattr(cfg, "LLM_BACKEND", "anthropic"),
            "aws_region": getattr(cfg, "AWS_REGION", "ap-northeast-2"),
            "aws_model_id": getattr(cfg, "AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1"),
            "anonymize": getattr(cfg, "ANONYMIZE_LLM", True),
            "anonymize_logic": getattr(cfg, "ANONYMIZE_LOGIC", True),
        }
    except Exception:
        return {
            "backend": os.getenv("LLM_BACKEND", "anthropic"),
            "aws_region": os.getenv("AWS_REGION", "ap-northeast-2"),
            "aws_model_id": os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1"),
            "anonymize": os.getenv("ANONYMIZE_LLM", "true").lower() == "true",
            "anonymize_logic": os.getenv("ANONYMIZE_LOGIC", "true").lower() == "true",
        }


# ─── Anthropic 클라이언트 ───────────────────────────────────────────

def _get_anthropic_client():
    """Anthropic 클라이언트 반환. API 키 없으면 None. (캐시 없음 — 키 변경 즉시 반영)"""
    from pathlib import Path
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_path, override=True)
    except ImportError:
        if _env_path.exists():
            with open(_env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


# ─── Bedrock 클라이언트 ─────────────────────────────────────────────

@st.cache_resource
def _get_bedrock_client(region: str):
    """boto3 Bedrock Runtime 클라이언트 반환. boto3 없으면 None."""
    try:
        import boto3
        return boto3.client("bedrock-runtime", region_name=region)
    except ImportError:
        logger.warning("boto3가 설치되지 않았습니다. `pip install boto3` 실행 필요.")
        return None
    except Exception as e:
        logger.warning("Bedrock 클라이언트 생성 실패: %s", e)
        return None


# ─── 통합 호출 ──────────────────────────────────────────────────────

def is_llm_available() -> bool:
    """현재 설정된 백엔드로 LLM 사용 가능 여부."""
    cfg = _load_config()
    if cfg["backend"] == "bedrock":
        return _get_bedrock_client(cfg["aws_region"]) is not None
    return _get_anthropic_client() is not None


def _call_claude(prompt: str, max_tokens: int = 1024) -> str:
    """
    Claude API 호출 (Anthropic 또는 Bedrock).
    실패 시 빈 문자열 반환.

    FIXME(M2-B): 이 함수는 M2-B 에서 core.ai.LLMGateway.analyze() 로 마이그레이션됩니다.
    새 코드는 `from core.ai import get_gateway, CommentaryRequest` 를 사용하세요.
    모든 내부 호출 지점(11군데)도 함께 이관 예정 — 본 파일 내부 FIXME 주석 참조.
    """
    cfg = _load_config()

    if cfg["backend"] == "bedrock":
        return _call_bedrock(prompt, max_tokens, cfg)
    return _call_anthropic(prompt, max_tokens)


# ─── 보안 시스템 프롬프트 (prompts.py에서 통합 import) ──────────────────

SECURE_SYSTEM_PROMPT = SECURE_SYSTEM_CONTEXT


def _call_anthropic(prompt: str, max_tokens: int) -> str:
    """Anthropic 직접 API 호출 (보안 시스템 프롬프트 적용)."""
    client = _get_anthropic_client()
    if client is None:
        return ""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=SECURE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Anthropic API 호출 실패: %s", e)
        return ""


def _call_bedrock(prompt: str, max_tokens: int, cfg: dict) -> str:
    """AWS Bedrock 호출 (보안 시스템 프롬프트 적용)."""
    client = _get_bedrock_client(cfg["aws_region"])
    if client is None:
        return ""
    try:
        import json
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": SECURE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(
            modelId=cfg["aws_model_id"],
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("Bedrock API 호출 실패: %s", e)
        return ""


# ─── Locus 컨텍스트 연동 ────────────────────────────────────────────


def _get_locus_context_for_llm(
    sector_id: str,
    locus_ids: list[str],
    max_loci: int = 10,
) -> str:
    """
    LLM용 Locus 컨텍스트를 생성한다.

    DataGuard.sanitize_locus_context()와 build_locus_context_prompt()를
    조합하여 안전한 공간 컨텍스트를 생성한다.

    Args:
        sector_id: Sector 식별자
        locus_ids: 포함할 Locus ID 목록
        max_loci: 최대 Locus 수

    Returns:
        LLM용 안전한 공간 컨텍스트 문자열
    """
    if not locus_ids:
        return ""

    try:
        from core.registry.locus_registry import get_registry
        registry = get_registry(sector_id)

        # DataGuard를 통해 안전한 컨텍스트 생성
        safe_context = _data_guard.sanitize_locus_context(registry, locus_ids[:max_loci])
        return safe_context
    except Exception as e:
        logger.warning("Locus 컨텍스트 생성 실패: %s", e)
        return ""


# ─── 3단 구조 파싱 유틸리티 ──────────────────────────────────────────


import re


def parse_structured_insight(text: str) -> dict[str, str]:
    """
    [WHAT]...[WHY]...[NOTE] 태그를 파싱한다.

    태그 없는 응답(기존 비구조화)은 fallback으로 what에 전체 텍스트 할당.

    Args:
        text: LLM 응답 텍스트

    Returns:
        {"what": str, "why": str, "note": str}
    """
    result = {"what": "", "why": "", "note": ""}

    for tag in ("WHAT", "WHY", "NOTE"):
        pattern = rf"\[{tag}\]\s*(.+?)(?=\[(?:WHAT|WHY|NOTE)\]|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            result[tag.lower()] = match.group(1).strip()

    # fallback: 태그 없으면 전체를 what으로
    if not any(result.values()):
        result["what"] = text.strip()

    return result


# ─── 익명화 통합 ────────────────────────────────────────────────────

def _get_anonymizer():
    """세션 스코프 Anonymizer 반환 (로직 추상화 포함)."""
    cfg = _load_config()
    cache_key = f"anonymizer_{cfg['anonymize_logic']}"

    if cache_key not in st.session_state:
        from src.pipeline.anonymizer import Anonymizer
        st.session_state[cache_key] = Anonymizer(anonymize_logic=cfg["anonymize_logic"])
    return st.session_state[cache_key]


def _call_claude_anon(
    prompt: str,
    max_tokens: int = 1024,
    *,
    worker_names: Sequence[str] | None = None,
    company_names: Sequence[str] | None = None,
    zone_names: Sequence[str] | None = None,
) -> str:
    """
    익명화 래퍼가 적용된 Claude 호출.

    1) prompt 내 이름 마스킹 → 2) LLM 호출 → 3) 응답 복원
    ANONYMIZE_LLM=false 이면 그대로 호출.
    """
    cfg = _load_config()

    if cfg["anonymize"] and (worker_names or company_names or zone_names):
        anon = _get_anonymizer()
        masked_prompt = anon.mask(
            prompt,
            worker_names=worker_names,
            company_names=company_names,
            zone_names=zone_names,
        )
        logger.debug(
            "Anonymizer: %d names masked in prompt (%d chars)",
            len(anon.get_mapping()),
            len(masked_prompt),
        )
        # FIXME(M2-B): _call_claude_anon 래퍼 전체를 LLMGateway.analyze() 로 대체.
        # LLMGateway 는 AnonymizationPipeline 을 내부에서 강제로 통과시키므로 이 래핑이 불필요해짐.
        response = _call_claude(masked_prompt, max_tokens)
        if response and not response.startswith("["):
            return anon.unmask(response)
        return response

    # FIXME(M2-B): 동일 — LLMGateway.analyze() 로 대체.
    return _call_claude(prompt, max_tokens)


# ─── 1. 일별 데이터 요약 (해석 보조) ─────────────────────────────────

@st.cache_data(ttl=LLM_CACHE, show_spinner=False)
def cached_daily_summary(
    date_str: str,
    sector_id: str,
    total_access: int,
    total_tward: int,
    tward_rate: float,
    avg_ewi: float,
    avg_cre: float,
    high_cre_count: int,
    confined_workers: int,
    hv_workers: int,
    top_spaces: str,
    companies: int,
) -> str:
    """
    일별 데이터 요약 생성.

    domain_packs/construction/prompts.py 템플릿을 사용하고,
    DataGuard를 통해 민감 정보를 필터링한다.
    """
    # DataGuard를 통해 데이터 필터링
    raw_data = {
        "date": date_str,
        "total_workers": total_access,
        "tward_count": total_tward,
        "tward_rate": tward_rate,
        "avg_ewi": avg_ewi,
        "avg_cre": avg_cre,
        "high_cre_count": high_cre_count,
        "confined_workers": confined_workers,
        "hv_workers": hv_workers,
        "companies": companies,
    }
    safe_data = _data_guard.sanitize_for_llm(raw_data)

    # 공간별 현황 문자열 조합
    space_lines = []
    if safe_data.get("confined_workers", 0) > 0:
        space_lines.append(f"- 밀폐공간 진입: {safe_data['confined_workers']}명")
    if safe_data.get("hv_workers", 0) > 0:
        space_lines.append(f"- 고압전 구역 진입: {safe_data['hv_workers']}명")
    space_summary = "\n".join(space_lines) if space_lines else ""

    # prompts.py 템플릿 사용
    prompt = domain_build_daily_prompt(
        date_str=safe_data.get("date", date_str),
        total_workers=safe_data.get("total_workers", 0),
        avg_ewi=safe_data.get("avg_ewi", 0.0),
        avg_cre=safe_data.get("avg_cre", 0.0),
        high_risk_count=safe_data.get("high_cre_count", 0),
        space_summary=space_summary,
    )

    # 추가 규칙 (기존 보안 제약 유지)
    prompt += "\n\n규칙:\n- 2~3문장으로 수치를 읽어주세요\n- 지표 계산 공식은 언급하지 마세요\n- 원인 추정이나 구체적 조치 권고는 하지 마세요\n- 한국어, 간결체"

    # FIXME(M2-B): cached_daily_summary → role="overview_commentator" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=600)


# ─── 2. 이상 수치 플래그 ──────────────────────────────────────────────

@st.cache_data(ttl=LLM_CACHE, show_spinner=False)
def cached_anomaly_flag(
    date_str: str,
    sector_id: str,
    anomaly_summary: str,
) -> str:
    """이상 탐지 결과를 한 문단으로 요약. DataGuard 적용."""
    # DataGuard를 통해 이상 요약 필터링
    safe_data = _data_guard.sanitize_for_llm({"anomaly_summary": anomaly_summary})
    sanitized_summary = safe_data.get("anomaly_summary", anomaly_summary)

    from domain_packs.construction.prompts import build_anomaly_prompt
    prompt = build_anomaly_prompt(sanitized_summary)

    # 추가 규칙 (기존 보안 제약 유지)
    prompt += "\n\n규칙:\n- 2~3문장으로 간결하게 요약\n- 원인 분석이나 조치 권고는 하지 마세요\n- 한국어, 간결체"

    # FIXME(M2-B): cached_anomaly_flag → role="anomaly_reporter" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=500)


# ─── 3. 주간 트렌드 요약 ─────────────────────────────────────────────

@st.cache_data(ttl=LLM_CACHE, show_spinner=False)
def cached_weekly_trend_analysis(
    sector_id: str,
    date_range: str,
    trend_summary: str,
) -> str:
    # DataGuard를 통해 트렌드 요약 필터링
    safe_summary = _data_guard.sanitize_for_llm({"trend_summary": trend_summary})
    sanitized_trend = safe_summary.get("trend_summary", trend_summary)

    # prompts.py 템플릿 사용
    prompt = domain_build_weekly_prompt(
        date_range=date_range,
        trend_summary=sanitized_trend,
        weekly_insights_text="(주간 인사이트 데이터 없음)",
    )

    # 추가 규칙 (기존 보안 제약 유지)
    prompt += "\n\n추가 규칙:\n- 원인 추정이나 구체적 권고는 하지 마세요\n- 한국어, 간결체"

    # FIXME(M2-B): cached_weekly_trend_analysis → role="trend_analyst"(신설) 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=800)


# ─── 4. 리포트용 데이터 해석 ─────────────────────────────────────────

@st.cache_data(ttl=LLM_CACHE, show_spinner=False)
def cached_insight_narrative(
    date_str: str,
    sector_id: str,
    insights_summary: str,
    kpi_summary: str,
) -> str:
    """
    리포트용 데이터 해석.

    domain_packs/construction/prompts.py 템플릿을 사용하고,
    DataGuard를 통해 민감 정보를 필터링한다.
    """
    # DataGuard를 통해 데이터 필터링
    safe_kpi = _data_guard.sanitize_for_llm({"kpi_summary": kpi_summary})
    safe_insights = _data_guard.sanitize_for_llm({"insights_summary": insights_summary})

    # prompts.py 템플릿 사용
    prompt = domain_build_insight_prompt(
        date_str=date_str,
        kpi_summary=safe_kpi.get("kpi_summary", kpi_summary),
        insights_text=safe_insights.get("insights_summary", insights_summary),
        include_space_context=True,
    )

    # 추가 규칙 (기존 보안 제약 유지: 권고 금지)
    prompt += "\n\n추가 규칙:\n- 지표 계산 공식이나 알고리즘을 언급하지 마세요\n- 깊은 원인 분석이나 조치 권고는 하지 마세요\n- 한국어, 보고서 문체"

    # FIXME(M2-B): cached_insight_narrative → role="productivity_analyst" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=800)


# ─── 5. 작업자 분석 (build_worker_prompt 연동) ────────────────────────

@st.cache_data(ttl=LLM_CACHE, show_spinner=False)
def cached_worker_analysis(
    sector_id: str,
    worker_id: str,
    ewi: float,
    cre: float,
    work_minutes: int,
    top_locus_ids: list[str] | None = None,
) -> str:
    """
    개별 작업자 분석 생성.

    domain_packs/construction/prompts.py의 build_worker_prompt를 사용하고,
    DataGuard를 통해 민감 정보를 필터링한다.

    Args:
        sector_id: Sector 식별자
        worker_id: 작업자 식별자 (익명화됨, 예: "작업자_A")
        ewi: 작업강도 지표
        cre: 위험노출 지표
        work_minutes: 근무 시간 (분)
        top_locus_ids: 주요 방문 공간 ID 목록

    Returns:
        LLM 분석 결과 문자열
    """
    # 작업자 지표 문자열 구성 (이미 익명화된 ID 사용)
    metrics_parts = [
        f"- 작업자: {worker_id}",
        f"- 작업강도(EWI): {ewi:.3f}",
        f"- 위험노출(CRE): {cre:.3f}",
        f"- 근무시간: {work_minutes}분 ({work_minutes // 60}시간 {work_minutes % 60}분)",
    ]

    # Locus 컨텍스트 추가 (있는 경우)
    if top_locus_ids:
        locus_context = _get_locus_context_for_llm(sector_id, top_locus_ids, max_loci=5)
        if locus_context:
            metrics_parts.append(f"\n주요 방문 공간:\n{locus_context}")

    worker_metrics = "\n".join(metrics_parts)

    # prompts.py 템플릿 사용
    prompt = domain_build_worker_prompt(worker_metrics)

    # 추가 규칙 (기존 보안 제약 유지)
    prompt += "\n\n규칙:\n- 2문장으로 간결하게 요약\n- 개인 식별 정보는 언급하지 마세요\n- 원인 분석이나 조치 권고는 하지 마세요\n- 한국어, 간결체"

    # FIXME(M2-B): cached_worker_analysis → role="productivity_analyst" (worker-level) 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=400)


# ─── (삭제됨) cached_high_risk_analysis ──────────────────────────────
# v2에서 제거: 고위험 작업자 AI 분석 → 데이터 테이블로 대체
# 공간 데이터/프롬프트 미성숙으로 LLM 패턴 분석 부정확 위험


# ─── UI 렌더링 ────────────────────────────────────────────────────────

# WHAT/WHY/NOTE 칩 스타일
INSIGHT_LAYER_CHIPS = {
    "WHAT": {
        "bg": "rgba(0, 174, 239, 0.12)",
        "color": "#00AEEF",
    },
    "WHY": {
        "bg": "rgba(154, 181, 212, 0.10)",
        "color": "#9AB5D4",
    },
    "NOTE": {
        "bg": "rgba(0, 200, 151, 0.10)",
        "color": "#00C897",
    },
}


def _render_insight_layer(label: str, content: str) -> str:
    """개별 인사이트 레이어 HTML 생성."""
    if not content:
        return ""

    chip_style = INSIGHT_LAYER_CHIPS.get(label.upper(), INSIGHT_LAYER_CHIPS["WHAT"])
    chip_html = (
        f"<span style='display:inline-block; padding:2px 7px; border-radius:4px; "
        f"font-size:0.7rem; font-weight:700; background:{chip_style['bg']}; "
        f"color:{chip_style['color']}; margin-right:8px;'>{label.upper()}</span>"
    )

    # 텍스트 색상
    text_color = "#D5E5FF" if label.upper() == "WHAT" else "#9AB5D4"
    font_weight = "600" if label.upper() == "WHAT" else "400"

    return (
        f"<div style='display:flex; align-items:flex-start; margin-bottom:8px;'>"
        f"{chip_html}"
        f"<span style='color:{text_color}; font-size:0.84rem; font-weight:{font_weight}; "
        f"line-height:1.5;'>{content}</span></div>"
    )


def render_data_comment(title: str, content: str, show_timestamp: bool = False):
    """
    데이터 해석 코멘트 렌더링 — 3단 구조화 버전.

    변경점:
    - parse_structured_insight()로 WHAT/WHY/NOTE 분리
    - 각 레이어별 칩 + 색상 구분
    - show_timestamp: 생성 시각 표시 여부 (기본: False)
    """
    if not content:
        return

    is_error = content.startswith("[API 오류") or content.startswith("[Bedrock API 오류")
    if is_error:
        return  # 에러는 조용히 무시

    # 3단 구조 파싱
    parsed = parse_structured_insight(content)

    # HTML 구성
    layers_html = ""
    for layer in ["what", "why", "note"]:
        if parsed.get(layer):
            layers_html += _render_insight_layer(layer, parsed[layer])

    # 타임스탬프 (선택적)
    timestamp_html = ""
    if show_timestamp:
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        timestamp_html = (
            f"<div style='text-align:right; color:#5A6A7A; font-size:0.72rem; "
            f"margin-top:8px;'>생성: {now_str}</div>"
        )

    st.markdown(
        f"<div style='background:#0D1520; border-left:3px solid #1E4A6A; "
        f"border-radius:6px; padding:12px 16px; margin:8px 0;'>"
        f"<div style='color:#5A8FA6; font-size:0.75rem; margin-bottom:10px;'>"
        f"{title}</div>"
        f"{layers_html}"
        f"{timestamp_html}</div>",
        unsafe_allow_html=True,
    )


# ─── 하위 호환 (기존 코드에서 호출하는 경우) ──────────────────────────

def render_ai_card(title: str, content: str, is_loading: bool = False):
    """하위 호환 래퍼 → render_data_comment로 위임."""
    if is_loading:
        return
    render_data_comment(title, content)


# ─── Deep Space 전용 LLM 함수 ─────────────────────────────────────────────


@st.cache_data(ttl=LLM_INSIGHT, show_spinner=False)
def cached_prediction_insight(
    current_locus: str,
    predictions: str,
    locus_context: str,
) -> str:
    """이동 예측 결과에 건설현장 맥락 부여.

    Args:
        current_locus: 현재 위치 이름
        predictions: 예측 이동 요약 (Top-K)
        locus_context: 관련 공간 맥락 (DataGuard 경유)

    Returns:
        LLM 해석 결과 (3단 구조: WHAT/WHY/NOTE)
    """
    if not is_llm_available():
        return ""

    # DataGuard로 민감 정보 필터링
    safe_context = locus_context
    if _data_guard and locus_context:
        safe_data = _data_guard.sanitize_for_llm({"locus_context": locus_context})
        safe_context = safe_data.get("locus_context", locus_context)

    prompt = build_prediction_context_prompt(
        current_locus=current_locus,
        predictions=predictions,
        locus_context=safe_context,
    )

    # FIXME(M2-B): cached_prediction_insight → role="prediction_explainer" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=400)


@st.cache_data(ttl=LLM_INSIGHT, show_spinner=False)
def cached_anomaly_insight(
    anomaly_description: str,
    perplexity: str,
    locus_context: str,
) -> str:
    """이상 이동 결과에 건설현장 맥락 부여.

    Args:
        anomaly_description: 이상 이동 설명 (from -> to)
        perplexity: Perplexity 점수
        locus_context: 관련 공간 맥락 (DataGuard 경유)

    Returns:
        LLM 해석 결과 (3단 구조: WHAT/WHY/NOTE)
    """
    if not is_llm_available():
        return ""

    safe_context = locus_context
    if _data_guard and locus_context:
        safe_data = _data_guard.sanitize_for_llm({"locus_context": locus_context})
        safe_context = safe_data.get("locus_context", locus_context)

    prompt = build_anomaly_context_prompt(
        anomaly_description=anomaly_description,
        perplexity=perplexity,
        locus_context=safe_context,
    )

    # FIXME(M2-B): cached_anomaly_insight → role="anomaly_reporter" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=400)


@st.cache_data(ttl=LLM_INSIGHT, show_spinner=False)
def cached_spatial_insight(
    summary: str,
    congested_spaces: str,
    locus_context: str,
) -> str:
    """현장 시뮬레이션 결과에 건설현장 맥락 부여.

    Args:
        summary: 현장 요약 (총 인원, 작업/이동/휴식 등)
        congested_spaces: 혼잡 공간 목록
        locus_context: 관련 공간 맥락 (DataGuard 경유)

    Returns:
        LLM 해석 결과 (3단 구조: WHAT/WHY/NOTE)
    """
    if not is_llm_available():
        return ""

    safe_context = locus_context
    if _data_guard and locus_context:
        safe_data = _data_guard.sanitize_for_llm({"locus_context": locus_context})
        safe_context = safe_data.get("locus_context", locus_context)

    prompt = build_spatial_insight_prompt(
        summary=summary,
        congested_spaces=congested_spaces,
        locus_context=safe_context,
    )

    # FIXME(M2-B): cached_spatial_insight → role="deep_space_agent" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=400)


@st.cache_data(ttl=LLM_INSIGHT, show_spinner=False)
def cached_locus_enrichment(
    locus_info: str,
    movement_pattern: str,
    time_context: str,
) -> str:
    """Locus 맥락 해석.

    Args:
        locus_info: 공간 정보
        movement_pattern: 이동 패턴 설명
        time_context: 시간대 정보

    Returns:
        LLM 해석 결과 (3단 구조: WHAT/WHY/NOTE)
    """
    if not is_llm_available():
        return ""

    prompt = build_locus_enrichment_prompt(
        locus_info=locus_info,
        movement_pattern=movement_pattern,
        time_context=time_context,
    )

    # FIXME(M2-B): cached_locus_enrichment → role="deep_space_agent" 로 LLMGateway.analyze() 마이그레이션.
    return _call_claude(prompt, max_tokens=300)
