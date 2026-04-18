"""
건설현장 LLM 프롬프트 템플릿
============================
핵심 로직 미포함, 맥락과 템플릿만 제공.
DataGuard를 적용한 안전한 프롬프트 — 민감 정보 없음.

출처:
  - src/intelligence/site_context.py (Y1_SITE_CONTEXT, build_analysis_prompt)
  - src/dashboard/llm_deepcon.py (LLM 호출 패턴)

사용법:
    from domain_packs.construction.prompts import (
        SYSTEM_CONTEXT,
        build_daily_prompt,
        build_weekly_prompt,
    )
"""
from __future__ import annotations

# ─── 시스템 컨텍스트 ─────────────────────────────────────────────────────────
# LLM이 분석을 수행할 때 참고하는 현장 배경 정보

SYSTEM_CONTEXT = """
당신은 건설현장 공간 데이터 분석 보조입니다.
반도체 클러스터 건설현장의 작업자 이동 데이터를 기반으로 분석합니다.

역할:
- 수치를 요약하고 패턴을 설명
- 데이터 기반 관찰 제공
- 구체적인 권고나 조치 지시는 하지 않음

건설현장 특성:
- BLE 태그 기반 1분 단위 위치 기록
- EWI (유효작업집중도): 고활성x1.0 + 저활성x0.5 + 대기x0.2 / 근무시간
- CRE (위험노출도): 공간위험(40%) + 동적밀집(15%) + 개인위험(45%)
- 밀폐공간/고압전 구역: 단독작업 금지 (2인 1조 원칙)
""".strip()


# ─── 공간 유형 설명 ─────────────────────────────────────────────────────────

SPACE_TYPE_DESCRIPTIONS = """
공간 유형별 특성:
- TRANSIT (통과형): 게이트, 타각기, 호이스트 — 체류 시간 짧음 (평균 <5분)
- SHORT_STAY (단기체류): 휴게실, 흡연실 — 체류 시간 10~30분
- LONG_STAY (작업구역): FAB, CUB, 야외 — 체류 시간 30분+
- HAZARD_ZONE (고위험): 밀폐공간, 고압전 — 장시간 체류 시 경고 필요
- ADMIN (관리): 사무실 — 행정 업무
""".strip()


# ─── 일별 분석 프롬프트 ─────────────────────────────────────────────────────

DAILY_SUMMARY_TEMPLATE = """
## 분석 날짜: {date}

### 주요 지표
- 총 출입 인원: {total_workers}명
- 평균 작업강도(EWI): {avg_ewi:.3f}
- 평균 위험노출(CRE): {avg_cre:.3f}
- 고위험 작업자: {high_risk_count}명

### 공간별 현황
{space_summary}

위 수치를 기반으로 오늘의 현장 상황을 2-3문장으로 요약해주세요.
수치를 나열하지 말고, 데이터가 의미하는 바를 해석해주세요.
""".strip()


def build_daily_prompt(
    date_str: str,
    total_workers: int,
    avg_ewi: float,
    avg_cre: float,
    high_risk_count: int,
    space_summary: str = "",
) -> str:
    """
    일별 분석 LLM 프롬프트 생성.

    Args:
        date_str: 분석 날짜 (YYYY-MM-DD)
        total_workers: 총 출입 인원
        avg_ewi: 평균 EWI
        avg_cre: 평균 CRE
        high_risk_count: 고위험(CRE >= 0.6) 작업자 수
        space_summary: 공간별 현황 문자열

    Returns:
        LLM 프롬프트 문자열
    """
    return DAILY_SUMMARY_TEMPLATE.format(
        date=date_str,
        total_workers=total_workers,
        avg_ewi=avg_ewi,
        avg_cre=avg_cre,
        high_risk_count=high_risk_count,
        space_summary=space_summary or "(공간별 현황 없음)",
    )


# ─── 인사이트 기반 분석 프롬프트 ────────────────────────────────────────────

INSIGHT_ANALYSIS_TEMPLATE = """
{system_context}

## 오늘의 현장 데이터 ({date})
{kpi_summary}

## AI 인사이트 엔진이 감지한 핵심 발견
{insights_text}

위의 현장 컨텍스트와 인사이트를 종합하여, 현장 소장에게 보고할 핵심 브리핑을 작성하세요.

작성 원칙:
1. 인사이트를 단순 나열하지 말고, 건설현장 맥락에서 왜 중요한지 해석하세요
2. 즉각 조치가 필요한 사항은 명확히 구분하세요
3. 수치보다 의미에 집중하세요 (예: "CRE 0.65" -> "고위험 임계점 초과")
4. 구체적인 권고사항은 포함하지 마세요 (데이터 해석만)
5. 3-4문단, 한국어로 작성하세요
""".strip()


def build_insight_prompt(
    date_str: str,
    kpi_summary: str,
    insights_text: str,
    include_space_context: bool = True,
) -> str:
    """
    인사이트 기반 LLM 분석 프롬프트 생성.

    Args:
        date_str: 분석 날짜
        kpi_summary: KPI 요약 문자열
        insights_text: 인사이트 요약 문자열
        include_space_context: 공간 유형 설명 포함 여부

    Returns:
        LLM 프롬프트 문자열
    """
    context = SYSTEM_CONTEXT
    if include_space_context:
        context = f"{SYSTEM_CONTEXT}\n\n{SPACE_TYPE_DESCRIPTIONS}"

    return INSIGHT_ANALYSIS_TEMPLATE.format(
        system_context=context,
        date=date_str,
        kpi_summary=kpi_summary,
        insights_text=insights_text,
    )


# ─── 주간 분석 프롬프트 ─────────────────────────────────────────────────────

WEEKLY_ANALYSIS_TEMPLATE = """
{system_context}

## 주간 데이터 ({date_range})
{trend_summary}

## AI 인사이트 엔진이 감지한 주간 핵심 발견
{weekly_insights_text}

위의 주간 데이터를 종합하여 다음을 분석하세요:
1. 이번 주 가장 주목할 변화나 이상 패턴
2. 지난 주 대비 개선/악화된 영역
3. 패턴의 의미 해석 (데이터를 나열하지 마세요)

3-4문단, 한국어로 작성하세요.
구체적인 조치 권고는 포함하지 마세요.
""".strip()


def build_weekly_prompt(
    date_range: str,
    trend_summary: str,
    weekly_insights_text: str,
) -> str:
    """
    주간 트렌드 분석 LLM 프롬프트 생성.

    Args:
        date_range: 분석 기간 (예: "2026-03-18 ~ 2026-03-24")
        trend_summary: 주간 트렌드 요약 문자열
        weekly_insights_text: 주간 인사이트 문자열

    Returns:
        LLM 프롬프트 문자열
    """
    return WEEKLY_ANALYSIS_TEMPLATE.format(
        system_context=SYSTEM_CONTEXT,
        date_range=date_range,
        trend_summary=trend_summary,
        weekly_insights_text=weekly_insights_text,
    )


# ─── 이상 탐지 알림 프롬프트 ────────────────────────────────────────────────

ANOMALY_ALERT_TEMPLATE = """
## 이상 탐지 알림

감지된 이상:
{anomaly_description}

이 이상이 건설현장에서 갖는 의미를 1-2문장으로 설명해주세요.
수치나 기술적 세부사항보다 현장 관점에서의 의미에 집중하세요.
""".strip()


def build_anomaly_prompt(anomaly_description: str) -> str:
    """
    이상 탐지 알림 프롬프트 생성.

    Args:
        anomaly_description: 이상 상황 설명

    Returns:
        LLM 프롬프트 문자열
    """
    return ANOMALY_ALERT_TEMPLATE.format(
        anomaly_description=anomaly_description,
    )


# ─── 작업자 분석 프롬프트 ───────────────────────────────────────────────────

WORKER_ANALYSIS_TEMPLATE = """
## 작업자 분석

작업자 지표:
{worker_metrics}

이 작업자의 오늘 근무 패턴을 2문장으로 요약해주세요.
개인 식별 정보는 언급하지 마세요.
""".strip()


def build_worker_prompt(worker_metrics: str) -> str:
    """
    작업자 분석 프롬프트 생성.

    Args:
        worker_metrics: 작업자 지표 문자열 (익명화됨)

    Returns:
        LLM 프롬프트 문자열
    """
    return WORKER_ANALYSIS_TEMPLATE.format(
        worker_metrics=worker_metrics,
    )


# ─── Locus 컨텍스트 템플릿 ─────────────────────────────────────────────

LOCUS_CONTEXT_TEMPLATE = """
공간 정보:
{locus_descriptions}
""".strip()


def build_locus_context_prompt(
    locus_descriptions: list[str],
    max_loci: int = 10,
) -> str:
    """
    Locus 컨텍스트 프롬프트를 생성한다.

    LocusRegistry.to_natural_language() 결과를 입력받아
    LLM 프롬프트에 삽입할 컨텍스트를 생성한다.

    Args:
        locus_descriptions: 자연어 설명 리스트
        max_loci: 최대 포함 Locus 수 (토큰 제한)

    Returns:
        LLM 프롬프트용 공간 컨텍스트 문자열
    """
    truncated = locus_descriptions[:max_loci]
    if len(locus_descriptions) > max_loci:
        truncated.append(f"... 외 {len(locus_descriptions) - max_loci}개 구역")

    return LOCUS_CONTEXT_TEMPLATE.format(
        locus_descriptions="\n".join(f"- {desc}" for desc in truncated),
    )


# ─── 통합 시스템 컨텍스트 (보안 규칙 포함) ─────────────────────────────────

SECURE_SYSTEM_CONTEXT = f"""{SYSTEM_CONTEXT}

역할 제한:
- 수치 데이터를 간결하게 요약하고 읽어주는 역할만 수행합니다.
- 깊은 원인 분석, 구체적 조치 권고, 전문가적 판단은 하지 않습니다.
- "~해야 합니다", "~하는 것이 좋습니다" 같은 권고성 문장을 사용하지 않습니다.

출력 규칙:
- 한국어, 간결체 (2~4문장)
- 수치 변화를 "상승/하락/유지"로 표현
- 평소 대비 변화를 "평소 대비 N%" 형식으로 표현
- 알고리즘 공식이나 계산 방법을 언급하지 않습니다.

출력 형식 (3단 구조):
[WHAT] 수치/패턴 요약 1문장
[WHY] 주목할 이유 1문장
[NOTE] 현장 관점 해석 1문장
""".strip()


# ─── 출력 형식 지시 (기존 템플릿에 추가용) ──────────────────────────────────

OUTPUT_FORMAT_INSTRUCTION = """

출력 형식:
[WHAT] {수치/패턴 요약 1문장}
[WHY] {주목할 이유 1문장}
[NOTE] {현장 관점 해석 1문장}
""".strip()


# ─── Deep Space 전용 프롬프트 ─────────────────────────────────────────────

PREDICTION_CONTEXT_TEMPLATE = """
당신은 건설현장 공간 데이터 해석 전문가입니다.
Deep Space 모델이 작업자의 다음 이동 위치를 예측했습니다.

현재 위치: {current_locus}
예측 이동: {predictions}
공간 속성: {locus_context}

이 예측 결과의 건설현장 관점 의미를 2~3문장으로 해석하세요.
- 밀폐공간/고소작업/고위험 구역 관련 안전 맥락이 있다면 언급하세요.
- 수치를 나열하지 말고 의미를 해석하세요.
- "~해야 합니다" 같은 권고성 문장은 사용하지 마세요.

{output_format}
""".strip()


ANOMALY_CONTEXT_TEMPLATE = """
당신은 건설현장 이상 이동 탐지 해석 전문가입니다.
Deep Space 모델이 비정상적인 이동 패턴을 감지했습니다.

이상 이동: {anomaly_description}
Perplexity: {perplexity}
관련 공간 속성: {locus_context}

이 이상 이동의 건설현장 관점 의미를 2~3문장으로 해석하세요.
- 안전 위험이 있다면 언급하세요.
- 수치를 나열하지 말고 의미를 해석하세요.
- "~해야 합니다" 같은 권고성 문장은 사용하지 마세요.

{output_format}
""".strip()


SPATIAL_INSIGHT_TEMPLATE = """
당신은 건설현장 공간 상태 해석 전문가입니다.
현장 시뮬레이션에서 공간별 상태를 분석했습니다.

현장 요약: {summary}
혼잡 공간: {congested_spaces}
공간 속성: {locus_context}

현재 현장 상태의 건설현장 관점 의미를 2~3문장으로 해석하세요.
- 혼잡/병목/안전 위험 관련 패턴이 있다면 언급하세요.
- 수치를 나열하지 말고 의미를 해석하세요.
- "~해야 합니다" 같은 권고성 문장은 사용하지 마세요.

{output_format}
""".strip()


LOCUS_CONTEXT_ENRICHMENT_TEMPLATE = """
당신은 건설현장 공간 맥락 분석 전문가입니다.
특정 공간(Locus)에 대한 이동 데이터와 속성이 주어졌습니다.

공간 정보: {locus_info}
이동 패턴: {movement_pattern}
시간대: {time_context}

이 공간의 현재 상태를 건설현장 관점에서 1~2문장으로 요약하세요.
- 안전/운영 맥락만 간결하게.
- "~해야 합니다" 같은 권고성 문장은 사용하지 마세요.

{output_format}
""".strip()


def build_prediction_context_prompt(
    current_locus: str,
    predictions: str,
    locus_context: str,
) -> str:
    """이동 예측 해석 LLM 프롬프트 생성."""
    return PREDICTION_CONTEXT_TEMPLATE.format(
        current_locus=current_locus,
        predictions=predictions,
        locus_context=locus_context or "(공간 정보 없음)",
        output_format=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_anomaly_context_prompt(
    anomaly_description: str,
    perplexity: str,
    locus_context: str,
) -> str:
    """이상 이동 해석 LLM 프롬프트 생성."""
    return ANOMALY_CONTEXT_TEMPLATE.format(
        anomaly_description=anomaly_description,
        perplexity=perplexity,
        locus_context=locus_context or "(공간 정보 없음)",
        output_format=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_spatial_insight_prompt(
    summary: str,
    congested_spaces: str,
    locus_context: str,
) -> str:
    """현장 시뮬레이션 해석 LLM 프롬프트 생성."""
    return SPATIAL_INSIGHT_TEMPLATE.format(
        summary=summary,
        congested_spaces=congested_spaces,
        locus_context=locus_context or "(공간 정보 없음)",
        output_format=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_locus_enrichment_prompt(
    locus_info: str,
    movement_pattern: str,
    time_context: str,
) -> str:
    """Locus 맥락 해석 LLM 프롬프트 생성."""
    return LOCUS_CONTEXT_ENRICHMENT_TEMPLATE.format(
        locus_info=locus_info,
        movement_pattern=movement_pattern,
        time_context=time_context,
        output_format=OUTPUT_FORMAT_INSTRUCTION,
    )
