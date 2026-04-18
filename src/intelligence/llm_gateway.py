"""
LLM Gateway (Y1 전용) — LEGACY (M2-A, 2026-04-18 이후 deprecated)
================================================================
⚠️ FIXME(M2-B): 이 모듈은 `core.ai.LLMGateway` 로 단일화됩니다.
  신규 코드는 반드시 `from core.ai import get_gateway, CommentaryRequest` 를 사용하세요.
  기존 `LLMGateway.analyze(tab_id=..., packed_text=...)` 호출은 M2-B 에서
  `core.ai.LLMGateway.analyze(CommentaryRequest(role=..., context=..., ...))` 로 이관됩니다.

M15X LLMGateway 구조를 채택하되 Y1 전용 시스템 프롬프트로 재작성.

흐름:
  1. AnonymizationPipeline → K=20 익명화 (우회 불가)
  2. _build_prompt()       → Y1 BASE_SYSTEM_PROMPT + TAB_CONTEXT 조합
  3. _audit_log()          → 세션 내 전송 내용 감사
  4. _call_claude_with_system() → Anthropic 또는 Bedrock 호출

탭 ID:
  - "daily"       : EWI/CRE 일별 작업자 분석
  - "congestion"  : 공간 혼잡도 분석
  - "weekly"      : 주간 트렌드 분석
  - "deep_space"  : Agentic AI / Deep Space 예측 해석

llm_deepcon.py 관계:
  - llm_deepcon.py는 유지 (Deep Space 탭 하위 호환)
  - Wave 3에서 일별/혼잡도/주간 탭이 점진적으로 이 Gateway로 전환
  - 현재 llm_deepcon.generate_insight()가 이 Gateway를 thin wrapper로 호출 예정

사용:
    from src.intelligence.llm_gateway import LLMGateway

    result = LLMGateway.analyze(
        tab_id="daily",
        packed_text=packed_text,
        company_names=list(worker_df["company_name"].dropna().unique()),
        date_list=[date_str],
    )
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


# ─── Y1 탭별 컨텍스트 ────────────────────────────────────────────────────────

_TAB_CONTEXT: dict[str, str] = {
    "daily": """
분석 주제: 일별 작업자 현황 (EWI/CRE/SII)
- EWI(유효작업집중도): 고활성×1.0 + 저활성×0.5 + 대기×0.2 / 근무시간
- CRE(건설위험노출도): 공간위험(40%) + 동적밀집(15%) + 개인위험(45%)
- SII: EWI × 공간위험도 — 고강도 + 고위험 공간 동시 작업자 탐지
- 주간/야간 교대 분리 해석: 야간 작업자는 수면 리듬 차이로 EWI 패턴이 다름
- earlywork/daywork/extensionwork/nightwork/overnightwork 플래그 기반 교대 유형 구분
""".strip(),

    "congestion": """
분석 주제: 공간 혼잡도 (213개 locus, FAB/CUB/WWT/야외/게이트 구역)
- 피크 혼잡: 식사시간(11~13시), 교대 전환(06시/18시) 집중 현상은 정상
- FAB vs CUB: 공정 단계에 따라 집중 구역이 주기적으로 이동
- 154kV/야외 구역: 다수 작업자 통과 지점 — 고위험 교차점
- 밀폐공간(CUB 내) 과밀: 2인1조 원칙 준수 확인 필요
- GW-XXX(locus ID)는 공간 코드명으로, 구역 유형(FAB/CUB/WWT 등)으로 해석
""".strip(),

    "weekly": """
분석 주제: 주간 트렌드 (EWI/CRE/이상탐지)
- 요일 패턴: 월요일 EWI 낮음(주초 적응), 금요일 소폭 하락 일반적
- CRE 급등일: 고소/밀폐 특수 작업 또는 인력 집중 배치 가능성
- 주간 이상탐지 결과: 7종 이상 탐지기 중 발화된 항목 해석
- 주간 교대 비율 변화: 야간 비율 증가 시 공정 진행 상황 반영 가능
""".strip(),

    "deep_space": """
분석 주제: Deep Space Transformer 예측 결과 해석
- 예측값(predicted_): 40일 학습 기반 Top-1 60.1%, Top-3 82.5% 정확도
- 예측 신뢰도: 반드시 "(예측)" 표시 및 정확도 수준 병기
- Agentic AI 예측(혼잡/병목/안전/생산성): 수치 요약만, 원인 단정 금지
- Prediction Journal 정확도 추이: 상승/하락 방향성만 언급
- Group Top-1 89.5% vs Locus Top-1 60%: 구역 수준 예측이 더 신뢰 가능
""".strip(),
}

# ─── Y1 BASE SYSTEM PROMPT ───────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """
당신은 반도체 클러스터 건설현장 데이터 해석 보조입니다.
SK하이닉스 Y-Project (경기도 용인, FAB/CUB/WWT/154kV/야외 복합 구조)의
작업자 이동 데이터를 분석합니다.

[현장 기본 특성]
- 일일 ~9,000명, 208개 협력업체 동시 투입
- 교대 패턴: 주간(06:00~18:00), 야간(18:00~06:00), 연장야간 포함
  earlywork=1: 일근, daywork=1: 주간, extensionwork=1: 연장,
  nightwork=1: 야간, overnightwork=1: 야간연장
- BLE 음영: 213개 Gateway 중 건설 단계에 따라 미설치 구역 존재. 평균 50%+ 음영 정상
- T-Ward: 활성신호 기반. active_ratio ≥ 0.90 = 고활성 작업, < 0.40 = 대기

[해석 지침]
- 야간 EWI 낮음: "피로 누적"이 아니라 "교대 복귀 직후"일 수 있음 → 교대 비율 먼저 확인
- 주간/야간 동시 작업자 수 급증: 교대 전환 구간(06시/18시) 자연 현상
- 특정 locus(GW-XXX) 밀집: FAB 공정 단계 집중 배치가 원인일 수 있음
- CRE ≥ 0.5 비율 > 10%: 안전 관리 강화 필요 신호
- Gap_ratio 높은 구역: 인프라 미완성(BLE 미설치) 구역 — 데이터 부재 ≠ 작업 없음

[비교 컨텍스트 해석 — 데이터에 포함된 경우 반드시 활용]
데이터에 "## 비교 컨텍스트" 섹션이 포함된 경우:
- [A] 기준선: 오늘 수치가 최근 14일 평균 대비 어떤 위치인지 WHAT에서 명시
  예: "오늘 EWI(0.52)는 최근 14일 평균(0.48) 대비 +0.04로 다소 높은 수준입니다."
- [B] 요일 패턴: 같은 요일(최근 4주) 대비 특이점이 있으면 언급
- [C] 직전일 변화: 방향성(▲/▼)을 WHY 해석에 활용
- [D] 순위: "N일 중 X위" 정보를 상대적 위치 표현에 활용
컨텍스트가 없거나 N일 데이터가 적으면 해당 비교는 생략.

[역할 제한]
- 수치 요약 및 패턴 해석만 수행
- 원인 단정, 특정 조치 권고, "~해야 합니다" 사용 금지
- 알고리즘 수식/계산 방법 언급 금지
- 특정 작업자/업체 직접 언급 금지 (코드명만 허용)

[출력 형식 — 반드시 준수]
[WHAT] 데이터가 보여주는 주요 현상 2~3문장 (수치 직접 인용, 비교 컨텍스트 활용)
[WHY] 패턴의 현장 맥락 해석 1~2문장 (원인 단정 금지, "~로 보임" 형식)
[NOTE] 해석 한계 또는 추가 확인 필요 사항 1문장

한국어, 간결체 사용.
""".strip()


# ─── 프롬프트 빌더 ───────────────────────────────────────────────────────────

def _build_prompt(tab_id: str, data_text: str) -> str:
    """탭별 컨텍스트 + 데이터를 조합한 유저 프롬프트 생성."""
    tab_ctx = _TAB_CONTEXT.get(tab_id, "")
    return f"""
{tab_ctx}

---

{data_text}

---

위 데이터를 바탕으로 건설현장 현장 소장 관점의 인사이트를 제공하세요.
출력 형식([WHAT]/[WHY]/[NOTE])을 반드시 사용하세요.
""".strip()


# ─── 감사 로그 ───────────────────────────────────────────────────────────────

def _audit_log(tab_id: str, prompt_text: str) -> None:
    """세션 내 전송 내용 감사 로그."""
    try:
        key = "_llm_gateway_audit_log"
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append({
            "time":    datetime.now().isoformat(),
            "tab":     tab_id,
            "chars":   len(prompt_text),
            "preview": prompt_text[:200],
        })
    except Exception:
        pass


def get_audit_log() -> list[dict]:
    """세션 내 감사 로그 반환 (관리자 탭용)."""
    try:
        return st.session_state.get("_llm_gateway_audit_log", [])
    except Exception:
        return []


# ─── 내부 API 호출 ───────────────────────────────────────────────────────────

def _call_claude_with_system(prompt: str, system: str, max_tokens: int) -> str:
    """
    강화된 시스템 프롬프트로 Claude 호출.
    llm_deepcon.py의 _call_anthropic/_call_bedrock 패턴과 동일하되
    system 파라미터를 직접 주입한다.
    """
    import os
    from pathlib import Path

    # .env 로드
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)
    except ImportError:
        pass

    try:
        backend     = getattr(cfg, "LLM_BACKEND",          "anthropic")
        aws_region  = getattr(cfg, "AWS_REGION",            "ap-northeast-2")
        aws_model   = getattr(cfg, "AWS_BEDROCK_MODEL_ID",  "anthropic.claude-sonnet-4-6-v1")
    except Exception:
        backend    = os.getenv("LLM_BACKEND",         "anthropic")
        aws_region = os.getenv("AWS_REGION",           "ap-northeast-2")
        aws_model  = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1")

    if backend == "bedrock":
        return _bedrock_call(prompt, system, max_tokens, aws_region, aws_model)
    return _anthropic_call(prompt, system, max_tokens)


def _anthropic_call(prompt: str, system: str, max_tokens: int) -> str:
    # FIXME(M2-B): 이 함수는 core.ai.LLMGateway._call_sync() 로 대체됩니다.
    #   중복 anthropic.Anthropic 인스턴스를 제거하고 model 하드코딩도 환경변수로 이관.
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY 미설정")
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Anthropic 호출 실패: %s", e)
        return ""


def _bedrock_call(
    prompt: str, system: str, max_tokens: int, region: str, model_id: str
) -> str:
    import json
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("Bedrock 호출 실패: %s", e)
        return ""


# ─── 메인 Gateway ─────────────────────────────────────────────────────────────

class LLMGateway:
    """
    DeepCon Y1 전용 LLM 분석 게이트웨이.

    탭별 전문 분석을 단일 진입점으로 제공.
    Y1 특성: 3교대(주간/야간/연장야간), 213 locus(v2), ~9,000명/일

    흐름:
      1) packed_text → AnonymizationPipeline (K=20, 우회 불가)
      2) _build_prompt() → Y1 BASE_SYSTEM_PROMPT + TAB_CONTEXT
      3) _audit_log()    → 세션 감사
      4) _call_claude_with_system() → Anthropic 또는 Bedrock
    """

    @staticmethod
    def analyze(
        tab_id: str,
        packed_text: str,
        *,
        company_names: list[str] | None = None,
        worker_names: list[str] | None = None,
        zone_names: list[str] | None = None,
        date_list: list[str] | None = None,
        max_tokens: int = 900,
    ) -> str:
        """
        탭별 AI 분석 실행.

        Args:
            tab_id:        탭 식별자 ("daily"|"congestion"|"weekly"|"deep_space")
            packed_text:   DataPackager가 생성한 풍부한 데이터 텍스트
            company_names: 익명화할 업체명 목록 (자동 코드로 치환)
            worker_names:  익명화할 작업자명 목록
            zone_names:    익명화할 구역명 목록
            date_list:     날짜 목록 (상대화 처리)
            max_tokens:    LLM 최대 토큰

        Returns:
            LLM 응답 텍스트 (빈 문자열 = 실패)
        """
        from src.intelligence.anonymization_pipeline import AnonymizationPipeline

        # Step 1: 익명화 파이프라인 (우회 불가)
        safe_text = AnonymizationPipeline.run(
            packed_text,
            company_names=company_names,
            worker_names=worker_names,
            zone_names=zone_names,
            date_list=date_list,
        )

        # Step 2: 프롬프트 조립
        user_prompt = _build_prompt(tab_id, safe_text)

        # Step 3: 감사 로그
        _audit_log(tab_id, user_prompt)

        logger.info(
            "LLMGateway.analyze: tab=%s, chars=%d, companies=%d",
            tab_id,
            len(user_prompt),
            len(company_names or []),
        )

        # Step 4: LLM 호출 (Y1 BASE_SYSTEM_PROMPT 적용)
        return _call_claude_with_system(
            prompt=user_prompt,
            system=_BASE_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )

    @staticmethod
    def analyze_daily(
        data_dict: dict,
        shift_filter: str | None = None,
    ) -> str:
        """
        EWI/CRE 일별 분석.

        Args:
            data_dict:    DataPackager.worker()가 생성한 텍스트 또는 dict
            shift_filter: 교대 필터 ("주간"|"야간"|"연장야간"|None=전체)

        Returns:
            LLM 분석 결과
        """
        packed = data_dict if isinstance(data_dict, str) else str(data_dict)
        if shift_filter:
            packed = f"[교대 필터: {shift_filter}]\n\n{packed}"
        return LLMGateway.analyze(
            tab_id="daily",
            packed_text=packed,
            company_names=data_dict.get("company_names") if isinstance(data_dict, dict) else None,
            date_list=data_dict.get("date_list") if isinstance(data_dict, dict) else None,
        )

    @staticmethod
    def analyze_congestion(data_dict: dict | str) -> str:
        """
        공간 혼잡도 분석.

        Args:
            data_dict: DataPackager.congestion()이 생성한 텍스트 또는 dict

        Returns:
            LLM 분석 결과
        """
        packed = data_dict if isinstance(data_dict, str) else str(data_dict)
        return LLMGateway.analyze(
            tab_id="congestion",
            packed_text=packed,
            date_list=data_dict.get("date_list") if isinstance(data_dict, dict) else None,
        )

    @staticmethod
    def analyze_weekly(data_dict: dict | str) -> str:
        """
        주간 패턴 분석.

        Args:
            data_dict: DataPackager.weekly()가 생성한 텍스트 또는 dict

        Returns:
            LLM 분석 결과
        """
        packed = data_dict if isinstance(data_dict, str) else str(data_dict)
        return LLMGateway.analyze(
            tab_id="weekly",
            packed_text=packed,
            date_list=data_dict.get("date_list") if isinstance(data_dict, dict) else None,
        )

    @staticmethod
    def analyze_deep_space(data_dict: dict | str) -> str:
        """
        Agentic AI 예측 결과 해석.

        Args:
            data_dict: DataPackager.deep_space()가 생성한 텍스트 또는 dict

        Returns:
            LLM 분석 결과
        """
        packed = data_dict if isinstance(data_dict, str) else str(data_dict)
        return LLMGateway.analyze(
            tab_id="deep_space",
            packed_text=packed,
            max_tokens=900,
        )

    @staticmethod
    def is_available() -> bool:
        """LLM 사용 가능 여부."""
        from src.dashboard.llm_deepcon import is_llm_available
        return is_llm_available()
