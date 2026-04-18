"""
core.security.anonymizer.k_anonymity — K-Anonymity 파이프라인 (Y1 전용)
=========================================================================
Upgrade v3 M2-B: `src/intelligence/anonymization_pipeline.py` 흡수.
기존 모듈은 shim 으로 전환, 실제 로직은 여기로 이관.

보안 5단계 (Y1 특성 반영):
  Step 1: k-Anonymity    — 집단 크기 < K_ANON_MIN(20) 억제
  Step 2: 식별자 차단    — 이름/업체명/user_no → 세션 코드 (Anonymizer 위임)
  Step 3: 날짜 상대화    — "2026-03-10" → "분석 1일차(월)"
  Step 4: 수치 맥락화    — 절대값 + 상대 변화율 병기
  Step 5: 알고리즘 추상  — EWI/CRE 수식 등 제거 (ANONYMIZE_LOGIC)

Y1 K=20 근거:
  9,185명/208개 업체 규모에서 소규모 하도급(<20명) 식별 방지.
  M15X(200명 규모)의 K=10 대비 상향.

Locus 규칙:
  GW-XXX ID는 공간 코드(IP 아님)로 노출 허용.
  공간 고유명(건물 내 구체 위치)은 구역 그룹명으로 치환.

사용:
    from core.security.anonymizer.k_anonymity import (
        KAnonymizationPipeline, K_ANON_MIN,
    )

    safe_text = KAnonymizationPipeline.run(
        text,
        company_names=["A사"],
        date_list=["20260310"],
    )
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ─── k-Anonymity 임계값 (Y1) ─────────────────────────────────────────────────
# 20명 미만 집단 억제. 프로젝트별 상수로 노출, env 로 override 가능.
K_ANON_MIN = 20

# ─── 요일 레이블 ─────────────────────────────────────────────────────────────
_DOW_KR: list[str] = ["월", "화", "수", "목", "금", "토", "일"]

# ─── Y1 locus 구역 그룹 매핑 (공간 실명 → 그룹명) ────────────────────────────
# GW-XXX ID 자체는 노출 허용, 공간 고유명만 치환.
_LOCUS_GROUP_MAP: dict[str, str] = {
    "FAB":         "FAB 구역",
    "CUB":         "CUB 구역",
    "WWT":         "WWT 구역",
    "GATE":        "출입 게이트",
    "154kV":       "전력 설비 구역",
    "야외":        "야외 작업 구역",
    "타워크레인":  "야외 작업 구역",
    "본진":        "현장 지원 구역",
    "전진":        "전진 구역",
}


# ─── config helpers ──────────────────────────────────────────────────────────
def _load_cfg() -> tuple[bool, bool]:
    """(ANONYMIZE_LLM, ANONYMIZE_LOGIC) 반환."""
    try:
        import config as cfg
        return (
            getattr(cfg, "ANONYMIZE_LLM",   True),
            getattr(cfg, "ANONYMIZE_LOGIC", True),
        )
    except Exception:
        return True, True


def _get_session_anonymizer():
    """
    세션 스코프 ``Anonymizer`` 반환.
    같은 세션 내에서 업체명 코드가 일관되게 유지된다.

    Note:
        ``src.pipeline.anonymizer.Anonymizer`` 는 코드명 치환식 (Company_A) 익명화
        기능을 담당. K-anonymity 와 달리 LLM 에 전달되는 세션 코드 매핑을 제공.
        M2-B 에서 LLMGateway 로 마이그레이션되면 이 헬퍼도 교체될 예정.
    """
    try:
        import streamlit as st
        _, anonymize_logic = _load_cfg()
        key = f"_k_anon_pipeline_{anonymize_logic}"
        if key not in st.session_state:
            from src.pipeline.anonymizer import Anonymizer
            st.session_state[key] = Anonymizer(anonymize_logic=anonymize_logic)
        return st.session_state[key]
    except Exception:
        # Streamlit 컨텍스트 외부 — 임시 인스턴스
        from src.pipeline.anonymizer import Anonymizer
        _, anonymize_logic = _load_cfg()
        return Anonymizer(anonymize_logic=anonymize_logic)


# ─── 메인 파이프라인 ─────────────────────────────────────────────────────────
class KAnonymizationPipeline:
    """
    K-anonymity + LLM 전송 전 데이터 익명화 단일 파이프라인 (Y1 전용).

    모든 메서드는 classmethods — 인스턴스화 불필요.
    K_ANON_MIN = 20 (M15X 의 10 에서 상향).
    """

    # ─── 메인 진입점 ──────────────────────────────────────────────────────
    @classmethod
    def run(
        cls,
        text: str,
        *,
        company_names: Iterable[str] | None = None,
        worker_names: Iterable[str] | None = None,
        zone_names: Iterable[str] | None = None,
        date_list: Iterable[str] | None = None,
    ) -> str:
        """
        텍스트를 K-anonymity 파이프라인에 통과시킨다.

        Args:
            text:          익명화할 원본 텍스트
            company_names: 업체명 (Company_A 등 세션 코드로 치환)
            worker_names:  작업자명 (Worker_001 등)
            zone_names:    구역명 (Zone_001 등)
            date_list:     날짜 ("20260310" 또는 "2026-03-10")

        Returns:
            익명화된 텍스트
        """
        anonymize_llm, _ = _load_cfg()
        if not anonymize_llm:
            return text

        anon = _get_session_anonymizer()

        # Step 2: 식별자 → 코드 치환
        text = anon.mask(
            text,
            worker_names=list(worker_names) if worker_names else [],
            company_names=list(company_names) if company_names else [],
            zone_names=list(zone_names) if zone_names else [],
        )

        # Step 3: 날짜 상대화
        if date_list:
            text = cls._relativize_dates(text, list(date_list))

        # user_no 패턴 제거
        text = re.sub(r"\buser_no\s*[:=]\s*\d+", "[작업자ID 제거됨]", text)

        return text

    @classmethod
    def get_company_code(cls, company_name: str) -> str:
        """업체명의 세션 내 익명 코드를 반환 (Company_A 형식)."""
        anon = _get_session_anonymizer()
        return anon._register(company_name, "company")

    # ─── k-Anonymity ──────────────────────────────────────────────────────
    @classmethod
    def filter_small_groups(
        cls,
        rows: list[dict],
        count_key: str = "n_workers",
    ) -> list[dict]:
        """
        집단 크기가 K_ANON_MIN(20) 미만인 항목을 억제.
        수치 컬럼 → None, 카운트 키 → "<K명 (비공개)" 표시.
        """
        result: list[dict] = []
        for item in rows:
            count = item.get(count_key, 0)
            if isinstance(count, (int, float)) and count < K_ANON_MIN:
                suppressed: dict[str, Any] = {}
                for k, v in item.items():
                    if k == count_key:
                        suppressed[k] = f"<{K_ANON_MIN}명 (비공개)"
                    elif isinstance(v, (int, float)):
                        suppressed[k] = None
                    else:
                        suppressed[k] = v
                result.append(suppressed)
            else:
                result.append(item)
        return result

    @classmethod
    def check_k_anonymity(cls, count: int) -> bool:
        """단일 집단의 k-Anonymity 충족 여부 (Y1: K=20)."""
        return count >= K_ANON_MIN

    # ─── 날짜 상대화 ──────────────────────────────────────────────────────
    @classmethod
    def _relativize_dates(cls, text: str, date_list: list[str]) -> str:
        """
        절대 날짜를 상대적 표현으로 치환.

        "2026-03-10" → "분석 1일차(월)"
        "20260310"   → "분석 1일차(월)"
        """
        parsed: list[tuple[str, str, datetime]] = []

        for d in date_list:
            try:
                d = d.strip()
                if len(d) == 8 and d.isdigit():
                    dt    = datetime.strptime(d, "%Y%m%d")
                    dash  = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    plain = d
                elif len(d) >= 10:
                    dt    = datetime.strptime(d[:10], "%Y-%m-%d")
                    dash  = d[:10]
                    plain = d[:10].replace("-", "")
                else:
                    continue
                parsed.append((dash, plain, dt))
            except Exception:
                continue

        parsed.sort(key=lambda x: x[2])
        for idx, (dash, plain, dt) in enumerate(parsed, 1):
            dow   = _DOW_KR[dt.weekday()]
            label = f"분석 {idx}일차({dow})"
            text  = text.replace(dash,  label)
            text  = text.replace(plain, label)

        return text

    # ─── 수치 맥락화 헬퍼 ─────────────────────────────────────────────────
    @staticmethod
    def relativize(
        value: float,
        baseline: float,
        label: str,
        unit: str = "",
    ) -> str:
        """
        절대값을 상대적 변화와 함께 표현.

        >>> KAnonymizationPipeline.relativize(0.52, 0.48, "EWI")
        'EWI 0.52 (기준 대비 +8.3%)'
        """
        if baseline and baseline != 0:
            pct  = (value - baseline) / baseline * 100
            sign = "+" if pct >= 0 else ""
            return f"{label} {value:.2f}{unit} (기준 대비 {sign}{pct:.1f}%)"
        return f"{label} {value:.2f}{unit}"

    @staticmethod
    def level_label(value: float, thresholds: dict[str, float]) -> str:
        """
        값에 따른 레벨 레이블.

        thresholds: {"높음": 0.7, "보통": 0.4, "낮음": 0.0}
        """
        for label, thresh in sorted(thresholds.items(), key=lambda x: -x[1]):
            if value >= thresh:
                return label
        return list(thresholds.keys())[-1]

    # ─── Y1 전용: Locus 익명화 ────────────────────────────────────────────
    @staticmethod
    def anonymize_locus(locus_id: str, locus_name: str | None = None) -> str:
        """
        Y1 전용 Locus 익명화 규칙.

        GW-XXX 형식 locus ID는 공간 코드로 그대로 노출해도 무방.
        공간 실명(건물 내 구체 위치)은 구역 그룹명으로 치환.
        """
        if not locus_name:
            return locus_id
        for keyword, group_name in _LOCUS_GROUP_MAP.items():
            if keyword in locus_name:
                return f"{locus_id} ({group_name})"
        return locus_id

    # ─── 사전 구조 데이터 익명화 ──────────────────────────────────────────
    @classmethod
    def sanitize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        딕셔너리에서 민감 키를 제거.

        제거 키: user_no, user_name, twardid, x, y, phone, email, cellphone
        """
        BLOCKED_KEYS = {
            "user_no", "user_name", "twardid", "worker_name",
            "x", "y", "phone", "email", "cellphone",
        }
        return {k: v for k, v in data.items() if k.lower() not in BLOCKED_KEYS}


# ─── 레거시 이름 alias (기존 import 경로 호환) ──────────────────────────────
# src.intelligence.anonymization_pipeline 의 AnonymizationPipeline 을 흡수.
# 충돌 회피: core.security.anonymizer.llm_pipeline.AnonymizationPipeline 과 다른 클래스이므로
# 여기서는 KAnonymizationPipeline 이 정식명, K-anonymity 전용 alias 는 별도 제공하지 않음.

__all__ = [
    "K_ANON_MIN",
    "KAnonymizationPipeline",
]
