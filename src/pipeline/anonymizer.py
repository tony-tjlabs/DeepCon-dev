"""
src.pipeline.anonymizer — DEPRECATED shim (M2-A, 2026-04-18)
=============================================================
이 모듈은 얇은 호환성 shim 이다. 새 코드는 다음을 import 하라:

    from core.security.anonymizer import (
        mask_name, mask_names_in_df, mask_name_series,
        mask_user_no, mask_twardid, mask_free_text,
        AnonymizationPipeline,
    )

역사:
  - v1 (M1 이전): 이 파일이 진짜 구현체였음.
  - M1 (2026-04-18 오전): core.security.anonymizer 로 PII 마스킹 규칙을 통합하고,
                         `src/utils/anonymizer.py` 는 shim 화. 이 파일은 여러 호출자
                         (llm_deepcon, report_context, anonymization_pipeline) 때문에
                         즉시 shim 화하지 못하고 잔존.
  - M2-A (2026-04-18 오후): 이제 shim 화. 기존 `Anonymizer` 클래스(코드명 치환식)는
                            호환을 위해 유지하되 DeprecationWarning.
                            단순 마스킹 함수는 core 로 위임.

보존 API (호환성용):
  - mask_name(name)                         → core.security.anonymizer
  - mask_name_series(series)                → core.security.anonymizer
  - Anonymizer(anonymize_logic=True)        → 이 파일 내 유지 (코드명 치환식)
  - relativize_metrics(...)                 → 이 파일 내 유지
  - anonymize_metrics_for_llm(...)          → 이 파일 내 유지
  - anonymize_high_risk_list(...)           → 이 파일 내 유지

M2-B 에서 `Anonymizer` 의존 지점(llm_deepcon, report_context)을 LLMGateway 로
마이그레이션하면 이 파일은 순수 shim 으로 더 얇아진다.
"""
from __future__ import annotations

import logging
import re
import warnings
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# ─── 단순 마스킹 함수 — core.security.anonymizer 로 위임 ─────────────
from core.security.anonymizer.core import (  # noqa: E402,F401
    mask_name,
    mask_name_series,
    mask_names_in_df,
    mask_user_no,
    mask_twardid,
    mask_free_text,
)

# 단 1회 shim import 경고
_warned = False


def _warn_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        "src.pipeline.anonymizer 는 deprecated 입니다. "
        "단순 마스킹은 `from core.security.anonymizer import mask_name` 로, "
        "LLM 익명화는 `core.ai.llm_gateway.LLMGateway` 로 이전하세요. "
        "(M2-B 에서 완전 이관 예정)",
        DeprecationWarning,
        stacklevel=3,
    )


_warn_once()


# ─── Anonymizer 클래스 (코드명 치환식) — 호환성 유지 ─────────────────
# 이 클래스는 "홍길동 → Worker_001" 처럼 숫자/알파벳 코드로 치환하는 방식이다.
# core.security.anonymizer 의 "홍길동 → 홍** (A-xxxxx)" 해시 기반과는 다른 목적 —
# 원문 복원이 가능한 세션 매핑 방식이므로 기존 llm_deepcon 플로우에서 필요.
# M2-B 에서 LLMGateway 가 이 기능을 대체하면 단계적으로 제거.

# 로직 추상화 매핑 (EWI/CRE 등 핵심 공식을 일반 용어로 변환)
LOGIC_ABSTRACTIONS = {
    # EWI 관련
    r"EWI\s*=\s*[^,\n\]]+": "작업 집중도 지수(계산식 비공개)",
    r"H_high\s*[×x*]\s*[\d.]+\s*\+\s*H_low\s*[×x*]\s*[\d.]+": "가중 활동 시간 합계",
    r"H_high\s*[×x*]\s*1\.0\s*\+\s*H_low\s*[×x*]\s*0\.5\s*\+\s*H_standby\s*[×x*]\s*0\.2": "가중 활동 시간(비공개)",
    r"active_ratio\s*[>=<]+\s*0\.\d+": "활성도 기준 적용",
    r"active_ratio\s*≥\s*0\.\d+": "활성도 기준 적용",
    # CRE 관련
    r"CRE\s*=\s*[^,\n\]]+": "복합 위험 노출도(계산식 비공개)",
    r"P_norm\s*\+\s*S_norm\s*\+\s*D_norm": "개인/공간/밀집 위험 통합",
    r"0\.\d+\s*[×x*]\s*[PFSDR]_\w+": "가중치 적용(비공개)",
    r"\d+\.\d+\s*[×x*]\s*P_norm\s*\+\s*\d+\.\d+\s*[×x*]\s*S_norm\s*\+\s*\d+\.\d+\s*[×x*]\s*D_norm": "위험 통합(비공개)",
    # 보정 관련
    r"DBSCAN": "클러스터링 알고리즘",
    r"Run-Length": "노이즈 보정",
    r"플리커\s*제거": "신호 보정",
    r"MAX_FLICKER_RUN\s*=\s*\d+": "보정 파라미터(비공개)",
    r"eps\s*=\s*[\d.]+": "파라미터(비공개)",
    r"min_samples\s*=\s*\d+": "파라미터(비공개)",
    # Word2Vec/K-means
    r"dim\s*=\s*\d+": "벡터 차원(비공개)",
    r"window\s*=\s*\d+": "윈도우 크기(비공개)",
    r"n_clusters\s*=\s*\d+": "클러스터 수(비공개)",
    r"Word2Vec": "임베딩 모델",
    r"K-means|KMeans|k-means": "클러스터링 모델",
    # 공간 위험 가중치
    r"confined_space.*?2\.0": "밀폐공간(고위험)",
    r"high_voltage.*?2\.0": "고압전(고위험)",
    r"mechanical_room.*?1\.8": "기계실(위험)",
    r"w_space\s*=\s*[\d.]+": "공간 위험 가중치(비공개)",
}


class Anonymizer:
    """
    DEPRECATED (M2-A) — 세션 코드(Worker_001, Company_A) 치환식 익명화.

    M2-B 에서 `core.ai.llm_gateway.LLMGateway` 로 마이그레이션 예정.
    당분간 호환성을 위해 유지.
    """

    def __init__(self, anonymize_logic: bool = True) -> None:
        _warn_once()
        self._map: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._counters = {"worker": 0, "company": 0, "zone": 0}
        self._anonymize_logic = anonymize_logic

    def mask(
        self,
        text: str,
        *,
        worker_names: Iterable[str] | None = None,
        company_names: Iterable[str] | None = None,
        zone_names: Iterable[str] | None = None,
    ) -> str:
        """텍스트 내 모든 알려진 이름을 익명 코드로 치환."""
        for name in _unique_sorted(worker_names):
            self._register(name, "worker")
        for name in _unique_sorted(company_names):
            self._register(name, "company")
        for name in _unique_sorted(zone_names):
            self._register(name, "zone")

        # 길이 내림차순 치환 (부분 치환 방지)
        pairs = sorted(self._map.items(), key=lambda kv: len(kv[0]), reverse=True)
        for original, masked in pairs:
            text = text.replace(original, masked)

        if self._anonymize_logic:
            text = self._mask_logic(text)
        return text

    def _mask_logic(self, text: str) -> str:
        for pattern, replacement in LOGIC_ABSTRACTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def unmask(self, text: str) -> str:
        """익명 코드를 원래 이름으로 복원 (로직 추상화는 비가역)."""
        pairs = sorted(self._reverse.items(), key=lambda kv: len(kv[0]), reverse=True)
        for masked, original in pairs:
            text = text.replace(masked, original)
        return text

    def get_mapping(self) -> dict[str, str]:
        return dict(self._map)

    def reset(self) -> None:
        self._map.clear()
        self._reverse.clear()
        self._counters = {"worker": 0, "company": 0, "zone": 0}

    def _register(self, name: str, category: str) -> str:
        name = name.strip()
        if not name or name in self._map:
            return self._map.get(name, "")
        self._counters[category] += 1
        idx = self._counters[category]
        if category == "worker":
            code = f"Worker_{idx:03d}"
        elif category == "company":
            code = f"Company_{_alpha_code(idx)}"
        else:
            code = f"Zone_{idx:03d}"
        self._map[name] = code
        self._reverse[code] = name
        return code


# ─── helpers ─────────────────────────────────────────────────────
def _unique_sorted(names: Iterable[str] | None) -> list[str]:
    if not names:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    result.sort(key=len, reverse=True)
    return result


def _alpha_code(n: int) -> str:
    """1→A, 2→B, ..., 26→Z, 27→AA, ..."""
    result = []
    while n > 0:
        n -= 1
        result.append(chr(65 + n % 26))
        n //= 26
    return "".join(reversed(result))


# ─── 수치 상대화 함수 (LLM 전송용) ───────────────────────────────────
def relativize_metrics(
    current_avg: float,
    baseline_avg: float,
    metric_name: str = "지표",
) -> str:
    """
    절대 수치 대신 상대적 변화로 표현.
    M2-B 에서 LLMGateway 내부에서 통합 처리 예정.
    """
    if baseline_avg == 0 or pd.isna(baseline_avg):
        if current_avg > 0.7:
            return f"{metric_name}가 높은 수준"
        elif current_avg > 0.4:
            return f"{metric_name}가 보통 수준"
        else:
            return f"{metric_name}가 낮은 수준"

    pct_change = (current_avg - baseline_avg) / baseline_avg * 100
    if current_avg >= 0.7:
        level = "상위 수준"
    elif current_avg >= 0.5:
        level = "중상위 수준"
    elif current_avg >= 0.3:
        level = "중간 수준"
    else:
        level = "하위 수준"

    if abs(pct_change) < 5:
        direction = "평소와 유사"
    elif pct_change > 0:
        direction = f"평소 대비 약 {abs(pct_change):.0f}% 높음"
    else:
        direction = f"평소 대비 약 {abs(pct_change):.0f}% 낮음"

    return f"{metric_name}가 {direction} ({level})"


def anonymize_metrics_for_llm(
    worker_count: int,
    avg_ewi: float,
    avg_cre: float,
    baseline_ewi: float = 0.5,
    baseline_cre: float = 0.35,
) -> str:
    """LLM 프롬프트용 메트릭 익명화 요약."""
    if worker_count < 100:
        worker_range = "소규모(100명 미만)"
    elif worker_count < 1000:
        worker_range = "중규모(100~1000명)"
    elif worker_count < 5000:
        worker_range = "대규모(1000~5000명)"
    else:
        worker_range = "초대규모(5000명 이상)"

    ewi_desc = relativize_metrics(avg_ewi, baseline_ewi, "작업 집중도")
    cre_desc = relativize_metrics(avg_cre, baseline_cre, "위험 노출도")

    return f"""
작업자 규모: {worker_range}
{ewi_desc}
{cre_desc}
    """.strip()


def anonymize_high_risk_list(
    high_risk_count: int,
    total_count: int,
) -> str:
    """고위험 작업자 수를 비율로 표현."""
    if total_count == 0:
        return "데이터 없음"
    ratio = high_risk_count / total_count * 100
    if ratio < 1:
        return "고위험 작업자 비율 매우 낮음 (1% 미만)"
    elif ratio < 5:
        return f"고위험 작업자 비율 낮음 (약 {ratio:.1f}%)"
    elif ratio < 10:
        return f"고위험 작업자 비율 보통 (약 {ratio:.1f}%)"
    else:
        return f"고위험 작업자 비율 높음 (약 {ratio:.1f}%)"


__all__ = [
    # core 위임
    "mask_name",
    "mask_name_series",
    "mask_names_in_df",
    "mask_user_no",
    "mask_twardid",
    "mask_free_text",
    # 호환성 유지 (M2-B 에서 이관)
    "Anonymizer",
    "LOGIC_ABSTRACTIONS",
    "relativize_metrics",
    "anonymize_metrics_for_llm",
    "anonymize_high_risk_list",
]
