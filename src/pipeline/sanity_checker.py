"""
Sanity Checker — 비상식 패턴 탐지 (Physical/Physiological Plausibility Check)
============================================================================
단순한 데이터 오류 감지가 아닌 **현실 물리/생리적 불가능 패턴** 탐지.

설계 원칙:
  - "사람은 화장실도 가고 이동도 한다" — 정상 인간 행동 패턴 기반
  - 통계적 이상치(anomaly_detector.py)와 역할 분리:
      sanity_checker.py : 파이프라인 처리 시 룰 기반 물리/생리 불가 패턴
      anomaly_detector.py: 대시보드 표시 시 통계적 이상치 탐지 (기존 유지)
  - 컬럼 없으면 해당 규칙 skip (방어적 처리)
  - worker.parquet에 sanity 컬럼 4개 추가 (pipeline 통합)

추가 컬럼:
    sanity_flags      : 트리거된 규칙 ID 목록 (JSON 문자열, 예: '["NO_BREAK_LONG_SHIFT"]')
    sanity_flag_count : 트리거된 규칙 수 (int)
    sanity_severity   : 최고 심각도 (HIGH / MEDIUM / LOW / OK)
    is_suspicious     : flag_count > 0 AND severity in (HIGH, MEDIUM) (bool)

규칙 목록:
    [물리적 불가]
    NO_BREAK_LONG_SHIFT    : 6h+ 근무 중 휴게 0분
    NO_TRANSIT_LONG_SHIFT  : 4h+ 근무 중 이동 0분
    ALL_DAY_HIGH_ACTIVE    : 4h+ 근무 중 고활성 비율 95% 초과
    SINGLE_LOCUS_ALL_DAY   : 6h+ 근무 중 방문 공간 1개
    EXCESSIVE_WORK_HOURS   : 근무시간 12h 초과
    WORK_ZONE_RATIO_PERFECT: 작업구역 비율 99% 초과 (4h+ 근무)

    [데이터 정합성 의심]
    EWI_TOO_PERFECT        : EWI 0.95 초과 (4h+ 근무)
    HELMET_ABANDONED       : 헬멧 방치 의심 (이미 processor.py 계산값 참조)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ─── 심각도 우선순위 ────────────────────────────────────────────────────
_SEVERITY_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "OK": 0}


def _max_severity(severities: list[str]) -> str:
    """심각도 목록에서 최고 심각도 반환."""
    if not severities:
        return "OK"
    return max(severities, key=lambda s: _SEVERITY_ORDER.get(s, 0))


# ─── Sanity 규칙 정의 ───────────────────────────────────────────────────
# 각 규칙은 dict: id, desc, severity, required_cols, condition
# condition(row_dict) → bool (True = 규칙 트리거됨)

def _make_rules() -> list[dict[str, Any]]:
    """규칙 정의 목록 반환. 함수로 감싸 import 시점 부작용 방지."""

    def _get(row: dict, col: str, default=0):
        v = row.get(col, default)
        if v is None or (isinstance(v, float) and v != v):  # NaN 체크
            return default
        return v

    rules = [
        # ── 물리적 불가 ──────────────────────────────────────────────
        {
            "id": "NO_BREAK_LONG_SHIFT",
            # ★ 강화 조건 (2026-04-17):
            # 단순 rest=0은 Y1 현장에서 REST_AREA BLE 미감지 때문에 과다 트리거 → 제외
            # "6h+ 근무 + rest=0 + transit=0 + unique_loci <= 2" 조합만 진짜 비상식
            # (이동도 없고 한 공간에만 있으면서 휴게도 없는 경우)
            "desc": "6h+ 근무 중 휴게·이동 모두 0분이며 방문 공간 2개 이하 — 기계가 아닌 이상 불가능",
            "severity": "HIGH",
            "required_cols": {"work_minutes"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 360
                and _get(r, "rest_minutes", 0) == 0
                and _get(r, "rest_min", 0) == 0
                and _get(r, "transit_min", 0) == 0
                and _get(r, "transition_count", 0) == 0
                and _get(r, "unique_loci", 99) <= 2
            ),
        },
        {
            "id": "NO_TRANSIT_LONG_SHIFT",
            "desc": "6시간 이상 근무 중 이동 0분이며 방문 공간 2개 이하 — 하루 종일 한 공간에서 꼼짝 안 함",
            "severity": "MEDIUM",
            "required_cols": {"work_minutes"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 360
                and _get(r, "transit_min", 0) == 0
                and _get(r, "transition_count", 0) == 0
                and _get(r, "unique_loci", 99) <= 2
            ),
        },
        {
            "id": "ALL_DAY_HIGH_ACTIVE",
            "desc": "4시간 이상 근무 중 고활성 비율 95% 초과 — 4시간 내내 몸을 움직이는 건 불가능",
            "severity": "HIGH",
            "required_cols": {"work_minutes", "high_active_min"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 240
                and _get(r, "high_active_min") / max(_get(r, "work_minutes"), 1) > 0.95
            ),
        },
        {
            "id": "SINGLE_LOCUS_ALL_DAY",
            "desc": "6시간 이상 근무 중 방문 공간 1개 이하 — 화장실도 안 가는 사람은 없다",
            "severity": "HIGH",
            "required_cols": {"work_minutes", "unique_loci"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 360
                and _get(r, "unique_loci", 99) <= 1
            ),
        },
        {
            "id": "EXCESSIVE_WORK_HOURS",
            # ★ Y1 현장 현실 반영 (2026-04-17):
            # 교대 근무 특성상 하루 12시간 근무는 흔함 → 18시간(1080분) 초과만 의심
            # 일반 12h 초과는 MEDIUM에서 제외 (과다 트리거 방지)
            "desc": "근무시간 18시간 초과 — 단순 교대 초과를 넘는 극단적 장시간 근무",
            "severity": "MEDIUM",
            "required_cols": {"work_minutes"},
            "condition": lambda r: _get(r, "work_minutes") > 1080,
        },
        {
            "id": "WORK_ZONE_RATIO_PERFECT",
            "desc": "작업구역 비율 99% 초과 (4h+ 근무) — 이동/휴게 없이 작업구역만 있는 건 센서 이상",
            "severity": "HIGH",
            "required_cols": {"work_minutes", "work_zone_minutes"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 240
                and _get(r, "work_zone_minutes") / max(_get(r, "work_minutes"), 1) > 0.99
            ),
        },
        # ── 데이터 정합성 의심 ──────────────────────────────────────
        {
            "id": "EWI_TOO_PERFECT",
            "desc": "EWI 0.95 초과 (4h+ 근무) — 완벽한 효율은 현실적으로 불가능",
            "severity": "MEDIUM",
            "required_cols": {"work_minutes", "ewi"},
            "condition": lambda r: (
                _get(r, "work_minutes") >= 240
                and _get(r, "ewi") is not None
                and _get(r, "ewi") > 0.95
            ),
        },
        {
            "id": "HELMET_ABANDONED",
            "desc": "헬멧 방치 의심 — 퇴근 후 헬멧만 현장에 남겨진 것으로 추정",
            "severity": "HIGH",
            "required_cols": {"helmet_abandoned"},
            "condition": lambda r: bool(_get(r, "helmet_abandoned", False)),
        },
    ]
    return rules


# ─── 핵심 함수 ──────────────────────────────────────────────────────────

def check_worker_sanity(worker_df: pd.DataFrame) -> pd.DataFrame:
    """
    worker_df에 sanity 관련 컬럼 4개 추가.

    Args:
        worker_df: processor.py의 calc_basic_metrics() + add_metrics_to_worker() 결과

    Returns:
        worker_df에 아래 컬럼이 추가된 DataFrame:
            sanity_flags      : JSON 문자열 형태의 트리거 규칙 ID 목록
                                예) '["NO_BREAK_LONG_SHIFT", "SINGLE_LOCUS_ALL_DAY"]'
                                없으면 '[]'
            sanity_flag_count : int — 트리거 규칙 수
            sanity_severity   : str — 최고 심각도 (HIGH / MEDIUM / LOW / OK)
            is_suspicious     : bool — flag_count > 0 AND severity in (HIGH, MEDIUM)

    Note:
        - 필요 컬럼 없으면 해당 규칙 skip (방어적 처리)
        - worker_df를 copy하지 않고 직접 컬럼 추가 (파이프라인 효율)
    """
    if worker_df.empty:
        worker_df["sanity_flags"]      = "[]"
        worker_df["sanity_flag_count"] = 0
        worker_df["sanity_severity"]   = "OK"
        worker_df["is_suspicious"]     = False
        return worker_df

    rules = _make_rules()
    available_cols = set(worker_df.columns)

    # 실제 적용 가능한 규칙만 필터 (필요 컬럼 존재 여부)
    applicable_rules: list[dict] = []
    for rule in rules:
        required = rule.get("required_cols", set())
        if required.issubset(available_cols):
            applicable_rules.append(rule)
        else:
            missing = required - available_cols
            logger.debug("규칙 '%s' skip — 컬럼 없음: %s", rule["id"], missing)

    if not applicable_rules:
        logger.warning("적용 가능한 sanity 규칙 없음 (필요 컬럼 부재). 기본값 설정.")
        worker_df["sanity_flags"]      = "[]"
        worker_df["sanity_flag_count"] = 0
        worker_df["sanity_severity"]   = "OK"
        worker_df["is_suspicious"]     = False
        return worker_df

    logger.info("Sanity check: %d개 규칙 적용 (총 %d개 중)", len(applicable_rules), len(rules))

    # ── 벡터화 불가 규칙이므로 itertuples로 효율 처리 ──────────────────
    # 필요 컬럼만 dict 변환 (불필요한 컬럼 제외하여 메모리 절약)
    needed_cols = set()
    for r in applicable_rules:
        needed_cols.update(r.get("required_cols", set()))
    # 추가 참조 컬럼 (condition에서 _get으로 접근하는 것들)
    extra_cols = {
        "rest_minutes", "rest_min", "transit_min", "transition_count",
        "unique_loci", "helmet_abandoned"
    }
    needed_cols.update(extra_cols & available_cols)

    # 행별 규칙 적용
    all_flags:     list[str] = []  # JSON 문자열 목록
    all_counts:    list[int] = []
    all_severities: list[str] = []

    for row_tuple in worker_df[list(needed_cols)].itertuples(index=False):
        row_dict = row_tuple._asdict()

        triggered_ids:    list[str] = []
        triggered_sevs:   list[str] = []

        for rule in applicable_rules:
            try:
                if rule["condition"](row_dict):
                    triggered_ids.append(rule["id"])
                    triggered_sevs.append(rule["severity"])
            except Exception as e:
                logger.debug("규칙 '%s' 실행 오류: %s", rule["id"], e)

        all_flags.append(json.dumps(triggered_ids, ensure_ascii=False))
        all_counts.append(len(triggered_ids))
        all_severities.append(_max_severity(triggered_sevs))

    worker_df["sanity_flags"]      = all_flags
    worker_df["sanity_flag_count"] = all_counts
    worker_df["sanity_severity"]   = all_severities
    worker_df["is_suspicious"]     = [
        cnt > 0 and sev in ("HIGH", "MEDIUM")
        for cnt, sev in zip(all_counts, all_severities)
    ]

    # 결과 요약 로그
    n_total    = len(worker_df)
    n_suspicious = int(sum(all_counts_v > 0 for all_counts_v in all_counts))
    n_high     = sum(1 for s in all_severities if s == "HIGH")
    n_medium   = sum(1 for s in all_severities if s == "MEDIUM")
    n_low      = sum(1 for s in all_severities if s == "LOW")

    logger.info(
        "Sanity check 완료: 총 %d명 / 이상 감지 %d명 (%.1f%%) "
        "[HIGH:%d, MEDIUM:%d, LOW:%d]",
        n_total, n_suspicious, n_suspicious / max(n_total, 1) * 100,
        n_high, n_medium, n_low,
    )

    return worker_df


# ─── 규칙 메타 정보 조회 ────────────────────────────────────────────────

def get_rule_descriptions() -> dict[str, dict]:
    """
    규칙 ID → {desc, severity} 매핑 반환.
    대시보드에서 플래그 설명 표시 시 사용.
    """
    return {
        r["id"]: {"desc": r["desc"], "severity": r["severity"]}
        for r in _make_rules()
    }


def get_sanity_summary(worker_df: pd.DataFrame) -> dict:
    """
    worker_df (sanity 컬럼 포함 가정)에서 요약 통계 반환.

    Returns:
        {
            "total": int,
            "suspicious": int,
            "suspicious_pct": float,
            "high": int, "medium": int, "low": int,
            "rule_counts": {rule_id: int},
        }
    """
    if "sanity_flag_count" not in worker_df.columns:
        return {"total": len(worker_df), "suspicious": 0, "suspicious_pct": 0.0,
                "high": 0, "medium": 0, "low": 0, "rule_counts": {}}

    n_total = len(worker_df)
    sev = worker_df["sanity_severity"]
    suspicious = int((worker_df["sanity_flag_count"] > 0).sum())

    # 규칙별 트리거 횟수 집계
    rule_counts: dict[str, int] = {}
    for flags_json in worker_df["sanity_flags"].dropna():
        try:
            for rule_id in json.loads(flags_json):
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "total":           n_total,
        "suspicious":      suspicious,
        "suspicious_pct":  round(suspicious / max(n_total, 1) * 100, 1),
        "high":            int((sev == "HIGH").sum()),
        "medium":          int((sev == "MEDIUM").sum()),
        "low":             int((sev == "LOW").sum()),
        "rule_counts":     rule_counts,
    }
