"""
Data Validator — 데이터 정합성 검증 모듈 (v3, Check DSL 기반)
================================================================
파이프라인 처리 후 데이터 품질을 검증하여 신뢰할 수 있는 분석 결과를 보장한다.

v3 변경사항 (Upgrade v3 T-03)
  - 선언적 `Check` DSL 도입: 검증 규칙을 데이터클래스로 표현
  - **NaN-safe 처리 일괄 적용**: `fillna(False).astype(bool)` 패턴
    → 이전 버전의 "Cannot mask with non-boolean array containing NA / NaN values" 버그 근본 해결
  - 기존 3개 validator(run_all_validations / generate_quality_report)와의 출력 호환성 유지

검증 항목:
  1. BLE 커버리지: 근무 시간 대비 BLE 감지 비율
  2. Journey 연속성: Adjacency 기반 유효 이동 비율
  3. 출퇴근 일관성: AccessLog vs TWardData 시간 오차

사용법 (기존 호환):
    from src.pipeline.validator import run_all_validations, generate_quality_report

    results = run_all_validations(journey_df, worker_df, access_df, spatial_graph)
    report = generate_quality_report(results)

신규 DSL 사용법:
    from src.pipeline.validator import Check, run_checks

    checks = [
        Check(id="NAN_FREE_LOCUS", description="locus_id NaN 없음",
              severity="error",
              predicate=lambda ctx: ctx["journey"]["locus_id"].isna()),
        ...
    ]
    results = run_checks(checks, ctx={"journey": jdf, "worker": wdf})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.spatial.graph import SpatialGraph

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 공통 유틸 — NaN-safe bool mask 변환
# ═════════════════════════════════════════════════════════════════════

def _safe_bool_mask(mask: Any) -> pd.Series:
    """
    NaN-safe boolean mask 변환.

    Pandas boolean 타입 (BooleanArray), nullable bool, float(NaN 포함),
    object(True/False/NaN) 등 어떤 입력이 와도 안전하게 bool mask로 변환.

    - NaN/None/pd.NA → False
    - 나머지 → bool 캐스팅

    이 함수는 validator v3 NaN 버그의 **핵심 수정 포인트**.
    """
    if not isinstance(mask, pd.Series):
        mask = pd.Series(mask)
    return mask.fillna(False).astype(bool)


def _safe_invert(mask: Any) -> pd.Series:
    """NaN-safe `~` 연산. (NaN → False 처리 후 반전)"""
    return ~_safe_bool_mask(mask)


# ═════════════════════════════════════════════════════════════════════
# 선언적 Check DSL
# ═════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """단일 Check 실행 결과."""
    check_id: str
    description: str
    severity: str              # "error" | "warn" | "info"
    passed: bool
    violations: int = 0
    total: int = 0
    error: str | None = None   # 예외 발생 시 메시지

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "severity": self.severity,
            "passed": self.passed,
            "violations": self.violations,
            "total": self.total,
            "error": self.error,
        }


@dataclass
class Check:
    """
    선언적 검증 규칙.

    Attributes:
        id: 유니크 식별자 (대문자 + 언더스코어)
        description: 사람이 읽는 설명
        severity: "error" | "warn" | "info"
        predicate: (ctx: dict) → pd.Series[bool]
                   True가 violation을 의미 (관례)
                   또는 단일 bool 값(True=violation)

    예시:
        Check(id="NULL_LOCUS",
              description="locus_id가 NaN인 행 없음",
              severity="error",
              predicate=lambda ctx: ctx["journey"]["locus_id"].isna())
    """
    id: str
    description: str
    severity: str
    predicate: Callable[[dict], Any]

    def run(self, ctx: dict) -> CheckResult:
        try:
            raw = self.predicate(ctx)
            # 단일 bool (스칼라) 처리
            if isinstance(raw, (bool, np.bool_)):
                violated = bool(raw)
                return CheckResult(
                    check_id=self.id,
                    description=self.description,
                    severity=self.severity,
                    passed=(not violated),
                    violations=(1 if violated else 0),
                    total=1,
                )
            # Series 처리 (NaN-safe)
            mask = _safe_bool_mask(raw)
            violations = int(mask.sum())
            total = int(len(mask))
            return CheckResult(
                check_id=self.id,
                description=self.description,
                severity=self.severity,
                passed=(violations == 0),
                violations=violations,
                total=total,
            )
        except Exception as e:
            logger.warning("Check %s 실행 실패: %s", self.id, e)
            return CheckResult(
                check_id=self.id,
                description=self.description,
                severity=self.severity,
                passed=False,
                error=str(e),
            )


def run_checks(checks: list[Check], ctx: dict) -> list[CheckResult]:
    """DSL Check 목록 실행."""
    return [c.run(ctx) for c in checks]


# ═════════════════════════════════════════════════════════════════════
# 기본 Check 카탈로그 (DSL 기반, 파이프라인 내부 NaN-safe 보증용)
# ═════════════════════════════════════════════════════════════════════

DEFAULT_CHECKS: list[Check] = [
    Check(
        id="NAN_FREE_LOCUS_ID",
        description="journey.locus_id NaN 없음",
        severity="error",
        predicate=lambda ctx: ctx["journey"]["locus_id"].isna() if "journey" in ctx and not ctx["journey"].empty else False,
    ),
    Check(
        id="NAN_FREE_USER_NO",
        description="journey.user_no NaN 없음",
        severity="error",
        predicate=lambda ctx: ctx["journey"]["user_no"].isna() if "journey" in ctx and not ctx["journey"].empty else False,
    ),
    Check(
        id="WORK_ZONE_CAP",
        description="work_zone_minutes ≤ work_minutes + 1분 (float 오차 허용)",
        severity="warn",
        predicate=lambda ctx: (
            (ctx["worker"]["work_zone_minutes"].astype(float)
             > ctx["worker"]["work_minutes"].astype(float).fillna(0) + 1.0)
            if "worker" in ctx and not ctx["worker"].empty
               and "work_zone_minutes" in ctx["worker"].columns
               and "work_minutes" in ctx["worker"].columns
            else False
        ),
    ),
    Check(
        id="TIME_MONOTONIC",
        description="user별 timestamp 단조증가",
        severity="warn",
        predicate=lambda ctx: (
            ctx["journey"]
                .sort_values(["user_no", "timestamp"])
                .groupby("user_no")["timestamp"]
                .diff()
                .dt.total_seconds()
                .lt(0)
            if "journey" in ctx and not ctx["journey"].empty
               and "timestamp" in ctx["journey"].columns
            else False
        ),
    ),
]


# ═════════════════════════════════════════════════════════════════════
# 레거시 호환 레이어 (기존 ValidationLevel / ValidationResult / 3개 validator)
# ═════════════════════════════════════════════════════════════════════

class ValidationLevel(Enum):
    """검증 결과 등급."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationResult:
    """검증 결과 (기존 API)."""

    check_name: str
    level: ValidationLevel
    metric: float
    threshold_pass: float
    threshold_warning: float
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "level": self.level.value,
            "metric": round(self.metric, 4),
            "threshold_pass": self.threshold_pass,
            "threshold_warning": self.threshold_warning,
            "message": self.message,
            **self.details,
        }


# ─── BLE 커버리지 검증 ─────────────────────────────────────────────────────


def validate_ble_coverage(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
) -> ValidationResult:
    """
    BLE 커버리지 검증.

    계산: 평균 (1 - gap_ratio) = BLE 감지 비율
    임계값:
      - PASS: >= 70%
      - WARNING: 50-70%
      - FAIL: < 50%
    """
    THRESHOLD_PASS = 0.70
    THRESHOLD_WARNING = 0.50

    if worker_df.empty or "gap_ratio" not in worker_df.columns:
        return ValidationResult(
            check_name="BLE 커버리지",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="gap_ratio 데이터 없음 — 검증 불가",
            details={"reason": "missing_column"},
        )

    # gap_ratio가 NaN인 경우 제외 (work_minutes=0인 작업자)
    valid_workers = worker_df[worker_df["gap_ratio"].notna()]
    if valid_workers.empty:
        return ValidationResult(
            check_name="BLE 커버리지",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="유효한 gap_ratio 데이터 없음",
            details={"reason": "no_valid_data"},
        )

    # 평균 커버리지 = 1 - 평균 gap_ratio
    avg_gap_ratio = valid_workers["gap_ratio"].mean()
    coverage = 1.0 - avg_gap_ratio

    # 등급 판정
    if coverage >= THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"BLE 커버리지 양호 ({coverage * 100:.1f}%)"
    elif coverage >= THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"BLE 커버리지 주의 ({coverage * 100:.1f}%) — 부분 음영 구역 존재"
    else:
        level = ValidationLevel.FAIL
        message = f"BLE 커버리지 부족 ({coverage * 100:.1f}%) — 음영 구역 점검 필요"

    # 상세 정보 (NaN-safe: 비교 결과가 nullable bool일 수 있으므로 _safe_bool_mask 사용)
    gap = valid_workers["gap_ratio"]
    coverage_dist = {
        "정상 (>=70%)":     int(_safe_bool_mask(gap <= 0.30).sum()),
        "부분음영 (50-70%)": int(_safe_bool_mask((gap > 0.30) & (gap <= 0.50)).sum()),
        "음영 (30-50%)":    int(_safe_bool_mask((gap > 0.50) & (gap <= 0.70)).sum()),
        "미측정 (<30%)":    int(_safe_bool_mask(gap > 0.70).sum()),
    }

    return ValidationResult(
        check_name="BLE 커버리지",
        level=level,
        metric=coverage,
        threshold_pass=THRESHOLD_PASS,
        threshold_warning=THRESHOLD_WARNING,
        message=message,
        details={
            "avg_gap_ratio": round(avg_gap_ratio, 4),
            "total_workers": len(valid_workers),
            "coverage_distribution": coverage_dist,
        },
    )


# ─── Journey 연속성 검증 ──────────────────────────────────────────────────


def validate_journey_continuity(
    journey_df: pd.DataFrame,
    spatial_graph: "SpatialGraph | None" = None,
) -> ValidationResult:
    """
    Journey 연속성 검증 (비인접 직접 이동 비율).

    ★ v3 NaN-safe: `locus_changed`, `is_valid_transition` 컬럼에 NaN이 섞여 있어도
    boolean indexing / ~ 연산이 안전하게 동작하도록 _safe_bool_mask() 일괄 적용.

    계산: 비인접 직접 이동 수 / 전체 이동 수
    임계값:
      - PASS: < 5%
      - WARNING: 5-15%
      - FAIL: >= 15%
    """
    THRESHOLD_PASS = 0.05
    THRESHOLD_WARNING = 0.15

    if journey_df.empty:
        return ValidationResult(
            check_name="Journey 연속성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="Journey 데이터 없음",
            details={"reason": "empty_data"},
        )

    # is_valid_transition 컬럼이 있으면 사용
    if "is_valid_transition" in journey_df.columns:
        # ★ NaN-safe: locus_changed에 NaN이 있을 수 있음 (tokenizer merge 이후)
        if "locus_changed" in journey_df.columns:
            changed_mask = _safe_bool_mask(journey_df["locus_changed"])
            transitions = journey_df[changed_mask]
        else:
            # locus_changed 없으면 prev_locus와 비교 (NaN 안전)
            prev = journey_df.get("prev_locus", pd.Series("", index=journey_df.index))
            transitions = journey_df[journey_df["locus_id"].fillna("") != prev.fillna("")]

        if len(transitions) == 0:
            return ValidationResult(
                check_name="Journey 연속성",
                level=ValidationLevel.PASS,
                metric=1.0,
                threshold_pass=THRESHOLD_PASS,
                threshold_warning=THRESHOLD_WARNING,
                message="이동 기록 없음 (단일 위치 체류)",
                details={"total_transitions": 0},
            )

        # ★ 핵심 NaN 버그 수정 지점:
        # 이전: invalid_count = (~transitions["is_valid_transition"]).sum()
        #       → is_valid_transition이 nullable bool/float(NaN)일 때 실패
        # 이후: _safe_invert()로 NaN → False 처리 후 반전
        invalid_mask = _safe_invert(transitions["is_valid_transition"])
        invalid_count = int(invalid_mask.sum())
        total_transitions = int(len(transitions))
        invalid_ratio = invalid_count / total_transitions if total_transitions > 0 else 0.0

    elif spatial_graph is not None and "prev_locus" in journey_df.columns:
        # SpatialGraph로 직접 검증
        invalid_df = spatial_graph.detect_impossible_transitions(journey_df)
        invalid_count = int(len(invalid_df))

        # 전체 이동 수 (NaN-safe)
        if "locus_changed" in journey_df.columns:
            total_transitions = int(_safe_bool_mask(journey_df["locus_changed"]).sum())
        else:
            total_transitions = int(
                (journey_df["locus_id"].fillna("") != journey_df["prev_locus"].fillna("")).sum()
            )

        if total_transitions == 0:
            return ValidationResult(
                check_name="Journey 연속성",
                level=ValidationLevel.PASS,
                metric=1.0,
                threshold_pass=THRESHOLD_PASS,
                threshold_warning=THRESHOLD_WARNING,
                message="이동 기록 없음",
                details={"total_transitions": 0},
            )

        invalid_ratio = invalid_count / total_transitions

    else:
        return ValidationResult(
            check_name="Journey 연속성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="검증 불가 — is_valid_transition 또는 SpatialGraph 필요",
            details={"reason": "missing_dependency"},
        )

    # 연속성 = 1 - 비정상 비율
    continuity = 1.0 - invalid_ratio

    # 등급 판정
    if invalid_ratio < THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"Journey 연속성 양호 (비정상 {invalid_ratio * 100:.1f}%)"
    elif invalid_ratio < THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"Journey 연속성 주의 (비정상 {invalid_ratio * 100:.1f}%) — BLE 음영 또는 보정 필요"
    else:
        level = ValidationLevel.FAIL
        message = f"Journey 연속성 부족 (비정상 {invalid_ratio * 100:.1f}%) — 데이터 품질 점검 필요"

    return ValidationResult(
        check_name="Journey 연속성",
        level=level,
        metric=continuity,
        threshold_pass=1.0 - THRESHOLD_PASS,
        threshold_warning=1.0 - THRESHOLD_WARNING,
        message=message,
        details={
            "total_transitions": int(total_transitions),
            "invalid_transitions": int(invalid_count),
            "invalid_ratio": round(invalid_ratio, 4),
        },
    )


# ─── 출퇴근 일관성 검증 ───────────────────────────────────────────────────


def validate_work_hours_consistency(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
) -> ValidationResult:
    """
    출퇴근 기록 일관성 검증.

    계산: AccessLog 첫/마지막 기록 시간 vs TWardData in/out_datetime 오차
    임계값:
      - PASS: 평균 오차 < 10분
      - WARNING: 10-30분
      - FAIL: >= 30분
    """
    THRESHOLD_PASS = 10.0  # 분
    THRESHOLD_WARNING = 30.0

    if journey_df.empty or worker_df.empty:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="데이터 없음",
            details={"reason": "empty_data"},
        )

    # 필요한 컬럼 확인
    if "in_datetime" not in worker_df.columns or "timestamp" not in journey_df.columns:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="검증 불가 — in_datetime 또는 timestamp 없음",
            details={"reason": "missing_column"},
        )

    # AccessLog 기준 첫/마지막 기록 시간
    access_times = journey_df.groupby("user_no")["timestamp"].agg(["min", "max"])
    access_times.columns = ["access_first", "access_last"]

    # 병합
    merged = worker_df.merge(access_times, left_on="user_no", right_index=True, how="inner")

    if merged.empty:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="매칭된 작업자 없음",
            details={"reason": "no_match"},
        )

    # 시간 오차 계산 (분)
    errors = []

    for _, row in merged.iterrows():
        in_dt = row.get("in_datetime")
        access_first = row.get("access_first")

        if pd.notna(in_dt) and pd.notna(access_first):
            try:
                # datetime 변환
                if isinstance(in_dt, str):
                    in_dt = pd.to_datetime(in_dt)
                if isinstance(access_first, str):
                    access_first = pd.to_datetime(access_first)

                error_min = abs((access_first - in_dt).total_seconds() / 60)
                errors.append(error_min)
            except Exception:
                continue

    if not errors:
        return ValidationResult(
            check_name="출퇴근 일관성",
            level=ValidationLevel.WARNING,
            metric=0.0,
            threshold_pass=THRESHOLD_PASS,
            threshold_warning=THRESHOLD_WARNING,
            message="유효한 비교 데이터 없음",
            details={"reason": "no_valid_comparison"},
        )

    avg_error = np.mean(errors)
    median_error = np.median(errors)

    # 등급 판정
    if avg_error < THRESHOLD_PASS:
        level = ValidationLevel.PASS
        message = f"출퇴근 기록 일관성 양호 (평균 오차 {avg_error:.1f}분)"
    elif avg_error < THRESHOLD_WARNING:
        level = ValidationLevel.WARNING
        message = f"출퇴근 기록 일관성 주의 (평균 오차 {avg_error:.1f}분)"
    else:
        level = ValidationLevel.FAIL
        message = f"출퇴근 기록 불일치 (평균 오차 {avg_error:.1f}분) — 데이터 점검 필요"

    return ValidationResult(
        check_name="출퇴근 일관성",
        level=level,
        metric=avg_error,
        threshold_pass=THRESHOLD_PASS,
        threshold_warning=THRESHOLD_WARNING,
        message=message,
        details={
            "avg_error_min": round(avg_error, 2),
            "median_error_min": round(median_error, 2),
            "sample_count": len(errors),
        },
    )


# ─── 통합 검증 ────────────────────────────────────────────────────────────


def run_all_validations(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    access_df: pd.DataFrame | None = None,
    spatial_graph: "SpatialGraph | None" = None,
) -> list[ValidationResult]:
    """
    모든 검증 실행.

    Args:
        journey_df: Journey 데이터
        worker_df: 작업자 데이터 (EWI/CRE 포함)
        access_df: AccessLog 원본 (선택)
        spatial_graph: SpatialGraph (선택)

    Returns:
        ValidationResult 리스트
    """
    results = [
        validate_ble_coverage(journey_df, worker_df),
        validate_journey_continuity(journey_df, spatial_graph),
        validate_work_hours_consistency(journey_df, worker_df),
    ]

    return results


def generate_quality_report(
    results: list[ValidationResult],
    dsl_checks: list[CheckResult] | None = None,
) -> dict:
    """
    일별 품질 리포트 생성.

    Args:
        results: 레거시 ValidationResult 리스트
        dsl_checks: 선언적 Check 실행 결과 (선택) — 있으면 리포트에 추가 기록

    Returns:
        {
            "overall_score": float (0~1),
            "overall_level": str (pass/warning/fail),
            "checks": [ValidationResult.to_dict(), ...],   # 레거시
            "dsl_checks": [CheckResult.to_dict(), ...],    # 신규 (있을 때만)
            "summary": str,
        }
    """
    if not results:
        return {
            "overall_score": 0.0,
            "overall_level": "fail",
            "checks": [],
            "summary": "검증 결과 없음",
        }

    # 점수 계산 (PASS=1.0, WARNING=0.5, FAIL=0.0)
    level_scores = {
        ValidationLevel.PASS: 1.0,
        ValidationLevel.WARNING: 0.5,
        ValidationLevel.FAIL: 0.0,
    }

    scores = [level_scores[r.level] for r in results]
    overall_score = float(np.mean(scores))

    # 전체 등급
    if overall_score >= 0.8:
        overall_level = "pass"
    elif overall_score >= 0.5:
        overall_level = "warning"
    else:
        overall_level = "fail"

    # 요약 생성
    pass_count = sum(1 for r in results if r.level == ValidationLevel.PASS)
    warn_count = sum(1 for r in results if r.level == ValidationLevel.WARNING)
    fail_count = sum(1 for r in results if r.level == ValidationLevel.FAIL)

    summary_parts = []
    if pass_count > 0:
        summary_parts.append(f"통과 {pass_count}개")
    if warn_count > 0:
        summary_parts.append(f"주의 {warn_count}개")
    if fail_count > 0:
        summary_parts.append(f"실패 {fail_count}개")

    summary = " / ".join(summary_parts) if summary_parts else "검증 항목 없음"

    report: dict = {
        "overall_score": round(overall_score, 2),
        "overall_level": overall_level,
        "checks": [r.to_dict() for r in results],
        "summary": summary,
    }

    # DSL check 결과 병합 (있을 때만)
    if dsl_checks:
        report["dsl_checks"] = [r.to_dict() for r in dsl_checks]
        # error severity 위반이 있으면 전체 등급 강등
        dsl_errors = [r for r in dsl_checks if r.severity == "error" and not r.passed]
        if dsl_errors:
            report["overall_level"] = "fail" if overall_level == "pass" else overall_level
            report["dsl_errors"] = len(dsl_errors)

    return report


# ═════════════════════════════════════════════════════════════════════
# 상위 진입점 — 파이프라인에서 단일 호출
# ═════════════════════════════════════════════════════════════════════

def run_validation_suite(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    access_df: pd.DataFrame | None = None,
    spatial_graph: "SpatialGraph | None" = None,
    extra_checks: list[Check] | None = None,
) -> dict:
    """
    레거시 3개 validator + DSL DEFAULT_CHECKS + extra_checks 를 모두 실행.

    파이프라인 processor.py에서 이 함수 하나만 호출하면 된다.

    Returns:
        generate_quality_report() 형식의 dict (dsl_checks 포함)
    """
    legacy_results = run_all_validations(journey_df, worker_df, access_df, spatial_graph)

    ctx = {
        "journey": journey_df if journey_df is not None else pd.DataFrame(),
        "worker":  worker_df  if worker_df  is not None else pd.DataFrame(),
        "access":  access_df  if access_df  is not None else pd.DataFrame(),
    }
    all_checks = list(DEFAULT_CHECKS)
    if extra_checks:
        all_checks.extend(extra_checks)
    dsl_results = run_checks(all_checks, ctx)

    return generate_quality_report(legacy_results, dsl_checks=dsl_results)
