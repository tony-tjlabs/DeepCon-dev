"""
이상 탐지 엔진 — Anomaly Detector
===================================
현재 날짜의 데이터를 baseline(과거 N일 평균)과 비교하여
통계적으로 유의미한 이상을 감지한다.

설계 원칙:
  - 정보의 나열이 아닌 "핵심만" 추출
  - 단일 날짜 데이터만으로도 동작 (baseline 없으면 절대 기준 사용)
  - baseline 있으면 상대 비교로 더 정확한 탐지
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.intelligence.models import Insight, Severity
from src.utils.anonymizer import mask_name
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH as _CRE_HIGH

logger = logging.getLogger(__name__)

# ─── 임계값 상수 (constants.py를 재노출) ──────────────────────────
CRE_CRITICAL        = _CRE_HIGH  # CRE 고위험 임계 (constants.CRE_HIGH=0.5)
CRE_WARNING          = 0.3        # CRE 주의 임계 (CRE_MEDIUM과 동일)
FATIGUE_HIGH         = 0.6        # 피로도 고위험 (생리학 기준, 별도 유지)
ALONE_RATIO_DANGER   = 0.5        # 단독작업 위험 비율
CONFINED_LONG_MIN    = 60     # 밀폐공간 장시간 (분)
SIGMA_THRESHOLD      = 2.0    # 이상 탐지 시그마
COMPANY_CHANGE_PCT   = 0.20   # 업체 CRE 변동 기준 (20%)


def detect_anomalies(
    results: dict,
    baseline: Optional[dict] = None,
) -> list[Insight]:
    """
    마스터 이상 탐지. 모든 서브 탐지기를 실행하여 인사이트 목록 반환.

    Args:
        results: load_daily_results() 반환값 (worker, space, company, journey, meta)
        baseline: 과거 N일 평균 통계 (None이면 절대 기준만 사용)

    Returns:
        list[Insight] — 심각도 순 정렬
    """
    worker_df  = results.get("worker", pd.DataFrame())
    space_df   = results.get("space", pd.DataFrame())
    company_df = results.get("company", pd.DataFrame())
    meta       = results.get("meta", {})

    insights = []

    if not worker_df.empty:
        insights.extend(_detect_high_risk_workers(worker_df, baseline))
        insights.extend(_detect_fatigue_clusters(worker_df))
        insights.extend(_detect_alone_risk(worker_df))
        insights.extend(_detect_long_confined(worker_df))

    if not space_df.empty:
        insights.extend(_detect_space_overcrowding(space_df, baseline))

    if not company_df.empty:
        insights.extend(_detect_company_risk_shift(company_df, baseline))

    # 피로+고위험 복합 패턴
    if not worker_df.empty:
        insights.extend(_detect_fatigue_risk_combo(worker_df))

    return sorted(insights, key=lambda i: -i.severity)


# ─── 개별 탐지기 ──────────────────────────────────────────────────

def _detect_high_risk_workers(
    worker_df: pd.DataFrame,
    baseline: Optional[dict],
) -> list[Insight]:
    """CRE ≥ 0.6 고위험 작업자 탐지. baseline 대비 급증 시 severity 상향."""
    if "cre" not in worker_df.columns:
        return []

    critical = worker_df[worker_df["cre"] >= CRE_CRITICAL]
    n_critical = len(critical)
    if n_critical == 0:
        return []

    # baseline 대비 비교
    severity = Severity.HIGH
    baseline_count = None
    if baseline and "avg_high_cre_count" in baseline:
        baseline_count = baseline["avg_high_cre_count"]
        if n_critical > baseline_count * 1.5:
            severity = Severity.CRITICAL

    top_workers = critical.nlargest(3, "cre")
    names = [f"{mask_name(r.get('user_name', '?'))}({r['cre']:.2f})" for _, r in top_workers.iterrows()]

    desc = f"CRE ≥ {CRE_CRITICAL} 작업자가 {n_critical}명 감지되었습니다."
    if baseline_count is not None:
        desc += f" 평소 평균({baseline_count:.0f}명) 대비 {'증가' if n_critical > baseline_count else '유사'}합니다."

    return [Insight(
        category="safety",
        severity=severity,
        title=f"고위험(CRE≥{CRE_CRITICAL}) 작업자 {n_critical}명",
        description=desc,
        evidence={"count": n_critical, "top_workers": names, "baseline": baseline_count},
        affected=critical["user_no"].tolist()[:10],
        recommendation="고위험 작업자의 작업 구역 확인 및 안전관리자 배치 강화",
        source="anomaly_detector",
    )]


def _detect_fatigue_clusters(worker_df: pd.DataFrame) -> list[Insight]:
    """피로도 ≥ 0.6 작업자 집단 탐지."""
    if "fatigue_score" not in worker_df.columns:
        return []

    fatigued = worker_df[worker_df["fatigue_score"] >= FATIGUE_HIGH]
    n = len(fatigued)
    if n < 3:  # 3명 미만은 개별 이슈
        return []

    # 같은 업체에 집중되어 있는지 확인
    company_counts = fatigued["company_name"].dropna().value_counts()
    if company_counts.empty:
        top_company, top_count = "미확인", 0
    else:
        top_company = company_counts.index[0]
        top_count = int(company_counts.iloc[0])

    severity = Severity.HIGH if n >= 10 else Severity.MEDIUM
    cluster_info = ""
    if top_count >= 3:
        cluster_info = f" 특히 {top_company}에서 {top_count}명이 집중됩니다."

    return [Insight(
        category="safety",
        severity=severity,
        title=f"고피로 작업자 {n}명 감지",
        description=f"피로도 ≥ {FATIGUE_HIGH} 작업자가 {n}명입니다.{cluster_info}"
                    f" 장시간 연속 고강도 활동으로 사고 위험이 높습니다.",
        evidence={"count": n, "top_company": top_company, "top_company_count": top_count},
        affected=fatigued["user_no"].tolist()[:10],
        recommendation=f"피로 누적 작업자 휴식 시간 확보. {top_company} 교대 스케줄 점검 권고",
        source="anomaly_detector",
    )]


def _detect_alone_risk(worker_df: pd.DataFrame) -> list[Insight]:
    """밀폐공간/고위험 구역 단독작업 탐지."""
    if "alone_ratio" not in worker_df.columns or "confined_minutes" not in worker_df.columns:
        return []

    # 밀폐공간 진입 + 높은 단독작업 비율
    at_risk = worker_df[
        (worker_df["confined_minutes"] > 0) &
        (worker_df["alone_ratio"] >= ALONE_RATIO_DANGER)
    ]
    n = len(at_risk)
    if n == 0:
        return []

    severity = Severity.CRITICAL if n >= 5 else Severity.HIGH

    return [Insight(
        category="compliance",
        severity=severity,
        title=f"밀폐공간 단독작업 위반 의심 {n}명",
        description=f"밀폐공간 진입 작업자 중 {n}명의 단독작업 비율이 {ALONE_RATIO_DANGER*100:.0f}% 이상입니다. "
                    f"밀폐공간 작업 시 2인 1조 원칙 위반 가능성이 있습니다.",
        evidence={"count": n, "threshold": ALONE_RATIO_DANGER},
        affected=at_risk["user_no"].tolist()[:10],
        recommendation="밀폐공간 감시자(2인 1조) 배치 현황 점검. CCTV 확인 권고",
        source="anomaly_detector",
    )]


def _detect_long_confined(worker_df: pd.DataFrame) -> list[Insight]:
    """밀폐공간 장시간 체류 탐지."""
    if "confined_minutes" not in worker_df.columns:
        return []

    long_stay = worker_df[worker_df["confined_minutes"] >= CONFINED_LONG_MIN]
    n = len(long_stay)
    if n == 0:
        return []

    max_worker = long_stay.nlargest(1, "confined_minutes").iloc[0]
    max_min = int(max_worker["confined_minutes"])
    max_name = mask_name(max_worker.get("user_name", "?"))

    return [Insight(
        category="safety",
        severity=Severity.HIGH if max_min >= 120 else Severity.MEDIUM,
        title=f"밀폐공간 장시간 체류 {n}명 (최대 {max_min}분)",
        description=f"밀폐공간에 {CONFINED_LONG_MIN}분 이상 체류한 작업자가 {n}명입니다. "
                    f"최대 체류자는 {max_name} ({max_min}분)입니다.",
        evidence={"count": n, "max_minutes": max_min, "max_worker": max_name},
        affected=long_stay["user_no"].tolist()[:10],
        recommendation="장시간 밀폐공간 작업자 환기 상태 및 산소 농도 점검",
        source="anomaly_detector",
    )]


def _detect_space_overcrowding(
    space_df: pd.DataFrame,
    baseline: Optional[dict],
) -> list[Insight]:
    """공간별 인원 과밀집 탐지 (baseline 대비)."""
    if "unique_workers" not in space_df.columns:
        return []

    if baseline and "space_worker_avg" in baseline:
        # baseline 대비 2시그마 초과 공간
        avg_map = baseline["space_worker_avg"]  # {locus_token: avg_workers}
        std_map = baseline.get("space_worker_std", {})

        overcrowded = []
        for _, row in space_df.iterrows():
            token = row.get("locus_token", "")
            workers = row["unique_workers"]
            avg = avg_map.get(token, workers)
            std = std_map.get(token, avg * 0.3)
            # std=0 보호: 변동 없는 공간은 최소 std로 대체
            # 최소 std=3.0 — 1~2명 변동으로 과잉 탐지 방지
            if std < 1e-6:
                std = max(avg * 0.1, 3.0)
            if (workers - avg) / std > SIGMA_THRESHOLD:
                overcrowded.append({
                    "token": token,
                    "workers": int(workers),
                    "avg": round(avg, 1),
                    "sigma": round((workers - avg) / std, 1),
                })

        if not overcrowded:
            return []

        top = sorted(overcrowded, key=lambda x: -x["sigma"])[:3]
        names = [f"{x['token']}({x['workers']}명, 평소 {x['avg']}명)" for x in top]

        return [Insight(
            category="space",
            severity=Severity.MEDIUM,
            title=f"공간 과밀집 {len(overcrowded)}곳 감지",
            description=f"평소 대비 작업자가 유의미하게 많은 구역이 있습니다: {', '.join(names)}",
            evidence={"overcrowded_spaces": top},
            affected=[x["token"] for x in top],
            recommendation="과밀집 구역 작업 분산 또는 시간대 조정 검토",
            source="anomaly_detector",
        )]

    return []


def _detect_company_risk_shift(
    company_df: pd.DataFrame,
    baseline: Optional[dict],
) -> list[Insight]:
    """업체별 CRE 급변 탐지."""
    if "avg_cre" not in company_df.columns or baseline is None:
        return []
    if "company_cre_avg" not in baseline:
        return []

    prev = baseline["company_cre_avg"]  # {company_name: avg_cre}
    changed = []

    for _, row in company_df.iterrows():
        name = row.get("company_name", "")
        cre = row["avg_cre"]
        prev_cre = prev.get(name)
        if prev_cre is None or prev_cre < 0.1:
            continue
        change = (cre - prev_cre) / prev_cre
        if change > COMPANY_CHANGE_PCT:
            changed.append({"company": name, "cre": round(cre, 3),
                            "prev": round(prev_cre, 3), "change_pct": round(change * 100, 1)})

    if not changed:
        return []

    top = sorted(changed, key=lambda x: -x["change_pct"])[:3]
    names = [f"{x['company']}(+{x['change_pct']}%)" for x in top]

    return [Insight(
        category="safety",
        severity=Severity.MEDIUM,
        title=f"업체 위험도 급등 {len(changed)}곳",
        description=f"평소 대비 CRE가 {COMPANY_CHANGE_PCT*100:.0f}% 이상 상승한 업체: {', '.join(names)}",
        evidence={"changed_companies": top},
        affected=[x["company"] for x in top],
        recommendation="해당 업체 안전교육 이수 현황 및 작업 환경 점검",
        source="anomaly_detector",
    )]


def _detect_fatigue_risk_combo(worker_df: pd.DataFrame) -> list[Insight]:
    """피로도 + CRE 복합 위험 (둘 다 높은 작업자)."""
    required = {"fatigue_score", "cre"}
    if not required.issubset(worker_df.columns):
        return []

    combo = worker_df[
        (worker_df["fatigue_score"] >= 0.5) &
        (worker_df["cre"] >= CRE_WARNING)
    ]
    n = len(combo)
    if n < 2:
        return []

    top = combo.nlargest(3, "cre")
    names = [f"{mask_name(r.get('user_name', '?'))}(피로{r['fatigue_score']:.2f}, CRE{r['cre']:.2f})"
             for _, r in top.iterrows()]

    return [Insight(
        category="safety",
        severity=Severity.CRITICAL if n >= 5 else Severity.HIGH,
        title=f"피로+고위험 복합 위험 작업자 {n}명",
        description=f"피로도와 위험노출도가 동시에 높은 작업자 {n}명입니다. "
                    f"피로 상태에서 고위험 환경에 노출되면 사고 확률이 크게 증가합니다.",
        evidence={"count": n, "top_workers": names},
        affected=combo["user_no"].tolist()[:10],
        recommendation="해당 작업자 즉시 휴식 배치 또는 저위험 구역으로 재배치 권고",
        source="anomaly_detector",
    )]


# ─── Baseline 계산 ─────────────────────────────────────────────────

def compute_baseline(
    processed_dates: list[str],
    current_date: str,
    sector_id: str | None = None,
    window: int = 7,
) -> dict:
    """
    과거 N일의 처리 결과에서 baseline 통계 산출.

    ★ Perf v2: space/company parquet만 읽음 (각 수KB).
    worker parquet(1.7MB)에서는 CRE 고위험 수만 필요하므로 컬럼 선택 로드.
    """
    from src.pipeline.cache_manager import _date_dir

    # 현재 날짜 제외, 최근 window일
    past = [d for d in processed_dates if d < current_date][-window:]
    if len(past) < 2:
        return {}

    baseline = {
        "days": len(past),
        "avg_high_cre_count": 0.0,
        "space_worker_avg": {},
        "space_worker_std": {},
        "company_cre_avg": {},
    }

    cre_counts = []
    space_workers: dict[str, list] = {}
    company_cres: dict[str, list] = {}

    for d in past:
        try:
            date_dir = _date_dir(d, sector_id)

            # ★ Perf: worker에서 cre 컬럼만 로드 (1.7MB → ~40KB)
            wp = date_dir / "worker.parquet"
            wdf = pd.read_parquet(wp, columns=["cre"]) if wp.exists() else pd.DataFrame()

            # space/company는 수KB이므로 전체 로드 OK
            sp = date_dir / "space.parquet"
            sdf = pd.read_parquet(sp) if sp.exists() else pd.DataFrame()
            cp = date_dir / "company.parquet"
            cdf = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()

            if "cre" in wdf.columns:
                cre_counts.append(len(wdf[wdf["cre"] >= CRE_CRITICAL]))

            if "unique_workers" in sdf.columns and "locus_token" in sdf.columns:
                for _, row in sdf.iterrows():
                    token = row.get("locus_token", "")
                    space_workers.setdefault(token, []).append(row["unique_workers"])

            if "avg_cre" in cdf.columns and "company_name" in cdf.columns:
                for _, row in cdf.iterrows():
                    name = row.get("company_name", "")
                    company_cres.setdefault(name, []).append(row["avg_cre"])

        except Exception as e:
            logger.warning(f"Baseline 계산 실패 ({d}): {e}")
            continue

    if cre_counts:
        baseline["avg_high_cre_count"] = float(np.mean(cre_counts))

    for token, vals in space_workers.items():
        avg_val = float(np.mean(vals))
        std_val = float(np.std(vals)) if len(vals) > 1 else avg_val * 0.3
        # std=0 보호 (모든 날짜에서 동일한 인원) — 최소 3.0
        if std_val < 1e-6:
            std_val = max(avg_val * 0.1, 3.0)
        baseline["space_worker_avg"][token] = avg_val
        baseline["space_worker_std"][token] = std_val

    for name, vals in company_cres.items():
        baseline["company_cre_avg"][name] = float(np.mean(vals))

    logger.info(f"Baseline 계산 완료: {len(past)}일 ({past[0]}~{past[-1]})")
    return baseline
