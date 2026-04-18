"""
Data Packager (Y1 전용) — 탭별 LLM 전송용 데이터 패키징
=========================================================
각 탭의 원시 DataFrame을 받아 LLM이 유의미한 인사이트를 생성할 수 있는
구조화된 텍스트 블록으로 변환한다.

Y1 특성:
  - K_ANON_MIN = 20 (M15X의 10에서 상향)
  - HIGH_ACTIVE_THRESHOLD = 0.90, LOW_ACTIVE_THRESHOLD = 0.40
  - 3교대 (day/night/unknown), 213 locus (GW-XXX 코드)
  - ~9,000명/일, 208개 협력업체

메서드:
    DataPackager.pack_daily(worker_df, journey_df, date_str, shift_filter)
    DataPackager.pack_congestion(space_df, journey_df, date_str)
    DataPackager.pack_weekly(worker_df_list, dates)
    DataPackager.pack_deep_space(prediction_df, accuracy_metrics)

사용:
    from src.intelligence.data_packager import DataPackager

    text = DataPackager.pack_daily(worker_df, journey_df, date_str="20260409")
    result = LLMGateway.analyze(tab_id="daily", packed_text=text, ...)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.intelligence.anonymization_pipeline import (
    AnonymizationPipeline as AP,
    K_ANON_MIN,
)

logger = logging.getLogger(__name__)

# ─── 임계값 상수 (constants.py를 재노출 — 단일 진실 공급원) ─────────────
from src.pipeline.metrics import HIGH_ACTIVE_THRESHOLD, LOW_ACTIVE_THRESHOLD
from src.metrics.constants import CRE_HIGH, EWI_HIGH

# EWI 중위 등급 (data_packager 전용, 별도 개념)
EWI_MID   = 0.4

# 교대 유형 레이블
SHIFT_LABEL: dict[str, str] = {
    "day":     "주간",
    "night":   "야간",
    "unknown": "미분류",
}

# BLE 커버리지 레이블 (worker.parquet ble_coverage 컬럼 값)
BLE_COVERAGE_LABELS = ["정상", "부분음영", "음영", "미측정"]


# ─── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _dist(series: pd.Series, label: str, unit: str = "") -> str:
    """분포 통계 1줄 요약 (평균/중앙값/std/P90)."""
    s = series.dropna()
    if s.empty:
        return f"- {label}: 데이터 없음"
    p90 = s.quantile(0.90)
    return (
        f"- {label}: 평균 {s.mean():.3f}{unit}, "
        f"중앙값 {s.median():.3f}{unit}, "
        f"표준편차 {s.std():.3f}{unit}, "
        f"P90 {p90:.3f}{unit}"
    )


def _pct(value: float | None, total: float | None) -> str:
    """비율 문자열 (분모 0 방어)."""
    if value is None or total is None or total == 0:
        return "N/A"
    return f"{value / total * 100:.1f}%"


def _company_rows(
    df: pd.DataFrame,
    company_col: str = "company_name",
    count_col: str = "worker_count",
    metric_cols: list[str] | None = None,
    top_n: int = 5,
    sort_col: str | None = None,
) -> list[str]:
    """업체별 통계 행 생성. k-Anonymity 미달(K=20) 항목 억제."""
    if df is None or df.empty or company_col not in df.columns:
        return []

    sort_col = sort_col or count_col
    if sort_col in df.columns:
        df = df.nlargest(top_n, sort_col)

    lines: list[str] = []
    for _, r in df.iterrows():
        cnt = r.get(count_col, 0)
        if not isinstance(cnt, (int, float)) or cnt < K_ANON_MIN:
            continue

        company_name = str(r.get(company_col, ""))
        code = AP.get_company_code(company_name) if company_name else "?"

        parts = [f"작업자 {cnt:.0f}명"]
        for col in (metric_cols or []):
            val = r.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                col_label = col.replace("avg_", "").replace("_", " ").upper()
                parts.append(
                    f"{col_label} {val:.3f}" if isinstance(val, float)
                    else f"{col_label} {val}"
                )
        lines.append(f"  - {code}: {', '.join(parts)}")

    return lines


# ─── 공개 패키저 ─────────────────────────────────────────────────────────────

class DataPackager:
    """
    DeepCon Y1 전용 LLM 전송용 데이터 텍스트 생성기.

    Y1 특성: HIGH_ACTIVE=0.90, LOW_ACTIVE=0.40, 3교대, 213 locus
    K_ANON_MIN = 20 (M15X의 10에서 상향, Y1 협력업체 다수 소규모 하도급 보호)

    모든 메서드는 @staticmethod — 인스턴스화 불필요.
    """

    # ─── 1. 일별 작업자 분석 (Daily) ─────────────────────────────────

    @staticmethod
    def pack_daily(
        worker_df: pd.DataFrame,
        journey_df: pd.DataFrame | None,
        date_str: str,
        shift_filter: str | None = None,
    ) -> str:
        """
        일별 작업자 분석 탭용 구조화 텍스트.

        포함:
          - EWI/CRE/SII 분포 (평균/중앙값/std/P90, 등급별 인원)
          - 3교대별 EWI 비교 (day/night/unknown)
          - 고활성/저활성/대기 시간 분포 (HIGH_ACTIVE=0.90, LOW_ACTIVE=0.40)
          - BLE 커버리지 분포 (정상/부분음영/음영/미측정)
          - 상위 5개 업체 EWI/CRE 평균 (k-Anonymity=20 적용)
          - 총 작업자수, 근무시간 분포
        """
        if worker_df is None or worker_df.empty:
            return "일별 작업자 데이터가 없습니다."

        df = worker_df.copy()
        if shift_filter and "shift_type" in df.columns:
            shift_en = {"주간": "day", "야간": "night", "미분류": "unknown"}.get(
                shift_filter, shift_filter
            )
            df = df[df["shift_type"] == shift_en]

        n_workers = len(df)
        lines: list[str] = [f"## 일별 작업자 분석 데이터 ({date_str})"]
        if shift_filter:
            lines.append(f"- 교대 필터: {shift_filter}")
        lines.append(f"- 총 작업자: {n_workers}명")

        # 근무시간
        if "work_minutes" in df.columns:
            wm = df["work_minutes"].dropna()
            if not wm.empty:
                lines.append(
                    f"- 평균 근무시간: {wm.mean():.0f}분({wm.mean()/60:.1f}h), "
                    f"최대 {wm.max():.0f}분, 중앙값 {wm.median():.0f}분"
                )

        # EWI 분포
        lines.append("\n### EWI (유효작업집중도) 분포")
        if "ewi" in df.columns:
            lines.append(_dist(df["ewi"], "EWI"))
            ewi = df["ewi"].dropna()
            if not ewi.empty:
                high = (ewi >= EWI_HIGH).sum()
                mid  = ((ewi >= EWI_MID) & (ewi < EWI_HIGH)).sum()
                low  = (ewi < EWI_MID).sum()
                lines.append(
                    f"  → 상위(≥{EWI_HIGH}): {high}명({_pct(high, n_workers)}), "
                    f"중간({EWI_MID}~{EWI_HIGH}): {mid}명({_pct(mid, n_workers)}), "
                    f"하위(<{EWI_MID}): {low}명({_pct(low, n_workers)})"
                )
        else:
            lines.append("- EWI 데이터 없음")

        # CRE 분포
        lines.append("\n### CRE (건설위험노출도) 분포")
        if "cre" in df.columns:
            lines.append(_dist(df["cre"], "CRE"))
            cre = df["cre"].dropna()
            if not cre.empty:
                high_cre = (cre >= CRE_HIGH).sum()
                lines.append(
                    f"  → 고위험(CRE≥{CRE_HIGH}): {high_cre}명({_pct(high_cre, n_workers)}) "
                    f"— 안전 관리 주의 필요"
                )
        else:
            lines.append("- CRE 데이터 없음")

        # SII 분포
        if "sii" in df.columns:
            sii = df["sii"].dropna()
            if not sii.empty:
                lines.append("\n### SII (공간위험-작업강도 지수)")
                lines.append(_dist(df["sii"], "SII"))

        # 3교대별 EWI 비교
        if "shift_type" in df.columns and "ewi" in df.columns and not shift_filter:
            lines.append("\n### 교대별 EWI 비교")
            for shift_en, shift_kr in SHIFT_LABEL.items():
                grp = df[df["shift_type"] == shift_en]
                if grp.empty:
                    continue
                ewi_grp = grp["ewi"].dropna()
                if ewi_grp.empty:
                    continue
                lines.append(
                    f"- {shift_kr}({len(grp)}명): EWI 평균 {ewi_grp.mean():.3f}, "
                    f"중앙값 {ewi_grp.median():.3f}"
                )

        # 활성레벨 시간 분포 (HIGH=0.90, LOW=0.40 기준)
        lines.append("\n### 활성레벨 시간 분포 (HIGH≥0.90, LOW<0.40)")
        activity_cols = {
            "high_active_min": f"고활성(active_ratio≥{HIGH_ACTIVE_THRESHOLD})",
            "low_active_min":  f"저활성({LOW_ACTIVE_THRESHOLD}~{HIGH_ACTIVE_THRESHOLD})",
            "standby_min":     f"대기(<{LOW_ACTIVE_THRESHOLD})",
            "rest_min":        "휴식",
            "transit_min":     "이동",
            "gap_min":         "BLE음영",
        }
        activity_parts: list[str] = []
        for col, label in activity_cols.items():
            if col in df.columns:
                avg = df[col].dropna().mean()
                if not np.isnan(avg) and avg > 0:
                    activity_parts.append(f"{label} {avg:.0f}분")
        if activity_parts:
            lines.append(f"- 1인당 평균: {', '.join(activity_parts)}")

        # GAP 비율 (음영 비율)
        if "gap_ratio" in df.columns:
            avg_gap = df["gap_ratio"].dropna().mean()
            if not np.isnan(avg_gap):
                lines.append(
                    f"- 평균 BLE음영(GAP) 비율: {avg_gap:.1%} "
                    f"(건설현장 BLE 특성상 30~50% 정상)"
                )

        # BLE 커버리지 분포
        if "ble_coverage" in df.columns:
            lines.append("\n### BLE 커버리지 분포")
            cov_dist = df["ble_coverage"].value_counts(normalize=True) * 100
            for label in BLE_COVERAGE_LABELS:
                pct = cov_dist.get(label, 0)
                cnt = (df["ble_coverage"] == label).sum()
                lines.append(f"- {label}: {cnt}명({pct:.1f}%)")

        # 상위 5개 업체 (k-Anonymity=20)
        if "company_name" in df.columns:
            company_agg = df.groupby("company_name").agg(
                worker_count=("user_no" if "user_no" in df.columns else "ewi", "count"),
                avg_ewi=("ewi", "mean") if "ewi" in df.columns else ("company_name", "count"),
                avg_cre=("cre", "mean") if "cre" in df.columns else ("company_name", "count"),
            ).reset_index()

            eligible = company_agg[company_agg["worker_count"] >= K_ANON_MIN]
            if not eligible.empty:
                overall_ewi = df["ewi"].dropna().mean() if "ewi" in df.columns else None
                ewi_str = f", 전체 평균 EWI {overall_ewi:.3f}" if overall_ewi and not np.isnan(overall_ewi) else ""
                lines.append(f"\n### 업체별 현황 ({K_ANON_MIN}명 이상 업체{ewi_str})")
                rows = _company_rows(
                    eligible,
                    company_col="company_name",
                    count_col="worker_count",
                    metric_cols=["avg_ewi", "avg_cre"],
                    top_n=5,
                    sort_col="worker_count",
                )
                lines.extend(rows if rows else ["  - 해당 없음"])

        return "\n".join(lines)

    # ─── 2. 공간 혼잡도 (Congestion) ─────────────────────────────────

    @staticmethod
    def pack_congestion(
        space_df: pd.DataFrame,
        journey_df: pd.DataFrame | None,
        date_str: str,
    ) -> str:
        """
        공간 혼잡도 탭용 구조화 텍스트.

        포함:
          - 구역 그룹별 (FAB/CUB/WWT/본진) 평균 체류 인원
          - 상위 5개 혼잡 구역 (locus_id, 최대/평균 인원)
          - journey에서 시간대별 피크 혼잡 구역 Top5 (있을 경우)
          - 밀폐공간/고압전 구역 현황
        """
        if space_df is None or space_df.empty:
            return "공간 혼잡도 데이터가 없습니다."

        lines: list[str] = [f"## 공간 혼잡도 데이터 ({date_str})"]
        n_spaces = len(space_df)
        lines.append(f"- 분석 공간 수: {n_spaces}개")

        # 전체 체류 인원 분포 (space_df의 unique_workers 기준)
        if "unique_workers" in space_df.columns:
            uw = space_df["unique_workers"].dropna()
            if not uw.empty:
                lines.append(
                    f"- 공간별 평균 체류 작업자: {uw.mean():.1f}명, "
                    f"최대 {uw.max():.0f}명"
                )

        # 밀폐공간/고압전 구역
        if "is_confined" in space_df.columns:
            n_confined = space_df["is_confined"].sum()
            lines.append(f"- 밀폐공간 구역 수: {n_confined}개")

        if "is_high_voltage" in space_df.columns:
            n_hv = space_df["is_high_voltage"].sum()
            lines.append(f"- 고압전 구역 수: {n_hv}개")

        # 구역 그룹별 집계 (locus_name 컬럼이 있을 경우 키워드 기반 분류)
        # locus_name이 없거나 GW-XXX 코드만 있으면 전체 분포 통계로 대체
        if "unique_workers" in space_df.columns:
            name_col = next(
                (c for c in ["locus_name", "locus_token"] if c in space_df.columns),
                None,
            )
            has_keyword_tokens = (
                name_col is not None
                and not space_df[name_col].dropna().str.match(r"^GW-\d+$").all()
            )
            if has_keyword_tokens and name_col:
                lines.append("\n### 구역 그룹별 평균 체류 인원")
                group_map: dict[str, list[str]] = {
                    "FAB 구역":  ["fab"],
                    "CUB 구역":  ["cub"],
                    "WWT 구역":  ["wwt"],
                    "현장 지원": ["본진", "support", "breakroom", "smoking", "restroom"],
                    "게이트":    ["gate", "timeclock"],
                    "이동 구역": ["transit", "hoist"],
                    "야외 작업": ["outdoor", "crane"],
                    "전력 설비": ["154kv", "power", "ut_"],
                }
                for group_label, keywords in group_map.items():
                    mask = space_df[name_col].str.lower().apply(
                        lambda t: any(k in str(t) for k in keywords)
                    )
                    sub = space_df[mask]
                    if not sub.empty:
                        avg_w   = sub["unique_workers"].mean()
                        total_w = sub["unique_workers"].sum()
                        lines.append(
                            f"- {group_label} ({len(sub)}개 구역): "
                            f"평균 {avg_w:.1f}명/구역, 합계 {total_w:.0f}명"
                        )
            else:
                # GW-XXX 코드만 있을 경우 분위수 분포 제공
                lines.append("\n### 구역별 체류 인원 분포 (213개 GW 구역)")
                uw = space_df["unique_workers"].dropna()
                if not uw.empty:
                    lines.append(
                        f"- 평균 {uw.mean():.0f}명, 중앙값 {uw.median():.0f}명, "
                        f"P75 {uw.quantile(0.75):.0f}명, 최대 {uw.max():.0f}명"
                    )
                    top_n_loci = space_df.nlargest(5, "unique_workers")["locus_id"].tolist()
                    lines.append(
                        f"- 상위 5개 구역 (locus_id): {', '.join(str(l) for l in top_n_loci)}"
                    )

        # 혼잡 상위 5개 구역 (locus_id로 표시, 실명 노출 안 함)
        if "unique_workers" in space_df.columns:
            top5 = space_df.nlargest(5, "unique_workers")
            lines.append("\n### 혼잡도 상위 5개 구역")
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                locus_id   = row.get("locus_id", f"구역{rank}")
                token      = row.get("locus_token", "")
                n_workers  = row.get("unique_workers", 0)
                n_minutes  = row.get("total_person_minutes", 0)
                avg_ratio  = row.get("avg_active_ratio", None)
                is_conf    = row.get("is_confined", False)
                is_hv      = row.get("is_high_voltage", False)
                flags      = []
                if is_conf:
                    flags.append("밀폐")
                if is_hv:
                    flags.append("고압전")
                flag_str   = f" [{'/'.join(flags)}]" if flags else ""
                ratio_str  = f", 활성비율 {avg_ratio:.2f}" if avg_ratio is not None else ""
                lines.append(
                    f"  - {rank}위 {locus_id}({token}){flag_str}: "
                    f"체류 작업자 {n_workers}명, "
                    f"누적 {n_minutes}인·분{ratio_str}"
                )

        # Journey 기반 시간대별 피크 (있을 경우)
        if journey_df is not None and not journey_df.empty:
            lines.append("\n### 시간대별 혼잡도 (journey 기반)")
            try:
                jdf = journey_df.copy()
                if "timestamp" in jdf.columns:
                    jdf["hour"] = pd.to_datetime(jdf["timestamp"]).dt.hour
                    hourly = (
                        jdf.groupby(["hour", "locus_id"])["user_no"]
                        .nunique()
                        .reset_index(name="n_workers")
                    )
                    # 시간대별 최고 혼잡 구역
                    peak_per_hour = hourly.loc[
                        hourly.groupby("hour")["n_workers"].idxmax()
                    ]
                    peak_hours = peak_per_hour.nlargest(5, "n_workers")
                    for _, r in peak_hours.iterrows():
                        lines.append(
                            f"  - {int(r['hour'])}시: 최대 {r['n_workers']}명 "
                            f"({r['locus_id']})"
                        )
            except Exception as e:
                logger.debug("journey hourly peak failed: %s", e)

        return "\n".join(lines)

    # ─── 3. 주간 트렌드 (Weekly) ─────────────────────────────────────

    @staticmethod
    def pack_weekly(
        worker_df_list: list[pd.DataFrame],
        dates: list[str],
    ) -> str:
        """
        주간 트렌드 탭용 구조화 텍스트.

        포함:
          - 요일별 EWI 평균 트렌드 (날짜 상대화)
          - 주간 총 작업자 추이
          - 업체별 주간 EWI 변화율 (Top5, k-Anonymity=20)
          - CRE 급등일 감지
        """
        if not worker_df_list or not dates:
            return "주간 데이터가 없습니다."

        if len(worker_df_list) != len(dates):
            logger.warning(
                "worker_df_list(%d) != dates(%d), truncating",
                len(worker_df_list), len(dates)
            )
            n = min(len(worker_df_list), len(dates))
            worker_df_list = worker_df_list[:n]
            dates          = dates[:n]

        lines: list[str] = [f"## 주간 트렌드 분석 ({dates[0]}~{dates[-1]})"]
        lines.append(f"- 분석 일수: {len(dates)}일")

        # 일별 KPI 집계
        from datetime import datetime as _dt
        _DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]

        lines.append("\n### 일별 EWI/CRE/작업자 추이")
        lines.append("| 일차 | 요일 | 작업자 | EWI평균 | CRE평균 | 고위험(CRE≥0.5) |")
        lines.append("|------|------|--------|---------|---------|----------------|")

        ewi_by_date: list[float] = []
        cre_by_date: list[float] = []
        n_by_date:   list[int]   = []

        for idx, (df, d) in enumerate(zip(worker_df_list, dates), 1):
            if df is None or df.empty:
                lines.append(f"| {idx} | - | 0 | - | - | - |")
                continue
            try:
                dow = _dt.strptime(d, "%Y%m%d").weekday()
                dow_str = _DOW_KR[dow]
            except Exception:
                dow_str = "-"

            n_w      = len(df)
            ewi_avg  = df["ewi"].dropna().mean() if "ewi" in df.columns else float("nan")
            cre_avg  = df["cre"].dropna().mean() if "cre" in df.columns else float("nan")
            high_cre = (df["cre"] >= CRE_HIGH).sum() if "cre" in df.columns else 0

            ewi_s = f"{ewi_avg:.3f}" if not np.isnan(ewi_avg) else "-"
            cre_s = f"{cre_avg:.3f}" if not np.isnan(cre_avg) else "-"

            lines.append(
                f"| {idx}({d}) | {dow_str} | {n_w} | {ewi_s} | {cre_s} | {high_cre}명 |"
            )

            if not np.isnan(ewi_avg):
                ewi_by_date.append(ewi_avg)
            if not np.isnan(cre_avg):
                cre_by_date.append(cre_avg)
            n_by_date.append(n_w)

        # 주간 요약 통계
        if ewi_by_date:
            ewi_s_arr = pd.Series(ewi_by_date)
            lines.append(f"\n### 주간 EWI 요약")
            lines.append(
                f"- 범위: {ewi_s_arr.min():.3f} ~ {ewi_s_arr.max():.3f}, "
                f"평균 {ewi_s_arr.mean():.3f}, 표준편차 {ewi_s_arr.std():.3f}"
            )
            if len(ewi_by_date) >= 2:
                trend = ewi_by_date[-1] - ewi_by_date[0]
                direction = "상승" if trend > 0.01 else ("하락" if trend < -0.01 else "유지")
                lines.append(
                    f"- 추세: {direction} ({ewi_by_date[0]:.3f} → {ewi_by_date[-1]:.3f})"
                )

        # CRE 급등일 감지
        if cre_by_date and len(cre_by_date) >= 3:
            cre_arr = pd.Series(cre_by_date)
            cre_mean = cre_arr.mean()
            cre_std  = cre_arr.std()
            spikes   = [(dates[i], v) for i, v in enumerate(cre_by_date)
                        if v > cre_mean + 1.5 * cre_std]
            if spikes:
                lines.append("\n### CRE 급등일 (평균+1.5σ 초과)")
                for d_spike, v_spike in spikes:
                    lines.append(
                        f"  - {d_spike}: CRE {v_spike:.3f} "
                        f"(기준 {cre_mean:.3f}±{cre_std:.3f})"
                    )

        # 작업자 추이
        if n_by_date:
            n_arr = pd.Series(n_by_date)
            lines.append(f"\n### 주간 작업자 추이")
            lines.append(
                f"- 범위: {n_arr.min():.0f}~{n_arr.max():.0f}명, "
                f"평균 {n_arr.mean():.0f}명"
            )

        # 업체별 주간 EWI 변화율 (k-Anonymity=20, 첫날/마지막날 비교)
        if len(worker_df_list) >= 2:
            df_first = worker_df_list[0]
            df_last  = worker_df_list[-1]
            if (
                df_first is not None and not df_first.empty
                and df_last is not None and not df_last.empty
                and "company_name" in df_first.columns
                and "ewi" in df_first.columns
            ):
                agg_first = df_first.groupby("company_name").agg(
                    n=("user_no" if "user_no" in df_first.columns else "ewi", "count"),
                    ewi_first=("ewi", "mean"),
                ).reset_index()
                agg_last = df_last.groupby("company_name").agg(
                    n=("user_no" if "user_no" in df_last.columns else "ewi", "count"),
                    ewi_last=("ewi", "mean"),
                ).reset_index()

                merged = agg_first.merge(agg_last, on="company_name", suffixes=("_f", "_l"))
                merged = merged[
                    (merged["n_f"] >= K_ANON_MIN) & (merged["n_l"] >= K_ANON_MIN)
                ]

                if not merged.empty:
                    merged["ewi_change"] = merged["ewi_last"] - merged["ewi_first"]
                    top_change = merged.reindex(
                        merged["ewi_change"].abs().nlargest(5).index
                    )
                    lines.append(
                        f"\n### 업체별 EWI 주간 변화 ({K_ANON_MIN}명 이상 업체, "
                        f"1일차({dates[0]}) vs 마지막({dates[-1]}))"
                    )
                    for _, r in top_change.iterrows():
                        code   = AP.get_company_code(str(r["company_name"]))
                        change = r["ewi_change"]
                        sign   = "▲" if change > 0 else "▼"
                        lines.append(
                            f"  - {code}: {r['ewi_first']:.3f} → {r['ewi_last']:.3f} "
                            f"({sign}{abs(change):.3f})"
                        )

        return "\n".join(lines)

    # ─── 4. Deep Space 예측 해석 (Deep Space) ────────────────────────

    @staticmethod
    def pack_deep_space(
        prediction_df: pd.DataFrame | None,
        accuracy_metrics: dict[str, Any] | None,
    ) -> str:
        """
        Deep Space Transformer 예측 결과 해석용 구조화 텍스트.

        포함:
          - Prediction Journal 최근 정확도 (Group Top-1/Top-3, Locus Top-1/Top-3)
          - 오늘 예측 vs 실제 비교 요약 (있을 경우)
          - 예측 오류 빈발 주의 구역 (있을 경우)

        Args:
            prediction_df:    오늘 예측 결과 DataFrame (컬럼: user_no, predicted, actual 등)
            accuracy_metrics: Prediction Journal accuracy_log 최근 항목
                              (group_top1, group_top3, top1_accuracy, top3_accuracy 등)
        """
        lines: list[str] = ["## Deep Space 예측 분석 데이터"]
        lines.append(
            "- 모델: Deep Space Transformer v2 (params=3.4M, "
            "Top-1 60.1%, Top-3 82.5%, Group Top-1 89.5%)"
        )

        # Prediction Journal 정확도
        if accuracy_metrics:
            lines.append("\n### Prediction Journal 정확도 (최근 평가)")
            group_top1  = accuracy_metrics.get("group_top1")
            group_top3  = accuracy_metrics.get("group_top3")
            locus_top1  = accuracy_metrics.get("top1_accuracy")
            locus_top3  = accuracy_metrics.get("top3_accuracy")
            n_eval      = accuracy_metrics.get("total_evaluated", 0)
            cong_mae    = accuracy_metrics.get("congestion_mae")

            if group_top1 is not None:
                lines.append(
                    f"- Group Top-1: {group_top1:.1%}, Group Top-3: "
                    f"{group_top3:.1%}" if group_top3 else f"- Group Top-1: {group_top1:.1%}"
                )
            if locus_top1 is not None:
                lines.append(
                    f"- Locus Top-1: {locus_top1:.1%}, Locus Top-3: "
                    f"{locus_top3:.1%}" if locus_top3 else f"- Locus Top-1: {locus_top1:.1%}"
                )
            if n_eval:
                lines.append(f"- 평가 작업자 수: {n_eval}명")
            if cong_mae is not None:
                lines.append(f"- 혼잡도 예측 MAE: {cong_mae:.1f}인·분")

        # 오늘 예측 결과 요약 (prediction_df)
        if prediction_df is not None and not prediction_df.empty:
            lines.append("\n### 오늘 예측 vs 실제 (샘플)")
            n_pred = len(prediction_df)
            lines.append(f"- 예측 작업자 수: {n_pred}명")

            # 예측 정확도 계산 (predicted/actual 컬럼 있을 경우)
            if "predicted" in prediction_df.columns and "actual" in prediction_df.columns:
                correct_top1 = (
                    prediction_df["predicted"] == prediction_df["actual"]
                ).sum()
                lines.append(
                    f"- 오늘 Top-1 일치율: {correct_top1}/{n_pred} "
                    f"({_pct(correct_top1, n_pred)})"
                )

            # 예측 오류 빈발 구역 (오류가 많은 locus)
            if "predicted" in prediction_df.columns and "actual" in prediction_df.columns:
                error_df = prediction_df[
                    prediction_df["predicted"] != prediction_df["actual"]
                ]
                if not error_df.empty and "actual" in error_df.columns:
                    error_loci = error_df["actual"].value_counts().head(3)
                    if not error_loci.empty:
                        lines.append("\n### 주의 구역 (예측 오류 빈발 locus Top3)")
                        for locus, cnt in error_loci.items():
                            lines.append(
                                f"  - {locus}: {cnt}건 오류 "
                                f"({_pct(cnt, n_pred)} 오류율)"
                            )

        if not accuracy_metrics and (prediction_df is None or prediction_df.empty):
            lines.append("\n- 예측 데이터 없음 (Prediction Journal 데이터 미확인)")

        return "\n".join(lines)
