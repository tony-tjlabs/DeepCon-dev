"""
Journey Reconstruction Engine
==============================
AccessLog(ground truth) + T-Ward BLE -> 복원된 완전한 작업자 Journey

핵심 설계 원칙:
1. AccessLog = Ground Truth: 작업자 현장 체류 시간 범위의 절대 기준
2. BLE는 위치 정보 제공: 1분 단위, 음영(gap) 존재
3. 음영 보정: 앞뒤 위치 모두 아는 경우만 보정(귀속). 양끝 음영은 "미확인"
4. Coverage threshold: BLE 감지율이 기준 미달인 작업자는 집계 제외
5. 단일 모집단: threshold 통과 작업자만 모든 KPI에 동일하게 사용

Coverage 계산:
  coverage_ratio = BLE 감지 분 / AccessLog 체류 총 분
  예: AccessLog 600분 체류, BLE 300분 감지 -> 50%

음영 보정:
  - 내부 음영: [위치A ... (gap) ... 위치B] -> gap을 A->B 이동으로 귀속
  - 앞 음영:  [(gap) ... 위치B] -> "미확인(unknown_start)" 처리
  - 뒤 음영:  [위치A ... (gap)] -> "미확인(unknown_end)" 처리

Threshold 권고값 (데이터 분포 기반):
  - 전체 31일 평균 coverage: 63.5%, 중앙값: 74.0%
  - 30% 미만: 21.4% (BLE 거의 없음 - 신뢰 불가)
  - 30-50%: 14.2% (부분 데이터 - 이동 경로 왜곡 가능)
  - 50% 이상: 64.4% (분석 가능한 신뢰 범위)
  -> 권고 threshold: 30% (너무 엄격하면 데이터 손실 과다)
     MAT/EOD 분석만 할 경우: 50% 이상이 권장
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─── 기본 Threshold ────────────────────────────────────────────────────────
DEFAULT_COVERAGE_THRESHOLD = 0.30   # 30% = 최소 신뢰 기준
TRANSIT_COVERAGE_THRESHOLD = 0.50   # 50% = MAT/EOD 계산 권장 기준

# ─── 음영 분류 토큰 ───────────────────────────────────────────────────────
SHADOW_TRANSIT_TOKEN = "shadow_transit"   # 앞뒤 위치 모두 아는 음영 (이동 귀속)
SHADOW_UNKNOWN_START = "shadow_unknown"   # 앞 위치 모르는 음영 (시작 부분)
SHADOW_UNKNOWN_END   = "shadow_unknown"   # 뒤 위치 모르는 음영 (종료 부분)
SHADOW_UNKNOWN_TOKEN = "shadow_unknown"   # 일반 미확인 음영

# 음영 최소 gap 기준 (분): 이 이상이어야 음영으로 간주
MIN_SHADOW_GAP_MIN = 2


def calc_coverage_ratio(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    작업자별 BLE coverage ratio 계산.

    coverage_ratio = BLE 감지 분 / AccessLog 현장 체류 총 분

    Args:
        journey_df: 1분 단위 BLE 기록 (user_no, timestamp, in_datetime, out_datetime)
        worker_df: 작업자별 집계 (user_no, in_datetime, out_datetime, work_minutes)

    Returns:
        worker_df에 coverage_ratio, coverage_pct, coverage_label 컬럼 추가된 DataFrame
    """
    df = worker_df.copy()

    # in_datetime / out_datetime이 없으면 recorded_minutes 기반 fallback
    has_access = (
        "in_datetime" in df.columns
        and "out_datetime" in df.columns
        and "work_minutes" in df.columns
    )

    if has_access:
        # AccessLog 기반: work_minutes = (out - in) 분
        access_total = df["work_minutes"].fillna(0)

        # BLE 감지 분 = journey_df에서 작업자별 레코드 수
        ble_counts = (
            journey_df.groupby("user_no").size().reset_index(name="_ble_count")
        )
        df = df.merge(ble_counts, on="user_no", how="left")
        df["_ble_count"] = df["_ble_count"].fillna(0)

        # Coverage ratio (0-1)
        df["coverage_ratio"] = np.where(
            access_total > 0,
            (df["_ble_count"] / access_total).clip(0, 1),
            0.0,
        )
        df = df.drop(columns=["_ble_count"])
    else:
        # Fallback: recorded_minutes 기반 (AccessLog 없는 경우)
        logger.warning("calc_coverage_ratio: AccessLog 없음 — recorded_minutes 기반 추정")
        if "recorded_minutes" in df.columns:
            # 평균 근무시간을 기준으로 추정 (신뢰도 낮음)
            avg_work = df["recorded_minutes"].median() * 2  # 보수적 추정
            df["coverage_ratio"] = (
                df["recorded_minutes"] / avg_work.clip(min=1)
            ).clip(0, 1)
        else:
            df["coverage_ratio"] = 0.0

    # Coverage percentage (0-100)
    df["coverage_pct"] = (df["coverage_ratio"] * 100).round(1)

    # Coverage label (카테고리)
    df["coverage_label"] = pd.cut(
        df["coverage_ratio"],
        bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=["불충분(<30%)", "부분(30-50%)", "양호(50-70%)", "충분(>70%)"],
    ).astype(str)

    logger.info(
        "Coverage 계산 완료: n=%d, mean=%.1f%%, median=%.1f%%",
        len(df),
        df["coverage_pct"].mean(),
        df["coverage_pct"].median(),
    )
    return df


def _find_shadow_gaps(
    worker_journey: pd.DataFrame,
    in_time: pd.Timestamp,
    out_time: pd.Timestamp,
) -> list[dict]:
    """
    AccessLog 체류 기간 내 BLE 음영 구간 탐색.

    Args:
        worker_journey: 단일 작업자 1분 단위 BLE 기록 (timestamp, locus_token 등)
        in_time: AccessLog 입장 시각
        out_time: AccessLog 퇴장 시각

    Returns:
        음영 구간 리스트: [{
            "gap_start": Timestamp,  # 음영 시작 (직전 BLE + 1분)
            "gap_end": Timestamp,    # 음영 종료 (직후 BLE - 1분)
            "gap_min": float,        # 음영 길이 (분)
            "before_token": str,     # 음영 직전 위치 (없으면 None)
            "after_token": str,      # 음영 직후 위치 (없으면 None)
            "gap_type": str,         # "transit"|"unknown_start"|"unknown_end"|"internal"
        }]
    """
    if worker_journey.empty or pd.isna(in_time) or pd.isna(out_time):
        return []

    # AccessLog 범위 내 BLE 기록만
    j = worker_journey[
        (worker_journey["timestamp"] >= in_time)
        & (worker_journey["timestamp"] <= out_time)
    ].sort_values("timestamp").reset_index(drop=True)

    gaps = []

    # ── 앞 음영: AccessLog 시작 ~ 첫 BLE 기록 ──────────────────────
    if not j.empty and (j["timestamp"].iloc[0] - in_time).total_seconds() / 60 > MIN_SHADOW_GAP_MIN:
        gap_start = in_time
        gap_end = j["timestamp"].iloc[0]
        gap_min = (gap_end - gap_start).total_seconds() / 60
        gaps.append({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_min": round(gap_min, 1),
            "before_token": None,
            "after_token": j["locus_token"].iloc[0],
            "gap_type": "unknown_start",
        })

    # ── 내부 음영: BLE 기록들 사이의 연속성 단절 ──────────────────
    if len(j) >= 2:
        for i in range(len(j) - 1):
            ts_curr = j["timestamp"].iloc[i]
            ts_next = j["timestamp"].iloc[i + 1]
            gap_min = (ts_next - ts_curr).total_seconds() / 60

            if gap_min > MIN_SHADOW_GAP_MIN:
                before_tok = j["locus_token"].iloc[i]
                after_tok = j["locus_token"].iloc[i + 1]

                # 앞뒤 위치 모두 알면 "transit" (이동 귀속 가능)
                # 같은 위치면 "internal" (동일 장소 BLE 미감지)
                if before_tok == after_tok:
                    gtype = "internal"
                else:
                    gtype = "transit"

                gaps.append({
                    "gap_start": ts_curr + pd.Timedelta(minutes=1),
                    "gap_end": ts_next,
                    "gap_min": round(gap_min - 1, 1),  # 전후 1분은 BLE 기록 포함
                    "before_token": before_tok,
                    "after_token": after_tok,
                    "gap_type": gtype,
                })

    # ── 뒤 음영: 마지막 BLE 기록 ~ AccessLog 종료 ─────────────────
    if not j.empty and (out_time - j["timestamp"].iloc[-1]).total_seconds() / 60 > MIN_SHADOW_GAP_MIN:
        gap_start = j["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
        gap_end = out_time
        gap_min = (gap_end - gap_start).total_seconds() / 60
        gaps.append({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_min": round(gap_min, 1),
            "before_token": j["locus_token"].iloc[-1],
            "after_token": None,
            "gap_type": "unknown_end",
        })

    # ── AccessLog 전체가 BLE 없는 경우 ────────────────────────────
    if j.empty:
        total_min = (out_time - in_time).total_seconds() / 60
        gaps.append({
            "gap_start": in_time,
            "gap_end": out_time,
            "gap_min": round(total_min, 1),
            "before_token": None,
            "after_token": None,
            "gap_type": "unknown_start",  # 전체 미확인
        })

    return gaps


def reconstruct_worker_journey(
    worker_journey: pd.DataFrame,
    in_time: pd.Timestamp | None,
    out_time: pd.Timestamp | None,
    user_no: int,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """
    단일 작업자 Journey 복원 (음영 보정 + 귀속).

    보정 방식:
    - 내부 transit 음영: gap 분을 shadow_transit 레코드로 채움 (이동 시간으로 귀속)
    - 내부 internal 음영 (같은 장소): gap 분을 직전 locus로 귀속 (체류 연장)
    - 앞/뒤 음영: shadow_unknown 레코드로 채움 (미확인)

    Args:
        worker_journey: 단일 작업자 BLE 기록 (timestamp, locus_token, ...)
        in_time: AccessLog 입장 시각 (None이면 보정 불가)
        out_time: AccessLog 퇴장 시각 (None이면 보정 불가)
        user_no: 작업자 번호 (로그용)
        coverage_threshold: 이 값 미만이면 excluded=True

    Returns:
        (reconstructed_df, meta_dict)
        - reconstructed_df: 원본 + 보정 레코드 (is_shadow 컬럼 추가)
        - meta_dict: {
            "user_no": int,
            "coverage_ratio": float,
            "shadow_transit_min": float,    # 이동 귀속된 음영 시간
            "shadow_unknown_min": float,    # 미확인 음영 시간
            "n_gaps": int,
            "excluded": bool,
        }
    """
    meta = {
        "user_no": user_no,
        "coverage_ratio": 0.0,
        "shadow_transit_min": 0.0,
        "shadow_unknown_min": 0.0,
        "n_gaps": 0,
        "excluded": True,
    }

    # AccessLog 없으면 보정 불가
    if pd.isna(in_time) or pd.isna(out_time):
        meta["excluded"] = True
        meta["coverage_ratio"] = 0.0
        result = worker_journey.copy()
        result["is_shadow"] = False
        result["shadow_type"] = ""
        return result, meta

    # 체류 총 분
    total_min = (out_time - in_time).total_seconds() / 60
    if total_min <= 0:
        meta["excluded"] = True
        result = worker_journey.copy()
        result["is_shadow"] = False
        result["shadow_type"] = ""
        return result, meta

    # AccessLog 범위 내 BLE 기록 수
    j_in_range = worker_journey[
        (worker_journey["timestamp"] >= in_time)
        & (worker_journey["timestamp"] <= out_time)
    ]
    ble_count = len(j_in_range)
    coverage_ratio = min(ble_count / total_min, 1.0)
    meta["coverage_ratio"] = round(coverage_ratio, 4)

    # Threshold 미달이면 excluded=True이지만 보정은 계속 진행 (UI 투명성)
    meta["excluded"] = coverage_ratio < coverage_threshold

    # 음영 구간 탐색
    gaps = _find_shadow_gaps(worker_journey, in_time, out_time)
    meta["n_gaps"] = len(gaps)

    # 원본 레코드에 shadow 여부 표시
    result = worker_journey.copy()
    result["is_shadow"] = False
    result["shadow_type"] = ""

    shadow_records = []

    for gap in gaps:
        gap_min = gap["gap_min"]
        if gap_min <= 0:
            continue

        gtype = gap["gap_type"]
        before_tok = gap["before_token"]
        after_tok = gap["after_token"]

        if gtype == "transit":
            # 이동 귀속: shadow_transit 레코드 생성
            shadow_token = SHADOW_TRANSIT_TOKEN
            meta["shadow_transit_min"] += gap_min
        elif gtype == "internal":
            # 동일 장소 체류: 직전 locus로 귀속
            shadow_token = before_tok if before_tok else SHADOW_UNKNOWN_TOKEN
            meta["shadow_transit_min"] += gap_min  # 이동 없는 체류이므로 transit으로 분류 안 함
        else:
            # 미확인 (unknown_start, unknown_end)
            shadow_token = SHADOW_UNKNOWN_TOKEN
            meta["shadow_unknown_min"] += gap_min

        # 1분 단위 레코드 생성
        gap_start = gap["gap_start"]
        gap_end = gap["gap_end"]

        # 음영 구간의 분별 timestamps 생성
        # gap_end는 포함하지 않음 (다음 BLE 기록이 있으면 중복 방지)
        n_records = max(1, int(round(gap_min)))
        for i in range(n_records):
            ts = gap_start + pd.Timedelta(minutes=i)
            if ts >= gap_end:
                break
            shadow_records.append({
                "user_no": user_no,
                "timestamp": ts,
                "locus_token": shadow_token,
                "locus_id": f"shadow_{gtype}",
                "is_shadow": True,
                "shadow_type": gtype,
                # 부가 정보
                "before_token": before_tok,
                "after_token": after_tok,
                "signal_count": 0,
                "active_count": 0,
                "active_ratio": 0.0,
            })

    meta["shadow_transit_min"] = round(meta["shadow_transit_min"], 1)
    meta["shadow_unknown_min"] = round(meta["shadow_unknown_min"], 1)

    if shadow_records:
        shadow_df = pd.DataFrame(shadow_records)
        # 원본에 없는 컬럼은 NaN으로 채움
        for col in result.columns:
            if col not in shadow_df.columns:
                shadow_df[col] = np.nan
        result = pd.concat([result, shadow_df], ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)

    return result, meta


def reconstruct_all_journeys(
    journey_df: pd.DataFrame,
    worker_df: pd.DataFrame,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 작업자 Journey 복원 + Coverage 계산.

    Args:
        journey_df: 1분 단위 BLE 기록 전체
        worker_df: 작업자별 집계 (in_datetime, out_datetime, work_minutes 포함)
        coverage_threshold: BLE coverage 최소 기준 (기본 30%)

    Returns:
        (reconstructed_journey_df, coverage_df)
        - reconstructed_journey_df: 음영 보정 포함 전체 journey
        - coverage_df: 작업자별 coverage 및 포함/제외 정보
    """
    logger.info(
        "Journey Reconstruction 시작: %d workers, threshold=%.0f%%",
        len(worker_df),
        coverage_threshold * 100,
    )

    all_reconstructed = []
    all_metas = []

    # 작업자별 처리
    worker_grps = {uid: grp for uid, grp in journey_df.groupby("user_no")}

    for _, wrow in worker_df.iterrows():
        user_no = wrow["user_no"]
        in_time = wrow.get("in_datetime")
        out_time = wrow.get("out_datetime")

        if pd.isna(in_time) if in_time is not None else True:
            in_time = None
        if pd.isna(out_time) if out_time is not None else True:
            out_time = None

        worker_journey = worker_grps.get(user_no, pd.DataFrame())

        if worker_journey.empty and (in_time is None or out_time is None):
            # BLE도 없고 AccessLog도 없음 -> 완전 제외
            all_metas.append({
                "user_no": user_no,
                "coverage_ratio": 0.0,
                "shadow_transit_min": 0.0,
                "shadow_unknown_min": 0.0,
                "n_gaps": 0,
                "excluded": True,
            })
            continue

        reconstructed, meta = reconstruct_worker_journey(
            worker_journey=worker_journey,
            in_time=in_time,
            out_time=out_time,
            user_no=user_no,
            coverage_threshold=coverage_threshold,
        )
        all_reconstructed.append(reconstructed)
        all_metas.append(meta)

    # 결과 합치기
    if all_reconstructed:
        reconstructed_df = pd.concat(all_reconstructed, ignore_index=True)
    else:
        reconstructed_df = journey_df.copy()
        reconstructed_df["is_shadow"] = False
        reconstructed_df["shadow_type"] = ""

    coverage_df = pd.DataFrame(all_metas)

    # Merge worker info
    if not worker_df.empty:
        merge_cols = ["user_no"]
        for col in ["user_name", "company_name", "in_datetime", "out_datetime", "work_minutes"]:
            if col in worker_df.columns:
                merge_cols.append(col)
        coverage_df = coverage_df.merge(
            worker_df[merge_cols], on="user_no", how="left"
        )

    # Coverage label
    coverage_df["coverage_pct"] = (coverage_df["coverage_ratio"] * 100).round(1)
    coverage_df["coverage_label"] = pd.cut(
        coverage_df["coverage_ratio"],
        bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=["불충분(<30%)", "부분(30-50%)", "양호(50-70%)", "충분(>70%)"],
    ).astype(str)

    n_total = len(coverage_df)
    n_excluded = coverage_df["excluded"].sum()
    n_included = n_total - n_excluded
    logger.info(
        "Reconstruction 완료: total=%d, included=%d (%.1f%%), excluded=%d",
        n_total,
        n_included,
        n_included / n_total * 100 if n_total > 0 else 0,
        n_excluded,
    )

    return reconstructed_df, coverage_df


def filter_by_coverage(
    transit_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """
    Transit DataFrame에서 coverage threshold 통과한 작업자만 필터링.

    Args:
        transit_df: transit_analyzer 출력 (user_no, mat_minutes, lbt_minutes, eod_minutes, ...)
        coverage_df: reconstruct_all_journeys() 출력 coverage 정보
        coverage_threshold: 필터링 기준

    Returns:
        (filtered_transit_df, stats_dict)
        - filtered_transit_df: threshold 통과 작업자만
        - stats_dict: {
            "n_total": int,
            "n_included": int,
            "n_excluded_low_coverage": int,
            "threshold_pct": float,
        }
    """
    if coverage_df.empty or transit_df.empty:
        return transit_df, {
            "n_total": len(transit_df),
            "n_included": len(transit_df),
            "n_excluded_low_coverage": 0,
            "threshold_pct": coverage_threshold * 100,
        }

    # included 작업자 목록
    included_workers = coverage_df[
        ~coverage_df["excluded"]
    ]["user_no"].tolist()

    n_total = len(transit_df)
    filtered = transit_df[transit_df["user_no"].isin(included_workers)].copy()
    n_included = len(filtered)

    # Coverage ratio/pct를 transit_df에 병합
    cov_merge = coverage_df[["user_no", "coverage_ratio", "coverage_pct", "coverage_label"]].copy()
    filtered = filtered.merge(cov_merge, on="user_no", how="left")

    stats = {
        "n_total": n_total,
        "n_included": n_included,
        "n_excluded_low_coverage": n_total - n_included,
        "threshold_pct": coverage_threshold * 100,
    }

    logger.info(
        "Coverage filter: total=%d -> included=%d (threshold=%.0f%%)",
        n_total,
        n_included,
        coverage_threshold * 100,
    )

    return filtered, stats


def get_coverage_threshold_recommendation(coverage_df: pd.DataFrame) -> dict:
    """
    데이터 분포를 분석하여 적정 threshold를 권고한다.

    Returns:
        {
            "recommended": float (0-1),
            "rationale": str,
            "distribution": dict,
        }
    """
    if coverage_df.empty or "coverage_ratio" not in coverage_df.columns:
        return {
            "recommended": DEFAULT_COVERAGE_THRESHOLD,
            "rationale": "데이터 없음 - 기본값 적용",
            "distribution": {},
        }

    ratios = coverage_df["coverage_ratio"].dropna()
    n = len(ratios)

    dist = {
        "n_total": n,
        "mean_pct": round(ratios.mean() * 100, 1),
        "median_pct": round(ratios.median() * 100, 1),
        "pct_below_30": round((ratios < 0.30).mean() * 100, 1),
        "pct_below_50": round((ratios < 0.50).mean() * 100, 1),
        "pct_above_70": round((ratios >= 0.70).mean() * 100, 1),
    }

    # 30% threshold: 전체의 20-25% 제외 예상 (적정)
    # 50% threshold: 전체의 35-40% 제외 (너무 엄격)
    below_30_pct = dist["pct_below_30"]

    if below_30_pct <= 25:
        recommended = 0.30
        rationale = (
            f"30% 미만 작업자가 {below_30_pct:.1f}%로 적정. "
            f"30% threshold로 BLE 신뢰도 최소 기준 확보 권장."
        )
    elif below_30_pct <= 35:
        recommended = 0.20
        rationale = (
            f"30% 미만 작업자가 {below_30_pct:.1f}%로 다소 많음. "
            f"20% threshold로 완화하여 분석 모집단 확보 권장."
        )
    else:
        recommended = 0.10
        rationale = (
            f"30% 미만 작업자가 {below_30_pct:.1f}%로 매우 많음 (BLE 환경 열악). "
            f"10% threshold로 최소한의 필터만 적용 권장."
        )

    return {
        "recommended": recommended,
        "rationale": rationale,
        "distribution": dist,
    }
