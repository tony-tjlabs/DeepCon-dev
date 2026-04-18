"""
Lone Work Detection — 단독 작업 감지 모듈
=========================================
1분 단위 journey 데이터에서 #no_lone 태그 구역에 혼자 있는 작업자 탐지.

건설현장 안전 규칙:
  - #no_lone 태그 구역 (밀폐공간, 고압전 등)에서 단독 작업 금지
  - 30분 이상 단독 → Critical (즉시 조치)
  - 10분 이상 단독 → Warning (주의)
"""
from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def detect_lone_work_realtime(
    journey_df: pd.DataFrame,
    locus_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    단독 작업 실시간 탐지.

    Args:
        journey_df: timestamp, user_no, locus_id, user_name 컬럼 필수
        locus_df: locus_id, tags, locus_name 컬럼 필수

    Returns:
        DataFrame with columns:
            user_no, user_name, locus_id, locus_name,
            lone_minutes (단독 체류 분), total_minutes (전체 체류 분),
            lone_ratio (단독 비율), latest_timestamp (마지막 단독 시간),
            status ("critical" | "warning" | "normal")

        빈 DataFrame if no lone work detected.
    """
    if journey_df.empty or locus_df.empty:
        return pd.DataFrame()

    # 1) #no_lone 태그 구역 필터
    if "tags" not in locus_df.columns:
        return pd.DataFrame()

    no_lone_loci = set(
        locus_df[locus_df["tags"].fillna("").astype(str).str.contains("#no_lone", na=False)]["locus_id"]
    )

    if not no_lone_loci:
        return pd.DataFrame()

    # 2) 해당 구역 체류 기록 필터
    df = journey_df[journey_df["locus_id"].isin(no_lone_loci)].copy()
    if df.empty:
        return pd.DataFrame()

    # timestamp가 datetime인지 확인
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 3) 1분 단위로 각 locus의 동시 체류 인원 계산
    df["minute"] = df["timestamp"].dt.floor("min")
    occupancy = df.groupby(["locus_id", "minute"]).agg(
        occupants=("user_no", "nunique"),
        users=("user_no", lambda x: list(x.unique())),
    ).reset_index()

    # 4) 혼자 있는 경우 추출
    lone_records = occupancy[occupancy["occupants"] == 1].copy()
    if lone_records.empty:
        return pd.DataFrame()

    # 5) 작업자별 단독 시간 집계
    lone_records["user_no"] = lone_records["users"].apply(lambda x: x[0])
    lone_summary = lone_records.groupby(["user_no", "locus_id"]).agg(
        lone_minutes=("minute", "count"),
        latest_timestamp=("minute", "max"),
    ).reset_index()

    # 6) 전체 체류 시간 계산
    total_stay = df.groupby(["user_no", "locus_id"]).agg(
        total_minutes=("minute", "nunique"),
    ).reset_index()

    result = lone_summary.merge(total_stay, on=["user_no", "locus_id"], how="left")
    result["lone_ratio"] = result["lone_minutes"] / result["total_minutes"].clip(lower=1)

    # 7) 작업자명 조인
    if "user_name" in journey_df.columns:
        user_names = journey_df[["user_no", "user_name"]].drop_duplicates()
        result = result.merge(user_names, on="user_no", how="left")
    else:
        result["user_name"] = result["user_no"].astype(str)

    # 8) 구역명 조인
    if "locus_name" in locus_df.columns:
        locus_names = locus_df[["locus_id", "locus_name"]].drop_duplicates()
        result = result.merge(locus_names, on="locus_id", how="left")
    else:
        result["locus_name"] = result["locus_id"]

    # 9) 상태 판정
    def _status(row: pd.Series) -> str:
        if row["lone_minutes"] >= 30:  # 30분 이상 단독
            return "critical"
        elif row["lone_minutes"] >= 10:  # 10분 이상
            return "warning"
        return "normal"

    result["status"] = result.apply(_status, axis=1)

    # 정렬: critical -> warning -> normal, lone_minutes 내림차순
    status_order = {"critical": 0, "warning": 1, "normal": 2}
    result["_status_order"] = result["status"].map(status_order)
    result = result.sort_values(
        ["_status_order", "lone_minutes"],
        ascending=[True, False]
    ).drop(columns=["_status_order"])

    return result


def get_current_lone_workers(
    journey_df: pd.DataFrame,
    locus_df: pd.DataFrame,
    window_minutes: int = 10,
) -> pd.DataFrame:
    """
    현재 단독 작업 중인 작업자 (최근 N분 내).

    Args:
        journey_df: 전체 journey
        locus_df: Locus 정보
        window_minutes: 최근 N분 (기본 10분)

    Returns:
        현재 단독 작업 중인 작업자 목록
    """
    if journey_df.empty:
        return pd.DataFrame()

    # timestamp가 datetime인지 확인
    if not pd.api.types.is_datetime64_any_dtype(journey_df["timestamp"]):
        journey_df = journey_df.copy()
        journey_df["timestamp"] = pd.to_datetime(journey_df["timestamp"])

    latest_time = journey_df["timestamp"].max()
    cutoff = latest_time - pd.Timedelta(minutes=window_minutes)
    recent = journey_df[journey_df["timestamp"] >= cutoff]

    return detect_lone_work_realtime(recent, locus_df)


def get_lone_work_summary(
    journey_df: pd.DataFrame,
    locus_df: pd.DataFrame,
) -> dict:
    """
    단독 작업 요약 통계.

    Returns:
        {
            "total_lone_workers": int,
            "critical_count": int,
            "warning_count": int,
            "no_lone_loci_count": int,
        }
    """
    if journey_df.empty or locus_df.empty:
        return {
            "total_lone_workers": 0,
            "critical_count": 0,
            "warning_count": 0,
            "no_lone_loci_count": 0,
        }

    no_lone_loci = locus_df[
        locus_df["tags"].fillna("").astype(str).str.contains("#no_lone", na=False)
    ] if "tags" in locus_df.columns else pd.DataFrame()

    lone_df = detect_lone_work_realtime(journey_df, locus_df)

    return {
        "total_lone_workers": len(lone_df["user_no"].unique()) if not lone_df.empty else 0,
        "critical_count": len(lone_df[lone_df["status"] == "critical"]) if not lone_df.empty else 0,
        "warning_count": len(lone_df[lone_df["status"] == "warning"]) if not lone_df.empty else 0,
        "no_lone_loci_count": len(no_lone_loci),
    }
