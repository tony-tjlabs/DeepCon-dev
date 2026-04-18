"""
Congestion Analysis Module — 공간별 혼잡도 분석
=================================================
journey.parquet에서 시간대별 공간별 동시 체류 인원을 계산.

핵심 메트릭:
  - occupancy: 특정 시간 빈(bin)에 특정 공간에 있는 고유 작업자 수
  - peak_occupancy: 하루 중 해당 공간의 최대 동시 인원
  - hourly_profile: 시간대(0~23h) × 공간 히트맵 데이터
  - congestion_ratio: peak / capacity (capacity 정의 시 활용)

적용: Y1_SKHynix, M15X_SKHynix 공통
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_mode(x: pd.Series) -> str:
    """mode()를 1번만 호출하여 최빈값 반환. 빈 Series면 'unknown'."""
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else "unknown"


def compute_congestion(
    journey_df: pd.DataFrame,
    time_bin_minutes: int = 30,
    locus_dict: dict | None = None,
) -> pd.DataFrame:
    """
    시간 빈별 공간별 동시 체류 인원 계산.

    Parameters
    ----------
    journey_df : DataFrame
        journey.parquet 로드 결과 (timestamp, user_no, locus_id, locus_token 필수)
    time_bin_minutes : int
        시간 빈 크기 (분). 기본 30분.
    locus_dict : dict, optional
        locus_id → {locus_name, ...} 매핑

    Returns
    -------
    DataFrame
        columns: time_bin, hour, locus_id, locus_name, locus_token, worker_count
    """
    if journey_df.empty or "timestamp" not in journey_df.columns:
        return pd.DataFrame()

    df = journey_df[["timestamp", "user_no", "locus_id", "locus_token"]].copy()
    df = df.dropna(subset=["locus_id"])

    # 시간 빈 생성
    freq = f"{time_bin_minutes}min"
    df["time_bin"] = df["timestamp"].dt.floor(freq)

    # 빈별 공간별 고유 작업자 수 (locus_id 기준, token은 별도 매핑)
    congestion = (
        df.groupby(["time_bin", "locus_id"])["user_no"]
        .nunique()
        .reset_index(name="worker_count")
    )

    congestion["hour"] = congestion["time_bin"].dt.hour

    # locus_token 매핑 (최빈값 사용)
    token_map = df.groupby("locus_id")["locus_token"].agg(
        _safe_mode
    ).to_dict()
    congestion["locus_token"] = congestion["locus_id"].map(token_map).fillna("unknown")

    # locus_name 매핑
    if locus_dict:
        congestion["locus_name"] = congestion["locus_id"].map(
            lambda lid: locus_dict.get(lid, {}).get("locus_name", lid)
        )
    else:
        congestion["locus_name"] = congestion["locus_id"]

    return congestion.sort_values(["time_bin", "locus_id"]).reset_index(drop=True)


def compute_hourly_profile(
    journey_df: pd.DataFrame,
    locus_dict: dict | None = None,
) -> pd.DataFrame:
    """
    시간대(0~23h) × 공간 평균 체류 인원 히트맵 데이터.

    Returns
    -------
    DataFrame
        columns: hour, locus_id, locus_name, locus_token, avg_workers, max_workers
    """
    if journey_df.empty:
        return pd.DataFrame()

    df = journey_df[["timestamp", "user_no", "locus_id", "locus_token"]].copy()
    df = df.dropna(subset=["locus_id"])
    df["hour"] = df["timestamp"].dt.hour

    # locus_token 매핑 (최빈값 사용)
    token_map = df.groupby("locus_id")["locus_token"].agg(
        _safe_mode
    ).to_dict()

    # 시간대별 공간별 고유 작업자 수 (분 단위 → 시간대 평균)
    df["minute_bin"] = df["timestamp"].dt.floor("1min")
    per_minute = (
        df.groupby(["hour", "minute_bin", "locus_id"])["user_no"]
        .nunique()
        .reset_index(name="worker_count")
    )

    hourly = (
        per_minute.groupby(["hour", "locus_id"])["worker_count"]
        .agg(avg_workers="mean", max_workers="max")
        .reset_index()
    )

    hourly["avg_workers"] = hourly["avg_workers"].round(1)
    hourly["locus_token"] = hourly["locus_id"].map(token_map).fillna("unknown")

    if locus_dict:
        hourly["locus_name"] = hourly["locus_id"].map(
            lambda lid: locus_dict.get(lid, {}).get("locus_name", lid)
        )
    else:
        hourly["locus_name"] = hourly["locus_id"]

    return hourly.sort_values(["hour", "locus_id"]).reset_index(drop=True)


def compute_congestion_summary(
    congestion_df: pd.DataFrame,
) -> dict:
    """
    혼잡도 요약 통계.

    Returns
    -------
    dict with keys:
        peak_space: 최대 혼잡 공간명
        peak_count: 최대 동시 인원
        peak_time: 최대 혼잡 시간대
        busiest_hour: 가장 붐비는 시간(시)
        quietest_hour: 가장 한산한 시간(시)
        total_spaces: 분석 대상 공간 수
        avg_occupancy: 전체 평균 점유
    """
    if congestion_df.empty:
        return {
            "peak_space": "—", "peak_count": 0, "peak_time": "—",
            "busiest_hour": 0, "quietest_hour": 0,
            "total_spaces": 0, "avg_occupancy": 0,
        }

    # 최대 혼잡 레코드
    peak_row = congestion_df.loc[congestion_df["worker_count"].idxmax()]
    peak_time_str = peak_row["time_bin"].strftime("%H:%M") if hasattr(peak_row["time_bin"], "strftime") else str(peak_row["time_bin"])

    # 시간대별 총 인원
    hourly_total = congestion_df.groupby("hour")["worker_count"].sum()
    busiest_h = int(hourly_total.idxmax()) if not hourly_total.empty else 0
    quietest_h = int(hourly_total.idxmin()) if not hourly_total.empty else 0

    return {
        "peak_space": peak_row.get("locus_name", peak_row["locus_id"]),
        "peak_count": int(peak_row["worker_count"]),
        "peak_time": peak_time_str,
        "busiest_hour": busiest_h,
        "quietest_hour": quietest_h,
        "total_spaces": congestion_df["locus_id"].nunique(),
        "avg_occupancy": round(congestion_df["worker_count"].mean(), 1),
    }


def compute_space_ranking(
    congestion_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    공간별 혼잡도 랭킹 (최대/평균 동시 인원 기준).

    Returns
    -------
    DataFrame
        columns: locus_id, locus_name, locus_token, max_workers, avg_workers, peak_hour
    """
    if congestion_df.empty:
        return pd.DataFrame()

    agg = congestion_df.groupby(["locus_id", "locus_name", "locus_token"]).agg(
        max_workers=("worker_count", "max"),
        avg_workers=("worker_count", "mean"),
        total_person_bins=("worker_count", "sum"),
    ).reset_index()

    # 각 공간의 피크 시간대
    peak_hours = (
        congestion_df.groupby(["locus_id", "hour"])["worker_count"]
        .sum()
        .reset_index()
    )
    idx = peak_hours.groupby("locus_id")["worker_count"].idxmax()
    peak_hour_map = peak_hours.loc[idx].set_index("locus_id")["hour"].to_dict()
    agg["peak_hour"] = agg["locus_id"].map(peak_hour_map)

    agg["avg_workers"] = agg["avg_workers"].round(1)
    return agg.nlargest(top_n, "max_workers").reset_index(drop=True)


def compute_multi_day_congestion(
    date_list: list[str],
    sector_id: str,
    locus_dict: dict | None = None,
) -> pd.DataFrame:
    """
    복수 날짜의 혼잡도 데이터 로드 + 집계.
    각 날짜별 시간대별 공간별 평균 인원 계산.

    Returns
    -------
    DataFrame
        columns: date, hour, locus_id, locus_name, avg_workers, max_workers
    """
    from src.pipeline.cache_manager import _date_dir

    rows = []
    for date_str in date_list:
        journey_path = _date_dir(date_str, sector_id) / "journey.parquet"
        if not journey_path.exists():
            continue
        try:
            jdf = pd.read_parquet(journey_path, columns=["timestamp", "user_no", "locus_id", "locus_token"])
            hourly = compute_hourly_profile(jdf, locus_dict)
            if not hourly.empty:
                hourly["date"] = date_str
                rows.append(hourly)
        except Exception as e:
            logger.warning(f"혼잡도 로드 실패 ({date_str}): {e}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compute_day_of_week_pattern(
    multi_day_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    요일별 시간대별 평균 혼잡도.

    Parameters
    ----------
    multi_day_df : DataFrame from compute_multi_day_congestion

    Returns
    -------
    DataFrame
        columns: day_of_week, day_name, hour, avg_workers
    """
    if multi_day_df.empty or "date" not in multi_day_df.columns:
        return pd.DataFrame()

    df = multi_day_df.copy()
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["day_of_week"] = df["dt"].dt.dayofweek
    day_names = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["day_name"] = df["day_of_week"].map(day_names)

    pattern = (
        df.groupby(["day_of_week", "day_name", "hour"])["avg_workers"]
        .mean()
        .reset_index()
    )
    pattern["avg_workers"] = pattern["avg_workers"].round(1)
    return pattern.sort_values(["day_of_week", "hour"]).reset_index(drop=True)
