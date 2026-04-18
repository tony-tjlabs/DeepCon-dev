"""
Bulk Loader — 합쳐진 CSV에서 날짜별 데이터 추출
================================================
날짜 범위가 하나의 파일에 통합된 경우를 청크 기반으로 처리.

배경:
    2026-04 부터 Y1 데이터는 날짜별 파일이 아닌 기간 단위로 제공됨.
    예) Y1_TWardData_20260401 ~ 20260409.csv  (45M 행, 9일 통합)
        Y1_AccessLog_20260401 ~ 20260409.csv  (130K 행, 9일 통합)

    날짜별 파일 (Y1_TWardData_20260301.csv 등) 은 기존 방식 그대로 처리됨.
    이 모듈은 합쳐진 파일에서 특정 날짜 데이터만 추출하는 역할만 담당.

사용:
    from src.pipeline.bulk_loader import extract_for_date

    df = extract_for_date(
        file_path="data/raw/Y1_SKHynix/Y1_TWardData_20260401 ~ 20260409.csv",
        time_col="Time",
        date_str="20260403",
    )
    # → 해당 날짜 원본 컬럼 그대로 반환 (loader.py 정규화 로직에 그대로 통과 가능)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500_000  # 50만 행 단위 (~40MB)


# =============================================================================
# Time Parsing
# =============================================================================

def parse_time_col(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    시간 컬럼 파싱 — timezone strip + millisecond 제거 → datetime64[ns].

    지원 포맷:
        "2026-04-03 23:59:00.000 +0900"  (TWardData Time)
        "2026-04-03 06:31:38.000 +0900"  (AccessLog Entry_time)
    """
    if series.empty:
        return pd.Series(dtype="datetime64[ns]")

    s = (
        series.astype(str)
        .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)  # timezone 제거
        .str.replace(r"\.\d+$", "", regex=True)                  # 밀리초 제거
        .str.strip()
    )
    result = pd.to_datetime(s, errors="coerce")

    new_nulls = result.isna().sum() - series.isna().sum()
    if new_nulls > 0:
        logger.warning(f"[{col_name}] {new_nulls:,}개 행 시간 파싱 실패 (총 {len(series):,}행)")

    return result


# =============================================================================
# Date Extraction
# =============================================================================

def extract_for_date(
    file_path: "Path | str",
    time_col: str,
    date_str: str,
    encoding: str = "cp949",
) -> pd.DataFrame:
    """
    합쳐진 CSV에서 특정 날짜(date_str) 데이터만 청크 기반으로 추출.

    - 원본 컬럼을 그대로 유지 (시간 컬럼 포함)
    - 시간 컬럼은 필터링용으로만 파싱하고 원본 문자열로 복원
      → loader.py의 기존 정규화 로직이 그대로 적용 가능

    Args:
        file_path : CSV 파일 경로
        time_col  : 날짜 기준 컬럼명 ("Time" 또는 "Entry_time")
        date_str  : 추출할 날짜 "YYYYMMDD"
        encoding  : 파일 인코딩 (기본 cp949)

    Returns:
        해당 날짜의 raw DataFrame. 데이터 없으면 빈 DataFrame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"파일 없음: {file_path}")
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []

    try:
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False,
            chunksize=CHUNK_SIZE,
        ):
            if time_col not in chunk.columns:
                logger.error(f"컬럼 없음: '{time_col}' in {file_path.name}")
                return pd.DataFrame()

            # 원본 문자열 보존 (정규화는 loader.py에서 수행)
            original_col = chunk[time_col].copy()

            # 필터링용 파싱 (임시)
            parsed = parse_time_col(chunk[time_col], time_col)
            mask = parsed.dt.strftime("%Y%m%d") == date_str

            if mask.any():
                matched = chunk.loc[mask].copy()
                matched[time_col] = original_col[mask].values  # 원본 문자열 복원
                chunks.append(matched)

    except Exception as e:
        logger.error(f"extract_for_date 실패: {file_path.name}, date={date_str}, 에러: {e}")
        return pd.DataFrame()

    if not chunks:
        logger.warning(f"데이터 없음: {file_path.name} [{date_str}]")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"추출 완료: {file_path.name} [{date_str}] → {len(df):,} rows")
    return df
