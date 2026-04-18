"""
Data Loader — Raw CSV 로딩 모듈 (v4, 2026-04-11)
===================================
AccessLog(출입 이력) + TWardData(T-Ward 이동 위치)를 로드하고 join하여
파이프라인이 사용할 수 있는 형태로 반환.

[v3 추가]
  - classify_shift(): Entry 시간 기준 주간/야간 분류
  - load_daily_data(): 야간 근무자 D+1 TWardData 자동 보완
  - 헬멧 방치 탐지를 위한 exit_source 컬럼 추가

Q. AccessLog vs TWardData 역할 분리는?
A.  AccessLog  = 생체인식 출입 이력 (in/out 시간, 업체, T-Ward ID)
                 → 총 출입자 카운트의 참값 (T-Ward 미착용자 포함)
    TWardData  = 헬멧 T-Ward BLE 이동 위치 (1분 단위, 좌표/신호)
                 → T-Ward 착용 작업자의 실제 이동 데이터
    join key   = User_no

[v4 추가]
  - 합쳐진 파일 지원: "Y1_TWardData_20260401 ~ 20260409.csv" 형식 자동 감지
  - detect_raw_dates(): 합쳐진 파일의 날짜 범위를 파일명에서 파싱하여 확장
  - load_daily_data(): 날짜별 파일 없으면 합쳐진 파일에서 bulk_loader로 추출
  - load_access_log() / load_tward_data(): Path 또는 DataFrame 모두 허용

Q. 주간/야간 분류 기준은?
A.  주간 (day):   Entry 04:00 ~ 16:59
    야간 (night): Entry 17:00 ~ 23:59 또는 00:00 ~ 03:59
    → 야간 작업자는 자정을 넘겨 다음날 퇴근하므로 D+1 TWardData 보완 필요

Q. 야간 근무자 BLE 보완은?
A.  out_datetime이 D+1인 야간 작업자의 경우
    D+1 TWardData에서 해당 user_no의 오전 데이터(~ exit_time)를 추가.
    → 이로써 is_work_hour 범위가 자정을 넘겨 정확하게 계산됨.

Q. 총 출입자 vs 이동 작업자 차이는?
A.  총 출입자    = AccessLog unique user_no 수 (참값, 생체인식 기준)
    이동 작업자  = TWardData unique user_no 수 (T-Ward 착용 + BLE 수신)
    차이 = T-Ward 미착용자 + BLE 음영 작업자
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 날짜 포맷 헬퍼 ──────────────────────────────────────────────────
def _dot_date_repl(m: re.Match) -> str:
    """
    마침표 구분 날짜의 월/일을 제로패딩하여 대시 구분으로 변환.
    "2026.3.9 17:24" → regex 매치 ".3.9 " → "-03-09 "
    """
    month = m.group(1).zfill(2)
    day   = m.group(2).zfill(2)
    return f"-{month}-{day} "


# ─── 날짜 유틸 ─────────────────────────────────────────────────────
def _date_from_filename(fname: str) -> str | None:
    """파일명에서 날짜 추출 (YYYYMMDD 형식)."""
    m = re.search(r"(\d{8})", fname)
    return m.group(1) if m else None


def _adjacent_date_str(date_str: str, delta: int) -> str:
    """YYYYMMDD → ±delta일 YYYYMMDD 반환."""
    d = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=delta)
    return d.strftime("%Y%m%d")


def _expand_combined_dates(fname: str) -> list[str]:
    """
    합쳐진 파일명에서 날짜 범위 확장.

    "Y1_TWardData_20260401 ~ 20260409.csv" → ["20260401", ..., "20260409"]
    날짜가 2개 미만이면 빈 리스트 반환.
    """
    dates = re.findall(r"\d{8}", fname)
    if len(dates) < 2:
        return []
    start = datetime.strptime(dates[0], "%Y%m%d")
    end   = datetime.strptime(dates[1], "%Y%m%d")
    result = []
    d = start
    while d <= end:
        result.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return result


def _find_file_for_date(raw_dir: Path, pattern: str, date_str: str) -> tuple["Path | None", bool]:
    """
    date_str 데이터를 포함하는 파일 탐색.

    탐색 우선순위:
      1. 날짜별 파일: *{pattern}*{date_str}*.csv  (~가 없는 파일)
      2. 합쳐진 파일: *{pattern}*.csv  (~가 있고, 파일명 날짜 범위에 date_str 포함)

    Returns:
        (file_path, is_combined)
        is_combined=True → bulk_loader.extract_for_date() 사용 필요
    """
    target_dt = datetime.strptime(date_str, "%Y%m%d")

    # 1. 날짜별 파일 우선
    for f in raw_dir.glob(f"*{pattern}*.csv"):
        if "~" in f.name:
            continue
        if _date_from_filename(f.name) == date_str:
            return f, False

    # 2. 합쳐진 파일
    for f in sorted(raw_dir.glob(f"*{pattern}*.csv")):
        if "~" not in f.name:
            continue
        dates = re.findall(r"\d{8}", f.name)
        if len(dates) >= 2:
            start_dt = datetime.strptime(dates[0], "%Y%m%d")
            end_dt   = datetime.strptime(dates[1], "%Y%m%d")
            if start_dt <= target_dt <= end_dt:
                return f, True

    return None, False


def detect_raw_dates(raw_dir: Path) -> list[str]:
    """
    raw_dir에서 AccessLog + TWardData 쌍이 모두 있는 날짜 목록 반환.

    날짜별 파일 (Y1_TWardData_20260301.csv) 과
    합쳐진 파일 (Y1_TWardData_20260401 ~ 20260409.csv) 모두 지원.

    반환: ["20260301", "20260302", ..., "20260409"]
    """
    def _collect_dates(glob_pattern: str) -> set[str]:
        dates: set[str] = set()
        for f in raw_dir.glob(glob_pattern):
            if "~" in f.name:
                dates.update(_expand_combined_dates(f.name))
            else:
                d = _date_from_filename(f.name)
                if d:
                    dates.add(d)
        return dates

    access_dates = _collect_dates("*AccessLog*.csv")
    tward_dates  = _collect_dates("*TWardData*.csv")
    return sorted(access_dates & tward_dates)


# ─── Shift 분류 ────────────────────────────────────────────────────
def classify_shift(in_datetime: pd.Series) -> pd.Series:
    """
    Entry 시간(in_datetime) 기준 근무 Shift 분류.

    주간 (day):   04:00 ~ 16:59 입장
    야간 (night): 17:00 ~ 23:59 또는 00:00 ~ 03:59 입장
    unknown:      NaT

    Note:
      - 야간 작업자는 대부분 다음날 퇴근 (out_datetime = D+1)
      - EWI/CRE 분리 분석 및 D+1 BLE 보완 여부 판단에 사용
    """
    hour   = in_datetime.dt.hour
    result = pd.Series("unknown", index=in_datetime.index, dtype="object")
    known  = in_datetime.notna()
    result.loc[known & (hour >= 4) & (hour < 17)] = "day"
    result.loc[known & ((hour >= 17) | (hour < 4))] = "night"
    return result


# ─── AccessLog 로드 (출입 이력 — 생체인식 Gate) ────────────────────
def load_access_log(source: "Path | pd.DataFrame") -> pd.DataFrame:
    """
    AccessLog CSV 로드 → 출입 이력 + 업체 정보 정규화.

    source:
        Path       → CSV 파일 직접 로드 (날짜별 파일)
        pd.DataFrame → 이미 추출된 날짜 데이터 (bulk_loader.extract_for_date 결과)

    원본 컬럼:
        User_no, Worker_name, Cellphone, User_record_id,
        SCon_company_name, SCon_company_code,
        EmploymentStatus_Hycon, HyCon_company_name, HyCon_company_code,
        Entry_time, Exit_time, SCON_record_id, T-Ward ID

    반환 컬럼:
        user_no (int), user_name (str),
        company_name (str), company_code (str),
        in_datetime (datetime, KST), out_datetime (datetime, KST),
        work_minutes (float),
        shift_type (str): "day" / "night" / "unknown"
        exit_source (str): "access_log" / "missing"
        twardid (str|None), has_tward (bool)

    Note:
      - 야간 작업자의 Exit_time은 D+1 날짜를 포함할 수 있음
        (e.g., "2026-03-21 06:00:00") → 올바르게 파싱됨
      - Exit NaT = 퇴근 미기록 (exit_source="missing")
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source, encoding="cp949", low_memory=False)

    # ── user_no 정규화 ─────────────────────────────────────────────
    df["user_no"] = pd.to_numeric(df["User_no"], errors="coerce").astype("Int64")

    # ── 이름 ──────────────────────────────────────────────────────
    df["user_name"] = df["Worker_name"].fillna("").astype(str)

    # ── 업체명: HyCon 우선, 없으면 SCon fallback ─────────────────
    hcon = df["HyCon_company_name"].fillna("").astype(str)
    scon = df["SCon_company_name"].fillna("").astype(str)
    df["company_name"] = hcon.where(hcon != "", scon).where(hcon != "", "미확인")
    df["company_name"] = df["company_name"].replace("", "미확인")

    hcon_code = df["HyCon_company_code"].fillna("").astype(str)
    scon_code = df["SCon_company_code"].fillna("").astype(str)
    df["company_code"] = hcon_code.where(hcon_code != "", scon_code)

    # ── 출입 시간 파싱 (KST 벽시계 시간 유지) ────────────────────
    # 지원 포맷:
    #   Y1:   "2026-03-19 05:58:48.000 +0900" / "2026-03-19 17:10:17"
    #   M15X: "2026.3.9 17:24" / "2026.3.11 6:00" (마침표 구분, 제로패딩 없음)
    for src, dst in [("Entry_time", "in_datetime"), ("Exit_time", "out_datetime")]:
        if src in df.columns:
            cleaned = (
                df[src].astype(str)
                .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)
                .str.replace(r"\.\d{3}$", "", regex=True)   # ms 제거 (.000)
                .str.strip()
            )
            # 1차: Y1 포맷 (YYYY-MM-DD HH:MM:SS)
            parsed = pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")
            # 2차: 마침표 구분 포맷 (YYYY.M.D H:MM 등) — 1차 실패분만
            nat_mask = parsed.isna() & (cleaned != "nan") & (cleaned != "")
            if nat_mask.any():
                # "2026.3.9 17:24" → "2026-03-09 17:24:00"
                dot_cleaned = (
                    cleaned[nat_mask]
                    .str.replace(r"\.(\d{1,2})\.(\d{1,2})\s", _dot_date_repl, regex=True)
                )
                parsed_dot = pd.to_datetime(dot_cleaned, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                # HH:MM만 있는 경우 (초 없음)
                still_nat = parsed_dot.isna()
                if still_nat.any():
                    parsed_dot[still_nat] = pd.to_datetime(
                        dot_cleaned[still_nat], format="%Y-%m-%d %H:%M", errors="coerce"
                    )
                parsed[nat_mask] = parsed_dot
            df[dst] = parsed
        else:
            df[dst] = pd.NaT

    # ── 근무 시간 (분) ────────────────────────────────────────────
    # 야간 작업자: in=D 20:00, out=D+1 06:00 → 600분 (정확)
    # NaT out: clip → 0 (퇴근 미기록)
    delta = df["out_datetime"] - df["in_datetime"]
    df["work_minutes"] = (delta.dt.total_seconds() / 60).clip(lower=0).fillna(0)

    # ── Shift 분류 + exit_source ───────────────────────────────────
    df["shift_type"]  = classify_shift(df["in_datetime"])
    df["exit_source"] = np.where(df["out_datetime"].notna(), "access_log", "missing")

    # ── 퇴근 미기록 작업자 fallback (exit_source="missing") ───────
    # in_datetime은 있으나 out_datetime이 NaT인 작업자에 대해
    # 같은 shift_type의 평균 근무시간(분)을 in_datetime에 더해 out_datetime 추정
    _missing_mask = (df["exit_source"] == "missing") & df["in_datetime"].notna()
    _estimated_count = 0
    if _missing_mask.any():
        # shift_type별 평균 work_minutes 계산 (정상 퇴근 기록 기준)
        _valid = df[(df["exit_source"] == "access_log") & (df["work_minutes"] > 0)]
        _avg_wm_by_shift = _valid.groupby("shift_type")["work_minutes"].mean()

        # "unknown" shift → "day" 평균 사용 (건설현장 주간 작업 대부분)
        _day_fallback_wm = _avg_wm_by_shift.get("day", None)

        # 각 missing 작업자에게 shift별 평균 근무시간을 더해 out_datetime 추정
        # Note: Timedelta를 정수분으로 반올림하여 datetime64[us] 호환 보장
        for shift_type, avg_wm in _avg_wm_by_shift.items():
            _shift_mask = _missing_mask & (df["shift_type"] == shift_type)
            if _shift_mask.any():
                _td = pd.Timedelta(minutes=round(avg_wm))
                _est_out = df.loc[_shift_mask, "in_datetime"] + _td
                df.loc[_shift_mask, "out_datetime"] = _est_out

        # unknown shift → day 평균 적용
        _unknown_still = _missing_mask & (df["shift_type"] == "unknown") & df["out_datetime"].isna()
        if _unknown_still.any() and _day_fallback_wm is not None:
            _td = pd.Timedelta(minutes=round(_day_fallback_wm))
            _est_out = df.loc[_unknown_still, "in_datetime"] + _td
            df.loc[_unknown_still, "out_datetime"] = _est_out

        # work_minutes 재계산 (추정 적용 대상만)
        _estimated_mask = _missing_mask & df["out_datetime"].notna()
        if _estimated_mask.any():
            _delta = df.loc[_estimated_mask, "out_datetime"] - df.loc[_estimated_mask, "in_datetime"]
            _wm = _delta.dt.total_seconds() / 60

            # 안전장치: 0 < work_minutes <= 1440 범위만 유효
            _valid_wm = (_wm > 0) & (_wm <= 1440)
            _est_idx = _wm.index
            _good_idx = _est_idx[_valid_wm.values]
            _bad_idx  = _est_idx[~_valid_wm.values]

            df.loc[_good_idx, "work_minutes"] = _wm.loc[_good_idx].values
            df.loc[_good_idx, "exit_source"] = "estimated"
            _estimated_count = len(_good_idx)

            # 범위 밖 → 원래대로 missing 복원
            if len(_bad_idx) > 0:
                df.loc[_bad_idx, "out_datetime"] = pd.NaT
                df.loc[_bad_idx, "work_minutes"] = 0

        logger.info(
            f"퇴근 미기록 fallback: {int(_missing_mask.sum())}명 대상 → "
            f"{_estimated_count}명 추정 적용 (exit_source='estimated')"
        )

    # ── T-Ward ID ─────────────────────────────────────────────────
    tward_col = "T-Ward ID"
    if tward_col in df.columns:
        df["twardid"] = df[tward_col].astype(str).str.strip()
        df["twardid"] = df["twardid"].replace({"nan": None, "": None})
    else:
        df["twardid"] = None

    df["has_tward"] = df["twardid"].notna()

    # ── 중복 user_no 처리 ─────────────────────────────────────────
    # 같은 날 동일인 여러 출입 (야간→주간 교대 등)
    # → shift_type별 분리하여 work_minutes 합산 (야간 교대 데이터 보존)
    # → downstream에서 user_no unique 필요 → 합산 후 단일 행으로 축소
    if df["user_no"].duplicated().any():
        # 나머지 컬럼은 work_minutes가 가장 큰 레코드의 값 사용
        df = df.sort_values("work_minutes", ascending=False)
        first_rows = df.drop_duplicates(subset=["user_no"], keep="first").set_index("user_no")

        # 합산/min/max 집계: 전체 시간 범위 보존 + work_minutes 합산
        agg = df.groupby("user_no").agg(
            work_minutes_sum=("work_minutes", "sum"),
            in_earliest=("in_datetime", "min"),
            out_latest=("out_datetime", "max"),
        )
        first_rows["work_minutes"] = agg["work_minutes_sum"]
        first_rows["in_datetime"] = agg["in_earliest"]
        first_rows["out_datetime"] = agg["out_latest"]

        # exit_source: 하나라도 access_log이면 access_log
        exit_agg = df.groupby("user_no")["exit_source"].apply(
            lambda x: "access_log" if (x == "access_log").any() else "missing"
        )
        first_rows["exit_source"] = exit_agg

        df = first_rows.reset_index()

    keep = ["user_no", "user_name", "company_name", "company_code",
            "in_datetime", "out_datetime", "work_minutes",
            "shift_type", "exit_source",
            "twardid", "has_tward"]
    return df[keep].reset_index(drop=True)


# ─── TWardData 로드 (T-Ward 이동 위치 — BLE) ──────────────────────
def load_tward_data(source: "Path | pd.DataFrame") -> pd.DataFrame:
    """
    TWardData CSV 로드 → 1분 단위 이동 위치 정규화.

    source:
        Path       → CSV 파일 직접 로드 (날짜별 파일)
        pd.DataFrame → 이미 추출된 날짜 데이터 (bulk_loader.extract_for_date 결과)

    원본 컬럼:
        User_no, Time, Worker_name,
        Building, Level, Place, X, Y,
        Signal_count, ActiveSignal_count

    반환 컬럼:
        timestamp (datetime, 1분 단위),
        user_no (int), user_name (str),
        building_name (str), floor_name (str), spot_name (str),
        x (float), y (float),
        signal_count (int), active_count (int),
        active_ratio (float)
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source, encoding="cp949", low_memory=False)

    # ── timestamp 파싱 ─────────────────────────────────────────────
    # 포맷: "2026-03-19 23:59:00.000 +0900"
    cleaned_time = (
        df["Time"].astype(str)
        .str.replace(r"\s*[+-]\d{2}:?\d{2}$", "", regex=True)
        .str.replace(r"\.\d+$", "", regex=True)
        .str.strip()
    )
    df["timestamp"] = pd.to_datetime(cleaned_time, format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # ── 컬럼 정규화 ───────────────────────────────────────────────
    df["user_no"]       = pd.to_numeric(df["User_no"], errors="coerce").astype("Int64")
    df["user_name"]     = df["Worker_name"].fillna("").astype(str)
    df["building_name"] = df["Building"].fillna("").astype(str)
    df["floor_name"]    = df["Level"].fillna("").astype(str)
    df["spot_name"]     = df["Place"].fillna("unknown").astype(str)
    df["x"]             = pd.to_numeric(df["X"], errors="coerce")
    df["y"]             = pd.to_numeric(df["Y"], errors="coerce")
    df["signal_count"]  = pd.to_numeric(df["Signal_count"],       errors="coerce").fillna(0).astype(int)
    df["active_count"]  = pd.to_numeric(df["ActiveSignal_count"], errors="coerce").fillna(0).astype(int)

    # ── 활성 비율 (벡터화) ─────────────────────────────────────────
    mask = df["signal_count"] > 0
    df["active_ratio"] = 0.0
    df.loc[mask, "active_ratio"] = (
        df.loc[mask, "active_count"] / df.loc[mask, "signal_count"]
    )

    keep = ["timestamp", "user_no", "user_name",
            "building_name", "floor_name", "spot_name",
            "x", "y", "signal_count", "active_count", "active_ratio"]
    return df[keep].reset_index(drop=True)


# ─── 통합 로드 + Join ──────────────────────────────────────────────
def load_daily_data(raw_dir: Path, date_str: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    하루치 AccessLog(출입 이력) + TWardData(T-Ward 이동 위치) 로드 후 join.

    [v3 야간 근무 보완]
    야간 작업자(Entry >= 17:00)의 out_datetime이 D+1인 경우:
      → D+1 TWardData에서 해당 user_no의 exit_time 이전 BLE 기록을 추가
      → is_work_hour 필터가 자정을 넘겨 정확하게 적용됨

    반환:
        journey_df  : TWardData에 출입 정보(업체/시간)가 join된 이동 데이터
        access_df   : AccessLog 전체 (T-Ward 미착용자 포함)
        meta        : 로딩 통계 dict (shift 분류 포함)
    """
    # ── 파일 탐색 (날짜별 파일 우선, 없으면 합쳐진 파일) ────────────
    access_file, access_combined = _find_file_for_date(raw_dir, "AccessLog", date_str)
    tward_file,  tward_combined  = _find_file_for_date(raw_dir, "TWardData", date_str)

    if not access_file:
        raise FileNotFoundError(f"AccessLog 없음: {raw_dir} [{date_str}]")
    if not tward_file:
        raise FileNotFoundError(f"TWardData 없음: {raw_dir} [{date_str}]")

    # ── D 날짜 로드 ────────────────────────────────────────────────
    if access_combined:
        from src.pipeline.bulk_loader import extract_for_date
        access_df = load_access_log(extract_for_date(access_file, "Entry_time", date_str))
    else:
        access_df = load_access_log(access_file)

    if tward_combined:
        from src.pipeline.bulk_loader import extract_for_date
        tward_df = load_tward_data(extract_for_date(tward_file, "Time", date_str))
    else:
        tward_df = load_tward_data(tward_file)

    # ── 야간 작업자 D+1 BLE 보완 ──────────────────────────────────
    # 야간 근무자 중 out_datetime이 D+1 날짜인 경우 (퇴근 시간이 자정 이후)
    # → D+1 TWardData에서 exit_time 이전 BLE 기록 추가
    next_date_str = _adjacent_date_str(date_str, 1)
    next_date_pd  = pd.Timestamp(datetime.strptime(next_date_str, "%Y%m%d"))
    tward_next_file, tward_next_combined = _find_file_for_date(raw_dir, "TWardData", next_date_str)

    night_with_next_exit = access_df[
        (access_df["shift_type"] == "night")
        & access_df["out_datetime"].notna()
        & (access_df["out_datetime"].dt.date == next_date_pd.date())
    ]

    night_supplement_count = 0
    incomplete_night_workers = 0
    if not night_with_next_exit.empty and not tward_next_file:
        # ★ D+1 TWardData 없음 → 야간 작업자 BLE 불완전 (경고)
        incomplete_night_workers = len(night_with_next_exit)
        logger.warning(
            f"D+1 TWardData({next_date_str}) 없음 — "
            f"야간 작업자 {incomplete_night_workers}명 BLE 불완전"
        )
    if not night_with_next_exit.empty and tward_next_file:
        if tward_next_combined:
            from src.pipeline.bulk_loader import extract_for_date
            tward_next = load_tward_data(extract_for_date(tward_next_file, "Time", next_date_str))
        else:
            tward_next = load_tward_data(tward_next_file)

        # 야간 작업자의 D+1 새벽 BLE만 추출 (exit_time 이전까지)
        exit_map = (
            night_with_next_exit
            .set_index("user_no")["out_datetime"]
            .to_dict()
        )
        night_nos = set(night_with_next_exit["user_no"].dropna())

        mask_night = tward_next["user_no"].isin(night_nos)
        tward_next_night = tward_next[mask_night].copy()

        # 각 작업자의 exit_time 이전 레코드만 유지 (벡터화)
        tward_next_night["_exit"] = tward_next_night["user_no"].map(exit_map)
        tward_next_night = tward_next_night[
            tward_next_night["timestamp"] <= tward_next_night["_exit"]
        ].drop(columns=["_exit"])

        night_supplement_count = len(tward_next_night)
        tward_df = pd.concat([tward_df, tward_next_night], ignore_index=True)

    # ── Join: TWardData ← AccessLog (user_no 기준, left join) ─────
    join_cols = ["user_no", "company_name", "company_code",
                 "in_datetime", "out_datetime", "work_minutes",
                 "shift_type", "exit_source",
                 "twardid", "has_tward"]
    join_cols = [c for c in join_cols if c in access_df.columns]

    journey_df = tward_df.merge(
        access_df[join_cols],
        on="user_no",
        how="left",
    )

    # ── 미매핑 처리 ───────────────────────────────────────────────
    journey_df["company_name"] = journey_df["company_name"].fillna("미확인")
    journey_df["has_tward"]    = journey_df["has_tward"].infer_objects(copy=False).fillna(False).astype(bool)
    journey_df["shift_type"]   = journey_df["shift_type"].fillna("unknown")
    journey_df["exit_source"]  = journey_df["exit_source"].fillna("missing")

    # ── is_work_hour 계산 ───────────────────────────────────────
    # ★ v2: out_datetime이 NaT인 경우 FAR_FUTURE로 채우면 모든 BLE가
    #        근무시간 판정 → 비정상 EWI. in/out 모두 있어야 판정.
    _FAR_PAST   = pd.Timestamp("1900-01-01")
    in_dt  = journey_df["in_datetime"].fillna(_FAR_PAST)
    out_dt = journey_df["out_datetime"].fillna(_FAR_PAST)  # ★ NaT → FAR_PAST (is_work_hour=False)
    ts     = journey_df["timestamp"].fillna(_FAR_PAST)
    journey_df["is_work_hour"] = (
        journey_df["in_datetime"].notna()
        & journey_df["out_datetime"].notna()  # ★ exit도 있어야 판정 가능
        & journey_df["timestamp"].notna()
        & (ts >= in_dt)
        & (ts <= out_dt)
    )
    journey_df["missing_exit"] = journey_df["out_datetime"].isna()

    # ── Shift별 통계 (메타) ───────────────────────────────────────
    day_workers   = int(access_df[access_df["shift_type"] == "day"  ]["user_no"].nunique())
    night_workers = int(access_df[access_df["shift_type"] == "night"]["user_no"].nunique())
    missing_exit  = int((access_df["exit_source"] == "missing").sum())

    meta = {
        "date_str":              date_str,
        "total_records":         len(journey_df),
        "total_workers_access":  access_df["user_no"].nunique(),
        "total_workers_move":    int(
            tward_df.groupby("user_no")["active_count"]
            .sum()
            .pipe(lambda s: (s >= 10).sum())
        ),  # ★ 활성 신호 10회 이상인 작업자만 카운팅 (비활성 헬멧 제외)
        "tward_holders":         int(access_df["has_tward"].sum()),
        "companies":             access_df["company_name"].nunique(),
        "spots":                 journey_df["spot_name"].nunique(),
        "day_workers":           day_workers,
        "night_workers":         night_workers,
        "missing_exit_workers":  missing_exit,
        "night_supplement_records": night_supplement_count,
        "incomplete_night_workers": incomplete_night_workers,
        "time_start":            str(journey_df["timestamp"].min()),
        "time_end":              str(journey_df["timestamp"].max()),
    }

    return journey_df, access_df, meta
