"""
integrity/context.py — 정합성 탭 데이터 로드 계층
==================================================
Streamlit 캐시 기반 loader 모음 + `SubTabContext` dataclass.

각 서브탭 render 함수는 이 모듈의 캐시 함수를 직접 호출하거나
`SubTabContext` 의 lazy 메서드로 필요한 데이터를 얻는다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.integrity.helpers import (
    ACTIVE_LOW_THRESHOLD,
    HELMET_SUSPECT_MIN_RUN,
    _find_raw_file,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 캐시 로더
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# 캐시 로더
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=600)
def _load_journey(sector_id: str, date_str: str, journey_path: str,
                  _mtime: float = 0.0) -> pd.DataFrame:
    """journey.parquet 로드 — 핵심 컬럼만.

    _mtime: 파일 수정 시각 (float) — 재처리 후 캐시 자동 무효화용.
    캐시 키에 포함되어 parquet이 바뀌면 즉시 새 데이터를 로드한다.
    """
    try:
        cols = [
            "timestamp", "user_no", "user_name", "company_name",
            "spot_name", "locus_id", "locus_name", "locus_token",
            "signal_count", "active_count", "active_ratio",
            "is_gap_filled", "gap_confidence", "gap_method",   # ★ gap_method 추가
            "is_low_confidence",
            "activity_level", "is_valid_transition",
            "block_id", "block_type", "block_duration_min",
            "is_transition", "locus_changed",
            "is_work_hour", "has_tward",
        ]
        # parquet에 없는 컬럼은 조용히 제외
        import pyarrow.parquet as pq
        available = set(pq.read_schema(journey_path).names)
        cols = [c for c in cols if c in available]
        df = pd.read_parquet(journey_path, columns=cols)

        # nullable boolean(pd.NA) → 일반 bool 변환
        # "boolean value of NA is ambiguous" 에러 예방
        bool_fill: list[tuple[str, bool]] = [
            ("is_gap_filled",       False),   # 보정 여부 — 없으면 미보정
            ("is_low_confidence",   False),   # 저신뢰 여부
            ("is_transition",       False),   # 전이 여부
            ("locus_changed",       False),   # locus 변경 여부
            ("is_work_hour",        True),    # 작업 시간대 — 없으면 유효로 가정
            ("has_tward",           False),   # T-Ward 착용 여부
            ("is_valid_transition", True),    # 비유효 전이 — NA는 유효로 가정
        ]
        for col, fill_val in bool_fill:
            if col in df.columns:
                df[col] = df[col].fillna(fill_val).astype(bool)

        return df
    except Exception as e:
        logger.warning(f"journey 로드 실패: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def _load_worker(sector_id: str, date_str: str, worker_path: str) -> pd.DataFrame:
    """worker.parquet 로드."""
    try:
        cols = ["user_no", "user_name", "company_name", "shift_type",
                "work_minutes", "ble_coverage", "ble_coverage_pct",
                "gap_ratio", "gap_min", "recorded_work_min",
                "helmet_abandoned", "ewi_reliable"]
        df = pd.read_parquet(worker_path, columns=cols)

        # nullable boolean → bool 변환
        for col, fill_val in [("helmet_abandoned", False), ("ewi_reliable", False)]:
            if col in df.columns:
                df[col] = df[col].fillna(fill_val).astype(bool)

        return df
    except Exception as e:
        logger.warning(f"worker 로드 실패: {e}")
        return pd.DataFrame()


# ─── Raw CSV 로더 ────────────────────────────────────────────────────────

def _find_raw_file(raw_dir: Path, prefix: str, date_str: str) -> "Path | None":
    """날짜에 해당하는 raw CSV 파일 경로를 반환 (단일일/범위 파일 모두 지원)."""
    exact = raw_dir / f"{prefix}_{date_str}.csv"
    if exact.exists():
        return exact
    target = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    for f in sorted(raw_dir.glob(f"{prefix}_*.csv")):
        stem = f.stem.replace(f"{prefix}_", "")
        if "~" in stem:
            parts = [p.strip() for p in stem.split("~")]
            if len(parts) == 2:
                try:
                    d0 = parts[0].replace(" ", "")[:8]
                    d1 = parts[1].replace(" ", "")[:8]
                    s = pd.Timestamp(f"{d0[:4]}-{d0[4:6]}-{d0[6:8]}")
                    e = pd.Timestamp(f"{d1[:4]}-{d1[4:6]}-{d1[6:8]}")
                    if s <= target <= e:
                        return f
                except Exception:
                    pass
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def _load_access_log_for_date(raw_dir: str, date_str: str) -> pd.DataFrame:
    """AccessLog CSV → 해당 날짜와 관련된 출입기록 전체.

    포함 기준 (OR):
      - Entry_time 날짜 == date_str  (당일 입장)
      - Exit_time  날짜 == date_str  (전날 야간근무 → 당일 퇴장)

    야간 교대 등으로 날짜가 걸쳐있는 기록이 누락되지 않도록 양방향 필터.
    """
    # 당일 파일 또는 당일이 포함된 범위 파일 검색
    fpath = _find_raw_file(Path(raw_dir), "Y1_AccessLog", date_str)

    # 전날도 확인 (야간교대: 전날 입장 → 당일 퇴장)
    prev_date_str = (
        pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        - pd.Timedelta(days=1)
    ).strftime("%Y%m%d")
    fpath_prev = _find_raw_file(Path(raw_dir), "Y1_AccessLog", prev_date_str)

    frames: list[pd.DataFrame] = []

    def _read_and_parse(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, encoding="cp949", low_memory=False)
        df["Entry_time"] = pd.to_datetime(
            df["Entry_time"].astype(str).str[:19], errors="coerce"
        )
        df["Exit_time"] = pd.to_datetime(
            df["Exit_time"].astype(str).str[:19], errors="coerce"
        )
        return df

    try:
        if fpath is not None:
            df_cur = _read_and_parse(fpath)
            # 당일 입장 OR 당일 퇴장
            mask_cur = (
                (df_cur["Entry_time"].dt.strftime("%Y%m%d") == date_str) |
                (df_cur["Exit_time"].dt.strftime("%Y%m%d") == date_str)
            )
            frames.append(df_cur[mask_cur])

        if fpath_prev is not None and fpath_prev != fpath:
            df_prev = _read_and_parse(fpath_prev)
            # 전날 입장 + 당일 퇴장 (야간 교대)
            mask_prev = (
                (df_prev["Entry_time"].dt.strftime("%Y%m%d") == prev_date_str) &
                (df_prev["Exit_time"].dt.strftime("%Y%m%d") == date_str)
            )
            frames.append(df_prev[mask_prev])

    except Exception as e:
        logger.warning(f"AccessLog 로드 실패: {e}")
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    # 중복 제거 (같은 파일에서 양쪽 조건 모두 걸릴 수 있는 경우 대비)
    if "User_record_id" in result.columns:
        result = result.drop_duplicates(subset="User_record_id")
    result = result.sort_values("Entry_time", na_position="last").reset_index(drop=True)
    return result


@st.cache_data(show_spinner=False, ttl=1200)
def _compute_helmet_suspect_by_user(
    journey_path: str, locus_csv: str, _mtime: float = 0.0,
) -> dict[int, int]:
    """해당 날짜의 journey.parquet를 1회 스캔하여 user_no → 헬멧 방치 의심 분 수 매핑.

    헬멧 방치 기준 = WORK_AREA + (active_ratio ≤ 0.40 OR activity_level == ESTIMATED)
                    가 연속 30분 이상 지속된 총 분 수.

    작업자 셀렉트박스의 "헬멧 방치 의심 시간 많은 순" 정렬에 사용.
    캐시는 journey.parquet 수정시각(_mtime)으로 무효화.
    """
    try:
        cols_needed = ["timestamp", "user_no", "locus_id",
                       "active_ratio", "activity_level", "is_gap_filled"]
        import pyarrow.parquet as pq
        available = set(pq.read_schema(journey_path).names)
        cols = [c for c in cols_needed if c in available]
        jdf = pd.read_parquet(journey_path, columns=cols)
        if jdf.empty:
            return {}

        # locus_id → locus_type
        locus = pd.read_csv(locus_csv, encoding="utf-8")
        if "locus_type" not in locus.columns or "locus_id" not in locus.columns:
            return {}
        lid2lt = locus.set_index("locus_id")["locus_type"].to_dict()
        jdf["_ltype"] = jdf["locus_id"].map(lid2lt)

        # 티어
        ar = jdf["active_ratio"].fillna(0.0)
        is_gap = jdf.get("is_gap_filled", pd.Series(False, index=jdf.index)).fillna(False).astype(bool)
        if "activity_level" in jdf.columns:
            is_gap = is_gap | jdf["activity_level"].fillna("").eq("ESTIMATED")
        is_low = ar <= ACTIVE_LOW_THRESHOLD

        # gap-filled는 "저활성" 으로 치지 않음 (ESTIMATED는 근거가 약함 → 판정 유보)
        suspect = (jdf["_ltype"] == "WORK_AREA") & is_low & (~is_gap)
        jdf = jdf.assign(_suspect=suspect.values)

        # user별 run-length 집계
        # user_no로 정렬 후 (user_no, suspect) 경계에서 run 생성
        jdf = jdf.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
        # 각 user 내에서 run id
        user_change = jdf["user_no"] != jdf["user_no"].shift()
        susp_change = jdf["_suspect"] != jdf["_suspect"].shift()
        jdf["_run"] = (user_change | susp_change).cumsum()

        susp_only = jdf[jdf["_suspect"]]
        if susp_only.empty:
            return {}

        run_sizes = susp_only.groupby(["_run", "user_no"]).size()
        # 30분 이상인 run만 합산
        long_runs = run_sizes[run_sizes >= HELMET_SUSPECT_MIN_RUN]
        if long_runs.empty:
            return {}
        by_user = long_runs.groupby(level="user_no").sum()
        return {int(k): int(v) for k, v in by_user.items()}
    except Exception as e:
        logger.warning(f"helmet suspect 집계 실패: {e}")
        return {}

@st.cache_data(show_spinner=False, ttl=3600)
def _load_tward_full_size(fpath: str) -> int:
    """Raw TWardData CSV 전체 행 수 — Raw 여부 검증용.

    파일 자체의 row count(헤더 제외)를 반환. 탭 상단에 "원본 전체 N행 / 본 작업자 M행"
    으로 표시해 사용자가 가공 여부를 즉시 판단할 수 있도록 한다.
    """
    if not fpath:
        return 0
    try:
        # nrows/usecols 없이 전체 읽되 1컬럼만 — 성능 개선
        df = pd.read_csv(fpath, encoding="cp949", usecols=[0], low_memory=False)
        return int(len(df))
    except Exception as e:
        logger.warning(f"TWardData 전체 행수 계산 실패: {e}")
        return 0


@st.cache_data(show_spinner=False, ttl=3600)
def _load_tward_for_date(raw_dir: str, date_str: str) -> pd.DataFrame:
    """TWardData CSV → 해당 날짜의 1분 단위 원본 BLE 기록 전체."""
    fpath = _find_raw_file(Path(raw_dir), "Y1_TWardData", date_str)
    if fpath is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(fpath, encoding="cp949", low_memory=False)
        df["Time"] = pd.to_datetime(
            df["Time"].astype(str).str[:19], errors="coerce"
        )
        mask = df["Time"].dt.strftime("%Y%m%d") == date_str
        return df[mask].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"TWardData 로드 실패: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def _load_locus_meta(locus_csv: str) -> pd.DataFrame:
    """locus_v2.csv → locus_id 기준 메타 테이블 (locus_type·building·floor·function·좌표)."""
    try:
        df = pd.read_csv(locus_csv, encoding="utf-8")
        keep = ["locus_id", "locus_name", "locus_type", "building", "floor",
                "function", "dwell_category", "hazard_level", "congestion_prone",
                "location_x", "location_y"]
        sub = df[[c for c in keep if c in df.columns]].copy()
        # journey에 이미 locus_name이 있으므로 메타쪽은 locus_meta_name으로 구분
        if "locus_name" in sub.columns:
            sub = sub.rename(columns={"locus_name": "locus_meta_name"})
        # 좌표도 journey.x/y와 구분
        if "location_x" in sub.columns:
            sub = sub.rename(columns={"location_x": "locus_x", "location_y": "locus_y"})
        return sub
    except Exception as e:
        logger.warning(f"locus_v2 로드 실패: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def _build_place_ltype_map(journey_path: str, locus_csv: str) -> dict:
    """섹터 전체 journey.parquet 기반 spot_name → dominant locus_type 매핑 (캐시됨).

    per-row (분 단위) 집계 후 mode를 취하므로 단순 locus 개수 기준보다 정확.
    동일 spot_name에 WORK_AREA 497분 + REST_AREA 70분이면 → WORK_AREA.
    """
    try:
        jdf = pd.read_parquet(journey_path, columns=["spot_name", "locus_id"])
        locus = pd.read_csv(locus_csv, encoding="utf-8")
        lid_to_ltype = locus.set_index("locus_id")["locus_type"].to_dict()
        jdf["_lt"] = jdf["locus_id"].map(lid_to_ltype)
        jdf["spot_name"] = jdf["spot_name"].fillna("").str.strip()
        valid = jdf[(jdf["spot_name"] != "") & jdf["_lt"].notna()]
        if valid.empty:
            return {}
        mode_series = (valid.groupby("spot_name")["_lt"]
                       .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
                       .dropna())
        return mode_series.to_dict()
    except Exception as e:
        logger.warning(f"place_ltype_map 구축 실패: {e}")
        return {}

def _daily_integrity_one(args: tuple) -> dict | None:
    """단일 날짜 정합성 집계 — 병렬 worker용 (모듈 전역 함수, picklable)."""
    d, jp = args
    try:
        df = pd.read_parquet(
            jp,
            columns=["is_gap_filled", "gap_confidence",
                     "is_low_confidence", "is_valid_transition",
                     "signal_count", "activity_level"],
        )
        n = len(df)
        if n == 0:
            return None
        # nullable boolean → bool 변환 (벡터화)
        gap_filled = df["is_gap_filled"].fillna(False).astype(bool)
        low_conf   = df["is_low_confidence"].fillna(False).astype(bool)
        valid_tr   = df["is_valid_transition"].fillna(True).astype(bool)
        signal_s   = df["signal_count"]

        gap_dist = df["gap_confidence"].value_counts().to_dict()

        return {
            "date":             d,
            "total_rows":       n,
            "gap_filled_rate":  int(gap_filled.sum()) / n * 100,
            "low_conf_rate":    int(low_conf.sum()) / n * 100,
            "invalid_tr_rate":  int((~valid_tr).sum()) / n * 100,
            "zero_signal_rate": int((signal_s == 0).sum()) / n * 100,
            "avg_signal":       float(signal_s.mean()),
            "gap_high":         gap_dist.get("high", 0),
            "gap_medium":       gap_dist.get("medium", 0),
            "gap_low":          gap_dist.get("low", 0),
            "gap_none":         gap_dist.get("none", 0),
        }
    except Exception as e:
        logger.warning(f"[{d}] 집계 실패: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=1200)
def _compute_daily_integrity_stats(sector_id: str) -> pd.DataFrame:
    """
    전체 날짜별 정합성 통계 집계.
    journey.parquet 순회하며 보정률/신호품질 요약.

    ★ 성능: ThreadPoolExecutor로 I/O 병렬화 (40일 × 순차 읽기 → 병렬 읽기).
    """
    from concurrent.futures import ThreadPoolExecutor
    from src.pipeline.cache_manager import detect_processed_dates

    paths = cfg.get_sector_paths(sector_id)
    tasks: list[tuple[str, Path]] = []
    for d in detect_processed_dates(sector_id):
        jp = paths["processed_dir"] / d / "journey.parquet"
        if jp.exists():
            tasks.append((d, jp))

    if not tasks:
        return pd.DataFrame()

    # I/O bound → 스레드 풀 (GIL 해제됨). max_workers는 디스크·메모리 여유에 맞춤.
    max_w = min(len(tasks), 8)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        for r in pool.map(_daily_integrity_one, tasks):
            if r is not None:
                rows.append(r)

    # 날짜 정렬 (병렬 결과 순서 보장용)
    rows.sort(key=lambda x: x["date"])
    return pd.DataFrame(rows) if rows else pd.DataFrame()



# ═══════════════════════════════════════════════════════════════════════
# SubTabContext — 서브탭 간 공통 문맥
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SubTabContext:
    """정합성 서브탭이 필요한 데이터를 lazy 로드하는 공용 컨텍스트.

    각 서브탭은 `ctx.sector_id / ctx.paths` 만 직접 읽고,
    journey/worker/access 등은 필요 시점에 `get_*` 메서드로 조회한다.
    """
    sector_id: str
    date_str: str | None = None
    paths: dict = field(default_factory=dict)
    _journey: pd.DataFrame | None = None
    _worker: pd.DataFrame | None = None
    _access: pd.DataFrame | None = None
    _tward: pd.DataFrame | None = None
    _locus_meta: pd.DataFrame | None = None

    @classmethod
    def build(cls, sector_id: str, date_str: str | None = None) -> "SubTabContext":
        paths = cfg.get_sector_paths(sector_id)
        return cls(sector_id=sector_id, date_str=date_str, paths=paths)

    def _journey_path(self) -> Path | None:
        if not self.date_str:
            return None
        p = self.paths["processed_dir"] / self.date_str / "journey.parquet"
        return p if p.exists() else None

    def _worker_path(self) -> Path | None:
        if not self.date_str:
            return None
        p = self.paths["processed_dir"] / self.date_str / "worker.parquet"
        return p if p.exists() else None

    def get_journey(self) -> pd.DataFrame:
        if self._journey is not None:
            return self._journey
        jp = self._journey_path()
        if jp is None:
            self._journey = pd.DataFrame()
        else:
            self._journey = _load_journey(
                self.sector_id, self.date_str or "", str(jp),
                _mtime=jp.stat().st_mtime,
            )
        return self._journey

    def get_worker(self) -> pd.DataFrame:
        if self._worker is not None:
            return self._worker
        wp = self._worker_path()
        self._worker = _load_worker(
            self.sector_id, self.date_str or "", str(wp)
        ) if wp else pd.DataFrame()
        return self._worker

    def get_access(self) -> pd.DataFrame:
        if self._access is not None:
            return self._access
        if not self.date_str:
            self._access = pd.DataFrame()
        else:
            self._access = _load_access_log_for_date(
                str(self.paths["raw_dir"]), self.date_str,
            )
        return self._access

    def get_tward(self) -> pd.DataFrame:
        if self._tward is not None:
            return self._tward
        if not self.date_str:
            self._tward = pd.DataFrame()
        else:
            self._tward = _load_tward_for_date(
                str(self.paths["raw_dir"]), self.date_str,
            )
        return self._tward

    def get_locus_meta(self) -> pd.DataFrame:
        if self._locus_meta is not None:
            return self._locus_meta
        locus_csv = self.paths.get("locus_v2_csv") or self.paths.get("locus_csv")
        if locus_csv and Path(locus_csv).exists():
            self._locus_meta = _load_locus_meta(str(locus_csv))
        else:
            self._locus_meta = pd.DataFrame()
        return self._locus_meta
