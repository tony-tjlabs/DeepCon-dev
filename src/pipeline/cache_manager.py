"""
Cache Manager — Processed 데이터 저장/로드
============================================
일별 처리 결과를 Parquet로 저장하고,
이미 처리된 날짜를 감지하여 중복 처리를 방지.

파일 구조:
    processed/{sector_id}/{YYYYMMDD}/
        journey.parquet    ← 작업자별 1분 위치
        worker.parquet     ← 작업자별 지표
        space.parquet      ← 공간별 지표
        company.parquet    ← 업체별 지표
        meta.json          ← 처리 메타 정보

모든 public 함수는 sector_id 파라미터를 받는다.
sector_id를 생략하면 기본값(cfg.SECTOR_ID)을 사용한다.

★ Performance (v1.1 → v1.2 M4-T30):
  - TTL 정책 중앙화 (core.cache.policy):
      load_daily_results / load_journey / load_meta_only → DAILY_PARQUET (1h)
      load_multi_day_results                              → MULTI_DAY_AGG (30m)
      get_cache_status / detect_processed_dates           → STATUS (60s)
  - load_meta_only: meta.json만 읽기 (pipeline_tab 요약용)
  - load_multi_day_results: journey 스킵 옵션 (주간 탭 메모리 절약 ~175MB)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

import config as cfg
from core.cache.policy import DAILY_PARQUET, MULTI_DAY_AGG, STATUS

# ★ journey.parquet는 필수가 아님 (Cloud 배포 시 제외됨, 35MB/일)
CACHE_FILES_REQUIRED = ["worker.parquet", "space.parquet",
                        "company.parquet", "meta.json"]
CACHE_FILES = CACHE_FILES_REQUIRED + ["journey.parquet"]  # 전체 목록 (로컬용)


# ═══════════════════════════════════════════════════════════════════
# 스키마 버전 관리 (Upgrade v3 T-06)
# ═══════════════════════════════════════════════════════════════════

class SchemaVersionMismatch(Exception):
    """
    meta.json 의 schema_versions 가 config.CACHE_SCHEMA_VERSION 과 불일치.

    대시보드 쪽에서 잡아서 사용자에게 "재처리" CTA 표시하는 용도.
    """
    def __init__(self, sector: str, date: str, file: str,
                 expected: int, found: int | None):
        self.sector = sector
        self.date = date
        self.file = file
        self.expected = expected
        self.found = found
        super().__init__(
            f"schema mismatch: sector={sector} date={date} file={file} "
            f"expected=v{expected} found={found!r}"
        )


def validate_schema(
    date_str: str,
    sector_id: str | None = None,
    *,
    strict_legacy: bool = False,
) -> None:
    """
    meta.json 의 schema_versions 를 config.CACHE_SCHEMA_VERSION 과 대조.

    Args:
        strict_legacy:
            False(기본) = `schema_versions` 키 자체가 없는 legacy meta는 관용 처리 (pass)
            True        = legacy도 불일치로 간주 (엄격 모드; 강제 전체 재처리 시)

    Raises:
        SchemaVersionMismatch: 불일치 시 첫 발견된 파일에 대해 raise
        FileNotFoundError: meta.json 자체가 없을 때
    """
    date_dir = _date_dir(date_str, sector_id)
    meta_path = date_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 없음: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    current = meta.get("schema_versions")
    if current is None:
        if strict_legacy:
            raise SchemaVersionMismatch(
                sector=_resolved_sector(sector_id),
                date=date_str,
                file="(legacy_meta_no_schema_versions)",
                expected=cfg.CACHE_SCHEMA_VERSION.get("meta", 0),
                found=None,
            )
        # legacy meta는 관용 처리
        logger.debug("legacy meta (no schema_versions): %s", meta_path)
        return

    for key, expected in cfg.CACHE_SCHEMA_VERSION.items():
        found = current.get(key)
        if found != expected:
            raise SchemaVersionMismatch(
                sector=_resolved_sector(sector_id),
                date=date_str,
                file=key,
                expected=expected,
                found=found,
            )


# ─── 내부 헬퍼 ────────────────────────────────────────────────────
def _resolved_sector(sector_id: str | None) -> str:
    """sector_id가 None이면 cfg 기본값 반환."""
    return sector_id if sector_id else cfg.SECTOR_ID


def _processed_dir(sector_id: str | None) -> Path:
    """processed 디렉토리 경로. 로컬/Cloud 동일 (리포 내 data/processed/)."""
    return cfg.PROCESSED_DIR / _resolved_sector(sector_id)


def _date_dir(date_str: str, sector_id: str | None) -> Path:
    return _processed_dir(sector_id) / date_str


# ─── 날짜 감지 ────────────────────────────────────────────────────
def is_processed(date_str: str, sector_id: str | None = None) -> bool:
    """해당 날짜의 필수 처리 결과가 존재하는지 확인 (journey 제외)."""
    d = _date_dir(date_str, sector_id)
    return all((d / f).exists() for f in CACHE_FILES_REQUIRED)


def detect_unprocessed_dates(
    raw_dir: Path,
    sector_id: str | None = None,
) -> list[str]:
    """
    raw_dir에서 처리 안 된 날짜 목록 반환.

    meta.json이 없으면 미처리로 간주 → 재처리.
    """
    try:
        from src.pipeline.loader import detect_raw_dates
        all_dates = detect_raw_dates(raw_dir)
    except ImportError:
        return []  # Cloud 환경: loader 없음
    return [d for d in all_dates if not is_processed(d, sector_id)]


@st.cache_data(ttl=STATUS, show_spinner=False)
def detect_processed_dates(sector_id: str | None = None) -> list[str]:
    """처리 완료된 날짜 목록 반환 (오름차순). ★ 10분 캐시 (save 시 무효화)."""
    proc_dir = _processed_dir(sector_id)
    if not proc_dir.exists():
        return []
    dates = []
    for d in sorted(proc_dir.iterdir()):
        if d.is_dir() and re.match(r"\d{8}$", d.name) and is_processed(d.name, sector_id):
            dates.append(d.name)
    return dates


# ─── dtype 다운캐스팅 (메모리 최적화) ────────────────────────────────
def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 타입 다운캐스팅 (메모리 ~42% 절약).

    - float64 -> float32 (EWI, CRE 등 0~1 범위)
    - int64 -> int32 (카운트 값, ID 등)
    - object (문자열) -> category (반복 값이 많은 경우)
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            # ID 및 카운트 값은 int32로 충분 (2B 범위)
            if df[col].max() < 2_000_000_000 and df[col].min() > -2_000_000_000:
                df[col] = df[col].astype("int32")
        elif dtype == "object":
            # 반복 값이 많으면 category로 변환 (locus_id, company_name 등)
            n_unique = df[col].nunique()
            n_total = len(df)
            if n_unique < n_total * 0.5:  # 50% 미만 고유값이면 category
                df[col] = df[col].astype("category")
    return df


# ─── 저장 ─────────────────────────────────────────────────────────
def save_daily_results(
    results: dict[str, pd.DataFrame],
    meta: dict,
    date_str: str,
    sector_id: str | None = None,
    use_zstd: bool = True,
    downcast: bool = True,
) -> Path:
    """
    처리 결과를 Parquet + JSON으로 저장.

    meta.json을 마지막에 저장 -> 완료 표시 패턴.
    저장 후 관련 캐시 무효화.

    Args:
        results: {"journey", "worker", "space", "company"} DataFrames
        meta: 메타데이터 딕셔너리
        date_str: 날짜 문자열 (YYYYMMDD)
        sector_id: Sector ID
        use_zstd: True면 zstd 압축 (디스크 ~42% 절약, 로드 약간 느림)
        downcast: True면 dtype 다운캐스팅 (메모리 ~42% 절약)

    Returns:
        저장된 디렉토리 Path
    """
    date_dir = _date_dir(date_str, sector_id)
    date_dir.mkdir(parents=True, exist_ok=True)

    compression = "zstd" if use_zstd else "snappy"

    for name in ["journey", "worker", "space", "company", "coverage"]:
        df = results.get(name)
        if df is not None and not df.empty:
            if downcast:
                df = _downcast_dtypes(df)
            df.to_parquet(
                date_dir / f"{name}.parquet",
                index=False,
                engine="pyarrow",
                compression=compression,
            )

    # meta.json 마지막 저장 (완료 표시)
    meta["processed_at"]    = datetime.now().isoformat()
    meta["cache_version"]   = cfg.CACHE_VERSION
    # ★ v3 T-06: 중앙 관리 스키마 버전 기록 — 불일치 시 자동 재처리 트리거
    meta["schema_versions"] = dict(cfg.CACHE_SCHEMA_VERSION)
    with open(date_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # ★ 저장 후 캐시 무효화 (새 데이터 반영)
    detect_processed_dates.clear()
    load_daily_results.clear()
    load_meta_only.clear()
    load_multi_day_results.clear()
    get_cache_status.clear()

    # ★ Summary Index 증분 업데이트 (meta.json 저장 직후)
    try:
        from src.pipeline.summary_index import update_date_entry
        update_date_entry(date_str, {**results, "meta": meta}, sector_id)
    except Exception as e:
        logger.debug(f"summary_index 업데이트 실패: {e}")

    return date_dir


# ─── 로드 ─────────────────────────────────────────────────────────
@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def load_daily_results(
    date_str: str,
    sector_id: str | None = None,
    include_journey: bool = False,
) -> dict:
    """
    처리된 하루치 결과 로드. ★ 5분 캐시.

    반환: { "worker", "space", "company", "meta" }
    include_journey=True 시 "journey" 도 포함 (5M행, ~35MB — 느림).

    ★ 웹 배포 성능: include_journey=False(기본)이면 worker+space+company+meta만
    로드 → ~0.3초. journey 필요 시 load_journey()를 별도 호출 권장.
    """
    date_dir = _date_dir(date_str, sector_id)
    if not is_processed(date_str, sector_id):
        raise FileNotFoundError(f"{date_str} 처리 결과 없음 (sector={sector_id})")

    results = {}
    load_targets = ["worker", "space", "company"]
    if include_journey:
        load_targets = ["journey"] + load_targets

    for name in load_targets:
        p = date_dir / f"{name}.parquet"
        results[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    with open(date_dir / "meta.json", encoding="utf-8") as f:
        results["meta"] = json.load(f)

    return results


# ─── Journey 컬럼 프루닝 상수 ──────────────────────────────────────
# 핵심 컬럼만 로드하면 메모리 ~40% 절약 (43MB → ~25MB/일)
JOURNEY_CORE_COLUMNS = [
    "user_no", "timestamp", "locus_id", "locus_token",
    "active_ratio", "is_work_hour",
]

# 탭별 필요 컬럼
JOURNEY_COLUMNS_BY_TAB = {
    "summary": ["user_no", "locus_id", "active_ratio", "is_work_hour"],
    "safety": ["user_no", "timestamp", "locus_id", "is_work_hour"],
    "congestion": ["user_no", "timestamp", "locus_id"],
    "deep_space": ["user_no", "timestamp", "locus_id"],
    "pattern": JOURNEY_CORE_COLUMNS,
}


@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def load_journey(
    date_str: str,
    sector_id: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    journey.parquet 로드. ★ 5분 캐시 + 컬럼 프루닝 옵션.

    Cloud 환경에서 journey.parquet 없으면 journey_slim.parquet(6컬럼)로
    자동 fallback. 반환 DataFrame에 `_is_slim=True` attr 표시.

    Args:
        date_str: 날짜 문자열 (YYYYMMDD)
        sector_id: Sector ID (None이면 기본값 사용)
        columns: 로드할 컬럼 목록 (None이면 전체 로드)

    Returns:
        DataFrame (slim fallback 시 6컬럼 subset)
    """
    date_dir = _date_dir(date_str, sector_id)
    p      = date_dir / "journey.parquet"
    p_slim = date_dir / "journey_slim.parquet"

    # ── 파일 선택: full → slim 순으로 시도 ───────────────────────
    if p.exists():
        target, is_slim = p, False
    elif p_slim.exists():
        target, is_slim = p_slim, True
        logger.info(f"journey_slim fallback: {date_str} ({sector_id})")
    else:
        return pd.DataFrame()

    try:
        import pyarrow.parquet as pq
        schema    = pq.read_schema(target)
        available = set(schema.names)

        if columns:
            valid_cols = [c for c in columns if c in available]
            df = pd.read_parquet(target, columns=valid_cols) if valid_cols else pd.read_parquet(target)
        else:
            df = pd.read_parquet(target)
    except Exception as e:
        logger.debug(f"스키마 읽기 실패, 전체 로드로 fallback: {e}")
        df = pd.read_parquet(target)

    # slim 여부를 DataFrame 속성으로 전달 (탭에서 안내 메시지 표시용)
    df.attrs["_is_slim"] = is_slim
    return df


def is_journey_slim(df: pd.DataFrame) -> bool:
    """load_journey() 반환값이 slim fallback 버전인지 확인."""
    return bool(df.attrs.get("_is_slim", False))


@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def load_meta_only(
    date_str: str,
    sector_id: str | None = None,
) -> dict | None:
    """
    meta.json만 읽기 (Parquet 로드 없음). ★ pipeline_tab 요약용.

    load_daily_results()는 4개 Parquet + JSON을 모두 읽지만,
    처리 현황 테이블에는 meta만 필요하므로 이 함수를 사용.
    """
    date_dir = _date_dir(date_str, sector_id)
    meta_path = date_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"meta.json 읽기 실패 ({meta_path}): {e}")
        return None


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def load_multi_day_results(
    _date_tuple: tuple[str, ...],
    sector_id: str | None = None,
    skip_journey: bool = False,
) -> dict[str, pd.DataFrame | list]:
    """
    여러 날짜의 결과 합산 (주간 리포트용). ★ 5분 캐시.

    Args:
        _date_tuple: 날짜 튜플 (st.cache_data 해시용, list 대신 tuple)
        sector_id: Sector ID
        skip_journey: True면 journey.parquet 스킵 (~35MB/일 메모리 절약)

    반환: { "worker", "space", "company", "metas" [, "journey"] }
    """
    worker_dfs  = []
    space_dfs   = []
    company_dfs = []
    journey_dfs = []
    metas       = []

    for date_str in _date_tuple:
        if not is_processed(date_str, sector_id):
            continue

        if skip_journey:
            # ★ Perf: journey.parquet(~35MB/일) 스킵 — 직접 필요한 파일만 로드
            date_dir = _date_dir(date_str, sector_id)
            for df_list, name in [
                (worker_dfs, "worker"),
                (space_dfs, "space"),
                (company_dfs, "company"),
            ]:
                p = date_dir / f"{name}.parquet"
                if p.exists():
                    df_list.append(pd.read_parquet(p).assign(date=date_str))

            meta_path = date_dir / "meta.json"
            if meta_path.exists():
                import json as _json
                with open(meta_path, encoding="utf-8") as f:
                    metas.append(_json.load(f))
        else:
            r = load_daily_results(date_str, sector_id)
            for df_list, key in [
                (worker_dfs, "worker"),
                (space_dfs, "space"),
                (company_dfs, "company"),
            ]:
                df = r.get(key, pd.DataFrame())
                if not df.empty:
                    df_list.append(df.assign(date=date_str))

            jdf = r.get("journey", pd.DataFrame())
            if not jdf.empty:
                journey_dfs.append(jdf.assign(date=date_str))

            if r.get("meta"):
                metas.append(r["meta"])

    result = {
        "worker":  pd.concat(worker_dfs,  ignore_index=True) if worker_dfs  else pd.DataFrame(),
        "space":   pd.concat(space_dfs,   ignore_index=True) if space_dfs   else pd.DataFrame(),
        "company": pd.concat(company_dfs, ignore_index=True) if company_dfs else pd.DataFrame(),
        "metas":   metas,
    }
    if not skip_journey:
        result["journey"] = pd.concat(journey_dfs, ignore_index=True) if journey_dfs else pd.DataFrame()
    return result


# ─── 처리 현황 요약 ────────────────────────────────────────────────
@st.cache_data(ttl=STATUS, show_spinner=False)
def get_cache_status(
    _raw_dir: str,
    sector_id: str | None = None,
) -> dict:
    """
    파이프라인 탭용 캐시 상태 요약. ★ 60초 캐시.

    주의: raw_dir은 str로 받음 (Path → st.cache_data 해시 불가).

    processed_dates  : 캐시 폴더에 존재하는 모든 처리완료 날짜
    processed_in_raw : raw가 있고 처리도 완료된 날짜 (파이프라인 정상 완료)
    orphaned_dates   : raw는 삭제됐지만 캐시가 남아 있는 날짜
    unprocessed_dates: raw는 있지만 아직 처리 안 된 날짜
    """
    raw_dir   = Path(_raw_dir)
    try:
        from src.pipeline.loader import detect_raw_dates
        all_raw = detect_raw_dates(raw_dir)
    except ImportError:
        all_raw = []  # Cloud 환경: loader 없음
    processed = detect_processed_dates(sector_id)
    raw_set   = set(all_raw)
    proc_set  = set(processed)

    processed_in_raw  = sorted(raw_set & proc_set)
    orphaned_dates    = sorted(proc_set - raw_set)
    unprocessed_dates = sorted(raw_set - proc_set)

    return {
        "raw_dates":          all_raw,
        "processed_dates":    processed,
        "processed_in_raw":   processed_in_raw,
        "orphaned_dates":     orphaned_dates,
        "unprocessed_dates":  unprocessed_dates,
        "total_raw":          len(all_raw),
        "total_processed":    len(processed_in_raw),
        "total_unprocessed":  len(unprocessed_dates),
        "total_orphaned":     len(orphaned_dates),
    }
