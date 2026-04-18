"""
DeepCon Pipeline CLI — 단일 진입점 (Upgrade v3 T-04)
====================================================
기존 3개 ad-hoc 스크립트를 대체한다:
  - reprocess_y1_correct.py (루트, archive 이동 예정)
  - reprocess_batch.py       (루트, archive 이동 예정)
  - scripts/reprocess_all.py

서브커맨드
  reprocess : raw → processed. --from/--to, --date, --missing-only,
              --incremental, --force, --dry-run 플래그 지원
  verify    : meta.json / schema / validation.error 검증
  clean     : .bak / 오래된 로그 정리
  index     : summary_index.json 재빌드

공통 옵션
  --sector {Y1_SKHynix,M15X_SKHynix}   (SECTOR_REGISTRY에서 자동 제공)
  --log-dir PATH                       (기본: logs/cli)

예시
  python -m src.pipeline.cli reprocess --sector Y1_SKHynix --from 20260301 --to 20260409
  python -m src.pipeline.cli reprocess --sector Y1_SKHynix --date 20260409 --force
  python -m src.pipeline.cli reprocess --sector Y1_SKHynix --missing-only
  python -m src.pipeline.cli reprocess --sector Y1_SKHynix --incremental
  python -m src.pipeline.cli verify    --sector Y1_SKHynix --date 20260409
  python -m src.pipeline.cli clean     --sector Y1_SKHynix --target bak --keep-days 7
  python -m src.pipeline.cli index     rebuild --sector Y1_SKHynix

증분(incremental) 판정 조건 — `_needs_reprocess()`
  (A) meta.json 부재
  (B) meta.schema_versions 가 config.CACHE_SCHEMA_VERSION 와 불일치
  (C) meta.validation.error 존재
  (D) raw 파일 mtime > meta.processed_at
  (그 외 → 스킵)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import unittest.mock as _umock
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Streamlit mock — CLI 환경에서 @st.cache_data 우회
# ─────────────────────────────────────────────────────────────────────
# NOTE: config 로드 전에 설치해야 함 (streamlit_secrets 접근 경로 존재 시)
class _MockCacheData:
    def __call__(self, func=None, *args, **kwargs):
        if func is not None and callable(func):
            func.clear = lambda: None
            return func
        def decorator(f):
            f.clear = lambda: None
            return f
        return decorator
    def clear(self):
        pass

_mock_st = _umock.MagicMock()
_mock_st.cache_data    = _MockCacheData()
_mock_st.cache_resource = _MockCacheData()
# secrets는 빈 dict 역할
class _SecretsShim(dict):
    def get(self, k, d=None): return os.environ.get(k, d)
_mock_st.secrets = _SecretsShim()
# 이미 streamlit 가 로드되어 있으면 건드리지 않음 (테스트/일부 환경)
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _mock_st

# ─────────────────────────────────────────────────────────────────────
# 프로젝트 루트 path 추가 (python -m src.pipeline.cli 실행 시)
# ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────
# lazy imports — streamlit mock 설치 이후에 로드
# ─────────────────────────────────────────────────────────────────────
import config as cfg  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# 로깅
# ═══════════════════════════════════════════════════════════════════
logger = logging.getLogger("deepcon.cli")


def _setup_logging(command: str, log_dir: Path, verbose: bool = False) -> Path:
    """
    stdout + 파일 동시 로그.

    반환: 로그 파일 경로
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{command}_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-7s  %(message)s"
    datefmt = "%H:%M:%S"

    # root logger 재설정 (기존 streamlit 로거가 섞일 수 있음)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(logging.Formatter(fmt, datefmt))
    stream_h.setLevel(level)
    root.addHandler(stream_h)

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setFormatter(logging.Formatter(fmt, datefmt))
    file_h.setLevel(logging.DEBUG)   # 파일엔 항상 DEBUG
    root.addHandler(file_h)

    logger.info("로그 파일: %s", log_path)
    return log_path


# ═══════════════════════════════════════════════════════════════════
# 공용 헬퍼 — 날짜 / 해시 / raw mtime
# ═══════════════════════════════════════════════════════════════════

def _valid_date(s: str) -> str:
    try:
        datetime.strptime(s, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"날짜 형식 YYYYMMDD 필요: {s!r}")
    return s


def _date_range(date_from: str, date_to: str) -> list[str]:
    """
    YYYYMMDD 범위 전개. from <= to 보장.
    raw 디렉토리에 없는 날짜도 포함 → `_needs_reprocess` 가 판단.
    """
    from datetime import timedelta
    d0 = datetime.strptime(date_from, "%Y%m%d").date()
    d1 = datetime.strptime(date_to,   "%Y%m%d").date()
    if d0 > d1:
        raise ValueError(f"--from({date_from}) > --to({date_to})")
    out = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def _raw_files_for_date(raw_dir: Path, date_str: str, sector_id: str) -> list[Path]:
    """
    해당 date 에 매칭되는 raw CSV 목록.
    날짜별 파일 + 합쳐진 파일(bulk) 모두 허용.
    """
    info = cfg.SECTOR_REGISTRY.get(sector_id, {})
    access_prefix = info.get("access_log_prefix", f"{sector_id}_AccessLog")
    tward_prefix  = info.get("tward_prefix",      f"{sector_id}_TWardData")

    if not raw_dir.exists():
        return []

    matches: list[Path] = []
    for f in raw_dir.glob("*.csv"):
        name = f.name
        # 날짜별 파일
        if f"_{date_str}" in name and (name.startswith(access_prefix) or name.startswith(tward_prefix)):
            matches.append(f)
        # 합쳐진 파일 "..._YYYYMMDD ~ YYYYMMDD.csv" 내부에 date_str 포함 여부는
        # loader.detect_raw_dates() 가 보장 → 여기서는 prefix 매칭만 충분
        elif (name.startswith(access_prefix) or name.startswith(tward_prefix)) and "~" in name:
            matches.append(f)
    return matches


def _raw_fingerprint(raw_files: list[Path]) -> str:
    """
    raw 파일들의 (mtime, size) 조합 SHA256 해시 — 빠른 변경 감지용.
    full content hash는 비쌈 → mtime+size 로 충분.
    """
    h = hashlib.sha256()
    for p in sorted(raw_files):
        try:
            st = p.stat()
            h.update(f"{p.name}|{int(st.st_mtime)}|{st.st_size}\n".encode())
        except FileNotFoundError:
            continue
    return h.hexdigest()[:16]


def _raw_max_mtime(raw_files: list[Path]) -> float:
    return max((p.stat().st_mtime for p in raw_files if p.exists()), default=0.0)


def _parse_iso(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# _needs_reprocess — 증분 판정
# ═══════════════════════════════════════════════════════════════════

def _needs_reprocess(
    date_str: str,
    sector_id: str,
    raw_dir: Path,
) -> tuple[bool, str]:
    """
    재처리가 필요한지 + 사유 문자열 반환.

    판정 우선순위:
      (A) meta.json 부재
      (B) schema_versions 불일치
      (C) validation.error 존재
      (D) raw mtime > processed_at
      → 그 외 False
    """
    paths   = cfg.get_sector_paths(sector_id)
    proc_dir = Path(paths["processed_dir"])
    meta_path = proc_dir / date_str / "meta.json"

    # (A)
    if not meta_path.exists():
        return True, "meta.json 없음"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return True, f"meta.json 파싱 실패({e})"

    # (B)
    current = meta.get("schema_versions")
    if current is None:
        # legacy meta → 이번 cycle에서 재처리 권장 (schema 기록 목적)
        return True, "legacy meta (schema_versions 없음)"
    for key, expected in cfg.CACHE_SCHEMA_VERSION.items():
        if current.get(key) != expected:
            return True, f"schema mismatch: {key} expected={expected} found={current.get(key)}"

    # (C)
    validation = meta.get("validation", {}) or {}
    if validation.get("error"):
        return True, f"validation.error: {validation['error']!s:.80s}"

    # (D)
    raw_files = _raw_files_for_date(raw_dir, date_str, sector_id)
    if not raw_files:
        # raw 자체 없음 — 기존 처리 유지. 재처리 대상 아님.
        return False, "raw 파일 없음 (skip)"
    raw_mtime  = _raw_max_mtime(raw_files)
    proc_mtime = _parse_iso(meta.get("processed_at"))
    if raw_mtime > proc_mtime + 1.0:   # 1초 여유 (저장 레이스)
        return True, f"raw mtime 신규 (raw={raw_mtime:.0f} > proc={proc_mtime:.0f})"

    return False, "up-to-date"


# ═══════════════════════════════════════════════════════════════════
# 공용 — Gateway / spot_map 한 번만 로드
# ═══════════════════════════════════════════════════════════════════

class _SectorResources:
    """sector별 spot_map + gateway_index — 여러 날짜 처리 시 재사용."""

    def __init__(self, sector_id: str):
        from src.spatial.loader import load_spot_name_map
        self.sector_id = sector_id
        self.paths     = cfg.get_sector_paths(sector_id)
        self.spot_map  = load_spot_name_map(sector_id)
        self.gateway_index = None

        if cfg.LOCUS_VERSION == "v2":
            try:
                from src.pipeline.sward_mapper import GatewayIndex
                gw_csv = self.paths.get("gateway_csv")
                if gw_csv and Path(gw_csv).exists():
                    self.gateway_index = GatewayIndex.from_csv(gw_csv)
                    logger.info("GatewayIndex 초기화: %d개 Gateway",
                                len(self.gateway_index._gw_meta))
                else:
                    logger.warning("gateway_csv 없음: %s", gw_csv)
            except Exception as e:
                logger.warning("GatewayIndex 초기화 실패, v1 fallback: %s", e)


# ═══════════════════════════════════════════════════════════════════
# reprocess 서브커맨드
# ═══════════════════════════════════════════════════════════════════

def _reprocess_one(
    date_str: str,
    resources: _SectorResources,
    raw_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    단일 날짜 재처리. (성공 여부, 메시지) 반환.
    """
    from src.pipeline.loader import load_daily_data
    from src.pipeline.processor import process_daily
    from src.pipeline.cache_manager import save_daily_results

    if dry_run:
        return True, "dry-run"

    t0 = time.time()
    # 1. Raw 로드
    journey_df, access_df, meta = load_daily_data(raw_dir, date_str)
    if journey_df.empty and access_df.empty:
        return False, "데이터 없음 (journey/access 모두 empty)"

    # 2. 처리
    def _step(pct: int, msg: str):
        logger.info("  [%3d%%] %s", pct, msg)

    result = process_daily(
        journey_df=journey_df,
        access_df=access_df,
        spot_map=resources.spot_map,
        date_str=date_str,
        progress_callback=_step,
        gateway_index=resources.gateway_index,
    )

    # 3. stats 병합 후 저장
    meta.update(result.pop("stats", {}))
    save_daily_results(result, meta, date_str, resources.sector_id)

    elapsed = time.time() - t0
    return True, (
        f"workers={meta.get('total_workers_access','?')} "
        f"journey={len(result.get('journey', []))} "
        f"({elapsed:.1f}s)"
    )


def cmd_reprocess(args: argparse.Namespace) -> int:
    sector_id = args.sector
    raw_dir   = Path(cfg.get_sector_paths(sector_id)["raw_dir"])

    # 대상 날짜 결정
    if args.date:
        candidates = [args.date]
    elif args.date_from and args.date_to:
        candidates = _date_range(args.date_from, args.date_to)
    elif args.missing_only or args.incremental:
        # raw_dir에 있는 모든 날짜
        try:
            from src.pipeline.loader import detect_raw_dates
            candidates = sorted(detect_raw_dates(raw_dir))
        except Exception as e:
            logger.error("detect_raw_dates 실패: %s", e)
            return 2
    else:
        logger.error(
            "대상 날짜 지정 필요: --date / --from ~ --to / --missing-only / --incremental"
        )
        return 2

    if not candidates:
        logger.error("대상 날짜 없음 (raw_dir=%s)", raw_dir)
        return 2

    # _needs_reprocess 필터
    if args.force:
        targets = candidates
        logger.info("--force: 필터 없이 %d일 재처리 강행", len(targets))
    else:
        targets = []
        skipped_reasons: dict[str, int] = {}
        for d in candidates:
            need, reason = _needs_reprocess(d, sector_id, raw_dir)
            if need:
                targets.append(d)
                logger.info("[target] %s — %s", d, reason)
            else:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        for reason, cnt in skipped_reasons.items():
            logger.info("[skip]   %d일 — %s", cnt, reason)

    if not targets:
        logger.info("재처리 필요 없음 (%d일 검사)", len(candidates))
        return 0

    logger.info("=" * 60)
    logger.info("재처리 대상: %d일 (sector=%s, dry_run=%s)",
                len(targets), sector_id, args.dry_run)
    logger.info("=" * 60)

    if args.dry_run:
        for d in targets:
            logger.info("[dry-run] %s", d)
        logger.info("dry-run 완료: %d일 처리 대상", len(targets))
        return 0

    # 자원 1회 초기화
    resources = _SectorResources(sector_id)

    ok, fail = 0, []
    t_batch = time.time()
    try:
        from tqdm import tqdm
        it = tqdm(list(enumerate(targets, 1)),
                  desc=f"reprocess[{sector_id}]",
                  total=len(targets))
    except ImportError:
        it = enumerate(targets, 1)

    for idx, date_str in it:
        logger.info("[%d/%d] %s 시작", idx, len(targets), date_str)
        try:
            success, msg = _reprocess_one(date_str, resources, raw_dir,
                                          dry_run=False)
            if success:
                logger.info("[%d/%d] %s OK — %s", idx, len(targets), date_str, msg)
                ok += 1
            else:
                logger.warning("[%d/%d] %s SKIP — %s", idx, len(targets), date_str, msg)
                fail.append((date_str, msg))
        except Exception as e:
            logger.error("[%d/%d] %s FAIL — %s\n%s", idx, len(targets), date_str, e,
                         traceback.format_exc())
            fail.append((date_str, str(e)))

    elapsed = time.time() - t_batch
    logger.info("=" * 60)
    logger.info("재처리 완료: 성공 %d / 실패 %d (총 %.1f분)", ok, len(fail), elapsed / 60)
    for d, e in fail:
        logger.warning("  %s: %s", d, e)

    # summary_index 증분 갱신 (save 시점에 각 날짜가 자체 갱신하므로 여기선 생략)
    # 필요 시 `index rebuild` 서브커맨드 별도 실행

    return 0 if not fail else 1


# ═══════════════════════════════════════════════════════════════════
# verify 서브커맨드
# ═══════════════════════════════════════════════════════════════════

def cmd_verify(args: argparse.Namespace) -> int:
    """
    재처리 없이 meta.json 의 건강 상태만 점검.
    - schema_versions 일치 여부
    - validation.error 유무
    - 필수 parquet 존재 여부
    """
    from src.pipeline.cache_manager import (
        validate_schema, SchemaVersionMismatch, is_processed,
    )

    sector_id = args.sector
    paths = cfg.get_sector_paths(sector_id)
    proc_dir = Path(paths["processed_dir"])

    if args.date:
        candidates = [args.date]
    elif args.date_from and args.date_to:
        candidates = _date_range(args.date_from, args.date_to)
    else:
        # 모든 processed 날짜
        candidates = sorted(
            d.name for d in proc_dir.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        ) if proc_dir.exists() else []

    if not candidates:
        logger.error("검증 대상 날짜 없음 (proc_dir=%s)", proc_dir)
        return 2

    logger.info("=" * 60)
    logger.info("verify: %d일 (sector=%s)", len(candidates), sector_id)
    logger.info("=" * 60)

    counts = {"ok": 0, "missing": 0, "schema": 0, "validation_err": 0, "legacy": 0}
    issues: list[tuple[str, str]] = []

    for d in candidates:
        # 파일 존재
        if not is_processed(d, sector_id):
            counts["missing"] += 1
            issues.append((d, "missing parquet/meta"))
            continue

        # schema
        try:
            validate_schema(d, sector_id, strict_legacy=False)
        except SchemaVersionMismatch as e:
            counts["schema"] += 1
            issues.append((d, f"schema: {e}"))
            continue

        # meta 읽기 (validation.error 확인 + legacy 여부)
        meta_path = proc_dir / d / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        is_legacy = ("schema_versions" not in meta)

        v = meta.get("validation", {}) or {}
        if v.get("error"):
            counts["validation_err"] += 1
            issues.append((d, f"validation.error: {v['error']!s:.80s}"))
            continue

        if is_legacy:
            counts["legacy"] += 1
            issues.append((d, "legacy meta (schema_versions 없음)"))
            continue

        counts["ok"] += 1

    logger.info("-" * 60)
    logger.info("OK: %(ok)d | missing: %(missing)d | schema_mismatch: %(schema)d | "
                "legacy: %(legacy)d | validation.error: %(validation_err)d",
                counts)
    if issues:
        logger.info("문제 목록:")
        for d, msg in issues[:20]:
            logger.info("  %s: %s", d, msg)
        if len(issues) > 20:
            logger.info("  ... +%d 추가 (로그 파일 전체 기록)", len(issues) - 20)

    # 반환 코드: 이슈 있으면 1
    return 0 if (counts["missing"] + counts["schema"] + counts["validation_err"]) == 0 else 1


# ═══════════════════════════════════════════════════════════════════
# clean 서브커맨드
# ═══════════════════════════════════════════════════════════════════

def cmd_clean(args: argparse.Namespace) -> int:
    """
    .bak 파일 / 오래된 로그 / orphaned cache 정리.
    """
    sector_id = args.sector
    proc_dir  = Path(cfg.get_sector_paths(sector_id)["processed_dir"])
    cutoff    = time.time() - args.keep_days * 86400

    removed = 0
    freed   = 0

    def _del(p: Path):
        nonlocal removed, freed
        try:
            sz = p.stat().st_size
            if args.dry_run:
                logger.info("[dry-run] rm %s (%.1f MB)", p, sz / 1024 / 1024)
            else:
                p.unlink()
                logger.info("rm %s (%.1f MB)", p, sz / 1024 / 1024)
            removed += 1
            freed += sz
        except Exception as e:
            logger.warning("rm 실패 %s: %s", p, e)

    if args.target in ("bak", "all"):
        for f in proc_dir.rglob("*.bak"):
            if f.stat().st_mtime < cutoff:
                _del(f)

    if args.target in ("log", "all"):
        log_dir = Path("logs") / "cli"
        if log_dir.exists():
            for f in log_dir.glob("*.log"):
                if f.stat().st_mtime < cutoff:
                    _del(f)

    logger.info("=" * 60)
    logger.info("clean 완료: 삭제 %d개 / 회수 %.1f MB (dry_run=%s)",
                removed, freed / 1024 / 1024, args.dry_run)
    return 0


# ═══════════════════════════════════════════════════════════════════
# index 서브커맨드
# ═══════════════════════════════════════════════════════════════════

def cmd_index(args: argparse.Namespace) -> int:
    if args.action != "rebuild":
        logger.error("지원 액션: rebuild")
        return 2

    from src.pipeline.summary_index import build_summary_index

    sector_id = args.sector
    logger.info("summary_index 재빌드: sector=%s", sector_id)
    t0 = time.time()
    idx = build_summary_index(sector_id)
    elapsed = time.time() - t0
    dates = idx.get("dates", {})
    logger.info("재빌드 완료: %d일 (%.1fs)", len(dates), elapsed)
    return 0


# ═══════════════════════════════════════════════════════════════════
# main / argparse
# ═══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    sectors = list(cfg.SECTOR_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.cli",
        description="DeepCon Pipeline CLI — raw→processed 파이프라인 단일 진입점",
    )
    parser.add_argument("--log-dir", default="logs/cli",
                        help="로그 저장 디렉토리 (기본: logs/cli)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG 레벨 로그 출력")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── reprocess ────────────────────────────────────────────────
    rp = sub.add_parser("reprocess", help="raw→processed 재처리")
    rp.add_argument("--sector", required=True, choices=sectors)
    grp = rp.add_mutually_exclusive_group()
    grp.add_argument("--date", type=_valid_date, help="단일 날짜 (YYYYMMDD)")
    grp.add_argument("--missing-only", action="store_true",
                     help="meta.json 없는 날짜만")
    grp.add_argument("--incremental", action="store_true",
                     help="증분: raw_hash/schema/validation.error 기반 재처리 판정")
    rp.add_argument("--from", dest="date_from", type=_valid_date,
                    help="시작 날짜 (YYYYMMDD)")
    rp.add_argument("--to",   dest="date_to",   type=_valid_date,
                    help="종료 날짜 (YYYYMMDD)")
    rp.add_argument("--force", action="store_true",
                    help="_needs_reprocess 필터 건너뛰고 강행")
    rp.add_argument("--dry-run", action="store_true",
                    help="실제 처리 없이 대상 날짜만 나열")
    rp.set_defaults(func=cmd_reprocess)

    # ── verify ────────────────────────────────────────────────
    vf = sub.add_parser("verify", help="processed 상태 검증")
    vf.add_argument("--sector", required=True, choices=sectors)
    vf.add_argument("--date", type=_valid_date, help="단일 날짜")
    vf.add_argument("--from", dest="date_from", type=_valid_date)
    vf.add_argument("--to",   dest="date_to",   type=_valid_date)
    vf.set_defaults(func=cmd_verify)

    # ── clean ────────────────────────────────────────────────
    cl = sub.add_parser("clean", help=".bak / 로그 정리")
    cl.add_argument("--sector", required=True, choices=sectors)
    cl.add_argument("--target", choices=["bak", "log", "all"], default="bak")
    cl.add_argument("--keep-days", type=int, default=7)
    cl.add_argument("--dry-run", action="store_true")
    cl.set_defaults(func=cmd_clean)

    # ── index ────────────────────────────────────────────────
    ix = sub.add_parser("index", help="summary_index 관리")
    ix.add_argument("action", choices=["rebuild"])
    ix.add_argument("--sector", required=True, choices=sectors)
    ix.set_defaults(func=cmd_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.command, Path(args.log_dir), verbose=args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("사용자 중단 (Ctrl-C)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
