"""
Prediction Journal — 예측 정확도 추적 (M4-T33 월별 분할 + 90일 retention)
==========================================================================
Agentic AI / Deep Space 예측을 저장하고, 다음 날 실제 데이터와 비교하여
예측 정확도를 추적한다.

저장 구조 (M4-T33 이후)
----------------------
    data/index/{sector_id}/
        prediction_journal_YYYY-MM.jsonl      ← 활성 월별 파일 (append-only)
        prediction_journal_archive_YYYY-MM.jsonl.gz
                                              ← 90일 초과 시 아카이브
        prediction_journal.json (legacy)       ← 마이그레이션 전 파일 (자동 읽기 병합)
        prediction_journal.json.bak.gz         ← 마이그레이션 시 생성되는 백업

각 .jsonl 줄은 다음 두 타입 중 하나:
    {"type": "entry",    "date": "YYYYMMDD", "data": {...}}
    {"type": "accuracy", "date": "YYYYMMDD", "data": {...}}

메모리 표현(기존 코드 호환 형태):
    {
      "sector_id": "...",
      "entries":      { "YYYYMMDD": {...}, ... },
      "accuracy_log": { "YYYYMMDD": {...}, ... },
    }

API
---
- save_predictions / evaluate_accuracy / get_accuracy_history — 기존과 동일 시그니처
- _load_journal / _save_journal 내부 구현만 월별 jsonl 로 교체
- load_recent_months(sector_id, months=None) — M4-T33 신규 read helper
- archive_old_files(sector_id, retention_days=90) — retention 정책
"""
from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.spatial.locus_group import build_locus_group_map, get_group, evaluate_group_accuracy

logger = logging.getLogger(__name__)

# ─── 설정 ────────────────────────────────────────────────────────
_LEGACY_JOURNAL_FILENAME = "prediction_journal.json"
_MONTHLY_PREFIX          = "prediction_journal_"       # + YYYY-MM.jsonl
_ARCHIVE_PREFIX          = "prediction_journal_archive_"  # + YYYY-MM.jsonl.gz

# 오래된 entries 보존 기준 — 기존 로직 유지 (읽기 시점에서 최근 N개만 보관)
MAX_ENTRIES_IN_MEMORY = 30

# 90일 이상 오래된 월 파일은 gzip 아카이브로 이동
DEFAULT_RETENTION_DAYS = 90

# 시간 허용 오차 (분) — 기존 상수 유지
TIME_TOLERANCE_MIN = 15


# ═══════════════════════════════════════════════════════════════════
# 경로 헬퍼
# ═══════════════════════════════════════════════════════════════════

def _index_dir(sector_id: str) -> Path:
    import config as cfg
    path = cfg.INDEX_DIR / sector_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_path(sector_id: str) -> Path:
    return _index_dir(sector_id) / _LEGACY_JOURNAL_FILENAME


def _month_file(sector_id: str, ym: str) -> Path:
    """ym = 'YYYY-MM' → .jsonl 경로."""
    return _index_dir(sector_id) / f"{_MONTHLY_PREFIX}{ym}.jsonl"


def _archive_file(sector_id: str, ym: str) -> Path:
    return _index_dir(sector_id) / f"{_ARCHIVE_PREFIX}{ym}.jsonl.gz"


def _journal_path(sector_id: str) -> Path:
    """legacy 경로 — backward compatibility (일부 호출자가 직접 참조할 수 있음)."""
    return _legacy_path(sector_id)


def _date_to_ym(date_str: str) -> str | None:
    """'20260315' → '2026-03'."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
        return d.strftime("%Y-%m")
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════
# 월별 jsonl I/O
# ═══════════════════════════════════════════════════════════════════

def _iter_month_files(sector_id: str) -> list[Path]:
    """sector 의 월별 jsonl 파일 목록 (최신순)."""
    d = _index_dir(sector_id)
    files = sorted(
        d.glob(f"{_MONTHLY_PREFIX}*.jsonl"),
        reverse=True,
    )
    return files


def _read_month_file(path: Path) -> tuple[dict, dict]:
    """
    단일 월 jsonl → (entries, accuracy_log).
    깨진 줄은 경고 후 skip.
    """
    entries: dict[str, dict] = {}
    accuracy: dict[str, dict] = {}
    if not path.exists():
        return entries, accuracy
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("%s:%d JSON parse error (%s) — skip", path.name, lineno, e)
                    continue
                rec_type = rec.get("type")
                date = rec.get("date")
                data = rec.get("data") or {}
                if not date:
                    continue
                # append-only 구조에서 같은 날짜 재저장 시 마지막이 유효
                if rec_type == "entry":
                    entries[date] = data
                elif rec_type == "accuracy":
                    accuracy[date] = data
    except OSError as e:
        logger.warning(f"Month file read failed {path}: {e}")
    return entries, accuracy


def _append_records(path: Path, records: list[dict]) -> None:
    """월 파일에 jsonl append."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _rewrite_month_file(path: Path, entries: dict, accuracy: dict) -> None:
    """월 파일 전체 재작성 (같은 날짜 여러 번 append 누적 시 압축용)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for date in sorted(entries.keys()):
            f.write(json.dumps({"type": "entry", "date": date, "data": entries[date]},
                               ensure_ascii=False) + "\n")
        for date in sorted(accuracy.keys()):
            f.write(json.dumps({"type": "accuracy", "date": date, "data": accuracy[date]},
                               ensure_ascii=False) + "\n")
    tmp.replace(path)


# ═══════════════════════════════════════════════════════════════════
# Legacy JSON 자동 병합
# ═══════════════════════════════════════════════════════════════════

def _read_legacy(sector_id: str) -> dict:
    """prediction_journal.json (구버전) 로드. 없으면 빈 dict."""
    p = _legacy_path(sector_id)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Legacy journal read failed {p}: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════
# Public: load / save (내부 전용)
# ═══════════════════════════════════════════════════════════════════

def _load_journal(sector_id: str) -> dict:
    """
    Journal 로드 — 최근 N개월 jsonl + legacy JSON 병합.

    Returns dict in 기존 호환 형태:
        {"sector_id": ..., "entries": {...}, "accuracy_log": {...}}
    """
    entries: dict[str, dict] = {}
    accuracy: dict[str, dict] = {}

    # 1) 최근 월 파일들 (최신 몇 개월만 — MAX_ENTRIES_IN_MEMORY 기준)
    month_files = _iter_month_files(sector_id)
    # 여유있게 최근 6개월까지 읽어도 충분히 작음 (~수 MB 수준)
    for mf in month_files[:6]:
        e, a = _read_month_file(mf)
        # 기존 값이 있으면 최신 month 우선 (월별 파일은 최신순)
        for date, data in e.items():
            entries.setdefault(date, data)
        for date, data in a.items():
            accuracy.setdefault(date, data)

    # 2) legacy JSON 병합 (아직 마이그레이션되지 않았을 수 있음)
    legacy = _read_legacy(sector_id)
    for date, data in (legacy.get("entries") or {}).items():
        entries.setdefault(date, data)
    for date, data in (legacy.get("accuracy_log") or {}).items():
        accuracy.setdefault(date, data)

    # 3) 메모리 상한 (최근 N개만)
    if len(entries) > MAX_ENTRIES_IN_MEMORY:
        keep = sorted(entries.keys())[-MAX_ENTRIES_IN_MEMORY:]
        entries = {k: entries[k] for k in keep}

    return {
        "sector_id":    sector_id,
        "entries":      entries,
        "accuracy_log": accuracy,
    }


def _save_journal(sector_id: str, journal: dict) -> None:
    """
    Journal 저장 — 월별 jsonl 로 분할 append.

    journal 은 _load_journal 가 반환한 dict (또는 그 copy).
    각 날짜별로 해당 월 파일을 찾아 append (중복 날짜는 마지막 레코드가 유효).

    Side effect: 90일 retention 정책 적용 — 오래된 월 파일 gzip 아카이브.
    """
    entries = journal.get("entries", {}) or {}
    accuracy = journal.get("accuracy_log", {}) or {}

    # 월별 버킷팅
    by_month_entries: dict[str, dict] = {}
    by_month_accuracy: dict[str, dict] = {}

    for date, data in entries.items():
        ym = _date_to_ym(date)
        if ym is None:
            logger.debug("Skip invalid date %r in entries", date)
            continue
        by_month_entries.setdefault(ym, {})[date] = data

    for date, data in accuracy.items():
        ym = _date_to_ym(date)
        if ym is None:
            logger.debug("Skip invalid date %r in accuracy", date)
            continue
        by_month_accuracy.setdefault(ym, {})[date] = data

    # 각 월 파일 전체 재작성 (append-only 단순 + 중복 제거)
    touched_months = set(by_month_entries.keys()) | set(by_month_accuracy.keys())
    for ym in touched_months:
        path = _month_file(sector_id, ym)
        # 기존 파일 병합 후 재작성 (다른 sector 의 동시 append 는 없음 가정)
        existing_e, existing_a = _read_month_file(path)
        existing_e.update(by_month_entries.get(ym, {}))
        existing_a.update(by_month_accuracy.get(ym, {}))
        _rewrite_month_file(path, existing_e, existing_a)

    # retention 정책
    try:
        archive_old_files(sector_id, retention_days=DEFAULT_RETENTION_DAYS)
    except Exception as e:
        logger.warning(f"Retention 정책 실패 (무시): {e}")


# ═══════════════════════════════════════════════════════════════════
# Retention 정책
# ═══════════════════════════════════════════════════════════════════

def archive_old_files(sector_id: str, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """
    `retention_days` 보다 오래된 월 파일을 gzip 아카이브로 이동.

    판정: 월 파일 이름의 YYYY-MM > today - retention_days 범위 밖이면 아카이브.

    Returns:
        아카이브된 파일 수
    """
    d = _index_dir(sector_id)
    cutoff = datetime.now() - timedelta(days=retention_days)
    # cutoff 가 속한 "월의 1일"보다 이전 월들은 모두 아카이브
    cutoff_ym = cutoff.strftime("%Y-%m")

    archived = 0
    for path in sorted(d.glob(f"{_MONTHLY_PREFIX}*.jsonl")):
        ym = path.stem[len(_MONTHLY_PREFIX):]  # "YYYY-MM"
        if not ym or len(ym) != 7 or ym[4] != "-":
            continue
        if ym >= cutoff_ym:
            continue  # 아직 보존 기간 내
        # 이미 아카이브가 있으면 skip (덮어쓰기 방지)
        archive_path = _archive_file(sector_id, ym)
        if archive_path.exists():
            logger.debug("Archive already exists: %s — skip", archive_path.name)
            continue
        try:
            with open(path, "rb") as src, gzip.open(archive_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            archived += 1
            logger.info("Archived %s → %s", path.name, archive_path.name)
        except OSError as e:
            logger.warning(f"Archive 실패 {path}: {e}")
    return archived


# ═══════════════════════════════════════════════════════════════════
# 예측 저장 (기존 API 유지)
# ═══════════════════════════════════════════════════════════════════

def save_predictions(
    sector_id: str,
    date_str: str,
    predictions: dict[str, list[tuple[str, float]]],
    current_loci: dict[str, str],
    congestion_alerts: list | None = None,
) -> bool:
    """
    당일 예측 결과를 Journal에 저장.

    Args:
        sector_id: Sector ID
        date_str: 분석 날짜 (YYYYMMDD)
        predictions: {user_no: [(locus_id, prob), ...]} — predict_next_batch 결과
        current_loci: {user_no: current_locus_id}
        congestion_alerts: CongestionAlert 리스트 (선택)

    Returns:
        True if saved successfully
    """
    try:
        journal = _load_journal(sector_id)

        # 개인별 Top-3 예측 저장 (user_no → {predicted, prob, current, top3})
        pred_entries: dict = {}
        for user_no, preds in predictions.items():
            if not preds:
                continue
            pred_entries[user_no] = {
                "predicted": preds[0][0],
                "prob":      round(preds[0][1], 4),
                "current":   current_loci.get(user_no, ""),
                "top3":      [(loc, round(p, 4)) for loc, p in preds[:3]],
            }

        # 혼잡도 예측 저장
        cong_entries: dict = {}
        if congestion_alerts:
            for a in congestion_alerts:
                cong_entries[a.locus_id] = {
                    "predicted_count": a.predicted_count,
                    "current_count":   a.current_count,
                    "capacity":        a.capacity,
                    "congestion_pct":  round(a.congestion_pct, 3),
                }

        journal["entries"][date_str] = {
            "saved_at":                datetime.now().isoformat(),
            "total_workers":           len(pred_entries),
            "predictions":             pred_entries,
            "congestion_predictions":  cong_entries,
        }

        _save_journal(sector_id, journal)
        logger.info(f"Prediction journal saved: {date_str}, {len(pred_entries)} workers")
        return True

    except Exception as e:
        logger.warning(f"Prediction journal 저장 실패: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 정확도 평가 (기존 API)
# ═══════════════════════════════════════════════════════════════════

def evaluate_accuracy(
    sector_id: str,
    pred_date_str: str,
    actual_sequences: dict[str, list[str]],
) -> dict | None:
    """
    저장된 예측과 실제 다음날 데이터를 비교하여 정확도를 계산.

    Args:
        sector_id: Sector ID
        pred_date_str: 예측이 저장된 날짜 (YYYYMMDD)
        actual_sequences: 다음날 실제 시퀀스 {user_no: [locus1, locus2, ...]}

    Returns:
        정확도 dict 또는 None (예측 데이터 없음)
    """
    journal = _load_journal(sector_id)
    entry = journal.get("entries", {}).get(pred_date_str)

    if not entry or not entry.get("predictions"):
        return None

    predictions = entry["predictions"]
    total = 0
    top1_correct = 0
    top3_correct = 0

    for user_no, pred_info in predictions.items():
        actual_seq = actual_sequences.get(user_no, [])
        if not actual_seq:
            continue

        actual_first = actual_seq[0]
        total += 1

        if pred_info["predicted"] == actual_first:
            top1_correct += 1

        top3_loci = [loc for loc, _ in pred_info.get("top3", [(pred_info["predicted"], 0)])]
        if actual_first in top3_loci:
            top3_correct += 1

    if total == 0:
        return None

    # 혼잡도 예측 평가 (MAE)
    cong_preds = entry.get("congestion_predictions", {})
    congestion_mae = None
    if cong_preds:
        from collections import Counter
        actual_counts = Counter()
        for seq in actual_sequences.values():
            if seq:
                actual_counts[seq[0]] += 1

        errors = []
        for locus_id, pred_info in cong_preds.items():
            actual_count = actual_counts.get(locus_id, 0)
            predicted = pred_info["predicted_count"]
            errors.append(abs(predicted - actual_count))

        if errors:
            congestion_mae = round(float(np.mean(errors)), 2)

    result = {
        "evaluated_at":      datetime.now().isoformat(),
        "total_evaluated":   total,
        "top1_correct":      top1_correct,
        "top1_accuracy":     round(top1_correct / total, 4),
        "top3_correct":      top3_correct,
        "top3_accuracy":     round(top3_correct / total, 4),
    }
    if congestion_mae is not None:
        result["congestion_mae"] = congestion_mae

    # 그룹(건물_층) 단위 정확도
    group_map = build_locus_group_map(sector_id)
    if group_map:
        group_result = evaluate_group_accuracy(predictions, actual_sequences, group_map)
        result["group_top1"] = group_result["top1"]
        result["group_top3"] = group_result["top3"]
        result["group_n"]    = group_result["n"]

    # accuracy_log 에 저장
    journal.setdefault("accuracy_log", {})[pred_date_str] = result
    _save_journal(sector_id, journal)

    logger.info(
        f"Prediction accuracy [{pred_date_str}]: "
        f"Locus Top-1 {result['top1_accuracy']:.1%}, Top-3 {result['top3_accuracy']:.1%} | "
        f"Group Top-1 {result.get('group_top1', 0):.1%}, Top-3 {result.get('group_top3', 0):.1%} "
        f"(n={total})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# 이력 조회 (기존 API)
# ═══════════════════════════════════════════════════════════════════

def get_accuracy_history(sector_id: str) -> list[dict]:
    """
    전체 정확도 이력 반환 (날짜순).

    Returns:
        [{"date": "20260315", "top1_accuracy": 0.60, ...}, ...]
    """
    journal = _load_journal(sector_id)
    log = journal.get("accuracy_log", {})

    history = []
    for date_str, metrics in sorted(log.items()):
        entry = {"date": date_str}
        entry.update(metrics)
        history.append(entry)
    return history


# ═══════════════════════════════════════════════════════════════════
# 단기 예측 정확도 평가 (기존 유지 — 내부 구현 변경 없음)
# ═══════════════════════════════════════════════════════════════════

def evaluate_intraday_accuracy(
    journey_df,
    model,
    tokenizer,
    horizons_min: list[int] = (30, 60, 120),
    cutoff_hours: list[int] = (9, 10, 11, 12, 13, 14, 15),
    max_workers: int = 2000,
    sector_id: str = "Y1_SKHynix",
) -> dict:
    """같은 날 단기 예측 정확도 평가 (구현 동일)."""
    import pandas as pd
    from src.dashboard.deep_space.helpers import predict_next_batch

    if journey_df is None or journey_df.empty or model is None:
        return {}

    df = journey_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "is_work_hour" in df.columns:
        df = df[df["is_work_hour"]].copy()

    locus_col = "locus_token" if "locus_token" in df.columns else "locus_id"
    df = df.sort_values(["user_no", "timestamp"])

    worker_groups = {}
    for user_no, grp in df.groupby("user_no"):
        ts_list = grp["timestamp"].tolist()
        locus_list = grp[locus_col].tolist()
        if len(ts_list) >= 5:
            worker_groups[user_no] = (ts_list, locus_list)

    if len(worker_groups) > max_workers:
        import random
        keys = random.sample(list(worker_groups.keys()), max_workers)
        worker_groups = {k: worker_groups[k] for k in keys}

    horizon_results = {h: {"top1": 0, "top3": 0, "group_top1": 0, "group_top3": 0, "n": 0}
                       for h in horizons_min}

    group_map = build_locus_group_map(sector_id)

    for cutoff_h in cutoff_hours:
        batch_ids = []
        batch_seqs = []
        batch_actuals = {}

        for user_no, (ts_list, locus_list) in worker_groups.items():
            cutoff_idx = None
            for i, ts in enumerate(ts_list):
                if ts.hour >= cutoff_h:
                    cutoff_idx = i
                    break
            if cutoff_idx is None or cutoff_idx < 3:
                continue

            seq_prefix = locus_list[:cutoff_idx]
            cutoff_time = ts_list[cutoff_idx - 1]

            has_any = False
            for h in horizons_min:
                target_time = cutoff_time + pd.Timedelta(minutes=h)
                tolerance = pd.Timedelta(minutes=TIME_TOLERANCE_MIN)
                window_loci = set()
                best_idx = None
                best_diff = float("inf")
                for j in range(cutoff_idx, len(ts_list)):
                    diff_td = ts_list[j] - target_time
                    diff_sec = abs(diff_td.total_seconds())
                    if abs(diff_td) <= tolerance:
                        window_loci.add(locus_list[j])
                        if diff_sec < best_diff:
                            best_diff = diff_sec
                            best_idx = j
                    if ts_list[j] > target_time + tolerance:
                        break
                if window_loci:
                    if h not in batch_actuals:
                        batch_actuals[h] = {}
                    batch_actuals[h][len(batch_seqs)] = (locus_list[best_idx], window_loci)
                    has_any = True

            if has_any:
                batch_ids.append(user_no)
                batch_seqs.append(seq_prefix)

        if not batch_seqs:
            continue

        try:
            preds = predict_next_batch(model, tokenizer, batch_seqs, top_k=3, use_cache=False)
        except Exception:
            continue

        for h in horizons_min:
            actuals = batch_actuals.get(h, {})
            for idx, (actual_locus, window_loci) in actuals.items():
                if idx >= len(preds):
                    continue
                pred_list = preds[idx]
                if not pred_list:
                    continue

                horizon_results[h]["n"] += 1
                pred_top1 = pred_list[0][0]
                top3_loci = [loc for loc, _ in pred_list[:3]]

                if pred_top1 in window_loci:
                    horizon_results[h]["top1"] += 1
                if window_loci & set(top3_loci):
                    horizon_results[h]["top3"] += 1

                if group_map:
                    actual_groups = {get_group(loc, group_map) for loc in window_loci}
                    pred_grp = get_group(pred_top1, group_map)
                    if pred_grp in actual_groups:
                        horizon_results[h]["group_top1"] += 1
                    top3_grps = {get_group(loc, group_map) for loc in top3_loci}
                    if actual_groups & top3_grps:
                        horizon_results[h]["group_top3"] += 1

    result = {"horizons": {}, "total_samples": 0}
    for h in horizons_min:
        n = horizon_results[h]["n"]
        if n > 0:
            entry = {
                "top1": round(horizon_results[h]["top1"] / n, 4),
                "top3": round(horizon_results[h]["top3"] / n, 4),
                "n":    n,
            }
            if group_map:
                entry["group_top1"] = round(horizon_results[h]["group_top1"] / n, 4)
                entry["group_top3"] = round(horizon_results[h]["group_top3"] / n, 4)
            result["horizons"][h]     = entry
            result["total_samples"]  += n
    return result


def get_pending_evaluations(sector_id: str, available_dates: list[str]) -> list[str]:
    """
    예측 저장 완료 + 정확도 평가 미완료 + 다음 날 데이터 존재 날짜 목록.
    """
    journal = _load_journal(sector_id)
    entries = journal.get("entries", {})
    accuracy_log = journal.get("accuracy_log", {})

    date_set = set(available_dates)
    pending = []

    for pred_date in sorted(entries.keys()):
        if pred_date in accuracy_log:
            continue
        try:
            d = datetime.strptime(pred_date, "%Y%m%d")
            next_date = (d + timedelta(days=1)).strftime("%Y%m%d")
            if next_date in date_set:
                pending.append(pred_date)
        except ValueError:
            continue

    return pending
