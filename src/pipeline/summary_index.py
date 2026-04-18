"""
Summary Index — 전체 날짜 KPI 요약 캐시
=========================================
processed/ 디렉토리의 모든 날짜 데이터를 한 곳에 집약한 경량 JSON.

구조:
    data/index/{sector_id}/summary_index.json
    {
      "sector_id": "Y1_SKHynix",
      "updated_at": "2026-03-21T10:00:00",
      "dates": {
        "20260316": {
          "total_workers_access": 9185,
          "total_workers_move": 8152,
          "avg_ewi": 0.62,
          "avg_cre": 0.34,
          "high_cre_count": 12,
          "high_fatigue_count": 8,
          "companies": 85,
          "correction_rate": 0.07,
          "top_insight_title": "FAB CRE 급등",
          "top_insight_severity": 3,
          "top_insight_category": "safety",
          "processed_at": "2026-03-16T..."
        },
        ...
      }
    }

목적:
  - 앱 시작 시 이것만 로드(수KB) → 사이드바 트렌드/날짜 목록 즉시 표시
  - 날짜별 Parquet은 날짜 선택 시 온디맨드 로드
  - save_daily_results() 이후 자동 업데이트 (증분 방식)

★ Perf: build_summary_index()는 Parquet 대신 meta.json만 읽음
  → 5일×1KB = ~5KB (vs 5일×35MB = ~175MB)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config as cfg
from core.cache.policy import SUMMARY_INDEX

# ★ 임계값 단일 소스 (하드코딩 금지) — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_HIGH

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "summary_index.json"


# ─── 경로 헬퍼 ──────────────────────────────────────────────────────
def _index_dir(sector_id: str) -> Path:
    """sector별 index 디렉토리."""
    return cfg.INDEX_DIR / sector_id


def _index_path(sector_id: str) -> Path:
    return _index_dir(sector_id) / _INDEX_FILENAME


# ─── 로드 ─────────────────────────────────────────────────────────
@st.cache_data(ttl=SUMMARY_INDEX, show_spinner=False)
def load_summary_index(sector_id: str) -> dict:
    """
    summary_index.json 로드. ★ SUMMARY_INDEX 정책 (10분).

    반환: {
        "sector_id": str,
        "updated_at": str,
        "dates": { date_str: DateSummary dict }
    }
    없으면 빈 구조 반환.
    """
    path = _index_path(sector_id)
    if not path.exists():
        return {"sector_id": sector_id, "updated_at": "", "dates": {}}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"summary_index 로드 실패: {e}")
        return {"sector_id": sector_id, "updated_at": "", "dates": {}}


def get_date_summary(date_str: str, sector_id: str) -> dict | None:
    """특정 날짜의 요약 반환. 없으면 None."""
    idx = load_summary_index(sector_id)
    return idx.get("dates", {}).get(date_str)


# ─── 저장 ─────────────────────────────────────────────────────────
def _save_index(index: dict, sector_id: str) -> None:
    """index 딕셔너리를 파일에 저장."""
    path = _index_path(sector_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, default=str)
    # 캐시 무효화
    load_summary_index.clear()


# ─── 단일 날짜 업데이트 ────────────────────────────────────────────
def update_date_entry(
    date_str: str,
    results: dict,
    sector_id: str,
    insight_report=None,
) -> None:
    """
    단일 날짜 처리 완료 후 summary index에 해당 날짜 항목을 추가/갱신.

    Args:
        date_str: "YYYYMMDD"
        results:  save_daily_results()에 넘기는 dict + meta
        sector_id: Sector ID
        insight_report: InsightReport 객체 (있으면 top insight 저장)
    """
    meta       = results.get("meta", {})
    worker_df  = results.get("worker", pd.DataFrame())

    entry: dict = {
        "total_workers_access": meta.get("total_workers_access", 0),
        "total_workers_move":   meta.get("total_workers_move", 0),
        "tward_holders":        meta.get("tward_holders", 0),
        "companies":            meta.get("companies", 0),
        "correction_rate":      meta.get("correction_rate", 0.0),
        "processed_at":         meta.get("processed_at", ""),
    }

    # Worker 지표 (NaN 방어 + 단일 상수 참조)
    # ★ 단일화 감사 M2: avg_ewi/avg_cre는 ewi_reliable(음영 ≤20%) 작업자만 평균 계산
    # → daily/productivity_tab의 apply_tward_reliable() 기준과 일치시켜 탭 간 수치 통일
    if not worker_df.empty:
        _rel = (
            worker_df[worker_df["ewi_reliable"] == True]
            if "ewi_reliable" in worker_df.columns else worker_df
        )
        if "ewi" in worker_df.columns:
            _ewi_rel = _rel["ewi"]
            entry["avg_ewi"]       = round(float(_ewi_rel.mean(skipna=True)), 4) if _ewi_rel.notna().any() else 0.0
            # high_count는 전체 대상 (전수 노출)
            _ewi_all = worker_df["ewi"]
            entry["high_ewi_count"] = int((_ewi_all.fillna(0) >= EWI_HIGH).sum())
        if "cre" in worker_df.columns:
            _cre_rel = _rel["cre"]
            entry["avg_cre"]       = round(float(_cre_rel.mean(skipna=True)), 4) if _cre_rel.notna().any() else 0.0
            _cre_all = worker_df["cre"]
            entry["high_cre_count"] = int((_cre_all.fillna(0) >= CRE_HIGH).sum())
        if "fatigue_score" in worker_df.columns:
            _f = worker_df["fatigue_score"]
            entry["avg_fatigue"]        = round(float(_f.mean(skipna=True)), 4) if _f.notna().any() else 0.0
            entry["high_fatigue_count"] = int((_f.fillna(0) >= 0.6).sum())  # fatigue 기준은 별도(생리학적)
        if "sii" in worker_df.columns:
            _s_rel = _rel["sii"]
            entry["avg_sii"] = round(float(_s_rel.mean(skipna=True)), 4) if _s_rel.notna().any() else 0.0
        # 신뢰 작업자 수 (분모 가시화)
        entry["reliable_workers"] = int(len(_rel))

    # Top Insight
    if insight_report and getattr(insight_report, "insights", None):
        top = insight_report.insights[0]
        entry["top_insight_title"]    = top.title
        entry["top_insight_severity"] = int(top.severity)
        entry["top_insight_category"] = top.category

    # 기존 index 로드 후 항목 추가
    idx = load_summary_index.__wrapped__(sector_id) if hasattr(
        load_summary_index, "__wrapped__"
    ) else _load_raw(sector_id)

    idx["sector_id"] = sector_id
    idx.setdefault("dates", {})[date_str] = entry
    _save_index(idx, sector_id)
    logger.info(f"summary_index 업데이트: {sector_id}/{date_str}")


# ─── 전체 재빌드 ───────────────────────────────────────────────────
def build_summary_index(sector_id: str) -> dict:
    """
    processed/ 디렉토리 전체를 스캔하여 summary_index.json 재빌드.

    ★ Perf: meta.json만 읽음 (Parquet 로드 없음).
    처리된 모든 날짜를 순회하므로 처음 한 번만 실행 권장.
    이후에는 update_date_entry()로 증분 업데이트.
    """
    from src.pipeline.cache_manager import detect_processed_dates, _date_dir

    processed = detect_processed_dates(sector_id)
    if not processed:
        logger.info(f"build_summary_index: 처리 완료 날짜 없음 ({sector_id})")
        return {"sector_id": sector_id, "updated_at": "", "dates": {}}

    idx = {"sector_id": sector_id, "dates": {}}
    for date_str in processed:
        date_d = _date_dir(date_str, sector_id)
        meta_path = date_d / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        # worker.parquet에서 지표 추출 (경량 컬럼 선택)
        worker_path = date_d / "worker.parquet"
        entry: dict = {
            "total_workers_access": meta.get("total_workers_access", 0),
            "total_workers_move":   meta.get("total_workers_move", 0),
            "tward_holders":        meta.get("tward_holders", 0),
            "companies":            meta.get("companies", 0),
            "correction_rate":      meta.get("correction_rate", 0.0),
            "processed_at":         meta.get("processed_at", ""),
        }
        if worker_path.exists():
            try:
                # 컬럼 선택 로드 — 존재하지 않는 컬럼은 pyarrow가 무시
                target_cols = ["ewi", "cre", "fatigue_score", "sii", "ewi_reliable"]
                try:
                    wdf = pd.read_parquet(worker_path, columns=target_cols)
                except Exception:
                    # 컬럼 불일치 시 전체 로드 후 필터
                    wdf = pd.read_parquet(worker_path)
                    wdf = wdf[[c for c in target_cols if c in wdf.columns]]
                # ★ M2: 평균은 ewi_reliable 필터 후, 고위험 count는 전수로 계산
                _rel = (
                    wdf[wdf["ewi_reliable"] == True]
                    if "ewi_reliable" in wdf.columns else wdf
                )
                if "ewi" in wdf.columns:
                    _ewi_rel = _rel["ewi"]
                    entry["avg_ewi"]        = round(float(_ewi_rel.mean(skipna=True)), 4) if _ewi_rel.notna().any() else 0.0
                    entry["high_ewi_count"] = int((wdf["ewi"].fillna(0) >= EWI_HIGH).sum())
                if "cre" in wdf.columns:
                    _cre_rel = _rel["cre"]
                    entry["avg_cre"]        = round(float(_cre_rel.mean(skipna=True)), 4) if _cre_rel.notna().any() else 0.0
                    entry["high_cre_count"] = int((wdf["cre"].fillna(0) >= CRE_HIGH).sum())
                if "fatigue_score" in wdf.columns:
                    _f = wdf["fatigue_score"]
                    entry["avg_fatigue"]        = round(float(_f.mean(skipna=True)), 4) if _f.notna().any() else 0.0
                    entry["high_fatigue_count"] = int((_f.fillna(0) >= 0.6).sum())
                entry["reliable_workers"] = int(len(_rel))
                if "sii" in wdf.columns:
                    _s = _rel["sii"]
                    entry["avg_sii"] = round(float(_s.mean(skipna=True)), 4) if _s.notna().any() else 0.0
            except Exception as e:
                logger.warning(f"worker.parquet 컬럼 선택 실패 ({date_str}): {e}")

        idx["dates"][date_str] = entry

    _save_index(idx, sector_id)
    logger.info(f"build_summary_index 완료: {sector_id}, {len(idx['dates'])}일")
    return idx


def _load_raw(sector_id: str) -> dict:
    """캐시 우회 직접 로드 (update_date_entry 내부 전용)."""
    path = _index_path(sector_id)
    if not path.exists():
        return {"sector_id": sector_id, "updated_at": "", "dates": {}}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sector_id": sector_id, "updated_at": "", "dates": {}}


# ─── 트렌드 헬퍼 ──────────────────────────────────────────────────
def get_trend_series(
    sector_id: str,
    metric: str = "avg_cre",
    n: int = 7,
) -> pd.DataFrame:
    """
    summary index에서 특정 지표의 최근 N일 시계열 반환.

    반환: DataFrame(date, {metric})  — 날짜 오름차순
    """
    idx   = load_summary_index(sector_id)
    dates = sorted(idx.get("dates", {}).keys())[-n:]
    rows  = []
    for d in dates:
        val = idx["dates"][d].get(metric)
        if val is not None:
            rows.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", metric: val})
    return pd.DataFrame(rows)


def get_weekly_comparison(sector_id: str) -> dict:
    """
    이번 주 vs 지난 주 평균 KPI 비교.

    반환: {
        "this_week": { avg_ewi, avg_cre, high_cre_count, ... },
        "last_week": { ... },
        "delta": { ... }          # 이번주 - 지난주
    }
    """
    idx   = load_summary_index(sector_id)
    dates = sorted(idx.get("dates", {}).keys())

    if len(dates) < 2:
        return {}

    # 최근 5일 = 이번 주, 그 이전 5일 = 지난 주
    this_week_dates = dates[-5:]
    last_week_dates = dates[-10:-5] if len(dates) >= 10 else dates[:-5]

    metrics = ["avg_ewi", "avg_cre", "high_cre_count", "high_fatigue_count", "avg_fatigue"]

    def _avg(date_list: list[str], m: str) -> float | None:
        vals = [idx["dates"][d].get(m) for d in date_list if idx["dates"][d].get(m) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    this_w = {m: _avg(this_week_dates, m) for m in metrics}
    last_w = {m: _avg(last_week_dates, m) for m in metrics}
    delta  = {
        m: round(this_w[m] - last_w[m], 4)
        if (this_w[m] is not None and last_w[m] is not None) else None
        for m in metrics
    }

    return {"this_week": this_w, "last_week": last_w, "delta": delta}
