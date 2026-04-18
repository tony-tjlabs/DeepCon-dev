"""
Context Builder (Y1 전용) — 역사적 맥락 데이터 구축
====================================================
anchor_date 기준으로 과거 데이터에서 4가지 비교 컨텍스트 블록을 생성한다.

[A] 최근 14일 기준선  — 평균/표준편차 범위 (M15X 7일 → Y1 2주로 확장)
[B] 같은 요일 패턴    — 최근 4주(4개) 동일 요일 습관성 파악
[C] 직전일 대비 변화  — 단기 트렌드 감지
[D] 전체 기간 내 순위 — 이날이 얼마나 특이한지 (summary_index 기반)

Y1 특성:
  - n_baseline = 14 (M15X 7일 → Y1 2주)
  - n_same_dow = 4  (최근 4주 같은 요일 = 8주 패턴)
  - build_transit() 제거 (Y1에 해당 없음)
  - build_space(): 구역 그룹별 체류 인원 기준선 신규
  - build_deep_space(): Prediction Journal 정확도 트렌드 신규
  - 주말 플래그 처리: 주말(토/일) 데이터는 기준선에서 별도 처리
    (Y1 주말은 유지보수 인원만 → 평일 기준선과 혼용 시 왜곡)

사용:
    from src.intelligence.context_builder import ContextBuilder

    ctx = ContextBuilder.build_daily("20260409", "Y1_SKHynix")
    ctx_s = ContextBuilder.build_space("Y1_SKHynix", "20260409")
    ctx_ds = ContextBuilder.build_deep_space("Y1_SKHynix", "20260409")
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.cache.policy import DAILY_PARQUET

# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

logger = logging.getLogger(__name__)

_WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

# Y1 전용 기준선 파라미터
_N_BASELINE_Y1  = 14  # 최근 N일 기준선 (M15X 7일 → Y1 2주)
_N_SAME_DOW_Y1  = 4   # 같은 요일 최근 N개 (4주 패턴)
_WEEKEND_DAYS   = {5, 6}  # 토(5)/일(6) — 유지보수 인원만

# 경량 로드용 컬럼 목록
_WORKER_COLS = [
    "user_no", "ewi", "cre", "sii",
    "high_active_min", "low_active_min", "standby_min",
    "gap_ratio", "shift_type",
]
_SPACE_COLS = [
    "locus_id", "locus_token", "unique_workers",
    "total_person_minutes", "is_confined", "is_high_voltage",
]

_METRIC_LABELS: dict[str, str] = {
    "n_workers":         "작업자 수(명)",
    "ewi":               "EWI(작업집중도)",
    "cre":               "CRE(위험노출도)",
    "sii":               "SII(공간위험-작업강도)",
    "high_cre_pct":      "고위험 비율(%)",
    "gap_ratio":         "GAP 비율",
    "avg_ewi":           "EWI(평균)",
    "avg_cre":           "CRE(평균)",
    "high_cre_count":    "고위험자 수(명)",
}


# ── 슬림 로더 (st.cache_data 적용) ─────────────────────────────────────────

@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def _load_worker_slim(date_str: str, sector_id: str) -> Optional[dict]:
    """
    worker.parquet 핵심 컬럼만 읽어 집계 dict 반환.
    DataFrame 전체가 아닌 float 집계값만 캐시 → 메모리 절약.
    """
    try:
        import config as cfg
        p = cfg.PROCESSED_DIR / sector_id / date_str / "worker.parquet"
        if not p.exists():
            return None

        import pyarrow.parquet as pq
        schema = pq.read_schema(str(p))
        cols   = [c for c in schema.names if c in _WORKER_COLS]
        if not cols:
            return None

        df = pd.read_parquet(p, columns=cols)
        if df.empty:
            return None

        rec: dict = {"n_workers": len(df)}
        for col in ["ewi", "cre", "sii", "gap_ratio"]:
            if col in df.columns:
                s = df[col].dropna()
                if len(s):
                    rec[col]           = float(s.mean())
                    rec[f"{col}_std"]  = float(s.std()) if len(s) > 1 else 0.0

        if "cre" in df.columns:
            cre = df["cre"].dropna()
            rec["high_cre_pct"] = float((cre >= CRE_HIGH).mean() * 100) if len(cre) else 0.0

        # 교대 비율
        if "shift_type" in df.columns:
            shift_dist = df["shift_type"].value_counts(normalize=True)
            for k, v in shift_dist.items():
                rec[f"shift_{k}_ratio"] = round(float(v), 3)

        return rec
    except Exception as e:
        logger.debug("worker slim load failed %s/%s: %s", sector_id, date_str, e)
        return None


@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def _load_space_slim(date_str: str, sector_id: str) -> Optional[dict]:
    """
    space.parquet 핵심 컬럼만 읽어 구역 그룹별 집계 dict 반환.
    """
    try:
        import config as cfg
        p = cfg.PROCESSED_DIR / sector_id / date_str / "space.parquet"
        if not p.exists():
            return None

        import pyarrow.parquet as pq
        schema = pq.read_schema(str(p))
        cols   = [c for c in schema.names if c in _SPACE_COLS]
        if not cols:
            return None

        df = pd.read_parquet(p, columns=cols)
        if df.empty:
            return None

        rec: dict = {"n_spaces": len(df)}

        # 구역 그룹별 평균 체류 인원 집계
        if "locus_token" in df.columns and "unique_workers" in df.columns:
            group_map = {
                "fab":       ["fab"],
                "cub":       ["cub"],
                "wwt":       ["wwt"],
                "support":   ["breakroom", "smoking", "restroom", "본진"],
                "gate":      ["gate", "timeclock"],
                "transit":   ["transit", "hoist"],
            }
            for group_key, keywords in group_map.items():
                mask = df["locus_token"].str.lower().apply(
                    lambda t: any(k in str(t) for k in keywords)
                )
                sub = df[mask]
                if not sub.empty:
                    rec[f"space_{group_key}_avg_workers"] = float(
                        sub["unique_workers"].mean()
                    )

        # 전체 통계
        if "unique_workers" in df.columns:
            uw = df["unique_workers"].dropna()
            rec["space_avg_workers"] = float(uw.mean()) if len(uw) else 0.0
            rec["space_max_workers"] = float(uw.max()) if len(uw) else 0.0

        return rec
    except Exception as e:
        logger.debug("space slim load failed %s/%s: %s", sector_id, date_str, e)
        return None


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _is_weekend(date_str: str) -> bool:
    """날짜가 주말(토/일)인지 확인."""
    try:
        return datetime.strptime(date_str, "%Y%m%d").weekday() in _WEEKEND_DAYS
    except ValueError:
        return False


def _select_dates(
    anchor_date: str,
    sector_id: str,
    n_prior: int,
    n_same_dow: int,
    exclude_weekend_from_baseline: bool = True,
) -> tuple[list[str], list[str], Optional[str]]:
    """
    최근 N일, 같은 요일 M개, 직전일 날짜 선택.

    Y1 주말 처리:
      - exclude_weekend_from_baseline=True: 기준선에서 주말 제외 (평일만)
      - 같은 요일 선택 시에는 주말 포함 (주말-주말 비교가 의미 있음)

    Returns: (recent_dates, same_dow_dates, prev_date)
    """
    from src.pipeline.cache_manager import detect_processed_dates
    processed = sorted(detect_processed_dates(sector_id))
    prior     = [d for d in processed if d < anchor_date]

    # 기준선: 주말 제외 옵션
    if exclude_weekend_from_baseline and not _is_weekend(anchor_date):
        baseline_pool = [d for d in prior if not _is_weekend(d)]
    else:
        baseline_pool = prior

    recent    = baseline_pool[-n_prior:]
    prev_date = prior[-1] if prior else None

    try:
        anchor_dow = datetime.strptime(anchor_date, "%Y%m%d").weekday()
        same_dow   = [
            d for d in prior
            if datetime.strptime(d, "%Y%m%d").weekday() == anchor_dow
        ][-n_same_dow:]
    except ValueError:
        same_dow = []

    return recent, same_dow, prev_date


def _stats_block(records: list[dict], key: str) -> Optional[dict]:
    """여러 날짜 집계 dict에서 특정 key의 기술통계 계산."""
    vals = [r[key] for r in records if key in r and r[key] is not None]
    if not vals:
        return None
    s = pd.Series(vals, dtype=float)
    return {
        "mean": round(float(s.mean()), 3),
        "std":  round(float(s.std()), 3) if len(vals) > 1 else 0.0,
        "min":  round(float(s.min()), 3),
        "max":  round(float(s.max()), 3),
        "n":    len(vals),
    }


def _rank_from_index(
    anchor_date: str, metric_key: str, sector_id: str
) -> tuple[int, int]:
    """
    summary_index에서 metric_key의 전체 날짜 중 anchor_date 순위 반환.
    Returns: (rank, total) — rank는 1-based, 높은 값이 1위
    """
    try:
        from src.pipeline.summary_index import load_summary_index
        idx        = load_summary_index(sector_id)
        dates_data = idx.get("dates", {})
        pairs      = [
            (d, v[metric_key])
            for d, v in dates_data.items()
            if metric_key in v and v[metric_key] is not None
        ]
        if not pairs:
            return 0, 0
        pairs.sort(key=lambda x: x[1], reverse=True)
        total = len(pairs)
        rank  = next((i + 1 for i, (d, _) in enumerate(pairs) if d == anchor_date), 0)
        return rank, total
    except Exception as e:
        logger.debug("rank from index failed: %s", e)
        return 0, 0


def _fmt_baseline(records: list[dict], keys: list[str], n_days: int) -> list[str]:
    """최근 N일 기준선 텍스트 블록."""
    if not records:
        return []
    lines = [f"### [A] 최근 {n_days}일 기준선 ({len(records)}일 데이터, 주말 제외)"]
    for key in keys:
        st_ = _stats_block(records, key)
        if st_:
            label = _METRIC_LABELS.get(key, key)
            lines.append(
                f"- {label}: 평균 {st_['mean']} (±{st_['std']}, "
                f"범위 {st_['min']}~{st_['max']}, N={st_['n']})"
            )
    return lines


def _fmt_dow(
    records: list[dict], keys: list[str], weekday_str: str, n: int
) -> list[str]:
    """같은 요일 패턴 텍스트 블록."""
    if not records:
        return []
    lines = [f"### [B] {weekday_str} 패턴 (최근 {len(records)}회 평균)"]
    for key in keys:
        st_ = _stats_block(records, key)
        if st_:
            label = _METRIC_LABELS.get(key, key)
            lines.append(f"- {label}: {st_['mean']} (±{st_['std']}, N={st_['n']})")
    return lines


def _fmt_trend(
    anchor_rec: dict, prev_rec: dict, keys: list[str]
) -> list[str]:
    """직전일 대비 변화 텍스트 블록."""
    if not anchor_rec or not prev_rec:
        return []
    lines = ["### [C] 직전일 대비 변화"]
    for key in keys:
        curr = anchor_rec.get(key)
        prev = prev_rec.get(key)
        if curr is None or prev is None:
            continue
        delta = curr - prev
        pct   = (delta / abs(prev) * 100) if prev != 0 else 0.0
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        label = _METRIC_LABELS.get(key, key)
        lines.append(
            f"- {label}: {round(prev, 3)} → {round(curr, 3)} "
            f"({arrow}{abs(delta):.3f} / {pct:+.1f}%)"
        )
    return lines


def _fmt_rank(
    anchor_date: str,
    sector_id: str,
    rank_specs: list[tuple[str, str, bool]],
) -> list[str]:
    """
    전체 기간 순위 텍스트 블록.
    rank_specs: [(idx_key, display_label, higher_is_better), ...]
    """
    lines: list[str] = []
    for idx_key, label, higher_better in rank_specs:
        rank, total = _rank_from_index(anchor_date, idx_key, sector_id)
        if rank and total:
            note = "높을수록 좋음" if higher_better else "낮을수록 좋음"
            lines.append(f"- {label}: {total}일 중 {rank}위 ({note})")
    if lines:
        lines.insert(0, "### [D] 전체 기간 내 순위")
    return lines


# ── 공개 API ─────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    anchor_date 기준 역사적 맥락 텍스트 생성기 (Y1 전용).

    M15X 대비 Y1 변경사항:
      - n_baseline = 14 (M15X 7 → Y1 14)
      - n_same_dow = 4  (M15X 3 → Y1 4)
      - build_transit() 제거 (Y1에 해당 없음)
      - build_space() 신규 (구역 그룹별 체류 인원 기준선)
      - build_deep_space() 신규 (Prediction Journal 정확도 7일 rolling)
      - 주말 기준선 분리 (Y1 주말 = 유지보수 소규모 인원)
    """

    @staticmethod
    def build_daily(
        anchor_date: str,
        sector_id: str,
        n_baseline: int = _N_BASELINE_Y1,
        n_same_dow: int = _N_SAME_DOW_Y1,
    ) -> str:
        """
        일별 작업자 분석용 비교 컨텍스트 텍스트.

        [A] 최근 14일 EWI/CRE/고위험비율/작업자수 기준선 (주말 제외)
        [B] 같은 요일 패턴 (최근 4주)
        [C] 직전일 대비 변화
        [D] 전체 기간 내 순위 (summary_index 기반)

        주말 anchor_date: 기준선에 주말 포함 (주말-주말 비교)
        """
        try:
            anchor_dt = datetime.strptime(anchor_date, "%Y%m%d")
        except ValueError:
            return ""

        weekday_str    = _WEEKDAY_KR[anchor_dt.weekday()] + "요일"
        is_anchor_wknd = anchor_dt.weekday() in _WEEKEND_DAYS
        weekend_note   = " (주말: 유지보수 인원 위주)" if is_anchor_wknd else ""

        recent, same_dow, prev_date = _select_dates(
            anchor_date, sector_id, n_baseline, n_same_dow,
            exclude_weekend_from_baseline=not is_anchor_wknd,
        )

        if not recent and not same_dow:
            return ""

        # 데이터 로드
        recent_recs = [r for r in (_load_worker_slim(d, sector_id) for d in recent) if r]
        dow_recs    = [r for r in (_load_worker_slim(d, sector_id) for d in same_dow) if r]
        anchor_rec  = _load_worker_slim(anchor_date, sector_id) or {}
        prev_rec    = _load_worker_slim(prev_date, sector_id) if prev_date else {}

        metric_keys = ["n_workers", "ewi", "cre", "high_cre_pct"]

        sections: list[str] = [
            f"\n## 비교 컨텍스트 (기준일: {anchor_date} {weekday_str}{weekend_note})"
        ]
        sections += _fmt_baseline(recent_recs, metric_keys, n_baseline)
        if dow_recs:
            sections += _fmt_dow(dow_recs, metric_keys, weekday_str, n_same_dow)
        if prev_rec:
            sections += _fmt_trend(anchor_rec, prev_rec, metric_keys)

        rank_specs = [
            ("avg_ewi",        "EWI",         True),
            ("avg_cre",        "CRE",         False),
            ("high_cre_count", "고위험자 수",  False),
        ]
        sections += _fmt_rank(anchor_date, sector_id, rank_specs)

        return "\n".join(sections)

    @staticmethod
    def build_space(
        sector_id: str,
        anchor_date: str,
        n_baseline: int = _N_BASELINE_Y1,
        n_same_dow: int = _N_SAME_DOW_Y1,
    ) -> str:
        """
        공간 혼잡도 분석용 비교 컨텍스트 텍스트 (Y1 신규).

        [A] 최근 14일 구역 그룹별 평균 체류 인원 기준선
        [B] 같은 요일 패턴
        [C] 직전일 대비 변화

        구역 그룹: fab / cub / wwt / support / gate / transit
        """
        try:
            anchor_dt = datetime.strptime(anchor_date, "%Y%m%d")
        except ValueError:
            return ""

        weekday_str = _WEEKDAY_KR[anchor_dt.weekday()] + "요일"
        recent, same_dow, prev_date = _select_dates(
            anchor_date, sector_id, n_baseline, n_same_dow,
        )

        if not recent and not same_dow:
            return ""

        recent_recs = [r for r in (_load_space_slim(d, sector_id) for d in recent) if r]
        dow_recs    = [r for r in (_load_space_slim(d, sector_id) for d in same_dow) if r]
        anchor_rec  = _load_space_slim(anchor_date, sector_id) or {}
        prev_rec    = _load_space_slim(prev_date, sector_id) if prev_date else {}

        space_keys = [
            "space_avg_workers", "space_max_workers",
            "space_fab_avg_workers", "space_cub_avg_workers",
            "space_wwt_avg_workers",
        ]
        # 레이블 추가
        space_labels = {
            "space_avg_workers":      "전체 평균 체류(명/구역)",
            "space_max_workers":      "최대 체류(명)",
            "space_fab_avg_workers":  "FAB 평균 체류(명/구역)",
            "space_cub_avg_workers":  "CUB 평균 체류(명/구역)",
            "space_wwt_avg_workers":  "WWT 평균 체류(명/구역)",
        }
        _METRIC_LABELS.update(space_labels)

        sections: list[str] = [
            f"\n## 공간 비교 컨텍스트 (기준일: {anchor_date} {weekday_str})"
        ]

        if recent_recs:
            lines = [f"### [A] 최근 {n_baseline}일 공간 기준선 ({len(recent_recs)}일 데이터)"]
            for key in space_keys:
                st_ = _stats_block(recent_recs, key)
                if st_:
                    label = space_labels.get(key, key)
                    lines.append(
                        f"- {label}: 평균 {st_['mean']:.1f} "
                        f"(±{st_['std']:.1f}, N={st_['n']})"
                    )
            sections += lines

        if dow_recs:
            lines = [f"### [B] {weekday_str} 공간 패턴 (최근 {len(dow_recs)}회)"]
            for key in space_keys:
                st_ = _stats_block(dow_recs, key)
                if st_:
                    label = space_labels.get(key, key)
                    lines.append(f"- {label}: {st_['mean']:.1f} (±{st_['std']:.1f})")
            sections += lines

        if prev_rec and anchor_rec:
            sections += _fmt_trend(anchor_rec, prev_rec, space_keys)

        return "\n".join(sections)

    @staticmethod
    def build_deep_space(
        sector_id: str,
        anchor_date: str,
        n_rolling: int = 7,
    ) -> str:
        """
        Deep Space Prediction Journal 정확도 트렌드 컨텍스트 (Y1 신규).

        최근 n_rolling일 rolling 정확도를 계산하여 LLM에 전달.
        - Group Top-1/Top-3, Locus Top-1/Top-3 트렌드
        - 최근 7일 평균 vs 전체 평균 비교

        Prediction Journal 구조:
            accuracy_log: dict[date_str -> {evaluated_at, group_top1, top1_accuracy, ...}]
        """
        try:
            import config as cfg
            pj_path = cfg.INDEX_DIR / sector_id / "prediction_journal.json"
            if not pj_path.exists():
                return ""

            with open(pj_path, encoding="utf-8") as f:
                pj = json.load(f)
        except Exception as e:
            logger.debug("prediction_journal load failed: %s", e)
            return ""

        acc_log = pj.get("accuracy_log", {})
        if not acc_log:
            return ""

        # date-sorted entries
        sorted_dates = sorted(acc_log.keys())
        prior_dates  = [d for d in sorted_dates if d < anchor_date]

        if not prior_dates:
            return ""

        recent_n = prior_dates[-n_rolling:]
        entries  = [acc_log[d] for d in recent_n if d in acc_log]

        if not entries:
            return ""

        sections: list[str] = [
            f"\n## Deep Space 정확도 컨텍스트 (기준일: {anchor_date})"
        ]

        # 최근 rolling 평균
        def _avg(key: str) -> Optional[float]:
            vals = [e[key] for e in entries if key in e and e[key] is not None]
            return float(np.mean(vals)) if vals else None

        g1  = _avg("group_top1")
        g3  = _avg("group_top3")
        l1  = _avg("top1_accuracy")
        l3  = _avg("top3_accuracy")
        mae = _avg("congestion_mae")

        lines = [
            f"### 최근 {len(recent_n)}일 rolling 정확도 평균"
        ]
        if g1 is not None:
            g3_str = f", Group Top-3: {g3:.1%}" if g3 is not None else ""
            lines.append(f"- Group Top-1: {g1:.1%}{g3_str}")
        if l1 is not None:
            l3_str = f", Locus Top-3: {l3:.1%}" if l3 is not None else ""
            lines.append(f"- Locus Top-1: {l1:.1%}{l3_str}")
        if mae is not None:
            lines.append(f"- 혼잡도 MAE: {mae:.1f}인·분")
        sections += lines

        # 트렌드 방향성 (첫 날 vs 마지막 날)
        if len(recent_n) >= 2:
            first_entry = acc_log.get(recent_n[0], {})
            last_entry  = acc_log.get(recent_n[-1], {})
            trends: list[str] = []
            for key, label in [
                ("group_top1", "Group Top-1"),
                ("top1_accuracy", "Locus Top-1"),
            ]:
                f_val = first_entry.get(key)
                l_val = last_entry.get(key)
                if f_val is not None and l_val is not None:
                    delta = l_val - f_val
                    arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "→")
                    trends.append(
                        f"- {label}: {arrow} {f_val:.1%} → {l_val:.1%}"
                    )
            if trends:
                sections.append("\n### 정확도 트렌드 방향성 (rolling 기간 내)")
                sections.extend(trends)

        # 전체 평균 대비 비교
        all_entries = list(acc_log.values())
        if all_entries and len(all_entries) > n_rolling:
            g1_all = _avg_list([e.get("group_top1") for e in all_entries])
            l1_all = _avg_list([e.get("top1_accuracy") for e in all_entries])
            sections.append("\n### 전체 기간 평균 대비 (참조)")
            if g1_all is not None and g1 is not None:
                diff = g1 - g1_all
                sign = "+" if diff >= 0 else ""
                sections.append(
                    f"- Group Top-1: 전체평균 {g1_all:.1%} vs "
                    f"최근 {n_rolling}일 {g1:.1%} ({sign}{diff:.1%})"
                )
            if l1_all is not None and l1 is not None:
                diff = l1 - l1_all
                sign = "+" if diff >= 0 else ""
                sections.append(
                    f"- Locus Top-1: 전체평균 {l1_all:.1%} vs "
                    f"최근 {n_rolling}일 {l1:.1%} ({sign}{diff:.1%})"
                )

        return "\n".join(sections)


def _avg_list(vals: list) -> Optional[float]:
    """None 제거 후 평균 계산."""
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None
