"""
트렌드 예측기 — Trend Forecaster
=================================
누적 데이터(7일+)를 기반으로 주요 지표 추세를 분석하고,
임계값 도달 시점을 예측하여 사전 경고를 생성한다.

분석 항목:
  1. 지표 추세 예측 — "현 추세 유지 시 N일 후 CRE 임계값 초과"
  2. 요일 패턴 감지 — "월요일 EWI 항상 낮음" 등
  3. 주간 변화 비교 — 이번 주 vs 지난 주
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.intelligence.models import Insight, Severity
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, EWI_LOW

logger = logging.getLogger(__name__)

# 감시할 지표 + 임계값 — constants.py 상수 참조
_WATCH_METRICS = [
    {"key": "avg_cre",     "label": "평균 CRE",   "threshold": CRE_HIGH, "direction": "above"},
    {"key": "avg_ewi",     "label": "평균 EWI",   "threshold": 0.2,      "direction": "below"},
    {"key": "high_cre_pct", "label": "고위험 비율", "threshold": 15.0,     "direction": "above"},
]

_DAY_NAMES_KR = ["월", "화", "수", "목", "금", "토", "일"]


def forecast_trends(
    processed_dates: list[str] | None,
    current_date: str,
    sector_id: str | None = None,
) -> list[Insight]:
    """
    마스터 트렌드 분석. 7일 이상 데이터가 있어야 활성화.
    """
    if not processed_dates or len(processed_dates) < 5:
        return []

    insights = []

    try:
        daily_stats = _collect_daily_stats(processed_dates, sector_id)
        if len(daily_stats) < 5:
            return []

        insights.extend(_forecast_thresholds(daily_stats))
        insights.extend(_detect_weekly_change(daily_stats, current_date))

    except Exception as e:
        logger.warning(f"트렌드 분석 실패: {e}")

    return insights


def _collect_daily_stats(
    dates: list[str],
    sector_id: str | None,
) -> pd.DataFrame:
    """
    날짜별 핵심 통계를 meta + worker(컬럼 선택)에서 수집.
    ★ Perf: journey 스킵, worker도 필요 컬럼만 로드.
    """
    from src.pipeline.cache_manager import load_meta_only, _date_dir

    rows = []
    for d in dates:
        try:
            meta = load_meta_only(d, sector_id) or {}

            # ★ Perf: worker에서 ewi/cre/fatigue만 로드 (~40KB vs 1.7MB)
            date_dir = _date_dir(d, sector_id)
            wp = date_dir / "worker.parquet"
            cols = ["ewi", "cre", "fatigue_score"]
            wdf = pd.read_parquet(wp, columns=cols) if wp.exists() else pd.DataFrame()

            row = {"date": d}
            row["total_access"] = meta.get("total_workers_access", 0)
            row["total_tward"] = meta.get("total_workers_move", 0)

            if not wdf.empty:
                if "ewi" in wdf.columns:
                    row["avg_ewi"] = round(wdf["ewi"].mean(), 4)
                if "cre" in wdf.columns:
                    row["avg_cre"] = round(wdf["cre"].mean(), 4)
                    n_total = len(wdf)
                    n_high = len(wdf[wdf["cre"].fillna(0) >= CRE_HIGH])
                    row["high_cre_pct"] = round(n_high / n_total * 100, 2) if n_total > 0 else 0
                if "fatigue_score" in wdf.columns:
                    row["avg_fatigue"] = round(wdf["fatigue_score"].mean(), 4)

            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)


def _forecast_thresholds(stats: pd.DataFrame) -> list[Insight]:
    """선형 회귀로 임계값 도달 시점 예측."""
    insights = []

    for metric in _WATCH_METRICS:
        key = metric["key"]
        if key not in stats.columns:
            continue

        values = stats[key].dropna().values
        if len(values) < 5:
            continue

        x = np.arange(len(values), dtype=float)
        try:
            slope, intercept = np.polyfit(x, values, 1)
        except Exception:
            continue

        # 추세 방향 확인
        if abs(slope) < 1e-6:
            continue  # 거의 변화 없음

        threshold = metric["threshold"]
        direction = metric["direction"]
        current = values[-1]

        # 현재값이 이미 임계값을 넘었으면 스킵 (이상 탐지가 담당)
        if direction == "above" and current >= threshold:
            continue
        if direction == "below" and current <= threshold:
            continue

        # 임계값 도달까지 남은 일수 계산
        if direction == "above" and slope > 0:
            days_to_threshold = (threshold - current) / slope
        elif direction == "below" and slope < 0:
            days_to_threshold = (threshold - current) / slope
        else:
            continue  # 추세가 임계값 반대 방향

        if days_to_threshold <= 0 or days_to_threshold > 30:
            continue  # 30일 넘으면 의미 없음

        days_int = int(round(days_to_threshold))
        severity = Severity.HIGH if days_int <= 7 else Severity.MEDIUM

        trend_dir = "상승" if slope > 0 else "하락"
        daily_change = abs(slope)

        insights.append(Insight(
            category="trend",
            severity=severity,
            title=f"{metric['label']} {days_int}일 후 임계값 도달 예측",
            description=(
                f"{metric['label']}이(가) 일평균 {daily_change:.4f}씩 {trend_dir} 중입니다. "
                f"현재 {current:.3f} → 현 추세 유지 시 약 {days_int}일 후 "
                f"{'위험' if direction == 'above' else '주의'} 임계값({threshold})에 도달합니다."
            ),
            evidence={
                "current": round(current, 4),
                "threshold": threshold,
                "slope_per_day": round(slope, 5),
                "days_to_threshold": days_int,
                "data_points": len(values),
            },
            recommendation=f"{'위험도 상승' if direction == 'above' else '생산성 저하'} 추세 "
                          f"원인 분석 및 선제적 조치 필요",
            source="trend_forecaster",
        ))

    return insights


def _detect_weekly_change(stats: pd.DataFrame, current_date: str) -> list[Insight]:
    """이번 주 vs 지난 주 비교 (최근 5일 vs 이전 5일)."""
    if len(stats) < 8:
        return []

    recent = stats.tail(5)  # 최근 5일 (이번 주)
    previous = stats.iloc[-10:-5] if len(stats) >= 10 else stats.head(5)  # 이전 5일

    insights = []
    compare_cols = [
        ("avg_cre", "평균 CRE", "higher_is_worse"),
        ("avg_ewi", "평균 EWI", "neutral"),
        ("high_cre_pct", "고위험 비율(%)", "higher_is_worse"),
    ]

    changes = []
    for col, label, direction in compare_cols:
        if col not in recent.columns or col not in previous.columns:
            continue
        r_mean = recent[col].mean()
        p_mean = previous[col].mean()
        if p_mean == 0:
            continue
        change_pct = (r_mean - p_mean) / abs(p_mean) * 100

        if abs(change_pct) >= 10:  # 10% 이상 변화만 의미 있음
            changes.append({
                "label": label,
                "recent": round(r_mean, 4),
                "previous": round(p_mean, 4),
                "change_pct": round(change_pct, 1),
                "direction": direction,
            })

    if not changes:
        return []

    # 가장 큰 변화만 인사이트로
    worst = max(changes, key=lambda c: abs(c["change_pct"]))

    direction_str = "상승" if worst["change_pct"] > 0 else "하락"
    is_bad = (worst["direction"] == "higher_is_worse" and worst["change_pct"] > 0)
    severity = Severity.MEDIUM if is_bad else Severity.LOW

    desc_parts = [f"{c['label']}: {c['previous']:.3f} → {c['recent']:.3f} ({c['change_pct']:+.1f}%)"
                  for c in changes]

    insights.append(Insight(
        category="trend",
        severity=severity,
        title=f"주간 변화: {worst['label']} {worst['change_pct']:+.1f}%",
        description=f"최근 5일 대비 이전 5일 비교:\n" + "\n".join(f"  - {p}" for p in desc_parts),
        evidence={"changes": changes},
        recommendation="변화 원인 파악 및 개선/악화 추세 모니터링",
        source="trend_forecaster",
    ))

    return insights
