"""
core.ai.context_builders — 탭별 LLM context dict 빌더 (M2-B T-15~19)
=====================================================================
각 탭이 "오늘 현장 어떻게 돌아가고 있나" 를 LLM 에 묻기 위해 필요한
요약 컨텍스트를 **500 tokens 이내** 로 압축한다.

설계 원칙
---------
- 원본 DataFrame 은 받되, 반환 dict 는 **이미 PII 마스킹된 상태**.
  (LLMGateway 내부에서 AnonymizationPipeline 이 한 번 더 돌지만,
  여기서도 1차 마스킹을 거쳐 context dict 자체에 raw 이름·user_no 가 남지 않게 한다.)
- 업체명은 `Company_A`, `Company_B` 처럼 코드화 (실명 추정 방지).
- user_no 는 SHA256(8자리) 해시.
- 숫자는 소수점 3자리 반올림 (프롬프트 노이즈 감소).
- 빈 DataFrame/None 에 안전해야 함 — 모든 builder 는 최악의 경우 `{}`/fallback 반환.

각 build_* 함수:
    input:  탭에서 이미 로드한 DataFrame/집계 결과
    output: dict[str, Any] — JSON 직렬화 가능, 500 tokens 이내 권장
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ─── 공통 유틸 ────────────────────────────────────────────────────

def _hash_user(user_no: Any) -> str:
    """user_no → 8자리 해시 (sector 내 고유성만 유지, 역추적 불가)."""
    if user_no is None or pd.isna(user_no):
        return "unknown"
    h = hashlib.sha256(str(user_no).encode("utf-8")).hexdigest()
    return f"u{h[:8]}"


def _codify_companies(names: list[str]) -> dict[str, str]:
    """
    업체명 → 코드 매핑 (Company_A, Company_B, ...).
    동일 세션·동일 context 내에서는 같은 업체가 같은 코드를 받도록 순서 정렬.
    """
    seen: dict[str, str] = {}
    idx = 0
    for n in names:
        if n is None or pd.isna(n):
            continue
        key = str(n).strip()
        if key and key not in seen:
            letter = ""
            i = idx
            # A, B, ..., Z, AA, AB, ...
            while True:
                letter = chr(65 + (i % 26)) + letter
                i = i // 26 - 1
                if i < 0:
                    break
            seen[key] = f"Company_{letter}"
            idx += 1
    return seen


def _r(x: Any, digits: int = 3) -> float | None:
    """float 반올림 — None/NaN 안전."""
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), digits)
    except Exception:
        return None


def _infer_weekday(date_str: str | None) -> str:
    """'20260409' → 'weekday' | 'weekend'."""
    if not date_str:
        return "unknown"
    try:
        dt = pd.to_datetime(date_str, format="%Y%m%d")
        return "weekend" if dt.dayofweek >= 5 else "weekday"
    except Exception:
        return "unknown"


def _shift_distribution(worker_df: pd.DataFrame) -> dict[str, int]:
    """shift_type 컬럼에서 day/night/ext_night/unknown 카운트."""
    if worker_df.empty or "shift_type" not in worker_df.columns:
        return {}
    counts = worker_df["shift_type"].fillna("unknown").value_counts().to_dict()
    mapping = {
        "day": "day",
        "night": "night",
        "extended_night": "ext_night",
        "unknown": "unknown",
    }
    out: dict[str, int] = {}
    for k, v in counts.items():
        out[mapping.get(k, k)] = int(v)
    return out


# ═══════════════════════════════════════════════════════════════════
# T-15 Overview Commentator
# ═══════════════════════════════════════════════════════════════════

def build_overview_context(
    *,
    sector_id: str,
    latest_date: str | None,
    summary_df: pd.DataFrame,
    worker_df: pd.DataFrame | None,
    top_n_companies: int = 3,
) -> dict[str, Any]:
    """
    🏠 현장 개요 탭용 context.

    Args:
        sector_id:    "Y1_SKHynix" 등
        latest_date:  "20260409" — 가장 최신 처리 일자
        summary_df:   _build_summary_df() 결과 (일별 KPI)
        worker_df:    최신일 worker.parquet (없으면 None)
        top_n_companies: 상위 업체 수 (기본 3)

    Returns:
        ~300 토큰 짜리 요약 dict.
    """
    ctx: dict[str, Any] = {
        "sector_id": sector_id,
        "today_date": latest_date,
        "weekday_weekend": _infer_weekday(latest_date),
    }

    # ── 일별 KPI 추이 (가장 최근 7일) ────────────────────────────
    if summary_df is not None and not summary_df.empty:
        df = summary_df.sort_values("date").tail(7).copy()
        ctx["days_available"] = int(len(summary_df))
        today_row = df.iloc[-1] if not df.empty else None
        prev_row = df.iloc[-2] if len(df) >= 2 else None

        if today_row is not None:
            ctx["today_kpi"] = {
                "workers_access":  int(today_row.get("workers_access", 0) or 0),
                "tward_holders":   int(today_row.get("tward_holders", 0) or 0),
                "wear_rate_pct":   _r(today_row.get("wear_rate", 0)),
                "companies":       int(today_row.get("companies", 0) or 0),
                "avg_ewi":         _r(today_row.get("avg_ewi", 0)),
                "avg_cre":         _r(today_row.get("avg_cre", 0)),
                "avg_fatigue":     _r(today_row.get("avg_fatigue", 0)),
                "high_ewi_count":  int(today_row.get("high_ewi_count", 0) or 0),
            }

        if prev_row is not None and today_row is not None:
            a_today = today_row.get("workers_access", 0) or 0
            a_prev = prev_row.get("workers_access", 0) or 0
            ewi_today = today_row.get("avg_ewi", 0) or 0
            ewi_prev = prev_row.get("avg_ewi", 0) or 0
            ctx["yesterday_comparison"] = {
                "workers_delta":     int(a_today - a_prev),
                "workers_delta_pct": _r(((a_today - a_prev) / a_prev * 100) if a_prev else 0, 2),
                "ewi_delta":         _r(ewi_today - ewi_prev),
            }

        # 최근 7일 평균 (today 제외) — 평균 대비 오늘 위치
        if len(df) >= 2:
            hist = df.iloc[:-1]
            ctx["recent_7d_avg"] = {
                "workers_access": _r(hist["workers_access"].mean(), 0),
                "avg_ewi":        _r(hist["avg_ewi"].mean()),
                "avg_cre":        _r(hist["avg_cre"].mean()),
                "wear_rate_pct":  _r(hist["wear_rate"].mean(), 1),
            }

    # ── 최신일 worker 기반 상위 업체 + shift 분포 ────────────────
    if worker_df is not None and not worker_df.empty:
        ctx["shift_distribution"] = _shift_distribution(worker_df)

        if "company_name" in worker_df.columns and "ewi" in worker_df.columns:
            # ewi_reliable 만 집계 (신뢰성)
            if "ewi_reliable" in worker_df.columns:
                reliable = worker_df[worker_df["ewi_reliable"] == True]  # noqa: E712
            else:
                reliable = worker_df
            agg = (
                reliable.groupby("company_name")
                .agg(
                    workers=("user_no", "nunique"),
                    avg_ewi=("ewi", "mean"),
                    avg_work_min=("work_minutes", "mean") if "work_minutes" in reliable.columns else ("ewi", "count"),
                )
                .reset_index()
                .query("workers >= 5")
                .sort_values("workers", ascending=False)
                .head(top_n_companies)
            )

            # 업체명 → 코드화
            mapping = _codify_companies(agg["company_name"].tolist())
            top_list = []
            for _, r in agg.iterrows():
                top_list.append({
                    "code":         mapping.get(r["company_name"], "Company_X"),
                    "workers":      int(r["workers"]),
                    "avg_ewi":      _r(r["avg_ewi"]),
                    "avg_work_min": _r(r.get("avg_work_min", 0), 0),
                })
            ctx["top_companies"] = top_list

    return ctx


# ═══════════════════════════════════════════════════════════════════
# T-16 Zone Time Analyst
# ═══════════════════════════════════════════════════════════════════

def build_zone_time_context(
    *,
    sector_id: str,
    date_str: str,
    worker_df: pd.DataFrame,
    company_df: pd.DataFrame | None = None,
    hourly_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    ⏱ 작업시간 탭용 context.

    핵심:
      - 업체별 작업공간 비율 Top/Bottom 3
      - 음영지역 비율 (BLE 미수신 = gap_min / work_minutes)
      - shift 분포
      - 개인 이상 (work_zone 과다/과소)
    """
    ctx: dict[str, Any] = {
        "sector_id": sector_id,
        "date": date_str,
        "weekday_weekend": _infer_weekday(date_str),
    }

    if worker_df is None or worker_df.empty:
        return ctx

    # shift 분포
    ctx["shift_distribution"] = _shift_distribution(worker_df)

    # 전체 평균
    def _safe_mean(col: str) -> float | None:
        if col not in worker_df.columns:
            return None
        v = pd.to_numeric(worker_df[col], errors="coerce").dropna()
        return _r(v.mean()) if len(v) else None

    wm_total  = pd.to_numeric(worker_df.get("work_minutes", pd.Series([])), errors="coerce").fillna(0).sum()
    wz_total  = pd.to_numeric(worker_df.get("work_zone_minutes", pd.Series([])), errors="coerce").fillna(0).sum()
    gap_total = pd.to_numeric(worker_df.get("gap_min", pd.Series([])), errors="coerce").fillna(0).sum()

    ctx["overall"] = {
        "workers":              int(worker_df["user_no"].nunique() if "user_no" in worker_df.columns else len(worker_df)),
        "avg_work_min":         _safe_mean("work_minutes"),
        "avg_work_zone_min":    _safe_mean("work_zone_minutes"),
        "work_zone_ratio":      _r(wz_total / wm_total if wm_total else 0),
        "shadow_ratio":         _r(gap_total / wm_total if wm_total else 0),  # BLE 음영비율
    }

    # 업체별 Top/Bottom 3 (work_zone_minutes / work_minutes 비율 기준)
    if "company_name" in worker_df.columns and "work_zone_minutes" in worker_df.columns:
        grp = (
            worker_df.groupby("company_name")
            .agg(
                workers=("user_no", "nunique"),
                total_work_min=("work_minutes", "sum"),
                total_zone_min=("work_zone_minutes", "sum"),
                avg_work_min=("work_minutes", "mean"),
            )
            .reset_index()
        )
        grp = grp[grp["workers"] >= 5]
        # 캡핑: work_zone > work_minutes 인 float 오차는 min 으로 보정
        grp["zone_ratio"] = (
            grp[["total_work_min", "total_zone_min"]].min(axis=1)
            / grp["total_work_min"].replace(0, 1)
        )

        if not grp.empty:
            # Top 3 (높은 작업공간 비율)
            top = grp.nlargest(3, "zone_ratio")
            # Bottom 3 (낮은 작업공간 비율)
            bot = grp.nsmallest(3, "zone_ratio")
            all_names = top["company_name"].tolist() + bot["company_name"].tolist()
            mapping = _codify_companies(all_names)

            def _pack(r):
                return {
                    "code":            mapping.get(r["company_name"], "Company_X"),
                    "workers":         int(r["workers"]),
                    "zone_ratio":      _r(r["zone_ratio"]),
                    "avg_work_min":    _r(r["avg_work_min"], 0),
                }

            ctx["companies_top_by_zone_ratio"]    = [_pack(r) for _, r in top.iterrows()]
            ctx["companies_bottom_by_zone_ratio"] = [_pack(r) for _, r in bot.iterrows()]

    # 개인 이상: 음영 비율 60%+ 인 작업자 수 (shadow 과다)
    if "gap_min" in worker_df.columns and "work_minutes" in worker_df.columns:
        wm = pd.to_numeric(worker_df["work_minutes"], errors="coerce").fillna(0)
        gm = pd.to_numeric(worker_df["gap_min"], errors="coerce").fillna(0)
        mask = (wm > 60) & (gm / wm.replace(0, 1) > 0.6)
        ctx["individual_anomalies"] = {
            "shadow_over_60pct_count": int(mask.sum()),
        }

    # 시간대별 분포 (Peak hour + WORK 카테고리 비율)
    if hourly_df is not None and not hourly_df.empty and "hour" in hourly_df.columns:
        try:
            value_col = None
            for c in ("workers", "count"):
                if c in hourly_df.columns:
                    value_col = c
                    break
            if value_col:
                by_hour = hourly_df.groupby("hour")[value_col].sum()
            else:
                by_hour = hourly_df.groupby("hour").size()
            peak_hour = int(by_hour.idxmax())
            ctx["peak_hour"] = peak_hour

            # WORK / TRANSIT / REST / OTHER 총합
            if "category" in hourly_df.columns and value_col:
                cat_totals = hourly_df.groupby("category")[value_col].sum()
                total = int(cat_totals.sum())
                if total > 0:
                    ctx["category_distribution"] = {
                        str(c): _r(cat_totals.get(c, 0) / total, 3)
                        for c in ("WORK", "TRANSIT", "REST", "OTHER")
                        if c in cat_totals.index
                    }
        except Exception as e:
            log.debug("hourly_df aggregation failed: %s", e)

    return ctx


# ═══════════════════════════════════════════════════════════════════
# T-17 Productivity Analyst
# ═══════════════════════════════════════════════════════════════════

def build_productivity_context(
    *,
    sector_id: str,
    date_str: str | None,
    worker_df: pd.DataFrame,
    space_df: pd.DataFrame | None = None,
    date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """
    🏭 생산성 탭용 context.

    핵심:
      - 층별(building_floor) EWI/CRE 평균 Top/Bottom 3 (space_df)
      - 전체 평균 EWI/CRE/active_ratio
      - 헬멧 방치 의심 인원수 (work_zone + low_active_min 기반 추정)
      - coverage_ratio 낮은 구역 언급
    """
    ctx: dict[str, Any] = {
        "sector_id": sector_id,
        "date": date_str,
        "date_range": list(date_range) if date_range else None,
    }

    if worker_df is None or worker_df.empty:
        return ctx

    # ── 전체 평균 ─────────────────────────────────────────────────
    def _safe_mean(col: str) -> float | None:
        if col not in worker_df.columns:
            return None
        v = pd.to_numeric(worker_df[col], errors="coerce").dropna()
        return _r(v.mean()) if len(v) else None

    # EWI 는 신뢰 가능한 것만
    if "ewi_reliable" in worker_df.columns:
        rel_df = worker_df[worker_df["ewi_reliable"] == True]  # noqa: E712
    else:
        rel_df = worker_df

    def _safe_mean_on(df_: pd.DataFrame, col: str) -> float | None:
        if col not in df_.columns:
            return None
        v = pd.to_numeric(df_[col], errors="coerce").dropna()
        return _r(v.mean()) if len(v) else None

    ctx["overall"] = {
        "workers":            int(worker_df["user_no"].nunique() if "user_no" in worker_df.columns else len(worker_df)),
        "reliable_workers":   int(len(rel_df)),
        "avg_ewi":            _safe_mean_on(rel_df, "ewi"),
        "avg_cre":            _safe_mean("cre"),
        "avg_sii":            _safe_mean("sii"),
        "avg_fatigue":        _safe_mean("fatigue_score"),
    }

    # ── 헬멧 방치 의심 (M1 구현: work_zone + active_ratio < 0.40 + 30분+ 연속)
    # worker_df 레벨에서는 근사: low_active_min >= 30 && work_zone_minutes > 0
    helmet_count = 0
    if "low_active_min" in worker_df.columns and "work_zone_minutes" in worker_df.columns:
        low = pd.to_numeric(worker_df["low_active_min"], errors="coerce").fillna(0)
        wz = pd.to_numeric(worker_df["work_zone_minutes"], errors="coerce").fillna(0)
        helmet_count = int(((low >= 30) & (wz > 60)).sum())
    ctx["helmet_abandon_suspect_count"] = helmet_count

    # ── 층/구역별 Top/Bottom 3 (space_df 기반) ─────────────────────
    if space_df is not None and not space_df.empty:
        df = space_df.copy()
        # 집계 가능한 key 찾기
        key_col = None
        for c in ("building_floor", "building", "locus_name", "locus_id"):
            if c in df.columns:
                key_col = c
                break
        ewi_col = "avg_ewi" if "avg_ewi" in df.columns else ("ewi" if "ewi" in df.columns else None)
        cov_col = "coverage_ratio" if "coverage_ratio" in df.columns else None

        if key_col and ewi_col:
            agg_df = (
                df.groupby(key_col)
                .agg(
                    avg_ewi=(ewi_col, "mean"),
                    total_workers=("total_workers" if "total_workers" in df.columns else ewi_col, "sum" if "total_workers" in df.columns else "count"),
                )
                .reset_index()
                .dropna(subset=["avg_ewi"])
                .sort_values("avg_ewi", ascending=False)
            )
            if len(agg_df) >= 2:
                top = agg_df.head(3)
                bot = agg_df.tail(3).iloc[::-1]  # 역순
                ctx["floors_top_3_by_ewi"] = [
                    {"zone": str(r[key_col]), "avg_ewi": _r(r["avg_ewi"]),
                     "workers": int(r.get("total_workers", 0) or 0)}
                    for _, r in top.iterrows()
                ]
                ctx["floors_bottom_3_by_ewi"] = [
                    {"zone": str(r[key_col]), "avg_ewi": _r(r["avg_ewi"]),
                     "workers": int(r.get("total_workers", 0) or 0)}
                    for _, r in bot.iterrows()
                ]

            # low coverage 구역 (coverage < 30%)
            if cov_col:
                low_cov = df[pd.to_numeric(df[cov_col], errors="coerce") < 0.3]
                ctx["low_coverage_zones"] = int(low_cov[key_col].nunique()) if key_col else int(len(low_cov))

    return ctx


# ═══════════════════════════════════════════════════════════════════
# T-18 Safety Analyst
# ═══════════════════════════════════════════════════════════════════

def build_safety_context(
    *,
    sector_id: str,
    date_str: str | None,
    worker_df: pd.DataFrame,
    space_df: pd.DataFrame | None = None,
    date_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """
    🦺 안전성 탭용 context.

    핵심:
      - 고위험 구역 Top 3 (CRE 기준)
      - 전체 CRE/SII 평균, 고위험 인원수
      - 밀폐공간(confined_minutes>0), 고압(high_voltage_minutes>0) 노출
      - alone_ratio 높은 작업자 수 (고립작업 의심)
    """
    ctx: dict[str, Any] = {
        "sector_id": sector_id,
        "date": date_str,
        "date_range": list(date_range) if date_range else None,
    }

    if worker_df is None or worker_df.empty:
        return ctx

    def _safe_mean(col: str) -> float | None:
        if col not in worker_df.columns:
            return None
        v = pd.to_numeric(worker_df[col], errors="coerce").dropna()
        return _r(v.mean()) if len(v) else None

    ctx["overall"] = {
        "workers":       int(worker_df["user_no"].nunique() if "user_no" in worker_df.columns else len(worker_df)),
        "avg_cre":       _safe_mean("cre"),
        "avg_sii":       _safe_mean("sii"),
        "avg_fatigue":   _safe_mean("fatigue_score"),
    }

    # 고위험 인원수
    if "cre" in worker_df.columns:
        cre = pd.to_numeric(worker_df["cre"], errors="coerce").fillna(0)
        ctx["high_cre_count"] = int((cre >= 0.5).sum())
    if "sii" in worker_df.columns:
        sii = pd.to_numeric(worker_df["sii"], errors="coerce").fillna(0)
        ctx["high_sii_count"] = int((sii >= 0.5).sum())

    # 밀폐/고압 노출
    if "confined_minutes" in worker_df.columns:
        cm = pd.to_numeric(worker_df["confined_minutes"], errors="coerce").fillna(0)
        ctx["confined_exposure"] = {
            "total_min":   int(cm.sum()),
            "workers":     int((cm > 0).sum()),
            "long_stay":   int((cm >= 60).sum()),  # 1시간 이상
        }
    if "high_voltage_minutes" in worker_df.columns:
        hv = pd.to_numeric(worker_df["high_voltage_minutes"], errors="coerce").fillna(0)
        ctx["high_voltage_exposure"] = {
            "total_min":  int(hv.sum()),
            "workers":    int((hv > 0).sum()),
            "long_stay":  int((hv >= 60).sum()),
        }

    # 고립작업
    if "alone_ratio" in worker_df.columns:
        ar = pd.to_numeric(worker_df["alone_ratio"], errors="coerce").fillna(0)
        ctx["lone_work_suspect_count"] = int((ar >= 0.7).sum())

    # 고위험 구역 (space_df 기반)
    if space_df is not None and not space_df.empty:
        df = space_df.copy()
        cre_col = "avg_cre" if "avg_cre" in df.columns else ("cre" if "cre" in df.columns else None)
        haz_col = "hazard_level" if "hazard_level" in df.columns else None
        key_col = None
        for c in ("locus_name", "locus_id", "building_floor"):
            if c in df.columns:
                key_col = c
                break

        if cre_col and key_col:
            high = df.sort_values(cre_col, ascending=False).head(3)
            ctx["high_risk_zones_top3"] = [
                {
                    "zone":     str(r[key_col]),
                    "avg_cre":  _r(r[cre_col]),
                    "hazard":   str(r[haz_col]) if haz_col and haz_col in r else "unknown",
                }
                for _, r in high.iterrows()
            ]

    # 교대 분포 (안전 판단 보조)
    ctx["shift_distribution"] = _shift_distribution(worker_df)

    return ctx


# ═══════════════════════════════════════════════════════════════════
# T-19 Integrity Auditor
# ═══════════════════════════════════════════════════════════════════

def build_integrity_context(
    *,
    sector_id: str,
    date_str: str | None,
    stats_df: pd.DataFrame | None = None,
    meta_json: dict | None = None,
    worker_df: pd.DataFrame | None = None,
    physical_validation: dict | None = None,
) -> dict[str, Any]:
    """
    🔬 데이터 정합성 탭용 context.

    Args:
        stats_df: 일별 보정 통계 (gap_filled_rate, low_conf_rate, invalid_tr_rate 등)
        meta_json: meta.json 로드 결과 (validation.error 등)
        worker_df: 최신일 worker (helmet_abandoned 추정 보조)
        physical_validation: {impossible_count, cross_floor_count, ...} (physical_validator 결과)
    """
    ctx: dict[str, Any] = {
        "sector_id": sector_id,
        "date": date_str,
    }

    if stats_df is not None and not stats_df.empty:
        latest = stats_df.iloc[-1].to_dict() if "date" in stats_df.columns else {}
        ctx["latest_stats"] = {
            "gap_filled_pct":       _r(latest.get("gap_filled_rate", 0), 2),
            "low_confidence_pct":   _r(latest.get("low_conf_rate", 0), 2),
            "invalid_tr_pct":       _r(latest.get("invalid_tr_rate", 0), 2),
            "avg_signal_per_min":   _r(latest.get("avg_signal", 0), 1),
            "zero_signal_pct":      _r(latest.get("zero_signal_rate", 0), 2),
            "total_rows":           int(latest.get("total_rows", 0) or 0),
        }
        # 전체 기간 평균
        ctx["period_avg"] = {
            "gap_filled_pct":    _r(stats_df["gap_filled_rate"].mean(), 2) if "gap_filled_rate" in stats_df.columns else None,
            "low_conf_pct":      _r(stats_df["low_conf_rate"].mean(), 2) if "low_conf_rate" in stats_df.columns else None,
            "invalid_tr_pct":    _r(stats_df["invalid_tr_rate"].mean(), 2) if "invalid_tr_rate" in stats_df.columns else None,
            "days":              int(len(stats_df)),
        }

    if meta_json:
        val = meta_json.get("validation", {}) if isinstance(meta_json, dict) else {}
        ctx["validation"] = {
            "has_error":       bool(val.get("error")),
            "error_summary":   str(val.get("error", ""))[:120] if val.get("error") else None,
            "schema_version":  meta_json.get("schema_version") if isinstance(meta_json, dict) else None,
        }

    if physical_validation:
        ctx["physical_validation"] = {
            "impossible_count":   int(physical_validation.get("impossible_count", 0) or 0),
            "warn_fast_count":    int(physical_validation.get("warn_fast_count", 0) or 0),
            "total_checked":      int(physical_validation.get("total_checked", 0) or 0),
            "impossible_pct":     _r(physical_validation.get("impossible_pct", 0), 2),
        }

    # 헬멧 방치 의심 (M1)
    if worker_df is not None and not worker_df.empty:
        if "low_active_min" in worker_df.columns and "work_zone_minutes" in worker_df.columns:
            low = pd.to_numeric(worker_df["low_active_min"], errors="coerce").fillna(0)
            wz = pd.to_numeric(worker_df["work_zone_minutes"], errors="coerce").fillna(0)
            ctx["helmet_abandon_suspect_count"] = int(((low >= 30) & (wz > 60)).sum())

    return ctx


# ═══════════════════════════════════════════════════════════════════
# Deep Space / Anomaly / Prediction (T-20)
# ═══════════════════════════════════════════════════════════════════

def build_deep_space_context(
    *,
    sector_id: str,
    summary: str,
    congested_spaces: str,
    locus_context: str,
    current_locus: str | None = None,
    predictions: str | None = None,
) -> dict[str, Any]:
    """
    🧠 Deep Space 시뮬레이션/예측용 context.

    기존 `cached_spatial_insight` 인자를 구조화.
    """
    ctx: dict[str, Any] = {
        "sector_id":          sector_id,
        "situation_summary":  str(summary)[:400] if summary else "",
        "congested_spaces":   str(congested_spaces)[:200] if congested_spaces else "",
        "locus_context":      str(locus_context)[:400] if locus_context else "",
    }
    if current_locus:
        ctx["current_locus"] = str(current_locus)[:100]
    if predictions:
        ctx["predictions_top_k"] = str(predictions)[:200]
    return ctx


def build_anomaly_context(
    *,
    sector_id: str,
    anomaly_description: str,
    perplexity: str,
    locus_context: str,
) -> dict[str, Any]:
    """🧠 이상 이동 해설용 context."""
    return {
        "sector_id":            sector_id,
        "anomaly_description":  str(anomaly_description)[:300] if anomaly_description else "",
        "perplexity":           str(perplexity)[:50],
        "locus_context":        str(locus_context)[:400] if locus_context else "",
    }


def build_prediction_context(
    *,
    sector_id: str,
    current_locus: str,
    predictions: str,
    locus_context: str,
) -> dict[str, Any]:
    """🧠 이동 예측 해설용 context."""
    return {
        "sector_id":       sector_id,
        "current_locus":   str(current_locus)[:100],
        "predictions":     str(predictions)[:300],
        "locus_context":   str(locus_context)[:400] if locus_context else "",
    }


# ═══════════════════════════════════════════════════════════════════
# V-10 fix — report_context용
# ═══════════════════════════════════════════════════════════════════

def build_report_section_context(
    *,
    sector_id: str,
    section_name: str,
    data_summary: str,
    static_context: str,
) -> dict[str, Any]:
    """
    📄 PDF 리포트 섹션별 맥락 생성용 — report_context._call_llm_for_context 대체.
    """
    return {
        "sector_id":       sector_id,
        "section":         str(section_name)[:60],
        "data_summary":    str(data_summary)[:500],
        "static_context":  str(static_context)[:300],
    }


__all__ = [
    "build_overview_context",
    "build_zone_time_context",
    "build_productivity_context",
    "build_safety_context",
    "build_integrity_context",
    "build_deep_space_context",
    "build_anomaly_context",
    "build_prediction_context",
    "build_report_section_context",
]
