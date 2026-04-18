"""
Daily Tab - Summary Section
===========================
요약 탭 관련 함수들.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.styles import (
    metric_card, metric_card_sm, section_header, sub_header,
    insight_card, PLOTLY_DARK, PLOTLY_LEGEND,
)
from src.dashboard.llm_deepcon import (
    cached_daily_summary, render_data_comment, is_llm_available,
)
from src.dashboard.components import (
    alert_center,
    validation_badge,
    build_space_alerts,
)
from src.utils.anonymizer import mask_name
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH, CRE_CRITICAL, CRE_MEDIUM, EWI_HIGH
# ★ 색상 단일 소스
from src.dashboard.styles import CHART_COLORS

_DARK = PLOTLY_DARK
_LEG = PLOTLY_LEGEND

# T-Ward 보유율 경고 임계값
TWARD_RATE_DANGER = 50    # 위험 (빨간)
TWARD_RATE_WARNING = 65   # 경고 (노란)


# ══════════════════════════════════════════════════════════════════
# Alert Center용 알림 리스트 구축
# ══════════════════════════════════════════════════════════════════

def _build_alert_list(meta, worker_df, space_df, locus_dict, has_ewi, has_cre) -> list[dict]:
    """Alert Center용 알림 리스트 구축.

    Returns:
        list[dict]: 각 dict는 {severity, title, detail, count} 포함
        severity: "critical" | "high" | "medium" | "low"
    """
    alerts = []

    # 1) CRE Critical — constants.py의 CRE_CRITICAL 기준 (단일 진실 공급원)
    if has_cre and "cre" in worker_df.columns:
        high_cre = worker_df[worker_df["cre"].fillna(0) >= CRE_CRITICAL]
        if len(high_cre) > 0:
            alerts.append({
                "severity": "critical",
                "title": f"CRE >= {CRE_CRITICAL:.1f} 고위험 {len(high_cre)}명",
                "detail": "즉시 휴식 권고 필요",
                "count": len(high_cre),
            })

    # 2) 밀폐공간 2시간 이상 (Critical)
    if "confined_minutes" in worker_df.columns:
        long_confined = worker_df[worker_df["confined_minutes"] >= 120]
        if len(long_confined) > 0:
            alerts.append({
                "severity": "critical",
                "title": f"밀폐공간 2h+ 체류 {len(long_confined)}명",
                "detail": "즉시 확인 필요",
                "count": len(long_confined),
            })

    # 3) 장시간 근무 12h+ (High)
    if "work_minutes" in worker_df.columns:
        overtime = worker_df[worker_df["work_minutes"] >= 720]
        if len(overtime) > 0:
            top_name = mask_name(overtime.nlargest(1, "work_minutes").iloc[0].get("user_name", "-"))
            top_hours = overtime["work_minutes"].max() / 60
            alerts.append({
                "severity": "high",
                "title": f"장시간 근무 {len(overtime)}명 (12h+)",
                "detail": f"최장: {top_name} ({top_hours:.1f}h)",
                "count": len(overtime),
            })

    # 4) 고밀집 공간 (High)
    if not space_df.empty and "unique_workers" in space_df.columns:
        top_spaces = space_df.nlargest(1, "unique_workers")
        if len(top_spaces) > 0:
            top_space = top_spaces.iloc[0]
            if top_space["unique_workers"] >= 200:
                name = locus_dict.get(top_space["locus_id"], {}).get("locus_name", top_space["locus_id"])
                alerts.append({
                    "severity": "high",
                    "title": f"고밀집: {name} ({int(top_space['unique_workers'])}명)",
                    "detail": "혼잡도 탭에서 확인",
                    "count": 1,
                })

    # 5) T-Ward 보유율 저조 (Medium)
    total_access = meta.get("total_workers_access", 0)
    total_tward = meta.get("tward_holders", meta.get("total_workers_move", 0))
    if total_access > 0:
        rate = total_tward / total_access * 100
        if rate < TWARD_RATE_WARNING:
            alerts.append({
                "severity": "medium",
                "title": f"T-Ward 보유율 {rate:.1f}%",
                "detail": f"{total_access:,}명 중 {total_tward:,}명 보유",
                "count": 1,
            })

    return alerts


def _build_alert_list_with_journey(
    meta, worker_df, space_df, locus_dict, has_ewi, has_cre, journey_df=None
) -> list[dict]:
    """
    기존 알림 + enriched locus 기반 공간 알림 통합.
    """
    alerts = _build_alert_list(meta, worker_df, space_df, locus_dict, has_ewi, has_cre)

    if journey_df is not None and not journey_df.empty and locus_dict:
        space_alerts = build_space_alerts(journey_df, locus_dict, space_df)
        alerts.extend(space_alerts)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(alerts, key=lambda x: severity_order.get(x.get("severity", "low"), 3))


# ══════════════════════════════════════════════════════════════════
# 건설현장 핵심 현황 패널
# ══════════════════════════════════════════════════════════════════

def _render_site_status_panel(meta, worker_df, space_df, locus_dict, has_ewi, has_cre):
    """
    건설현장 관리자에게 즉시 필요한 정보를 알림 카드로 요약.
    """
    import numpy as np
    alerts = []  # (icon, title, detail, color, severity)

    # 1) 장시간 근무자 (12시간 이상)
    if "work_minutes" in worker_df.columns:
        overtime = worker_df[worker_df["work_minutes"] >= 720]
        if len(overtime) > 0:
            top3 = overtime.nlargest(3, "work_minutes")
            names = ", ".join(
                f"{mask_name(r['user_name'])}({r['work_minutes']/60:.1f}h)"
                for _, r in top3.iterrows()
            )
            alerts.append((
                "clock", f"장시간 근무 {len(overtime)}명 (12h+)",
                f"상위: {names}",
                "#FF6B35", 3,
            ))

    # 2) 밀폐공간 진입
    if "confined_minutes" in worker_df.columns:
        confined = worker_df[worker_df["confined_minutes"] > 0]
        if len(confined) > 0:
            long_conf = confined[confined["confined_minutes"] >= 30]
            detail = f"총 {len(confined)}명 진입"
            if len(long_conf) > 0:
                detail += f" | {len(long_conf)}명 30분 이상 체류"
            alerts.append(("lock", "밀폐공간 진입 감지", detail, "#FF4C4C", 4))

    # 3) 고위험 작업자 (CRE >= CRE_HIGH) — constants.py 기준
    if has_cre:
        high_cre = worker_df[worker_df["cre"].fillna(0) >= CRE_HIGH]
        if len(high_cre) > 0:
            top_co = high_cre["company_name"].value_counts().head(3)
            co_str = ", ".join(f"{co}({n}명)" for co, n in top_co.items())
            alerts.append((
                "warning", f"고위험 작업자 {len(high_cre)}명 (CRE>={CRE_HIGH:.1f})",
                f"주요 업체: {co_str}",
                "#FF4C4C", 4,
            ))

    # 4) T-Ward 보유율 저조
    total_access = meta.get("total_workers_access", 0)
    total_tward = meta.get("tward_holders", meta.get("total_workers_move", 0))
    tward_rate = round(total_tward / total_access * 100, 1) if total_access > 0 else 0
    if tward_rate < TWARD_RATE_DANGER and total_access > 0:
        alerts.append((
            "signal", f"T-Ward 보유율 저조: {tward_rate:.1f}%",
            f"출입 {total_access:,}명 중 T-Ward 보유 {total_tward:,}명",
            "#FFB300", 2,
        ))

    # 5) 공간 과밀
    if not space_df.empty and "unique_workers" in space_df.columns:
        top_space = space_df.nlargest(1, "unique_workers").iloc[0]
        top_name = locus_dict.get(top_space["locus_id"], {}).get("locus_name", top_space["locus_id"])
        uw = top_space["unique_workers"]
        if uw >= 200:
            alerts.append((
                "building", f"고밀집 공간: {top_name} ({uw}명)",
                "혼잡도 분석 탭에서 상세 확인",
                "#FF6B35", 3,
            ))

    # 렌더링
    if not alerts:
        st.markdown(
            "<div style='background:#0D2A1A; border:1px solid #00C897; "
            "border-radius:10px; padding:12px 16px; margin-bottom:12px;'>"
            "<span style='color:#00C897; font-size:0.92rem; font-weight:600;'>"
            "OK 특이사항 없음</span>"
            "<span style='color:#7A8FA6; font-size:0.82rem; margin-left:12px;'>"
            "안전/근무/혼잡 기준 모두 정상 범위</span></div>",
            unsafe_allow_html=True,
        )
        return

    # 심각도 순 정렬
    alerts.sort(key=lambda x: x[4], reverse=True)

    # 카드 렌더링 (최대 4개)
    for icon, title, detail, color, _ in alerts[:4]:
        st.markdown(
            f"<div style='background:#111820; border-left:4px solid {color}; "
            f"border-radius:8px; padding:10px 14px; margin:6px 0;'>"
            f"<div style='color:{color}; font-size:0.92rem; font-weight:600;'>"
            f"{title}</div>"
            f"<div style='color:#9AB5D4; font-size:0.82rem; margin-top:3px;'>"
            f"{detail}</div></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Journey 보정 + 데이터 검증 배너
# ══════════════════════════════════════════════════════════════════

def _render_correction_banner(meta: dict):
    """Journey 보정 상태 + 데이터 검증 배너."""
    corrected = meta.get("journey_corrected", False)
    c_count = meta.get("corrected_records", 0)
    c_ratio = meta.get("correction_ratio", 0.0)
    w_corrected = meta.get("workers_corrected", None)
    unmapped = meta.get("unmapped_ratio", 0.0)

    if corrected:
        w_str = f" | 보정 작업자: <b style='color:#00C897'>{w_corrected}명</b>" if w_corrected else ""
        st.markdown(
            f"<div style='background:#0D2A1A; border:1px solid #00C897; border-radius:8px; "
            f"padding:10px 16px; margin-top:12px; font-size:0.83rem; color:#C8D6E8;'>"
            f"OK <b>Journey 보정 적용됨</b> - "
            f"보정: <b style='color:#00C897'>{c_count:,}건 ({c_ratio:.1f}%)</b>"
            f"{w_str} | Unmapped: {unmapped:.1f}%</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#1A1A0D; border:1px solid #3A3A1A; border-radius:8px; "
            f"padding:10px 16px; margin-top:12px; font-size:0.83rem; color:#7A8FA6;'>"
            f"Journey 보정 미적용 (Raw 데이터) | "
            f"Unmapped: <b style='color:#FFB300'>{unmapped:.1f}%</b></div>",
            unsafe_allow_html=True,
        )

    # 데이터 검증 결과 뱃지
    validation = meta.get("validation", {})
    if validation and not validation.get("error"):
        level = validation.get("overall_level", "warning")
        score = validation.get("overall_score", 0)
        summary = validation.get("summary", "")
        badge_html = validation_badge(level, score)
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; margin-top:8px;'>"
            f"<span style='color:#7A8FA6; font-size:0.82rem;'>데이터 품질:</span>"
            f"{badge_html}"
            f"<span style='color:#7A8FA6; font-size:0.78rem;'>{summary}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# 인사이트 카드 렌더링
# ══════════════════════════════════════════════════════════════════

def _render_insight_section(date_str: str, sector_id: str):
    """인사이트 엔진 결과를 카드로 표시."""
    report = st.session_state.get("_insights")
    if report is None or not hasattr(report, "insights") or not report.insights:
        return

    from src.intelligence.models import SEVERITY_BADGE, Severity
    sev_counts = {}
    for i in report.insights:
        badge_icon, badge_label = SEVERITY_BADGE.get(i.severity, ("o", "?"))
        sev_counts[badge_label] = sev_counts.get(badge_label, 0) + 1

    badges_html = " ".join(
        f"<span style='background:{'#FF4C4C' if k=='긴급' else '#FF8C00' if k=='경고' else '#FFB300' if k=='주의' else '#4A90D9'};"
        f"color:white; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; margin-right:6px;'>"
        f"{k} {v}</span>"
        for k, v in sev_counts.items()
    )

    st.markdown(
        f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; border-left:4px solid #00AEEF; "
        f"border-radius:10px; padding:16px 20px; margin-bottom:16px;'>"
        f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>"
        f"<div style='color:#00AEEF; font-size:0.95rem; font-weight:700;'>"
        f"데이터 기반 알림 ({len(report.insights)}건)</div>"
        f"<div>{badges_html}</div></div>",
        unsafe_allow_html=True,
    )

    for ins in report.top(5):
        badge_icon, badge_label = ins.badge
        cat_icon, cat_label = ins.category_label
        sev_key = "critical" if ins.severity >= Severity.CRITICAL else (
            "high" if ins.severity >= Severity.HIGH else "medium")
        st.markdown(
            insight_card(
                title=f"{badge_icon} {cat_icon} {ins.title}",
                description=ins.description,
                detail=f">> {ins.recommendation}" if ins.recommendation else "",
                severity=sev_key,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 요약 탭 메인 렌더
# ══════════════════════════════════════════════════════════════════

def render_summary(meta, worker_df, space_df, locus_dict, date_str, sid, has_ewi, has_cre):
    """요약 탭 전체 렌더링."""
    total_access = meta.get("total_workers_access", 0)
    total_tward = meta.get("tward_holders", meta.get("total_workers_move", 0))
    tward_rate = round(total_tward / total_access * 100, 1) if total_access > 0 else 0

    # Alert Center 최상단
    alerts = _build_alert_list(meta, worker_df, space_df, locus_dict, has_ewi, has_cre)
    if alerts:
        alert_center(alerts)

    # Validation Badge 표시
    validation_results = meta.get("validation_results", {})
    if validation_results:
        validation_badge(validation_results)

    st.markdown(section_header(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} 현황"), unsafe_allow_html=True)

    # 건설현장 핵심 현황 패널
    _render_site_status_panel(meta, worker_df, space_df, locus_dict, has_ewi, has_cre)

    # 핵심 인사이트 (Intelligence Engine)
    _render_insight_section(date_str, sid)

    # KPI 카드
    cols = st.columns(6)
    kpis = [
        ("생체인식 출입인원", f"{total_access:,}명", None),
        ("T-Ward 보유자", f"{total_tward:,}명", None),
        ("T-Ward 보유율 (보유자/출입)", f"{tward_rate:.1f}%", "#F5A623" if tward_rate < TWARD_RATE_WARNING else "#00C897"),
        ("참여 업체", f"{meta.get('companies', 0)}개", None),
        ("평균 근무", f"{worker_df['work_minutes'].mean()/60:.1f}h" if "work_minutes" in worker_df.columns else "-", None),
        ("밀폐공간 진입", f"{int((worker_df.get('confined_minutes', pd.Series(0)) > 0).sum())}명",
         "#FF4C4C" if (worker_df.get("confined_minutes", pd.Series(0)) > 0).any() else "#00C897"),
    ]
    for col, (label, val, color) in zip(cols, kpis):
        with col:
            if color:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>"
                    f"{val}</div><div class='metric-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(metric_card(label, val), unsafe_allow_html=True)

    # EWI/CRE 요약 KPI
    if has_ewi or has_cre:
        st.markdown("")
        ec_cols = st.columns(4)
        if has_ewi:
            avg_ewi = round(worker_df["ewi"].mean(), 3)
            with ec_cols[0]:
                color = CHART_COLORS["critical"] if avg_ewi >= EWI_HIGH else (CHART_COLORS["medium"] if avg_ewi >= 0.2 else CHART_COLORS["rest"])
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{avg_ewi:.3f}</div>"
                    f"<div class='metric-label'>평균 EWI (생산성)</div></div>", unsafe_allow_html=True)
            with ec_cols[1]:
                hi_ewi = int((worker_df["ewi"].fillna(0) >= EWI_HIGH).sum())
                st.markdown(metric_card(f"고강도 작업자 (EWI>={EWI_HIGH:.1f})", f"{hi_ewi}명"), unsafe_allow_html=True)
        if has_cre:
            avg_cre = round(worker_df["cre"].mean(), 3)
            with ec_cols[2]:
                color = CHART_COLORS["critical"] if avg_cre >= CRE_HIGH else (CHART_COLORS["medium"] if avg_cre >= CRE_MEDIUM else CHART_COLORS["rest"])
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{avg_cre:.3f}</div>"
                    f"<div class='metric-label'>평균 CRE (위험노출)</div></div>", unsafe_allow_html=True)
            with ec_cols[3]:
                hi_cre = int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum())
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{'#FF4C4C' if hi_cre>0 else '#00C897'}'>"
                    f"{hi_cre}명</div><div class='metric-label'>고위험 작업자 (CRE>={CRE_HIGH:.1f})</div></div>",
                    unsafe_allow_html=True)

    # 날씨 정보 (선택적)
    try:
        from src.pipeline.weather import fetch_weather
        _w_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        _w_df = fetch_weather(_w_date, _w_date)
        if not _w_df.empty:
            _w_row = _w_df.iloc[0]
            _WEATHER_EMOJI = {"Sunny": "sun", "Rain": "rain", "Snow": "snow", "Unknown": "cloud"}
            _w_tag = _w_row.get("weather", "Unknown")
            _t_max = _w_row.get("temp_max")
            _t_min = _w_row.get("temp_min")
            _precip = _w_row.get("precipitation", 0)
            _snow = _w_row.get("snowfall", 0)

            parts = [f"{_w_tag}"]
            if _t_max is not None and _t_min is not None:
                parts.append(f"{_t_min:.0f}~{_t_max:.0f}C")
            if _precip and float(_precip) > 0:
                parts.append(f"강수 {float(_precip):.1f}mm")
            if _snow and float(_snow) > 0:
                parts.append(f"적설 {float(_snow):.1f}cm")

            st.markdown(
                f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; border-radius:8px; "
                f"padding:8px 14px; margin:6px 0; display:inline-block;'>"
                f"<span style='color:#C8D6E8; font-size:0.88rem;'>"
                f"{' | '.join(parts)}</span></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Journey 보정 배너
    _render_correction_banner(meta)

    st.divider()

    # 공간 상위 현황
    if not space_df.empty:
        top5 = space_df.nlargest(5, "total_person_minutes")
        top5["locus_name"] = top5["locus_id"].map(
            lambda lid: locus_dict.get(lid, {}).get("locus_name", lid))

        with st.expander("i 공간별 체류 지표 설명"):
            st.markdown("""
| 지표 | 의미 | 예시 |
|------|------|------|
| **체류량 (인/분)** | 해당 공간에서 모든 작업자가 체류한 총 시간의 합 | 100명 x 10분 = 1,000 인/분 |
| **인원 (명)** | 해당 날짜에 1회 이상 방문한 고유 작업자 수 | 순방문자 수 |
| **평균 체류 (분)** | 총 체류량 / 방문 인원 | 1인당 평균 머문 시간 |

> 체류량이 높을수록 작업자들이 오래 머무는 핵심 공간이며, 안전/생산성 관리의 우선 대상입니다.
            """)

        cols2 = st.columns(5)
        for col, (_, r) in zip(cols2, top5.iterrows()):
            pm = r['total_person_minutes']
            uw = r['unique_workers']
            avg_min = round(pm / uw, 1) if uw > 0 else 0
            if pm >= 10000:
                pm_disp = f"{pm/10000:.1f}만"
            else:
                pm_disp = f"{pm:,}"
            with col:
                st.markdown(
                    f"<div style='background:#111820; border:1px solid #2A3A4A; border-radius:8px; "
                    f"padding:10px; text-align:center;'>"
                    f"<div style='color:#00AEEF; font-size:0.75rem; font-weight:600;'>{r['locus_name']}</div>"
                    f"<div style='color:#F5A623; font-size:1.2rem; font-weight:700;'>{pm_disp}</div>"
                    f"<div style='color:#C8D6E8; font-size:0.68rem; margin:1px 0;'>인/분 (체류량)</div>"
                    f"<div style='color:#7A8FA6; font-size:0.68rem;'>worker {uw:,}명 | 평균 {avg_min}분</div></div>",
                    unsafe_allow_html=True)

    # 데이터 품질 (BLE 커버리지)
    if "ble_coverage" in worker_df.columns:
        cov = worker_df["ble_coverage"].value_counts()
        total_w = len(worker_df)
        normal = cov.get("정상", 0)
        partial = cov.get("부분음영", 0)
        shadow = cov.get("음영", 0)
        unmeas = cov.get("미측정", 0)
        reliable_pct = round(normal / total_w * 100, 1) if total_w else 0
        problem_pct = round((shadow + unmeas) / total_w * 100, 1) if total_w else 0

        with st.expander(
            f"signal 데이터 품질 - BLE 커버리지 정상 {reliable_pct}% | 음영+미측정 {problem_pct}%",
            expanded=problem_pct > 30,
        ):
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.markdown(metric_card_sm("정상", f"{normal:,}명", "#00C897"), unsafe_allow_html=True)
            with q2:
                st.markdown(metric_card_sm("부분음영", f"{partial:,}명", "#FFB300"), unsafe_allow_html=True)
            with q3:
                st.markdown(metric_card_sm("음영", f"{shadow:,}명", "#FF6B35"), unsafe_allow_html=True)
            with q4:
                st.markdown(metric_card_sm("미측정", f"{unmeas:,}명", "#FF4C4C"), unsafe_allow_html=True)

            st.caption(
                "정상: BLE 커버리지 80%+ | 부분음영: 50~80% | "
                "음영: 20~50% | 미측정: 20% 미만 (EWI/CRE 신뢰 불가)"
            )

            if unmeas > 0:
                st.warning(
                    f"**미측정 작업자 {unmeas}명** - 출퇴근은 확인되나 BLE 신호 거의 없음. "
                    f"T-Ward 미착용, 센서 음영, 또는 장비 고장 가능성. "
                    f"이 작업자들의 EWI/CRE는 신뢰할 수 없습니다."
                )

    # 데이터 해석 (LLM 보조, 접이식)
    if is_llm_available():
        with st.expander("speech 데이터 해석 보기", expanded=False):
            result = cached_daily_summary(
                date_str=date_str, sector_id=sid,
                total_access=total_access, total_tward=total_tward, tward_rate=tward_rate,
                avg_ewi=round(worker_df["ewi"].mean(), 3) if has_ewi else 0,
                avg_cre=round(worker_df["cre"].mean(), 3) if has_cre else 0,
                high_cre_count=int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum()) if has_cre else 0,
                confined_workers=int((worker_df.get("confined_minutes", pd.Series(0)) > 0).sum()),
                hv_workers=int((worker_df.get("high_voltage_minutes", pd.Series(0)) > 0).sum()),
                top_spaces="",
                companies=meta.get("companies", 0),
            )
            render_data_comment("오늘의 수치 요약", result)
