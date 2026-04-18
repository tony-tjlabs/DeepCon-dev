"""
integrity/sanity_check.py — 이상 패턴 · 비상식 패턴 탐지
========================================================
tab3: `⚠️ 이상 패턴 감지` — 규칙 5종 (gap_ratio, invalid_tr, helmet,
       DEEP_INACTIVE, zero_signal)
tab5: `🚨 비상식 패턴 (Sanity)` — 파이프라인 sanity_checker 결과 렌더

두 서브탭 모두 자동 탐지 → 테이블/차트 → CSV 다운로드 흐름.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    COLORS,
    CHART_COLORS,
    PLOTLY_DARK,
    metric_card,
    metric_card_sm,
    section_header,
    sub_header,
)
from src.dashboard.date_utils import get_date_selector

logger = logging.getLogger(__name__)


def _render_anomaly_detection(sector_id: str) -> None:
    """이상 감지 — 비정상 패턴 자동 플래그."""
    from src.pipeline.cache_manager import detect_processed_dates

    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    st.markdown(section_header("비정상 패턴 자동 감지"), unsafe_allow_html=True)
    st.caption(
        "⚠ 전문가 검토 필요 — 아래 규칙에 해당하는 작업자를 자동 플래그합니다. "
        "오검출 가능성이 있으므로 반드시 개별 타임라인으로 확인하세요."
    )

    # 날짜 선택
    date_str = get_date_selector(
        list(reversed(processed)),
        key="anomaly_date",
        default_index=0,
        label="분석 날짜",
        show_label=True,
    ) or processed[-1]

    paths = cfg.get_sector_paths(sector_id)
    journey_path = paths["processed_dir"] / date_str / "journey.parquet"
    worker_path  = paths["processed_dir"] / date_str / "worker.parquet"

    if not journey_path.exists() or not worker_path.exists():
        st.error("parquet 파일이 없습니다.")
        return

    with st.spinner("이상 패턴 감지 중..."):
        anomalies = _detect_anomalies(sector_id, date_str,
                                       str(journey_path), str(worker_path))

    if anomalies.empty:
        st.success("🎉 감지된 이상 패턴이 없습니다.")
        return

    # KPI: 유형별 카운트
    type_counts = anomalies["anomaly_type"].value_counts()
    n_types = len(type_counts)
    cols = st.columns(max(1, min(4, n_types)))
    for i, (atype, cnt) in enumerate(type_counts.items()):
        with cols[i % len(cols)]:
            st.markdown(metric_card_sm(atype, f"{cnt}건",
                                       color=COLORS["warning"]),
                        unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # 결과 테이블
    st.markdown(sub_header(f"감지 결과 ({len(anomalies)}건)"),
                unsafe_allow_html=True)
    st.dataframe(
        anomalies[[
            "user_no", "user_name", "company_name", "anomaly_type",
            "severity", "detail", "gap_filled_pct", "invalid_tr_count",
            "work_minutes",
        ]].round(1),
        use_container_width=True, height=420,
    )

    # 다운로드
    csv = anomalies.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ 이상 감지 결과 CSV 다운로드",
        data=csv,
        file_name=f"anomalies_{date_str}.csv",
        mime="text/csv",
    )


@st.cache_data(show_spinner=False, ttl=600)
def _detect_anomalies(
    sector_id: str, date_str: str,
    journey_path: str, worker_path: str,
) -> pd.DataFrame:
    """
    규칙 기반 이상 감지.

    규칙:
      1. 음영 과다           — gap_ratio > 0.80
      2. 비유효 전이 다수    — invalid_tr_count ≥ 5
      3. 헬멧 방치 의심      — helmet_abandoned=True
      4. 장시간 DEEP_INACTIVE — deep_inactive_min ≥ 480 (8시간)
      5. 0-signal 연속 초과  — zero_signal_pct ≥ 70 (7할 이상 무신호)

    ★ 성능: worker.parquet 컬럼 프루닝 + journey.parquet 단일 통합 로드
       (기존: jstats용 3컬럼 + extra용 3컬럼 = 별개 2회 로드 → 단일 5컬럼 로드)
    """
    # worker_df: 판정에 필요한 컬럼만 로드
    worker_cols = [
        "user_no", "user_name", "company_name",
        "work_minutes", "gap_ratio",
        "helmet_abandoned",
    ]
    try:
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(worker_path).names)
        worker_cols = [c for c in worker_cols if c in avail]
        worker_df = pd.read_parquet(worker_path, columns=worker_cols)
    except Exception:
        worker_df = pd.read_parquet(worker_path)
    if worker_df.empty:
        return pd.DataFrame()

    # ★ journey.parquet 단일 로드로 jstats + extra 모두 계산 (중복 로드 제거)
    try:
        jdf = pd.read_parquet(
            journey_path,
            columns=["user_no", "is_gap_filled", "is_valid_transition",
                     "activity_level", "signal_count"],
        )
        # nullable boolean → bool (벡터화)
        gap_filled = jdf["is_gap_filled"].fillna(False).astype(bool)
        valid_tr   = jdf["is_valid_transition"].fillna(True).astype(bool)

        jdf["_gap"]     = gap_filled.astype(int)
        jdf["_invalid"] = (~valid_tr).astype(int)
        jdf["_deep"]    = (jdf["activity_level"] == "DEEP_INACTIVE").astype(int)
        jdf["_zero"]    = (jdf["signal_count"] == 0).astype(int)

        agg = jdf.groupby("user_no").agg(
            n_total=("signal_count", "count"),
            n_gap=("_gap", "sum"),
            n_invalid=("_invalid", "sum"),
            deep_inactive_min=("_deep", "sum"),
            zero_signal_count=("_zero", "sum"),
        ).reset_index()
        agg["gap_filled_pct"]   = agg["n_gap"] / agg["n_total"] * 100
        agg["invalid_tr_count"] = agg["n_invalid"]
        agg["zero_signal_pct"]  = agg["zero_signal_count"] / agg["n_total"] * 100
        jstats = agg[["user_no", "gap_filled_pct", "invalid_tr_count",
                      "deep_inactive_min", "zero_signal_pct"]]
    except Exception:
        jstats = pd.DataFrame(columns=["user_no", "gap_filled_pct", "invalid_tr_count",
                                         "deep_inactive_min", "zero_signal_pct"])

    merged = worker_df.merge(jstats, on="user_no", how="left")

    anomalies = []

    def _flag(row, atype, severity, detail):
        anomalies.append({
            "user_no":         int(row["user_no"]),
            "user_name":       str(row.get("user_name", "")),
            "company_name":    str(row.get("company_name", "미확인")),
            "anomaly_type":    atype,
            "severity":        severity,
            "detail":          detail,
            "gap_filled_pct":  float(row.get("gap_filled_pct", 0) or 0),
            "invalid_tr_count": int(row.get("invalid_tr_count", 0) or 0),
            "work_minutes":    float(row.get("work_minutes", 0) or 0),
        })

    # ★ 성능: iterrows 대신 itertuples (메모리 효율 + 속도 ~5-10배)
    # 필요 컬럼을 리스트로 준비 (없는 컬럼은 기본값 할당)
    _cols_needed = ["user_no", "user_name", "company_name",
                    "work_minutes", "gap_ratio", "helmet_abandoned",
                    "gap_filled_pct", "invalid_tr_count",
                    "deep_inactive_min", "zero_signal_pct"]
    for _c in _cols_needed:
        if _c not in merged.columns:
            merged[_c] = 0 if _c not in ("user_name", "company_name", "helmet_abandoned") else ("" if _c != "helmet_abandoned" else False)

    # float/int 결측 정리 (itertuples 시 NaN 분기 줄이기)
    for _c in ("gap_ratio", "invalid_tr_count", "deep_inactive_min",
               "zero_signal_pct", "work_minutes", "gap_filled_pct"):
        merged[_c] = merged[_c].fillna(0)
    merged["helmet_abandoned"] = merged["helmet_abandoned"].fillna(False).astype(bool)

    for t in merged[_cols_needed].itertuples(index=False, name="W"):
        gap_ratio = float(t.gap_ratio)
        inv_tr    = int(t.invalid_tr_count)
        helmet_ab = bool(t.helmet_abandoned)
        deep_min  = int(t.deep_inactive_min)
        zero_pct  = float(t.zero_signal_pct)
        work_min  = float(t.work_minutes)
        _row = {
            "user_no": int(t.user_no),
            "user_name": str(t.user_name or ""),
            "company_name": str(t.company_name or "미확인"),
            "gap_filled_pct": float(t.gap_filled_pct),
            "invalid_tr_count": inv_tr,
            "work_minutes": work_min,
        }

        # metrics.py의 ble_coverage 등급 "미측정" 기준(gap_ratio > 0.8)과 통일 (M3)
        if gap_ratio > 0.80:
            anomalies.append({**_row, "anomaly_type": "음영 과다 (>80%)", "severity": "HIGH",
                              "detail": f"gap_ratio={gap_ratio*100:.0f}%, BLE 미감지 대부분"})
        if inv_tr >= 5:
            anomalies.append({**_row, "anomaly_type": "비유효 전이 다수 (≥5건)", "severity": "MEDIUM",
                              "detail": f"{inv_tr}건의 물리적 비정상 전이"})
        if helmet_ab:
            anomalies.append({**_row, "anomaly_type": "헬멧 방치 의심", "severity": "HIGH",
                              "detail": "장시간 동일 locus + 저활성 → 헬멧만 남기고 이탈 가능성"})
        if deep_min >= 480 and work_min >= 480:
            anomalies.append({**_row, "anomaly_type": "장시간 DEEP_INACTIVE (≥8h)", "severity": "MEDIUM",
                              "detail": f"{deep_min}분 정지 / 체류 {work_min:.0f}분"})
        if zero_pct >= 70 and work_min >= 60:
            anomalies.append({**_row, "anomaly_type": "0-signal 연속 초과 (≥70%)", "severity": "HIGH",
                              "detail": f"{zero_pct:.0f}% 행이 signal_count=0 (T-Ward 미작동 의심)"})

    return pd.DataFrame(anomalies)

def _render_sanity_check(sector_id: str) -> None:
    """
    비상식 패턴 탐지 — 현실 물리/생리적 불가능 패턴 전문가 검토 도구.

    파이프라인 처리 시 worker.parquet에 저장된 sanity 컬럼을 읽어 표시.
    sanity 컬럼이 없는 구 parquet은 즉석에서 재계산(소급 적용).
    """
    import json
    from src.pipeline.cache_manager import detect_processed_dates

    # ── 인라인 집계 헬퍼 (sanity 컬럼이 worker.parquet에 이미 저장됨) ─
    def _sanity_summary(df: "pd.DataFrame") -> dict:
        n = len(df)
        sev = df.get("sanity_severity", pd.Series(dtype=str))
        suspicious = int((df["sanity_flag_count"] > 0).sum())
        rule_counts: dict[str, int] = {}
        for flags_json in df.get("sanity_flags", pd.Series(dtype=str)).dropna():
            try:
                for rule_id in json.loads(flags_json):
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "total": n, "suspicious": suspicious,
            "suspicious_pct": round(suspicious / max(n, 1) * 100, 1),
            "high":   int((sev == "HIGH").sum()),
            "medium": int((sev == "MEDIUM").sum()),
            "low":    int((sev == "LOW").sum()),
            "rule_counts": rule_counts,
        }

    # ── CLOUD_MODE 이외에서만 소급 계산 지원 ──────────────────────
    _cloud = getattr(cfg, "CLOUD_MODE", False)
    if not _cloud:
        try:
            from src.pipeline.sanity_checker import check_worker_sanity
        except ImportError:
            check_worker_sanity = None  # type: ignore
    else:
        check_worker_sanity = None  # type: ignore

    st.markdown(section_header("비상식 패턴 탐지 (Sanity Check)"), unsafe_allow_html=True)
    st.caption(
        "하루 종일 휴식/이동 없이 단일 공간에서 고활성 작업하는 등 "
        "현실적으로 불가능한 패턴을 규칙 기반으로 탐지합니다. "
        "데이터 오류, T-Ward 이상, 헬멧 방치 등의 원인을 나타낼 수 있습니다."
    )

    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    date_str = get_date_selector(
        list(reversed(processed)),
        key="sanity_date",
        default_index=0,
        label="분석 날짜",
        show_label=True,
    ) or processed[-1]

    paths = cfg.get_sector_paths(sector_id)
    worker_path = paths["processed_dir"] / date_str / "worker.parquet"

    if not worker_path.exists():
        st.error(f"worker.parquet 없음: {worker_path}")
        return

    with st.spinner("Sanity check 중..."):
        try:
            import pyarrow.parquet as pq
            avail = set(pq.read_schema(str(worker_path)).names)
            # sanity 컬럼이 있으면 그대로 로드, 없으면 소급 계산
            _has_sanity = "sanity_flag_count" in avail

            _load_cols = [
                "user_no", "user_name", "company_name",
                "work_minutes", "rest_minutes", "rest_min",
                "transit_min", "transition_count", "unique_loci",
                "high_active_min", "work_zone_minutes", "ewi",
                "helmet_abandoned",
            ]
            if _has_sanity:
                _load_cols += ["sanity_flags", "sanity_flag_count",
                               "sanity_severity", "is_suspicious"]
            _load_cols = [c for c in _load_cols if c in avail]
            wdf = pd.read_parquet(str(worker_path), columns=_load_cols)

            if not _has_sanity:
                if check_worker_sanity is not None:
                    st.warning(
                        "이 날짜의 parquet은 Sanity 컬럼이 없습니다 (구 버전). "
                        "즉석 소급 계산 결과를 표시합니다 — 파이프라인 재실행을 권장합니다.",
                        icon="⚠️",
                    )
                    wdf = check_worker_sanity(wdf)
                else:
                    st.info(
                        "ℹ️ 이 날짜의 parquet에 Sanity 컬럼이 없습니다.  \n"
                        "클라우드 모드에서는 소급 계산을 지원하지 않습니다.  \n"
                        "로컬 파이프라인에서 재처리 후 Drive에 업로드하세요."
                    )
                    return
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            return

    if wdf.empty or "sanity_flag_count" not in wdf.columns:
        st.warning("Sanity 데이터를 계산할 수 없습니다.")
        return

    summary = _sanity_summary(wdf)
    rule_descs: dict = {}  # 규칙 ID → 가독성 레이블 변환은 하단에서 직접 처리

    # ── KPI ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        color = COLORS["danger"] if summary["high"] > 0 else COLORS["success"]
        st.markdown(
            metric_card("HIGH 이상", f"{summary['high']}명", color=color),
            unsafe_allow_html=True,
        )
    with c2:
        color = COLORS["warning"] if summary["medium"] > 0 else COLORS["text_muted"]
        st.markdown(
            metric_card("MEDIUM 이상", f"{summary['medium']}명", color=color),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("전체 이상 감지", f"{summary['suspicious']}명",
                        color=COLORS["warning"]),
            unsafe_allow_html=True,
        )
    with c4:
        pct = summary["suspicious_pct"]
        color = (COLORS["danger"] if pct > 5
                 else COLORS["warning"] if pct > 2
                 else COLORS["text_muted"])
        st.markdown(
            metric_card("이상 감지 비율", f"{pct:.1f}%", color=color),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if summary["suspicious"] == 0:
        st.success("감지된 비상식 패턴이 없습니다.")
        return

    # ── 규칙 설명 테이블 ──────────────────────────────────────────────
    with st.expander("규칙 정의 보기"):
        rows = [
            {"규칙 ID": rid, "심각도": meta["severity"], "설명": meta["desc"]}
            for rid, meta in rule_descs.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 규칙별 감지 현황 막대차트 ─────────────────────────────────────
    rule_counts = summary["rule_counts"]
    if rule_counts:
        import plotly.graph_objects as go
        labels, vals, colors = [], [], []
        for rid, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
            sev = rule_descs.get(rid, {}).get("severity", "LOW")
            labels.append(rid.replace("_", " ").title())
            vals.append(cnt)
            colors.append(
                CHART_COLORS["critical"] if sev == "HIGH" else
                CHART_COLORS["medium"]   if sev == "MEDIUM" else
                CHART_COLORS["low"]
            )

        fig = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            text=vals, textposition="outside",
            textfont=dict(color=COLORS["text"], size=12),
        ))
        fig.update_layout(**{
            **PLOTLY_DARK,
            "title": "규칙별 감지 현황",
            "height": max(300, len(labels) * 40 + 90),
            "margin": dict(l=230, r=70, t=50, b=20),
            "xaxis": dict(title="감지 작업자 수 (명)", tickfont_color=COLORS["text_muted"]),
            "yaxis": dict(autorange="reversed", tickfont_color=COLORS["text"]),
        })
        st.plotly_chart(fig, use_container_width=True)

    # ── 심각도별 작업자 분포 ──────────────────────────────────────────
    col_dist, col_top = st.columns([1, 2])
    with col_dist:
        st.markdown(sub_header("심각도별 분포"), unsafe_allow_html=True)
        sev_counts = (
            wdf[wdf["is_suspicious"]]["sanity_severity"]
            .value_counts()
            .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        )
        sev_colors = [CHART_COLORS["critical"], CHART_COLORS["medium"], CHART_COLORS["low"]]
        import plotly.graph_objects as go
        fig2 = go.Figure(go.Pie(
            labels=sev_counts.index.tolist(),
            values=sev_counts.values.tolist(),
            marker_colors=sev_colors,
            hole=0.5,
            textinfo="label+value",
            textfont=dict(color=COLORS["text"]),
        ))
        fig2.update_layout(**{**PLOTLY_DARK,
                              "margin": dict(l=10, r=10, t=20, b=10),
                              "height": 260})
        st.plotly_chart(fig2, use_container_width=True)

    with col_top:
        st.markdown(sub_header("플래그 多 상위 작업자"), unsafe_allow_html=True)
        top_df = (
            wdf[wdf["is_suspicious"]]
            .nlargest(10, "sanity_flag_count")[
                ["user_name", "company_name", "work_minutes",
                 "sanity_severity", "sanity_flag_count", "sanity_flags"]
            ]
            .copy()
        )

        def _fmt(flags_json: str) -> str:
            try:
                return ", ".join(json.loads(flags_json))
            except Exception:
                return str(flags_json or "")

        top_df["sanity_flags"] = top_df["sanity_flags"].apply(_fmt)
        top_df = top_df.rename(columns={
            "user_name":         "작업자",
            "company_name":      "업체",
            "work_minutes":      "근무(분)",
            "sanity_severity":   "심각도",
            "sanity_flag_count": "플래그",
            "sanity_flags":      "감지 규칙",
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True, height=320)

    # ── 전체 상세 테이블 (다운로드 포함) ─────────────────────────────
    st.markdown(sub_header(f"전체 이상 작업자 ({summary['suspicious']}명)"),
                unsafe_allow_html=True)
    full_df = wdf[wdf["is_suspicious"]].copy()
    full_df["sanity_flags"] = full_df["sanity_flags"].apply(
        lambda x: ", ".join(json.loads(x)) if x else ""
    )
    full_df = full_df.sort_values(
        ["sanity_severity", "sanity_flag_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    _disp_cols = ["user_name", "company_name", "work_minutes",
                  "sanity_severity", "sanity_flag_count", "sanity_flags"]
    _disp_cols = [c for c in _disp_cols if c in full_df.columns]
    st.dataframe(
        full_df[_disp_cols].rename(columns={
            "user_name":         "작업자",
            "company_name":      "업체",
            "work_minutes":      "근무시간(분)",
            "sanity_severity":   "심각도",
            "sanity_flag_count": "플래그 수",
            "sanity_flags":      "감지 규칙",
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    csv_bytes = full_df[_disp_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드",
        data=csv_bytes,
        file_name=f"sanity_check_{date_str}.csv",
        mime="text/csv",
    )
