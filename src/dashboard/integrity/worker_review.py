"""
integrity/worker_review.py — 작업자 상세 검토 서브탭
====================================================
정합성 탭의 첫 번째(그리고 가장 큰) 서브탭.
날짜·업체·작업자를 선택하고 7개 inner 탭으로 Raw → 보정 Journey
변환 과정을 교차 검증한다.

Inner 탭:
  1. 📋 출입기록 (타각기)
  2. 📡 Raw BLE
  3. 📊 신호 품질
  4. 🔍 Raw vs 보정 비교
  5. ⏱ 보정 Journey
  6. 📍 Locus 이동 맵
  7. 🚶 물리적 이동 검증 (physical_validator.py 위임)
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src.dashboard.styles import (
    COLORS,
    CHART_COLORS,
    PLOTLY_DARK,
    PLOTLY_LEGEND,
    metric_card,
    metric_card_sm,
    section_header,
    sub_header,
)
from src.dashboard.date_utils import get_date_selector, format_date_label
from src.dashboard.integrity.helpers import (
    CAT_COLORS,
    ACTIVITY_COLORS,
    GAP_COLORS,
    LOCUS_TYPE_COLOR,
    LOCUS_TYPE_KO,
    _SPACE_COLORS,
    _LTYPE_TO_SPACE,
    _SPACE_DISPLAY_ORDER,
    _ACT_TIER_COLORS,
    _ACT_TIER_KO,
    ACTIVE_HIGH_THRESHOLD,
    ACTIVE_LOW_THRESHOLD,
    HELMET_SUSPECT_MIN_RUN,
    _make_donut,
    _locus_display_name,
    _sort_place_names,
    _enrich_journey_with_locus,
    _compute_act_tier,
)
from src.dashboard.integrity.context import (
    _load_journey,
    _load_worker,
    _load_access_log_for_date,
    _load_tward_for_date,
    _load_tward_full_size,
    _load_locus_meta,
    _build_place_ltype_map,
    _compute_helmet_suspect_by_user,
)
from src.dashboard.integrity.physical_validator import _render_physical_validation

logger = logging.getLogger(__name__)

def _render_worker_detail(sector_id: str) -> None:
    """작업자 상세 — 출입기록 · Raw BLE · 보정 Journey 비교."""
    from src.pipeline.cache_manager import detect_processed_dates

    processed = detect_processed_dates(sector_id)
    if not processed:
        st.info("처리된 데이터가 없습니다.")
        return

    paths = cfg.get_sector_paths(sector_id)

    # ── 1) 날짜 + 업체 필터 ──────────────────────────────────────────
    col_date, col_company = st.columns([1, 2])
    with col_date:
        date_str = get_date_selector(
            list(reversed(processed)),
            key="integrity_date",
            default_index=0,
            label="검토 날짜",
            show_label=True,
        ) or processed[-1]

    journey_path = paths["processed_dir"] / date_str / "journey.parquet"
    slim_path    = paths["processed_dir"] / date_str / "journey_slim.parquet"
    agg_path     = paths["processed_dir"] / date_str / "journey_agg.parquet"
    worker_path  = paths["processed_dir"] / date_str / "worker.parquet"

    if not worker_path.exists():
        st.error("해당 날짜의 worker.parquet가 없습니다.")
        return

    import config as _cfg

    # ── journey_slim 온디맨드 다운로드 (정합성 탭 KPI 활성화) ──────────
    # journey.parquet도 없고 slim도 없을 때, Cloud 환경이면 Drive에서 받아옴
    if not journey_path.exists() and not slim_path.exists():
        if _cfg.CLOUD_MODE:
            with st.spinner(
                f"☁️ 정합성 데이터 다운로드 중… ({date_str} · 약 10~15초)"
            ):
                try:
                    from src.pipeline.drive_storage import init_drive_storage_from_secrets
                    _drive = init_drive_storage_from_secrets(sector_id)
                    if _drive:
                        _drive.ensure_journey_slim(
                            sector_id, date_str, _cfg.PROCESSED_DIR
                        )
                except Exception as _e:
                    logger.warning(f"정합성 journey_slim 다운로드 실패: {_e}")

    # ── Cloud: full journey 온디맨드 다운로드 UI ──────────────────────
    # journey.parquet 없을 때(= slim만 있거나 아무것도 없을 때) 다운로드 배너 표시.
    # 사용자가 버튼을 눌러야 다운로드 시작 (대용량 파일이므로 자동 X).
    if not journey_path.exists() and _cfg.CLOUD_MODE:
        _dl_key = f"_journey_dl_{sector_id}_{date_str}"

        # 진행 중이 아닐 때만 배너 표시
        if not st.session_state.get(f"{_dl_key}_done"):
            try:
                from src.pipeline.drive_storage import init_drive_storage_from_secrets
                _drive_info = init_drive_storage_from_secrets(sector_id)
                _file_size_bytes = (
                    _drive_info.get_drive_file_size(sector_id, date_str, "journey.parquet")
                    if _drive_info else None
                )
            except Exception:
                _drive_info = None
                _file_size_bytes = None

            _size_label = (
                f"{round(_file_size_bytes / 1024 / 1024, 0):.0f} MB"
                if _file_size_bytes else "알 수 없음"
            )
            _est_sec = max(30, round((_file_size_bytes or 30_000_000) / 1_000_000 * 1.2))
            _est_label = f"약 {_est_sec}초" if _est_sec < 120 else f"약 {_est_sec // 60}분"

            with st.container():
                st.info(
                    f"☁️ **전체 데이터({_size_label})를 다운로드하면 "
                    f"Raw BLE · 신호 품질 · 보정 비교 탭을 모두 사용할 수 있습니다.** "
                    f"예상 소요 시간: **{_est_label}** · 다운로드 후 자동 새로 고침됩니다.",
                    icon="📥",
                )

                _btn_col, _skip_col = st.columns([2, 5])
                with _btn_col:
                    _do_download = st.button(
                        f"📥 전체 데이터 다운로드 ({_size_label})",
                        key=f"{_dl_key}_btn",
                        type="primary",
                        use_container_width=True,
                    )

                if _do_download and _drive_info:
                    _status_box = st.status(
                        f"☁️ journey.parquet 다운로드 중… ({date_str})",
                        expanded=True,
                    )
                    with _status_box:
                        _prog_bar  = st.progress(0.0)
                        _prog_text = st.empty()

                        def _on_progress(pct: float, label: str) -> None:
                            _prog_bar.progress(min(pct, 1.0))
                            _prog_text.markdown(
                                f"<span style='color:#9AB5D4;font-size:0.85rem;'>"
                                f"다운로드 중: **{label}**</span>",
                                unsafe_allow_html=True,
                            )

                        try:
                            _ok, _reason = _drive_info.ensure_journey_full(
                                sector_id, date_str, _cfg.PROCESSED_DIR,
                                progress_callback=_on_progress,
                            )
                            if _ok:
                                _prog_bar.progress(1.0)
                                _prog_text.empty()
                                _status_box.update(
                                    label="✅ 다운로드 완료! 데이터를 새로 로드합니다…",
                                    state="complete",
                                    expanded=False,
                                )
                                st.session_state[f"{_dl_key}_done"] = True
                                st.rerun()
                            else:
                                _status_box.update(
                                    label=f"❌ 다운로드 실패: {_reason}",
                                    state="error",
                                    expanded=True,
                                )
                        except Exception as _e:
                            _status_box.update(
                                label=f"❌ 오류: {_e}",
                                state="error",
                                expanded=True,
                            )

    worker_df = _load_worker(sector_id, date_str, str(worker_path))
    if worker_df.empty:
        st.warning("작업자 데이터가 없습니다.")
        return

    # 업체 목록
    companies = ["전체"] + sorted(
        worker_df["company_name"].dropna().unique().tolist()
    )
    with col_company:
        sel_company = st.selectbox(
            "업체 필터",
            options=companies,
            index=0,
            key="integrity_company_filter",
        )

    # ── 2) 이름/user_no 검색 + 정렬 ───────────────────────────────────
    col_search, col_sort = st.columns([1, 1])
    with col_search:
        search_q = st.text_input(
            "이름 / User_no 검색",
            placeholder="예: 홍길동  또는  12345",
            key="integrity_search_q",
        ).strip()
    with col_sort:
        sort_mode = st.selectbox(
            "정렬 기준",
            options=[
                "체류 시간 많은 순",
                "헬멧 방치 의심 시간 많은 순",
                "이름 가나다순",
                "User_no 순",
            ],
            index=0,
            key="integrity_worker_sort",
            help=(
                "헬멧 방치 의심: WORK_AREA + 저활성(active_ratio ≤ 0.40)이 "
                "연속 30분 이상 지속된 총 분 수. "
                "journey.parquet를 섹터 전체 1회 스캔하여 계산하고 캐시함."
            ),
        )

    col_sel = st.container()

    # 필터 적용
    filtered = worker_df.copy()
    if sel_company != "전체":
        filtered = filtered[filtered["company_name"] == sel_company]
    if search_q:
        q_lower = search_q.lower()
        mask_name = filtered["user_name"].fillna("").str.lower().str.contains(q_lower)
        mask_no   = filtered["user_no"].astype(str).str.contains(search_q)
        filtered  = filtered[mask_name | mask_no]

    # ── 헬멧 방치 의심 시간: 해당 날짜 journey 1회 스캔하여 전체 작업자 집계 ──
    susp_map: dict[int, int] = {}
    if sort_mode == "헬멧 방치 의심 시간 많은 순":
        locus_csv_p = paths.get("locus_v2_csv") or paths.get("locus_csv")
        if locus_csv_p and Path(locus_csv_p).exists() and journey_path.exists():
            susp_map = _compute_helmet_suspect_by_user(
                str(journey_path), str(locus_csv_p),
                _mtime=journey_path.stat().st_mtime,
            )
        filtered["_susp_min"] = filtered["user_no"].map(susp_map).fillna(0).astype(int)
    else:
        filtered["_susp_min"] = 0

    # 정렬 적용
    if sort_mode == "헬멧 방치 의심 시간 많은 순":
        filtered = filtered.sort_values(
            ["_susp_min", "work_minutes"],
            ascending=[False, False], na_position="last",
        )
    elif sort_mode == "이름 가나다순":
        filtered = filtered.sort_values("user_name", na_position="last")
    elif sort_mode == "User_no 순":
        filtered = filtered.sort_values("user_no", na_position="last")
    else:  # 체류 시간 많은 순 (기본)
        filtered = filtered.sort_values("work_minutes", ascending=False, na_position="last")

    if filtered.empty:
        st.warning("검색 조건에 해당하는 작업자가 없습니다.")
        return

    with col_sel:
        # 라벨: 정렬 기준이 헬멧 방치일 때 "헬멧 NN분" 추가 표시
        def _mk_label(r) -> str:
            base = f"{r['user_name']}  ({r['user_no']})  ·  {str(r['company_name'])[:20]}"
            if pd.notna(r.get("work_minutes")):
                base += f"  ·  체류 {r['work_minutes']:.0f}분"
            if sort_mode == "헬멧 방치 의심 시간 많은 순" and r.get("_susp_min", 0) > 0:
                base += f"  ·  ⚠헬멧 {int(r['_susp_min'])}분"
            return base

        labels = [_mk_label(r) for _, r in filtered.iterrows()]
        sel_idx = st.selectbox(
            f"작업자 선택  ({len(filtered):,}명 / 전체 {len(worker_df):,}명)",
            options=range(len(filtered)),
            format_func=lambda i: labels[i],
            index=0,
            key="integrity_worker_select",
        )

    selected_user_no = int(filtered.iloc[sel_idx]["user_no"])
    user_info = filtered.iloc[sel_idx]

    st.divider()

    # ── 3) 데이터 로드 ──────────────────────────────────────────────
    raw_dir = str(paths["raw_dir"])

    # locus 메타 (탭 여러 곳에서 사용)
    locus_csv = paths.get("locus_v2_csv") or paths.get("locus_csv")
    locus_meta = (
        _load_locus_meta(str(locus_csv))
        if locus_csv and Path(locus_csv).exists()
        else pd.DataFrame()
    )

    # ── 섹터 전체 spot_name → locus_type 매핑 (도넛용 단일 소스) ─────
    # per-row mode 기반 — 개별 작업자 journey가 아닌 전체 집계로 정확
    place_ltype_map: dict = {}
    if locus_csv and Path(locus_csv).exists() and journey_path.exists():
        place_ltype_map = _build_place_ltype_map(str(journey_path), str(locus_csv))

    with st.spinner("데이터 로딩 중..."):
        # journey.parquet 우선, 없으면 slim fallback (Cloud 환경)
        if journey_path.exists():
            _jp_mtime = journey_path.stat().st_mtime
            jdf = _load_journey(sector_id, date_str, str(journey_path), _mtime=_jp_mtime)
        elif slim_path.exists():
            _slim_mtime = slim_path.stat().st_mtime
            jdf = _load_journey(sector_id, date_str, str(slim_path), _mtime=_slim_mtime)
        else:
            jdf = pd.DataFrame()
        user_jdf = (
            jdf[jdf["user_no"] == selected_user_no]
            .sort_values("timestamp")
            .reset_index(drop=True)
        ) if not jdf.empty else pd.DataFrame()

        # ★ journey_agg: Cloud 환경에서 journey/slim 없을 때 사전 집계 데이터 로드
        # (~3.5MB/일, Drive에서 자동 다운로드됨)
        agg_df = pd.DataFrame()
        if _cfg.CLOUD_MODE and not journey_path.exists() and not slim_path.exists():
            if agg_path.exists():
                try:
                    agg_df = pd.read_parquet(str(agg_path))
                except Exception as _e:
                    logger.warning(f"journey_agg 로드 실패: {_e}")
        user_agg_df = (
            agg_df[agg_df["user_no"] == selected_user_no].copy()
            if not agg_df.empty else pd.DataFrame()
        )

        # locus 메타 조인 (locus_type, building, floor, function, locus_x/y 등)
        if not locus_meta.empty and not user_jdf.empty:
            user_jdf = _enrich_journey_with_locus(user_jdf, locus_meta)

        access_df  = _load_access_log_for_date(raw_dir, date_str)
        tward_df   = _load_tward_for_date(raw_dir, date_str)

        user_access = access_df[access_df["User_no"] == selected_user_no] if not access_df.empty else pd.DataFrame()
        user_tward  = tward_df[tward_df["User_no"] == selected_user_no].sort_values("Time").reset_index(drop=True) if not tward_df.empty else pd.DataFrame()

    # ── 4) 타각기 기준 총 체류 시간 계산 (벡터화) ──────────────────────
    # 기존 iterrows → vectorized timedelta (수십 배 빠름)
    access_total_min: float = 0.0
    if not user_access.empty:
        _et = pd.to_datetime(user_access["Entry_time"], errors="coerce")
        _xt = pd.to_datetime(user_access["Exit_time"],  errors="coerce")
        _valid = _et.notna() & _xt.notna()
        if _valid.any():
            _deltas = ((_xt[_valid] - _et[_valid]).dt.total_seconds() / 60)
            access_total_min = float(_deltas[_deltas > 0].sum())
    # fallback: AccessLog 없으면 worker.parquet work_minutes 사용
    if access_total_min <= 0:
        access_total_min = float(user_info.get("work_minutes") or 0)

    # ── 5) KPI 카드 ─────────────────────────────────────────────────
    _render_worker_kpi(user_jdf, user_info, user_agg_df=user_agg_df)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # slim 여부 판단 — Cloud에서 journey_slim 사용 시 signal_count 등 없음
    _is_slim = not user_jdf.empty and "signal_count" not in user_jdf.columns
    # agg-only 모드 — Cloud에서 journey/slim 없고 journey_agg만 있을 때
    _is_agg_only = user_jdf.empty and not user_agg_df.empty
    _SLIM_NOTE = (
        "☁️ **Cloud 환경 — 슬림 데이터** · 이 탭은 전체 `journey.parquet`가 필요합니다.  \n"
        "신호 품질·보정 상세는 로컬 환경 또는 전체 데이터 다운로드 후 확인 가능합니다."
    )
    _AGG_NOTE = (
        "☁️ **Cloud 환경 — 집계 데이터** · 이 탭은 분 단위 `journey.parquet`가 필요합니다.  \n"
        "전체 `journey.parquet` 다운로드 후 Raw BLE · 신호 품질 · 보정 상세를 확인할 수 있습니다."
    )

    # ── 6) 탭 (근거-쌓기 순서: Raw → 신호품질 → 비교 → 보정 → 맵 → 물리검증) ──
    tab_access, tab_raw, tab_signal, tab_compare, tab_journey, tab_map, tab_phys = st.tabs([
        "📋 출입기록 (타각기)",
        "📡 Raw BLE",
        "📊 신호 품질",
        "🔍 Raw vs 보정 비교",
        "⏱ 보정 Journey",
        "📍 Locus 이동 맵",
        "🚶 물리적 이동 검증",
    ])

    with tab_access:
        _render_access_record(user_access, user_info)

    with tab_raw:
        if _is_agg_only:
            st.info(_AGG_NOTE)
        elif _is_slim:
            st.info(_SLIM_NOTE)
        else:
            _render_raw_ble(user_tward, date_str, access_total_min, locus_meta, user_jdf,
                            place_ltype_map=place_ltype_map,
                            raw_dir=raw_dir, user_no=selected_user_no)

    with tab_signal:
        if _is_agg_only:
            st.info(_AGG_NOTE)
        elif user_jdf.empty:
            st.info("신호 품질 데이터가 없습니다.")
        elif _is_slim:
            st.info(_SLIM_NOTE)
        else:
            _render_signal_quality(user_jdf)

    with tab_compare:
        if _is_agg_only:
            st.info(_AGG_NOTE)
        elif _is_slim:
            st.info(_SLIM_NOTE)
        else:
            _render_journey_comparison(user_tward, user_jdf, access_total_min,
                                       place_ltype_map=place_ltype_map,
                                       locus_meta=locus_meta,
                                       user_access=user_access)

    with tab_journey:
        if _is_agg_only:
            _render_locus_agg_table(user_agg_df, locus_meta)
        elif user_jdf.empty:
            st.info("보정된 journey 데이터가 없습니다.")
        elif _is_slim:
            st.info(_SLIM_NOTE)
        else:
            _render_corrected_journey(user_jdf, access_total_min=access_total_min,
                                      user_info=user_info)

    with tab_map:
        if _is_agg_only:
            _render_locus_map_agg(user_agg_df, locus_meta)
        else:
            _render_locus_map(user_jdf, locus_meta)

    with tab_phys:
        if _is_agg_only:
            st.info(_AGG_NOTE)
        else:
            _render_physical_validation(sector_id, user_jdf)

def _render_worker_kpi(
    user_jdf: pd.DataFrame,
    user_info,
    user_agg_df: pd.DataFrame | None = None,
) -> None:
    """선택된 작업자의 KPI 요약.

    journey.parquet 없는 Cloud 환경에서도 안전하게 동작하도록
    컬럼 존재 여부를 확인하고 없으면 0으로 처리.
    user_agg_df 가 있으면 agg 집계에서 파생 (Cloud agg-only 모드).
    """
    if user_agg_df is None:
        user_agg_df = pd.DataFrame()

    def _col_sum(col: str, default=0) -> int:
        return int(user_jdf[col].sum()) if col in user_jdf.columns else default

    def _col_inv_sum(col: str, default=0) -> int:
        return int((~user_jdf[col]).sum()) if col in user_jdf.columns else default

    # agg-only 모드: user_jdf 없고 agg 있을 때 파생값 사용
    if user_jdf.empty and not user_agg_df.empty:
        n_total    = int(user_agg_df["total_min"].sum())
        n_gap      = int(user_agg_df["gap_filled_min"].sum())
        n_low_conf = int(user_agg_df["low_confidence_min"].sum())
        n_invalid  = 0   # agg에 없음
        n_zero     = 0   # agg에 없음
        n_transit  = 0   # agg에 없음
    else:
        n_total    = len(user_jdf)
        n_gap      = _col_sum("is_gap_filled")
        n_low_conf = _col_sum("is_low_confidence")
        n_invalid  = _col_inv_sum("is_valid_transition")
        n_zero     = int((user_jdf["signal_count"] == 0).sum()) if "signal_count" in user_jdf.columns else 0
        n_transit  = _col_sum("is_transition")

    gap_pct   = n_gap / n_total * 100 if n_total > 0 else 0
    lowc_pct  = n_low_conf / n_total * 100 if n_total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(metric_card("총 기록", f"{n_total}분"), unsafe_allow_html=True)
    with col2:
        gap_color = (COLORS["danger"] if gap_pct >= 50
                     else COLORS["warning"] if gap_pct >= 20
                     else COLORS["success"])
        st.markdown(metric_card("음영 보정", f"{gap_pct:.1f}%",
                                color=gap_color),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("저신뢰 구간", f"{lowc_pct:.1f}%"),
                    unsafe_allow_html=True)
    with col4:
        inv_color = COLORS["danger"] if n_invalid > 0 else COLORS["success"]
        st.markdown(metric_card("비유효 전이", f"{n_invalid}건",
                                color=inv_color),
                    unsafe_allow_html=True)
    with col5:
        st.markdown(metric_card("총 전이", f"{n_transit}건"),
                    unsafe_allow_html=True)

    # 메타 정보 한 줄
    if user_info is not None:
        shift_txt = user_info.get("shift_type", "unknown")
        work_min  = user_info.get("work_minutes", 0) or 0
        cov_lbl   = user_info.get("ble_coverage", "?")
        cov_pct   = user_info.get("ble_coverage_pct", 0) or 0
        helmet_ab = bool(user_info.get("helmet_abandoned", False))
        ewi_rel   = bool(user_info.get("ewi_reliable", False))

        helmet_badge = (
            "<span style='background:#2A1A1A; color:#FF4C4C; "
            "padding:2px 8px; border-radius:4px; margin-right:6px; font-size:0.75rem;'>"
            "⚠ 헬멧 방치 의심</span>"
        ) if helmet_ab else ""
        ewi_badge = (
            "<span style='background:#1A2A1A; color:#00C897; "
            "padding:2px 8px; border-radius:4px; margin-right:6px; font-size:0.75rem;'>"
            "✓ EWI 유효</span>"
            if ewi_rel else
            "<span style='background:#2A241A; color:#FFB300; "
            "padding:2px 8px; border-radius:4px; margin-right:6px; font-size:0.75rem;'>"
            "ⓘ EWI 저신뢰</span>"
        )

        st.markdown(
            f"""
            <div style='margin-top:10px; padding:10px 14px;
                        background:#0D1B2A; border-left:3px solid #00AEEF; border-radius:4px;
                        font-size:0.82rem; color:#9AB5D4;'>
                <b style='color:#D5E5FF;'>교대</b> {shift_txt} ·
                <b style='color:#D5E5FF;'>체류</b> {work_min:.0f}분 ·
                <b style='color:#D5E5FF;'>BLE 커버리지</b> {cov_lbl} ({cov_pct:.1f}%) &nbsp;
                {helmet_badge}{ewi_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

def _render_access_record(user_access: pd.DataFrame, user_info) -> None:
    """타각기 출입기록 — AccessLog CSV에서 로드한 원본.

    모든 기록을 보여준다: 당일 입장, 당일 퇴장(야간교대), 복수 출입 모두 포함.
    날짜가 걸치는 경우(야간 교대)는 날짜 태그로 명시.

    구현 주의:
      st.markdown HTML 블록은 CommonMark 규칙상 빈 줄이 포함되면 블록이 종료되어
      이후 태그가 텍스트로 렌더링된다.
      → 모든 카드 HTML을 빈 줄 없이 단일 문자열로 조립 후 한 번에 st.markdown 호출.
    """
    st.markdown(section_header("출입기록 (타각기 원본)"), unsafe_allow_html=True)

    if user_access.empty:
        if cfg.CLOUD_MODE:
            st.info(
                "☁️ Cloud 환경에서는 타각기 원본 데이터(AccessLog CSV)가 제공되지 않습니다.  \n"
                "출입 시간은 worker.parquet의 **체류 시간(work_minutes)** 으로 대체 집계됩니다."
            )
        else:
            st.info("해당 날짜의 AccessLog 파일에 출입 기록이 없습니다. (날짜 범위 파일이 없거나 미출입)")
        return

    n_rec = len(user_access)
    st.markdown(
        f"<p style='color:#7A8FA6; font-size:0.83rem; margin-bottom:12px;'>"
        f"총 <b style='color:#D5E5FF;'>{n_rec}건</b>의 출입 기록 — "
        f"야간 교대(전날 입장·당일 퇴장) 및 복수 출입 모두 포함</p>",
        unsafe_allow_html=True,
    )

    def _fmt_dt(ts) -> tuple[str, str]:
        """(날짜 'MM/DD', 시간 'HH:MM:SS') 반환. NaT → ('', '미기록')"""
        if pd.isna(ts):
            return "", "미기록"
        t = pd.Timestamp(ts)
        return t.strftime("%m/%d"), t.strftime("%H:%M:%S")

    # ── enumerate로 순번 부여 (iterrows 인덱스 ≠ 순번) ─────────────────
    for seq_num, (_, row) in enumerate(user_access.iterrows(), start=1):
        entry_t  = row.get("Entry_time")
        exit_t   = row.get("Exit_time")
        tward_id = row.get("T-Ward ID", "")
        company  = str(row.get("SCon_company_name", "") or row.get("HyCon_company_name", "") or "")
        emp_stat = str(row.get("EmploymentStatus_Hycon", "") or "")
        rec_id   = str(row.get("User_record_id", "") or "")

        entry_date, entry_time_s = _fmt_dt(entry_t)
        exit_date,  exit_time_s  = _fmt_dt(exit_t)

        # 체류 시간
        if pd.notna(entry_t) and pd.notna(exit_t):
            delta_min = (pd.Timestamp(exit_t) - pd.Timestamp(entry_t)).total_seconds() / 60
            dur_str = f"{delta_min:.0f}분 ({delta_min/60:.1f}h)" if delta_min >= 0 else "시간 역전 ⚠"
        else:
            dur_str = "—"

        # 야간 교대 / 현장 체류 중 배지
        is_night_shift = (
            pd.notna(entry_t) and pd.notna(exit_t) and
            pd.Timestamp(entry_t).date() != pd.Timestamp(exit_t).date()
        )
        still_on_site = pd.notna(entry_t) and pd.isna(exit_t)

        # 배지 HTML — 빈 줄 없이 한 줄로 조립
        badges = ""
        if is_night_shift:
            badges += ("<span style='background:#1A1A3A;color:#A78BFA;"
                       "padding:2px 8px;border-radius:4px;font-size:0.72rem;margin-left:8px;'>"
                       "🌙 야간 교대</span>")
        if still_on_site:
            badges += ("<span style='background:#1A2A1A;color:#00C897;"
                       "padding:2px 8px;border-radius:4px;font-size:0.72rem;margin-left:8px;'>"
                       "📍 현장 체류 중</span>")

        # T-Ward
        tward_ok    = pd.notna(tward_id) and str(tward_id) not in ["nan", ""]
        tward_color = "#00C897" if tward_ok else "#FF4C4C"
        tward_label = str(tward_id) if tward_ok else "미등록"

        # 날짜 prefix (날짜가 있을 때만)
        entry_date_html = (f"<span style='color:#A78BFA;font-size:0.72rem;margin-right:4px;'>{entry_date}</span>"
                           if entry_date else "")
        exit_date_html  = (f"<span style='color:#A78BFA;font-size:0.72rem;margin-right:4px;'>{exit_date}</span>"
                           if exit_date else "")

        border_color = "#3A2A6A" if is_night_shift else "#1E3A5A"

        # ── 카드 HTML: 빈 줄이 생기지 않도록 join으로 조립 ──────────────
        # CommonMark: <div> 블록은 빈 줄을 만나면 종료 → 이후 태그가 텍스트로 출력됨
        # 대책: 줄 목록을 join("\n")으로 이어붙이되 빈 줄을 절대 포함하지 않는다.
        lines = [
            f"<div style='background:#0D1B2A;border:1px solid {border_color};"
            f"border-radius:8px;padding:14px 20px;margin-bottom:10px;'>",
            f"<div style='display:flex;align-items:center;margin-bottom:10px;'>"
            f"<span style='color:#7A8FA6;font-size:0.75rem;'>기록 {seq_num}</span>"
            f"{badges}</div>",
            "<div style='display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;'>",
            # 입장
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>입장 시간</div>"
            f"<div style='color:#D5E5FF;font-size:1.05rem;font-weight:600;'>{entry_date_html}{entry_time_s}</div></div>",
            # 화살표
            "<div style='color:#3A5A7A;font-size:1.4rem;padding-top:10px;'>→</div>",
            # 퇴장
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>퇴장 시간</div>"
            f"<div style='color:#D5E5FF;font-size:1.05rem;font-weight:600;'>{exit_date_html}{exit_time_s}</div></div>",
            # 체류
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>체류 시간</div>"
            f"<div style='color:#FFB300;font-size:1.05rem;font-weight:600;'>{dur_str}</div></div>",
            # T-Ward
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>T-Ward ID</div>"
            f"<div style='color:{tward_color};font-size:0.9rem;font-weight:600;font-family:monospace;'>{tward_label}</div></div>",
            # 업체
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>업체</div>"
            f"<div style='color:#9AB5D4;font-size:0.85rem;'>{company}</div></div>",
            # 고용형태
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>고용형태</div>"
            f"<div style='color:#9AB5D4;font-size:0.85rem;'>{emp_stat}</div></div>",
            # 기록 ID
            "<div><div style='color:#7A8FA6;font-size:0.75rem;margin-bottom:2px;'>기록 ID</div>"
            f"<div style='color:#6A8AA6;font-size:0.75rem;font-family:monospace;'>{rec_id}</div></div>",
            "</div>",   # flex row 닫기
            "</div>",   # 카드 닫기
        ]
        st.markdown("\n".join(lines), unsafe_allow_html=True)

    # 원본 테이블 전체 보기
    with st.expander("📋 출입기록 원본 테이블 (AccessLog 전체 컬럼)"):
        disp_cols = [c for c in [
            "User_no", "Worker_name", "Entry_time", "Exit_time",
            "T-Ward ID", "SCon_company_name", "HyCon_company_name",
            "EmploymentStatus_Hycon", "User_record_id",
        ] if c in user_access.columns]
        st.dataframe(user_access[disp_cols], use_container_width=True)


def _render_raw_ble(
    user_tward: pd.DataFrame,
    date_str: str,
    access_total_min: float = 0.0,
    locus_meta: "pd.DataFrame | None" = None,
    user_jdf: "pd.DataFrame | None" = None,
    place_ltype_map: "dict | None" = None,
    raw_dir: "str | None" = None,
    user_no: "int | None" = None,
) -> None:
    """TWardData — 1분 단위 원본 BLE 신호 타임라인 + 공간 유형별 체류 도넛.

    access_total_min: 타각기(AccessLog) 기준 총 체류시간 — 도넛의 분모.
    place_ltype_map : 섹터 전체 journey 기반 spot_name→locus_type (단일 소스).
    locus_meta     : locus_v2 메타 — Place → locus_type 매핑에 사용.
    user_jdf       : journey.parquet — spot_name+locus_id → locus_type 매핑에 사용 (우선).
    raw_dir, user_no: 원본 파일 경로/작업자 번호 — Raw 여부 검증 표시용.
    TWardData에 기록 없는 구간은 '음영(미수신)'으로 분류.
    """
    st.markdown(section_header("Raw BLE (TWardData 원본, 1분 단위)"), unsafe_allow_html=True)

    # ── 데이터 출처 / Raw 여부 검증 패널 ───────────────────────────────
    # "이 탭이 정말 Raw CSV 원본인지" 를 사용자가 즉시 확인할 수 있도록
    # - 원본 파일 경로 (Y1_TWardData_YYYYMMDD.csv 또는 범위 파일)
    # - 파일 전체 행 수 vs 이 작업자 기록 수
    # - 적용된 필터 명시 (user_no + date 단일 필터만 — 가공 없음)
    if raw_dir and user_no is not None:
        try:
            fpath = _find_raw_file(Path(raw_dir), "Y1_TWardData", date_str)
        except Exception:
            fpath = None

        n_user   = int(len(user_tward))
        # 원본 파일 전체 행 수 (캐시됨: _load_tward_full_size)
        n_total_raw = _load_tward_full_size(str(fpath)) if fpath else 0
        fsize_kb = (Path(fpath).stat().st_size / 1024) if fpath else 0.0

        fpath_disp = (
            str(fpath).replace("/Users/tony/Desktop/TJLABS/TJLABS_Research/", "")
            if fpath else "(파일 없음)"
        )

        st.markdown(
            f"""
            <div style='background:#0B1728; border:1px dashed #3A5A7A; border-radius:6px;
                        padding:10px 14px; margin-bottom:10px; font-size:0.78rem;
                        color:#9AB5D4; line-height:1.6;'>
                <b style='color:#00AEEF;'>📁 데이터 출처 (Raw 여부 검증)</b><br>
                <span style='color:#7A8FA6;'>파일:</span>
                <code style='color:#D5E5FF; font-size:0.76rem;'>{fpath_disp}</code>
                ({fsize_kb:,.0f} KB)<br>
                <span style='color:#7A8FA6;'>원본 전체 행 수:</span>
                <b style='color:#D5E5FF;'>{n_total_raw:,}행</b> &nbsp;·&nbsp;
                <span style='color:#7A8FA6;'>본 작업자 표시 행:</span>
                <b style='color:#00C897;'>{n_user}행</b><br>
                <span style='color:#7A8FA6;'>적용 필터:</span>
                <code style='color:#FFB300;'>Time.date == {date_str}</code> &nbsp;AND&nbsp;
                <code style='color:#FFB300;'>User_no == {user_no}</code>
                <span style='color:#6A7A95;'>(가공/보정 없음 — pd.read_csv(cp949) 결과 그대로)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if user_tward.empty:
        st.info("해당 날짜의 TWardData 파일에 기록이 없습니다.")
        return

    n_total  = len(user_tward)
    n_zero   = int((user_tward["Signal_count"] == 0).sum())
    n_active = int((user_tward["ActiveSignal_count"] > 0).sum())
    places   = user_tward["Place"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_sm("총 기록", f"{n_total}분"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_sm("0-신호", f"{n_zero}분",
                                   "#FF4C4C" if n_zero > 0 else "#00C897"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_sm("활성 신호", f"{n_active}분",
                                   "#00C897"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_sm("방문 장소", f"{places}곳"), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 장소 타임라인 (scatter) ──────────────────────────────────────
    st.markdown(sub_header("장소 타임라인 (Place 원본)"), unsafe_allow_html=True)

    df = user_tward.copy()
    df["active_ratio_raw"] = (
        df["ActiveSignal_count"] / df["Signal_count"].replace(0, float("nan"))
    ).fillna(0)

    # 장소별 색상 (Building 기준)
    buildings = df["Building"].fillna("미확인").unique()
    bldg_colors = {b: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                   for i, b in enumerate(buildings)}

    fig = go.Figure()
    for bldg in buildings:
        sub = df[df["Building"].fillna("미확인") == bldg]
        fig.add_trace(go.Scatter(
            x=sub["Time"],
            y=sub["Place"].fillna("미확인"),
            mode="markers",
            name=bldg,
            marker=dict(
                size=8,
                color=bldg_colors[bldg],
                line=dict(width=0.5, color="#0D1B2A"),
            ),
            customdata=list(zip(
                sub["Building"].fillna("미확인"),
                sub["Level"].fillna("?"),
                sub["Signal_count"],
                sub["ActiveSignal_count"],
                sub["active_ratio_raw"].round(2),
            )),
            hovertemplate=(
                "%{x|%H:%M}<br>"
                "장소: <b>%{y}</b><br>"
                "건물: %{customdata[0]}  층: %{customdata[1]}<br>"
                "신호: %{customdata[2]}  활성: %{customdata[3]}<br>"
                "활성비율: %{customdata[4]:.2f}<extra></extra>"
            ),
        ))

    # Y축: 유사 이름 그룹 정렬 (building/floor 기반)
    unique_places = df["Place"].fillna("미확인").unique().tolist()
    place_order = _sort_place_names(unique_places)

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=180, r=20, t=30, b=40)},
        height=max(300, 22 * len(place_order) + 80),
        xaxis=dict(title="시간", tickformat="%H:%M",
                   tickfont_color=COLORS["text_muted"], gridcolor="#2A3A4A"),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=place_order,
            tickfont=dict(size=9, color=COLORS["text"]),
        ),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True, key="plotly_1")

    # ── 신호 시계열 ───────────────────────────────────────────────────
    st.markdown(sub_header("BLE 신호 강도 시계열 (원본)"), unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Time"], y=df["Signal_count"],
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["accent"], width=1.5),
        name="Signal_count",
        hovertemplate="%{x|%H:%M}<br>signals: %{y}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=df["Time"], y=df["ActiveSignal_count"],
        mode="lines",
        line=dict(color=COLORS["success"], width=1.5),
        name="ActiveSignal_count",
        hovertemplate="%{x|%H:%M}<br>active: %{y}<extra></extra>",
    ))
    # 0-signal 강조
    zero_df = df[df["Signal_count"] == 0]
    if len(zero_df) > 0:
        fig2.add_trace(go.Scatter(
            x=zero_df["Time"], y=zero_df["Signal_count"],
            mode="markers",
            marker=dict(color=COLORS["danger"], size=5, symbol="triangle-down"),
            name="0-signal",
            hovertemplate="%{x|%H:%M}<br><b>0-signal</b><extra></extra>",
        ))
    fig2.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=60, r=20, t=20, b=40)},
        height=240,
        xaxis=dict(title="시간", tickformat="%H:%M"),
        yaxis=dict(title="신호 수"),
        hovermode="x unified",
        legend={**PLOTLY_LEGEND},
    )
    st.plotly_chart(fig2, use_container_width=True, key="plotly_2")

    # ── Raw 테이블 ────────────────────────────────────────────────────
    with st.expander("📋 TWardData 원본 테이블"):
        disp = df[["Time", "Building", "Level", "Place",
                   "Signal_count", "ActiveSignal_count", "active_ratio_raw",
                   "X", "Y"]].copy()
        disp["Time"] = disp["Time"].dt.strftime("%H:%M")
        st.dataframe(disp, use_container_width=True, height=400)

        csv = disp.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ Raw BLE CSV", csv,
                           f"raw_ble_{date_str}.csv", "text/csv")

    # ── 공간 유형별 체류 비율 도넛 (타각기 기준 전체 시간) ───────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(sub_header("공간 유형별 체류 비율 (타각기 기준)"), unsafe_allow_html=True)
    st.caption(
        "전체 시간 = 타각기(AccessLog) 출입 기준  ·  "
        "Place → locus_type 매핑으로 공간 유형 분류  ·  "
        "매핑 불가 또는 TWardData 미수신 구간 → 음영"
    )

    # ── Place → locus_type 매핑: 섹터 전체 journey 기반 단일 소스 ────
    # place_ltype_map은 _build_place_ltype_map()으로 미리 계산 (캐시됨)
    # key = spot_name(=Place), value = dominant locus_type (per-row mode)
    _plmap: dict = place_ltype_map if place_ltype_map else {}

    # apply() 벡터화: Signal_count==0 → 음영, 나머지는 Place→ltype→space 매핑
    df_raw_cat = df.copy()
    _space_ser = (
        df_raw_cat["Place"].astype(str).str.strip()
        .map(_plmap).map(_LTYPE_TO_SPACE).fillna("미분류")
    )
    df_raw_cat["space_cat"] = _space_ser.where(df_raw_cat["Signal_count"] != 0, "음영 (0-signal)")

    raw_counts_tward = df_raw_cat["space_cat"].value_counts()
    tward_total = int(raw_counts_tward.sum())

    # 타각기 기준 전체 시간
    donut_total = int(round(access_total_min)) if access_total_min > 0 else tward_total
    missing_min = max(0, donut_total - tward_total)

    # 최종 집계 (정해진 순서) — Raw에는 gap-fill 없으므로 건너뜀
    raw_counts: dict[str, int] = OrderedDict()
    for cat in _SPACE_DISPLAY_ORDER:
        if cat == "음영 (gap-fill)":
            continue  # Raw BLE에는 gap-fill 없음
        if cat == "음영 (미수신)":
            if missing_min > 0:
                raw_counts[cat] = missing_min
        else:
            v = int(raw_counts_tward.get(cat, 0))
            if v > 0:
                raw_counts[cat] = v

    labels_r = list(raw_counts.keys())
    values_r = list(raw_counts.values())
    colors_r = [_SPACE_COLORS.get(l, "#3A5A7A") for l in labels_r]

    col_donut, col_table = st.columns([1, 1])
    with col_donut:
        st.plotly_chart(
            _make_donut(labels_r, values_r, colors_r, donut_total),
            use_container_width=True, key="plotly_3",
        )

    with col_table:
        rows_raw = [
            {"카테고리": cat,
             "분 수": cnt,
             "비율 (%)": round(cnt / donut_total * 100, 1) if donut_total > 0 else 0}
            for cat, cnt in raw_counts.items()
        ]
        rows_raw.append({"카테고리": "── 합계 ──", "분 수": donut_total, "비율 (%)": 100.0})
        st.dataframe(pd.DataFrame(rows_raw), use_container_width=True, hide_index=True)
        st.markdown(
            f"<p style='color:#6A8AA6;font-size:0.72rem;margin-top:4px;'>"
            f"타각기 {donut_total}분 중 수신 {tward_total}분 / 미수신 {missing_min}분</p>",
            unsafe_allow_html=True,
        )


def _render_journey_comparison(
    user_tward: pd.DataFrame,
    user_jdf: pd.DataFrame,
    access_total_min: float = 0.0,
    place_ltype_map: "dict | None" = None,
    locus_meta: "pd.DataFrame | None" = None,
    user_access: "pd.DataFrame | None" = None,
) -> None:
    """Raw BLE Place vs 보정된 Journey 비교.

    access_total_min: 타각기 기준 총 체류시간 — 도넛 분모로 사용.
    보정 Journey의 Y축을 locus_name 또는 spot_name(place) 중 선택 가능.
    - locus 모드  : pipeline이 부여한 locus_id/name 기반 — 보정 후 공간 추상화 확인
    - place 모드  : journey.parquet의 spot_name(원본 Place와 동일 계통) — Raw와 직접 비교
    """
    st.markdown(section_header("Raw vs 보정 Journey 비교"), unsafe_allow_html=True)

    if user_tward.empty and user_jdf.empty:
        st.info("비교할 데이터가 없습니다.")
        return

    # ── 보정 Journey Y축 선택 ────────────────────────────────────────
    cor_mode = st.radio(
        "보정 Journey Y축",
        options=["place (spot_name)", "locus (locus_name)"],
        index=0,
        horizontal=True,
        key="compare_cor_mode",
        help=(
            "place: Raw와 동일 계통(spot_name) — 보정 전후 직접 비교에 적합\n"
            "locus: pipeline이 할당한 Gateway ID/Name — 공간 추상화 수준 확인"
        ),
    )
    use_locus = (cor_mode == "locus (locus_name)")

    # caption 업데이트
    mode_label = "locus_name" if use_locus else "spot_name (place 계통)"
    st.caption(
        f"왼쪽(파랑): TWardData의 원본 장소(Place) · "
        f"오른쪽(초록/주황): 보정된 Journey의 {mode_label} "
        f"(주황 = Gap-fill 보정 구간)"
    )

    # ── locus_id → 사람이 읽을 수 있는 이름 테이블 구축 (gap-fill 표시용) ──
    _lid_display: dict[str, str] = {}
    if locus_meta is not None and not locus_meta.empty:
        _lm = locus_meta
        _cols = [c for c in ["locus_id", "locus_meta_name", "building", "floor"] if c in _lm.columns]
        for row in _lm[_cols].itertuples(index=False):
            lid = str(getattr(row, "locus_id", "") or "")
            if lid:
                _lid_display[lid] = _locus_display_name(
                    lid,
                    str(getattr(row, "locus_meta_name", "") or ""),
                    str(getattr(row, "building", "") or ""),
                    str(getattr(row, "floor", "") or ""),
                )

    # ── 데이터 사전 준비 (두 차트 공통 Y축 계산을 위해 col 진입 전에) ──
    df_raw = user_tward.copy() if not user_tward.empty else pd.DataFrame()
    place_col_raw = df_raw["Place"].fillna("미확인") if not df_raw.empty else pd.Series(dtype=str)

    df_cor = pd.DataFrame()
    _is_gap = pd.Series(dtype=bool)
    if not user_jdf.empty:
        df_cor = user_jdf.copy()
        df_cor["timestamp"] = pd.to_datetime(df_cor["timestamp"])
        _is_gap = df_cor["is_gap_filled"].fillna(False).astype(bool)
        _lid_ser = df_cor["locus_id"].fillna("").astype(str) if "locus_id" in df_cor.columns else pd.Series("", index=df_cor.index)
        if use_locus:
            _lnm_ser = df_cor["locus_name"].fillna("").astype(str) if "locus_name" in df_cor.columns else pd.Series("", index=df_cor.index)
            df_cor["_ylabel"] = (
                _lid_ser.map(_lid_display)
                .fillna(_lnm_ser.where(_lnm_ser != "", _lid_ser))
                .fillna("UNKNOWN")
            )
        else:
            _sn_ser = (df_cor["spot_name"].fillna("").astype(str).str.strip().replace("", "미확인")
                       if "spot_name" in df_cor.columns
                       else pd.Series("미확인", index=df_cor.index))
            _gap_label = _lid_ser.map(_lid_display).fillna(_lid_ser.replace("", "추정구간"))
            df_cor["_ylabel"] = _gap_label.where(_is_gap, _sn_ser)

    # ── 공통 Y축: Raw + 보정 합집합 → 같은 행에서 좌우 비교 ───────────
    _raw_places: set = set(place_col_raw.unique()) if not place_col_raw.empty else set()
    _cor_places: set = set(df_cor["_ylabel"].unique()) if not df_cor.empty and "_ylabel" in df_cor.columns else set()
    _all_places = _raw_places | _cor_places
    shared_y_order = _sort_place_names(list(_all_places))
    chart_height = max(320, 20 * len(shared_y_order) + 80)

    # ── locus_id → locus_type (gap-fill 배경 밴드 + 도넛 공용) ────────
    _lid_to_ltype: dict = {}
    if locus_meta is not None and not locus_meta.empty and "locus_type" in locus_meta.columns:
        _lid_to_ltype = locus_meta.set_index("locus_id")["locus_type"].to_dict()

    # ── 공간 유형별 배경 밴드 색 ─────────────────────────────────────
    # 각 place의 locus_type을 _plmap에서 조회 → 해당 행에 옅은 색 배경
    _BAND_ALPHA = 0.07
    _BAND_COLORS: dict[str, str] = {
        "WORK_AREA":  f"rgba(0,200,151,{_BAND_ALPHA})",
        "TRANSIT":    f"rgba(0,174,239,{_BAND_ALPHA})",
        "REST_AREA":  f"rgba(255,179,0,{_BAND_ALPHA})",
        "GATE":       f"rgba(167,139,250,{_BAND_ALPHA})",
    }

    def _add_space_bands(fig: "go.Figure", y_order: list) -> "go.Figure":
        """공간 유형별 배경 밴드를 Y축 각 행에 추가."""
        _plm = place_ltype_map or {}
        for i, place in enumerate(y_order):
            ltype = _plm.get(str(place).strip(), "")
            # gap-fill 파생 이름은 _lid_display 역방향 조회
            if not ltype:
                for _lid, _dname in _lid_display.items():
                    if _dname == place:
                        ltype = _lid_to_ltype.get(_lid, "")
                        break
            band_color = _BAND_COLORS.get(ltype)
            if band_color:
                fig.add_hrect(
                    y0=i - 0.5, y1=i + 0.5,
                    fillcolor=band_color,
                    line_width=0,
                    layer="below",
                )
        return fig

    col_l, col_r = st.columns(2)

    # ── 왼쪽: Raw BLE ────────────────────────────────────────────────
    with col_l:
        st.markdown(sub_header("📡 Raw BLE (Place 원본)"), unsafe_allow_html=True)
        if df_raw.empty:
            st.info("TWardData 없음")
        else:
            fig_raw = go.Figure(go.Scatter(
                x=df_raw["Time"],
                y=place_col_raw,
                mode="markers",
                marker=dict(size=6, color=COLORS["accent"],
                            line=dict(width=0.3, color="#0D1B2A")),
                hovertemplate="%{x|%H:%M}<br><b>%{y}</b><extra></extra>",
            ))
            fig_raw.update_layout(
                **{**PLOTLY_DARK, "margin": dict(l=180, r=10, t=20, b=40)},
                height=chart_height,
                xaxis=dict(tickformat="%H:%M", gridcolor="#2A3A4A"),
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=shared_y_order,  # 합집합 순서
                    tickfont=dict(size=8),
                ),
                showlegend=False,
            )
            _add_space_bands(fig_raw, shared_y_order)
            st.plotly_chart(fig_raw, use_container_width=True, key="plotly_4")

    # ── 오른쪽: 보정 Journey ─────────────────────────────────────────
    with col_r:
        header_label = "locus_name" if use_locus else "spot_name (place)"
        st.markdown(sub_header(f"🗺 보정 Journey ({header_label})"), unsafe_allow_html=True)
        if df_cor.empty:
            st.info("journey.parquet 없음")
        else:
            colors_cor = _is_gap.map(
                {True: COLORS["warning"], False: COLORS["accent"]}
            ).fillna(COLORS["accent"])

            lid_col = df_cor["locus_id"].fillna("") if "locus_id" in df_cor.columns else pd.Series("", index=df_cor.index)
            df_cor["_locus_hover"] = lid_col

            fig_cor = go.Figure(go.Scatter(
                x=df_cor["timestamp"],
                y=df_cor["_ylabel"],
                mode="markers",
                marker=dict(size=6, color=colors_cor,
                            line=dict(width=0.3, color="#0D1B2A")),
                hovertemplate=("%{x|%H:%M}<br><b>%{y}</b>"
                               "<br><span style='color:#888'>%{customdata}</span><extra></extra>"),
                customdata=df_cor["_locus_hover"],
            ))
            fig_cor.update_layout(
                **{**PLOTLY_DARK, "margin": dict(l=180, r=10, t=20, b=40)},
                height=chart_height,
                xaxis=dict(tickformat="%H:%M", gridcolor="#2A3A4A"),
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=shared_y_order,  # 합집합 순서
                    tickfont=dict(size=8),
                ),
                showlegend=False,
            )
            _add_space_bands(fig_cor, shared_y_order)
            st.plotly_chart(fig_cor, use_container_width=True, key="plotly_5")

    # ── 보정 상세 통계 ──────────────────────────────────────────────
    if not user_jdf.empty:
        n = len(user_jdf)
        n_gap = int(user_jdf["is_gap_filled"].sum())
        n_raw = n - n_gap

        # place 모드에서 Raw와 일치 여부 간단 비교
        match_note = ""
        if (not use_locus
                and not user_tward.empty
                and "spot_name" in user_jdf.columns
                and "Place" in user_tward.columns):
            raw_places = set(user_tward["Place"].dropna().unique())
            cor_places = set(user_jdf["spot_name"].dropna().unique())
            common = raw_places & cor_places
            only_raw = raw_places - cor_places
            only_cor = cor_places - raw_places
            match_note = (
                f" &nbsp;|&nbsp; Raw-only <b style='color:#FF8C42'>{len(only_raw)}곳</b> · "
                f"보정-only <b style='color:#A78BFA'>{len(only_cor)}곳</b> · "
                f"공통 <b style='color:#00C897'>{len(common)}곳</b>"
            )

        st.markdown(
            f"<div style='background:#131E2A; padding:10px 14px; border-radius:6px;"
            f"font-size:0.82rem; color:#9AB5D4; margin-top:6px;'>"
            f"전체 <b style='color:#D5E5FF'>{n}분</b> 중 "
            f"<b style='color:#00AEEF'>{n_raw}분</b>은 원본 BLE 신호 기반, "
            f"<b style='color:#FFB300'>{n_gap}분</b>은 Gap-fill 보정 추정 "
            f"({n_gap/n*100:.1f}%){match_note}</div>",
            unsafe_allow_html=True,
        )

    # ── 공간 유형별 체류 비율 도넛 비교 ────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(sub_header("공간 유형별 체류 비율 비교"), unsafe_allow_html=True)
    st.caption(
        "왼쪽: Raw BLE — Place → 공간 유형 분류  ·  "
        "오른쪽: 보정 Journey — same_locus gap-fill은 실제 공간 유형, 이동 추정(graph/ratio)만 음영 처리"
    )

    # ── 타각기 기준 총 시간 ────────────────────────────────────────────
    tward_row_cnt = len(user_tward) if not user_tward.empty else 0
    donut_total   = int(round(access_total_min)) if access_total_min > 0 else tward_row_cnt
    missing_min   = max(0, donut_total - tward_row_cnt)

    # ── Place → locus_type 단일 소스 (도넛 집계용) ───────────────────
    _plmap: dict = place_ltype_map if place_ltype_map else {}

    # ── Raw BLE 카테고리 집계 (Place → 공간 유형) ─────────────────────
    df_tw = pd.DataFrame()
    rc_ordered: "dict[str, int]" = {}
    if not user_tward.empty:
        df_tw = user_tward.copy()
        # apply() 벡터화: Signal_count==0 → 음영, 나머지는 Place→ltype→space 매핑
        _cat_ser = (
            df_tw["Place"].astype(str).str.strip()
            .map(_plmap).map(_LTYPE_TO_SPACE).fillna("미분류")
        )
        df_tw["_cat"] = _cat_ser.where(df_tw["Signal_count"] != 0, "음영 (0-signal)")
        rc_base = df_tw["_cat"].value_counts()
        rc_ordered = OrderedDict()
        for cat in _SPACE_DISPLAY_ORDER:
            if cat == "음영 (gap-fill)":
                continue  # Raw BLE에는 gap-fill 없음
            if cat == "음영 (미수신)":
                if missing_min > 0:
                    rc_ordered[cat] = missing_min
            else:
                v = int(rc_base.get(cat, 0))
                if v > 0:
                    rc_ordered[cat] = v

    # ── 보정 Journey 카테고리 집계 (per-row locus_id → locus_type) ────
    # gap-fill은 "음영 (gap-fill)", 나머지는 locus_id로 직접 locus_type 조회
    jdf2 = pd.DataFrame()
    cc_ordered: "dict[str, int]" = {}
    if not user_jdf.empty:
        jdf2 = user_jdf.copy()
        is_gap2 = (
            jdf2["is_gap_filled"].fillna(False).astype(bool)
            if "is_gap_filled" in jdf2.columns
            else pd.Series(False, index=jdf2.index)
        )

        # apply() 벡터화: locus_id → locus_type → space 매핑
        # 1차: locus_id → ltype (per-row dict)
        _ltype_ser = (
            jdf2["locus_id"].fillna("").astype(str).str.strip()
            .map(_lid_to_ltype)
        )
        # fallback: locus_type 컬럼이 존재하면 사용
        if "locus_type" in jdf2.columns:
            _ltype_fallback = jdf2["locus_type"].fillna("").astype(str).str.strip()
            _ltype_ser = _ltype_ser.fillna(_ltype_fallback)
        _space_ser2 = _ltype_ser.map(_LTYPE_TO_SPACE).fillna("미분류")

        # ── gap-fill 분류 (same_locus 기반) ────────────────────────
        # same_locus: 갭 전후 locus 동일 → "그 자리에 계속 있었던 것" → 실제 공간 유형
        # graph_path / ratio_split: 추정 이동 경로 → 불확실 → 음영(gap-fill)
        # gap_method 컬럼 없는 구 parquet: gap_confidence="high"로 fallback
        gap_method_ser = (
            jdf2["gap_method"].fillna("").astype(str)
            if "gap_method" in jdf2.columns
            else pd.Series("", index=jdf2.index)
        )
        gap_conf = (
            jdf2["gap_confidence"].fillna("none").astype(str)
            if "gap_confidence" in jdf2.columns
            else pd.Series("none", index=jdf2.index)
        )
        # same_locus 메서드이거나(신규 parquet), gap_confidence=high(구 parquet) → 확실한 위치
        is_certain_gap = (gap_method_ser == "same_locus") | (gap_conf == "high")
        is_uncertain_gap = is_gap2 & ~is_certain_gap
        jdf2["_cat"] = _space_ser2.where(~is_uncertain_gap, "음영 (gap-fill)")

        cc_base     = jdf2["_cat"].value_counts()
        cor_total   = int(cc_base.sum())
        cor_missing = max(0, donut_total - cor_total)
        cc_ordered = OrderedDict()
        for cat in _SPACE_DISPLAY_ORDER:
            if cat == "음영 (미수신)":
                if cor_missing > 0:
                    cc_ordered[cat] = cor_missing
            else:
                v = int(cc_base.get(cat, 0))
                if v > 0:
                    cc_ordered[cat] = v

    col_d_raw, col_d_cor = st.columns(2)

    with col_d_raw:
        st.markdown(
            "<p style='text-align:center;color:#7A8FA6;font-size:0.82rem;"
            "margin-bottom:4px;'>📡 Raw BLE — 공간 유형 (Place 기준)</p>",
            unsafe_allow_html=True,
        )
        if rc_ordered:
            st.plotly_chart(
                _make_donut(
                    list(rc_ordered.keys()), list(rc_ordered.values()),
                    [_SPACE_COLORS.get(l, "#3A5A7A") for l in rc_ordered],
                    donut_total,
                ),
                use_container_width=True,
                key="plotly_cmp_raw",
            )
            st.markdown(
                "<p style='color:#6A8AA6;font-size:0.72rem;text-align:center;'>"
                f"타각기 {donut_total}분 기준 · 미수신 {missing_min}분 포함</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Raw BLE 데이터 없음")

    with col_d_cor:
        st.markdown(
            "<p style='text-align:center;color:#7A8FA6;font-size:0.82rem;"
            "margin-bottom:4px;'>🗺 보정 Journey — locus_type</p>",
            unsafe_allow_html=True,
        )
        if cc_ordered:
            st.plotly_chart(
                _make_donut(
                    list(cc_ordered.keys()), list(cc_ordered.values()),
                    [_SPACE_COLORS.get(l, "#3A5A7A") for l in cc_ordered],
                    donut_total,
                ),
                use_container_width=True,
                key="plotly_cmp_cor",
            )
            st.markdown(
                "<p style='color:#6A8AA6;font-size:0.72rem;text-align:center;'>"
                "same_locus gap-fill → 실제 공간 유형  ·  이동 추정 gap-fill → 음영</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("보정 Journey 데이터 없음")

    # ── 하루 Journey Gantt 차트 ──────────────────────────────────────────
    st.markdown(sub_header("하루 Journey 타임라인"), unsafe_allow_html=True)
    _render_journey_gantt(user_tward, user_jdf, place_ltype_map,
                          user_access=user_access)

    # ── 수치 비교 테이블 ─────────────────────────────────────────────────
    with st.expander("📊 Raw vs 보정 수치 상세 (타각기 기준)"):
        rows_cmp = []
        for cat, cnt in rc_ordered.items():
            rows_cmp.append({"구분": "Raw BLE", "카테고리": cat,
                              "분 수": cnt,
                              "비율 (%)": round(cnt / donut_total * 100, 1) if donut_total else 0})
        for cat, cnt in cc_ordered.items():
            rows_cmp.append({"구분": "보정 Journey", "카테고리": cat,
                              "분 수": cnt,
                              "비율 (%)": round(cnt / donut_total * 100, 1) if donut_total else 0})
        if rows_cmp:
            st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

    # ── 시간별 Raw vs 보정 상세 비교 표 ─────────────────────────────────
    with st.expander("🕐 시간별 Raw vs 보정 상세 비교 (1분 단위)"):
        _render_minute_comparison_table(user_tward, user_jdf, place_ltype_map)


def _render_journey_gantt(  # noqa: C901
    user_tward: pd.DataFrame,
    user_jdf: pd.DataFrame,
    place_ltype_map: "dict | None" = None,
    user_access: "pd.DataFrame | None" = None,
) -> None:
    """
    작업자 하루 Journey 가로 Gantt 타임라인 (Raw BLE vs 보정).

    ★ 테이블(시간별 Raw vs 보정 상세 비교)과 완전히 동일한 1분 단위 로직으로
      카테고리를 먼저 결정한 뒤 연속 세그먼트로 묶어서 그린다.

    ★ user_access (AccessLog) 기반 타각기 출입 범위를 추가 → 미수신 구간도 음영 표시.
    ★ 연속 30분 초과 음영 구간은 보정도 음영으로 표시 (gap-fill 신뢰도 낮음).

    카테고리 결정 규칙:
      Raw  row: Signal_count=0 또는 미수신(타각기 범위 내) → 음영
               GATE → 출입  /  REST_AREA → 휴게  /  else → 작업
      보정 row: journey entry 없거나 Raw도 음영 + 연속 30분+ 구간 → 음영
               gap-fill → ltype/btype 기반 실제 카테고리 (단, 30분 이하 구간만)
               gap-fill이면서 Raw는 신호 있음 → ltype+block_type 기반 실제 카테고리
               non-gap → ltype+block_type 기반 실제 카테고리
    """
    import plotly.graph_objects as go

    if user_tward.empty and user_jdf.empty:
        st.info("타임라인 데이터가 없습니다.")
        return

    _plmap: dict = place_ltype_map or {}

    # Gantt 타임라인 색상 — CHART_COLORS 단일 소스 기반 (전사 통일)
    GANTT_COLORS: dict = {
        "출입":    CHART_COLORS["gate"],        # #A78BFA
        "작업":    CHART_COLORS["work_area"],   # #00AEEF (파랑 = 작업공간)
        "이동 중": CHART_COLORS["transit"],     # #9AB5D4
        "휴게":    CHART_COLORS["rest_area"],   # #00C897 (녹색 = 휴게)
        "음영":    CHART_COLORS["gap"],         # #4A5A6A (회색)
    }
    CATEGORY_ORDER = ["출입", "작업", "이동 중", "휴게", "음영"]
    MIN_LABEL_MIN  = 8
    BAR_HEIGHT     = 0.5

    def _ltype_to_cat(ltype: str) -> "str | None":
        if ltype == "GATE":      return "출입"
        if ltype == "REST_AREA": return "휴게"
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — 테이블과 완전히 동일한 방식으로 1분 단위 데이터 딕셔너리 구축
    # ═══════════════════════════════════════════════════════════════════════

    # ── Raw BLE: HH:MM → {place, signal} ─────────────────────────────────
    raw_by_min: dict = {}   # {hhmm: {"place": str, "signal": int}}
    if not user_tward.empty:
        tw = user_tward.copy()
        tw["_ts"]  = pd.to_datetime(tw["Time"], errors="coerce")
        tw["_sig"] = pd.to_numeric(tw.get("Signal_count", 0), errors="coerce").fillna(0).astype(int)
        tw["_pl"]  = tw["Place"].fillna("").str.strip()
        tw = tw.dropna(subset=["_ts"])
        for _, row in tw.iterrows():
            hhmm = row["_ts"].strftime("%H:%M")
            raw_by_min[hhmm] = {"place": row["_pl"], "signal": int(row["_sig"])}

    # ── 보정 Journey: HH:MM → {place, ltype, btype, is_gap} ─────────────
    cor_by_min: dict = {}   # {hhmm: {"place": str, "ltype": str, "btype": str, "is_gap": bool}}
    if not user_jdf.empty:
        jdf = user_jdf.copy()
        jdf["_ts"] = pd.to_datetime(jdf["timestamp"], errors="coerce")
        jdf = jdf.dropna(subset=["_ts"])

        has_spot  = "spot_name"     in jdf.columns
        has_lid   = "locus_id"      in jdf.columns
        has_ltype = "locus_type"    in jdf.columns
        has_btype = "block_type"    in jdf.columns
        has_gap   = "is_gap_filled" in jdf.columns

        # locus_id → spot_name (비 gap 행 기반 — 테이블과 동일 로직)
        loc_spot: dict = {}
        if has_lid and has_spot:
            src = jdf[~jdf["is_gap_filled"].fillna(False)] if has_gap else jdf
            src = src[src["spot_name"].fillna("").str.strip() != ""]
            if not src.empty:
                loc_spot = (
                    src.groupby("locus_id")["spot_name"]
                    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
                    .to_dict()
                )

        for _, row in jdf.iterrows():
            hhmm = row["_ts"].strftime("%H:%M")
            # place: spot_name → loc_spot → locus_id (테이블과 동일)
            sn = str(row.get("spot_name", "")).strip() if has_spot else ""
            if not sn and has_lid:
                lid = str(row.get("locus_id", "")).strip()
                sn = loc_spot.get(lid, lid)
            cor_by_min[hhmm] = {
                "place":  sn,
                "ltype":  str(row.get("locus_type", "")).strip()  if has_ltype else "",
                "btype":  str(row.get("block_type",  "")).strip()  if has_btype else "",
                "is_gap": bool(row["is_gap_filled"]) if has_gap and pd.notna(row.get("is_gap_filled")) else False,
            }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — all_hhmm: 데이터 범위 + 타각기 shift 전체 범위 통합
    #
    # 데이터(raw/cor)에 없는 분도 타각기 범위 안이면 음영으로 표시해야 함.
    # user_access (AccessLog) entry/exit 시간을 이용해 shift 전체 분 목록 생성.
    # ═══════════════════════════════════════════════════════════════════════

    # 날짜 기준점
    ref_date_raw = (
        pd.to_datetime(user_tward["Time"].iloc[0], errors="coerce").date()
        if not user_tward.empty else None
    )
    ref_date_cor = (
        pd.to_datetime(user_jdf["timestamp"].iloc[0], errors="coerce").date()
        if not user_jdf.empty else None
    )
    ref_date = ref_date_raw or ref_date_cor

    # 타각기 기반 shift 전체 분 범위 추출 (벡터화 — pd.date_range)
    # 기존: iterrows + while 루프 × 1440분 = O(n × 1440)
    # 개선: date_range로 분 단위 전체 생성 후 set으로 집합 (수십 배 빠름)
    shift_minutes: set[str] = set()
    if user_access is not None and not user_access.empty:
        _et = pd.to_datetime(user_access["Entry_time"], errors="coerce")
        _xt = pd.to_datetime(user_access["Exit_time"],  errors="coerce")
        for et, xt in zip(_et, _xt):
            if pd.notna(et) and pd.notna(xt) and xt > et:
                start = et.floor("min")
                end   = xt.floor("min")
                rng = pd.date_range(start, end, freq="1min")
                shift_minutes.update(rng.strftime("%H:%M"))

    # 데이터 기반 분 범위 + shift 범위 합집합
    all_hhmm = sorted(set(raw_by_min.keys()) | set(cor_by_min.keys()) | shift_minutes)
    if not all_hhmm:
        st.info("시각화할 데이터가 없습니다.")
        return

    # ── 연속 음영 구간 길이 사전 계산 (30분 규칙용) ──────────────────────
    # 음영 기준: raw_by_min에 없거나 signal=0 인 분
    SHADOW_GATE = 30   # 연속 음영이 이 분 초과이면 보정도 음영
    shadow_set: set[str] = set()
    for h in all_hhmm:
        if h not in raw_by_min or raw_by_min[h]["signal"] == 0:
            shadow_set.add(h)

    # 연속 음영 길이 계산: 각 음영 분이 속한 연속 run의 길이
    shadow_run_len: dict[str, int] = {}
    i = 0
    hhmm_list = all_hhmm  # already sorted
    while i < len(hhmm_list):
        h = hhmm_list[i]
        if h in shadow_set:
            # 연속 run 시작 찾기
            run_start = i
            while i < len(hhmm_list) and hhmm_list[i] in shadow_set:
                # 연속 여부 확인 (1분 간격인지)
                if i > run_start:
                    prev_ts = pd.Timestamp(f"{ref_date or '2000-01-01'} {hhmm_list[i-1]}:00")
                    cur_ts  = pd.Timestamp(f"{ref_date or '2000-01-01'} {hhmm_list[i]}:00")
                    if (cur_ts - prev_ts).total_seconds() > 90:  # 1분 초과 간격 → 새 run
                        break
                i += 1
            run_len = i - run_start
            for j in range(run_start, i):
                shadow_run_len[hhmm_list[j]] = run_len
        else:
            i += 1

    raw_min_rows: list[dict] = []   # {ts, cat, place}
    cor_min_rows: list[dict] = []

    for hhmm in all_hhmm:
        try:
            ts = pd.Timestamp(f"{ref_date} {hhmm}:00")
        except Exception:
            continue

        # ── Raw 1분 카테고리 ────────────────────────────────────────────
        # TWardData entry 없는 분(타각기 범위 포함 "미수신") → 음영
        if hhmm in raw_by_min:
            r = raw_by_min[hhmm]
            if r["signal"] == 0:
                raw_cat = "음영"
            else:
                ltype = _plmap.get(r["place"], "")
                raw_cat = _ltype_to_cat(ltype) or "작업"
            raw_min_rows.append({"ts": ts, "cat": raw_cat, "place": r["place"]})
        else:
            # 미수신 = 음영
            raw_min_rows.append({"ts": ts, "cat": "음영", "place": ""})

        # ── 보정 1분 카테고리 ────────────────────────────────────────────
        # ★ 연속 30분 이상 음영 구간 → gap-fill이라도 음영 (신뢰도 부족)
        # ★ 30분 이하 음영 → gap-fill 실제 카테고리 사용
        is_long_shadow = shadow_run_len.get(hhmm, 0) > SHADOW_GATE

        if hhmm in cor_by_min and not is_long_shadow:
            c = cor_by_min[hhmm]
            # 1차: locus_type 직접 → 2차: place_ltype_map → 3차: block_type
            cor_cat = _ltype_to_cat(c["ltype"])
            if not cor_cat and c["place"]:
                cor_cat = _ltype_to_cat(_plmap.get(c["place"], ""))
            if not cor_cat:
                cor_cat = "이동 중" if c["btype"] == "TRANSIT" else "작업"
            cor_min_rows.append({"ts": ts, "cat": cor_cat, "place": c["place"]})
        else:
            # long shadow이거나 journey entry 없음 → 음영
            cor_min_rows.append({"ts": ts, "cat": "음영", "place": ""})

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — 연속 같은 (카테고리, 장소) 를 세그먼트로 묶기
    # ═══════════════════════════════════════════════════════════════════════
    def _to_segs(rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        segs: list[dict] = []
        cur_cat   = rows[0]["cat"]
        cur_place = rows[0]["place"]
        cur_start = rows[0]["ts"]
        cur_end   = rows[0]["ts"] + pd.Timedelta(minutes=1)
        cur_places: list[str] = [rows[0]["place"]] if rows[0]["place"] else []

        for r in rows[1:]:
            if r["cat"] == cur_cat and r["place"] == cur_place:
                cur_end = r["ts"] + pd.Timedelta(minutes=1)
                if r["place"]:
                    cur_places.append(r["place"])
            else:
                place = pd.Series(cur_places).mode().iloc[0] if cur_places else ""
                dur   = max(1, int(round((cur_end - cur_start).total_seconds() / 60)))
                segs.append({"start": cur_start, "end": cur_end,
                              "cat": cur_cat, "place": place, "dur": dur})
                cur_cat   = r["cat"]
                cur_place = r["place"]
                cur_start = r["ts"]
                cur_end   = r["ts"] + pd.Timedelta(minutes=1)
                cur_places = [r["place"]] if r["place"] else []

        place = pd.Series(cur_places).mode().iloc[0] if cur_places else ""
        dur   = max(1, int(round((cur_end - cur_start).total_seconds() / 60)))
        segs.append({"start": cur_start, "end": cur_end,
                     "cat": cur_cat, "place": place, "dur": dur})
        return segs

    raw_segs = _to_segs(raw_min_rows)
    cor_segs = _to_segs(cor_min_rows)

    if not raw_segs and not cor_segs:
        st.info("시각화할 Journey 세그먼트가 없습니다.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Plotly Gantt 렌더링
    # ═══════════════════════════════════════════════════════════════════════
    all_starts = [s["start"] for s in raw_segs + cor_segs]
    all_ends   = [s["end"]   for s in raw_segs + cor_segs]
    x_min = min(all_starts)
    x_max = max(all_ends)
    total_min = int((x_max - x_min).total_seconds() / 60)

    fig = go.Figure()
    ROWS         = [("📡 Raw BLE", raw_segs), ("🗺 보정", cor_segs)]
    legend_added: set = set()

    # 두 행 전체 시간 범위 확보 (투명 패딩)
    for row_label in ["📡 Raw BLE", "🗺 보정"]:
        fig.add_trace(go.Bar(
            x=[total_min], y=[row_label], base=[0],
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
            showlegend=False, hoverinfo="skip", width=BAR_HEIGHT,
        ))

    for row_label, segs in ROWS:
        for seg in segs:
            cat       = seg["cat"]
            color     = GANTT_COLORS.get(cat, "#888888")
            start_num = (seg["start"] - x_min).total_seconds() / 60
            dur_num   = max(1.0, (seg["end"] - seg["start"]).total_seconds() / 60)
            show_leg  = cat not in legend_added
            fig.add_trace(go.Bar(
                x=[dur_num], y=[row_label], base=[start_num],
                orientation="h",
                marker=dict(color=color, line=dict(width=0)),
                name=cat, legendgroup=cat, showlegend=show_leg,
                hovertemplate=(
                    f"<b>{cat}</b><br>"
                    f"장소: {seg['place'] or '—'}<br>"
                    f"{seg['start'].strftime('%H:%M')}~{seg['end'].strftime('%H:%M')} "
                    f"({seg['dur']}분)<extra></extra>"
                ),
                width=BAR_HEIGHT,
            ))
            if show_leg:
                legend_added.add(cat)

    # 작업 세그먼트 장소명 annotation (겹침 방지)
    MIN_ANNOT_GAP = 20
    last_ax: dict[str, float] = {}
    for row_label, segs in ROWS:
        for seg in sorted(
            [s for s in segs if s["cat"] == "작업" and s["dur"] >= MIN_LABEL_MIN and s["place"]],
            key=lambda s: s["start"],
        ):
            mid = (seg["start"] - x_min).total_seconds() / 60 + seg["dur"] / 2
            if mid - last_ax.get(row_label, -9999) < MIN_ANNOT_GAP:
                continue
            lbl = seg["place"][:13] + "…" if len(seg["place"]) > 13 else seg["place"]
            fig.add_annotation(
                x=mid, y=row_label, text=lbl, showarrow=False,
                font=dict(size=9, color="#111827"),
                bgcolor="rgba(255,255,255,0.82)", borderpad=2,
                xanchor="center", yanchor="middle",
            )
            last_ax[row_label] = mid

    # X축 tick
    tick_step = max(30, (total_min // 8 // 30) * 30)
    tick_vals = list(range(0, total_min + 1, tick_step))
    tick_text = [(x_min + pd.Timedelta(minutes=v)).strftime("%H:%M") for v in tick_vals]

    fig.update_layout(
        barmode="overlay",
        height=240,
        margin=dict(l=90, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickvals=tick_vals, ticktext=tick_text,
            range=[-1, total_min + 1],
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            zeroline=False, color="#9CA3AF",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=["🗺 보정", "📡 Raw BLE"],
            tickfont=dict(size=12), color="#9CA3AF",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0,
            font=dict(size=11, color="#D1D5DB"),
            bgcolor="rgba(0,0,0,0)", itemwidth=50,
        ),
        font=dict(color="#D1D5DB"),
    )
    for cat in CATEGORY_ORDER:
        for trace in fig.data:
            if trace.name == cat:
                trace.legendrank = CATEGORY_ORDER.index(cat)
                break

    st.plotly_chart(fig, use_container_width=True, key="gantt_journey")
    st.caption(
        "출입 = 타각기/게이트 구역  ·  작업 = 체류 블록  ·  이동 중 = TRANSIT 블록  ·  "
        "휴게 = 휴게실/흡연실  ·  음영 = Raw 신호 없음 / Raw도 없는 gap-fill"
    )


def _render_minute_comparison_table(
    user_tward: pd.DataFrame,
    user_jdf: pd.DataFrame,
    place_ltype_map: "dict | None" = None,
) -> None:
    """
    1분 단위 Raw BLE vs 보정 Journey 병렬 비교 표.

    세로축: 시간 (00:00 ~ 23:59)
    좌측: Raw BLE — Place, Signal_count
    우측: 보정 Journey — spot_name (Place 계통), 보정 유형
    활동 분류: Gantt와 동일 규칙 (locus_type 우선 → block_type 폴백)
    """
    _plmap: dict = place_ltype_map or {}
    if user_tward.empty and user_jdf.empty:
        st.info("비교할 데이터가 없습니다.")
        return

    # ── Raw BLE: "HH:MM" → {place, signal, active} ──────────────────
    raw_rows: dict = {}
    if not user_tward.empty:
        tw = user_tward.copy()
        tw["_ts"] = pd.to_datetime(tw["Time"], errors="coerce")
        tw = tw.dropna(subset=["_ts"])
        tw["_hhmm"] = tw["_ts"].dt.strftime("%H:%M")
        for _, row in tw.iterrows():
            raw_rows[row["_hhmm"]] = {
                "place":  str(row.get("Place",  "")).strip() or "—",
                "signal": int(row["Signal_count"])       if pd.notna(row.get("Signal_count"))       else 0,
                "active": int(row["ActiveSignal_count"]) if pd.notna(row.get("ActiveSignal_count")) else 0,
            }

    # ── 보정 Journey: "HH:MM" → {place, is_gap, method, conf} ───────
    # spot_name을 우선 사용 (Raw Place와 같은 계통)
    # gap-filled 행은 spot_name이 ""로 저장되므로, 비 gap 행에서 locus_id→spot_name 매핑을 먼저 빌드
    cor_rows: dict = {}
    if not user_jdf.empty:
        jdf = user_jdf.copy()
        jdf["_ts"]   = pd.to_datetime(jdf["timestamp"], errors="coerce")
        jdf = jdf.dropna(subset=["_ts"])
        jdf["_hhmm"] = jdf["_ts"].dt.strftime("%H:%M")

        has_spot   = "spot_name"      in jdf.columns
        has_lname  = "locus_name"     in jdf.columns
        has_lid    = "locus_id"       in jdf.columns
        has_gap    = "is_gap_filled"  in jdf.columns
        has_meth   = "gap_method"     in jdf.columns
        has_conf   = "gap_confidence" in jdf.columns
        has_btype  = "block_type"     in jdf.columns
        has_ltype  = "locus_type"     in jdf.columns

        # locus_id → 실제 장소명 매핑: 비 gap-filled 행의 spot_name 최빈값
        locus_spot_map: dict = {}
        if has_lid and has_spot:
            src = jdf[~jdf["is_gap_filled"].fillna(False)] if has_gap else jdf
            src = src[src["spot_name"].fillna("").str.strip() != ""]
            if not src.empty:
                locus_spot_map = (
                    src.groupby("locus_id")["spot_name"]
                    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
                    .to_dict()
                )

        for _, row in jdf.iterrows():
            hhmm = row["_hhmm"]
            # 보정 Journey 장소:
            #   1) spot_name (비어있지 않으면 — 실제 BLE 수신 행)
            #   2) locus_spot_map[locus_id] — 같은 locus의 비 gap 행에서 가져온 spot_name
            #   3) locus_name
            #   4) locus_id (최후 폴백)
            place = ""
            if has_spot and str(row.get("spot_name", "")).strip():
                place = str(row["spot_name"]).strip()
            if not place and has_lid:
                lid = str(row.get("locus_id", "")).strip()
                place = locus_spot_map.get(lid, "")
            if not place and has_lname and pd.notna(row.get("locus_name")) and str(row["locus_name"]).strip():
                place = str(row["locus_name"]).strip()
            if not place and has_lid:
                place = str(row.get("locus_id", "")).strip()
            # 활동 분류: Gantt와 동일 규칙 (locus_type 우선 → place_ltype_map → block_type)
            ltype_val = str(row.get("locus_type", "")).strip() if has_ltype else ""
            btype     = str(row.get("block_type",  "")).strip() if has_btype and pd.notna(row.get("block_type")) else ""
            if ltype_val == "GATE":
                activity = "출입"
            elif ltype_val == "REST_AREA":
                activity = "휴게"
            elif not ltype_val and place:
                # locus_type 없을 때 place_ltype_map 경유 (gap-fill 등)
                mapped = _plmap.get(place, "")
                if mapped == "GATE":
                    activity = "출입"
                elif mapped == "REST_AREA":
                    activity = "휴게"
                elif btype == "TRANSIT":
                    activity = "이동 중"
                elif btype == "WORK":
                    activity = "작업"
                else:
                    activity = ""
            elif btype == "TRANSIT":
                activity = "이동 중"
            elif btype == "WORK":
                activity = "작업"
            else:
                activity = ""
            cor_rows[hhmm] = {
                "place":    place or "—",
                "is_gap":   bool(row["is_gap_filled"]) if has_gap else False,
                "method":   str(row["gap_method"])     if has_meth and pd.notna(row.get("gap_method"))     else "",
                "conf":     str(row["gap_confidence"]) if has_conf and pd.notna(row.get("gap_confidence")) else "",
                "activity": activity,
            }

    # ── 전체 시간 범위 ────────────────────────────────────────────────
    all_hhmm = sorted(set(raw_rows.keys()) | set(cor_rows.keys()))
    if not all_hhmm:
        st.info("표시할 데이터가 없습니다.")
        return

    # ── 표 데이터 구성 ───────────────────────────────────────────────
    rows = []
    for hhmm in all_hhmm:
        r = raw_rows.get(hhmm, {})
        c = cor_rows.get(hhmm, {})

        raw_place  = r.get("place",  "—")
        raw_signal = r.get("signal", "")
        raw_active = r.get("active", "")
        cor_place  = c.get("place",  "—")
        is_gap     = c.get("is_gap", False)
        method     = c.get("method", "")
        conf       = c.get("conf",   "")

        # 보정 유형 태그
        if is_gap:
            tag = f"gap-fill · {method}" if method else "gap-fill"
        elif hhmm in cor_rows and hhmm not in raw_rows:
            tag = "보정-only"
        elif hhmm in raw_rows and hhmm not in cor_rows:
            tag = "raw-only"
        else:
            tag = ""

        rows.append({
            "시간":            hhmm,
            "📡 Raw — Place":  raw_place,
            "신호":            str(raw_signal) if raw_signal != "" else "—",
            "활성":            str(raw_active) if raw_active != "" else "—",
            "🗺 보정 — Place": cor_place,
            "활동":            c.get("activity", ""),
            "보정 유형":       tag,
            "신뢰도":          conf,
            "_is_gap":         is_gap,
            "_no_raw":         hhmm not in raw_rows,
        })

    df_table = pd.DataFrame(rows)
    n_gap = int(df_table["_is_gap"].sum())

    # ── 필터 옵션 ────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        show_all = st.checkbox(
            "전체 시간 표시", value=True, key="mct_show_all",
            help="ON: 모든 1분 행 표시  /  OFF: gap-fill·차이 행만"
        )
    with col_f2:
        only_gap = st.checkbox(
            "gap-fill만 보기", value=False, key="mct_only_gap",
            help="보정으로 채워진 구간만 표시"
        )
    with col_f3:
        st.caption(f"총 {len(df_table)}분 · gap-fill {n_gap}분 · Raw {len(raw_rows)}분")

    # ── 필터 적용 ────────────────────────────────────────────────────
    disp = df_table.copy()
    if only_gap:
        disp = disp[disp["_is_gap"]]
    elif not show_all:
        disp = disp[disp["_is_gap"] | disp["_no_raw"] | (disp["보정 유형"] != "")]

    if disp.empty:
        st.success("✅ 차이 없음 — Raw BLE와 보정 Journey가 완전히 일치합니다.")
        return

    disp_cols = ["시간", "📡 Raw — Place", "신호", "활성", "🗺 보정 — Place", "활동", "보정 유형", "신뢰도"]
    st.dataframe(
        disp[disp_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "시간":            st.column_config.TextColumn("시간",            width="small"),
            "📡 Raw — Place":  st.column_config.TextColumn("📡 Raw BLE",      width="medium"),
            "신호":            st.column_config.TextColumn("신호수",          width="small"),
            "활성":            st.column_config.TextColumn("활성신호",        width="small"),
            "🗺 보정 — Place": st.column_config.TextColumn("🗺 보정 Journey",  width="medium"),
            "활동":            st.column_config.TextColumn("활동 분류",       width="small"),
            "보정 유형":       st.column_config.TextColumn("보정 유형",       width="medium"),
            "신뢰도":          st.column_config.TextColumn("신뢰도",          width="small"),
        },
    )
    st.caption(
        f"표시: {len(disp)}행 / 전체 {len(df_table)}분  ·  "
        "gap-fill = BLE 음영 구간 보정값  ·  보정-only = Journey 있으나 Raw 없는 구간"
    )


def _render_locus_agg_table(user_agg_df: pd.DataFrame, locus_meta: pd.DataFrame) -> None:
    """
    Cloud agg-only 모드: journey_agg.parquet 기반 작업자 locus 체류 테이블.

    분 단위 journey 없을 때 집계 데이터로 "어디서 얼마나 있었는지"를 보여준다.
    """
    if user_agg_df.empty:
        st.info("집계 데이터가 없습니다.")
        return

    st.caption(
        "ℹ️ **집계 데이터 모드** · `journey_agg.parquet` 기반 (분 단위 상세 없음).  \n"
        "locus별 총 체류분, 근무시간 내 체류분, 활성도 분포를 표시합니다."
    )

    # locus_meta 조인 (locus_type, building, floor)
    display = user_agg_df.copy()
    if not locus_meta.empty and "locus_type" in locus_meta.columns:
        display = display.merge(
            locus_meta[["locus_id", "locus_type"]].rename(columns={"locus_type": "_ltype"}),
            on="locus_id", how="left",
        )
        display["locus_type"] = display["_ltype"].fillna("UNKNOWN")
        display.drop(columns=["_ltype"], inplace=True, errors="ignore")
    else:
        display["locus_type"] = "UNKNOWN"

    display = display.sort_values("total_min", ascending=False).reset_index(drop=True)

    # KPI 요약
    total = int(display["total_min"].sum())
    work  = int(display["work_hour_min"].sum())
    gap   = int(display["gap_filled_min"].sum())
    low_c = int(display["low_confidence_min"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("총 체류", f"{total}분"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("근무시간 체류", f"{work}분"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Gap-fill", f"{gap}분"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("저신뢰 구간", f"{low_c}분"), unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # 테이블 컬럼 정리
    col_map = {
        "locus_id":              "Locus ID",
        "locus_name":            "장소명",
        "building_name":         "건물",
        "floor_name":            "층",
        "locus_type":            "유형",
        "total_min":             "총 체류(분)",
        "work_hour_min":         "근무시간(분)",
        "gap_filled_min":        "Gap-fill(분)",
        "low_confidence_min":    "저신뢰(분)",
        "activity_high_min":     "고활성(분)",
        "activity_medium_min":   "중활성(분)",
        "activity_low_min":      "저활성(분)",
        "avg_active_ratio":      "평균 활성비율",
        "avg_signal_count":      "평균 신호수",
    }
    show_cols = [c for c in col_map if c in display.columns]
    tbl = display[show_cols].rename(columns=col_map)
    tbl["평균 활성비율"] = tbl["평균 활성비율"].map("{:.2f}".format) if "평균 활성비율" in tbl.columns else None
    tbl["평균 신호수"]   = tbl["평균 신호수"].map("{:.1f}".format) if "평균 신호수" in tbl.columns else None

    st.dataframe(tbl, use_container_width=True, height=400)


def _render_locus_map_agg(user_agg_df: pd.DataFrame, locus_meta: pd.DataFrame) -> None:
    """
    Cloud agg-only 모드: journey_agg.parquet 기반 locus 체류 히트맵.

    분 단위 이동 경로 대신 "locus별 총 체류분"으로 버블 크기를 표현.
    """
    if user_agg_df.empty:
        st.info("집계 데이터가 없습니다.")
        return
    if locus_meta.empty or "locus_x" not in locus_meta.columns:
        st.info("locus 좌표 메타데이터가 없습니다.")
        return

    st.caption(
        "ℹ️ **집계 데이터 모드** · `journey_agg.parquet` 기반 버블 맵.  \n"
        "버블 크기 = 총 체류분. 이동 경로(선)는 분 단위 `journey.parquet`가 필요합니다."
    )

    # locus_meta 조인
    df = user_agg_df.merge(
        locus_meta[["locus_id", "locus_x", "locus_y", "building", "floor",
                     "locus_type", "locus_meta_name"]],
        on="locus_id", how="left",
    )
    df = df.dropna(subset=["locus_x", "locus_y"])
    if df.empty:
        st.info("좌표 정보가 있는 locus 데이터가 없습니다.")
        return

    # 건물/층 선택
    bldg_floor_combos = (
        df[df["building"].notna()][["building", "floor"]]
        .dropna().drop_duplicates().sort_values(["building", "floor"])
    )
    view_options = ["🌐 전체"] + [
        f"{r.building} / {r.floor}"
        for r in bldg_floor_combos.itertuples(index=False)
    ]
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_view = st.selectbox(
            "건물 / 층 선택", view_options, key="locus_agg_map_sel",
        )

    if "전체" in selected_view:
        view_df = df.copy()
        bg_loci = locus_meta[locus_meta["building"].isna()]
        title_txt = "전체"
    else:
        bldg, floor = selected_view.split(" / ", 1)
        view_df = df[(df["building"] == bldg) & (df["floor"] == floor)]
        bg_loci = locus_meta[(locus_meta["building"] == bldg) & (locus_meta["floor"] == floor)]
        title_txt = f"{bldg} {floor}"

    with col_info:
        n_loci = len(view_df)
        total_min = int(view_df["total_min"].sum())
        st.markdown(
            f"<div style='padding:8px 12px; background:#0D1B2A; border-radius:6px; "
            f"font-size:0.83rem; color:#9AB5D4; margin-top:4px;'>"
            f"<b style='color:#D5E5FF'>{title_txt}</b> — "
            f"방문 locus <b style='color:#00C897'>{n_loci}곳</b> · "
            f"총 체류 <b style='color:#00AEEF'>{total_min}분</b></div>",
            unsafe_allow_html=True,
        )

    if view_df.empty:
        st.info(f"'{selected_view}' 구역에 해당하는 체류 데이터가 없습니다.")
        return

    fig = go.Figure()

    # 배경 locus
    if not bg_loci.empty:
        bg_meta_name = bg_loci.get("locus_meta_name", bg_loci.get("locus_id", pd.Series(dtype=str))).fillna("")
        fig.add_trace(go.Scatter(
            x=bg_loci["locus_x"], y=bg_loci["locus_y"],
            mode="markers+text",
            text=bg_loci["locus_id"],
            textposition="top center",
            textfont=dict(size=7, color="#4A6A8A"),
            marker=dict(size=12, color="#1E3A5A", opacity=0.25,
                        line=dict(width=1, color="#1E3A5A")),
            name="locus 배경",
            customdata=np.column_stack([
                bg_loci["locus_id"].values,
                bg_meta_name.values,
            ]),
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<extra></extra>",
        ))

    # 체류 버블 (크기 = total_min)
    max_min = view_df["total_min"].max()
    bubble_size = ((view_df["total_min"] / max(max_min, 1)) * 40 + 8).clip(8, 50).tolist()
    work_ratio = (view_df["work_hour_min"] / view_df["total_min"].clip(1)).fillna(0)
    meta_name_col = view_df.get("locus_meta_name", view_df.get("locus_name", pd.Series(dtype=str))).fillna("")

    fig.add_trace(go.Scatter(
        x=view_df["locus_x"],
        y=view_df["locus_y"],
        mode="markers+text",
        text=view_df["locus_id"],
        textposition="top center",
        textfont=dict(size=8, color="#C8D6E8"),
        name="체류 locus",
        marker=dict(
            size=bubble_size,
            color=work_ratio,
            colorscale="Blues",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title="근무시간 비율",
                tickvals=[0, 0.5, 1],
                ticktext=["0%", "50%", "100%"],
                len=0.6, thickness=12,
            ),
            opacity=0.85,
            line=dict(width=1, color="#0D1B2A"),
        ),
        customdata=np.column_stack([
            view_df["locus_id"].values,
            meta_name_col.values,
            view_df["total_min"].values,
            view_df["work_hour_min"].values,
            view_df["gap_filled_min"].values,
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
            "총 체류: <b>%{customdata[2]}분</b><br>"
            "근무시간: %{customdata[3]}분<br>"
            "Gap-fill: %{customdata[4]}분<br>"
            "x=%{x:.0f}, y=%{y:.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=20, r=60, t=30, b=20)},
        height=520,
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1,
                   gridcolor="#1A2A3A", zeroline=False),
        yaxis=dict(title="Y", autorange="reversed",
                   gridcolor="#1A2A3A", zeroline=False),
        title=dict(text=f"Locus 체류 버블 맵 — {title_txt} (집계 데이터)",
                   font=dict(size=13, color="#9AB5D4"), x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_locus_map(user_jdf: pd.DataFrame, locus_meta: pd.DataFrame) -> None:
    """
    작업자 locus 이동 경로 — 2D 공간 맵.

    journey.parquet의 locus_id → locus_v2 location_x/y 좌표를 이용해
    건물/층별 평면도 위에 이동 경로를 그린다.

    (주의) Raw TWardData의 Place 정보와 별개 — pipeline이 부여한 locus_id 기반.
    """
    if user_jdf.empty:
        st.info("journey 데이터가 없습니다.")
        return
    if locus_meta.empty or "locus_x" not in locus_meta.columns:
        st.info("locus 좌표 메타데이터가 없습니다.")
        return

    st.caption(
        "ℹ️ 이 맵은 pipeline이 부여한 **locus_id** 기반입니다. "
        "Raw TWardData의 Place와는 다르며, KDTree 매핑으로 X/Y → 최근접 Gateway를 할당한 결과입니다."
    )

    # user_jdf에 locus 좌표 조인 (이미 enriched면 locus_x가 있음)
    if "locus_x" not in user_jdf.columns:
        df = user_jdf.merge(
            locus_meta[["locus_id", "locus_x", "locus_y", "building", "floor",
                         "locus_type", "locus_meta_name"]],
            on="locus_id", how="left", suffixes=("", "_meta2"),
        )
    else:
        df = user_jdf.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── 건물/층 선택 ────────────────────────────────────────────────
    has_bldg = df["building"].notna()

    # 방문한 건물/층 목록
    bldg_floor_combos = (
        df[has_bldg][["building", "floor"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["building", "floor"])
    )
    view_options = ["🌐 공사현장 전체 (건물 미지정)"] + [
        f"{r.building} / {r.floor}"
        for r in bldg_floor_combos.itertuples(index=False)
    ]

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_view = st.selectbox(
            "건물 / 층 선택",
            view_options,
            key="locus_map_view_sel",
        )

    # ── 데이터 필터링 ────────────────────────────────────────────────
    if "전체" in selected_view:
        worker_path = df[~has_bldg].sort_values("timestamp").reset_index(drop=True)
        # 해당 구역 전체 locus 배경
        bg_loci = locus_meta[locus_meta["building"].isna()]
        title_txt = "공사현장 전체 (건물 미지정 locus)"
    else:
        bldg, floor = selected_view.split(" / ", 1)
        worker_path = df[
            (df["building"] == bldg) & (df["floor"] == floor)
        ].sort_values("timestamp").reset_index(drop=True)
        bg_loci = locus_meta[
            (locus_meta["building"] == bldg) & (locus_meta["floor"] == floor)
        ]
        title_txt = f"{bldg} {floor}"

    with col_info:
        n_pts = len(worker_path)
        n_loci = worker_path["locus_id"].nunique()
        n_gap = int(worker_path["is_gap_filled"].sum()) if "is_gap_filled" in worker_path.columns else 0
        st.markdown(
            f"<div style='padding:8px 12px; background:#0D1B2A; border-radius:6px; "
            f"font-size:0.83rem; color:#9AB5D4; margin-top:4px;'>"
            f"<b style='color:#D5E5FF'>{title_txt}</b> — "
            f"이동 기록 <b style='color:#00AEEF'>{n_pts}분</b> · "
            f"방문 locus <b style='color:#00C897'>{n_loci}곳</b> · "
            f"Gap-fill <b style='color:#FFB300'>{n_gap}분</b></div>",
            unsafe_allow_html=True,
        )

    if worker_path.empty:
        st.info(f"'{selected_view}' 구역에 해당하는 journey 기록이 없습니다.")
        return

    # ── Plotly 2D 맵 ────────────────────────────────────────────────
    fig = go.Figure()

    # 배경: 해당 층 전체 locus 포인트 (회색)
    if not bg_loci.empty:
        ltype_col = bg_loci.get("locus_type", pd.Series(dtype=str)).fillna("UNKNOWN")
        bg_colors = ltype_col.map(
            {**LOCUS_TYPE_COLOR, "UNKNOWN": "#1E3A5A"}
        ).fillna("#1E3A5A")
        bg_name = bg_loci.get("locus_meta_name", bg_loci.get("locus_id", pd.Series(dtype=str))).fillna("")
        fig.add_trace(go.Scatter(
            x=bg_loci["locus_x"],
            y=bg_loci["locus_y"],
            mode="markers+text",
            text=bg_loci["locus_id"],
            textposition="top center",
            textfont=dict(size=7, color="#4A6A8A"),
            marker=dict(
                size=14,
                color=bg_colors,
                opacity=0.25,
                line=dict(width=1, color=bg_colors, ),
            ),
            name="locus 배경",
            customdata=np.column_stack([
                bg_loci["locus_id"].values,
                bg_name.values,
                ltype_col.values,
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                "유형: %{customdata[2]}<br>"
                "x=%{x:.0f}, y=%{y:.0f}<extra></extra>"
            ),
        ))

    # 이동 경로 선 (흐릿한 선)
    fig.add_trace(go.Scatter(
        x=worker_path["locus_x"],
        y=worker_path["locus_y"],
        mode="lines",
        line=dict(color="rgba(0,174,239,0.25)", width=2),
        name="이동 경로",
        showlegend=False,
        hoverinfo="skip",
    ))

    # 작업자 방문 포인트 (시간 컬러 그라디언트)
    ts = worker_path["timestamp"]
    t_min, t_max = ts.min(), ts.max()
    if t_max > t_min:
        t_norm = ((ts - t_min) / (t_max - t_min)).tolist()
    else:
        t_norm = [0.5] * len(worker_path)

    # Gap-fill 구간은 X 마커, 일반은 circle
    symbols = worker_path.get("is_gap_filled", pd.Series([False]*len(worker_path))).map(
        {True: "x", False: "circle"}
    ).fillna("circle").tolist()

    act_col = (
        worker_path["activity_level"].fillna("")
        if "activity_level" in worker_path.columns
        else pd.Series([""] * len(worker_path), dtype=str)
    )
    gap_col = worker_path.get("is_gap_filled", pd.Series([False]*len(worker_path))).fillna(False)
    locus_meta_name_col = worker_path.get("locus_meta_name", worker_path.get("locus_name", pd.Series(dtype=str))).fillna("")

    fig.add_trace(go.Scatter(
        x=worker_path["locus_x"],
        y=worker_path["locus_y"],
        mode="markers",
        name="방문 locus",
        marker=dict(
            size=10,
            color=t_norm,
            colorscale="Plasma",
            showscale=True,
            symbol=symbols,
            colorbar=dict(
                title="시간",
                tickvals=[0, 0.5, 1],
                ticktext=[
                    ts.min().strftime("%H:%M"),
                    ts.iloc[len(ts)//2].strftime("%H:%M"),
                    ts.max().strftime("%H:%M"),
                ],
                len=0.6,
                thickness=12,
            ),
            line=dict(width=1, color="#0D1B2A"),
        ),
        customdata=np.column_stack([
            worker_path["locus_id"].values,
            locus_meta_name_col.values,
            ts.dt.strftime("%H:%M").values,
            act_col.values,
            gap_col.astype(str).values,
        ]),
        hovertemplate=(
            "<b>%{customdata[2]}</b> → <b>%{customdata[0]}</b> %{customdata[1]}<br>"
            "활성도: %{customdata[3]}<br>"
            "Gap-fill: %{customdata[4]}<br>"
            "x=%{x:.0f}, y=%{y:.0f}<extra></extra>"
        ),
    ))

    # 시작/종료 마커 강조
    for row, label, color, sym in [
        (worker_path.iloc[0],  "▶ 시작", "#00C897", "triangle-right"),
        (worker_path.iloc[-1], "■ 종료", "#FF4C4C", "square"),
    ]:
        fig.add_trace(go.Scatter(
            x=[row["locus_x"]], y=[row["locus_y"]],
            mode="markers+text",
            text=[label],
            textposition="top right",
            textfont=dict(size=10, color=color),
            marker=dict(size=14, color=color, symbol=sym,
                        line=dict(width=1, color="#0D1B2A")),
            name=label,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=20, r=60, t=30, b=20)},
        height=520,
        xaxis=dict(title="X", scaleanchor="y", scaleratio=1,
                   gridcolor="#1A2A3A", zeroline=False),
        yaxis=dict(title="Y", autorange="reversed",   # 좌상단 원점 (건설현장 좌표계)
                   gridcolor="#1A2A3A", zeroline=False),
        legend={**PLOTLY_LEGEND, "x": 0, "y": 1},
    )

    st.plotly_chart(fig, use_container_width=True, key="plotly_8")

    # locus 유형 범례
    st.markdown(
        "<div style='font-size:0.78rem; color:#7A8FA6; margin-top:2px; display:flex; gap:14px; flex-wrap:wrap;'>"
        + "".join(
            f"<span><b style='color:{c}'>●</b> {LOCUS_TYPE_KO.get(k, k)}</span>"
            for k, c in LOCUS_TYPE_COLOR.items()
        )
        + "<span style='opacity:0.5'>(배경 원 = 해당 층 전체 locus / 채색 점 = 작업자 방문)</span>"
        + "<span><b>✕</b> = Gap-fill 보정</span></div>",
        unsafe_allow_html=True,
    )

    # ── 방문 locus 상세 목록 ────────────────────────────────────────
    with st.expander(f"📋 방문 locus 목록 ({worker_path['locus_id'].nunique()}곳)"):
        visit_summary = (
            worker_path.groupby(["locus_id"])
            .agg(
                locus_name=("locus_meta_name", "first") if "locus_meta_name" in worker_path.columns else ("locus_name", "first"),
                locus_type=("locus_type", "first") if "locus_type" in worker_path.columns else ("locus_id", "count"),
                첫방문=("timestamp", lambda s: s.min().strftime("%H:%M")),
                마지막방문=("timestamp", lambda s: s.max().strftime("%H:%M")),
                체류분=("locus_id", "count"),
            )
            .reset_index()
            .sort_values("첫방문")
        )
        visit_summary["공간유형"] = visit_summary.get("locus_type", pd.Series(dtype=str)).map(LOCUS_TYPE_KO).fillna("미분류")
        st.dataframe(
            visit_summary[["locus_id", "locus_name", "공간유형", "첫방문", "마지막방문", "체류분"]],
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# 보정 Journey 탭 — 공간유형 × 활성도 상세 분석 (헬멧 방치 탐지 포함)
# ═══════════════════════════════════════════════════════════════════════
# 임계값/색상은 상단(라인 100 근처)에서 모듈 전역 상수로 정의됨.


def _detect_helmet_suspect_runs(
    df: pd.DataFrame, min_run: int = HELMET_SUSPECT_MIN_RUN
) -> pd.DataFrame:
    """WORK_AREA + LOW 활성도가 연속 min_run 분 이상 지속되는 구간 리스트 반환.

    반환 컬럼: start, end, minutes, locus_label, avg_active_ratio
    """
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "minutes", "locus_label", "avg_active_ratio"])

    d = df.sort_values("timestamp").reset_index(drop=True).copy()
    if "locus_type" not in d.columns:
        return pd.DataFrame(columns=["start", "end", "minutes", "locus_label", "avg_active_ratio"])

    d["_tier"] = _compute_act_tier(d)
    d["_suspect"] = (d["locus_type"] == "WORK_AREA") & (d["_tier"] == "LOW")

    # run-length 그룹핑
    run_id = (d["_suspect"] != d["_suspect"].shift()).cumsum()
    d["_run"] = run_id

    susp = d[d["_suspect"]].copy()
    if susp.empty:
        return pd.DataFrame(columns=["start", "end", "minutes", "locus_label", "avg_active_ratio"])

    def _label(g: pd.DataFrame) -> str:
        # 가장 많이 등장한 locus_name · building · floor
        loc = g["locus_name"].fillna("").astype(str).mode()
        loc_s = loc.iloc[0] if not loc.empty else ""
        bldg = g.get("building", pd.Series([""]*len(g))).fillna("").astype(str).mode()
        floor = g.get("floor", pd.Series([""]*len(g))).fillna("").astype(str).mode()
        bl = f"{bldg.iloc[0] if not bldg.empty else ''} {floor.iloc[0] if not floor.empty else ''}".strip()
        return f"{bl} · {loc_s}" if bl else (loc_s or "UNKNOWN")

    agg = susp.groupby("_run").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        minutes=("timestamp", "count"),
        avg_active_ratio=("active_ratio", "mean"),
    ).reset_index(drop=True)

    labels: list[str] = []
    for rid, g in susp.groupby("_run"):
        labels.append(_label(g))
    agg["locus_label"] = labels

    return agg[agg["minutes"] >= min_run].reset_index(drop=True)

def _render_corrected_journey(
    user_jdf: pd.DataFrame,
    access_total_min: float = 0.0,
    user_info=None,
) -> None:
    """보정된 Journey 상세 분석 — 공간유형 × 활성도 교차 분석으로
    '작업공간에 있었지만 실제로 활동하지 않음' (헬멧 방치) 을 드러낸다.

    섹션:
      A. 보정 Journey 총괄 KPI (5개 카드)
      B. 공간 유형 × 활성도 교차 테이블
      C. 시간대별 활성도 스택 바 차트
      D. 헬멧 방치 의심 구간 타임라인
      E. Journey Timeline (공간유형 × 활성도 색상)
    """
    st.markdown(section_header("보정 Journey — 공간 유형 × 활성도 상세"), unsafe_allow_html=True)
    st.caption(
        f"임계값: 고활성 ≥ {ACTIVE_HIGH_THRESHOLD:.2f}  ·  "
        f"저활성 ≤ {ACTIVE_LOW_THRESHOLD:.2f}  ·  "
        f"헬멧 방치 판정: 작업공간(WORK_AREA) + 저활성이 "
        f"연속 {HELMET_SUSPECT_MIN_RUN}분 이상 지속"
    )

    df = user_jdf.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    has_ltype = "locus_type" in df.columns
    if not has_ltype:
        st.warning(
            "locus_type 메타가 조인되지 않았습니다. "
            "공간 유형 기반 분석은 생략되고 기본 타임라인만 표시됩니다."
        )
        _render_locus_timeline(user_jdf)
        return

    # 활성도 티어 계산 (단일 소스)
    df["_tier"] = _compute_act_tier(df)

    total_min      = len(df)
    wa_mask        = df["locus_type"] == "WORK_AREA"
    wa_min         = int(wa_mask.sum())
    wa_high_min    = int((wa_mask & (df["_tier"] == "HIGH")).sum())
    wa_low_min     = int((wa_mask & (df["_tier"] == "LOW")).sum())

    # 헬멧 방치 의심 구간
    suspect_runs = _detect_helmet_suspect_runs(df, HELMET_SUSPECT_MIN_RUN)
    suspect_min  = int(suspect_runs["minutes"].sum()) if not suspect_runs.empty else 0
    suspect_pct  = (suspect_min / total_min * 100) if total_min > 0 else 0.0

    # ── 섹션 A: 총괄 KPI (5개 카드) ────────────────────────────────
    st.markdown(sub_header("A. 총괄 KPI"), unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    wa_pct   = wa_min / total_min * 100 if total_min > 0 else 0
    waH_pct  = wa_high_min / total_min * 100 if total_min > 0 else 0
    waL_pct  = wa_low_min / total_min * 100 if total_min > 0 else 0

    with col1:
        st.markdown(
            metric_card("총 체류", f"{total_min}분",
                        delta=f"{total_min/60:.1f}h"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            metric_card("작업공간 체류",
                        f"{wa_min}분",
                        delta=f"{wa_pct:.1f}%",
                        color=CHART_COLORS["work_area"]),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            metric_card("작업공간+고활성",
                        f"{wa_high_min}분",
                        delta=f"{waH_pct:.1f}% · 실제 작업",
                        color=CHART_COLORS["high_active"]),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            metric_card("작업공간+저활성",
                        f"{wa_low_min}분",
                        delta=f"{waL_pct:.1f}% · 정지/대기",
                        color=CHART_COLORS["low_active"]),
            unsafe_allow_html=True,
        )
    with col5:
        suspect_color = (COLORS["danger"]  if suspect_pct >= 15
                         else COLORS["warning"] if suspect_pct >= 5
                         else COLORS["success"])
        st.markdown(
            metric_card(f"헬멧 방치 의심 (≥{HELMET_SUSPECT_MIN_RUN}분 연속)",
                        f"{suspect_min}분",
                        delta=f"{suspect_pct:.1f}% · {len(suspect_runs)}개 구간",
                        delta_up=(suspect_pct < 5),
                        color=suspect_color),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 섹션 B: 공간 유형 × 활성도 교차 테이블 ─────────────────────
    st.markdown(sub_header("B. 공간 유형 × 활성도 교차 분석"), unsafe_allow_html=True)
    st.caption(
        "각 셀: 해당 공간·활성도 조합에서 머문 분 수 (전체 대비 %). "
        "작업공간 + 고활성 = 실제 작업, 작업공간 + 저활성 = 헬멧 방치 후보."
    )

    space_order = ["WORK_AREA", "TRANSIT", "REST_AREA", "GATE"]
    tier_order  = ["HIGH", "MID", "LOW", "GAP"]

    # 집계 (pivot)
    pivot_counts = (
        df.groupby([df["locus_type"].fillna("UNKNOWN"), df["_tier"]])
        .size()
        .unstack(fill_value=0)
    )
    # 행/열 순서 고정
    pivot_counts = pivot_counts.reindex(index=space_order, columns=tier_order, fill_value=0)
    pivot_counts.loc["합계"] = pivot_counts.sum(axis=0)
    pivot_counts["합계"] = pivot_counts.sum(axis=1)

    # 표시용 DataFrame: "NNN분 (PP%)"
    def _fmt_cell(v: int) -> str:
        pct = (v / total_min * 100) if total_min > 0 else 0
        return f"{int(v)}분 ({pct:.1f}%)" if v > 0 else "—"

    disp = pivot_counts.map(_fmt_cell) if hasattr(pivot_counts, "map") else pivot_counts.applymap(_fmt_cell)
    disp = disp.rename(
        index={"WORK_AREA": "작업공간", "TRANSIT": "이동공간",
               "REST_AREA": "휴게공간", "GATE": "출입공간",
               "UNKNOWN": "미분류"},
        columns={k: _ACT_TIER_KO[k] for k in tier_order}
    )
    disp.index.name = "공간 유형"
    st.dataframe(disp, use_container_width=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 섹션 C: 시간대별 활성도 스택 바 ────────────────────────────
    st.markdown(sub_header("C. 시간대별 활성도 구성 (1시간 단위)"), unsafe_allow_html=True)
    st.caption(
        "각 시간대에 고활성 / 중간 / 저활성 / 음영 분 수를 쌓아 표시. "
        "특정 시간대에 저활성(짙은 파랑) 비중이 높다면 해당 시간 실제 활동 없음."
    )

    df["_hour"] = df["timestamp"].dt.hour
    hour_tier = (
        df.groupby(["_hour", "_tier"]).size().unstack(fill_value=0)
        .reindex(columns=tier_order, fill_value=0)
        .sort_index()
    )

    fig_c = go.Figure()
    for tier in tier_order:
        if tier not in hour_tier.columns:
            continue
        fig_c.add_trace(go.Bar(
            x=hour_tier.index,
            y=hour_tier[tier],
            name=_ACT_TIER_KO[tier],
            marker_color=_ACT_TIER_COLORS[tier],
            hovertemplate=(
                f"{_ACT_TIER_KO[tier]}<br>"
                "시간: %{x}시<br>"
                "분: %{y}<extra></extra>"
            ),
        ))
    fig_c.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=50, r=20, t=10, b=40)},
        height=260,
        barmode="stack",
        xaxis=dict(title="시간대 (시)", dtick=1,
                   tickfont_color=COLORS["text_muted"], gridcolor="#2A3A4A"),
        yaxis=dict(title="분 수", tickfont_color=COLORS["text_muted"]),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig_c, use_container_width=True, key="corrected_journey_hourly")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 섹션 D: 헬멧 방치 의심 구간 리스트 ────────────────────────
    st.markdown(sub_header("D. 헬멧 방치 의심 구간"), unsafe_allow_html=True)
    if suspect_runs.empty:
        st.success(
            f"✅ 작업공간 + 저활성이 연속 {HELMET_SUSPECT_MIN_RUN}분 이상 지속된 구간이 없습니다. "
            f"정상적으로 작업 중이었을 가능성이 높습니다."
        )
    else:
        st.markdown(
            f"<p style='color:#FFB300; font-size:0.85rem; margin:4px 0 10px 0;'>"
            f"⚠ 총 <b>{len(suspect_runs)}개 구간, {suspect_min}분</b> 감지 — "
            f"전체 체류의 <b>{suspect_pct:.1f}%</b>"
            f"</p>",
            unsafe_allow_html=True,
        )
        # 구간별 카드 리스트
        for i, row in suspect_runs.iterrows():
            s = pd.Timestamp(row["start"]).strftime("%H:%M")
            e = pd.Timestamp(row["end"]).strftime("%H:%M")
            mins = int(row["minutes"])
            ar = float(row["avg_active_ratio"])
            lb = str(row["locus_label"])
            ar_color = (COLORS["danger"] if ar <= 0.20
                        else COLORS["warning"] if ar <= 0.30
                        else COLORS["text_muted"])
            st.markdown(
                f"""
                <div style='background:#2A1A1A; border-left:3px solid #FF4C4C;
                            border-radius:4px; padding:10px 14px; margin-bottom:6px;
                            font-size:0.85rem;'>
                    <b style='color:#FF8C42;'>{i+1}.</b>
                    <b style='color:#D5E5FF; font-family:monospace;'>
                        {s} ~ {e}
                    </b>
                    <span style='color:#FFB300; margin-left:8px;'>({mins}분)</span>
                    &nbsp;·&nbsp;
                    <span style='color:#9AB5D4;'>{lb}</span>
                    &nbsp;·&nbsp;
                    <span style='color:{ar_color};'>avg_active = {ar:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # 다운로드
        with st.expander("📋 의심 구간 테이블 / CSV 다운로드"):
            disp_sr = suspect_runs.copy()
            disp_sr["start"] = pd.to_datetime(disp_sr["start"]).dt.strftime("%H:%M")
            disp_sr["end"]   = pd.to_datetime(disp_sr["end"]).dt.strftime("%H:%M")
            disp_sr["avg_active_ratio"] = disp_sr["avg_active_ratio"].round(3)
            disp_sr = disp_sr.rename(columns={
                "start": "시작", "end": "종료", "minutes": "분",
                "locus_label": "위치", "avg_active_ratio": "평균 활성비율",
            })
            st.dataframe(disp_sr, use_container_width=True, hide_index=True)
            csv = disp_sr.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇ 의심 구간 CSV",
                               csv, "helmet_suspect_runs.csv", "text/csv")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── 섹션 E: Journey Timeline (공간유형 × 활성도) ──────────────
    st.markdown(sub_header("E. Journey Timeline"), unsafe_allow_html=True)
    st.caption(
        "각 분의 색 = 공간유형 × 활성도 조합. "
        "작업공간+저활성(빗금 x)이 연속으로 이어지면 헬멧 방치 후보. "
        "기존 '활성도 / 공간유형' 2가지 모드도 아래 라디오에서 선택 가능."
    )

    view_mode = st.radio(
        "표시 방식",
        ["공간 × 활성도 (통합)", "활성도만", "공간 유형만"],
        index=0,
        horizontal=True,
        key="corrected_journey_view_mode",
        help=(
            "공간 × 활성도 (통합): 공간유형으로 Y축을 쌓고, 색은 활성도로 표시. "
            "작업공간 + 저활성 = 빗금 패턴 (헬멧 방치 후보)."
        ),
    )

    if view_mode == "공간 × 활성도 (통합)":
        _render_space_x_activity_timeline(df, suspect_runs)
    else:
        # 기존 타임라인 — 내부 라디오 키 충돌 방지를 위해 별도 key 사용
        _render_locus_timeline(user_jdf, forced_mode=view_mode)


def _render_space_x_activity_timeline(
    df: pd.DataFrame, suspect_runs: pd.DataFrame,
) -> None:
    """섹션 E용 — 공간 × 활성도 통합 타임라인 (Gantt 스타일).

    Y축: 방문한 locus (공간유형 그룹 내에서 정렬)
    색 : 활성도 티어 (HIGH/MID/LOW/GAP)
    의심 구간(suspect_runs)은 연한 빨간 배경 shape로 강조.
    """
    d = df.copy()
    # Y 라벨
    bldg  = d.get("building", pd.Series(dtype=str)).fillna("").astype(str)
    floor = d.get("floor",    pd.Series(dtype=str)).fillna("").astype(str)
    lname = d["locus_name"].fillna(d["locus_id"]).fillna("UNKNOWN").astype(str)
    loc_part = (bldg + " " + floor).str.strip()
    d["locus_label"] = [
        f"{lp} | {ln}" if lp else ln
        for lp, ln in zip(loc_part, lname)
    ]

    # Y축 순서: 공간유형 그룹 우선 → 첫 방문 시각
    first_ts = d.groupby("locus_label")["timestamp"].min()
    ltype    = d.groupby("locus_label")["locus_type"].agg(
        lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else "UNKNOWN"
    )
    lt_rank = {"WORK_AREA": 0, "TRANSIT": 1, "REST_AREA": 2, "GATE": 3, "UNKNOWN": 4}
    order_df = pd.DataFrame({"first_ts": first_ts, "ltype": ltype})
    order_df["lt_rank"] = order_df["ltype"].map(lt_rank).fillna(9).astype(int)
    order_df = order_df.sort_values(["lt_rank", "first_ts"])
    locus_order = order_df.index.tolist()

    fig = go.Figure()

    # 의심 구간 배경 shape (suspect_runs)
    if not suspect_runs.empty:
        for _, r in suspect_runs.iterrows():
            fig.add_shape(
                type="rect",
                x0=pd.Timestamp(r["start"]), x1=pd.Timestamp(r["end"]) + pd.Timedelta(minutes=1),
                y0=-0.5, y1=len(locus_order) - 0.5,
                fillcolor="rgba(255,76,76,0.10)",
                line=dict(width=0),
                layer="below",
            )

    # 티어별 marker
    tier_symbol = {"HIGH": "square", "MID": "circle", "LOW": "circle", "GAP": "x"}
    for tier in ["HIGH", "MID", "LOW", "GAP"]:
        sub = d[d["_tier"] == tier]
        if sub.empty:
            continue
        hover_ltype = sub.get("locus_type", pd.Series(dtype=str)).fillna("").astype(str)
        fig.add_trace(go.Scatter(
            x=sub["timestamp"],
            y=sub["locus_label"],
            mode="markers",
            name=_ACT_TIER_KO[tier],
            marker=dict(
                color=_ACT_TIER_COLORS[tier],
                size=8,
                symbol=tier_symbol[tier],
                line=dict(width=0.5, color="#0D1B2A"),
            ),
            customdata=np.column_stack([
                sub["signal_count"].fillna(0).astype(int).values,
                sub["active_ratio"].fillna(0).round(2).values,
                hover_ltype.values,
                sub["activity_level"].fillna("").values,
            ]),
            hovertemplate=(
                "<b>%{x|%H:%M}</b><br>"
                "locus: <b>%{y}</b><br>"
                "공간유형: %{customdata[2]}<br>"
                "활성도 티어: <b>" + tier + f" ({_ACT_TIER_KO[tier]})</b><br>"
                "activity_level: %{customdata[3]}<br>"
                "신호: %{customdata[0]}  active_ratio: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=200, r=20, t=30, b=40)},
        height=max(320, 24 * len(locus_order) + 80),
        xaxis=dict(title="시간", tickformat="%H:%M",
                   tickfont_color=COLORS["text_muted"], gridcolor="#2A3A4A"),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=locus_order,
            tickfont=dict(size=9, color=COLORS["text"]),
        ),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True, key="corrected_journey_timeline")

    # 범례/가이드
    st.markdown(
        f"<div style='font-size:0.78rem; color:#7A8FA6; margin-top:4px;"
        f"display:flex; gap:18px; flex-wrap:wrap;'>"
        f"<span><b style='color:{_ACT_TIER_COLORS['HIGH']}'>■</b> 고활성 (≥{ACTIVE_HIGH_THRESHOLD})</span>"
        f"<span><b style='color:{_ACT_TIER_COLORS['MID']}'>●</b> 중간</span>"
        f"<span><b style='color:{_ACT_TIER_COLORS['LOW']}'>●</b> 저활성 (≤{ACTIVE_LOW_THRESHOLD})</span>"
        f"<span><b style='color:{_ACT_TIER_COLORS['GAP']}'>✕</b> 음영 (gap-fill)</span>"
        f"<span style='color:#FF8C82;'>▮ 빨간 배경: 헬멧 방치 의심 구간</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_locus_timeline(user_jdf: pd.DataFrame, forced_mode: str | None = None) -> None:
    """
    1분 단위 Locus 타임라인 (Gantt 스타일).
    - Y축: 방문한 locus (locus_name · building · floor · locus_type)
    - X축: 시간
    - 색 모드: 활성도(activity_level) / 공간 유형(locus_type) 선택
    """
    df = user_jdf.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    has_locus_meta = "locus_type" in df.columns

    # ── 색상 모드 선택 ─────────────────────────────────────────────
    # forced_mode: 상위 호출(보정 Journey 탭 섹션 E)에서 강제 지정 시 라디오 숨김
    if forced_mode is not None:
        use_locus_type = has_locus_meta and ("공간" in forced_mode)
    else:
        color_mode = st.radio(
            "색상 기준",
            ["활성도 (activity_level)", "공간 유형 (locus_type)"] if has_locus_meta else ["활성도 (activity_level)"],
            horizontal=True,
            key="locus_timeline_color_mode",
        )
        use_locus_type = has_locus_meta and "공간 유형" in color_mode

    # ── Y축 라벨 구성 ───────────────────────────────────────────────
    # locus_name · building · floor (locus 메타가 있으면 포함)
    if has_locus_meta:
        bldg  = df.get("building", pd.Series(dtype=str)).fillna("").astype(str)
        floor = df.get("floor",    pd.Series(dtype=str)).fillna("").astype(str)
        ltype = df.get("locus_type", pd.Series(dtype=str)).fillna("").astype(str)
        lname = df["locus_name"].fillna(df["locus_id"]).fillna("UNKNOWN").astype(str)

        # Y 라벨: "건물/층 | locus_name"
        def _make_label(b, fl, ln):
            loc_part = f"{b} {fl}".strip()
            return f"{loc_part} | {ln}" if loc_part else ln

        df["locus_label"] = [
            _make_label(b, f, n)
            for b, f, n in zip(bldg, floor, lname)
        ]
    else:
        df["locus_label"] = df["locus_name"].fillna(df["locus_id"]).fillna("UNKNOWN").astype(str)

    # 방문한 locus 순서 (처음 방문 시간 기준)
    locus_order = (
        df.groupby("locus_label")["timestamp"].min()
        .sort_values().index.tolist()
    )

    fig = go.Figure()

    if use_locus_type:
        # ── 공간 유형 기준 색상 ──────────────────────────────────────
        ltype_col = df.get("locus_type", pd.Series(["UNKNOWN"] * len(df))).fillna("UNKNOWN")
        for lt, ko in {**LOCUS_TYPE_KO, "UNKNOWN": "미분류"}.items():
            color = LOCUS_TYPE_COLOR.get(lt, "#6A7A95")
            sub = df[ltype_col == lt]
            if sub.empty:
                continue
            sym = sub["is_gap_filled"].map({True: "x", False: "circle"}).fillna("circle")
            # hover용 컬럼 준비
            hover_func   = sub.get("function",      pd.Series(dtype=str)).fillna("").astype(str)
            hover_dwcat  = sub.get("dwell_category", pd.Series(dtype=str)).fillna("").astype(str)
            hover_bldg   = sub.get("building",       pd.Series(dtype=str)).fillna("").astype(str)
            hover_floor  = sub.get("floor",          pd.Series(dtype=str)).fillna("").astype(str)
            hover_hazard = sub.get("hazard_level",   pd.Series(dtype=str)).fillna("").astype(str)
            fig.add_trace(go.Scatter(
                x=sub["timestamp"],
                y=sub["locus_label"],
                mode="markers",
                name=f"{ko} ({lt})",
                marker=dict(
                    color=color, size=8,
                    symbol=list(sym),
                    line=dict(width=0.5, color="#0D1B2A"),
                ),
                customdata=np.column_stack([
                    sub["signal_count"].values,
                    sub["active_ratio"].fillna(0).values,
                    sub["gap_confidence"].fillna("none").values,
                    hover_func.values,
                    hover_bldg.values,
                    hover_floor.values,
                    hover_hazard.values,
                    sub["activity_level"].fillna("").values,
                    hover_dwcat.values,
                ]),
                hovertemplate=(
                    "<b>%{x|%H:%M}</b><br>"
                    "locus: <b>%{y}</b><br>"
                    "유형: <b>" + ko + f" ({lt})</b><br>"
                    "건물: %{customdata[4]}  층: %{customdata[5]}<br>"
                    "기능: %{customdata[3]}  체류유형: %{customdata[8]}<br>"
                    "위험도: %{customdata[6]}<br>"
                    "활성도: %{customdata[7]}<br>"
                    "신호: %{customdata[0]}  active_ratio: %{customdata[1]:.2f}<br>"
                    "gap_conf: %{customdata[2]}<extra></extra>"
                ),
            ))
        legend_html = "".join([
            f"<span><b style='color:{c}'>●</b> {LOCUS_TYPE_KO.get(k, k)}</span>"
            for k, c in LOCUS_TYPE_COLOR.items()
        ]) + "<span><b style='color:#6A7A95'>●</b> 미분류</span>"
        legend_html += "&nbsp;&nbsp;<span style='opacity:0.7'><b>✕</b> = Gap-fill 보정</span>"
    else:
        # ── 활성도 기준 색상 (기존 방식) ────────────────────────────
        for level in ["HIGH_ACTIVE", "ACTIVE", "INACTIVE", "DEEP_INACTIVE", "ESTIMATED"]:
            sub = df[df["activity_level"] == level]
            if sub.empty:
                continue
            symbol = "x" if level == "ESTIMATED" else "circle"

            # 공간 메타가 있으면 hover에 추가
            hover_ltype  = sub.get("locus_type",     pd.Series(dtype=str)).fillna("").astype(str)
            hover_bldg   = sub.get("building",        pd.Series(dtype=str)).fillna("").astype(str)
            hover_floor  = sub.get("floor",           pd.Series(dtype=str)).fillna("").astype(str)
            hover_func   = sub.get("function",        pd.Series(dtype=str)).fillna("").astype(str)

            fig.add_trace(go.Scatter(
                x=sub["timestamp"],
                y=sub["locus_label"],
                mode="markers",
                name={
                    "HIGH_ACTIVE":   "고활성",
                    "ACTIVE":        "활성",
                    "INACTIVE":      "비활성",
                    "DEEP_INACTIVE": "정지",
                    "ESTIMATED":     "추정(Gap-fill)",
                }[level],
                marker=dict(
                    color=ACTIVITY_COLORS[level], size=7, symbol=symbol,
                    line=dict(width=0.5, color="#0D1B2A"),
                ),
                customdata=np.column_stack([
                    sub["signal_count"].values,
                    sub["active_ratio"].fillna(0).values,
                    sub["gap_confidence"].fillna("none").values,
                    sub["block_type"].fillna("").values,
                    hover_ltype.values,
                    hover_bldg.values,
                    hover_floor.values,
                    hover_func.values,
                ]),
                hovertemplate=(
                    "<b>%{x|%H:%M}</b><br>"
                    "locus: <b>%{y}</b><br>"
                    "공간유형: %{customdata[4]}  기능: %{customdata[7]}<br>"
                    "건물: %{customdata[5]}  층: %{customdata[6]}<br>"
                    "activity: <b>" + level + "</b><br>"
                    "신호: %{customdata[0]}  active_ratio: %{customdata[1]:.2f}<br>"
                    "gap_conf: %{customdata[2]}  block: %{customdata[3]}<extra></extra>"
                ),
            ))
        legend_html = (
            "<span><b style='color:#00C897'>●</b> 고활성 (BLE 10초)</span>"
            "<span><b style='color:#00AEEF'>●</b> 활성</span>"
            "<span><b style='color:#FFB300'>●</b> 비활성 (BLE 60초)</span>"
            "<span><b style='color:#6A7A95'>●</b> 정지</span>"
            "<span><b style='color:#FF8C42'>✕</b> Gap-fill 추정</span>"
        )

    # 비유효 전이 — 빨간 세로선
    invalid_tr = df[~df["is_valid_transition"]]
    if len(invalid_tr) > 0:
        for ts in invalid_tr["timestamp"]:
            fig.add_vline(x=ts, line_width=1, line_dash="dot",
                          line_color=COLORS["danger"], opacity=0.4)

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=200, r=20, t=40, b=40)},
        height=max(320, 24 * len(locus_order) + 80),
        xaxis=dict(title="시간", tickformat="%H:%M",
                   tickfont_color=COLORS["text_muted"],
                   gridcolor="#2A3A4A"),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=locus_order,
            tickfont=dict(size=9, color=COLORS["text"]),
        ),
        legend={**PLOTLY_LEGEND, "orientation": "h",
                "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True, key="plotly_9")

    st.markdown(
        f"<div style='font-size:0.78rem; color:#7A8FA6; margin-top:4px;"
        f"display:flex; gap:16px; flex-wrap:wrap;'>"
        f"{legend_html}"
        f"&nbsp;&nbsp;<span><b style='color:#FF4C4C'>┊</b> 비유효 전이</span></div>",
        unsafe_allow_html=True,
    )

    # ── locus 분포 요약 (locus_type 조인 시) ───────────────────────
    if has_locus_meta and "locus_type" in df.columns:
        with st.expander("📊 공간 유형별 체류 시간 분포"):
            type_min = (
                df.groupby("locus_type")["timestamp"]
                .count()
                .rename("체류(분)")
                .reset_index()
                .rename(columns={"locus_type": "공간 유형"})
            )
            type_min["공간 유형(한글)"] = type_min["공간 유형"].map(LOCUS_TYPE_KO).fillna("미분류")
            type_min = type_min.sort_values("체류(분)", ascending=False)

            col_t, col_b = st.columns([1, 2])
            with col_t:
                st.dataframe(type_min[["공간 유형(한글)", "공간 유형", "체류(분)"]],
                             use_container_width=True, hide_index=True)
            with col_b:
                fig_bar = go.Figure(go.Bar(
                    y=type_min["공간 유형(한글)"],
                    x=type_min["체류(분)"],
                    orientation="h",
                    marker_color=[LOCUS_TYPE_COLOR.get(lt, "#6A7A95")
                                  for lt in type_min["공간 유형"]],
                    text=type_min["체류(분)"].astype(str) + "분",
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    **{**PLOTLY_DARK, "margin": dict(l=80, r=60, t=10, b=20)},
                    height=160, showlegend=False,
                    xaxis=dict(title="체류 시간(분)"),
                    yaxis=dict(title=""),
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="plotly_10")


def _render_signal_quality(user_jdf: pd.DataFrame) -> None:
    """신호 품질 시계열 — signal_count, active_ratio, gap_confidence."""
    df = user_jdf.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("BLE 신호 갯수 (signal_count)",
                        "활성 비율 (active_ratio)",
                        "Gap-fill 신뢰도"),
    )

    # signal_count
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["signal_count"],
        mode="lines", fill="tozeroy",
        line=dict(color=COLORS["accent"], width=1.5),
        name="signal_count",
        hovertemplate="%{x|%H:%M}<br>signals: %{y}<extra></extra>",
    ), row=1, col=1)
    # 0-signal 행 강조
    zero_df = df[df["signal_count"] == 0]
    fig.add_trace(go.Scatter(
        x=zero_df["timestamp"], y=zero_df["signal_count"],
        mode="markers",
        marker=dict(color=COLORS["danger"], size=5, symbol="triangle-down"),
        name="0-signal (음영)",
        showlegend=False,
        hovertemplate="%{x|%H:%M}<br><b>0-signal</b><extra></extra>",
    ), row=1, col=1)

    # active_ratio (0~1)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["active_ratio"],
        mode="lines",
        line=dict(color=COLORS["success"], width=1.5),
        name="active_ratio",
        hovertemplate="%{x|%H:%M}<br>active: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    # 0.9 기준선 (HIGH_ACTIVE_THRESHOLD)
    fig.add_hline(y=0.9, line_dash="dot", line_color=COLORS["success"],
                  annotation_text="H=0.90", annotation_position="right",
                  annotation_font_size=10, row=2, col=1)
    fig.add_hline(y=0.4, line_dash="dot", line_color=COLORS["warning"],
                  annotation_text="L=0.40", annotation_position="right",
                  annotation_font_size=10, row=2, col=1)

    # gap_confidence — 스택바
    gap_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
    df["gap_level"] = df["gap_confidence"].map(gap_map).fillna(0)
    df["gap_color"] = df["gap_confidence"].map(GAP_COLORS).fillna("#1A2A3A")

    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["gap_level"],
        marker=dict(color=df["gap_color"]),
        name="gap_confidence",
        hovertemplate="%{x|%H:%M}<br>gap: %{customdata}<extra></extra>",
        customdata=df["gap_confidence"],
        showlegend=False,
    ), row=3, col=1)

    fig.update_layout(
        **{**PLOTLY_DARK, "margin": dict(l=60, r=20, t=40, b=30)},
        height=520,
        xaxis3=dict(title="시간", tickformat="%H:%M"),
        yaxis=dict(title="갯수"),
        yaxis2=dict(title="비율", range=[0, 1.1]),
        yaxis3=dict(
            title="gap 수준",
            tickvals=[0, 1, 2, 3],
            ticktext=["none", "low", "medium", "high"],
            range=[0, 3.5],
        ),
        hovermode="x unified",
        showlegend=False,
    )
    # subplot 제목 색상
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color=COLORS["text"], size=11)

    st.plotly_chart(fig, use_container_width=True, key="plotly_11")


def _render_raw_table(user_jdf: pd.DataFrame) -> None:
    """Raw 데이터 테이블 — 의심 행 하이라이트."""
    df = user_jdf.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")
    df["time_idx"]  = range(len(df))

    disp_cols = [
        "time_idx", "timestamp", "locus_id", "locus_name",
        "signal_count", "active_ratio", "activity_level",
        "is_gap_filled", "gap_confidence", "is_low_confidence",
        "is_transition", "is_valid_transition",
        "block_id", "block_type",
    ]
    disp_cols = [c for c in disp_cols if c in df.columns]

    # 필터 옵션
    col_filter, col_dl = st.columns([3, 1])
    with col_filter:
        show_only = st.radio(
            "표시 필터",
            ["전체", "보정 행만 (gap_filled)",
             "저신뢰 행만", "비유효 전이만", "0-signal만"],
            horizontal=True,
            key="integrity_raw_filter",
        )

    if show_only == "보정 행만 (gap_filled)":
        df = df[df["is_gap_filled"]]
    elif show_only == "저신뢰 행만":
        df = df[df["is_low_confidence"]]
    elif show_only == "비유효 전이만":
        df = df[~df["is_valid_transition"]]
    elif show_only == "0-signal만":
        df = df[df["signal_count"] == 0]

    st.caption(f"표시: {len(df):,} / 전체: {len(user_jdf):,}")

    if df.empty:
        st.info("해당 필터에 해당하는 행이 없습니다.")
        return

    # 스타일링 (pandas Styler)
    def _row_color(row):
        if row.get("is_gap_filled"):
            return ["background-color:#2a2015; color:#FFB300"] * len(row)
        if not row.get("is_valid_transition", True):
            return ["background-color:#2a1515; color:#FF4C4C"] * len(row)
        if row.get("is_low_confidence"):
            return ["background-color:#20282a; color:#9AB5D4"] * len(row)
        return [""] * len(row)

    try:
        styler = df[disp_cols].head(500).style.apply(_row_color, axis=1)
        st.dataframe(styler, use_container_width=True, height=420)
    except Exception:
        st.dataframe(df[disp_cols].head(500), use_container_width=True, height=420)

    with col_dl:
        csv = df[disp_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ CSV 다운로드",
            data=csv,
            file_name=f"integrity_user_{df['time_idx'].iloc[0] if len(df) else 0}.csv",
            mime="text/csv",
        )


def _sort_workers_by_filter(
    sector_id: str, date_str: str, journey_path: str,
    worker_df: pd.DataFrame, filter_mode: str,
) -> pd.DataFrame:
    """필터 모드에 따라 작업자 정렬 + 상위 200명 리턴."""
    if worker_df.empty:
        return pd.DataFrame()

    base = worker_df.copy()
    base["label_metric"] = ""

    if filter_mode == "음영 상위":
        base = base.sort_values("gap_ratio", ascending=False, na_position="last")
        base["label_metric"] = base["gap_ratio"].apply(
            lambda v: f"음영 {v*100:.0f}%" if pd.notna(v) else "음영 ?"
        )
    elif filter_mode == "보정 상위":
        # 보정율 계산을 위해 journey 필요 — 캐시된 통계 활용
        stats = _compute_worker_gap_stats(sector_id, date_str, journey_path)
        if not stats.empty:
            base = base.merge(stats, on="user_no", how="left")
            base["gap_filled_pct"] = base["gap_filled_pct"].fillna(0)
            base = base.sort_values("gap_filled_pct", ascending=False)
            base["label_metric"] = base["gap_filled_pct"].apply(
                lambda v: f"보정 {v:.0f}%"
            )
    elif filter_mode == "비유효 전이 상위":
        stats = _compute_worker_gap_stats(sector_id, date_str, journey_path)
        if not stats.empty:
            base = base.merge(stats, on="user_no", how="left")
            base["invalid_tr_count"] = base["invalid_tr_count"].fillna(0)
            base = base.sort_values("invalid_tr_count", ascending=False)
            base["label_metric"] = base["invalid_tr_count"].apply(
                lambda v: f"비유효 {int(v)}건"
            )
    else:
        # 전체
        base = base.sort_values("work_minutes", ascending=False, na_position="last")
        base["label_metric"] = base["work_minutes"].apply(
            lambda v: f"{v:.0f}분" if pd.notna(v) else "?"
        )

    base = base.head(200).reset_index(drop=True)
    # NaN company_name 처리
    base["company_name"] = base["company_name"].fillna("미확인").astype(str)
    return base


@st.cache_data(show_spinner=False, ttl=600)
def _compute_worker_gap_stats(sector_id: str, date_str: str, journey_path: str) -> pd.DataFrame:
    """작업자별 보정률/비유효 전이 집계."""
    try:
        df = pd.read_parquet(
            journey_path,
            columns=["user_no", "is_gap_filled", "is_valid_transition"],
        )
        # nullable boolean → bool
        df["is_gap_filled"]       = df["is_gap_filled"].fillna(False).astype(bool)
        df["is_valid_transition"] = df["is_valid_transition"].fillna(True).astype(bool)

        agg = df.groupby("user_no").agg(
            n_total=("is_gap_filled", "count"),
            n_gap=("is_gap_filled", "sum"),
            n_invalid=("is_valid_transition",
                       lambda s: int((~s.astype(bool)).sum())),
        ).reset_index()
        agg["gap_filled_pct"] = agg["n_gap"] / agg["n_total"] * 100
        agg["invalid_tr_count"] = agg["n_invalid"]
        return agg[["user_no", "gap_filled_pct", "invalid_tr_count"]]
    except Exception:
        return pd.DataFrame(columns=["user_no", "gap_filled_pct", "invalid_tr_count"])
