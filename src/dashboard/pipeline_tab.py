"""
Pipeline Tab — 데이터 파이프라인 관리 탭
=========================================
동작 흐름:
  1. 탭 진입 시 raw 폴더 자동 스캔
  2. 미처리 날짜 감지 → 목록 표시
  3. 버튼 클릭 → 전처리 실행 (진행 상황 실시간 표시)
  4. 완료 후 자동 rerun → 업데이트된 상태 표시
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import config as cfg
from src.dashboard.styles import badge, metric_card, section_header, COLORS
from src.pipeline.cache_manager import (
    get_cache_status,
    load_meta_only,
    save_daily_results,
)

# ── CLOUD_MODE 비의존 임포트 ─────────────────────────────────────────
# loader / processor / sward_mapper 는 파이프라인 IP 포함 모듈이므로
# CLOUD_MODE=True (Streamlit Cloud 배포) 환경에서는 임포트하지 않는다.
# raw CSV 없음 → 재처리 불필요 — Drive Sync 전용 UI로 대체.
try:
    import config as _cfg_check
    _CLOUD_MODE = getattr(_cfg_check, "CLOUD_MODE", False)
except ImportError:
    _CLOUD_MODE = True

if not _CLOUD_MODE:
    from src.pipeline.loader import load_daily_data        # noqa: F401
    from src.pipeline.processor import process_daily       # noqa: F401

from src.utils.weather import date_label


def render_pipeline_tab(sector_id: str | None = None):
    """파이프라인 탭 전체 렌더링."""
    sid      = sector_id or cfg.SECTOR_ID
    paths    = cfg.get_sector_paths(sid)
    raw_dir  = paths["raw_dir"]

    st.markdown(section_header("📡 데이터 파이프라인"), unsafe_allow_html=True)

    # ── CLOUD_MODE: 재처리 UI 비활성화 ──────────────────────────────
    if cfg.CLOUD_MODE:
        st.info(
            "☁️ **클라우드 모드** — 이 환경에서는 Raw 데이터 파일이 없으므로 재처리를 실행할 수 없습니다.  \n"
            "처리된 데이터는 Google Drive에서 자동으로 로드됩니다.  \n"
            "데이터 업데이트가 필요하면 로컬 환경에서 파이프라인 실행 후 Drive에 업로드하세요."
        )
        # Drive 동기화 섹션만 표시
        st.divider()
        render_drive_section(sid)
        return

    # ── 스캔 안내 + 재스캔 버튼 ──────────────────────────────────────
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        st.caption(
            f"📁 원본 폴더: `{raw_dir}`  |  "
            f"탭 접속 시 자동으로 미처리 데이터를 감지합니다."
        )
    with col_btn:
        if st.button("🔄 재스캔", key="rescan_btn", help="폴더를 다시 스캔합니다"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # ── 폴더 스캔 ─────────────────────────────────────────────────
    with st.spinner("데이터 폴더 스캔 중..."):
        status = get_cache_status(str(raw_dir), sid)

    # ── KPI 카드 ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            metric_card("원본 데이터", f"{status['total_raw']}일", "📁"),
            unsafe_allow_html=True,
        )
    with col2:
        # 처리완료 = raw 기준 (raw 없는 고아 캐시 제외)
        st.markdown(
            metric_card("처리 완료", f"{status['total_processed']}일", "✅"),
            unsafe_allow_html=True,
        )
    with col3:
        n_unproc = status["total_unprocessed"]
        val_html = (
            f"<div class='metric-value' style='color:#FFB300'>{n_unproc}일</div>"
            if n_unproc > 0
            else f"<div class='metric-value' style='color:#00C897'>0일</div>"
        )
        st.markdown(
            f"<div class='metric-card'>{val_html}"
            f"<div class='metric-label'>미처리 대기</div></div>",
            unsafe_allow_html=True,
        )

    # ── 고아 캐시 경고 ────────────────────────────────────────────
    if status["total_orphaned"] > 0:
        orphaned_str = ", ".join(status["orphaned_dates"])
        st.warning(
            f"⚠️ 원본 파일이 삭제된 캐시 {status['total_orphaned']}일 존재: **{orphaned_str}**  \n"
            f"해당 날짜는 분석 탭에서는 열람 가능하나, 재처리는 원본 파일 필요."
        )

    st.divider()

    # ── 미처리 날짜 목록 + 실행 버튼 ─────────────────────────────
    unprocessed = status["unprocessed_dates"]

    if unprocessed:
        st.markdown(
            section_header(f"⏳ 미처리 데이터 — {len(unprocessed)}일 감지됨"),
            unsafe_allow_html=True,
        )

        # 미처리 날짜 카드
        cols = st.columns(min(len(unprocessed), 4))
        for i, d in enumerate(unprocessed):
            date_fmt = date_label(d)
            with cols[i % 4]:
                st.markdown(
                    f"<div style='background:#1A2A3A; border:1px solid #FFB300; "
                    f"border-left:4px solid #FFB300; border-radius:10px; "
                    f"padding:14px 16px; text-align:center; margin-bottom:8px;'>"
                    f"<div style='font-size:1rem; font-weight:700; color:{COLORS['text_bright']}'>"
                    f"{date_fmt}</div>"
                    f"<div style='margin-top:6px'>{badge('처리 대기', 'warning')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 단일 실행 버튼 ────────────────────────────────────────
        st.markdown(
            f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; "
            f"border-radius:12px; padding:20px 24px; margin-bottom:16px;'>"
            f"<div style='color:{COLORS['text_bright']}; font-size:0.95rem; font-weight:600;'>"
            f"🚀 전처리 준비 완료</div>"
            f"<div style='color:#7A8FA6; font-size:0.83rem; margin-top:6px;'>"
            f"버튼을 클릭하면 {len(unprocessed)}일치 데이터를 순차 처리합니다.<br>"
            f"처리 시간: 약 {len(unprocessed) * 55}초 예상 (일당 ~55초, Journey 보정 포함)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            f"▶  미처리 {len(unprocessed)}일 전처리 실행",
            use_container_width=True,
            type="primary",
            key="run_pipeline_btn",
        ):
            success = _run_pipeline(unprocessed, raw_dir, sid)
            if success:
                st.rerun()   # 완료 후 자동 rerun → 업데이트된 상태 표시

    else:
        # 모두 처리된 상태
        st.markdown(
            "<div style='background:#0D2A1A; border:1px solid #00C897; "
            "border-radius:12px; padding:24px; text-align:center;'>"
            "<div style='font-size:1.5rem; margin-bottom:8px;'>✅</div>"
            "<div style='color:#00C897; font-size:1rem; font-weight:600;'>"
            "모든 원본 데이터가 처리되었습니다</div>"
            "<div style='color:#7A8FA6; font-size:0.83rem; margin-top:6px;'>"
            "새 raw 데이터 추가 후 [🔄 재스캔] 버튼을 클릭하세요.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── 재처리 섹션 ─────────────────────────────────────────────
    processed = status["processed_dates"]
    if processed:
        with st.expander("🔄 데이터 재처리 (처리 로직 변경 시)", expanded=False):
            st.caption(
                "전처리 로직이 변경된 경우, 기존 데이터를 재처리합니다. "
                "기존 캐시를 삭제하고 다시 처리합니다."
            )
            reprocess_mode = st.radio(
                "재처리 범위",
                ["전체 재처리", "날짜 선택"],
                horizontal=True,
                key="reprocess_mode",
            )
            if reprocess_mode == "전체 재처리":
                reprocess_dates = processed
            else:
                reprocess_dates = st.multiselect(
                    "재처리할 날짜",
                    processed,
                    format_func=date_label,
                    key="reprocess_dates",
                )

            if reprocess_dates and st.button(
                f"🔄 {len(reprocess_dates)}일 재처리 실행",
                type="secondary",
                key="reprocess_btn",
            ):
                # 기존 캐시 삭제 후 재처리
                import shutil
                proc_dir = cfg.PROCESSED_DIR / sid
                for d in reprocess_dates:
                    target = proc_dir / d
                    if target.exists():
                        shutil.rmtree(target)
                st.cache_data.clear()
                success = _run_pipeline(reprocess_dates, raw_dir, sid)
                if success:
                    st.rerun()

    # ── 처리 완료 현황 테이블 ────────────────────────────────────
    if processed:
        st.divider()
        st.markdown(
            section_header(f"✅ 처리 완료 — {len(processed)}일"),
            unsafe_allow_html=True,
        )
        _render_processed_summary(processed, sid)

        # ── Google Drive 업로드 섹션 ───────────────────────────────
        st.divider()
        _render_drive_upload_section(processed, sid)

        # ── Journey 임베딩 모델 학습 섹션 ────────────────────────
        st.divider()
        render_embedding_section(sid)

        # ── Deep Space Foundation Model 학습 섹션 ────────────────
        st.divider()
        render_deep_space_section(sid)


# ─── 파이프라인 실행 ──────────────────────────────────────────────
def _run_pipeline(
    date_list: list[str],
    raw_dir: Path,
    sector_id: str | None = None,
) -> bool:
    """
    미처리 날짜들 순차 전처리.

    반환:
        True  — 전체 성공
        False — 1건 이상 실패
    """
    import config as cfg
    spot_map = load_spot_name_map(sector_id)

    # v2: GatewayIndex 초기화
    gateway_index = None
    if cfg.LOCUS_VERSION == "v2":
        try:
            from src.pipeline.sward_mapper import GatewayIndex
            gw_csv = cfg.get_sector_paths(sector_id).get("gateway_csv")
            if gw_csv and Path(gw_csv).exists():
                gateway_index = GatewayIndex.from_csv(gw_csv)
                print(f"  ✓ GatewayIndex 초기화: {len(gateway_index._gw_meta)}개 Gateway")
        except Exception as e:
            print(f"  ⚠ GatewayIndex 초기화 실패, v1 fallback: {e}")

    total    = len(date_list)
    errors   = []

    # ── UI 컨테이너 ──────────────────────────────────────────────
    progress_bar = st.progress(0.0, text="전처리 준비 중...")
    status_box   = st.empty()   # 현재 단계 표시
    log_box      = st.empty()   # 실시간 로그
    logs: list[str] = []

    def _log(msg: str):
        logs.append(msg)
        log_box.code("\n".join(logs[-14:]), language=None)

    for idx, date_str in enumerate(date_list):
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        frac     = idx / total

        try:
            # 1. 데이터 로드
            progress_bar.progress(frac, text=f"[{idx+1}/{total}]  {date_fmt}  — 데이터 로딩 중...")
            status_box.markdown(
                f"<div style='background:#0D1B2A; border-left:4px solid #00AEEF; "
                f"padding:10px 16px; border-radius:6px; color:{COLORS['text_bright']}; font-size:0.88rem;'>"
                f"📥 <b>{date_fmt}</b> — AccessLog + TWardData 로딩 중...</div>",
                unsafe_allow_html=True,
            )
            _log(f"📥 [{date_fmt}] 로딩 시작")

            journey_df, access_df, meta = load_daily_data(raw_dir, date_str)
            _log(
                f"    ✓ AccessLog {meta['total_records']:,}건 · "
                f"T-Ward작업자 {meta['total_workers_move']:,}명 · "
                f"출입자 {meta['total_workers_access']:,}명"
            )

            # 2. 전처리
            progress_bar.progress(frac + 0.3 / total, text=f"[{idx+1}/{total}]  {date_fmt}  — 전처리 중...")
            status_box.markdown(
                f"<div style='background:#0D1B2A; border-left:4px solid #FFB300; "
                f"padding:10px 16px; border-radius:6px; color:{COLORS['text_bright']}; font-size:0.88rem;'>"
                f"⚙️ <b>{date_fmt}</b> — Locus 매핑 · Journey 구성 · 지표 계산 중...</div>",
                unsafe_allow_html=True,
            )
            _log(f"⚙️  [{date_fmt}] 전처리 시작")

            step_logs: list[str] = []

            def _step(pct: int, msg: str):
                step_logs.append(f"    [{pct:3d}%] {msg}")
                log_box.code(
                    "\n".join(logs + step_logs[-6:]), language=None
                )

            results = process_daily(journey_df, access_df, spot_map, date_str, _step, gateway_index=gateway_index)

            # 3. stats를 meta에 병합 (journey 보정 통계)
            meta.update(results.pop("stats", {}))

            # 4. 저장
            save_daily_results(results, meta, date_str, sector_id)
            _log(
                f"💾 [{date_fmt}] 저장 완료 → "
                f"작업자 {len(results['worker']):,}명 / "
                f"공간 {len(results['space'])}곳 / "
                f"업체 {len(results['company'])}개"
            )

        except Exception as e:
            errors.append(date_fmt)
            _log(f"❌ [{date_fmt}] 오류: {e}")
            st.error(f"**{date_fmt}** 처리 실패: {e}")

    # ── 완료 표시 ────────────────────────────────────────────────
    progress_bar.progress(1.0, text="전처리 완료!")
    status_box.empty()

    if errors:
        st.warning(f"⚠️ {len(errors)}일 처리 실패: {', '.join(errors)}")
        return False
    else:
        st.success(
            f"✅ {total}일 전처리 완료! 잠시 후 결과가 업데이트됩니다...",
            icon="🎉",
        )
        return True


# ─── 처리 완료 요약 테이블 ────────────────────────────────────────
def _render_processed_summary(
    processed: list[str],
    sector_id: str | None = None,
):
    """처리 완료 날짜별 요약 표시."""
    rows = []
    for date_str in reversed(processed):   # 최신 날짜 먼저
        try:
            meta = load_meta_only(date_str, sector_id) or {}

            # Journey 보정 상태
            corrected      = meta.get("journey_corrected", False)
            c_records      = meta.get("corrected_records", 0)
            c_ratio        = meta.get("correction_ratio", 0.0)
            w_corrected    = meta.get("workers_corrected", None)
            unmapped       = meta.get("unmapped_ratio", None)

            if corrected:
                w_str = f" / {w_corrected}명" if w_corrected is not None else ""
                correction_str = f"✅ {c_records:,}건 ({c_ratio:.1f}%){w_str}"
            else:
                correction_str = "미적용"
            unmapped_str = f"{unmapped:.1f}%" if unmapped is not None else "—"

            # ★ 데이터 검증 결과 (신규)
            validation = meta.get("validation", {})
            if validation and not validation.get("error"):
                level = validation.get("overall_level", "—")
                score = validation.get("overall_score", 0)
                level_icons = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
                validation_str = f"{level_icons.get(level, '❓')} {score:.2f}"
            else:
                validation_str = "—"

            # ★ 토큰화 상태 (신규)
            tokenization = meta.get("tokenization_enabled", False)
            token_stats = meta.get("tokenization_stats", {})
            if tokenization and not token_stats.get("error"):
                block_dist = token_stats.get("block_type_distribution", {})
                total_blocks = sum(block_dist.values()) if block_dist else 0
                tokenization_str = f"✅ {total_blocks:,}건"
            else:
                tokenization_str = "미적용"

            rows.append({
                "날짜":         date_label(date_str),
                "T-Ward 착용자":  f"{meta.get('total_workers_move', 0):,}명",
                "총 출입자":    f"{meta.get('total_workers_access', 0):,}명",
                "참여 업체":    f"{meta.get('companies', 0)}개",
                "레코드":       f"{meta.get('total_records', 0):,}건",
                "Journey 보정": correction_str,
                "토큰화":       tokenization_str,
                "검증":         validation_str,
                "Unmapped":     unmapped_str,
                "처리 일시":    meta.get("processed_at", "")[:16].replace("T", " "),
            })
        except Exception:
            rows.append({"날짜": date_str, "이동 작업자": "오류"})

    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Google Drive 업로드 섹션 ──────────────────────────────────────
def _render_drive_upload_section(
    processed_dates: list[str],
    sector_id: str | None = None,
):
    """
    처리 완료된 데이터를 Google Drive에 업로드하는 UI.
    drive_config.json + token.json 설정이 되어 있어야 활성화.
    """
    from src.pipeline.drive_uploader import is_drive_configured

    if not is_drive_configured():
        st.markdown(
            "<div style='background:#1A2A3A; border:1px solid #3A4A5A; "
            "border-radius:10px; padding:16px 20px;'>"
            "<div style='color:#7A8FA6; font-size:0.88rem;'>"
            "☁️ <b>Google Drive 배포</b> — 미설정</div>"
            "<div style='color:#5A6A7A; font-size:0.78rem; margin-top:6px;'>"
            "Streamlit Cloud 배포를 위해 Google Drive 연동이 필요합니다.<br>"
            "① <code>drive_config.json</code> 생성 (folder_id 설정)<br>"
            "② 터미널에서 <code>python upload_to_drive.py</code> 실행 (최초 인증)"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    sid = sector_id or cfg.SECTOR_ID

    st.markdown(
        section_header("☁️ Google Drive 배포"),
        unsafe_allow_html=True,
    )

    # 날짜 선택 (전체 or 선택)
    col_opt, col_info = st.columns([3, 2])
    with col_opt:
        upload_mode = st.radio(
            "업로드 범위",
            ["전체 (처리 완료 데이터 모두)", "날짜 선택"],
            horizontal=True,
            key="drive_upload_mode",
            label_visibility="collapsed",
        )

    if upload_mode == "날짜 선택":
        selected_dates = st.multiselect(
            "업로드할 날짜 선택",
            options=processed_dates,
            default=processed_dates[-3:] if len(processed_dates) >= 3 else processed_dates,
            format_func=date_label,
            key="drive_upload_dates",
        )
    else:
        selected_dates = processed_dates

    if not selected_dates:
        st.caption("업로드할 날짜를 선택해 주세요.")
        return

    # ── Drive 상태 사전 검사 ────────────────────────────────────
    try:
        from src.pipeline.drive_uploader import check_upload_status
    except ImportError as e:
        st.info(f"☁️ Google Drive 모듈 미설치 ({e}). 로컬 환경에서는 Drive 배포가 불필요합니다.")
        return

    with st.spinner("☁️ Google Drive 동기화 상태 확인 중..."):
        try:
            status = check_upload_status(selected_dates, sid)
        except ImportError as e:
            st.info(f"☁️ Google Drive 라이브러리 미설치: {e}")
            return
        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            return
        except Exception as e:
            st.error(f"❌ Drive 연결 실패: {e}")
            return

    n_new     = len(status["new"])
    n_changed = len(status["changed"])
    n_synced  = len(status["synced"])
    n_upload  = n_new + n_changed
    size_mb   = status["total_new_size"] / 1024 / 1024

    # 상태 카드
    c1, c2, c3 = st.columns(3)
    with c1:
        color = "#FFB300" if n_new > 0 else "#7A8FA6"
        st.markdown(
            f"<div style='text-align:center; padding:8px;'>"
            f"<div style='font-size:1.3rem; font-weight:700; color:{color};'>{n_new}</div>"
            f"<div style='font-size:0.78rem; color:#7A8FA6;'>신규 파일</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        color = "#FF4C4C" if n_changed > 0 else "#7A8FA6"
        st.markdown(
            f"<div style='text-align:center; padding:8px;'>"
            f"<div style='font-size:1.3rem; font-weight:700; color:{color};'>{n_changed}</div>"
            f"<div style='font-size:0.78rem; color:#7A8FA6;'>변경됨</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='text-align:center; padding:8px;'>"
            f"<div style='font-size:1.3rem; font-weight:700; color:#00C897;'>{n_synced}</div>"
            f"<div style='font-size:0.78rem; color:#7A8FA6;'>동기화 완료</div></div>",
            unsafe_allow_html=True,
        )

    if n_upload == 0:
        st.markdown(
            "<div style='background:#0D2A1A; border:1px solid #00C897; "
            "border-radius:10px; padding:16px; text-align:center; margin-top:8px;'>"
            "<div style='color:#00C897; font-size:0.92rem; font-weight:600;'>"
            "✅ Google Drive와 완전히 동기화되어 있습니다</div>"
            "<div style='color:#7A8FA6; font-size:0.78rem; margin-top:4px;'>"
            "업로드할 새 파일이 없습니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div style='background:#0D1B2A; border:1px solid #1A3A5C; "
        f"border-radius:10px; padding:14px 18px; margin:8px 0;'>"
        f"<div style='color:{COLORS['text_bright']}; font-size:0.88rem;'>"
        f"☁️ <b>{n_upload}개</b> 파일 업로드 예정 ({size_mb:.1f} MB)</div>"
        f"<div style='color:#7A8FA6; font-size:0.78rem; margin-top:4px;'>"
        f"{'신규 ' + str(n_new) + '개' if n_new else ''}"
        f"{' · ' if n_new and n_changed else ''}"
        f"{'변경 ' + str(n_changed) + '개' if n_changed else ''}"
        f" — 이미 동기화된 {n_synced}개는 스킵합니다."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    force_upload = st.checkbox(
        "🔄 강제 업로드 (재처리 후 전체 덮어쓰기)",
        key="force_upload_chk",
    )

    btn_label = (
        f"☁️  전체 {n_upload + n_synced}개 파일 강제 업로드"
        if force_upload
        else f"☁️  {n_upload}개 파일 Google Drive에 업로드"
    )

    if st.button(
        btn_label,
        use_container_width=True,
        type="primary",
        key="drive_upload_btn",
    ):
        upload_dates = selected_dates if force_upload else selected_dates
        _execute_drive_upload(upload_dates, sid, force=force_upload)


def _execute_drive_upload(date_list: list[str], sector_id: str, force: bool = False):
    """Google Drive 업로드 실행 (프로그래스 바 포함)."""
    from src.pipeline.drive_uploader import upload_processed_dates

    progress_bar = st.progress(0.0, text="Google Drive 업로드 준비 중...")
    log_box = st.empty()

    def _progress(current, total, message):
        frac = current / total if total > 0 else 0.0
        progress_bar.progress(frac, text=message)

    try:
        result = upload_processed_dates(
            date_list, sector_id, progress_callback=_progress, force=force
        )

        progress_bar.progress(1.0, text="업로드 완료!")

        # 결과 표시
        if result["errors"] == 0:
            st.success(
                f"✅ Google Drive 업로드 완료! "
                f"({result['uploaded']}개 업로드 / {result['skipped']}개 스킵)",
                icon="☁️",
            )
        else:
            st.warning(
                f"⚠️ {result['uploaded']}개 업로드 / "
                f"{result['skipped']}개 스킵 / {result['errors']}개 실패"
            )

        # 상세 로그
        if result["details"]:
            with st.expander("📋 업로드 상세 로그", expanded=False):
                st.code("\n".join(result["details"]), language=None)

    except RuntimeError as e:
        progress_bar.empty()
        st.error(f"⚠️ {e}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Google Drive 업로드 실패: {e}")


# ── Journey 임베딩 섹션 ──────────────────────────────────────────────────
def render_embedding_section(sector_id: str):
    """
    파이프라인 탭 내 Journey 임베딩 + 클러스터링 모델 학습 섹션.

    ★ Dev 환경 전용 (journey.parquet 필요).
    """
    from src.intelligence.journey_embedding import JourneyEmbedder, run_embedding_pipeline
    from src.pipeline.cache_manager import detect_processed_dates

    st.markdown(
        section_header("🧬 Journey 임베딩 모델"),
        unsafe_allow_html=True,
    )
    st.caption(
        "작업자 이동 시퀀스(Locus 토큰)를 Word2Vec으로 임베딩하고 "
        "K-means로 Journey 패턴을 자동 분류합니다."
    )

    emb        = JourneyEmbedder(sector_id)
    is_trained = emb.is_available()
    processed  = detect_processed_dates(sector_id)

    col_status, col_train = st.columns([3, 1])
    with col_status:
        if is_trained:
            # 메타 로드
            meta_path = emb.model_dir / "embedder_meta.json"
            meta_info = ""
            if meta_path.exists():
                import json as _json
                with open(meta_path, encoding="utf-8") as f:
                    m = _json.load(f)
                meta_info = (
                    f"&nbsp;·&nbsp; vocab {m.get('vocab_size',0)}개 Locus "
                    f"&nbsp;·&nbsp; {m.get('n_clusters',0)}개 클러스터"
                )
            st.markdown(
                f"<span style='color:#00C897'>✅ 모델 존재{meta_info}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:#FFB300'>⚠️ 모델 없음 — 학습이 필요합니다</span>",
                unsafe_allow_html=True,
            )

    with col_train:
        n_clusters = st.number_input(
            "클러스터 수", min_value=2, max_value=10, value=5, step=1,
            key="emb_n_clusters",
        )

    if not processed:
        st.info("처리 완료된 날짜가 없습니다.")
        return

    if st.button(
        "🚀 임베딩 모델 학습" if not is_trained else "🔄 모델 재학습",
        key="train_embedding_btn",
        type="primary" if not is_trained else "secondary",
    ):
        with st.spinner(f"Journey 임베딩 학습 중... ({len(processed)}일 데이터)"):
            try:
                run_embedding_pipeline(sector_id, processed, n_clusters=n_clusters)
                st.success(
                    f"✅ 임베딩 모델 학습 완료! "
                    f"({len(processed)}일 / {n_clusters}개 클러스터)",
                    icon="🧬",
                )
                # ── 모델 자동 Drive 업로드 ──
                _auto_upload_model(sector_id)
                st.rerun()
            except ImportError as e:
                st.error(f"⚠️ 의존성 없음: {e}\n\npip install gensim scikit-learn")
            except Exception as e:
                st.error(f"❌ 학습 실패: {e}")


def _auto_upload_model(sector_id: str):
    """모델 학습 후 자동으로 Google Drive에 업로드 (설정된 경우만)."""
    try:
        from src.pipeline.drive_uploader import is_drive_configured, upload_model_files
        if not is_drive_configured():
            return
        result = upload_model_files(sector_id)
        if result["uploaded"] > 0:
            st.toast(f"☁️ 모델 {result['uploaded']}개 파일 Drive에 업로드됨", icon="✅")
        elif result["skipped"] > 0:
            st.toast("☁️ 모델 이미 Drive 동기화 완료", icon="✅")
    except Exception as e:
        st.toast(f"⚠️ 모델 Drive 업로드 실패: {e}", icon="⚠️")


# ── Deep Space Transformer 학습 섹션 ──────────────────────────────────
def render_deep_space_section(sector_id: str):
    """
    Deep Space Foundation Model 학습 섹션.

    Transformer 기반 Journey MLM 사전학습.
    ★ Dev 환경 전용 (GPU 권장, CPU도 가능)
    """
    from src.pipeline.cache_manager import detect_processed_dates

    st.markdown(
        section_header("🚀 Deep Space Foundation Model"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Transformer 기반 Journey 패턴 학습 모델입니다. "
        "MLM(Masked Language Model) 방식으로 Locus 시퀀스를 사전학습합니다."
    )

    # 모델 상태 확인
    try:
        from src.model.trainer import get_model_info
        from src.model.tokenizer import get_tokenizer
        model_info = get_model_info(sector_id)
    except ImportError:
        st.warning("⚠️ PyTorch가 설치되지 않았습니다. `pip install torch`")
        return
    except Exception as e:
        model_info = {"exists": False}

    processed = detect_processed_dates(sector_id)

    # 상태 표시
    col_status, col_params = st.columns([3, 2])

    with col_status:
        if model_info.get("exists"):
            vocab_size = model_info.get("vocab_size", 0)
            n_params = model_info.get("n_params", 0)
            best_loss = model_info.get("best_val_loss", 0)
            best_epoch = model_info.get("best_epoch", 0)

            st.markdown(
                f"<span style='color:#00C897'>✅ 모델 존재</span> &nbsp;·&nbsp; "
                f"vocab {vocab_size} &nbsp;·&nbsp; "
                f"{n_params:,} params &nbsp;·&nbsp; "
                f"best loss {best_loss:.3f} (epoch {best_epoch})",
                unsafe_allow_html=True,
            )

            # 학습 이력 차트
            if model_info.get("train_losses"):
                with st.expander("📊 학습 이력", expanded=False):
                    import pandas as pd
                    history_df = pd.DataFrame({
                        "Epoch": range(1, len(model_info["train_losses"]) + 1),
                        "Train Loss": model_info["train_losses"],
                        "Val Loss": model_info["val_losses"],
                    })
                    st.line_chart(
                        history_df.set_index("Epoch")[["Train Loss", "Val Loss"]],
                        use_container_width=True,
                    )

                    # 정확도 표시
                    if model_info.get("val_acc_top1"):
                        last_idx = -1
                        st.markdown(
                            f"**최종 정확도**: "
                            f"Top-1 {model_info['val_acc_top1'][last_idx]:.1%} &nbsp;·&nbsp; "
                            f"Top-3 {model_info['val_acc_top3'][last_idx]:.1%} &nbsp;·&nbsp; "
                            f"Top-5 {model_info['val_acc_top5'][last_idx]:.1%}"
                        )
        else:
            st.markdown(
                "<span style='color:#FFB300'>⚠️ 모델 없음 — 학습이 필요합니다</span>",
                unsafe_allow_html=True,
            )

    with col_params:
        col_ep, col_bs = st.columns(2)
        with col_ep:
            epochs = st.number_input(
                "Epochs",
                min_value=5, max_value=100, value=10, step=5,
                key="deep_space_epochs",
            )
        with col_bs:
            batch_size = st.number_input(
                "Batch",
                min_value=8, max_value=128, value=64, step=8,
                key="deep_space_batch",
            )

    if not processed:
        st.info("처리 완료된 날짜가 없습니다.")
        return

    st.caption(f"학습 데이터: {len(processed)}일 (journey.parquet)")

    # 학습 버튼
    btn_label = "🚀 Deep Space 학습" if not model_info.get("exists") else "🔄 모델 재학습"
    btn_type = "primary" if not model_info.get("exists") else "secondary"

    if st.button(btn_label, key="train_deep_space_btn", type=btn_type):
        _run_deep_space_training(sector_id, processed, epochs, batch_size)


def _run_deep_space_training(
    sector_id: str,
    dates: list[str],
    epochs: int,
    batch_size: int,
):
    """Deep Space 모델 학습 실행."""
    try:
        from src.model.config import DeepSpaceConfig
        from src.model.trainer import train_deep_space
    except ImportError as e:
        st.error(f"⚠️ 의존성 없음: {e}\n\npip install torch")
        return

    config = DeepSpaceConfig(epochs=epochs, batch_size=batch_size)

    progress_bar = st.progress(0.0, text="Deep Space 학습 준비 중...")
    status_text = st.empty()

    def _progress_callback(epoch, total, train_loss, val_loss):
        frac = epoch / total
        progress_bar.progress(frac, text=f"Epoch {epoch}/{total}")
        status_text.markdown(
            f"<div style='font-size:0.85rem; color:{COLORS['text_bright']};'>"
            f"Train Loss: {train_loss:.4f} &nbsp;·&nbsp; Val Loss: {val_loss:.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )

    try:
        with st.spinner("Deep Space 모델 학습 중..."):
            model = train_deep_space(
                sector_id,
                dates,
                config=config,
                progress_callback=_progress_callback,
            )

        progress_bar.progress(1.0, text="학습 완료!")
        status_text.empty()

        st.success(
            f"✅ Deep Space 모델 학습 완료! ({epochs} epochs, {len(dates)}일 데이터)",
            icon="🚀",
        )

        # 모델 정보 표시
        from src.model.trainer import get_model_info
        info = get_model_info(sector_id)
        if info.get("exists"):
            st.markdown(
                f"**모델 정보**: "
                f"{info.get('n_params', 0):,} params &nbsp;·&nbsp; "
                f"Best Val Loss {info.get('best_val_loss', 0):.4f} "
                f"(Epoch {info.get('best_epoch', 0)})"
            )

        # Drive 업로드
        _auto_upload_deep_space_model(sector_id)

        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Deep Space 학습 실패: {e}")
        import traceback
        with st.expander("상세 오류"):
            st.code(traceback.format_exc())


def _auto_upload_deep_space_model(sector_id: str):
    """Deep Space 모델 자동 Drive 업로드."""
    try:
        from src.pipeline.drive_uploader import is_drive_configured
        if not is_drive_configured():
            return

        # Deep Space 모델 파일 업로드 (추후 구현)
        st.toast("☁️ Deep Space 모델 Drive 업로드 준비됨", icon="✅")
    except Exception as e:
        st.toast(f"⚠️ 모델 Drive 업로드 실패: {e}", icon="⚠️")
