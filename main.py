"""
DeepCon — Agentic AI based on Spatial Data
==========================================
Sector 기반 접근 제어 + 로그인 시스템

실행: streamlit run main.py

★ Performance (v1.1):
  - Lazy import: 탭 모듈은 페이지 라우팅 시점에만 로드
  - CLOUD_MODE: Drive 동기화, 파이프라인 탭 숨김
  - 사이드바: get_cache_status / get_spatial_summary 캐시 활용
"""
import logging
import streamlit as st

# ── 페이지 설정 (반드시 첫 번째) ─────────────────────────────────
st.set_page_config(
    page_title="DeepCon · Spatial AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config as cfg
from src.dashboard.auth import (
    is_logged_in, require_login, enforce_timeout,
    get_current_user, get_current_sector, set_current_sector,
    get_allowed_sectors, is_admin, logout,
)
from src.dashboard.styles import inject_css

# ★ 탭 모듈 lazy import 제거 — 페이지 라우팅 블록에서 inline import


# ─── CLOUD_MODE: Google Drive 동기화 ─────────────────────────────
_RECENT_DAYS   = 14   # Phase-1: 최근 N일 우선 로드 (빠른 시작)

def _bg_sync_full(drives_cfg: list[tuple]) -> None:
    """
    백그라운드 스레드: 나머지 전체 날짜 동기화.

    Phase-1(최근 14일)이 완료된 후 실행.
    Streamlit UI를 건드리지 않으므로 thread-safe.
    결과는 디스크에 저장되고, 사용자가 해당 날짜를 선택할 때 자동으로 로드됨.
    """
    import logging
    _log = logging.getLogger(__name__)
    for creds, folder_id, sector_id, local_dir in drives_cfg:
        try:
            from src.pipeline.drive_storage import DriveStorage
            drive = DriveStorage(creds, folder_id)
            drive._local_dir = local_dir
            new_count = drive.sync_all()
            if new_count:
                _log.info(f"[BG sync] {sector_id}: {new_count}개 추가 다운로드 완료")
        except Exception as e:
            _log.warning(f"[BG sync] {sector_id} 실패: {e}")


def _init_drive_cache() -> None:
    """
    CLOUD_MODE 데이터 초기화 (2-Phase Progressive Loading).

    Phase 1 (동기, 빠름):  최근 14일 데이터 다운로드 → 앱 즉시 사용 가능
    Phase 2 (백그라운드):  나머지 전체 날짜 → 스레드로 조용히 완료

    ★ Cold start UX: 64MB 전체 대신 23MB(14일)만 먼저 받아 시작 시간 단축
    """
    if not cfg.CLOUD_MODE:
        return
    if st.session_state.get("_drive_synced"):
        return

    import logging, threading
    _log = logging.getLogger(__name__)

    try:
        from src.pipeline.drive_storage import init_drive_storage_from_secrets
        drives = init_drive_storage_from_secrets()
        if drives is None:
            _log.info("DriveStorage: SA 키 없음 → git 데이터만 사용")
            st.session_state["_drive_synced"] = True
            st.session_state["_drive_status"] = "no_sa"
            return

        # ── Phase 1: 최근 14일 우선 동기화 ─────────────────────────
        total_new = 0
        bg_args: list[tuple] = []
        for sector_id, drive in drives.items():
            drive._local_dir = cfg.PROCESSED_DIR
            new_count = drive.sync_recent(n_dates=_RECENT_DAYS)
            total_new += new_count
            _log.info(f"Drive sync recent [{sector_id}]: {new_count} new files")
            # Phase 2 용 인자 수집 (credentials dict 복사)
            bg_args.append((
                drive._credentials_info,
                drive._folder_id,
                sector_id,
                cfg.PROCESSED_DIR,
            ))

        if total_new > 0:
            st.toast(f"☁️ {total_new}개 최신 파일 로드 완료", icon="✅")
            from src.pipeline.cache_manager import detect_processed_dates
            detect_processed_dates.clear()

        st.session_state["_drive_synced"] = True
        st.session_state["_drive_status"] = f"ok:{total_new}"
        st.session_state["_bg_sync_done"] = False

        # ── Phase 2: 나머지 날짜 백그라운드 동기화 ─────────────────
        t = threading.Thread(
            target=_bg_sync_full,
            args=(bg_args,),
            daemon=True,
        )
        t.start()
        _log.info("Phase-2 백그라운드 동기화 스레드 시작")

    except Exception as e:
        _log.warning(f"Drive sync failed: {e}")
        st.session_state["_drive_synced"] = True
        st.session_state["_drive_status"] = f"error:{e}"


# ─── 로그인 게이트 ────────────────────────────────────────────────
inject_css()
# M1 Security: 세션 만료(idle 30분 / absolute 8시간) 강제 로그아웃.
# require_login() 내부에서도 호출되지만, 상단에서 명시적으로 한 번 더
# 실행해 "만료된 세션으로 중간 코드가 실행되는 창"을 최소화한다.
enforce_timeout()
require_login()


# ─── 이하 코드는 로그인 성공 후에만 실행됨 ────────────────────────
def render_sidebar() -> tuple[str, str]:
    """
    사이드바 렌더링.
    반환: (page, current_sector_id)
    """
    user = get_current_user()

    with st.sidebar:
        # ── 브랜드 로고 ───────────────────────────────────────────
        st.markdown(
            """
            <div style='text-align:center; padding: 20px 0 12px 0;'>
                <div style='font-size:2.4rem;'>🌐</div>
                <div style='font-size:1.4rem; font-weight:800; color:#00AEEF;
                            letter-spacing:2px; font-family:monospace;'>DeepCon</div>
                <div style='font-size:0.72rem; color:#7A8FA6; margin-top:4px;
                            line-height:1.4;'>
                    Agentic AI based on<br>Spatial Data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Sector 선택 ───────────────────────────────────────────
        allowed_sectors = get_allowed_sectors()
        current_sector  = get_current_sector()

        if not allowed_sectors:
            st.warning("접근 가능한 Sector가 없습니다.")
            logout()
            return "", ""

        st.markdown(
            "<div style='font-size:0.8rem; color:#7A8FA6; margin-bottom:6px;'>"
            "Sector</div>",
            unsafe_allow_html=True,
        )

        sector_labels = {
            sid: f"{cfg.SECTOR_REGISTRY[sid]['icon']}  {cfg.SECTOR_REGISTRY[sid]['label']}"
            for sid in allowed_sectors
        }
        selected_sector = st.radio(
            "Sector 선택",
            options=list(sector_labels.keys()),
            format_func=lambda x: sector_labels[x],
            index=allowed_sectors.index(current_sector) if current_sector in allowed_sectors else 0,
            label_visibility="collapsed",
            key="sector_radio",
        )

        # Sector 변경 감지
        if selected_sector != current_sector:
            set_current_sector(selected_sector)
            st.cache_data.clear()
            st.rerun()

        # Sector 부제목
        sec_info = cfg.SECTOR_REGISTRY.get(selected_sector, {})
        st.markdown(
            f"<div style='font-size:0.75rem; color:#4A6A8A; margin-top:4px;"
            f"padding-left:4px;'>{sec_info.get('subtitle','')}</div>",
            unsafe_allow_html=True,
        )

        # Admin 전용: 추가 Sector 힌트
        if is_admin():
            coming = [
                sid for sid, info in cfg.SECTOR_REGISTRY.items()
                if info.get("status") == "coming_soon"
            ]
            if coming:
                for sid in coming:
                    info = cfg.SECTOR_REGISTRY[sid]
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#2A3A4A; padding:3px 0 0 4px;'>"
                        f"🔒 {info['icon']} {info['label']} <span style='color:#2A3A4A'>"
                        f"(준비중)</span></div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── 페이지 네비게이션 ────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.8rem; color:#7A8FA6; margin-bottom:6px;'>"
            "메뉴</div>",
            unsafe_allow_html=True,
        )

        # ★ 인사이트 & 예측 / 시스템 관리: 관리자만 표시
        #   (데이터 정합성은 모든 계정 공개 — 현장 고객사도 데이터 품질 직접 확인 가능)
        base_pages = [
            "🏠 현장 개요",
            "⏱ 작업시간 분석",
            "🏭 생산성 분석",
            "🦺 안전성 분석",
            "📅 기간 분석",
            "🔬 데이터 정합성",
        ]
        if is_admin():
            pages = base_pages + [
                "🧠 인사이트 & 예측",
                "⚙️ 시스템 관리",
            ]
        else:
            pages = base_pages

        page = st.radio(
            "페이지", pages,
            label_visibility="collapsed",
            key="page_radio",
        )

        st.divider()

        # ── 시스템 상태 ──────────────────────────────────────────
        try:
            # ★ Lazy import — 사이드바에서만 사용
            from src.pipeline.cache_manager import get_cache_status
            from src.spatial.loader import get_spatial_summary

            paths   = cfg.get_sector_paths(selected_sector)
            status  = get_cache_status(str(paths["raw_dir"]), selected_sector)
            spatial = get_spatial_summary(selected_sector)

            # ★ 간결한 시스템 상태 (핵심 3개만)
            n_unproc = status["total_unprocessed"]
            unproc_color = "#FFB300" if n_unproc > 0 else "#00C897"
            st.markdown(
                f"<div style='font-size:0.8rem; color:#9AB5D4; margin-bottom:6px;'>"
                f"시스템 상태</div>",
                unsafe_allow_html=True,
            )
            if n_unproc > 0:
                st.markdown(
                    f"<div style='margin-bottom:8px;'>"
                    f"<span style='color:{unproc_color}; font-size:1.2rem; font-weight:700;'>"
                    f"⏳ {n_unproc}일</span>"
                    f"<span style='color:#9AB5D4; font-size:0.78rem;'> 미처리</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            _ds = st.session_state.get("_drive_status", "")
            _bg_done = st.session_state.get("_bg_sync_done", True)
            _di = "☁️" if _ds.startswith("ok") else "⚠️" if _ds.startswith("error") else "📁"
            if _ds.startswith("ok"):
                _dl = "최근 14일 로드됨" + ("" if _bg_done else " · 전체 동기화 중…")
            elif _ds == "no_sa":
                _dl = "git 데이터"
            elif _ds.startswith("error"):
                _dl = "오류"
            else:
                _dl = ""
            _drive_line = f"<br>{_di} Drive: {_dl}" if _dl else ""
            st.markdown(
                f"<div style='font-size:0.78rem; color:#6A7A95; line-height:1.8;'>"
                f"✅ 처리 {status['total_processed']}일"
                f" · 🗺️ {spatial['locus_count']} Locus"
                f"{_drive_line}</div>",
                unsafe_allow_html=True,
            )
            if is_admin() and _ds.startswith("error"):
                with st.expander("☁️ Drive 오류", expanded=False):
                    st.caption(_ds[6:])
        except Exception:
            pass

        st.divider()

        # ── 사용자 정보 + 로그아웃 ──────────────────────────────
        st.markdown(
            f"<div style='font-size:0.82rem; color:#7A8FA6;'>"
            f"{user['icon']} <b style='color:#C8D6E8'>{user['label']}</b><br>"
            f"<span style='font-size:0.72rem; color:#3A4A5A;'>"
            f"{'관리자' if is_admin() else '클라이언트'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("🚪 로그아웃", use_container_width=True, key="logout_btn"):
            logout()

        st.markdown(
            f"<div style='text-align:center; color:#2A3A4A; font-size:0.72rem;"
            f"margin-top:12px;'>v{cfg.APP_VERSION} · TJLABS Research</div>",
            unsafe_allow_html=True,
        )

    return page, selected_sector


def _init_summary_index(sector_id: str) -> None:
    """
    앱 시작 시 summary_index.json 초기화.

    index 파일이 없거나 processed/ 날짜 수와 불일치하면 자동 재빌드.
    ★ Perf: meta.json만 읽음 (Parquet 로드 없음).
    """
    if st.session_state.get(f"_summary_init_{sector_id}"):
        return
    try:
        from src.pipeline.cache_manager import detect_processed_dates
        from src.pipeline.summary_index import load_summary_index, build_summary_index

        processed = detect_processed_dates(sector_id)
        if not processed:
            st.session_state[f"_summary_init_{sector_id}"] = True
            return

        idx = load_summary_index(sector_id)
        existing_dates = set(idx.get("dates", {}).keys())
        missing = [d for d in processed if d not in existing_dates]

        if missing:
            with st.spinner("📊 요약 인덱스 구축 중..."):
                build_summary_index(sector_id)

        st.session_state[f"_summary_init_{sector_id}"] = True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"summary_index 초기화 실패: {e}")
        st.session_state[f"_summary_init_{sector_id}"] = True


def _run_insight_engine(sector_id: str) -> None:
    """
    인사이트 엔진 실행 (날짜 변경 시 1회). 결과를 session_state에 저장.

    ★ Perf: journey.parquet(35MB) 스킵 — worker/space/company만 로드하여
    인사이트 계산 시간을 대폭 단축 (5×35MB 절약).
    """
    try:
        from src.pipeline.cache_manager import detect_processed_dates, _date_dir
        processed = detect_processed_dates(sector_id)
        if not processed:
            return

        latest_date = processed[-1]

        # 이미 같은 날짜로 인사이트 생성했으면 스킵
        insight_key = f"insight_{sector_id}_{latest_date}"
        if st.session_state.get("_insight_key") == insight_key:
            return

        import json
        import pandas as pd

        # ★ Perf: journey 스킵 — worker/space/company + meta만 로드
        date_dir = _date_dir(latest_date, sector_id)
        data = {}
        for name in ["worker", "space", "company"]:
            p = date_dir / f"{name}.parquet"
            data[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        meta_path = date_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data["meta"] = json.load(f)
        else:
            data["meta"] = {}

        if data["worker"].empty:
            return

        from src.intelligence.insight_aggregator import run_insight_pipeline
        report = run_insight_pipeline(data, latest_date, sector_id, processed)

        st.session_state["_insights"] = report
        st.session_state["_insight_data"] = data
        st.session_state["_insight_key"] = insight_key

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"인사이트 엔진 실패: {e}")


def main():
    # ★ CLOUD_MODE: Drive 동기화 (최초 1회)
    _init_drive_cache()

    page, sector_id = render_sidebar()
    if not sector_id:
        return

    sec_info = cfg.SECTOR_REGISTRY.get(sector_id, {})

    # ── 페이지 헤더 ──────────────────────────────────────────────
    st.markdown(
        f"<h1 style='color:#C8D6E8; font-size:1.7rem; margin-bottom:2px;'>"
        f"{sec_info.get('icon','🌐')} DeepCon &nbsp;"
        f"<span style='color:#00AEEF'>{sec_info.get('label','')}</span>"
        f"</h1>"
        f"<p style='color:#7A8FA6; font-size:0.88rem; margin-top:0;'>"
        f"Agentic AI based on Spatial Data &nbsp;·&nbsp; "
        f"{sec_info.get('subtitle','')}"
        f"</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Summary Index 초기화 (앱 시작 시 1회) ────────────────────
    _init_summary_index(sector_id)

    # ── 날씨 데이터 초기화 (1회, 세션 캐시) ────────────────────────
    try:
        from src.pipeline.cache_manager import detect_processed_dates
        from src.utils.weather import init_weather_data
        _processed_for_weather = detect_processed_dates(sector_id)
        if _processed_for_weather:
            init_weather_data(_processed_for_weather)
    except Exception:
        pass  # 날씨 실패 시 요일만 표시

    # ── 인사이트 파이프라인 (일별/주간 공통, 1회 계산) ──────────────
    _run_insight_engine(sector_id)

    # ── 페이지 라우팅 (★ Lazy import) ────────────────────────────
    def _safe_render(tab_name: str, importer):
        try:
            importer()
        except Exception as e:
            st.error(f"{tab_name} 탭 로드 실패: {e}")
            logging.exception("%s 탭 에러", tab_name)

    if "현장 개요" in page:
        _safe_render("현장 개요", lambda: __import__(
            "src.dashboard.overview_tab", fromlist=["render_overview_tab"]
        ).render_overview_tab(sector_id))
    elif "작업시간" in page:
        _safe_render("작업시간 분석", lambda: __import__(
            "src.dashboard.zone_time_tab", fromlist=["render_zone_time_tab"]
        ).render_zone_time_tab(sector_id))
    elif "생산성" in page:
        _safe_render("생산성 분석", lambda: __import__(
            "src.dashboard.productivity_tab", fromlist=["render_productivity_tab"]
        ).render_productivity_tab(sector_id))
    elif "안전성" in page:
        _safe_render("안전성 분석", lambda: __import__(
            "src.dashboard.safety_tab", fromlist=["render_safety_tab"]
        ).render_safety_tab(sector_id))
    elif "기간 분석" in page:
        _safe_render("기간 분석", lambda: __import__(
            "src.dashboard.period_tab", fromlist=["render_period_tab"]
        ).render_period_tab(sector_id))
    elif "데이터 정합성" in page:
        # 데이터 정합성은 전 계정 접근 허용 (2026-04-18~)
        # M3-A 분해 이후 신규 경로로 직접 import (shim 우회)
        _safe_render("데이터 정합성", lambda: __import__(
            "src.dashboard.integrity", fromlist=["render_integrity_tab"]
        ).render_integrity_tab(sector_id))
    elif "인사이트" in page and is_admin():
        _safe_render("인사이트 & 예측", lambda: __import__(
            "src.dashboard.deep_space_tab", fromlist=["render_deep_space_tab"]
        ).render_deep_space_tab(sector_id))
    elif "시스템 관리" in page and is_admin():
        _safe_render("시스템 관리", lambda: __import__(
            "src.dashboard.admin_tab", fromlist=["render_admin_tab"]
        ).render_admin_tab(sector_id))


if __name__ == "__main__":
    main()
