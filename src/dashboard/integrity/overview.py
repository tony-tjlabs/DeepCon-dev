"""
integrity/overview.py — 정합성 탭 메인 진입점
==============================================
`render_integrity_tab(sector_id)` 를 구현한다.

책임:
  - 상단 헤더 + M2-B AI 품질 감사 박스 (T-19)
  - 5개 서브탭 라우팅:
      1. 🔎 작업자 상세 검토 (worker_review)
      2. 📊 현장 보정 통계 (schema_check)
      3. ⚠️ 이상 패턴 감지 (sanity_check._render_anomaly_detection)
      4. 📉 BLE 커버리지 이상 (gap_analysis)
      5. 🚨 비상식 패턴 (sanity_check._render_sanity_check)
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import config as cfg
from src.dashboard.integrity.context import _compute_daily_integrity_stats
from src.dashboard.integrity.worker_review import _render_worker_detail
from src.dashboard.integrity.schema_check import _render_daily_stats
from src.dashboard.integrity.sanity_check import (
    _render_anomaly_detection,
    _render_sanity_check,
)
from src.dashboard.integrity.gap_analysis import _render_ble_coverage_gap

logger = logging.getLogger(__name__)


def render_integrity_tab(sector_id: str) -> None:
    """데이터 정합성 탭 메인 렌더링."""
    st.markdown(
        """
        <div style='margin-bottom:12px;'>
            <h2 style='color:#D5E5FF; margin-bottom:2px;'>🔬 데이터 정합성 검토</h2>
            <p style='color:#7A8FA6; font-size:0.88rem; margin-top:0;'>
                1분 단위 BLE 데이터 → 보정된 Journey 변환 과정의 투명성 확보.
                전문가는 이 도구로 <b style='color:#00AEEF'>언제 / 어디서 / 왜</b> 보정이
                일어났는지 개별 검토할 수 있습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── AI 코멘터리 (T-19) — D-5 사용자 요청 ─────────────────────
    try:
        from src.dashboard.components import ai_commentary_box
        from core.ai import build_integrity_context
        from src.dashboard.auth import get_current_user
        from src.pipeline.cache_manager import detect_processed_dates

        processed = detect_processed_dates(sector_id)
        latest_date = processed[-1] if processed else None

        # 일별 보정 통계 집계 (기존 _compute_daily_integrity_stats 재활용)
        try:
            stats_df = _compute_daily_integrity_stats(sector_id)
        except Exception:
            stats_df = None

        # 최신일 worker (헬멧 방치 의심 추정용)
        worker_df_ai = None
        meta_json_ai = None
        if latest_date:
            try:
                paths = cfg.get_sector_paths(sector_id)
                p = paths["processed_dir"] / latest_date / "worker.parquet"
                if p.exists():
                    try:
                        import pyarrow.parquet as _pq
                        _avail = set(_pq.read_schema(p).names)
                        _cols = [c for c in ("user_no", "low_active_min", "work_zone_minutes")
                                 if c in _avail]
                        worker_df_ai = pd.read_parquet(p, columns=_cols) if _cols else pd.read_parquet(p)
                    except Exception:
                        worker_df_ai = pd.read_parquet(p)
                meta_p = paths["processed_dir"] / latest_date / "meta.json"
                if meta_p.exists():
                    import json as _json
                    meta_json_ai = _json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        ai_ctx = build_integrity_context(
            sector_id=sector_id,
            date_str=str(latest_date) if latest_date else None,
            stats_df=stats_df,
            meta_json=meta_json_ai,
            worker_df=worker_df_ai,
        )
        ai_commentary_box(
            role="integrity_auditor",
            context=ai_ctx,
            sector_id=sector_id,
            date_str=str(latest_date) if latest_date else None,
            title="데이터 품질 AI 감사",
            spinner_text="데이터 품질 감사 중...",
            button_label="AI 품질 감사 실행 (Haiku)",
            user_role=get_current_user().get("role", "unknown"),
            tab="integrity",
            show_meta=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"AI 코멘터리 렌더 실패 (integrity): {e}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 작업자 상세 검토",
        "📊 현장 보정 통계",
        "⚠️ 이상 패턴 감지",
        "📉 BLE 커버리지 이상",
        "🚨 비상식 패턴 (Sanity)",
    ])

    with tab1:
        _render_worker_detail(sector_id)

    with tab2:
        _render_daily_stats(sector_id)

    with tab3:
        _render_anomaly_detection(sector_id)

    with tab4:
        _render_ble_coverage_gap(sector_id)

    with tab5:
        _render_sanity_check(sector_id)
