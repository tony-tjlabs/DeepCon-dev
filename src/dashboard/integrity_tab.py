"""
src.dashboard.integrity_tab — legacy import shim (M3-A 분해 이후)
===================================================================
원본 4,668 LOC 모듈을 `src.dashboard.integrity/` 패키지로 분해했다.
이 파일은 하위 호환을 위한 얇은 shim 으로만 남아 있으며, 신규 코드는
`from src.dashboard.integrity import render_integrity_tab` 을 사용한다.

하위 헬퍼(_render_worker_detail, _render_daily_stats 등)도 동일 이름으로
재노출하므로 외부에서 직접 참조하던 코드는 수정 없이 동작한다.
"""
from src.dashboard.integrity import render_integrity_tab

# 하위 호환 — 기존 이름으로도 import 가능하도록 공개
from src.dashboard.integrity.worker_review import _render_worker_detail
from src.dashboard.integrity.schema_check import _render_daily_stats
from src.dashboard.integrity.sanity_check import (
    _render_anomaly_detection,
    _render_sanity_check,
)
from src.dashboard.integrity.gap_analysis import _render_ble_coverage_gap
from src.dashboard.integrity.physical_validator import _render_physical_validation
from src.dashboard.integrity.context import (
    _compute_daily_integrity_stats,
    _compute_helmet_suspect_by_user,
    _load_journey,
    _load_worker,
    _load_access_log_for_date,
    _load_tward_for_date,
    _load_locus_meta,
    _build_place_ltype_map,
)

__all__ = [
    "render_integrity_tab",
    "_render_worker_detail",
    "_render_daily_stats",
    "_render_anomaly_detection",
    "_render_sanity_check",
    "_render_ble_coverage_gap",
    "_render_physical_validation",
    "_compute_daily_integrity_stats",
    "_compute_helmet_suspect_by_user",
    "_load_journey",
    "_load_worker",
    "_load_access_log_for_date",
    "_load_tward_for_date",
    "_load_locus_meta",
    "_build_place_ltype_map",
]
