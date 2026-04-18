"""
Dashboard Token Utilities
==========================
locus_v2.csv 기반 공간 토큰 분류 유틸리티.

이전에는 src.pipeline.metrics._get_token_sets 에서 직접 import 했으나,
CLOUD_MODE 환경에서는 pipeline 코드가 없으므로 UI 레이어에서 독립 제공.

로직: locus_v2.csv 의 locus_type / function 컬럼을 읽어 토큰 집합을 동적 빌드.
IP 없음 — 공간 메타데이터(locus_v2.csv) 읽기만 수행.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ─── v1 fallback 토큰 (locus CSV 없거나 v1 모드 시) ────────────────
_V1_WORK_TOKENS    = {"work_zone", "outdoor_work", "mechanical_room",
                      "confined_space", "high_voltage", "transit"}
_V1_TRANSIT_TOKENS = {"timeclock", "main_gate", "sub_gate"}
_V1_REST_TOKENS    = {"breakroom", "smoking_area", "dining_hall", "restroom", "parking_lot"}
_V1_ADMIN_TOKENS: set[str] = set()

# 섹터별 캐시
_cached: dict[str, dict[str, set[str]]] = {}


def _v1_token_sets() -> dict[str, set[str]]:
    return {
        "work":    _V1_WORK_TOKENS,
        "transit": _V1_TRANSIT_TOKENS,
        "rest":    _V1_REST_TOKENS,
        "admin":   _V1_ADMIN_TOKENS,
    }


def get_token_sets(sector_id: str | None = None) -> dict[str, set[str]]:
    """
    LOCUS_VERSION 에 따라 토큰 분류 집합 반환.

    v2: locus_v2.csv 의 locus_type/function 기반 동적 분류
    v1 / fallback: 하드코딩된 영문 토큰 집합

    Returns:
        {"work": set, "transit": set, "rest": set, "admin": set}
    """
    try:
        import config as cfg
    except ImportError:
        return _v1_token_sets()

    sid = sector_id or cfg.SECTOR_ID

    if sid in _cached:
        return _cached[sid]

    if getattr(cfg, "LOCUS_VERSION", "v1") != "v2":
        result = _v1_token_sets()
        _cached[sid] = result
        return result

    try:
        paths = cfg.get_sector_paths(sid)
        csv_path: Path | None = paths.get("locus_v2_csv")
        if not csv_path or not csv_path.exists():
            logger.warning("[token_utils] locus_v2.csv 없음 (%s) — v1 fallback", csv_path)
            result = _v1_token_sets()
            _cached[sid] = result
            return result

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        token_col = "locus_id" if "locus_id" in df.columns else "locus_name"

        ltype = df.get("locus_type", pd.Series(dtype=str)).fillna("").str.upper()
        func  = df.get("function",   pd.Series(dtype=str)).fillna("").str.upper()

        # REST: REST_AREA 또는 REST function
        rest_mask    = (ltype == "REST_AREA") | (func == "REST")
        rest_tokens  = set(df.loc[rest_mask, token_col].dropna())

        # TRANSIT/GATE
        transit_mask   = (ltype == "GATE") | (func == "ACCESS") | (ltype == "TRANSIT") | (func == "MOVE")
        transit_tokens = set(df.loc[transit_mask, token_col].dropna()) - rest_tokens

        # ADMIN (향후 확장용)
        admin_tokens: set[str] = set()

        # WORK: 나머지 WORK_AREA / VERTICAL
        work_mask  = ((ltype == "WORK_AREA") | (func == "WORK") | (ltype == "VERTICAL")) & ~rest_mask & ~transit_mask
        work_tokens = set(df.loc[work_mask, token_col].dropna())

        result = {
            "work":    work_tokens,
            "transit": transit_tokens,
            "rest":    rest_tokens,
            "admin":   admin_tokens,
        }
        _cached[sid] = result

    except Exception as exc:
        logger.warning("[token_utils] v2 토큰 로드 실패: %s — v1 fallback", exc)
        result = _v1_token_sets()
        _cached[sid] = result

    return result
