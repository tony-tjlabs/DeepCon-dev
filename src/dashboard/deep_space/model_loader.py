"""
Deep Space 모델 로더
====================
모델/토크나이저/데이터 로드 함수 집합.
torch는 함수 내부에서 lazy import.
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from config import PROCESSED_DIR, DEEP_SPACE_DIR, SPATIAL_DIR, LOCUS_VERSION
from core.cache.policy import DAILY_PARQUET, STATUS, SPATIAL_MODEL

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Deep Space 모델 로딩...")
def load_model(sector_id: str):
    """
    Deep Space 모델 + 토크나이저 로드.

    torch는 이 함수가 호출될 때만 import됨 (lazy).

    Returns:
        (model, tokenizer) or (None, None)
    """
    try:
        from src.model.transformer import DeepSpaceModel
        from src.model.tokenizer import LocusTokenizer

        model_dir = DEEP_SPACE_DIR / sector_id
        ckpt_path = model_dir / "checkpoint" / "best_model.pt"
        tok_path = model_dir / "tokenizer.json"

        if not ckpt_path.exists():
            return None, None
        model = DeepSpaceModel.from_pretrained(str(ckpt_path))
        model.eval()
        tokenizer = LocusTokenizer.load(str(tok_path))
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Deep Space 모델 로드 실패: {e}")
        return None, None


# Deep Space 서브탭에서 필요한 journey 컬럼 (전체 로드 방지 → Cloud OOM 방지)
_JOURNEY_COLUMNS = [
    "timestamp", "user_no", "locus_id", "locus_name", "locus_token",
    "seq", "is_work_hour", "is_transition", "activity_level",
    "building_name", "floor_name", "company_name",
]


@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def load_journey_data(
    sector_id: str,
    date_str: str,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """특정 날짜의 journey.parquet 로드 (컬럼 프루닝 적용).

    Args:
        columns: 읽을 컬럼 리스트. None이면 Deep Space 기본 컬럼만 로드.
                 빈 리스트([])면 전체 로드.
    """
    path = PROCESSED_DIR / sector_id / date_str / "journey.parquet"
    if not path.exists():
        return None
    if columns is not None and len(columns) == 0:
        return pd.read_parquet(path)
    use_cols = columns or _JOURNEY_COLUMNS
    try:
        return pd.read_parquet(path, columns=use_cols)
    except Exception:
        # 컬럼 불일치 시 전체 로드 fallback
        return pd.read_parquet(path)


@st.cache_data(ttl=DAILY_PARQUET, show_spinner=False)
def load_worker_data(sector_id: str, date_str: str) -> pd.DataFrame | None:
    """특정 날짜의 worker.parquet 로드."""
    path = PROCESSED_DIR / sector_id / date_str / "worker.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=STATUS, show_spinner=False)
def get_available_dates(sector_id: str) -> list[str]:
    """사용 가능한 날짜 목록."""
    proc_dir = PROCESSED_DIR / sector_id
    if not proc_dir.exists():
        return []
    return sorted([d.name for d in proc_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()])


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner=False)
def load_training_history(sector_id: str) -> dict | None:
    """학습 이력 로드."""
    import json
    hist_path = DEEP_SPACE_DIR / sector_id / "training_history.json"
    if not hist_path.exists():
        return None
    with open(hist_path, "r") as f:
        return json.load(f)


@st.cache_data(ttl=SPATIAL_MODEL, show_spinner=False)
def load_locus_info(sector_id: str) -> pd.DataFrame | None:
    """locus.csv 로드 (LOCUS_VERSION에 따라 v2/v1 자동 선택)."""
    locus_dir = SPATIAL_DIR / sector_id / "locus"
    # v2: locus_v2.csv 우선, 없으면 locus.csv fallback
    if LOCUS_VERSION == "v2":
        path = locus_dir / "locus_v2.csv"
        if not path.exists():
            path = locus_dir / "locus.csv"
    else:
        path = locus_dir / "locus.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_locus_meta(locus_info: pd.DataFrame | None) -> dict[str, dict]:
    """
    locus_info DataFrame에서 Locus 메타 정보 딕셔너리 생성 (vectorized).

    Returns:
        {locus_id: {"name": ..., "type": ...}}
    """
    if locus_info is None or locus_info.empty:
        return {}

    return {
        str(r.get("locus_id", "")): {
            "name": r.get("locus_name", r.get("name", str(r.get("locus_id", "")))),
            "type": r.get("locus_type", r.get("dwell_category", "")),
        }
        for r in locus_info.to_dict("records")
        if r.get("locus_id")
    }


# ─── 공간 용량 기본값 (config.py 단일 소스에서 re-export) ──────
from config import DEFAULT_CAPACITY_BY_TYPE  # noqa: E402 — 하위 호환 re-export


def detect_time_column(df: pd.DataFrame) -> str | None:
    """데이터프레임에서 시간 컬럼 이름 자동 감지."""
    for col in ["timestamp", "data_date", "minute", "time_index"]:
        if col in df.columns:
            return col
    return None
