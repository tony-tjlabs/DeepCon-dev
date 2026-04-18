"""
Date Utilities (Y1 전용) — 날짜 선택 및 날씨 유틸리티
======================================================
M15X date_utils.py 기반으로 DeepCon Y1에 맞게 재작성.

주요 변경:
  - Y1 현장 좌표 (SK하이닉스 경기도 용인): lat=37.2636, lon=127.0286
  - 날씨 데이터: src/utils/weather.py의 init_weather_data()가 session_state에
    사전 초기화 → fetch_weather_info()가 session_state에서 꺼내도록 조정
  - get_date_badge(): Y1 테마 컬러(#00AEEF) 적용
  - get_available_dates(): config.get_sector_paths() 기반 processed 폴더 탐색

사용:
    from src.dashboard.date_utils import get_date_selector, get_date_badge

    dates = get_available_dates("Y1_SKHynix")
    selected = get_date_selector(dates, key="daily_date")
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config as cfg

# ═══════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════

DAY_NAMES_KR = ["월", "화", "수", "목", "금", "토", "일"]

# Y1 건설현장 좌표 (SK하이닉스 경기도 용인 처인구)
Y1_LATITUDE  = 37.2636
Y1_LONGITUDE = 127.0286

# 날씨 이모지
WEATHER_ICONS: dict[str, str] = {
    "Sunny":   "☀️",
    "Rain":    "🌧️",
    "Snow":    "❄️",
    "Unknown": "🌤️",
}

# Y1 테마 컬러
_THEME_ACCENT = getattr(cfg, "THEME_ACCENT", "#00AEEF")
_THEME_BG     = getattr(cfg, "THEME_BG",     "#0D1B2A")


# ═══════════════════════════════════════════════════════════════════
# 날짜 목록 조회
# ═══════════════════════════════════════════════════════════════════

def get_available_dates(sector_id: str) -> list[str]:
    """
    processed 폴더에서 완료된 날짜 목록 반환.

    config.get_sector_paths()가 반환하는 processed_dir 아래에서
    meta.json이 존재하는 날짜 디렉토리만 반환한다.

    Parameters
    ----------
    sector_id : str
        Sector 식별자 (예: "Y1_SKHynix")

    Returns
    -------
    list[str]
        YYYYMMDD 형식 날짜 목록 (오름차순 정렬)
    """
    try:
        paths = cfg.get_sector_paths(sector_id)
        proc_dir: Path = paths["processed_dir"]
        if not proc_dir.exists():
            return []

        dates = []
        for d in sorted(proc_dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                name = d.name
                # YYYYMMDD 형식 검증
                if len(name) == 8 and name.isdigit():
                    dates.append(name)
        return dates
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# 날씨 정보 조회 (session_state 우선)
# ═══════════════════════════════════════════════════════════════════

def fetch_weather_info(dates: list[str]) -> dict[str, dict]:
    """
    날짜 목록에 대한 요일+날씨 정보 반환.

    Y1 조정: src/utils/weather.py의 init_weather_data()가 session_state에
    사전 초기화한 "_deepcon_weather_map"을 우선 사용한다.
    세션 데이터가 없거나 요청 날짜가 없으면 직접 API 호출로 fallback.

    Parameters
    ----------
    dates : YYYYMMDD 또는 YYYY-MM-DD 형식 날짜 목록

    Returns
    -------
    dict[date_str, {day_kr, weather, icon, temp_max, temp_min, label}]
    """
    if not dates:
        return {}

    # 날짜 형식 통일 (YYYY-MM-DD)
    dates_fmt: list[str] = []
    for d in sorted(dates):
        if len(d) == 8 and d.isdigit():
            dates_fmt.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        else:
            dates_fmt.append(d)

    # session_state에서 날씨 맵 우선 사용
    weather_map: dict[str, dict] = st.session_state.get("_deepcon_weather_map", {})

    # session_state에 없으면 API 직접 호출
    if not weather_map:
        try:
            from src.utils.weather import fetch_weather
            weather_df = fetch_weather(
                start_date=dates_fmt[0],
                end_date=dates_fmt[-1],
                lat=Y1_LATITUDE,
                lon=Y1_LONGITUDE,
            )
            if not weather_df.empty:
                weather_map = weather_df.set_index("date").to_dict("index")
        except Exception:
            weather_map = {}

    info: dict[str, dict] = {}
    for d_orig, d_fmt in zip(sorted(dates), dates_fmt):
        try:
            dt = datetime.strptime(d_fmt, "%Y-%m-%d")
        except ValueError:
            continue

        day_kr = DAY_NAMES_KR[dt.weekday()]
        w_data = weather_map.get(d_fmt, {})
        weather   = w_data.get("weather", "Unknown")
        temp_max  = w_data.get("temp_max")
        temp_min  = w_data.get("temp_min")

        icon     = WEATHER_ICONS.get(weather, "🌤️")
        temp_str = f" {temp_min:.0f}~{temp_max:.0f}" if temp_max is not None else ""

        mm_dd = f"{dt.month:02d}/{dt.day:02d}"
        info[d_orig] = {
            "day_kr":   day_kr,
            "weather":  weather,
            "icon":     icon,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "label":    f"{mm_dd} ({day_kr}) {icon}{temp_str}",
        }

    return info


# ═══════════════════════════════════════════════════════════════════
# 날짜 선택기 위젯
# ═══════════════════════════════════════════════════════════════════

def get_date_selector(
    dates: list[str],
    key: str = "date_selector",
    default_index: int | None = None,
    label: str = "날짜 선택",
    show_label: bool = False,
) -> str | None:
    """
    요일+날씨 포함 날짜 선택기.

    M15X date_utils.get_date_selector()와 동일 인터페이스.
    Y1의 session_state 날씨 캐시를 우선 사용한다.

    Parameters
    ----------
    dates : YYYYMMDD 형식 날짜 목록
    key : Streamlit widget 키
    default_index : 기본 선택 인덱스 (None이면 마지막 날짜)
    label : 라벨 텍스트
    show_label : 라벨 표시 여부

    Returns
    -------
    선택된 날짜 (YYYYMMDD 형식) 또는 None
    """
    if not dates:
        return None

    weather_info = fetch_weather_info(dates)
    date_labels  = [weather_info.get(d, {}).get("label", d) for d in dates]

    if default_index is None:
        default_index = len(dates) - 1

    selected_label = st.selectbox(
        label,
        date_labels,
        index=default_index,
        key=key,
        label_visibility="visible" if show_label else "collapsed",
    )

    try:
        idx = date_labels.index(selected_label)
        return dates[idx]
    except (ValueError, IndexError):
        return dates[-1] if dates else None


# ═══════════════════════════════════════════════════════════════════
# 날짜 포맷 유틸리티
# ═══════════════════════════════════════════════════════════════════

def get_weekday_korean(d: Any) -> str:
    """요일 한글 반환 (date 또는 datetime 객체)."""
    return DAY_NAMES_KR[d.weekday()]


def get_date_badge(d: Any, styles_module: Any = None) -> str:
    """
    날짜 뱃지 HTML (주중/주말).

    Y1 테마 컬러(#00AEEF) 적용.
    styles_module이 없으면 src.dashboard.styles.badge를 사용한다.
    """
    if styles_module is None:
        from src.dashboard.styles import badge
    else:
        badge = styles_module.badge

    if d.weekday() >= 5:
        return badge("주말", "warning")
    return badge("주중", "info")


def parse_date_str(date_str: str) -> datetime | None:
    """YYYYMMDD 또는 YYYY-MM-DD 형식 날짜 문자열 파싱."""
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def format_date_label(date_str: Any) -> str:
    """YYYYMMDD -> MM/DD 형식으로 변환. int/float 입력도 처리."""
    date_str = str(date_str).split(".")[0]
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[4:6]}/{date_str[6:]}"
    return date_str


def format_date_full(date_str: Any) -> str:
    """YYYYMMDD -> YYYY-MM-DD 형식으로 변환. int/float 입력도 처리."""
    date_str = str(date_str).split(".")[0]
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def get_weather_emoji(condition: str) -> str:
    """날씨 조건 문자열 -> 이모지 반환."""
    return WEATHER_ICONS.get(condition, "🌤️")
