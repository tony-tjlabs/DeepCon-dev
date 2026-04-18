"""
날씨 + 요일 유틸리티
====================
Open-Meteo Archive API로 날씨 데이터를 조회하고,
날짜 선택 UI에 요일/날씨 정보를 표시하기 위한 유틸리티.

참조: SandBox/Entrance_Analysis_Y1/src/metrics.py 패턴 적용
"""
from __future__ import annotations

import json
import logging
import ssl
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st

from core.cache.policy import WEATHER

logger = logging.getLogger(__name__)

# ── 건설현장 좌표 (Y1 SK하이닉스, 경기도 용인) ──────────────────────
_DEFAULT_LAT, _DEFAULT_LON = 37.235, 127.209

# ── 요일 한글 매핑 ───────────────────────────────────────────────────
_WEEKDAY_KR = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}

# ── 날씨 이모지 ──────────────────────────────────────────────────────
WEATHER_EMOJI = {"Sunny": "☀️", "Rain": "🌧️", "Snow": "❄️", "Unknown": "❓"}

# ── 건설현장 특이일 (공휴일, 설/추석 등) ──────────────────────────────
SPECIAL_DAYS: dict[str, str] = {
    "2026-01-01": "신정",
    "2026-02-16": "설 연휴",
    "2026-02-17": "설 연휴",
    "2026-02-18": "설 연휴",
    "2026-03-01": "삼일절",
    "2026-05-05": "어린이날",
    "2026-06-06": "현충일",
}


def _ssl_ctx():
    """SSL 컨텍스트 생성 (certifi 우선, 없으면 비검증)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


@st.cache_data(show_spinner=False, ttl=WEATHER)
def fetch_weather(
    start_date: str,
    end_date: str,
    lat: float = _DEFAULT_LAT,
    lon: float = _DEFAULT_LON,
) -> pd.DataFrame:
    """Open-Meteo Archive API로 일별 날씨 조회 (24시간 캐시).

    Parameters
    ----------
    start_date, end_date : str
        "YYYY-MM-DD" 형식
    lat, lon : float
        위도/경도 (기본: Y1 건설현장)

    Returns
    -------
    pd.DataFrame
        columns: date, precipitation, snowfall, temp_max, temp_min, weather
    """
    empty = pd.DataFrame(
        columns=["date", "precipitation", "snowfall", "temp_max", "temp_min", "weather"]
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Seoul",
    }
    for base_url in [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://api.open-meteo.com/v1/forecast",
    ]:
        try:
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=10, context=_ssl_ctx()) as resp:
                data = json.loads(resp.read().decode())
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            if dates:
                df = pd.DataFrame({
                    "date": dates,
                    "precipitation": daily.get("precipitation_sum"),
                    "snowfall": daily.get("snowfall_sum"),
                    "temp_max": daily.get("temperature_2m_max"),
                    "temp_min": daily.get("temperature_2m_min"),
                })
                df["weather"] = df.apply(
                    lambda r: "Snow" if (not pd.isna(r["snowfall"]) and r["snowfall"] > 0)
                    else ("Rain" if (not pd.isna(r["precipitation"]) and r["precipitation"] > 0)
                          else "Sunny"),
                    axis=1,
                )
                return df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
        except Exception:
            continue
    return empty


def _get_weather_map() -> dict[str, dict]:
    """세션 캐시에서 weather_map을 반환. 없으면 빈 딕셔너리."""
    return st.session_state.get("_deepcon_weather_map", {})


def init_weather_data(dates: list[str], date_format: str = "%Y%m%d"):
    """날짜 목록으로 날씨 데이터를 조회하여 세션에 캐시.

    Parameters
    ----------
    dates : list[str]
        날짜 문자열 목록 (예: ["20260301", "20260302", ...])
    date_format : str
        날짜 형식 (기본: "%Y%m%d")
    """
    if not dates:
        return
    # 이미 초기화되었으면 스킵
    cache_key = "_deepcon_weather_map"
    if cache_key in st.session_state:
        return

    try:
        date_objs = [pd.to_datetime(d, format=date_format) for d in dates]
        start = min(date_objs).strftime("%Y-%m-%d")
        end = max(date_objs).strftime("%Y-%m-%d")
        weather_df = fetch_weather(start, end)
        if not weather_df.empty:
            st.session_state[cache_key] = weather_df.set_index("date").to_dict("index")
        else:
            st.session_state[cache_key] = {}
    except Exception:
        logger.debug("날씨 데이터 조회 실패 — 요일만 표시됩니다.", exc_info=True)
        st.session_state[cache_key] = {}


def date_label(date_str: str, date_format: str = "%Y%m%d") -> str:
    """날짜 → '2026-03-05 (수) ☀️ -2~5°' 형식 레이블.

    Parameters
    ----------
    date_str : str
        날짜 문자열 (예: "20260305" 또는 "2026-03-05")
    date_format : str
        입력 날짜 형식 (기본: "%Y%m%d")

    Returns
    -------
    str
        포맷팅된 날짜 레이블
    """
    try:
        dt = pd.to_datetime(date_str, format=date_format)
    except Exception:
        return date_str

    dow = _WEEKDAY_KR.get(dt.dayofweek, "")
    iso_date = dt.strftime("%Y-%m-%d")

    # 날씨 정보
    weather_map = _get_weather_map()
    w = weather_map.get(iso_date, {})
    emoji = WEATHER_EMOJI.get(w.get("weather", ""), "")
    temp = ""
    if w.get("temp_min") is not None and w.get("temp_max") is not None:
        temp = f" {w['temp_min']:.0f}°~{w['temp_max']:.0f}°"

    # 특이일
    special = SPECIAL_DAYS.get(iso_date, "")

    label = f"{iso_date} ({dow})"
    if emoji or temp:
        label += f" {emoji}{temp}"
    if special:
        label += f" [{special}]"
    return label


def date_label_iso(iso_date: str) -> str:
    """ISO 형식 날짜 → 레이블 (date_input 위젯용).

    Parameters
    ----------
    iso_date : str
        "YYYY-MM-DD" 형식 날짜
    """
    return date_label(iso_date, date_format="%Y-%m-%d")


def date_label_short(date_str: str, date_format: str = "%Y%m%d") -> str:
    """날짜 → '03-05(수)' 형식 (차트 축 레이블용).

    Parameters
    ----------
    date_str : str
        날짜 문자열 (예: "20260305")
    date_format : str
        입력 날짜 형식 (기본: "%Y%m%d")
    """
    try:
        dt = pd.to_datetime(date_str, format=date_format)
    except Exception:
        return date_str
    dow = _WEEKDAY_KR.get(dt.dayofweek, "")
    weather_map = _get_weather_map()
    iso_date = dt.strftime("%Y-%m-%d")
    w = weather_map.get(iso_date, {})
    emoji = WEATHER_EMOJI.get(w.get("weather", ""), "")
    return f"{dt.strftime('%m-%d')}({dow}){emoji}"
