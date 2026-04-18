"""
날씨 데이터 수집 — Open-Meteo Archive API (무료, 키 불필요)
============================================================
Y1 건설현장 (경기도 용인시) 일별 날씨를 자동 수집.

수집 항목:
  - precipitation (mm): 일 강수량
  - snowfall (cm): 일 적설량
  - temp_max (°C): 최고기온
  - temp_min (°C): 최저기온
  - weather (str): 날씨 태그 (Sunny / Rain / Snow / Unknown)

사용법:
  from src.pipeline.weather import fetch_weather, enrich_weather
  weather_df = fetch_weather("2026-03-01", "2026-03-20")
  enriched = enrich_weather(my_df, date_col="date")

API: https://open-meteo.com (Open-Meteo Archive API)
위치: 경기도 용인시 (Y1_SKHynix 건설현장)
"""

import json
import logging
import ssl
import urllib.request
import urllib.parse
from datetime import datetime

import pandas as pd

# SSL 컨텍스트 (macOS certifi 이슈 방어)
def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

logger = logging.getLogger(__name__)

# ─── Y1 건설현장 좌표 (경기도 용인시) ─────────────────────────
Y1_LATITUDE: float = 37.235
Y1_LONGITUDE: float = 127.209

# ─── Open-Meteo API endpoints ────────────────────────────────
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _weather_tag(precipitation: float, snowfall: float) -> str:
    """강수/적설량으로 날씨 태그 결정."""
    if pd.isna(precipitation) and pd.isna(snowfall):
        return "Unknown"
    if not pd.isna(snowfall) and snowfall > 0:
        return "Snow"
    if not pd.isna(precipitation) and precipitation > 0:
        return "Rain"
    return "Sunny"


def fetch_weather(
    start_date: str,
    end_date: str,
    latitude: float = Y1_LATITUDE,
    longitude: float = Y1_LONGITUDE,
) -> pd.DataFrame:
    """
    Open-Meteo Archive API에서 일별 날씨 조회.

    Parameters
    ----------
    start_date : "YYYY-MM-DD" 형식
    end_date   : "YYYY-MM-DD" 형식
    latitude   : 위도 (기본: 용인시)
    longitude  : 경도 (기본: 용인시)

    Returns
    -------
    DataFrame with columns: date, precipitation, snowfall, temp_max, temp_min, weather
    """
    empty = pd.DataFrame(
        columns=["date", "precipitation", "snowfall", "temp_max", "temp_min", "weather"]
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Seoul",
    }

    url = f"{_ARCHIVE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"날씨 API 호출 실패: {e}")
        # Archive 실패 시 Forecast API 시도 (최근 데이터용)
        try:
            params_fc = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": "precipitation_sum,snowfall_sum,temperature_2m_max,temperature_2m_min",
                "timezone": "Asia/Seoul",
                "past_days": 30,
            }
            url_fc = f"{_FORECAST_URL}?{urllib.parse.urlencode(params_fc)}"
            req_fc = urllib.request.Request(url_fc)
            with urllib.request.urlopen(req_fc, timeout=10, context=_ssl_context()) as resp_fc:
                data = json.loads(resp_fc.read().decode())
        except Exception as e2:
            logger.warning(f"날씨 Forecast API도 실패: {e2}")
            return empty

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return empty

    df = pd.DataFrame({
        "date": dates,
        "precipitation": daily.get("precipitation_sum", [None] * len(dates)),
        "snowfall": daily.get("snowfall_sum", [None] * len(dates)),
        "temp_max": daily.get("temperature_2m_max", [None] * len(dates)),
        "temp_min": daily.get("temperature_2m_min", [None] * len(dates)),
    })

    df["weather"] = df.apply(
        lambda r: _weather_tag(r["precipitation"], r["snowfall"]), axis=1
    )

    # 요청 범위로 필터링
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)

    return df


def enrich_weather(
    df: pd.DataFrame,
    date_col: str = "date",
    latitude: float = Y1_LATITUDE,
    longitude: float = Y1_LONGITUDE,
) -> pd.DataFrame:
    """
    DataFrame에 날씨 컬럼을 left-join으로 추가.

    Parameters
    ----------
    df       : date_col 컬럼이 있는 DataFrame
    date_col : 날짜 컬럼 이름 (YYYYMMDD 또는 YYYY-MM-DD)

    Returns
    -------
    날씨 컬럼이 추가된 DataFrame
    """
    if df.empty or date_col not in df.columns:
        return df

    # date 형식 통일 (YYYYMMDD → YYYY-MM-DD)
    dates_raw = df[date_col].astype(str).unique()
    dates_fmt = []
    for d in sorted(dates_raw):
        if len(d) == 8 and d.isdigit():
            dates_fmt.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        else:
            dates_fmt.append(d)

    if not dates_fmt:
        return df

    weather_df = fetch_weather(dates_fmt[0], dates_fmt[-1], latitude, longitude)
    if weather_df.empty:
        return df

    # join key 생성
    out = df.copy()
    if len(str(dates_raw[0])) == 8:
        # YYYYMMDD 형식 → YYYY-MM-DD로 변환하여 join
        out["_date_join"] = out[date_col].astype(str).apply(
            lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        )
        out = out.merge(weather_df, left_on="_date_join", right_on="date",
                       how="left", suffixes=("", "_weather"))
        out.drop(columns=["_date_join", "date_weather"], errors="ignore", inplace=True)
        if "date" in out.columns and date_col != "date":
            out.drop(columns=["date"], errors="ignore", inplace=True)
    else:
        out = out.merge(weather_df, left_on=date_col, right_on="date",
                       how="left", suffixes=("", "_weather"))

    return out
