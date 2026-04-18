"""
core.cache.policy — 캐시 TTL 정책 중앙화 (Upgrade v3 M4-T30)
============================================================
모든 `@st.cache_data` 호출의 TTL을 한 곳에서 관리한다.

설계 의도
--------
이전까지 각 탭·로더는 `ttl=300`, `ttl=600`, `ttl=1200`, `ttl=3600` 같은
매직 넘버를 산발적으로 사용했다. 성능 튜닝·장애 대응 시 일괄 변경이
불가능했고, 의도가 분명하지 않아 불필요하게 긴/짧은 TTL이 섞여 있었다.

이 모듈은 캐시를 "데이터 성격"에 따라 분류하고, 각 카테고리의 TTL을
상수로 고정한다. 호출 지점은 `from core.cache.policy import DAILY_PARQUET`
처럼 카테고리만 import 한다 → 정책 변경 시 이 파일 하나만 수정하면 된다.

카테고리
--------
DAILY_PARQUET   (3600s = 1h)
    일별 processed parquet (worker/space/company/journey).
    같은 날짜를 여러 탭에서 반복 로드하므로 캐시 히트율이 매우 높다.
    한 번 처리된 일자의 parquet 파일은 재처리 전까지 바뀌지 않기 때문에
    1시간 TTL 이 안전하다.

MULTI_DAY_AGG   (1800s = 30m)
    다일 집계 (period_tab, productivity_tab._load_multi_day_*).
    I/O+집계 비용이 크고, 재처리/파이프라인 실행 후 무효화가 필요하다.
    30분은 "새 일자 추가 → 대시보드 반영" 지연의 상한.

STATUS          (60s)
    실시간 상태 (processed_dates, cache_status, summary_index 경량 조회).
    파이프라인이 새 일자를 추가하면 ~1분 내 UI 반영.

SPATIAL_MODEL   (86400s = 1d)
    locus_v2.csv, gateway/sward/adjacency CSV, Deep Space checkpoint meta.
    거의 바뀌지 않음. 변경 시 세션 재시작 또는 clear_cache() 로 무효화.

SUMMARY_INDEX   (600s = 10m)
    summary_index.json 기반 일별 KPI 추이.
    재처리 후 집계 재실행 주기에 맞춰 10분.

LLM_CACHE       (3600s = 1h)
    LLM 코멘터리/요약 결과 (cached_daily_summary, cached_weekly_trend_analysis).
    결정적 입력(context dict) → 동일 결과이므로 길게 유지.

LLM_INSIGHT     (600s = 10m)
    예측·이상탐지 같이 자주 재실행되는 LLM 해석 (cached_prediction_insight,
    cached_anomaly_insight, cached_spatial_insight).

WEATHER         (86400s = 1d)
    OpenWeather/기상청 일별 데이터. 하루 동안 바뀌지 않음.

변경 이력
--------
2026-04-18: 초기 버전 (M4 성능 튜닝).
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# 일별 parquet — 같은 날짜 재로드 많음, 파일 교체 드묾
# ═══════════════════════════════════════════════════════════════════
DAILY_PARQUET: int = 3600  # 1 hour

# ═══════════════════════════════════════════════════════════════════
# 다일 집계 — I/O 무거움, 재처리 시 무효화
# ═══════════════════════════════════════════════════════════════════
MULTI_DAY_AGG: int = 1800  # 30 minutes

# ═══════════════════════════════════════════════════════════════════
# 실시간 상태 — 파이프라인 실행 결과 빠르게 반영
# ═══════════════════════════════════════════════════════════════════
STATUS: int = 60  # 1 minute

# ═══════════════════════════════════════════════════════════════════
# 공간 모델 — 설정 파일, 거의 불변
# ═══════════════════════════════════════════════════════════════════
SPATIAL_MODEL: int = 86400  # 1 day

# ═══════════════════════════════════════════════════════════════════
# summary_index — KPI 추이, 재처리 주기
# ═══════════════════════════════════════════════════════════════════
SUMMARY_INDEX: int = 600  # 10 minutes

# ═══════════════════════════════════════════════════════════════════
# LLM 코멘터리 — 결정적 결과, 오래 유지
# ═══════════════════════════════════════════════════════════════════
LLM_CACHE: int = 3600  # 1 hour

# ═══════════════════════════════════════════════════════════════════
# LLM 인사이트/예측 해석 — 재실행 많음
# ═══════════════════════════════════════════════════════════════════
LLM_INSIGHT: int = 600  # 10 minutes

# ═══════════════════════════════════════════════════════════════════
# 외부 API (날씨 등) — 하루 단위 변경
# ═══════════════════════════════════════════════════════════════════
WEATHER: int = 86400  # 1 day


# ─── 메타: 정책 키 → 초 ──────────────────────────────────────────
POLICY: dict[str, int] = {
    "DAILY_PARQUET": DAILY_PARQUET,
    "MULTI_DAY_AGG": MULTI_DAY_AGG,
    "STATUS":        STATUS,
    "SPATIAL_MODEL": SPATIAL_MODEL,
    "SUMMARY_INDEX": SUMMARY_INDEX,
    "LLM_CACHE":     LLM_CACHE,
    "LLM_INSIGHT":   LLM_INSIGHT,
    "WEATHER":       WEATHER,
}


def describe() -> dict[str, int]:
    """현재 정책 스냅샷 (관리자 UI/로그용)."""
    return dict(POLICY)


__all__ = [
    "DAILY_PARQUET",
    "MULTI_DAY_AGG",
    "STATUS",
    "SPATIAL_MODEL",
    "SUMMARY_INDEX",
    "LLM_CACHE",
    "LLM_INSIGHT",
    "WEATHER",
    "POLICY",
    "describe",
]
