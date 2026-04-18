"""
DeepCon Project Configuration
==============================
모든 경로, 상수, 설정을 중앙 관리
"""
import os
from pathlib import Path

# ★ .env 파일 로드 (로컬 환경, config.py import 시 최초 1회)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH, override=True)
    except ImportError:
        # dotenv 없으면 수동 파싱
        with open(_ENV_PATH) as _f:
            for _line in _f:
                _line = _line.strip()
                if "=" in _line and not _line.startswith("#"):
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())


def _get_secret(key: str, default: str = "") -> str:
    """환경변수 → Streamlit secrets → default 순으로 조회."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

# ─── 배포 모드 ───────────────────────────────────────────────────
# Streamlit Cloud 배포 시 True → 파이프라인 숨김, Drive에서 데이터 로드
CLOUD_MODE = _get_secret("CLOUD_MODE", "false").lower() == "true"

# ─── 기본 경로 ─────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPATIAL_DIR   = DATA_DIR / "spatial_model"
INDEX_DIR     = DATA_DIR / "index"          # ★ Summary Index (날짜별 KPI 요약)
MODEL_DIR     = DATA_DIR / "model"          # ★ Journey 임베딩/클러스터링 모델
DEEP_SPACE_DIR = DATA_DIR / "deep_space"    # ★ Deep Space Transformer 모델

# ─── Raw 파일 패턴 ─────────────────────────────────────────────────
RAW_DATE_FORMAT  = "%Y%m%d"
RAW_ENCODING     = "cp949"

# ─── Processed 파일 패턴 ───────────────────────────────────────────
PROCESSED_ENCODING = "utf-8"

# ─── 데이터 처리 상수 ──────────────────────────────────────────────
MIN_SIGNAL_COUNT           = 1      # 최소 신호 수
MIN_ACTIVE_RATIO           = 0.0    # 최소 활성 비율
WORK_START_HOUR            = 5      # 05:00 이후 유효
WORK_END_HOUR              = 23     # 23:00 이전
JOURNEY_CORRECTION_ENABLED     = True   # Phase 1(슬라이딩 윈도우) + Phase 2(DBSCAN) 활성
JOURNEY_TOKENIZATION_ENABLED   = True   # Journey 의미 블록 토큰화 활성
DATA_VALIDATION_ENABLED        = True   # 데이터 정합성 검증 활성
GAP_ANALYSIS_ENABLED           = True   # BLE 음영 탐지·채우기 + 활성 레벨 분류 (M15X 이식)
JOURNEY_RECONSTRUCTION_ENABLED = True   # AccessLog 기반 BLE Coverage 계산 (M15X 이식)
COVERAGE_THRESHOLD             = 0.30   # Deep Space 훈련 포함 최소 BLE Coverage (30%)

# ─── Locus 버전 ──────────────────────────────────────────────────────
# "v1" = 기존 58개 Locus (spot_name_map.json 기반)
# "v2" = 신규 213개 Locus (S-Ward 좌표 기반)
LOCUS_VERSION = "v2"  # v2: S-Ward 좌표 기반 213개 Locus

# ─── Sector 레지스트리 ─────────────────────────────────────────────
# 새 Sector 추가 시 이 딕셔너리에만 등록하면 됨
SECTOR_REGISTRY: dict[str, dict] = {
    "Y1_SKHynix": {
        "label":    "SK하이닉스 Y1",
        "subtitle": "반도체 클러스터 건설현장",
        "icon":     "🏗️",
        "domain":   "construction",
        "status":   "active",          # "active" | "coming_soon"
        "access_log_prefix": "Y1_AccessLog",
        "tward_prefix":      "Y1_TWardData",
    },
    "M15X_SKHynix": {
        "label":    "SK하이닉스 M15X",
        "subtitle": "반도체 FAB 건설현장",
        "icon":     "🏭",
        "domain":   "construction",
        "status":   "active",
        "access_log_prefix": "M15X_AccessLog",
        "tward_prefix":      "M15X_TWardData",
    },
    # 추후 추가 예정 (status="coming_soon" → 사이드바에 잠금 표시)
    # "LotteMart_Singal": {
    #     "label": "롯데마트 신갈점", "subtitle": "유통 유동인구 분석",
    #     "icon": "🛒", "domain": "retail", "status": "coming_soon",
    # },
    # "ICN_Airport_T2": {
    #     "label": "인천공항 T2", "subtitle": "공항 공간 분석",
    #     "icon": "✈️", "domain": "airport", "status": "coming_soon",
    # },
}

# ─── 사용자(계정) 레지스트리 ────────────────────────────────────────
# sectors=None → 모든 Sector 접근 가능 (admin)
# sectors=[...] → 해당 Sector만 접근 가능 (client)
#
# ★ 비밀번호 정책 (Upgrade v3 M1, 2026-04-18):
#   - 평문 필드 `password` 는 제거됨.
#   - argon2id 해시만 사용한다: `.env` / Streamlit secrets 의
#     `DEEPCON_{ADMIN|Y1|M15X}_PW_HASH` 키를 src.dashboard.auth._USER_PWHASH_ENV
#     가 읽어 검증.
#   - DEPRECATED: `DEEPCON_*_PASSWORD` env 변수는 더 이상 참조되지 않는다.
#     Streamlit Cloud Secrets 에 남아 있으면 삭제할 것.
#   - 해시 생성: `scripts/rotate_passwords.py` 또는 `argon2.PasswordHasher().hash(plain)`

USER_REGISTRY: dict[str, dict] = {
    "administrator": {
        "label":    "Administrator",
        "icon":     "⚙️",
        "role":     "admin",
        "sectors":  None,              # None = 전체 Sector
    },
    "Y1_SKHynix": {
        "label":    "SK하이닉스 Y1",
        "icon":     "🏗️",
        "role":     "client",
        "sectors":  ["Y1_SKHynix"],
    },
    "M15X_SKHynix": {
        "label":    "SK하이닉스 M15X",
        "icon":     "🏭",
        "role":     "client",
        "sectors":  ["M15X_SKHynix"],
    },
}

# ─── Sector별 경로 헬퍼 ────────────────────────────────────────────
def get_sector_paths(sector_id: str) -> dict:
    """
    Sector ID → 경로 딕셔너리 반환.

    반환 키:
        raw_dir, processed_dir, spatial_dir,
        ssmp_dir, locus_dir, adjacency_dir,
        spot_name_map, locus_csv, locus_xlsx, adjacency
    """
    raw_dir     = RAW_DIR       / sector_id
    proc_dir    = PROCESSED_DIR / sector_id
    spatial_dir = SPATIAL_DIR   / sector_id
    locus_dir   = spatial_dir   / "locus"
    adj_dir     = spatial_dir   / "adjacency"
    return {
        "raw_dir":       raw_dir,
        "processed_dir": proc_dir,
        "spatial_dir":   spatial_dir,
        "ssmp_dir":      spatial_dir / "ssmp",
        "locus_dir":     locus_dir,
        "adjacency_dir": adj_dir,
        "spot_name_map": locus_dir   / "spot_name_map.json",
        "locus_csv":     locus_dir   / ("locus_v2.csv" if LOCUS_VERSION == "v2" else "locus.csv"),
        "locus_xlsx":    locus_dir   / "Locus.xlsx",
        "adjacency":     adj_dir     / "locus_adjacency.csv",
        # v2 경로
        "locus_v2_csv":  locus_dir   / "locus_v2.csv",
        "gateway_csv":   spatial_dir / "New_SSMP_20260403" / "SWard_data" / "Y1_GatewayPoint_20260403.csv",
        "locus_map_xlsx": locus_dir  / "Locus_SWard_Map.xlsx",
    }


def get_allowed_sectors_for_user(user_id: str) -> list[str]:
    """해당 사용자가 접근 가능한 활성 Sector 목록."""
    user    = USER_REGISTRY.get(user_id, {})
    allowed = user.get("sectors")                        # None = admin
    active  = [sid for sid, info in SECTOR_REGISTRY.items()
               if info.get("status") == "active"]
    if allowed is None:
        return active                                     # 전체 접근
    return [s for s in allowed if s in SECTOR_REGISTRY
            and SECTOR_REGISTRY[s].get("status") == "active"]


# ─── 하위 호환 (기존 코드가 cfg.XXX_SECTOR_DIR 등을 직접 쓰는 경우 대비) ──
# 새 코드는 get_sector_paths() 사용 권장
SECTOR_ID            = "Y1_SKHynix"
SECTOR_LABEL         = "SK하이닉스 Y1 반도체 클러스터"
RAW_SECTOR_DIR       = RAW_DIR       / SECTOR_ID
PROCESSED_SECTOR_DIR = PROCESSED_DIR / SECTOR_ID
SPATIAL_SECTOR_DIR   = SPATIAL_DIR   / SECTOR_ID
SSMP_DIR             = SPATIAL_SECTOR_DIR / "ssmp"
LOCUS_DIR            = SPATIAL_SECTOR_DIR / "locus"
ADJACENCY_DIR        = SPATIAL_SECTOR_DIR / "adjacency"
SPOT_NAME_MAP_FILE   = LOCUS_DIR / "spot_name_map.json"
LOCUS_CSV_FILE       = LOCUS_DIR / "locus.csv"
LOCUS_XLSX_FILE      = LOCUS_DIR / "Locus.xlsx"
ADJACENCY_FILE       = ADJACENCY_DIR / "locus_adjacency.csv"

# ─── Dashboard 설정 ────────────────────────────────────────────────
APP_TITLE     = "DeepCon"
APP_SUBTITLE  = "Agentic AI based on Spatial Data"
APP_ICON      = "🌐"
THEME_PRIMARY = "#1A3A5C"
THEME_ACCENT  = "#00AEEF"
THEME_BG      = "#0D1B2A"
THEME_CARD_BG = "#1A2A3A"
THEME_TEXT    = "#C8D6E8"
THEME_SUCCESS = "#00C897"
THEME_WARNING = "#FFB300"
THEME_DANGER  = "#FF4C4C"

# ─── 리포트 설정 ───────────────────────────────────────────────────
REPORT_OUTPUT_DIR = BASE_DIR / "output" / "reports"
WEEKLY_REPORT_DAY = 0   # 0=월요일

# ─── LLM 백엔드 설정 ─────────────────────────────────────────────
# "anthropic" (기본값) 또는 "bedrock" (AWS Bedrock)
LLM_BACKEND        = _get_secret("LLM_BACKEND", "anthropic")
AWS_REGION         = _get_secret("AWS_REGION", "ap-northeast-2")
AWS_BEDROCK_MODEL_ID = _get_secret("AWS_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-6-v1")

# ─── 익명화 설정 ─────────────────────────────────────────────────
# True(기본값): LLM 호출 전 작업자/업체/구역명 마스킹
ANONYMIZE_LLM = _get_secret("ANONYMIZE_LLM", "true").lower() == "true"
# True(기본값): LLM 호출 전 알고리즘 공식도 추상화
ANONYMIZE_LOGIC = _get_secret("ANONYMIZE_LOGIC", "true").lower() == "true"

# ─── 보안 모드 ───────────────────────────────────────────────────
# True(기본값): 방법론 탭에서 상세 알고리즘 숨김 (배포용)
# False: 개발 환경에서 상세 표시 (디버깅용)
SECURE_MODE = _get_secret("SECURE_MODE", "true").lower() == "true"

# ─── 공간 용량 기본값 (locus_type별) ────────────────────────────────
# 모든 모듈이 이 상수를 import하여 fallback capacity를 결정한다.
DEFAULT_CAPACITY_BY_TYPE: dict[str, int] = {
    "GATE": 50,
    "WORK": 100,
    "REST": 30,
    "HAZARD": 20,
    "ADMIN": 20,
    "FACILITY": 30,
    "TRANSPORT": 15,
}

# ─── 혼잡도 등급 임계값 (capacity 대비 비율, 0~1+) ─────────────────
# congestion_tab.py, locus_prediction.py 등에서 공통 사용.
# spatial_predictor.py의 CONGESTION_THRESHOLDS는 Agentic AI alert용으로 별도 유지.
CONGESTION_GRADE_THRESHOLDS: dict[str, float] = {
    "과밀": 1.0,   # >= 100%
    "혼잡": 0.8,   # >= 80%
    "보통": 0.6,   # >= 60%
    # 그 외 = "여유" (< 60%)
}

# ─── 버전 ──────────────────────────────────────────────────────────
APP_VERSION   = "0.1.0"
CACHE_VERSION = "v1"

# ─── 캐시 스키마 버전 (Upgrade v3 T-06) ─────────────────────────────
# 각 Parquet 산출물의 스키마 버전을 중앙 관리.
# 컬럼 추가/삭제/타입 변경 시 해당 키의 숫자를 +1 하면 자동으로 재처리 트리거.
#
# 정책:
#   - meta.json 에 `schema_versions` 필드로 기록 (save_daily_results)
#   - 로드 시 cache_manager.validate_schema()가 불일치 검출 → SchemaVersionMismatch
#   - `schema_versions` 없는 기존 meta는 "legacy"로 관용 처리 (warn만, 강제 재처리 X)
#   - CLI `reprocess --incremental` 이 schema mismatch를 재처리 조건 중 하나로 사용
#
# 현재 버전 기준선 (2026-04-18 수립):
#   journey 40 cols, worker 45 cols, space 9, company 10, coverage 7
#   향후 컬럼 변경 시 해당 키 bump.
CACHE_SCHEMA_VERSION: dict[str, int] = {
    "journey":  5,   # 현재 journey.parquet 스키마 (40개 컬럼)
    "worker":   8,   # worker.parquet (45개 컬럼) — v8 (2026-04-18): EWI 분모 정책 변경 (항상 work_minutes)
    "space":    3,   # space.parquet (9개 컬럼)
    "company":  3,   # company.parquet (10개 컬럼)
    "coverage": 2,   # coverage.parquet (7개 컬럼)
    "meta":     4,   # meta.json 키 구조 (validation DSL 포함 형식)
}
