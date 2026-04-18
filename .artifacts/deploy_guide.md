# DeepCon — GitHub Public + Streamlit Cloud 배포 가이드

> **날짜**: 2026-04-18
> **리팩터링**: pipeline IP 분리 완료 (v3.0.1)

---

## ✅ 리팩터링 결과

### Before (불가능했던 이유)
| 파일 | 문제 | 유형 |
|------|------|------|
| `daily/productivity.py:21` | `from src.pipeline.metrics import 상수6개` — 모듈 시작 시 강제 로드 | Top-level import |
| `zone_time_tab.py:38` | `from src.pipeline.metrics import _get_token_sets` — 동일 | Top-level import |
| `pipeline_tab.py:23-24` | `from src.pipeline.processor/loader import` — 동일 | Top-level import |
| `integrity/sanity_check.py` | `check_worker_sanity` 소급 계산 호출 | Runtime IP 호출 |
| `integrity/physical_validator.py` | `annotate_journey` 등 재계산 호출 | Runtime IP 호출 |

### After (해결)
| 수정 | 방법 |
|------|------|
| `daily/productivity.py` | `metric_constants.py`(신규) 에서 상수만 import — pipeline 의존 0 |
| `zone_time_tab.py` | `token_utils.py`(신규) 에서 locus CSV 읽기 함수 사용 — sanity_checker 인라인화 |
| `pipeline_tab.py` | `if not CLOUD_MODE:` 가드 — CLOUD_MODE 시 Drive Sync UI만 표시 |
| `integrity/sanity_check.py` | `_cloud` 가드 — CLOUD_MODE 시 parquet 컬럼만 읽음 |
| `integrity/physical_validator.py` | CLOUD_MODE 시 즉시 return + 안내 메시지 |

### Release pipeline/ 잔존 파일 (IP 아님)
```
src/pipeline/
├── __init__.py
├── cache_manager.py    ← parquet 읽기/쓰기 + schema 검증 (유틸)
├── summary_index.py    ← summary_index.json 읽기 (유틸)
├── drive_storage.py    ← Drive 다운로드 클라이언트
└── drive_uploader.py   ← Drive 업로드 유틸
```

---

## 📦 GitHub Public 배포 절차

### 1. git repo 크기 확인
`.gitignore`에 의해 제외되는 폴더:
- `data/processed/` (1.9 GB)
- `data/index/` (108 MB)
- `data/spatial_model/` (93 MB)
- `data/deep_space/` (57 MB)
- `data/model/` (128 KB)
- `data/raw/` (원본 CSV)
- `data/audit/` (LLM 감사 로그)

**Git에 포함될 코드**: ~13 MB ← 1 GB 제한 안전

### 2. GitHub 설정
```bash
cd /Users/tony/Desktop/TJLABS/TJLABS_Research/Release/DeepCon
git init
git add .
git commit -m "DeepCon v3.0.1 — Agentic AI based on Spatial Data"
# GitHub에서 신규 Public repo 생성 후:
git remote add origin https://github.com/tony-tjlabs/{REPO_NAME}.git
git push -u origin main
```

### 3. IP 검증 (커밋 전)
```bash
# pipeline IP 파일이 없는지 확인
ls src/pipeline/  # 4개만 있어야 함: __init__, cache_manager, summary_index, drive_storage*, drive_uploader

# dashboard 내 직접 pipeline import (CLOUD_MODE 가드 외부)가 없는지 확인
grep -rn "from src.pipeline.metrics\|from src.pipeline.processor\|from src.pipeline.loader\|from src.pipeline.sward_mapper" \
  src/dashboard/ --include="*.py"
# → 결과 0건이어야 함 (남은 건 모두 CLOUD_MODE 가드 내부)
```

---

## ☁️ Streamlit Cloud 배포 절차

### 1. Google Drive 준비

#### 폴더 구조
```
Google Drive (Service Account 공유):
├── DeepCon_Y1_SKHynix/              ← DRIVE_FOLDER_ID_Y1
│   ├── Y1_SKHynix_20260309_worker.parquet
│   ├── Y1_SKHynix_20260309_space.parquet
│   ├── Y1_SKHynix_20260309_company.parquet
│   ├── Y1_SKHynix_20260309_coverage.parquet
│   ├── Y1_SKHynix_20260309_meta.json
│   ├── model__Y1_SKHynix__w2v_model.pkl
│   └── ...
└── DeepCon_M15X_SKHynix/           ← DRIVE_FOLDER_ID_M15X
    ├── M15X_SKHynix_20260309_worker.parquet
    └── ...
```

**주의**: `journey.parquet` 는 35MB/일 × 40일 = 1.4GB → 업로드 제외 (`_SKIP_PATTERNS`에 이미 설정)

#### Service Account 생성 (최초 1회)
1. Google Cloud Console → IAM → Service Account 생성
2. JSON 키 다운로드
3. Drive 폴더를 SA 이메일에 공유 (Editor 권한)

### 2. 데이터 업로드 (로컬에서)
```bash
# 로컬 환경에서 처리된 데이터를 Drive에 업로드
cd SandBox/DeepCon
python -m src.pipeline.drive_uploader upload-all --sector Y1_SKHynix
python -m src.pipeline.drive_uploader upload-all --sector M15X_SKHynix
```

### 3. Streamlit Cloud Secrets 설정
```toml
# .streamlit/secrets.toml (Cloud Secrets UI에 입력)

# 배포 모드
CLOUD_MODE = "true"

# 비밀번호 해시 (argon2id — 평문 금지!)
DEEPCON_ADMIN_PW_HASH = "argon2id$v=19$..."
DEEPCON_Y1_PW_HASH = "argon2id$v=19$..."
DEEPCON_M15X_PW_HASH = "argon2id$v=19$..."

# Anthropic API
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-haiku-4-5"

# Google Drive (SA JSON의 각 필드를 분리 저장)
GDRIVE_SA_TYPE = "service_account"
GDRIVE_SA_PROJECT_ID = "..."
GDRIVE_SA_PRIVATE_KEY_ID = "..."
GDRIVE_SA_PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\n..."
GDRIVE_SA_CLIENT_EMAIL = "...@....iam.gserviceaccount.com"
GDRIVE_SA_CLIENT_ID = "..."
GDRIVE_SA_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
GDRIVE_SA_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Drive 폴더 ID (각 Sector별)
DRIVE_FOLDER_ID_Y1_SKHynix = "1ABC..."
DRIVE_FOLDER_ID_M15X_SKHynix = "1DEF..."
```

### 4. 앱 동작 흐름
```
Streamlit Cloud 앱 시작
  └─ main.py: _init_drive_cache()
       └─ CLOUD_MODE=True → DriveStorage 초기화
       └─ sync_all() → Drive에서 processed/index/model 다운로드
       └─ Streamlit 캐시 (/tmp/ or mounted 볼륨)에 저장
       └─ 이후 탭: cache_manager.load_daily() → 로컬 캐시 읽기
```

---

## 📊 용량 정리

| 위치 | 크기 | 비고 |
|------|------|------|
| GitHub repo | ~13 MB | 1 GB 제한 여유 |
| Google Drive (업로드) | ~260 MB | processed(worker/space/company/coverage) × 45일 + model |
| Streamlit Cloud 런타임 | ~260 MB | Drive에서 동기화 (journey.parquet 제외) |

**journey.parquet 제외 이유**: 35MB/일 × 45일 = 1.6GB → 렌더링에 불필요 (worker/space/company/coverage로 모든 UI 구동 가능)

---

## 🔐 IP 보호 정리

| 분류 | 파일 | GitHub |
|------|------|--------|
| **제외 (IP)** | `processor.py`, `metrics.py`, `loader.py`, `sward_mapper.py` | `.gitignore` 또는 미포함 |
| **제외 (IP)** | `corrector.py`, `tokenizer.py`, `gap_analyzer.py`, `sanity_checker.py`, `physical_validator.py` | 동일 |
| **포함 (유틸)** | `cache_manager.py`, `summary_index.py` | Parquet 읽기 — 알고리즘 없음 |
| **포함 (Drive)** | `drive_storage.py`, `drive_uploader.py` | Drive 연동 코드 |
| **포함 (UI)** | `src/dashboard/**`, `core/**`, `main.py`, `config.py` | 전체 UI + AI Gateway |
