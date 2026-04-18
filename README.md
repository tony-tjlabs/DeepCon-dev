# DeepCon v3.0.1 — Release

Agentic AI based on Spatial Data. 건설현장 BLE/T-Ward 이동 데이터 대시보드 + AI 코멘터리 + 감사 로그.

## 활성 Sector

- `Y1_SKHynix` — SK하이닉스 Y1 (40일, 2026-03-01 ~ 04-09)
- `M15X_SKHynix` — SK하이닉스 M15X (5일, 2026-03-09 ~ 03-13)

## 배포 — Streamlit Cloud

1. 이 폴더를 새 Git 저장소로 푸시.
2. Streamlit Cloud에서 저장소 연결 → `main.py` 지정.
3. Secrets에 아래 키 주입:

```
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL   = "claude-haiku-4-5"
CLOUD_MODE        = "true"

# argon2id 해시 (scripts/rotate_passwords.py 로 생성)
DEEPCON_ADMIN_PW_HASH = "$argon2id$v=19$m=65536,t=3,p=2$..."
DEEPCON_Y1_PW_HASH    = "$argon2id$v=19$m=65536,t=3,p=2$..."
DEEPCON_M15X_PW_HASH  = "$argon2id$v=19$m=65536,t=3,p=2$..."

# 사용자 번호 해싱 salt (랜덤 32자+)
DEEPCON_USER_NO_SALT  = "..."

# Google Drive 동기화 (CLOUD_MODE=true 필요)
[gcp_service_account]
type = "service_account"
# ... (SA JSON 전체 필드)
```

샘플: `secrets.toml.example`

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run main.py --server.port 8530
```

## 구성

| 영역 | 경로 |
|---|---|
| UI (8탭) | `src/dashboard/` |
| AI 게이트웨이 + 익명화 | `core/ai/`, `core/security/` |
| 데이터 처리 | `src/pipeline/` (CLI 제외) |
| 학습 모델 로딩 | `src/model/`, `src/intelligence/` |
| 캐시 데이터 | `data/processed/`, `data/index/`, `data/deep_space/`, `data/spatial_model/` |

## 비밀번호 교체

```bash
python scripts/rotate_passwords.py
```

argon2id 해시를 출력 → Streamlit Cloud Secrets의 `DEEPCON_*_PW_HASH` 에 붙여넣기.

## 안전

- 민감 파일 (`.env`, `token.json`, SA 키, 평문 `secrets.toml`) 전부 `.gitignore` 처리.
- Raw CSV (`data/raw/`) 배포 제외 — CLOUD_MODE=true 시 Drive 동기화.
- 모든 LLM 호출은 `core.ai.LLMGateway` 경유 (K=20 익명화 강제).
- 감사 로그 `data/audit/{sector}/{YYYY-MM}.jsonl` (PII 저장 금지, hash만).

## 버전

v3.0.1 (2026-04-18) — EWI 분모 정책 수정 (`work_minutes` 고정), worker schema 7→8, Y1 40일 + M15X 5일 전수 재처리.
