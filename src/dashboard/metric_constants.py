"""
Dashboard Metric Constants
==========================
파이프라인 IP 코드와 분리된 지표 상수 모음.

이전에는 src.pipeline.metrics 에서 직접 import 했으나,
CLOUD_MODE(Streamlit Cloud 배포) 환경에서는 pipeline 코드가 없으므로
UI 레이어에서 독립적으로 상수를 유지한다.

values 는 config.py / src/pipeline/metrics.py 와 반드시 동기화할 것.
"""

# ─── T-Ward 활성 임계값 ─────────────────────────────────────────────
# BLE 신호 기반 작업 활성도 분류 기준 (건설현장 공통)
HIGH_ACTIVE_THRESHOLD: float = 0.90   # 고활성: active_ratio >= 90% → 집중 작업
LOW_ACTIVE_THRESHOLD:  float = 0.40   # 저활성 하한: 이 미만은 대기로 분류

# ─── EWI 가중치 ─────────────────────────────────────────────────────
# EWI = (고활성×W_H + 저활성×W_L + 대기×W_S) / work_minutes
HIGH_WORK_WEIGHT: float = 1.0    # 고활성 작업
LOW_WORK_WEIGHT:  float = 0.5    # 저활성 작업
STANDBY_WEIGHT:   float = 0.2    # 대기 (크레인 대기·감독 등 실질 작업 포함)

# ─── EWI 신뢰도 임계값 ──────────────────────────────────────────────
# gap_ratio 가 이 값 이하이면 ewi_reliable=True (UI 참고용)
# gap_ratio = (work_minutes - recorded_work_min) / work_minutes
EWI_GAP_RELIABLE_THRESHOLD: float = 0.20  # 20% 이상 음영이면 unreliable 배지
