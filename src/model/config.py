"""
Deep Space Model Configuration
==============================
모델 하이퍼파라미터 및 학습 설정.

설계 원칙:
  - v2 (2026-04-11): MPS GPU 활용, Locus v2 vocab 218 기준
  - Streamlit Cloud 추론: CPU에서도 동작 (float32)
  - 로컬 학습: Apple MPS GPU (MacBook M-series)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeepSpaceConfig:
    """Deep Space 모델 하이퍼파라미터."""

    # ─── Vocabulary ─────────────────────────────────────────────
    # v2: 213 GW-XXX Locus + 5 특수 토큰 = 218
    vocab_size: int = 218

    # 특수 토큰 ID
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4

    # 특수 토큰 문자열
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"

    # ─── Transformer Architecture ───────────────────────────────
    # v2 mid: MPS GPU 활용, BERT-tiny 수준 (3.4M params)
    d_model: int = 256           # 임베딩 차원 (v1: 64)
    n_heads: int = 8             # 어텐션 헤드 수
    n_layers: int = 4            # Transformer 레이어 수
    d_ff: int = 1024             # FFN 중간 차원 (v1: 256)
    dropout: float = 0.1
    max_seq_len: int = 256       # 최대 시퀀스 길이 (v2 Locus 다양화로 증가)

    # ─── MLM Pre-training ───────────────────────────────────────
    mlm_probability: float = 0.15  # 마스킹 비율
    mlm_replace_prob: float = 0.8  # [MASK]로 교체 확률
    mlm_random_prob: float = 0.1   # 랜덤 토큰 교체 확률
    mlm_keep_prob: float = 0.1     # 원본 유지 확률

    # ─── Training ───────────────────────────────────────────────
    # v2: MPS GPU → lr 낮춤, 배치 크기 증가, 샘플 상한 4배 확장
    learning_rate: float = 3e-4    # 모델 커짐 → lr 보수적으로
    weight_decay: float = 0.01
    batch_size: int = 512          # MPS GPU → 큰 배치 OK
    epochs: int = 20               # GPU이므로 에포크 늘림
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    early_stop_patience: int = 5
    min_delta: float = 0.001
    max_train_samples: int = 200000  # v2: 40일 315K 중 200K 샘플링

    # ─── Checkpoint ─────────────────────────────────────────────
    save_every_n_epochs: int = 5
    keep_top_k_checkpoints: int = 3

    # ─── Inference ──────────────────────────────────────────────
    top_k_prediction: int = 5     # Top-K 예측 반환
    # v2 실측 기반: Median=3.9, P75=6.3, P90=9.1 → threshold=10 ≈ 상위 8% 이상 탐지
    anomaly_threshold: float = 10.0  # perplexity 이상 탐지 임계값 (v2 실측 재조정)

    @property
    def special_token_ids(self) -> list[int]:
        """특수 토큰 ID 목록."""
        return [
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        ]

    @property
    def special_tokens(self) -> dict[str, int]:
        """특수 토큰 문자열 → ID 매핑."""
        return {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id,
        }


# ─── 기본 설정 인스턴스 ─────────────────────────────────────────
DEFAULT_CONFIG = DeepSpaceConfig()


def get_config(**overrides) -> DeepSpaceConfig:
    """커스텀 설정 생성."""
    return DeepSpaceConfig(**overrides)
