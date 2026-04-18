"""
Deep Space Transformer Model
============================
Journey 시퀀스 학습을 위한 Transformer Encoder 아키텍처.

아키텍처:
  Input: [CLS] L1 L2 L3 ... Ln [SEP]
    |
    v
  [Locus Embedding + Positional Encoding]
    |
    v
  [Transformer Encoder] x N layers
    |
    v
  [MLM Head] -> vocab_size logits

핵심 설계:
  - BERT 스타일 Bidirectional Encoder
  - 학습 가능한 Positional Embedding
  - 경량 모델 (~10MB): 4 layers, 4 heads, d_model=128

사용법:
    config = DeepSpaceConfig()
    model = DeepSpaceModel(config, vocab_size=63)

    # Forward
    outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]  # [batch, seq_len, vocab_size]
"""
from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    학습 가능한 Positional Embedding.

    BERT 스타일: 각 위치에 대해 학습 가능한 임베딩 벡터.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # 위치 ID 버퍼 (0, 1, 2, ..., max_len-1)
        self.register_buffer(
            "position_ids",
            torch.arange(max_len).unsqueeze(0),  # [1, max_len]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        seq_len = x.size(1)
        position_ids = self.position_ids[:, :seq_len]  # [1, seq_len]
        position_embeddings = self.position_embeddings(position_ids)  # [1, seq_len, d_model]
        x = x + position_embeddings
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    단일 Transformer Encoder Layer.

    구조:
      1. Multi-Head Self-Attention + Residual + LayerNorm
      2. Feed-Forward Network + Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: [seq_len, seq_len] attention mask
            key_padding_mask: [batch, seq_len] PAD mask (True = ignore)

        Returns:
            [batch, seq_len, d_model]
        """
        # Self-Attention + Residual + Norm
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.attn_dropout(attn_out)
        x = self.attn_norm(x)

        # FFN + Residual + Norm
        ff_out = self.ff(x)
        x = x + self.ff_dropout(ff_out)
        x = self.ff_norm(x)

        return x


class DeepSpaceModel(nn.Module):
    """
    Deep Space Foundation Model.

    Journey 시퀀스 → MLM 예측.

    구조:
      - Locus Embedding Layer
      - Positional Encoding
      - Transformer Encoder Stack
      - MLM Head (Linear → vocab_size)

    모델 크기 (기본 설정):
      vocab_size=100, d_model=128, n_layers=4, n_heads=4
      → 약 5~10MB (50MB 이하 목표 충족)
    """

    def __init__(
        self,
        config: DeepSpaceConfig | None = None,
        vocab_size: int | None = None,
    ) -> None:
        """
        Args:
            config: 모델 설정 (None이면 기본값)
            vocab_size: Vocabulary 크기 (None이면 config.vocab_size)
        """
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        # vocab_size 결정
        self._vocab_size = vocab_size or self.config.vocab_size

        # Locus Embedding
        self.locus_embedding = nn.Embedding(
            num_embeddings=self._vocab_size,
            embedding_dim=self.config.d_model,
            padding_idx=self.config.pad_token_id,
        )

        # Positional Encoding
        self.position_encoding = PositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )

        # Transformer Encoder Stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])

        # MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self._vocab_size),
        )

        # 가중치 초기화
        self._init_weights()

        logger.info(
            f"DeepSpaceModel 초기화: vocab_size={self._vocab_size}, "
            f"d_model={self.config.d_model}, n_layers={self.config.n_layers}, "
            f"n_heads={self.config.n_heads}, params={self.num_parameters:,}"
        )

    def _init_weights(self) -> None:
        """가중치 초기화 (Xavier/He)."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_parameters(self) -> int:
        """총 파라미터 수."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """학습 가능한 파라미터 수."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] 토큰 ID
            attention_mask: [batch, seq_len] 1=유효, 0=PAD
            output_hidden_states: hidden states 반환 여부

        Returns:
            {
                "logits": [batch, seq_len, vocab_size],
                "last_hidden_state": [batch, seq_len, d_model],
                "hidden_states": list (optional),
            }
        """
        # Embedding
        x = self.locus_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.position_encoding(x)

        # key_padding_mask: PAD 위치 = True (무시)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # [batch, seq_len]

        # Transformer Encoder
        hidden_states = []
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            if output_hidden_states:
                hidden_states.append(x)

        # MLM Head
        logits = self.mlm_head(x)  # [batch, seq_len, vocab_size]

        outputs = {
            "logits": logits,
            "last_hidden_state": x,
        }
        if output_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        시퀀스 임베딩 추출 ([CLS] 토큰 벡터).

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            [batch, d_model] 시퀀스 임베딩
        """
        outputs = self.forward(input_ids, attention_mask)
        # [CLS] 토큰 위치 (index 0)의 hidden state
        return outputs["last_hidden_state"][:, 0, :]

    def predict_masked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        마스킹된 위치의 예측 확률 분포.

        Args:
            input_ids: [batch, seq_len] ([MASK] 포함)
            attention_mask: [batch, seq_len]

        Returns:
            [batch, seq_len, vocab_size] 확률 분포
        """
        outputs = self.forward(input_ids, attention_mask)
        return F.softmax(outputs["logits"], dim=-1)

    def save_pretrained(self, path: str) -> None:
        """모델 저장."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self._vocab_size,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "d_ff": self.config.d_ff,
                "dropout": self.config.dropout,
                "max_seq_len": self.config.max_seq_len,
                "pad_token_id": self.config.pad_token_id,
            },
        }, path)
        logger.info(f"모델 저장: {path}")

    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "DeepSpaceModel":
        """저장된 모델 로드."""
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {path}")

        # ★ 보안 고려사항:
        #   weights_only=True가 이상적 (pickle 역직렬화 공격 방지)이나,
        #   체크포인트에 config dict가 포함되어 weights_only=True가 실패할 수 있음.
        #   현재는 로컬 파일만 로드하므로 보안 위험은 낮으나,
        #   외부/원격 체크포인트를 로드할 경우 반드시 weights_only=True 전용
        #   저장 형식으로 마이그레이션 필요 (config를 별도 JSON으로 분리).
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            # 기존 체크포인트 호환 (config dict 포함 — pickle 역직렬화 필요)
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        cfg_data = checkpoint["config"]

        config = DeepSpaceConfig(
            d_model=cfg_data.get("d_model", 128),
            n_heads=cfg_data.get("n_heads", 4),
            n_layers=cfg_data.get("n_layers", 4),
            d_ff=cfg_data.get("d_ff", 512),
            dropout=cfg_data.get("dropout", 0.1),
            max_seq_len=cfg_data.get("max_seq_len", 512),
            pad_token_id=cfg_data.get("pad_token_id", 0),
        )

        model = cls(config=config, vocab_size=cfg_data.get("vocab_size", 100))
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"모델 로드: {path}")
        return model


# ─── MLM Loss 계산 ───────────────────────────────────────────────


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    MLM Cross-Entropy Loss.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] (-100 = 무시)
        ignore_index: 무시할 label 값

    Returns:
        스칼라 loss
    """
    # [batch * seq_len, vocab_size]
    logits_flat = logits.view(-1, logits.size(-1))
    # [batch * seq_len]
    labels_flat = labels.view(-1)

    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
    )


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    top_k: int = 1,
) -> float:
    """
    MLM Top-K 정확도 계산.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        ignore_index: 무시할 label 값
        top_k: Top-K 정확도 (1, 3, 5)

    Returns:
        정확도 (0~1)
    """
    # 마스킹된 위치만 평가
    mask = labels != ignore_index
    if not mask.any():
        return 0.0

    masked_logits = logits[mask]  # [n_masked, vocab_size]
    masked_labels = labels[mask]  # [n_masked]

    # Top-K 예측
    _, topk_preds = masked_logits.topk(top_k, dim=-1)  # [n_masked, top_k]

    # 정답이 Top-K에 포함되는지 확인
    correct = (topk_preds == masked_labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
