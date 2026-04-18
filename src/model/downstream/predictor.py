"""
Next Locus Predictor — 다음 이동 위치 예측
==========================================
마지막 N개 Locus 기반으로 다음 이동 위치 확률 분포 예측.

핵심 개념:
  - 현재 시퀀스 끝에 [MASK] 추가 → 모델이 예측
  - Top-K 예측 반환
  - 전이 확률 매트릭스 생성

사용법:
    from src.model.downstream.predictor import NextLocusPredictor

    predictor = NextLocusPredictor(model, tokenizer)
    probs = predictor.predict_next(["GW-351", "GW-233"], top_k=5)  # v2 locus_id
    # [("GW-172", 0.35), ("GW-241", 0.22), ...]

    # 전이 확률 매트릭스
    matrix = predictor.get_transition_matrix()
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG
from src.model.transformer import DeepSpaceModel
from src.model.tokenizer import LocusTokenizer

logger = logging.getLogger(__name__)


class NextLocusPredictor:
    """
    다음 Locus 예측기.

    방법:
      1. 입력 시퀀스: [CLS] L1 L2 ... Ln [MASK] [SEP]
      2. 모델이 [MASK] 위치의 토큰 예측
      3. 확률 분포에서 Top-K 반환
    """

    def __init__(
        self,
        model: DeepSpaceModel,
        tokenizer: LocusTokenizer,
        config: DeepSpaceConfig | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model: 학습된 DeepSpaceModel
            tokenizer: LocusTokenizer
            config: 설정 (top_k 등)
            device: "cpu" 또는 "cuda"
        """
        self.config = config or DEFAULT_CONFIG

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS는 Placeholder storage 에러 등 불안정 → CPU 사용
                device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def predict_next(
        self,
        current_sequence: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        다음 Locus Top-K 예측.

        Args:
            current_sequence: 현재까지의 Locus ID 리스트
            top_k: 반환할 상위 K개 (기본: config.top_k_prediction)

        Returns:
            [(locus_id, probability), ...] Top-K 리스트
        """
        top_k = top_k or self.config.top_k_prediction

        if not current_sequence:
            return []

        # 시퀀스 + [MASK] 구성
        # [CLS] L1 L2 ... Ln [MASK] [SEP]
        input_ids = [self.config.cls_token_id]
        for locus_id in current_sequence:
            tid = self.tokenizer.locus_to_token_id(locus_id)
            input_ids.append(tid)
        mask_position = len(input_ids)  # [MASK] 위치
        input_ids.append(self.config.mask_token_id)
        input_ids.append(self.config.sep_token_id)

        # 텐서 변환
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_tensor)

        # Forward
        outputs = self.model(input_tensor, attention_mask)
        logits = outputs["logits"]  # [1, seq_len, vocab_size]

        # [MASK] 위치의 logits
        mask_logits = logits[0, mask_position, :]  # [vocab_size]
        probs = F.softmax(mask_logits, dim=-1)  # [vocab_size]

        # 특수 토큰 제외하고 Top-K
        # 특수 토큰 확률을 0으로
        for special_id in self.config.special_token_ids:
            probs[special_id] = 0.0

        # Top-K
        topk_probs, topk_ids = probs.topk(top_k)
        topk_probs = topk_probs.cpu().numpy()
        topk_ids = topk_ids.cpu().numpy()

        results = []
        for prob, tid in zip(topk_probs, topk_ids):
            locus_id = self.tokenizer.token_id_to_locus(int(tid))
            if locus_id not in self.config.special_tokens:
                results.append((locus_id, float(prob)))

        return results

    @torch.no_grad()
    def predict_batch(
        self,
        sequences: list[list[str]],
        top_k: int | None = None,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """
        실제 배치 예측 (패딩 + 단일 forward pass).

        Args:
            sequences: Locus ID 시퀀스 리스트
            top_k: Top-K
            batch_size: GPU/CPU 배치 크기

        Returns:
            각 시퀀스에 대한 Top-K 리스트
        """
        top_k = top_k or self.config.top_k_prediction
        if not sequences:
            return []

        all_results: list[list[tuple[str, float]]] = []
        special_ids = set(self.config.special_token_ids)

        for batch_start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[batch_start: batch_start + batch_size]

            # 토큰화 + [MASK] 추가
            batch_ids = []
            mask_positions = []
            for seq in batch_seqs:
                if not seq:
                    batch_ids.append([self.config.cls_token_id, self.config.mask_token_id, self.config.sep_token_id])
                    mask_positions.append(1)
                else:
                    ids = [self.config.cls_token_id]
                    for lid in seq:
                        ids.append(self.tokenizer.locus_to_token_id(lid))
                    mask_positions.append(len(ids))
                    ids.append(self.config.mask_token_id)
                    ids.append(self.config.sep_token_id)
                    batch_ids.append(ids)

            # 패딩
            max_len = max(len(ids) for ids in batch_ids)
            padded = []
            attn_masks = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.config.pad_token_id] * pad_len)
                attn_masks.append([1] * len(ids) + [0] * pad_len)

            input_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            attn_tensor = torch.tensor(attn_masks, dtype=torch.long, device=self.device)

            outputs = self.model(input_tensor, attn_tensor)
            logits = outputs["logits"]  # [batch, max_len, vocab]

            for i, mask_pos in enumerate(mask_positions):
                probs = F.softmax(logits[i, mask_pos, :], dim=-1)
                for sid in special_ids:
                    probs[sid] = 0.0
                topk_probs, topk_ids = probs.topk(top_k)
                results = []
                for p, tid in zip(topk_probs.cpu().numpy(), topk_ids.cpu().numpy()):
                    lid = self.tokenizer.token_id_to_locus(int(tid))
                    if lid not in self.config.special_tokens:
                        results.append((lid, float(p)))
                all_results.append(results)

        return all_results

    @torch.no_grad()
    def get_probability(
        self,
        current_sequence: list[str],
        target_locus: str,
    ) -> float:
        """
        특정 Locus로 이동할 확률.

        Args:
            current_sequence: 현재 시퀀스
            target_locus: 예측 대상 Locus

        Returns:
            확률 (0~1)
        """
        if not current_sequence:
            return 0.0

        # 전체 예측 수행
        input_ids = [self.config.cls_token_id]
        for locus_id in current_sequence:
            input_ids.append(self.tokenizer.locus_to_token_id(locus_id))
        mask_position = len(input_ids)
        input_ids.append(self.config.mask_token_id)
        input_ids.append(self.config.sep_token_id)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        outputs = self.model(input_tensor)
        logits = outputs["logits"][0, mask_position, :]
        probs = F.softmax(logits, dim=-1)

        target_id = self.tokenizer.locus_to_token_id(target_locus)
        return float(probs[target_id].item())

    @torch.no_grad()
    def get_transition_matrix(
        self,
        context_length: int = 1,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, list[str]]:
        """
        전이 확률 매트릭스 생성 (배치 추론).

        각 Locus에서 다른 Locus로 이동할 확률 행렬.
        213개 Locus를 batch_size 단위로 묶어 한 번에 forward pass.

        Args:
            context_length: 컨텍스트 길이 (1 = 직전 Locus만 사용)
            batch_size: 배치 크기

        Returns:
            (transition_matrix, locus_ids)
            - transition_matrix: [n_locus, n_locus] 확률 행렬
            - locus_ids: 행/열 인덱스에 해당하는 Locus ID 리스트
        """
        locus_ids = self.tokenizer.locus_ids
        n_locus = len(locus_ids)
        special_ids = set(self.config.special_token_ids)

        matrix = np.zeros((n_locus, n_locus), dtype=np.float32)

        # locus_id → matrix index 매핑
        locus_to_idx = {lid: i for i, lid in enumerate(locus_ids)}
        # token_id → matrix index 매핑 (직접 인덱싱용)
        tid_to_idx = {}
        for lid in locus_ids:
            tid = self.tokenizer.locus_to_token_id(lid)
            tid_to_idx[tid] = locus_to_idx[lid]

        # 배치 단위로 전이 확률 계산
        for batch_start in range(0, n_locus, batch_size):
            batch_loci = locus_ids[batch_start: batch_start + batch_size]
            bs = len(batch_loci)

            # [CLS] Locus [MASK] [SEP] 형태로 고정 길이 4
            batch_ids = []
            for lid in batch_loci:
                tid = self.tokenizer.locus_to_token_id(lid)
                batch_ids.append([
                    self.config.cls_token_id,
                    tid,
                    self.config.mask_token_id,
                    self.config.sep_token_id,
                ])

            input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            attn_mask = torch.ones(bs, 4, dtype=torch.long, device=self.device)

            outputs = self.model(input_tensor, attn_mask)
            logits = outputs["logits"]  # [bs, 4, vocab]

            # [MASK] 위치 (index 2)의 확률
            mask_logits = logits[:, 2, :]  # [bs, vocab]
            probs = F.softmax(mask_logits, dim=-1).cpu().numpy()

            # 특수 토큰 확률 제거
            for sid in special_ids:
                probs[:, sid] = 0.0

            # 행렬에 직접 매핑
            for i_local in range(bs):
                i_global = batch_start + i_local
                for tid, j_idx in tid_to_idx.items():
                    matrix[i_global, j_idx] = probs[i_local, tid]

        # 행별 정규화 (합 = 1)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

        return matrix, locus_ids

    def get_top_transitions(
        self,
        from_locus: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        특정 Locus에서의 Top-K 이동 확률.

        Args:
            from_locus: 출발 Locus
            top_k: 상위 K개

        Returns:
            [(to_locus, probability), ...]
        """
        return self.predict_next([from_locus], top_k=top_k)


# ─── 편의 함수 ────────────────────────────────────────────────────


def get_predictor(sector_id: str) -> NextLocusPredictor | None:
    """
    학습된 모델로 Predictor 생성.

    Args:
        sector_id: Sector ID

    Returns:
        NextLocusPredictor 또는 None
    """
    from src.model.trainer import load_trained_model
    from src.model.tokenizer import get_tokenizer

    model = load_trained_model(sector_id)
    if model is None:
        return None

    tokenizer = get_tokenizer(sector_id)
    if tokenizer is None:
        return None

    return NextLocusPredictor(model, tokenizer)


def predict_next_location(
    sector_id: str,
    current_sequence: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]] | None:
    """
    편의 함수: 다음 위치 예측.

    Args:
        sector_id: Sector ID
        current_sequence: 현재 Locus 시퀀스
        top_k: Top-K

    Returns:
        [(locus_id, probability), ...] 또는 None
    """
    predictor = get_predictor(sector_id)
    if predictor is None:
        return None
    return predictor.predict_next(current_sequence, top_k)
