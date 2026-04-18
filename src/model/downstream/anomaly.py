"""
Anomaly Detector — Journey 이상 탐지
====================================
Perplexity 기반 이동 패턴 이상 탐지.

핵심 개념:
  - Perplexity = exp(평균 Cross-Entropy Loss)
  - 높은 perplexity = 모델이 예측하기 어려운 시퀀스 = 비정상 패턴
  - 임계값 초과 시 이상 탐지 플래그

사용법:
    from src.model.downstream.anomaly import AnomalyDetector

    detector = AnomalyDetector(model, tokenizer)
    score = detector.compute_perplexity(journey_sequence)
    is_anomaly = detector.is_anomaly(journey_sequence)

    # 배치 처리
    scores = detector.score_batch(sequences)
    anomalies = detector.detect_anomalies(sequences)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG
from src.model.transformer import DeepSpaceModel
from src.model.tokenizer import LocusTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """이상 탐지 결과."""
    user_no: str
    perplexity: float
    is_anomaly: bool
    anomaly_score: float  # 0~1 정규화 점수
    sequence_length: int
    details: dict | None = None


class AnomalyDetector:
    """
    Journey 이상 탐지기.

    방법:
      1. 각 위치에서 실제 토큰의 예측 확률 계산
      2. 평균 NLL (Negative Log-Likelihood) 계산
      3. Perplexity = exp(NLL)
      4. 임계값 초과 시 이상

    해석 (v2 실측 기반, 40일 Y1_SKHynix):
      - Perplexity 1~4:  정상 (Median=3.9)
      - Perplexity 4~7:  경계 (P75=6.3)
      - Perplexity 7~10: 주의 (P90=9.1)
      - Perplexity 10~15: 이상 (상위 ~8%, 기본 임계값=10)
      - Perplexity 15+:  강한 이상 (P95=12.2 초과)
    """

    def __init__(
        self,
        model: DeepSpaceModel,
        tokenizer: LocusTokenizer,
        config: DeepSpaceConfig | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model: 학습된 DeepSpaceModel
            tokenizer: LocusTokenizer
            config: 설정
            threshold: 이상 탐지 임계값 (None이면 config.anomaly_threshold)
            device: "cpu" 또는 "cuda"
        """
        self.config = config or DEFAULT_CONFIG
        self.threshold = threshold or self.config.anomaly_threshold

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
    def compute_perplexity(
        self,
        sequence: list[str],
        mask_chunk_size: int = 8,
    ) -> float:
        """
        MLM Pseudo-Perplexity 계산 (BERT-style).

        BERT는 causal LM이 아니므로 각 위치를 [MASK]로 대체한 뒤
        해당 위치에서 원래 토큰의 확률을 계산하는 pseudo-perplexity 사용.
        효율을 위해 chunk_size 개 위치를 동시에 마스킹하여 배치 처리.

        Args:
            sequence: Locus ID 시퀀스
            mask_chunk_size: 한 번에 마스킹할 위치 수 (메모리/속도 트레이드오프)

        Returns:
            Pseudo-Perplexity (낮을수록 정상, 높을수록 비정상)
        """
        if len(sequence) < 2:
            return 1.0

        input_ids = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )

        special_ids = set(self.config.special_token_ids)
        mask_id = self.config.mask_token_id

        # 마스킹 가능한 위치 (locus 토큰만)
        locus_positions = [
            i for i, tid in enumerate(input_ids)
            if tid not in special_ids
        ]
        if not locus_positions:
            return 1.0

        total_nll = 0.0
        n_valid = 0

        # chunk 단위로 배치 처리: chunk_size개 위치씩 각각 [MASK] 처리 후 배치 forward
        for chunk_start in range(0, len(locus_positions), mask_chunk_size):
            chunk_positions = locus_positions[chunk_start: chunk_start + mask_chunk_size]

            # 각 위치별 마스킹된 시퀀스 생성
            batch_ids = []
            for pos in chunk_positions:
                masked = list(input_ids)
                masked[pos] = mask_id
                batch_ids.append(masked)

            # 배치 forward
            input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            attn_mask = torch.ones_like(input_tensor)

            outputs = self.model(input_tensor, attn_mask)
            logits = outputs["logits"]  # [chunk, seq_len, vocab_size]

            for batch_i, pos in enumerate(chunk_positions):
                target_id = input_ids[pos]
                probs = F.softmax(logits[batch_i, pos], dim=-1)
                target_prob = probs[target_id].item()
                total_nll += -np.log(max(target_prob, 1e-10))
                n_valid += 1

        if n_valid == 0:
            return 1.0

        avg_nll = total_nll / n_valid
        return float(np.exp(avg_nll))

    def is_anomaly(
        self,
        sequence: list[str],
        threshold: float | None = None,
    ) -> bool:
        """
        이상 여부 판정.

        Args:
            sequence: Locus ID 시퀀스
            threshold: 커스텀 임계값

        Returns:
            True = 이상
        """
        threshold = threshold or self.threshold
        perplexity = self.compute_perplexity(sequence)
        return perplexity > threshold

    def score(
        self,
        sequence: list[str],
        user_no: str = "",
    ) -> AnomalyResult:
        """
        이상 탐지 점수 및 결과.

        Args:
            sequence: Locus ID 시퀀스
            user_no: 작업자 ID (선택)

        Returns:
            AnomalyResult
        """
        perplexity = self.compute_perplexity(sequence)

        # 0~1 정규화 (sigmoid 스타일)
        # perplexity 10 → 0.5, 20 → 0.73, 50 → 0.92
        anomaly_score = 1.0 - (1.0 / (1.0 + np.exp((perplexity - self.threshold) / 5)))

        return AnomalyResult(
            user_no=user_no,
            perplexity=perplexity,
            is_anomaly=perplexity > self.threshold,
            anomaly_score=float(anomaly_score),
            sequence_length=len(sequence),
        )

    def score_batch(
        self,
        sequences: list[list[str]],
        user_ids: list[str] | None = None,
    ) -> list[AnomalyResult]:
        """
        배치 이상 탐지.

        Args:
            sequences: Locus ID 시퀀스 리스트
            user_ids: 작업자 ID 리스트 (선택)

        Returns:
            AnomalyResult 리스트
        """
        if user_ids is None:
            user_ids = [""] * len(sequences)

        results = []
        for seq, uid in zip(sequences, user_ids):
            results.append(self.score(seq, uid))
        return results

    def detect_anomalies(
        self,
        sequences: list[list[str]],
        user_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[AnomalyResult]:
        """
        이상 시퀀스만 반환 (점수 높은 순).

        Args:
            sequences: 시퀀스 리스트
            user_ids: 작업자 ID 리스트
            top_k: 상위 K개만 반환 (None = 전체)

        Returns:
            이상 탐지된 AnomalyResult 리스트 (점수 내림차순)
        """
        results = self.score_batch(sequences, user_ids)

        # 이상만 필터링
        anomalies = [r for r in results if r.is_anomaly]

        # 점수 내림차순 정렬
        anomalies.sort(key=lambda r: r.perplexity, reverse=True)

        if top_k is not None:
            anomalies = anomalies[:top_k]

        return anomalies

    @torch.no_grad()
    def get_position_scores(
        self,
        sequence: list[str],
    ) -> list[tuple[str, float]]:
        """
        각 위치별 이상 점수 (MLM pseudo-perplexity, 위치별 NLL).

        Returns:
            [(locus_id, nll_score), ...] 리스트
        """
        if len(sequence) < 2:
            return []

        input_ids = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )

        special_ids = set(self.config.special_token_ids)
        mask_id = self.config.mask_token_id

        locus_positions = [
            i for i, tid in enumerate(input_ids)
            if tid not in special_ids
        ]

        results = []
        # 각 위치를 하나씩 마스킹하여 NLL 계산
        for pos in locus_positions:
            masked = list(input_ids)
            masked[pos] = mask_id
            input_tensor = torch.tensor([masked], dtype=torch.long, device=self.device)
            attn_mask = torch.ones_like(input_tensor)
            outputs = self.model(input_tensor, attn_mask)
            logits = outputs["logits"][0]

            target_id = input_ids[pos]
            probs = F.softmax(logits[pos], dim=-1)
            target_prob = probs[target_id].item()
            nll = -np.log(max(target_prob, 1e-10))

            locus_id = self.tokenizer.token_id_to_locus(target_id)
            results.append((locus_id, float(nll)))

        return results


# ─── 편의 함수 ────────────────────────────────────────────────────


def get_detector(
    sector_id: str,
    threshold: float | None = None,
) -> AnomalyDetector | None:
    """
    학습된 모델로 AnomalyDetector 생성.

    Args:
        sector_id: Sector ID
        threshold: 이상 임계값

    Returns:
        AnomalyDetector 또는 None
    """
    from src.model.trainer import load_trained_model
    from src.model.tokenizer import get_tokenizer

    model = load_trained_model(sector_id)
    if model is None:
        return None

    tokenizer = get_tokenizer(sector_id)
    if tokenizer is None:
        return None

    return AnomalyDetector(model, tokenizer, threshold=threshold)


def detect_journey_anomalies(
    sector_id: str,
    sequences: list[list[str]],
    user_ids: list[str] | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[AnomalyResult] | None:
    """
    편의 함수: Journey 이상 탐지.

    Args:
        sector_id: Sector ID
        sequences: Locus ID 시퀀스 리스트
        user_ids: 작업자 ID 리스트
        threshold: 이상 임계값
        top_k: 상위 K개만 반환

    Returns:
        AnomalyResult 리스트 또는 None
    """
    detector = get_detector(sector_id, threshold)
    if detector is None:
        return None
    return detector.detect_anomalies(sequences, user_ids, top_k)


def compute_journey_perplexity(
    sector_id: str,
    sequence: list[str],
) -> float | None:
    """
    편의 함수: 단일 시퀀스 perplexity.

    Args:
        sector_id: Sector ID
        sequence: Locus ID 시퀀스

    Returns:
        Perplexity 또는 None
    """
    detector = get_detector(sector_id)
    if detector is None:
        return None
    return detector.compute_perplexity(sequence)
