"""
Journey MLM Dataset — Masked Language Model 학습 데이터셋
========================================================
journey.parquet에서 작업자별 Locus 시퀀스를 추출하고,
MLM(Masked Language Modeling) 방식으로 마스킹.

핵심 개념:
  - 시퀀스: [CLS] + locus_ids + [SEP]
  - 15% 랜덤 마스크 (80% [MASK], 10% 랜덤, 10% 유지)
  - attention_mask로 PAD 처리

사용법:
    from src.model.dataset import JourneyMLMDataset, create_dataloader

    dataset = JourneyMLMDataset(
        tokenizer=tokenizer,
        journey_parquets=[path1, path2],
    )
    dataloader = create_dataloader(dataset, batch_size=32)
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG
from src.model.tokenizer import LocusTokenizer

logger = logging.getLogger(__name__)


class JourneyMLMDataset(Dataset):
    """
    Journey 시퀀스를 MLM 학습용으로 변환하는 Dataset.

    각 샘플:
      - input_ids: 마스킹된 토큰 시퀀스
      - attention_mask: PAD 위치 = 0, 나머지 = 1
      - labels: 마스킹된 위치만 원본 토큰, 나머지 = -100

    마스킹 전략 (BERT 스타일):
      - 선택된 토큰의 80%: [MASK]로 교체
      - 선택된 토큰의 10%: 랜덤 토큰으로 교체
      - 선택된 토큰의 10%: 원본 유지

    최적화:
      - 초기화 시 전 시퀀스 사전 토큰화 → numpy 배열 캐시 (epoch 반복 재계산 제거)
      - MLM 마스킹 numpy 배치 연산 (Python for 루프 제거)
      - locus_token_ids numpy 배열 캐시 (내부 루프 매번 리스트 생성 제거)
    """

    def __init__(
        self,
        tokenizer: LocusTokenizer,
        journey_parquets: list[Path] | None = None,
        sequences: list[list[str]] | None = None,
        config: DeepSpaceConfig | None = None,
        min_seq_len: int = 5,
    ) -> None:
        """
        Args:
            tokenizer: LocusTokenizer 인스턴스
            journey_parquets: journey.parquet 파일 경로 리스트
            sequences: 미리 추출된 Locus ID 시퀀스 리스트 (parquet 대신 사용)
            config: 모델 설정
            min_seq_len: 최소 시퀀스 길이 (미만은 무시)
        """
        self.tokenizer = tokenizer
        self.config = config or DEFAULT_CONFIG
        self.min_seq_len = min_seq_len

        # 시퀀스 로드
        if sequences is not None:
            raw_sequences = [s for s in sequences if len(s) >= min_seq_len]
        elif journey_parquets:
            raw_sequences = self._load_sequences(journey_parquets)
        else:
            raise ValueError("journey_parquets 또는 sequences 필요")

        # ── 사전 토큰화 (핵심 최적화) ──────────────────────────────
        # tokenizer.encode()를 epoch마다 반복하지 않고 한 번만 수행.
        # 결과를 int16 numpy 배열(max_seq_len 고정)로 저장 → 메모리 절약 + 빠른 슬라이싱.
        logger.info(f"사전 토큰화 시작: {len(raw_sequences):,}개 시퀀스...")
        max_len = self.config.max_seq_len
        n = len(raw_sequences)
        # int16: 0~218 범위 충분, 절반 메모리
        self._token_cache = np.zeros((n, max_len), dtype=np.int16)
        self._attn_cache  = np.zeros((n, max_len), dtype=np.int8)
        pad_id = self.config.pad_token_id

        for i, seq in enumerate(raw_sequences):
            ids = tokenizer.encode(
                seq,
                add_special_tokens=True,
                max_length=max_len,
                padding=True,
                truncation=True,
            )
            self._token_cache[i] = ids
            self._attn_cache[i]  = [0 if t == pad_id else 1 for t in ids]

        # 특수 토큰 마스크 (마스킹 불가 위치) — 한 번 계산
        special_set = set(self.config.special_token_ids)
        self._special_mask = np.array(
            [1 if t in special_set else 0 for t in range(max(special_set) + 1)],
            dtype=bool,
        )

        # locus 토큰 ID numpy 배열 캐시 (랜덤 교체용)
        self._locus_token_arr = np.array(tokenizer.locus_token_ids, dtype=np.int32)

        logger.info(f"JourneyMLMDataset 초기화 완료: {n:,}개 시퀀스 "
                    f"(캐시 {self._token_cache.nbytes / 1e6:.1f}MB)")

    def _load_sequences(self, parquet_paths: list[Path]) -> list[list[str]]:
        """journey.parquet에서 작업자별 Locus 시퀀스 추출 (벡터화 최적화)."""
        sequences = []

        for i, path in enumerate(parquet_paths):
            path = Path(path)
            if not path.exists():
                logger.warning(f"파일 없음: {path}")
                continue

            try:
                df = pd.read_parquet(
                    path,
                    columns=["user_no", "locus_id", "seq"],
                )
            except Exception as e:
                logger.warning(f"파일 로드 실패 ({path}): {e}")
                continue

            if "locus_id" not in df.columns:
                continue

            # NaN 제거 후 정렬 (벡터화)
            df = df.dropna(subset=["locus_id"])
            df["locus_id"] = df["locus_id"].astype(str)
            df = df.sort_values(["user_no", "seq"])

            # 연속 중복 제거 (벡터화): 이전 행과 같은 locus면 제거
            mask = (df["locus_id"] != df["locus_id"].shift()) | (df["user_no"] != df["user_no"].shift())
            df = df[mask]

            # groupby → 리스트 변환 (벡터화)
            grouped = df.groupby("user_no", sort=False)["locus_id"].agg(list)
            for locus_list in grouped:
                if len(locus_list) >= self.min_seq_len:
                    sequences.append(locus_list)

            logger.info(f"[{i+1}/{len(parquet_paths)}] {path.parent.name}: {len(grouped)}명 → {sum(1 for l in grouped if len(l) >= self.min_seq_len)}개 시퀀스")

        return sequences

    def __len__(self) -> int:
        return len(self._token_cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        단일 샘플 반환 (사전 토큰화 캐시 사용, MLM만 on-the-fly).

        Returns:
            {
                "input_ids": Tensor[max_seq_len],
                "attention_mask": Tensor[max_seq_len],
                "labels": Tensor[max_seq_len],
            }
        """
        # 캐시에서 int32로 읽기 (MLM 연산용)
        input_ids = self._token_cache[idx].astype(np.int32)
        attention_mask = self._attn_cache[idx]

        # MLM 마스킹 (numpy 배치 연산)
        masked_ids, labels = self._apply_mlm_mask_numpy(input_ids)

        return {
            "input_ids": torch.from_numpy(masked_ids).long(),
            "attention_mask": torch.from_numpy(attention_mask.astype(np.int32)).long(),
            "labels": torch.from_numpy(labels).long(),
        }

    def _apply_mlm_mask_numpy(
        self,
        input_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        numpy 배치 연산으로 MLM 마스킹 적용 (Python for 루프 없음).

        Returns:
            (masked_ids, labels) — 각 shape [max_seq_len], dtype int32
        """
        seq_len = len(input_ids)
        labels = np.full(seq_len, -100, dtype=np.int32)
        masked_ids = input_ids.copy()

        # 마스킹 가능 위치: PAD·특수 토큰 제외
        # special_mask는 특수 토큰 ID 범위 내만 커버, 범위 밖은 마스킹 가능
        spec_max = len(self._special_mask)
        is_special = np.zeros(seq_len, dtype=bool)
        in_range = input_ids < spec_max
        is_special[in_range] = self._special_mask[input_ids[in_range]]
        maskable = np.where(~is_special)[0]

        if len(maskable) == 0:
            return masked_ids, labels

        # 마스킹 대상 무작위 선택
        n_mask = max(1, int(len(maskable) * self.config.mlm_probability))
        chosen = np.random.choice(maskable, size=min(n_mask, len(maskable)), replace=False)

        # 원본 저장
        labels[chosen] = input_ids[chosen]

        # 교체 전략 결정 (배치 난수)
        rands = np.random.rand(len(chosen))
        mask_flag   = rands < self.config.mlm_replace_prob
        random_flag = (rands >= self.config.mlm_replace_prob) & \
                      (rands < self.config.mlm_replace_prob + self.config.mlm_random_prob)

        # 80% → [MASK]
        masked_ids[chosen[mask_flag]] = self.config.mask_token_id

        # 10% → 랜덤 locus 토큰
        n_random = random_flag.sum()
        if n_random > 0 and len(self._locus_token_arr) > 0:
            masked_ids[chosen[random_flag]] = np.random.choice(
                self._locus_token_arr, size=n_random
            )
        # 나머지 10% → 원본 유지 (이미 copy이므로 그대로)

        return masked_ids, labels


class JourneySequenceDataset(Dataset):
    """
    추론용 Dataset (마스킹 없음).

    각 샘플:
      - input_ids: 토큰 시퀀스
      - attention_mask: PAD 마스크
      - user_no: 작업자 ID (선택)
    """

    def __init__(
        self,
        tokenizer: LocusTokenizer,
        sequences: list[list[str]],
        user_ids: list[str] | None = None,
        config: DeepSpaceConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.user_ids = user_ids or [""] * len(sequences)
        self.config = config or DEFAULT_CONFIG

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.sequences[idx]

        input_ids = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
            max_length=self.config.max_seq_len,
            padding=True,
            truncation=True,
        )

        attention_mask = [
            0 if tid == self.config.pad_token_id else 1
            for tid in input_ids
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "user_no": self.user_ids[idx],
        }


# ─── DataLoader 생성 함수 ───────────────────────────────────────────


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    DataLoader 생성.

    Args:
        dataset: JourneyMLMDataset 또는 JourneySequenceDataset
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수 (Mac에서는 0 권장)
        pin_memory: GPU 메모리 고정

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def load_journey_sequences(
    sector_id: str,
    dates: list[str],
    processed_dir: Path | None = None,
) -> list[list[str]]:
    """
    복수 날짜의 journey.parquet에서 시퀀스 추출.

    Args:
        sector_id: Sector ID
        dates: YYYYMMDD 형식 날짜 리스트
        processed_dir: processed 디렉토리

    Returns:
        Locus ID 시퀀스 리스트
    """
    import config as cfg

    if processed_dir is None:
        processed_dir = cfg.PROCESSED_DIR

    parquet_paths = []
    for date_str in dates:
        path = processed_dir / sector_id / date_str / "journey.parquet"
        if path.exists():
            parquet_paths.append(path)

    if not parquet_paths:
        logger.warning(f"journey.parquet 없음: {sector_id}, dates={dates}")
        return []

    sequences = []
    for path in parquet_paths:
        try:
            df = pd.read_parquet(
                path,
                columns=["user_no", "locus_id", "seq"],
            )
            # 벡터화 시퀀스 추출 (groupby + tolist 대신)
            df = df.dropna(subset=["locus_id"])
            df = df.sort_values(["user_no", "seq"])
            df["locus_id"] = df["locus_id"].astype(str)

            # 연속 중복 제거 (같은 locus에 계속 있으면 하나로)
            df["prev_locus"] = df.groupby("user_no")["locus_id"].shift(1)
            df = df[df["locus_id"] != df["prev_locus"]].drop(columns=["prev_locus"])

            # 작업자별 시퀀스 길이 필터 (5 이상만)
            counts = df.groupby("user_no").size()
            valid_users = counts[counts >= 5].index
            df = df[df["user_no"].isin(valid_users)]

            # 빠른 시퀀스 변환
            for user_no, grp in df.groupby("user_no", sort=False):
                sequences.append(grp["locus_id"].tolist())
        except Exception as e:
            logger.warning(f"시퀀스 로드 실패 ({path}): {e}")

    return sequences


def create_mlm_dataset(
    sector_id: str,
    dates: list[str],
    tokenizer: LocusTokenizer,
    config: DeepSpaceConfig | None = None,
) -> JourneyMLMDataset:
    """
    MLM 학습용 Dataset 생성 (편의 함수).

    Args:
        sector_id: Sector ID
        dates: 날짜 리스트
        tokenizer: LocusTokenizer
        config: 모델 설정

    Returns:
        JourneyMLMDataset
    """
    sequences = load_journey_sequences(sector_id, dates)
    return JourneyMLMDataset(
        tokenizer=tokenizer,
        sequences=sequences,
        config=config,
    )
