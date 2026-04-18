"""
Locus Tokenizer — Journey 시퀀스 토큰화
======================================
locus.csv에서 Locus ID를 로드하고 정수 토큰으로 변환.

핵심 개념:
  - Locus = 단어(Word), Journey = 문장(Sentence)
  - 특수 토큰: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
  - locus_id → token_id 양방향 매핑

사용법:
    tokenizer = LocusTokenizer.from_locus_csv(locus_path)
    tokens = tokenizer.encode(["GW-351", "GW-233", "GW-172"])  # v2 locus_id
    loci = tokenizer.decode(tokens)

    tokenizer.save(output_path)
    tokenizer = LocusTokenizer.load(output_path)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class LocusTokenizer:
    """
    Locus ID → 정수 토큰 변환기.

    특수 토큰:
      [PAD] = 0  (패딩)
      [UNK] = 1  (미확인 Locus)
      [CLS] = 2  (시퀀스 시작)
      [SEP] = 3  (구분자)
      [MASK] = 4 (MLM 마스킹)

    Locus 토큰: 5번부터 시작
    """

    def __init__(self, config: DeepSpaceConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self._locus_to_id: dict[str, int] = {}
        self._id_to_locus: dict[int, str] = {}
        self._locus_info: dict[str, dict] = {}  # locus_id → 메타정보

        # 특수 토큰 초기화
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """특수 토큰 등록."""
        for token, token_id in self.config.special_tokens.items():
            self._locus_to_id[token] = token_id
            self._id_to_locus[token_id] = token

    @classmethod
    def from_locus_csv(
        cls,
        locus_path: str | Path,
        config: DeepSpaceConfig | None = None,
    ) -> "LocusTokenizer":
        """
        locus.csv에서 토크나이저 생성.

        Args:
            locus_path: locus.csv 파일 경로
            config: 모델 설정 (선택)

        Returns:
            초기화된 LocusTokenizer
        """
        tokenizer = cls(config)

        locus_path = Path(locus_path)
        if not locus_path.exists():
            raise FileNotFoundError(f"locus.csv not found: {locus_path}")

        df = pd.read_csv(locus_path)

        if "locus_id" not in df.columns:
            raise ValueError("locus_id 컬럼이 없습니다")

        # 특수 토큰 다음 ID부터 시작
        next_id = max(tokenizer.config.special_token_ids) + 1

        for _, row in df.iterrows():
            locus_id = str(row["locus_id"]).strip()
            if locus_id and locus_id not in tokenizer._locus_to_id:
                tokenizer._locus_to_id[locus_id] = next_id
                tokenizer._id_to_locus[next_id] = locus_id

                # 메타정보 저장
                tokenizer._locus_info[locus_id] = {
                    "token": row.get("token", ""),
                    "locus_name": row.get("locus_name", ""),
                    "locus_type": row.get("locus_type", ""),
                    "hazard_level": row.get("hazard_level", ""),
                }
                next_id += 1

        logger.info(
            f"LocusTokenizer 초기화: {len(tokenizer._locus_to_id) - 5}개 Locus, "
            f"vocab_size={tokenizer.vocab_size}"
        )
        return tokenizer

    # ─── 인코딩/디코딩 ──────────────────────────────────────────────

    def encode(
        self,
        locus_sequence: Iterable[str],
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        """
        Locus ID 시퀀스 → 토큰 ID 시퀀스.

        Args:
            locus_sequence: Locus ID 리스트
            add_special_tokens: [CLS], [SEP] 추가 여부
            max_length: 최대 길이 (None이면 config.max_seq_len)
            padding: PAD로 패딩 여부
            truncation: 최대 길이 초과 시 자르기

        Returns:
            토큰 ID 리스트
        """
        max_len = max_length or self.config.max_seq_len
        tokens = []

        # CLS 토큰
        if add_special_tokens:
            tokens.append(self.config.cls_token_id)

        # Locus → 토큰 ID
        for locus_id in locus_sequence:
            locus_id = str(locus_id).strip()
            token_id = self._locus_to_id.get(locus_id, self.config.unk_token_id)
            tokens.append(token_id)

        # SEP 토큰
        if add_special_tokens:
            tokens.append(self.config.sep_token_id)

        # Truncation
        if truncation and len(tokens) > max_len:
            tokens = tokens[:max_len]
            # SEP 유지
            if add_special_tokens:
                tokens[-1] = self.config.sep_token_id

        # Padding
        if padding and len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens.extend([self.config.pad_token_id] * pad_len)

        return tokens

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        토큰 ID 시퀀스 → Locus ID 시퀀스.

        Args:
            token_ids: 토큰 ID 리스트
            skip_special_tokens: 특수 토큰 제외 여부

        Returns:
            Locus ID 리스트
        """
        result = []
        special_ids = set(self.config.special_token_ids)

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            locus_id = self._id_to_locus.get(token_id, self.config.unk_token)
            result.append(locus_id)

        return result

    def encode_batch(
        self,
        sequences: list[list[str]],
        **kwargs,
    ) -> list[list[int]]:
        """배치 인코딩."""
        return [self.encode(seq, **kwargs) for seq in sequences]

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        **kwargs,
    ) -> list[list[str]]:
        """배치 디코딩."""
        return [self.decode(ids, **kwargs) for ids in batch_ids]

    # ─── 토큰 정보 ──────────────────────────────────────────────────

    def get_locus_info(self, locus_id: str) -> dict:
        """Locus 메타정보 조회."""
        return self._locus_info.get(locus_id, {})

    def locus_to_token_id(self, locus_id: str) -> int:
        """단일 Locus → 토큰 ID."""
        return self._locus_to_id.get(locus_id, self.config.unk_token_id)

    def token_id_to_locus(self, token_id: int) -> str:
        """단일 토큰 ID → Locus."""
        return self._id_to_locus.get(token_id, self.config.unk_token)

    @property
    def vocab_size(self) -> int:
        """실제 vocabulary 크기."""
        return len(self._locus_to_id)

    @property
    def locus_ids(self) -> list[str]:
        """등록된 Locus ID 목록 (특수 토큰 제외)."""
        return [
            locus for locus in self._locus_to_id.keys()
            if locus not in self.config.special_tokens
        ]

    @property
    def locus_token_ids(self) -> list[int]:
        """Locus 토큰 ID 목록 (특수 토큰 제외)."""
        return [
            tid for tid in self._id_to_locus.keys()
            if tid not in self.config.special_token_ids
        ]

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, locus_id: str) -> bool:
        return locus_id in self._locus_to_id

    # ─── 저장/로드 ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        토크나이저를 JSON으로 저장.

        저장 위치: data/deep_space/{sector_id}/tokenizer.json
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "vocab_size": self.vocab_size,
            "locus_to_id": self._locus_to_id,
            "locus_info": self._locus_info,
            "config": {
                "pad_token_id": self.config.pad_token_id,
                "unk_token_id": self.config.unk_token_id,
                "cls_token_id": self.config.cls_token_id,
                "sep_token_id": self.config.sep_token_id,
                "mask_token_id": self.config.mask_token_id,
                "max_seq_len": self.config.max_seq_len,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"토크나이저 저장: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LocusTokenizer":
        """JSON에서 토크나이저 로드."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"토크나이저 파일 없음: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Config 복원
        cfg_data = data.get("config", {})
        config = DeepSpaceConfig(
            pad_token_id=cfg_data.get("pad_token_id", 0),
            unk_token_id=cfg_data.get("unk_token_id", 1),
            cls_token_id=cfg_data.get("cls_token_id", 2),
            sep_token_id=cfg_data.get("sep_token_id", 3),
            mask_token_id=cfg_data.get("mask_token_id", 4),
            max_seq_len=cfg_data.get("max_seq_len", 512),
        )

        tokenizer = cls(config)

        # 매핑 복원
        locus_to_id = data.get("locus_to_id", {})
        for locus_id, token_id in locus_to_id.items():
            tokenizer._locus_to_id[locus_id] = int(token_id)
            tokenizer._id_to_locus[int(token_id)] = locus_id

        tokenizer._locus_info = data.get("locus_info", {})

        logger.info(f"토크나이저 로드: {path} (vocab_size={tokenizer.vocab_size})")
        return tokenizer


# ─── 편의 함수 ────────────────────────────────────────────────────


def get_tokenizer(
    sector_id: str,
    deep_space_dir: Path | None = None,
) -> LocusTokenizer | None:
    """
    저장된 토크나이저 로드 (없으면 None).

    Args:
        sector_id: Sector ID (예: "Y1_SKHynix")
        deep_space_dir: Deep Space 모델 디렉토리 (기본: config.DEEP_SPACE_DIR)

    Returns:
        LocusTokenizer 또는 None
    """
    import config as cfg

    if deep_space_dir is None:
        deep_space_dir = getattr(cfg, "DEEP_SPACE_DIR", cfg.DATA_DIR / "deep_space")

    tokenizer_path = deep_space_dir / sector_id / "tokenizer.json"
    if tokenizer_path.exists():
        try:
            return LocusTokenizer.load(tokenizer_path)
        except Exception as e:
            logger.warning(f"토크나이저 로드 실패: {e}")
    return None


def create_tokenizer(
    sector_id: str,
    locus_csv_path: Path | None = None,
    deep_space_dir: Path | None = None,
    save: bool = True,
) -> LocusTokenizer:
    """
    locus.csv에서 토크나이저 생성 및 저장.

    Args:
        sector_id: Sector ID
        locus_csv_path: locus.csv 경로 (기본: config에서 조회)
        deep_space_dir: 저장 디렉토리
        save: 저장 여부

    Returns:
        새 LocusTokenizer
    """
    import config as cfg

    if locus_csv_path is None:
        paths = cfg.get_sector_paths(sector_id)
        locus_csv_path = paths["locus_csv"]

    tokenizer = LocusTokenizer.from_locus_csv(locus_csv_path)

    if save:
        if deep_space_dir is None:
            deep_space_dir = getattr(cfg, "DEEP_SPACE_DIR", cfg.DATA_DIR / "deep_space")
        tokenizer_path = deep_space_dir / sector_id / "tokenizer.json"
        tokenizer.save(tokenizer_path)

    return tokenizer
