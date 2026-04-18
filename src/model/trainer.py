"""
Deep Space Model Trainer
========================
MLM 사전학습 및 학습 루프 관리.

기능:
  - 에포크별 학습/검증
  - Loss 및 Top-K 정확도 로깅
  - 체크포인트 저장 (best_model.pt)
  - Early Stopping
  - 학습 이력 JSON 저장

사용법:
    from src.model.trainer import DeepSpaceTrainer, train_deep_space

    # 방법 1: 편의 함수
    model = train_deep_space(sector_id, dates)

    # 방법 2: Trainer 직접 사용
    trainer = DeepSpaceTrainer(model, train_dataset, config, output_dir)
    trainer.train()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from src.model.config import DeepSpaceConfig, DEFAULT_CONFIG
from src.model.transformer import DeepSpaceModel, compute_mlm_loss, compute_accuracy
from src.model.dataset import JourneyMLMDataset, create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """학습 이력."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_acc_top1: list[float] = field(default_factory=list)
    train_acc_top3: list[float] = field(default_factory=list)
    train_acc_top5: list[float] = field(default_factory=list)
    val_acc_top1: list[float] = field(default_factory=list)
    val_acc_top3: list[float] = field(default_factory=list)
    val_acc_top5: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_train_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_acc_top1": self.train_acc_top1,
            "train_acc_top3": self.train_acc_top3,
            "train_acc_top5": self.train_acc_top5,
            "val_acc_top1": self.val_acc_top1,
            "val_acc_top3": self.val_acc_top3,
            "val_acc_top5": self.val_acc_top5,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_train_time": self.total_train_time,
        }


class DeepSpaceTrainer:
    """
    Deep Space 모델 학습기.

    특징:
      - CPU/GPU 자동 감지
      - Warmup + Linear Decay 스케줄러
      - Early Stopping
      - Gradient Clipping
      - 체크포인트 관리
    """

    def __init__(
        self,
        model: DeepSpaceModel,
        train_dataset: JourneyMLMDataset,
        config: DeepSpaceConfig | None = None,
        output_dir: str | Path = "output",
        val_split: float = 0.1,
        device: str | None = None,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> None:
        """
        Args:
            model: DeepSpaceModel 인스턴스
            train_dataset: 학습 데이터셋
            config: 학습 설정
            output_dir: 체크포인트 저장 디렉토리
            val_split: 검증 데이터 비율 (0~1)
            device: "cpu" 또는 "cuda" (None이면 자동 감지)
            progress_callback: 진행 콜백 (epoch, total_epochs, train_loss, val_loss)
        """
        self.config = config or DEFAULT_CONFIG
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device 설정 (CUDA > MPS > CPU 우선순위)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # 모델
        self.model = model.to(self.device)

        # 데이터 분할
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # DataLoader
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler (Warmup + Linear Decay)
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - warmup_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps],
        )

        # 학습 상태
        self.history = TrainingHistory()
        self.progress_callback = progress_callback
        self._early_stop_counter = 0

        logger.info(
            f"DeepSpaceTrainer 초기화: "
            f"train={train_size}, val={val_size}, "
            f"device={self.device}, epochs={self.config.epochs}"
        )

    def train(self) -> TrainingHistory:
        """
        전체 학습 루프.

        Returns:
            TrainingHistory
        """
        start_time = time.time()
        logger.info("학습 시작...")

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # 학습
            train_loss, train_metrics = self._train_epoch()

            # 검증
            val_loss, val_metrics = self._validate()

            epoch_time = time.time() - epoch_start

            # 이력 기록
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_acc_top1.append(train_metrics["top1"])
            self.history.train_acc_top3.append(train_metrics["top3"])
            self.history.train_acc_top5.append(train_metrics["top5"])
            self.history.val_acc_top1.append(val_metrics["top1"])
            self.history.val_acc_top3.append(val_metrics["top3"])
            self.history.val_acc_top5.append(val_metrics["top5"])
            self.history.learning_rates.append(self.optimizer.param_groups[0]["lr"])
            self.history.epoch_times.append(epoch_time)

            # 로깅
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Top-1: {val_metrics['top1']:.2%} | Top-3: {val_metrics['top3']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )

            # 콜백
            if self.progress_callback:
                self.progress_callback(epoch, self.config.epochs, train_loss, val_loss)

            # Best 모델 저장
            if val_loss < self.history.best_val_loss - self.config.min_delta:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                self._save_checkpoint("best_model.pt")
                self._early_stop_counter = 0
            else:
                self._early_stop_counter += 1

            # 정기 체크포인트
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch:03d}.pt")

            # Early Stopping
            if self._early_stop_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.history.total_train_time = time.time() - start_time

        # 학습 이력 저장
        self._save_history()

        logger.info(
            f"학습 완료: Best epoch={self.history.best_epoch}, "
            f"Best val_loss={self.history.best_val_loss:.4f}, "
            f"Total time={self.history.total_train_time:.1f}s"
        )

        return self.history

    def _train_epoch(self) -> tuple[float, dict]:
        """단일 에포크 학습."""
        self.model.train()
        total_loss = 0.0
        total_acc_top1 = 0.0
        total_acc_top3 = 0.0
        total_acc_top5 = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]

            loss = compute_mlm_loss(logits, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc_top1 += compute_accuracy(logits, labels, top_k=1)
            total_acc_top3 += compute_accuracy(logits, labels, top_k=3)
            total_acc_top5 += compute_accuracy(logits, labels, top_k=5)
            n_batches += 1

        return (
            total_loss / max(n_batches, 1),
            {
                "top1": total_acc_top1 / max(n_batches, 1),
                "top3": total_acc_top3 / max(n_batches, 1),
                "top5": total_acc_top5 / max(n_batches, 1),
            },
        )

    @torch.no_grad()
    def _validate(self) -> tuple[float, dict]:
        """검증."""
        self.model.eval()
        total_loss = 0.0
        total_acc_top1 = 0.0
        total_acc_top3 = 0.0
        total_acc_top5 = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]

            loss = compute_mlm_loss(logits, labels)

            total_loss += loss.item()
            total_acc_top1 += compute_accuracy(logits, labels, top_k=1)
            total_acc_top3 += compute_accuracy(logits, labels, top_k=3)
            total_acc_top5 += compute_accuracy(logits, labels, top_k=5)
            n_batches += 1

        return (
            total_loss / max(n_batches, 1),
            {
                "top1": total_acc_top1 / max(n_batches, 1),
                "top3": total_acc_top3 / max(n_batches, 1),
                "top5": total_acc_top5 / max(n_batches, 1),
            },
        )

    def _save_checkpoint(self, filename: str) -> None:
        """체크포인트 저장."""
        checkpoint_dir = self.output_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename

        self.model.save_pretrained(path)

        # 오래된 체크포인트 정리 (best_model은 유지)
        if "checkpoint_epoch" in filename:
            self._cleanup_old_checkpoints(checkpoint_dir)

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path) -> None:
        """오래된 체크포인트 삭제 (keep_top_k_checkpoints)."""
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_ckpt in checkpoints[self.config.keep_top_k_checkpoints:]:
            old_ckpt.unlink()

    def _save_history(self) -> None:
        """학습 이력 JSON 저장."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"학습 이력 저장: {history_path}")


# ─── 편의 함수 ────────────────────────────────────────────────────


def train_deep_space(
    sector_id: str,
    dates: list[str],
    config: DeepSpaceConfig | None = None,
    output_dir: Path | None = None,
    progress_callback: Callable | None = None,
) -> DeepSpaceModel:
    """
    Deep Space 모델 학습 원스텝 함수.

    Args:
        sector_id: Sector ID
        dates: 학습에 사용할 날짜 리스트
        config: 모델 설정
        output_dir: 출력 디렉토리 (기본: DEEP_SPACE_DIR/{sector_id})
        progress_callback: 진행 콜백

    Returns:
        학습된 DeepSpaceModel
    """
    import config as cfg
    from src.model.tokenizer import LocusTokenizer, create_tokenizer
    from src.model.dataset import create_mlm_dataset

    config = config or DEFAULT_CONFIG

    # 출력 디렉토리
    deep_space_dir = getattr(cfg, "DEEP_SPACE_DIR", cfg.DATA_DIR / "deep_space")
    if output_dir is None:
        output_dir = deep_space_dir / sector_id

    # 토크나이저 생성/로드
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = LocusTokenizer.load(tokenizer_path)
    else:
        tokenizer = create_tokenizer(sector_id, deep_space_dir=deep_space_dir)

    # 데이터셋 생성
    dataset = create_mlm_dataset(sector_id, dates, tokenizer, config)
    if len(dataset) == 0:
        raise ValueError(f"학습 데이터 없음: {sector_id}, dates={dates}")

    # CPU 최적화: 샘플링 상한 적용
    max_samples = getattr(config, "max_train_samples", 0)
    if max_samples and len(dataset) > max_samples:
        import random
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"샘플링 적용: {max_samples}개 / 전체 {len(indices) + (len(dataset) - len(indices))}개")

    logger.info(f"학습 데이터: {len(dataset)}개 시퀀스")

    # 모델 생성
    model = DeepSpaceModel(config=config, vocab_size=tokenizer.vocab_size)

    # 학습
    trainer = DeepSpaceTrainer(
        model=model,
        train_dataset=dataset,
        config=config,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
    trainer.train()

    # Best 모델 로드
    best_model_path = output_dir / "checkpoint" / "best_model.pt"
    if best_model_path.exists():
        model = DeepSpaceModel.from_pretrained(best_model_path)

    return model


def load_trained_model(sector_id: str) -> DeepSpaceModel | None:
    """
    학습된 모델 로드.

    Args:
        sector_id: Sector ID

    Returns:
        DeepSpaceModel 또는 None (모델 없음)
    """
    import config as cfg

    deep_space_dir = getattr(cfg, "DEEP_SPACE_DIR", cfg.DATA_DIR / "deep_space")
    model_path = deep_space_dir / sector_id / "checkpoint" / "best_model.pt"

    if not model_path.exists():
        return None

    try:
        return DeepSpaceModel.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"모델 로드 실패: {e}")
        return None


def get_model_info(sector_id: str) -> dict | None:
    """
    모델 정보 조회.

    Returns:
        {
            "exists": bool,
            "vocab_size": int,
            "n_params": int,
            "best_val_loss": float,
            "best_epoch": int,
            "training_time": float,
        }
    """
    import config as cfg

    deep_space_dir = getattr(cfg, "DEEP_SPACE_DIR", cfg.DATA_DIR / "deep_space")
    model_dir = deep_space_dir / sector_id

    model_path = model_dir / "checkpoint" / "best_model.pt"
    history_path = model_dir / "training_history.json"
    tokenizer_path = model_dir / "tokenizer.json"

    if not model_path.exists():
        return {"exists": False}

    info = {"exists": True}

    # 토크나이저 정보
    if tokenizer_path.exists():
        with open(tokenizer_path, encoding="utf-8") as f:
            tok_data = json.load(f)
            info["vocab_size"] = tok_data.get("vocab_size", 0)

    # 학습 이력
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
            info["best_val_loss"] = history.get("best_val_loss", 0)
            info["best_epoch"] = history.get("best_epoch", 0)
            info["training_time"] = history.get("total_train_time", 0)
            info["train_losses"] = history.get("train_losses", [])
            info["val_losses"] = history.get("val_losses", [])
            info["val_acc_top1"] = history.get("val_acc_top1", [])
            info["val_acc_top3"] = history.get("val_acc_top3", [])
            info["val_acc_top5"] = history.get("val_acc_top5", [])

    # 모델 파라미터 수
    try:
        model = DeepSpaceModel.from_pretrained(model_path)
        info["n_params"] = model.num_parameters
    except Exception:
        pass

    return info
