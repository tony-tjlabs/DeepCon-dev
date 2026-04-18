"""
Deep Space Foundation Model
===========================
Transformer 기반 Journey 패턴 학습/예측 모델.

핵심 개념:
  "Journey = 문장(Sentence), Locus = 단어(Word)"
  - 작업자의 하루 이동 경로 = 토큰 시퀀스
  - MLM(Masked Language Model) 방식 사전학습
  - 다운스트림: 이동 예측, 이상 탐지

사용법:
    # 학습 (Dev 환경)
    from src.model.trainer import train_deep_space
    model = train_deep_space(sector_id, dates)

    # 추론 (Cloud 포함)
    from src.model.inference import DeepSpaceInference
    inf = DeepSpaceInference(sector_id)
    probs = inf.predict_next(current_sequence)

모듈 구조:
    config.py       - 모델 하이퍼파라미터
    tokenizer.py    - Locus 토크나이저
    dataset.py      - Journey MLM 데이터셋
    transformer.py  - Transformer Encoder 아키텍처
    trainer.py      - 학습 루프
    inference.py    - 추론 API
    downstream/     - 다운스트림 태스크
        anomaly.py      - 이상 탐지 (perplexity 기반)
        predictor.py    - 다음 Locus 예측
"""
from src.model.config import DeepSpaceConfig
from src.model.tokenizer import LocusTokenizer
from src.model.transformer import DeepSpaceModel

__all__ = [
    "DeepSpaceConfig",
    "LocusTokenizer",
    "DeepSpaceModel",
]
