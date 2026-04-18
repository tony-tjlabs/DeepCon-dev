"""
DeepConRecord — Core 정규화 레코드
=================================
모든 도메인의 Raw 데이터를 통일된 형식으로 변환한 레코드.

데이터 흐름:
  Raw CSV → Domain Pack Parser → DeepConRecord → Locus Mapping → Journey

핵심 원칙:
  - 도메인 무관 공통 필드 (entity_id, timestamp, locus_id)
  - 도메인 특화 데이터는 domain_meta (dict)에 저장
  - Observability Class로 데이터 신뢰도 자동 분류

사용:
    from core.schema.record import DeepConRecord, ObservabilityClass

    record = DeepConRecord(
        sector="Y1_SKHynix",
        entity_id="TW-001",
        entity_type="PERSON_FIXED",
        timestamp=datetime(2026, 3, 18, 7, 30),
        raw_location="FAB 1F",
        locus_id="L-Y1-020",
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ObservabilityClass(str, Enum):
    """관측 가능성 등급.

    데이터 소스의 신뢰도와 추적 가능성을 분류.
    - FULL: T-Ward 태그, 고정 MAC → 개인 식별 + 연속 추적 가능
    - PARTIAL: iPhone/Android BLE → MAC 랜덤화, AST 기반 분석
    - AGGREGATE: MAC 카운트 → 개인 추적 불가, 집계 통계만
    """

    FULL = "FULL"
    PARTIAL = "PARTIAL"
    AGGREGATE = "AGGREGATE"


# entity_type → observability_class 자동 매핑
_ENTITY_OBSERVABILITY: dict[str, ObservabilityClass] = {
    "PERSON_FIXED": ObservabilityClass.FULL,
    "DEVICE": ObservabilityClass.FULL,
    "PERSON_RANDOM": ObservabilityClass.PARTIAL,
    "MAC_COUNT": ObservabilityClass.AGGREGATE,
}


def derive_observability(entity_type: str) -> ObservabilityClass:
    """entity_type에서 관측 가능성 등급을 자동 도출한다."""
    return _ENTITY_OBSERVABILITY.get(entity_type, ObservabilityClass.PARTIAL)


class DeepConRecord(BaseModel):
    """Core 정규화 레코드.

    모든 도메인의 원본 데이터를 이 형식으로 변환하여
    일관된 파이프라인 처리를 보장한다.

    Attributes:
        record_uid: 레코드 고유 ID (자동 생성)
        sector: Sector 식별자 (Y1_SKHynix 등)
        entity_id: 비식별화된 개체 ID (작업자/장치)
        entity_type: 개체 유형 (PERSON_FIXED/PERSON_RANDOM/DEVICE/MAC_COUNT)
        observability_class: 관측 가능성 등급 (자동 도출)
        timestamp: 관측 시각
        raw_location: 원본 위치 문자열 (spot_name 등)
        locus_id: 매핑된 Locus ID (매핑 전 None)
        confidence: 위치 확신도 (0.0~1.0)
        signal_strength: 신호 강도 (RSSI, dBm)
        activity_ratio: 활성 비율 (건설: active_count/signal_count)
        domain_meta_schema: 도메인 메타 스키마명 (construction_meta_v1 등)
        domain_meta: 도메인 특화 메타 데이터 (자유 형식)
        source_type: 데이터 소스 타입 (tward_csv, ble_flow_csv 등)
        parser_version: 파서 버전
        schema_version: DeepConRecord 스키마 버전
        ingested_at: 수집 시각 (자동)
        mapping_method: Locus 매핑 방법 (spot_name_map, rssi_argmax 등)
        mapping_confidence: 매핑 확신도 (0.0~1.0)
    """

    record_uid: str = Field(
        default_factory=lambda: str(uuid4()),
        description="레코드 고유 ID",
    )
    sector: str = Field(
        ...,
        description="Sector 식별자",
    )
    entity_id: str = Field(
        ...,
        description="비식별화된 개체 ID",
    )
    entity_type: str = Field(
        ...,
        description="개체 유형 (PERSON_FIXED/PERSON_RANDOM/DEVICE/MAC_COUNT)",
    )
    observability_class: ObservabilityClass = Field(
        default=ObservabilityClass.PARTIAL,
        description="관측 가능성 등급 (자동 도출)",
    )

    timestamp: datetime = Field(
        ...,
        description="관측 시각",
    )
    raw_location: str = Field(
        default="",
        description="원본 위치 문자열",
    )
    locus_id: Optional[str] = Field(
        default=None,
        description="매핑된 Locus ID",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="위치 확신도",
    )

    signal_strength: Optional[float] = Field(
        default=None,
        description="신호 강도 (RSSI, dBm)",
    )
    activity_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="활성 비율 (건설: active_count/signal_count)",
    )

    domain_meta_schema: str = Field(
        default="",
        description="도메인 메타 스키마명",
    )
    domain_meta: dict = Field(
        default_factory=dict,
        description="도메인 특화 메타 데이터",
    )

    source_type: str = Field(
        default="",
        description="데이터 소스 타입",
    )
    parser_version: str = Field(
        default="1.0",
        description="파서 버전",
    )
    schema_version: str = Field(
        default="1.1",
        description="DeepConRecord 스키마 버전",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="수집 시각 (UTC)",
    )
    mapping_method: Optional[str] = Field(
        default=None,
        description="Locus 매핑 방법",
    )
    mapping_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="매핑 확신도",
    )

    model_config = {"str_strip_whitespace": True}

    def model_post_init(self, __context: object) -> None:
        """observability_class를 entity_type에서 자동 도출."""
        if self.observability_class == ObservabilityClass.PARTIAL:
            self.observability_class = derive_observability(self.entity_type)
