"""
Locus 데이터 모델 — 범용 공간 단위 정의
========================================
모든 도메인(건설/리테일/공항)에서 공통으로 사용하는 Locus 속성 스키마.

핵심 개념:
  - Locus = 의미 있는 공간 단위 (Word)
  - Journey = Locus 시퀀스 (Sentence)
  - LocusBase는 도메인 무관 범용 속성만 포함
  - 도메인별 확장은 extensions.py에서 정의

Locus ID 체계:
  L-{SECTOR_CODE}-{SEQUENCE}
  예: L-Y1-001 (Y1_SKHynix 건설), L-LM-001 (LotteMart 리테일)

사용:
    from core.schema.locus import LocusBase, LocusType, DwellCategory

    locus = LocusBase(
        locus_id="L-Y1-020",
        locus_name="FAB 1F 작업구역",
        locus_type=LocusType.WORK,
        token="work_zone",
        building="FAB",
        floor="1F",
    )
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LocusType(str, Enum):
    """Locus 기능 분류 열거형.

    v1 기존 값: WORK, REST, GATE, FACILITY, ADMIN, TRANSPORT, HAZARD
    v2 추가 값: WORK_AREA, REST_AREA, CLEANROOM, TRANSITION, VERTICAL
    """

    # v1 기존 (backward compatible)
    WORK = "WORK"
    REST = "REST"
    GATE = "GATE"
    FACILITY = "FACILITY"
    ADMIN = "ADMIN"
    TRANSPORT = "TRANSPORT"
    HAZARD = "HAZARD"

    # v2 추가 (S-Ward 기반 세분화)
    WORK_AREA = "WORK_AREA"
    REST_AREA = "REST_AREA"
    CLEANROOM = "CLEANROOM"
    TRANSITION = "TRANSITION"
    VERTICAL = "VERTICAL"


class LocusScale(str, Enum):
    """Locus 공간 규모 열거형."""

    POINT = "POINT"       # 단일 지점 (휴게실, 흡연장)
    ZONE = "ZONE"         # 구역 (CUB 1F, 밀폐공간)
    BLOCK = "BLOCK"       # 블록 (FAB 전체, 야외 공사현장)
    FLOOR = "FLOOR"       # 층 단위
    ROOM = "ROOM"         # 개별 공간


class DwellCategory(str, Enum):
    """체류 성격 분류 열거형."""

    TRANSIT = "TRANSIT"           # 통과 지점 (< 5분)
    SHORT_STAY = "SHORT_STAY"     # 단시간 체류 (5~30분)
    LONG_STAY = "LONG_STAY"       # 장시간 체류 (30분~2시간)
    HAZARD_ZONE = "HAZARD_ZONE"   # 위험 구역 (체류 최소화 대상)
    ADMIN = "ADMIN"               # 사무/행정 구역


# locus.csv의 scale 컬럼 값 → LocusScale 매핑 (대소문자 무관)
_SCALE_ALIASES: dict[str, LocusScale] = {
    "point": LocusScale.POINT,
    "zone": LocusScale.ZONE,
    "block": LocusScale.BLOCK,
    "floor": LocusScale.FLOOR,
    "room": LocusScale.ROOM,
}


def parse_scale(raw: str) -> LocusScale:
    """CSV의 scale 문자열을 LocusScale 열거형으로 변환."""
    key = str(raw).strip().lower()
    return _SCALE_ALIASES.get(key, LocusScale.ZONE)


# locus.csv의 dwell_category 컬럼 값 → DwellCategory 매핑
_DWELL_ALIASES: dict[str, DwellCategory] = {
    "transit": DwellCategory.TRANSIT,
    "short_stay": DwellCategory.SHORT_STAY,
    "long_stay": DwellCategory.LONG_STAY,
    "hazard_zone": DwellCategory.HAZARD_ZONE,
    "admin": DwellCategory.ADMIN,
}


def parse_dwell_category(raw: str) -> DwellCategory:
    """CSV의 dwell_category 문자열을 DwellCategory 열거형으로 변환."""
    key = str(raw).strip().upper()
    return _DWELL_ALIASES.get(key.lower(), DwellCategory.SHORT_STAY)


class LocusBase(BaseModel):
    """도메인 무관 범용 Locus 속성.

    locus.csv에서 로드하여 Locus 레지스트리에 등록한다.
    도메인별 확장 속성은 extensions.py의 Extension 모델을 사용한다.
    """

    locus_id: str = Field(
        ...,
        description="전역 고유 식별자 (L-{SECTOR_CODE}-{SEQ})",
        examples=["L-Y1-001", "L-LM-001"],
    )
    locus_name: str = Field(
        ...,
        description="사람이 읽을 수 있는 장소명",
        examples=["본진 타각기 입구", "FAB 1F 작업구역"],
    )
    locus_type: LocusType = Field(
        ...,
        description="기능 분류 (WORK/REST/GATE/FACILITY/ADMIN/TRANSPORT/HAZARD)",
    )
    token: str = Field(
        default="",
        description="Deep Space 토큰 (timeclock, work_zone 등)",
    )
    building: str = Field(
        default="",
        description="소속 건물 (FAB, CUB, WWT 등)",
    )
    floor: str = Field(
        default="",
        description="소속 층 (1F, B1F 등)",
    )
    zone: str = Field(
        default="",
        description="소속 구역/존",
    )
    scale: LocusScale = Field(
        default=LocusScale.ZONE,
        description="공간 규모 (POINT/ZONE/BLOCK/FLOOR/ROOM)",
    )
    dwell_category: DwellCategory = Field(
        default=DwellCategory.SHORT_STAY,
        description="체류 성격 (TRANSIT/SHORT_STAY/LONG_STAY/HAZARD_ZONE/ADMIN)",
    )
    capacity: int = Field(
        default=0,
        ge=0,
        description="최대 수용 인원 (0=미지정)",
    )
    coordinates: Optional[dict[str, float]] = Field(
        default=None,
        description="대표 좌표 {'x': float, 'y': float}",
    )
    description: str = Field(
        default="",
        description="자연어 설명 (LLM 컨텍스트용)",
    )
    # ─── v2 S-Ward 기반 필드 (모두 Optional, v1 호환 유지) ─────────
    gateway_no: Optional[int] = Field(
        default=None,
        description="S-Ward(Gateway) 고유 번호 (v2 전용)",
    )
    gateway_name: Optional[str] = Field(
        default=None,
        description="S-Ward 원본 이름 (v2 전용)",
    )
    building_no: Optional[int] = Field(
        default=None,
        description="SSMP 건물 번호 (v2 전용)",
    )
    floor_no: Optional[int] = Field(
        default=None,
        description="SSMP 층 번호 (v2 전용)",
    )
    location_x: Optional[float] = Field(
        default=None,
        description="건물+층 내 정확 좌표 X (v2 전용)",
    )
    location_y: Optional[float] = Field(
        default=None,
        description="건물+층 내 정확 좌표 Y (v2 전용)",
    )
    locus_version: str = Field(
        default="v1",
        description="Locus 버전 (v1=기존 58개, v2=S-Ward 기반 213개)",
    )

    # ─── 기존 필드 (v1 호환) ──────────────────────────────────────
    sward_ids: list[str] = Field(
        default_factory=list,
        description="매핑된 S-Ward ID 목록",
    )
    sward_count: int = Field(
        default=0,
        ge=0,
        description="매핑된 S-Ward 수",
    )
    tags: str = Field(
        default="",
        description="공간 태그 (#gate #anchor #hazard_3 등)",
    )
    hazard_level: Optional[str] = Field(
        default=None,
        description="위험 등급 (low/medium/high/critical)",
    )
    hazard_grade: Optional[float] = Field(
        default=None,
        description="위험 등급 숫자 (2.0~5.0)",
    )
    risk_flags: Optional[str] = Field(
        default=None,
        description="위험 플래그 (crowded, height, outdoor, confined 등)",
    )
    function: str = Field(
        default="",
        description="기능 분류 (ACCESS, WORK, REST, ADMIN, FACILITY 등)",
    )

    model_config = {"str_strip_whitespace": True}

    @property
    def is_hazardous(self) -> bool:
        """위험 구역 여부 (hazard_grade >= 4 또는 hazard_level=critical)."""
        if self.hazard_level == "critical":
            return True
        if self.hazard_grade is not None and self.hazard_grade >= 4.0:
            return True
        return False

    @property
    def is_confined_space(self) -> bool:
        """밀폐공간 여부 (tags에 #confined 포함)."""
        return "#confined" in self.tags

    @property
    def requires_team(self) -> bool:
        """2인 이상 필수 여부 (tags에 #no_lone 또는 #team_2 포함)."""
        return "#no_lone" in self.tags or "#team_2" in self.tags
