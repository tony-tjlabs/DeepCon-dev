"""
도메인별 Locus 확장 속성
========================
각 Domain Pack이 LocusBase에 추가하는 도메인 특화 속성.

- ConstructionExtension: 건설현장 (위험등급, 밀폐공간, 고소작업)
- RetailExtension: 리테일 (상품 카테고리, 내부/외부)
- AirportExtension: 공항 (보안 레벨, 병목 취약)

사용:
    from core.schema.extensions import ConstructionExtension

    ext = ConstructionExtension(
        hazard_level="hazard_4",
        confined_space=True,
        minimum_crew=2,
    )
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ConstructionExtension(BaseModel):
    """건설 도메인 Locus 확장 속성.

    위험등급, 밀폐공간, 고소작업, 안전 태그 등 건설현장 특화 정보.
    """

    hazard_level: str = Field(
        default="hazard_2",
        description="위험 등급 (hazard_2 ~ hazard_5)",
    )
    confined_space: bool = Field(
        default=False,
        description="밀폐공간 여부",
    )
    height_risk: bool = Field(
        default=False,
        description="고소작업 위험 여부",
    )
    lone_work_prohibited: bool = Field(
        default=False,
        description="단독작업 금지 여부",
    )
    minimum_crew: int = Field(
        default=1,
        ge=1,
        description="최소 작업 인원",
    )
    safety_tags: list[str] = Field(
        default_factory=list,
        description="안전 태그 목록 (#no_lone, #team_2 등)",
    )
    work_permit_required: bool = Field(
        default=False,
        description="작업 허가서 필요 여부",
    )

    @classmethod
    def from_locus_tags(cls, tags: str, hazard_grade: float | None = None) -> "ConstructionExtension":
        """Locus 태그 문자열에서 건설 확장 속성을 추출한다.

        Args:
            tags: 공간 태그 문자열 (#gate #confined #hazard_5 등)
            hazard_grade: 위험 등급 숫자 (2.0~5.0)

        Returns:
            ConstructionExtension 인스턴스
        """
        tag_set = set(tags.lower().split())

        # hazard_level 결정
        grade = hazard_grade or 2.0
        hazard_level = f"hazard_{int(grade)}"

        # 플래그 추출
        confined = "#confined" in tag_set
        height = "#height" in tag_set
        no_lone = "#no_lone" in tag_set
        team_2 = "#team_2" in tag_set

        # 최소 인원 결정
        minimum_crew = 2 if (no_lone or team_2 or confined) else 1

        # 안전 태그 수집
        safety_tags = [t for t in tag_set if t.startswith("#") and t in {
            "#no_lone", "#team_2", "#confined", "#height",
            "#high_volt", "#outdoor", "#hazard_3", "#hazard_4", "#hazard_5",
        }]

        # 작업 허가서: 위험 4등급 이상 또는 밀폐공간
        work_permit_required = grade >= 4.0 or confined

        return cls(
            hazard_level=hazard_level,
            confined_space=confined,
            height_risk=height,
            lone_work_prohibited=no_lone or team_2,
            minimum_crew=minimum_crew,
            safety_tags=sorted(safety_tags),
            work_permit_required=work_permit_required,
        )


class RetailExtension(BaseModel):
    """리테일 도메인 Locus 확장 속성.

    상품 카테고리, 내부/외부 분류, 전환 구역 등 소매점 특화 정보.
    """

    product_category: str = Field(
        default="",
        description="상품 카테고리 (식품, 의류 등)",
    )
    is_interior: bool = Field(
        default=True,
        description="매장 내부 여부 (AST 계산 기준)",
    )
    display_type: str = Field(
        default="",
        description="진열 타입 (Gondola, Endcap 등)",
    )
    conversion_zone: bool = Field(
        default=False,
        description="전환율 측정 대상 구역 여부",
    )
    zone_attractiveness: float = Field(
        default=0.0,
        ge=0.0,
        description="구역 매력도 (기준 체류시간 대비 비율)",
    )


class AirportExtension(BaseModel):
    """공항 도메인 Locus 확장 속성.

    구역 기능, 보안 레벨, 병목 취약성 등 공항 특화 정보.
    """

    zone_function: str = Field(
        default="",
        description="구역 기능 (check_in, security, gate, lounge, retail, food_court)",
    )
    flight_related: bool = Field(
        default=False,
        description="항공편 연동 구역 여부",
    )
    security_level: int = Field(
        default=0,
        ge=0,
        le=3,
        description="보안 등급 (0=공개, 1=검색후, 2=출입제한, 3=승무원전용)",
    )
    public_access: bool = Field(
        default=True,
        description="일반 대중 접근 가능 여부",
    )
    bottleneck_prone: bool = Field(
        default=False,
        description="병목 취약 구역 여부",
    )
    target_throughput: int = Field(
        default=0,
        ge=0,
        description="목표 처리량 (시간당 인원)",
    )
