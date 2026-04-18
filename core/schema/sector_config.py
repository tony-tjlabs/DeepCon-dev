"""
SectorConfig — Sector별 전역 설정
=================================
sector_configs/{sector_id}.json에서 로드하는 Sector 설정.

사용:
    from core.schema.sector_config import SectorConfig

    config = SectorConfig.load("Y1_SKHynix", config_dir)
    print(config.primary_domain)  # "construction"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Raw 데이터 설정."""

    raw_encoding: str = Field(default="cp949", description="CSV 인코딩")
    raw_date_format: str = Field(default="%Y%m%d", description="날짜 형식")
    access_log_prefix: str = Field(default="", description="AccessLog 파일 접두사")
    tward_prefix: str = Field(default="", description="TWardData 파일 접두사")
    time_resolution_sec: int = Field(default=60, description="시간 해상도 (초)")


class ProcessingConfig(BaseModel):
    """처리 파이프라인 설정."""

    work_start_hour: int = Field(default=5, ge=0, le=23, description="근무 시작 시간")
    work_end_hour: int = Field(default=23, ge=0, le=23, description="근무 종료 시간")
    journey_correction_enabled: bool = Field(default=True, description="Journey 보정 활성화")
    min_signal_count: int = Field(default=1, ge=0, description="최소 신호 수")


class DisplayConfig(BaseModel):
    """대시보드 표시 설정."""

    theme_primary: str = Field(default="#1A3A5C", description="주 테마 색상")
    default_date_range_days: int = Field(default=7, ge=1, description="기본 날짜 범위(일)")


class SectorConfig(BaseModel):
    """Sector별 전역 설정.

    sector_configs/{sector_id}.json 파일에서 로드한다.
    모든 Sector 공통 필드를 Pydantic 모델로 검증한다.

    Attributes:
        sector_id: Sector 고유 식별자
        sector_code: 약어 (2~4자, Locus ID에 사용)
        label: 표시명
        subtitle: 부제
        icon: 아이콘 (이모지 또는 식별자)
        primary_domain: 주 도메인 (construction/retail/airport)
        active_packs: 활성화된 Domain Pack 목록
        status: 상태 (active/coming_soon)
        data_config: 데이터 설정
        processing_config: 처리 설정
        display_config: 표시 설정
    """

    sector_id: str = Field(..., description="Sector 고유 식별자")
    sector_code: str = Field(default="", description="Sector 약어 (Locus ID에 사용)")
    label: str = Field(default="", description="표시명")
    subtitle: str = Field(default="", description="부제")
    icon: str = Field(default="", description="아이콘")
    primary_domain: str = Field(default="construction", description="주 도메인")
    active_packs: list[str] = Field(default_factory=list, description="활성 Domain Pack 목록")
    status: str = Field(default="active", description="상태 (active/coming_soon)")
    data_config: DataConfig = Field(default_factory=DataConfig)
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    display_config: DisplayConfig = Field(default_factory=DisplayConfig)

    model_config = {"str_strip_whitespace": True}

    @classmethod
    def load(cls, sector_id: str, config_dir: Path) -> "SectorConfig":
        """sector_configs/{sector_id}.json에서 설정을 로드한다.

        Args:
            sector_id: Sector 식별자
            config_dir: sector_configs 디렉토리 경로

        Returns:
            SectorConfig 인스턴스

        Raises:
            FileNotFoundError: JSON 파일이 없을 때
            ValueError: JSON 파싱 또는 검증 실패 시
        """
        json_path = config_dir / f"{sector_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Sector 설정 파일 없음: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        # sector_id가 JSON에 없으면 파일명에서 추출
        if "sector_id" not in raw:
            raw["sector_id"] = sector_id

        logger.info("SectorConfig 로드: %s", sector_id)
        return cls.model_validate(raw)

    @classmethod
    def load_all(cls, config_dir: Path) -> dict[str, "SectorConfig"]:
        """config_dir 내 모든 Sector 설정을 로드한다.

        '_' 접두사 파일은 템플릿으로 간주하여 제외.

        Args:
            config_dir: sector_configs 디렉토리 경로

        Returns:
            {sector_id: SectorConfig} 딕셔너리
        """
        configs: dict[str, SectorConfig] = {}
        if not config_dir.exists():
            logger.warning("sector_configs 디렉토리 없음: %s", config_dir)
            return configs

        for json_file in sorted(config_dir.glob("*.json")):
            if json_file.stem.startswith("_"):
                continue
            try:
                cfg = cls.load(json_file.stem, config_dir)
                configs[cfg.sector_id] = cfg
            except Exception as e:
                logger.warning("Sector 설정 로드 실패 [%s]: %s", json_file.name, e)

        return configs
