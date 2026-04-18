"""
도메인 팩 로더 — domain_packs/{domain}/ 리소스 통합 로드
===========================================================
core/ 레이어가 domain_packs/ 리소스를 읽기 위한 중개 모듈.

핵심 기능:
  - load_construction_config(): config.yaml 로드
  - load_alert_rules(): rules.py ALERT_RULES 로드
  - get_domain_pack_path(): 도메인 팩 경로 반환
  - load_domain_attrs_csv(): locus_attrs.csv 로드 → dict

사용:
    from core.registry.domain_loader import (
        load_construction_config,
        load_alert_rules,
        get_domain_pack_path,
        load_domain_attrs_csv,
    )
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# domain_packs/ 루트 경로 (core/ 기준으로 계산)
_CORE_DIR = Path(__file__).resolve().parent.parent        # core/
_PROJECT_ROOT = _CORE_DIR.parent                          # DeepCon/
_DOMAIN_PACKS_DIR = _PROJECT_ROOT / "domain_packs"


def get_domain_pack_path(domain: str) -> Path:
    """도메인 팩 경로를 반환한다.

    Args:
        domain: 도메인 이름 (construction, retail, airport)

    Returns:
        도메인 팩 디렉토리 Path

    Raises:
        FileNotFoundError: 도메인 팩이 없을 때
    """
    pack_path = _DOMAIN_PACKS_DIR / domain
    if not pack_path.exists():
        raise FileNotFoundError(f"도메인 팩 없음: {pack_path}")
    return pack_path


def load_construction_config() -> dict[str, Any]:
    """domain_packs/construction/config.yaml을 로드한다.

    Returns:
        YAML 딕셔너리 (ewi, cre, fatigue, lone_work 등)

    Raises:
        FileNotFoundError: config.yaml이 없을 때
    """
    config_path = get_domain_pack_path("construction") / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"건설 도메인 설정 파일 없음: {config_path}")

    import yaml
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.info("건설 도메인 설정 로드 완료: %s", config_path)
    return data or {}


def load_alert_rules() -> list:
    """domain_packs/construction/rules.py의 ALERT_RULES를 로드한다.

    Returns:
        AlertRule 인스턴스 리스트
    """
    from domain_packs.construction.rules import ALERT_RULES
    logger.info("건설 도메인 알림 규칙 로드: %d개", len(ALERT_RULES))
    return list(ALERT_RULES)


def load_domain_attrs_csv(domain: str = "construction") -> dict[str, dict[str, Any]]:
    """도메인 팩의 locus_attrs.csv를 로드하여 딕셔너리로 반환한다.

    Args:
        domain: 도메인 이름

    Returns:
        {locus_id: {attr_name: value, ...}} 딕셔너리

    Raises:
        FileNotFoundError: CSV가 없을 때
    """
    pack_path = get_domain_pack_path(domain)
    csv_path = pack_path / "locus_attrs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"도메인 속성 파일 없음: {csv_path}")

    result: dict[str, dict[str, Any]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            locus_id = row.get("locus_id", "").strip()
            if not locus_id:
                continue
            result[locus_id] = dict(row)

    logger.info(
        "도메인 속성 로드: %s — %d개 Locus",
        csv_path.name, len(result),
    )
    return result
