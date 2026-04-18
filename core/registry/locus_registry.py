"""
LocusRegistry — Sector별 Locus 관리
====================================
locus.csv에서 Locus를 로드하고 조회/필터/자연어 변환을 제공한다.

핵심 기능:
  - load_from_csv(): locus.csv → LocusBase 인스턴스 목록
  - get() / get_by_type() / get_by_building(): Locus 조회
  - sward_to_locus(): S-Ward ID → Locus ID 매핑
  - to_natural_language(): LLM 컨텍스트용 자연어 변환
  - get_adjacency(): 인접 관계 그래프 로드

사용:
    from core.registry.locus_registry import LocusRegistry

    registry = LocusRegistry("Y1_SKHynix")
    registry.load_from_csv(locus_csv_path)

    locus = registry.get("L-Y1-020")
    print(registry.to_natural_language("L-Y1-055"))
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from core.schema.extensions import ConstructionExtension
from core.schema.locus import (
    DwellCategory,
    LocusBase,
    LocusScale,
    LocusType,
    parse_dwell_category,
    parse_scale,
)

logger = logging.getLogger(__name__)


class LocusRegistry:
    """Sector별 Locus 관리 레지스트리.

    단일 Sector의 모든 Locus를 메모리에 보유하고,
    다양한 조회/필터/변환 기능을 제공한다.

    Attributes:
        sector_id: Sector 식별자
    """

    def __init__(self, sector_id: str) -> None:
        """LocusRegistry를 초기화한다.

        Args:
            sector_id: Sector 식별자 (Y1_SKHynix 등)
        """
        self.sector_id = sector_id
        self._loci: dict[str, LocusBase] = {}
        self._sward_map: dict[str, str] = {}  # sward_id → locus_id
        self._adjacency: dict[str, list[str]] = {}
        self._domain_extensions: dict[str, ConstructionExtension] = {}  # locus_id → ext

    # ─── 로드 ──────────────────────────────────────────────────────

    def load_from_csv(self, csv_path: Path) -> None:
        """locus.csv에서 Locus 데이터를 로드한다.

        CSV 컬럼은 LocusBase 필드에 매핑된다.
        sward_ids 컬럼의 쉼표 구분 값으로 sward → locus 역매핑을 구축한다.

        Args:
            csv_path: locus.csv 파일 경로

        Raises:
            FileNotFoundError: CSV 파일이 없을 때
            ValueError: 필수 컬럼(locus_id)이 없을 때
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"locus.csv 파일 없음: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str).fillna("")

        if "locus_id" not in df.columns:
            raise ValueError("locus_id 컬럼이 없습니다")

        self._loci.clear()
        self._sward_map.clear()

        for _, row in df.iterrows():
            locus_id = row["locus_id"].strip()
            if not locus_id:
                continue

            # LocusType 파싱 (대소문자 무관)
            raw_type = row.get("locus_type", "WORK").strip().upper()
            try:
                locus_type = LocusType(raw_type)
            except ValueError:
                locus_type = LocusType.WORK

            # Scale 파싱
            scale = parse_scale(row.get("scale", "ZONE"))

            # DwellCategory 파싱
            dwell_cat = parse_dwell_category(row.get("dwell_category", "SHORT_STAY"))

            # sward_ids 파싱 (쉼표 구분)
            raw_swards = row.get("sward_ids", "")
            sward_ids = [s.strip() for s in raw_swards.split(",") if s.strip()]

            # sward_count
            sward_count_str = row.get("sward_count", "0")
            try:
                sward_count = int(float(sward_count_str)) if sward_count_str else len(sward_ids)
            except (ValueError, TypeError):
                sward_count = len(sward_ids)

            # hazard_grade 파싱
            hazard_grade_str = row.get("hazard_grade", "")
            hazard_grade: Optional[float] = None
            if hazard_grade_str:
                try:
                    hazard_grade = float(hazard_grade_str)
                except (ValueError, TypeError):
                    pass

            # v2 필드 파싱 (Optional, 없으면 None)
            gateway_no = None
            gw_no_str = row.get("gateway_no", "")
            if gw_no_str:
                try:
                    gateway_no = int(float(gw_no_str))
                except (ValueError, TypeError):
                    pass

            gateway_name = row.get("gateway_name", None) or None

            building_no = None
            bno_str = row.get("building_no", "")
            if bno_str:
                try:
                    building_no = int(float(bno_str))
                except (ValueError, TypeError):
                    pass

            floor_no = None
            fno_str = row.get("floor_no", "")
            if fno_str:
                try:
                    floor_no = int(float(fno_str))
                except (ValueError, TypeError):
                    pass

            location_x = None
            lx_str = row.get("location_x", "")
            if lx_str:
                try:
                    location_x = float(lx_str)
                except (ValueError, TypeError):
                    pass

            location_y = None
            ly_str = row.get("location_y", "")
            if ly_str:
                try:
                    location_y = float(ly_str)
                except (ValueError, TypeError):
                    pass

            locus_version = row.get("locus_version", "v1") or "v1"

            locus = LocusBase(
                locus_id=locus_id,
                locus_name=row.get("locus_name", ""),
                locus_type=locus_type,
                token=row.get("token", ""),
                building=row.get("building", ""),
                floor=row.get("floor", ""),
                zone=row.get("zone", row.get("zone_type", "")),
                scale=scale,
                dwell_category=dwell_cat,
                capacity=0,  # CSV에 capacity 컬럼 없음
                description=row.get("description", ""),
                # v2 S-Ward 필드
                gateway_no=gateway_no,
                gateway_name=gateway_name,
                building_no=building_no,
                floor_no=floor_no,
                location_x=location_x,
                location_y=location_y,
                locus_version=locus_version,
                # 기존 필드
                sward_ids=sward_ids,
                sward_count=sward_count,
                tags=row.get("tags", ""),
                hazard_level=row.get("hazard_level", None) or None,
                hazard_grade=hazard_grade,
                risk_flags=row.get("risk_flags", None) or None,
                function=row.get("function", ""),
            )

            self._loci[locus_id] = locus

            # S-Ward → Locus 역매핑
            for sward_id in sward_ids:
                self._sward_map[sward_id] = locus_id

        logger.info(
            "LocusRegistry [%s]: %d개 Locus 로드, %d개 S-Ward 매핑",
            self.sector_id,
            len(self._loci),
            len(self._sward_map),
        )

    def load_adjacency(self, adjacency_path: Path) -> None:
        """인접 관계 CSV를 로드한다.

        CSV 형식: from_locus_id, to_locus_id, weight (선택)

        Args:
            adjacency_path: locus_adjacency.csv 파일 경로
        """
        adjacency_path = Path(adjacency_path)
        if not adjacency_path.exists():
            logger.warning("인접 관계 파일 없음: %s", adjacency_path)
            return

        self._adjacency.clear()

        with open(adjacency_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_data in reader:
                from_id = row_data.get("from_locus_id", "").strip()
                to_id = row_data.get("to_locus_id", "").strip()
                if from_id and to_id:
                    self._adjacency.setdefault(from_id, []).append(to_id)
                    # 양방향 (무방향 그래프)
                    self._adjacency.setdefault(to_id, []).append(from_id)

        # 중복 제거
        for lid in self._adjacency:
            self._adjacency[lid] = sorted(set(self._adjacency[lid]))

        logger.info(
            "Adjacency 로드: %d개 노드, %d개 엣지",
            len(self._adjacency),
            sum(len(v) for v in self._adjacency.values()) // 2,
        )

    def load_domain_attrs(self, csv_path: Path) -> None:
        """도메인 팩의 locus_attrs.csv에서 건설 속성을 로드하여 각 Locus에 병합한다.

        CSV 컬럼: locus_id, hazard_level, hazard_grade, confined_space,
                  height_risk, minimum_crew, work_permit_required, dwell_category 등.

        각 Locus에 대해 ConstructionExtension 객체를 생성하여 내부 매핑에 보관한다.
        Locus의 hazard_grade, hazard_level, dwell_category도 CSV 값으로 갱신한다.

        Args:
            csv_path: locus_attrs.csv 파일 경로

        Raises:
            FileNotFoundError: CSV 파일이 없을 때
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"도메인 속성 파일 없음: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "locus_id" not in df.columns:
            raise ValueError("locus_id 컬럼이 없습니다")

        self._domain_extensions.clear()
        mapped_count = 0

        for _, row in df.iterrows():
            locus_id = row["locus_id"].strip()
            if not locus_id:
                continue

            locus = self._loci.get(locus_id)
            if locus is None:
                logger.warning("도메인 속성: Locus %s가 레지스트리에 없음 (무시)", locus_id)
                continue

            # hazard_grade 파싱
            hazard_grade_str = row.get("hazard_grade", "")
            hazard_grade = 2.0
            if hazard_grade_str:
                try:
                    hazard_grade = float(hazard_grade_str)
                except (ValueError, TypeError):
                    pass

            # ConstructionExtension 생성
            confined = row.get("confined_space", "False").strip().lower() == "true"
            height = row.get("height_risk", "False").strip().lower() == "true"
            min_crew_str = row.get("minimum_crew", "1")
            try:
                min_crew = int(min_crew_str) if min_crew_str else 1
            except (ValueError, TypeError):
                min_crew = 1
            work_permit = row.get("work_permit_required", "False").strip().lower() == "true"
            hazard_level = row.get("hazard_level", "low").strip()

            ext = ConstructionExtension(
                hazard_level=f"hazard_{int(hazard_grade)}",
                confined_space=confined,
                height_risk=height,
                lone_work_prohibited=confined or min_crew >= 2,
                minimum_crew=min_crew,
                safety_tags=[],
                work_permit_required=work_permit,
            )

            self._domain_extensions[locus_id] = ext

            # LocusBase 속성도 갱신 (pydantic v2 model copy 불필요 — 직접 대입)
            locus.hazard_grade = hazard_grade
            locus.hazard_level = hazard_level

            # dwell_category 갱신
            dwell_raw = row.get("dwell_category", "")
            if dwell_raw:
                locus.dwell_category = parse_dwell_category(dwell_raw)

            mapped_count += 1

        logger.info(
            "도메인 속성 로드 [%s]: %d개 Locus에 ConstructionExtension 매핑",
            self.sector_id,
            mapped_count,
        )

    def get_domain_extension(self, locus_id: str) -> Optional[ConstructionExtension]:
        """Locus의 도메인 확장 속성을 반환한다.

        Args:
            locus_id: Locus 식별자

        Returns:
            ConstructionExtension 또는 None
        """
        return self._domain_extensions.get(locus_id)

    # ─── 조회 ──────────────────────────────────────────────────────

    def get(self, locus_id: str) -> Optional[LocusBase]:
        """ID로 Locus를 조회한다.

        Args:
            locus_id: Locus 식별자

        Returns:
            LocusBase 인스턴스 또는 None
        """
        return self._loci.get(locus_id)

    def get_by_type(self, locus_type: LocusType) -> list[LocusBase]:
        """타입별 Locus 목록을 반환한다.

        Args:
            locus_type: LocusType 열거형

        Returns:
            해당 타입의 LocusBase 리스트
        """
        return [loc for loc in self._loci.values() if loc.locus_type == locus_type]

    def get_by_building(self, building: str) -> list[LocusBase]:
        """건물별 Locus 목록을 반환한다.

        Args:
            building: 건물명 (FAB, CUB, WWT 등)

        Returns:
            해당 건물의 LocusBase 리스트
        """
        building_lower = building.lower()
        return [
            loc for loc in self._loci.values()
            if loc.building.lower() == building_lower
        ]

    def get_hazardous(self) -> list[LocusBase]:
        """위험 구역 Locus 목록을 반환한다 (hazard_grade >= 4 또는 critical)."""
        return [loc for loc in self._loci.values() if loc.is_hazardous]

    def get_confined_spaces(self) -> list[LocusBase]:
        """밀폐공간 Locus 목록을 반환한다."""
        return [loc for loc in self._loci.values() if loc.is_confined_space]

    # ─── 매핑 ──────────────────────────────────────────────────────

    def sward_to_locus(self, sward_id: str) -> Optional[str]:
        """S-Ward ID를 Locus ID로 매핑한다.

        Args:
            sward_id: S-Ward 식별자 (SWD-000149 등)

        Returns:
            매핑된 Locus ID 또는 None
        """
        return self._sward_map.get(sward_id)

    def gateway_to_locus(self, gateway_no: int) -> Optional[str]:
        """Gateway 번호를 Locus ID로 매핑한다 (v2 전용).

        Args:
            gateway_no: S-Ward(Gateway) 고유 번호

        Returns:
            매핑된 Locus ID (GW-{gateway_no}) 또는 None
        """
        locus_id = f"GW-{gateway_no}"
        return locus_id if locus_id in self._loci else None

    def get_adjacency(self) -> dict[str, list[str]]:
        """인접 관계 그래프를 반환한다.

        Returns:
            {locus_id: [인접 locus_id, ...]} 딕셔너리
        """
        return dict(self._adjacency)

    def get_neighbors(self, locus_id: str) -> list[str]:
        """특정 Locus의 인접 Locus ID 목록을 반환한다.

        Args:
            locus_id: 기준 Locus ID

        Returns:
            인접 Locus ID 리스트
        """
        return self._adjacency.get(locus_id, [])

    # ─── 자연어 변환 ──────────────────────────────────────────────

    def to_natural_language(self, locus_id: str) -> str:
        """Locus를 LLM 컨텍스트용 자연어로 변환한다.

        예: "FAB 1F 작업구역(L-Y1-020). FAB 1F에 위치한 WORK 구역으로,
             위험등급 3. 장시간 체류 공간이며 S-Ward 11개가 매핑되어 있다."

        Args:
            locus_id: Locus 식별자

        Returns:
            자연어 설명 문자열 (Locus가 없으면 "알 수 없는 장소")
        """
        locus = self._loci.get(locus_id)
        if locus is None:
            return f"알 수 없는 장소 ({locus_id})"

        parts: list[str] = []

        # 기본 정보
        parts.append(f"{locus.locus_name}({locus.locus_id}).")

        # 위치 설명
        location_parts: list[str] = []
        if locus.building:
            location_parts.append(locus.building)
        if locus.floor:
            location_parts.append(locus.floor)

        if location_parts:
            parts.append(f"{' '.join(location_parts)}에 위치한 {locus.locus_type.value} 구역으로,")
        else:
            parts.append(f"{locus.locus_type.value} 구역으로,")

        # 위험 정보
        if locus.hazard_grade is not None and locus.hazard_grade > 0:
            parts.append(f"위험등급 {int(locus.hazard_grade)}.")
        elif locus.hazard_level:
            parts.append(f"위험수준 {locus.hazard_level}.")

        # 특수 위험
        if locus.is_confined_space:
            parts.append("밀폐공간으로 2인 이상 필수, 단독 진입 금지.")
        elif locus.requires_team:
            parts.append("단독작업 금지 구역.")

        # 체류 특성
        dwell_desc = {
            DwellCategory.TRANSIT: "통과 지점",
            DwellCategory.SHORT_STAY: "단시간 체류 공간",
            DwellCategory.LONG_STAY: "장시간 체류 공간",
            DwellCategory.HAZARD_ZONE: "위험 구역 (체류 최소화 대상)",
            DwellCategory.ADMIN: "사무/행정 구역",
        }
        dwell_text = dwell_desc.get(locus.dwell_category, "")
        if dwell_text:
            parts.append(f"{dwell_text}이며")

        # S-Ward 정보
        if locus.sward_count > 0:
            parts.append(f"S-Ward {locus.sward_count}개가 매핑되어 있다.")
        elif locus.sward_ids:
            parts.append(f"S-Ward {len(locus.sward_ids)}개가 매핑되어 있다.")
        else:
            parts.append("S-Ward 미매핑.")

        # 도메인 확장 속성 (건설)
        ext = self._domain_extensions.get(locus_id)
        if ext is not None:
            ext_details: list[str] = []
            if ext.height_risk:
                ext_details.append("고소작업 위험")
            if ext.work_permit_required:
                ext_details.append("작업허가서 필요")
            if ext.minimum_crew >= 2:
                ext_details.append(f"최소 {ext.minimum_crew}인 작업")
            if ext_details:
                parts.append(f"건설 속성: {', '.join(ext_details)}.")

        # 상세 설명 (있으면 추가)
        if locus.description:
            parts.append(locus.description)

        return " ".join(parts)

    def to_natural_language_brief(self, locus_id: str) -> str:
        """Locus를 간략 자연어로 변환한다 (한 줄).

        예: "FAB 1F 작업구역 (WORK, 위험등급3)"

        Args:
            locus_id: Locus 식별자

        Returns:
            간략 설명 문자열
        """
        locus = self._loci.get(locus_id)
        if locus is None:
            return f"알 수 없음 ({locus_id})"

        parts = [locus.locus_name]
        parts.append(f"({locus.locus_type.value}")

        if locus.hazard_grade is not None and locus.hazard_grade > 0:
            parts.append(f", 위험등급{int(locus.hazard_grade)})")
        else:
            parts.append(")")

        return "".join(parts)

    # ─── 속성 ──────────────────────────────────────────────────────

    @property
    def all_locus_ids(self) -> list[str]:
        """등록된 모든 Locus ID 목록."""
        return sorted(self._loci.keys())

    @property
    def vocab_size(self) -> int:
        """등록된 Locus 수."""
        return len(self._loci)

    @property
    def sward_map_size(self) -> int:
        """S-Ward → Locus 매핑 수."""
        return len(self._sward_map)

    def __len__(self) -> int:
        return len(self._loci)

    def __contains__(self, locus_id: str) -> bool:
        return locus_id in self._loci

    def __iter__(self):
        return iter(self._loci.values())

    def summary(self) -> dict:
        """레지스트리 요약 정보를 반환한다.

        Returns:
            {total, by_type, by_building, sward_mapped, hazardous, confined}
        """
        by_type: dict[str, int] = {}
        by_building: dict[str, int] = {}
        hazardous_count = 0
        confined_count = 0

        for loc in self._loci.values():
            by_type[loc.locus_type.value] = by_type.get(loc.locus_type.value, 0) + 1
            if loc.building:
                by_building[loc.building] = by_building.get(loc.building, 0) + 1
            if loc.is_hazardous:
                hazardous_count += 1
            if loc.is_confined_space:
                confined_count += 1

        return {
            "sector_id": self.sector_id,
            "total": len(self._loci),
            "by_type": by_type,
            "by_building": by_building,
            "sward_mapped": len(self._sward_map),
            "hazardous": hazardous_count,
            "confined": confined_count,
        }


# ─── 팩토리 함수 (캐시) ──────────────────────────────────────────


def get_registry(
    sector_id: str,
    *,
    include_domain_attrs: bool = True,
    include_adjacency: bool = True,
) -> LocusRegistry:
    """
    Sector별 LocusRegistry 싱글턴을 반환한다.

    Streamlit 앱에서는 @st.cache_resource로 캐시되어
    세션 간 재사용된다.

    Args:
        sector_id: Sector 식별자 (Y1_SKHynix 등)
        include_domain_attrs: domain_packs/ locus_attrs.csv 로드 여부
        include_adjacency: adjacency CSV 로드 여부

    Returns:
        초기화된 LocusRegistry 인스턴스

    Usage:
        # Streamlit 앱에서
        from core.registry.locus_registry import get_registry
        registry = get_registry("Y1_SKHynix")

        # 비-Streamlit 환경에서 (캐시 없음)
        registry = _create_registry("Y1_SKHynix")
    """
    try:
        import streamlit as st
        return _get_registry_cached(sector_id, include_domain_attrs, include_adjacency)
    except ImportError:
        # Streamlit 없는 환경 (테스트, 스크립트)
        return _create_registry(sector_id, include_domain_attrs, include_adjacency)


def _create_registry(
    sector_id: str,
    include_domain_attrs: bool = True,
    include_adjacency: bool = True,
) -> LocusRegistry:
    """Registry 생성 로직 (캐시 없음)."""
    import sys
    from pathlib import Path

    # config.py import
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    try:
        import config as cfg
    except ImportError:
        logger.error("config.py를 import할 수 없습니다.")
        return LocusRegistry(sector_id)

    paths = cfg.get_sector_paths(sector_id)
    registry = LocusRegistry(sector_id)

    # 1. locus CSV 로드 (LOCUS_VERSION에 따라 v1 또는 v2)
    locus_version = getattr(cfg, "LOCUS_VERSION", "v1")
    if locus_version == "v2":
        locus_csv = paths.get("locus_v2_csv")
        if locus_csv and locus_csv.exists():
            try:
                registry.load_from_csv(locus_csv)
            except Exception as e:
                logger.warning("locus_v2.csv 로드 실패, v1 fallback: %s", e)
                locus_csv = paths.get("locus_csv")
                if locus_csv and locus_csv.exists():
                    registry.load_from_csv(locus_csv)
        else:
            logger.warning("locus_v2.csv 없음, v1 fallback: %s", locus_csv)
            locus_csv = paths.get("locus_csv")
            if locus_csv and locus_csv.exists():
                registry.load_from_csv(locus_csv)
    else:
        locus_csv = paths.get("locus_csv")
        if locus_csv and locus_csv.exists():
            try:
                registry.load_from_csv(locus_csv)
            except Exception as e:
                logger.warning("locus.csv 로드 실패: %s", e)
        else:
            logger.warning("locus.csv 없음: %s", locus_csv)

    # 2. adjacency 로드
    if include_adjacency:
        adj_csv = paths.get("adjacency")
        if adj_csv and adj_csv.exists():
            try:
                registry.load_adjacency(adj_csv)
            except Exception as e:
                logger.warning("adjacency 로드 실패: %s", e)

    # 3. domain_packs/ 속성 로드
    if include_domain_attrs:
        sector_info = cfg.SECTOR_REGISTRY.get(sector_id, {})
        domain = sector_info.get("domain", "construction")

        try:
            from core.registry.domain_loader import get_domain_pack_path
            domain_pack_path = get_domain_pack_path(domain)

            # LOCUS_VERSION에 따라 attrs 파일 선택
            if locus_version == "v2":
                domain_attrs_csv = domain_pack_path / "locus_attrs_v2.csv"
                if not domain_attrs_csv.exists():
                    domain_attrs_csv = domain_pack_path / "locus_attrs.csv"
            else:
                domain_attrs_csv = domain_pack_path / "locus_attrs.csv"

            if domain_attrs_csv.exists():
                registry.load_domain_attrs(domain_attrs_csv)
        except FileNotFoundError:
            logger.info("도메인 팩 없음: %s (속성 로드 생략)", domain)
        except Exception as e:
            logger.warning("도메인 속성 로드 실패: %s", e)

    return registry


# Streamlit 캐시 래퍼 (별도 함수로 분리해야 @st.cache_resource 작동)
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _get_registry_cached(
        sector_id: str,
        include_domain_attrs: bool,
        include_adjacency: bool,
    ) -> LocusRegistry:
        """Streamlit 캐시 래퍼."""
        return _create_registry(sector_id, include_domain_attrs, include_adjacency)

except ImportError:
    # Streamlit 없는 환경 — 더미 함수 정의
    def _get_registry_cached(
        sector_id: str,
        include_domain_attrs: bool,
        include_adjacency: bool,
    ) -> LocusRegistry:
        """Streamlit 없는 환경 fallback."""
        return _create_registry(sector_id, include_domain_attrs, include_adjacency)
