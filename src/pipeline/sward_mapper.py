"""
S-Ward Mapper -- 좌표 기반 최근접 Gateway 할당 모듈
====================================================
TWardData의 (building, level, x, y) 좌표를 가장 가까운
S-Ward(Gateway)에 할당하여 Locus v2 ID를 반환한다.

핵심 개념:
  - GatewayIndex: building+level별 Gateway 좌표를 numpy 배열로 인덱싱
  - assign_nearest_sward: 단일 좌표 -> 최근접 Gateway
  - assign_batch: DataFrame 전체 -> 벡터화 배치 할당

사용:
    from src.pipeline.sward_mapper import GatewayIndex

    index = GatewayIndex.from_csv(gateway_csv_path)
    result = index.assign_single("FAB", "1F", 100.0, 200.0)
    print(result.locus_id, result.distance)

    df = index.assign_batch(journey_df)
    # -> "gw_locus_id", "gw_distance", "gw_confidence" 컬럼 추가
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 결과 데이터 클래스 ─────────────────────────────────────────────

@dataclass
class AssignmentResult:
    """최근접 Gateway 할당 결과."""

    gateway_no: int
    locus_id: str           # "GW-{gateway_no}"
    distance: float         # Euclidean distance
    confidence: str         # "high" | "medium" | "low"
    match_type: str         # "exact" | "inferred" | "gl_fallback" | "unmatched"


# ─── 층 이름 정규화 ─────────────────────────────────────────────────

# 층 표기 정규화: B1 -> B1F, b1f -> B1F, 2f -> 2F 등
_LEVEL_ALIASES: dict[str, str] = {
    "b1": "B1F",
    "b2": "B2F",
    "b1f": "B1F",
    "b2f": "B2F",
    "b1mf": "B1MF",
    "rf": "RF",
    "gl": "GL",
}


def normalize_level(level: str | None) -> str | None:
    """
    층 이름 정규화.

    B1 -> B1F, b1f -> B1F, 2f -> 2F, None -> None
    지원동의 B1/B2를 B1F/B2F로 통일한다.
    """
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return None
    level_str = str(level).strip()
    if not level_str:
        return None
    key = level_str.lower()
    if key in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[key]
    # 숫자+F 형태 (1F, 2F 등) -> 대문자
    return level_str.upper()


# ─── Building 추론 (Place 키워드) ────────────────────────────────────

# Place 키워드 -> building_name 매핑
_PLACE_BUILDING_MAP: list[tuple[str, str]] = [
    ("fab", "FAB"),
    ("cub", "CUB"),
    ("wwt", "WWT"),
    ("wtp", "WWT"),          # Water Treatment Plant
    ("154kv", "154kV"),
    ("154k", "154kV"),
    ("변전", "154kV"),
    ("저수조", "저수조"),
    ("비상발전", "비상발전기동"),
    ("지원_1bl", "지원_1BL"),
    ("지원_2bl", "지원_2BL"),
    ("지원_3bl", "지원_3BL"),
    ("1bl", "지원_1BL"),
    ("2bl", "지원_2BL"),
    ("3bl", "지원_3BL"),
]

# 야외/본진/전진 등은 building이 없는 sector-level 위치
_PLACE_GL_KEYWORDS: list[str] = [
    "타각기", "본진", "전진", "공사현장", "gate", "게이트",
    "흡연", "휴게", "보호구", "주차", "호이스트",
]


def infer_building_from_place(place: str | None) -> tuple[str | None, bool]:
    """
    Place(spot_name) 키워드에서 건물명을 추론한다.

    Returns:
        (building_name, is_outdoor)
        - building_name: 추론된 건물명 또는 None
        - is_outdoor: 야외/GL 위치 여부
    """
    if place is None or (isinstance(place, float) and np.isnan(place)):
        return None, True
    place_lower = str(place).strip().lower()
    if not place_lower:
        return None, True

    # 건물 키워드 매칭
    for keyword, building in _PLACE_BUILDING_MAP:
        if keyword in place_lower:
            return building, False

    # 야외/GL 키워드 매칭
    for keyword in _PLACE_GL_KEYWORDS:
        if keyword in place_lower:
            return None, True

    # 알 수 없음 (야외로 간주)
    return None, True


# ─── 거리 임계값 ────────────────────────────────────────────────────

DISTANCE_HIGH_CONFIDENCE = 30.0    # < 30: 높은 확신
DISTANCE_MEDIUM_CONFIDENCE = 100.0  # 30~100: 중간 확신
# > 100: 낮은 확신


def _confidence_from_distance(distance: float, match_type: str) -> str:
    """거리와 매칭 유형에서 확신도를 결정한다."""
    if match_type == "unmatched":
        return "low"
    if match_type == "gl_fallback":
        return "medium" if distance < DISTANCE_MEDIUM_CONFIDENCE else "low"
    if distance < DISTANCE_HIGH_CONFIDENCE:
        return "high"
    if distance < DISTANCE_MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


# ─── GatewayIndex ────────────────────────────────────────────────────

class GatewayIndex:
    """
    Building+Level별 Gateway 좌표 인덱스.

    각 (building, level) 그룹의 Gateway 좌표를 numpy 배열로 저장하여
    cdist 기반 벡터화 최근접 탐색을 지원한다.

    Attributes:
        _index: {(building, level): np.ndarray shape (N, 3)} -- [gw_no, x, y]
        _gw_meta: {gateway_no: {gateway_name, building_name, floor_name, ...}}
        _gl_index: np.ndarray -- GL(야외) Gateway 좌표 (fallback용)
    """

    def __init__(self) -> None:
        self._index: dict[tuple[str, str], np.ndarray] = {}
        self._gw_meta: dict[int, dict] = {}
        self._gl_index: np.ndarray | None = None
        self._all_gw_coords: np.ndarray | None = None  # 전체 Gateway [gw_no, x, y]

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        encoding: str = "euc-kr",
    ) -> "GatewayIndex":
        """
        GatewayPoint CSV에서 인덱스를 구축한다.

        Args:
            csv_path: Y1_GatewayPoint_YYYYMMDD.csv 경로
            encoding: CSV 인코딩 (기본 euc-kr)

        Returns:
            초기화된 GatewayIndex
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Gateway CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, encoding=encoding)

        required_cols = {"gateway_no", "gateway_name", "building_name", "floor_name",
                         "location_x", "location_y"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        index = cls()

        for _, row in df.iterrows():
            gw_no = int(row["gateway_no"])
            gw_name = str(row["gateway_name"]).strip()
            building = row["building_name"]
            floor = row["floor_name"]
            x = float(row["location_x"]) if pd.notna(row["location_x"]) else None
            y = float(row["location_y"]) if pd.notna(row["location_y"]) else None

            if x is None or y is None:
                logger.warning("Gateway %d (%s): 좌표 없음, 인덱스 제외", gw_no, gw_name)
                continue

            # 메타 저장
            building_str = str(building).strip() if pd.notna(building) else None
            floor_str = normalize_level(str(floor).strip()) if pd.notna(floor) else None

            index._gw_meta[gw_no] = {
                "gateway_name": gw_name,
                "building_name": building_str,
                "building_no": int(row["building_no"]) if pd.notna(row.get("building_no")) else None,
                "floor_name": floor_str,
                "floor_no": int(row["floor_no"]) if pd.notna(row.get("floor_no")) else None,
                "location_x": x,
                "location_y": y,
            }

            # building+level 키
            key = (building_str, floor_str)
            if key not in index._index:
                index._index[key] = []
            index._index[key].append([gw_no, x, y])

        # list -> numpy array 변환
        for key in index._index:
            index._index[key] = np.array(index._index[key], dtype=np.float64)

        # 전체 좌표 배열 (마지막 fallback용)
        all_coords = []
        for arr in index._index.values():
            all_coords.append(arr)
        if all_coords:
            index._all_gw_coords = np.vstack(all_coords)
        else:
            index._all_gw_coords = np.empty((0, 3))

        total_gw = sum(len(v) for v in index._index.values())
        logger.info(
            "GatewayIndex 구축: %d개 Gateway, %d개 building+level 그룹",
            total_gw,
            len(index._index),
        )

        return index

    # ─── 단일 할당 ─────────────────────────────────────────────────

    def assign_single(
        self,
        building: str | None,
        level: str | None,
        x: float,
        y: float,
        place: str | None = None,
    ) -> AssignmentResult:
        """
        단일 좌표를 최근접 Gateway에 할당한다.

        전략:
          1. building+level이 모두 있으면 -> 해당 그룹에서 탐색
          2. building이 NaN이면 -> Place에서 추론 시도
          3. 추론 실패 -> 전체 Gateway에서 탐색 (GL fallback)

        Args:
            building: 건물명 (NaN 가능)
            level: 층명 (NaN 가능)
            x: X 좌표
            y: Y 좌표
            place: spot_name (building 추론용)

        Returns:
            AssignmentResult
        """
        if np.isnan(x) or np.isnan(y):
            return AssignmentResult(
                gateway_no=-1,
                locus_id="unmatched",
                distance=float("inf"),
                confidence="low",
                match_type="unmatched",
            )

        point = np.array([[x, y]])

        # Strategy 1: building+level 정확 매칭
        building_str = str(building).strip() if building is not None and not (isinstance(building, float) and np.isnan(building)) else None
        level_str = normalize_level(level)

        if building_str and level_str:
            key = (building_str, level_str)
            if key in self._index:
                return self._find_nearest(point, self._index[key], "exact")

        # Strategy 2: Place에서 building 추론
        if building_str is None and place is not None:
            inferred_building, is_outdoor = infer_building_from_place(place)
            if inferred_building and level_str:
                key = (inferred_building, level_str)
                if key in self._index:
                    return self._find_nearest(point, self._index[key], "inferred")
            # building 추론 성공, level 없음 -> 해당 building의 전체 층 탐색
            if inferred_building:
                building_gws = self._get_building_gateways(inferred_building)
                if len(building_gws) > 0:
                    return self._find_nearest(point, building_gws, "inferred")

        # Strategy 3: 전체 Gateway에서 탐색 (GL fallback)
        if self._all_gw_coords is not None and len(self._all_gw_coords) > 0:
            return self._find_nearest(point, self._all_gw_coords, "gl_fallback")

        return AssignmentResult(
            gateway_no=-1,
            locus_id="unmatched",
            distance=float("inf"),
            confidence="low",
            match_type="unmatched",
        )

    def _find_nearest(
        self,
        point: np.ndarray,
        gw_array: np.ndarray,
        match_type: str,
    ) -> AssignmentResult:
        """gw_array에서 point에 가장 가까운 Gateway를 찾는다."""
        coords = gw_array[:, 1:3]  # x, y
        dists = np.sqrt(np.sum((coords - point) ** 2, axis=1))
        idx = np.argmin(dists)
        gw_no = int(gw_array[idx, 0])
        dist = float(dists[idx])
        confidence = _confidence_from_distance(dist, match_type)

        return AssignmentResult(
            gateway_no=gw_no,
            locus_id=f"GW-{gw_no}",
            distance=round(dist, 2),
            confidence=confidence,
            match_type=match_type,
        )

    def _get_building_gateways(self, building: str) -> np.ndarray:
        """특정 건물의 모든 Gateway를 반환한다."""
        arrays = []
        for (b, _), arr in self._index.items():
            if b == building:
                arrays.append(arr)
        if arrays:
            return np.vstack(arrays)
        return np.empty((0, 3))

    # ─── 배치 할당 ─────────────────────────────────────────────────

    def assign_batch(
        self,
        df: pd.DataFrame,
        building_col: str = "Building",
        level_col: str = "Level",
        x_col: str = "X",
        y_col: str = "Y",
        place_col: str = "Place",
    ) -> pd.DataFrame:
        """
        DataFrame 전체에 최근접 Gateway를 배치 할당한다.

        building+level별로 그룹화하여 벡터화 거리 계산.
        결과 컬럼: gw_locus_id, gw_gateway_no, gw_distance, gw_confidence, gw_match_type

        Args:
            df: TWardData DataFrame (Building, Level, X, Y, Place 컬럼 필요)
            building_col: 건물 컬럼명
            level_col: 층 컬럼명
            x_col: X좌표 컬럼명
            y_col: Y좌표 컬럼명
            place_col: Place 컬럼명

        Returns:
            원본 DataFrame에 gw_* 컬럼 추가
        """
        result = df.copy()
        n = len(result)

        # 결과 배열 초기화
        gw_nos = np.full(n, -1, dtype=np.int32)
        gw_dists = np.full(n, np.inf, dtype=np.float64)
        gw_match_types = np.full(n, "", dtype=object)

        # 컬럼 존재 확인 및 안전한 추출
        has_building = building_col in result.columns
        has_level = level_col in result.columns
        has_place = place_col in result.columns

        # X, Y를 float로 변환
        xs = pd.to_numeric(result[x_col], errors="coerce").values
        ys = pd.to_numeric(result[y_col], errors="coerce").values

        # Building+Level 준비
        if has_building:
            buildings = result[building_col].values
        else:
            buildings = np.full(n, None, dtype=object)

        if has_level:
            levels = result[level_col].apply(lambda v: normalize_level(v) if pd.notna(v) else None).values
        else:
            levels = np.full(n, None, dtype=object)

        if has_place:
            places = result[place_col].values
        else:
            places = np.full(n, None, dtype=object)

        # Phase 1: building+level 정확 매칭 (그룹별 벡터화)
        # building+level 키 생성
        bl_keys = []
        for i in range(n):
            b = str(buildings[i]).strip() if pd.notna(buildings[i]) else None
            l = levels[i]
            bl_keys.append((b, l))

        # 유효한 좌표 마스크
        valid_mask = ~(np.isnan(xs) | np.isnan(ys))

        # 그룹별 배치 처리
        group_map: dict[tuple, list[int]] = {}
        for i in range(n):
            if not valid_mask[i]:
                gw_match_types[i] = "unmatched"
                continue
            key = bl_keys[i]
            if key[0] is not None and key[1] is not None:
                group_map.setdefault(key, []).append(i)

        processed = np.zeros(n, dtype=bool)

        from scipy.spatial import cKDTree

        for key, indices in group_map.items():
            if key not in self._index:
                continue
            gw_arr = self._index[key]
            gw_coords = gw_arr[:, 1:3]
            tree = cKDTree(gw_coords)

            idx_arr = np.array(indices)
            points = np.column_stack([xs[idx_arr], ys[idx_arr]])

            min_dist, min_idx = tree.query(points, k=1)

            gw_nos[idx_arr] = gw_arr[min_idx, 0].astype(np.int32)
            gw_dists[idx_arr] = min_dist
            gw_match_types[idx_arr] = "exact"
            processed[idx_arr] = True

        # Phase 2: 미처리 레코드 -> Place 추론 (배치) + GL fallback (배치)
        unprocessed_mask = valid_mask & ~processed
        unprocessed_indices = np.where(unprocessed_mask)[0]

        if len(unprocessed_indices) > 0:
            # Phase 2a: Place에서 building 추론하여 그룹별 배치 처리
            inferred_groups: dict[tuple, list[int]] = {}
            remaining_indices: list[int] = []

            for i in unprocessed_indices:
                place = str(places[i]) if has_place and pd.notna(places[i]) else None
                inferred_building, _ = infer_building_from_place(place)

                if inferred_building:
                    lvl = levels[i]
                    if lvl:
                        key = (inferred_building, lvl)
                        if key in self._index:
                            inferred_groups.setdefault(key, []).append(i)
                            continue
                    # level 없으면 building 전체에서 탐색
                    bld_key = (inferred_building, "__all__")
                    inferred_groups.setdefault(bld_key, []).append(i)
                    continue

                remaining_indices.append(i)

            # 추론된 그룹 배치 처리
            for key, indices in inferred_groups.items():
                bld, lvl = key
                if lvl == "__all__":
                    gw_arr = self._get_building_gateways(bld)
                else:
                    gw_arr = self._index.get(key)

                if gw_arr is None or len(gw_arr) == 0:
                    remaining_indices.extend(indices)
                    continue

                gw_coords = gw_arr[:, 1:3]
                tree = cKDTree(gw_coords)
                idx_arr = np.array(indices)
                points = np.column_stack([xs[idx_arr], ys[idx_arr]])

                min_dist, min_idx = tree.query(points, k=1)

                gw_nos[idx_arr] = gw_arr[min_idx, 0].astype(np.int32)
                gw_dists[idx_arr] = min_dist
                gw_match_types[idx_arr] = "inferred"
                processed[idx_arr] = True

            # Phase 2b: GL fallback (전체 Gateway에서 KDTree 배치 탐색)
            remaining_arr = np.array(remaining_indices) if remaining_indices else np.array([], dtype=int)
            if len(remaining_arr) > 0 and self._all_gw_coords is not None and len(self._all_gw_coords) > 0:
                from scipy.spatial import cKDTree

                gw_coords = self._all_gw_coords[:, 1:3]
                tree = cKDTree(gw_coords)
                points = np.column_stack([xs[remaining_arr], ys[remaining_arr]])

                min_dist, min_idx = tree.query(points, k=1)

                gw_nos[remaining_arr] = self._all_gw_coords[min_idx, 0].astype(np.int32)
                gw_dists[remaining_arr] = min_dist
                gw_match_types[remaining_arr] = "gl_fallback"

        # 결과 컬럼 추가
        result["gw_gateway_no"] = gw_nos
        result["gw_locus_id"] = [
            f"GW-{gw}" if gw >= 0 else "unmatched"
            for gw in gw_nos
        ]
        result["gw_distance"] = np.round(gw_dists, 2)
        result["gw_match_type"] = gw_match_types
        result["gw_confidence"] = [
            _confidence_from_distance(d, mt) if mt else "low"
            for d, mt in zip(gw_dists, gw_match_types)
        ]

        # 통계 로깅
        match_counts = pd.Series(gw_match_types).value_counts()
        unmatched_pct = (gw_nos == -1).mean() * 100
        logger.info(
            "GatewayIndex.assign_batch: %d건 처리, unmatched %.1f%%, "
            "match_type: %s",
            n,
            unmatched_pct,
            match_counts.to_dict(),
        )

        if unmatched_pct > 1.0:
            logger.warning("unmatched 비율 %.1f%% (>1%%)", unmatched_pct)

        return result

    # ─── 정보 조회 ─────────────────────────────────────────────────

    def get_gateway_meta(self, gateway_no: int) -> dict | None:
        """Gateway 메타데이터를 반환한다."""
        return self._gw_meta.get(gateway_no)

    @property
    def gateway_count(self) -> int:
        """인덱싱된 Gateway 수."""
        return len(self._gw_meta)

    @property
    def group_count(self) -> int:
        """building+level 그룹 수."""
        return len(self._index)

    def summary(self) -> dict:
        """인덱스 요약."""
        return {
            "gateway_count": self.gateway_count,
            "group_count": self.group_count,
            "groups": {
                f"{b}_{l}": len(arr)
                for (b, l), arr in sorted(self._index.items(), key=lambda x: str(x[0]))
            },
        }


# ─── 편의 함수 ──────────────────────────────────────────────────────


def get_gateway_index(
    sector_id: str = "Y1_SKHynix",
) -> GatewayIndex | None:
    """
    Sector의 GatewayIndex를 로드한다.

    config.py의 경로에서 Gateway CSV를 찾아 인덱스를 구축한다.

    Args:
        sector_id: Sector ID

    Returns:
        GatewayIndex 또는 None (CSV 없으면)
    """
    try:
        import config as cfg
        paths = cfg.get_sector_paths(sector_id)
        gateway_csv = paths.get("gateway_csv")
        if gateway_csv and gateway_csv.exists():
            return GatewayIndex.from_csv(gateway_csv)

        # fallback: New_SSMP 디렉토리 직접 탐색
        spatial_dir = paths.get("spatial_dir")
        if spatial_dir:
            for csv_file in sorted(spatial_dir.rglob("*GatewayPoint*.csv")):
                try:
                    return GatewayIndex.from_csv(csv_file)
                except Exception:
                    continue

        logger.warning("Gateway CSV not found for sector %s", sector_id)
        return None
    except Exception as e:
        logger.warning("GatewayIndex 로드 실패: %s", e)
        return None
