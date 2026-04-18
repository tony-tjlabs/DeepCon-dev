"""
Spatial Graph — Locus 인접 관계 그래프 모듈
============================================
NetworkX 기반으로 Locus 간 인접 관계를 그래프로 구축하여
경로 분석, 비정상 이동 탐지, 이동 비용 계산을 지원한다.

핵심 기능:
  - 최단 경로 계산 (Dijkstra)
  - 인접 여부 판별
  - 비정상 이동(비인접 직접 이동) 탐지
  - 경로 비용 계산

사용법:
    graph = get_spatial_graph("Y1_SKHynix")
    path = graph.shortest_path("GW-351", "GW-233")   # v2 locus_id
    is_adj = graph.is_adjacent("GW-351", "GW-352")
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialGraph:
    """
    Locus 간 인접 관계 그래프.

    노드: Locus (locus_id + 속성)
    엣지: 인접 관계 (weight = transition_cost_min)
    """

    def __init__(
        self,
        adjacency_df: pd.DataFrame,
        locus_df: pd.DataFrame,
    ):
        """
        Args:
            adjacency_df: locus_adjacency.csv 데이터
                필수 컬럼: from_locus_id, to_locus_id
                선택 컬럼: transition_cost_min, connector_type, direction, bidirectional
            locus_df: locus.csv 데이터 (노드 속성용)
                필수 컬럼: locus_id
        """
        self.G: nx.Graph = nx.Graph()
        self._locus_ids: set[str] = set()
        self._build(adjacency_df, locus_df)

    def _build(
        self,
        adj_df: pd.DataFrame,
        locus_df: pd.DataFrame,
    ) -> None:
        """그래프 구축."""
        # 노드 추가 (locus_id + 속성)
        for _, row in locus_df.iterrows():
            locus_id = str(row.get("locus_id", ""))
            if not locus_id:
                continue
            self._locus_ids.add(locus_id)
            # NaN 제거한 속성 딕셔너리
            attrs = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            self.G.add_node(locus_id, **attrs)

        # 엣지 추가 (transition_cost_min = weight)
        if adj_df.empty:
            logger.warning("Adjacency 데이터가 비어 있음")
            return

        for _, row in adj_df.iterrows():
            from_locus = str(row.get("from_locus_id", ""))
            to_locus = str(row.get("to_locus_id", ""))

            if not from_locus or not to_locus:
                continue

            weight = float(row.get("transition_cost_min", 1.0))
            connector_type = str(row.get("connector_type", "open"))
            direction = str(row.get("direction", "horizontal"))
            bidirectional = row.get("bidirectional", True)

            # bidirectional이 문자열인 경우 처리
            if isinstance(bidirectional, str):
                bidirectional = bidirectional.lower() in ("true", "1", "yes")

            self.G.add_edge(
                from_locus,
                to_locus,
                weight=weight,
                connector_type=connector_type,
                direction=direction,
            )

        logger.info(
            f"SpatialGraph 구축 완료: {self.G.number_of_nodes()} 노드, "
            f"{self.G.number_of_edges()} 엣지"
        )

    @property
    def node_count(self) -> int:
        """노드 수."""
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """엣지 수."""
        return self.G.number_of_edges()

    def shortest_path(self, from_locus: str, to_locus: str) -> list[str]:
        """
        최단 경로 반환 (Dijkstra).

        Returns:
            경로가 존재하면 [from_locus, ..., to_locus]
            경로가 없으면 []
        """
        if from_locus not in self.G or to_locus not in self.G:
            return []
        try:
            return nx.shortest_path(self.G, from_locus, to_locus, weight="weight")
        except nx.NetworkXNoPath:
            return []

    def shortest_path_length(self, from_locus: str, to_locus: str) -> float:
        """최단 경로 길이 (비용 합계)."""
        if from_locus not in self.G or to_locus not in self.G:
            return float("inf")
        try:
            return nx.shortest_path_length(self.G, from_locus, to_locus, weight="weight")
        except nx.NetworkXNoPath:
            return float("inf")

    def is_adjacent(self, locus_a: str, locus_b: str) -> bool:
        """직접 연결 여부."""
        return self.G.has_edge(locus_a, locus_b)

    def get_neighbors(self, locus_id: str) -> list[str]:
        """인접 Locus 목록."""
        if locus_id not in self.G:
            return []
        return list(self.G.neighbors(locus_id))

    def path_cost(self, path: list[str]) -> float:
        """경로 비용 합계."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            edge_data = self.G.get_edge_data(path[i], path[i + 1])
            if edge_data:
                total += edge_data.get("weight", 1.0)
            else:
                # 연결되지 않은 경로
                total += float("inf")
        return total

    def get_node_attribute(self, locus_id: str, attr: str, default=None):
        """노드 속성 조회."""
        if locus_id not in self.G:
            return default
        return self.G.nodes[locus_id].get(attr, default)

    def detect_impossible_transitions(
        self,
        journey_df: pd.DataFrame,
        prev_locus_col: str = "prev_locus",
        locus_col: str = "locus_id",
    ) -> pd.DataFrame:
        """
        비인접 직접 이동 탐지.

        Args:
            journey_df: 이동 데이터
            prev_locus_col: 이전 Locus 컬럼명
            locus_col: 현재 Locus 컬럼명

        Returns:
            비정상 이동 행만 필터링한 DataFrame
        """
        if journey_df.empty:
            return pd.DataFrame()

        if prev_locus_col not in journey_df.columns:
            return pd.DataFrame()

        # 이전 locus와 현재 locus가 다르고, 인접하지 않은 경우
        def _is_invalid(row):
            prev = row.get(prev_locus_col, "")
            curr = row.get(locus_col, "")
            if not prev or not curr or prev == curr:
                return False
            # 둘 다 그래프에 있어야 함
            if prev not in self._locus_ids or curr not in self._locus_ids:
                return False
            return not self.is_adjacent(prev, curr)

        mask = journey_df.apply(_is_invalid, axis=1)
        return journey_df[mask].copy()

    def get_building_subgraph(self, building: str) -> "SpatialGraph":
        """특정 건물의 서브그래프 추출."""
        nodes = [
            n for n in self.G.nodes
            if self.G.nodes[n].get("building") == building
        ]
        subgraph = self.G.subgraph(nodes).copy()

        # 새 SpatialGraph 객체 생성 (빈 데이터로 초기화 후 그래프 교체)
        result = SpatialGraph(pd.DataFrame(), pd.DataFrame())
        result.G = subgraph
        result._locus_ids = set(nodes)
        return result

    def get_connected_components(self) -> list[set[str]]:
        """연결 컴포넌트 목록."""
        return [set(c) for c in nx.connected_components(self.G)]

    def to_dict(self) -> dict:
        """그래프 정보 딕셔너리 (디버깅용)."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "components": len(self.get_connected_components()),
            "nodes": list(self._locus_ids)[:10],  # 샘플
        }


# ─── 캐시된 싱글턴 접근 ─────────────────────────────────────────────────


@lru_cache(maxsize=4)
def get_spatial_graph(sector_id: str) -> SpatialGraph:
    """
    Sector별 SpatialGraph 싱글턴 캐시.

    Usage:
        graph = get_spatial_graph("Y1_SKHynix")
    """
    from src.spatial.loader import load_adjacency_df, load_locus_df

    adj_df = load_adjacency_df(sector_id)
    locus_df = load_locus_df(sector_id)

    return SpatialGraph(adj_df, locus_df)


def clear_graph_cache() -> None:
    """그래프 캐시 초기화."""
    get_spatial_graph.cache_clear()
