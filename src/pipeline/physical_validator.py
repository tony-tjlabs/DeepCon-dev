"""
물리적 이동 가능성 검증 (Physical Movement Feasibility)
=========================================================
1분 단위 journey에서 작업자의 locus 전이가 **물리적으로 가능한가**를 평가.

핵심 아이디어
  - 사람은 점프하지 못한다.
  - 같은 건물·층 내 수평 이동은 직선거리/보행속도로 상한이 있다.
  - 다른 층 이동은 계단·엘리베이터 시간을 포함해야 한다.
  - 다른 건물 이동은 야외 전이 시간이 필요하다.

모듈이 제공하는 기능
  - GatewayGeo: locus 중심 좌표/건물/층 lookup (캐시)
  - evaluate_transition(): 두 locus 사이 1분 전이의 물리 실현 가능성 평가
  - annotate_journey(): journey DataFrame 전체에 feasibility 컬럼 부착

feasibility 분류
  - OK          : 정상 이동 (동일 locus 유지 또는 인접 범위)
  - WARN_FAST   : 같은 건물·층에서 보행 한계 근접 (attention)
  - IMPOSSIBLE  : 보행으로 불가능한 속도 (플리커/오탐 의심)
  - CROSS_FLOOR : 층 변경 — 수직 이동 시간 내외로 판정
  - CROSS_BLDG  : 건물 변경 — 야외 전이 필요

보행 가정 (보수적):
  - 일반 보행 80 m/min, 빠른 걸음 120 m/min, 달리기 상한 200 m/min
  - 층 변경 최소 1분 (계단 1층), 엘리베이터 대기 포함 2~3분 전형
  - 건물 변경 최소 2분 (Y1 캠퍼스 내 인접 건물 기준)

좌표계 주의:
  - locus_v2의 location_x/y는 **층(floor) 단위 로컬 좌표계**.
    같은 (building, floor)에서만 직접 비교 가능.
    다른 (building, floor) 간 거리는 산출하지 않고 "카테고리 분류"로 처리.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 물리 상수 (보수적 추정) ──────────────────────────────────────
WALK_SPEED_NORMAL   = 80.0    # m/min, 일반 보행
WALK_SPEED_FAST     = 120.0   # m/min, 빠른 걸음
WALK_SPEED_MAX      = 200.0   # m/min, 달리기 상한 (건설현장 실질 상한)

MIN_FLOOR_CHANGE_MIN  = 1     # 층 변경 최소 시간 (계단 1층, 분)
MIN_BLDG_CHANGE_MIN   = 2     # 건물 변경 최소 시간 (분)

# feasibility 라벨
FEAS_OK         = "OK"
FEAS_WARN_FAST  = "WARN_FAST"
FEAS_IMPOSSIBLE = "IMPOSSIBLE"
FEAS_CROSS_FLOOR = "CROSS_FLOOR"
FEAS_CROSS_BLDG  = "CROSS_BLDG"

FEAS_ORDER = [FEAS_OK, FEAS_WARN_FAST, FEAS_CROSS_FLOOR, FEAS_CROSS_BLDG, FEAS_IMPOSSIBLE]


# ─── Gateway 좌표 인덱스 ──────────────────────────────────────────

@dataclass(frozen=True)
class LocusGeo:
    """locus 단위 위치 정보."""
    locus_id: str
    building: str
    floor: str
    x: float
    y: float
    building_no: float
    floor_no: float

    @property
    def has_coords(self) -> bool:
        return not (np.isnan(self.x) or np.isnan(self.y))


class GatewayGeo:
    """
    locus_v2.csv 기반 locus 위치 lookup.

    사용:
        geo = GatewayGeo.from_csv(Path("data/spatial_model/Y1/locus/locus_v2.csv"))
        info = geo.get("GW-256")
        dist = geo.horizontal_distance("GW-256", "GW-258")   # 동일 building+floor만 유효
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.set_index("locus_id")
        self._cache: dict[str, LocusGeo] = {}

    @classmethod
    def from_csv(cls, path: Path | str) -> "GatewayGeo":
        p = Path(path)
        if not p.exists():
            logger.warning("locus_v2.csv not found: %s", p)
            return cls(pd.DataFrame(columns=[
                "locus_id", "building", "floor",
                "location_x", "location_y", "building_no", "floor_no",
            ]).set_index("locus_id").reset_index())
        df = pd.read_csv(p)
        keep = ["locus_id", "building", "floor",
                "location_x", "location_y", "building_no", "floor_no"]
        df = df[[c for c in keep if c in df.columns]].copy()
        return cls(df)

    def get(self, locus_id: str) -> LocusGeo | None:
        if locus_id in self._cache:
            return self._cache[locus_id]
        if locus_id not in self._df.index:
            return None
        row = self._df.loc[locus_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        info = LocusGeo(
            locus_id=str(locus_id),
            building=str(row.get("building", "") or ""),
            floor=str(row.get("floor", "") or ""),
            x=float(row.get("location_x", np.nan)),
            y=float(row.get("location_y", np.nan)),
            building_no=float(row.get("building_no", np.nan)),
            floor_no=float(row.get("floor_no", np.nan)),
        )
        self._cache[locus_id] = info
        return info

    def horizontal_distance(self, a: str, b: str) -> float | None:
        """동일 (building, floor)에서의 유클리드 거리 (meter, 좌표단위 = m 가정)."""
        ga, gb = self.get(a), self.get(b)
        if ga is None or gb is None:
            return None
        if ga.building != gb.building or ga.floor != gb.floor:
            return None
        if not ga.has_coords or not gb.has_coords:
            return None
        return float(np.hypot(ga.x - gb.x, ga.y - gb.y))


# ─── 전이 평가 ────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransitionEval:
    """한 전이의 물리적 평가 결과."""
    prev_locus: str
    curr_locus: str
    dt_min: float
    category: str            # FEAS_* 상수
    distance_m: float | None  # 동일 층 내에서만 유효
    speed_m_per_min: float | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "prev_locus":      self.prev_locus,
            "curr_locus":      self.curr_locus,
            "dt_min":          self.dt_min,
            "feasibility":     self.category,
            "distance_m":      self.distance_m,
            "speed_m_per_min": self.speed_m_per_min,
            "reason":          self.reason,
        }


def evaluate_transition(
    prev_locus: str,
    curr_locus: str,
    dt_min: float,
    geo: GatewayGeo,
) -> TransitionEval:
    """
    한 쌍의 전이가 물리적으로 가능한지 평가.

    Args:
        prev_locus: 이전 locus_id
        curr_locus: 현재 locus_id
        dt_min:     이전→현재 소요 분 (1분 단위 일반적으로 1.0)
        geo:        GatewayGeo 인스턴스
    """
    # 동일 locus 또는 결측 → 자동 OK
    if prev_locus == curr_locus:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, 0.0, 0.0, "동일 locus 유지")
    if not prev_locus or not curr_locus or dt_min <= 0:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, None, None, "정보 부족 — 판정 생략")

    ga, gb = geo.get(prev_locus), geo.get(curr_locus)
    if ga is None or gb is None:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, None, None, "locus 좌표 미등록")

    # 결측 건물·층 정보는 판정 생략 (야외/경계 locus)
    def _missing(s: str) -> bool:
        return (not s) or s.lower() in ("nan", "none", "unknown")

    if _missing(ga.building) or _missing(gb.building):
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, None, None, "건물 정보 결측 — 판정 생략")

    # 건물 변경
    if ga.building != gb.building:
        if dt_min < MIN_BLDG_CHANGE_MIN:
            return TransitionEval(prev_locus, curr_locus, dt_min,
                                  FEAS_IMPOSSIBLE, None, None,
                                  f"건물 변경({ga.building}→{gb.building})이 {dt_min:.0f}분 내 불가 — 최소 {MIN_BLDG_CHANGE_MIN}분 필요")
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_CROSS_BLDG, None, None,
                              f"건물 변경 {ga.building}→{gb.building} ({dt_min:.0f}분)")

    if _missing(ga.floor) or _missing(gb.floor):
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, None, None, "층 정보 결측 — 판정 생략")

    # 층 변경 (같은 건물)
    if ga.floor != gb.floor:
        if dt_min < MIN_FLOOR_CHANGE_MIN:
            return TransitionEval(prev_locus, curr_locus, dt_min,
                                  FEAS_IMPOSSIBLE, None, None,
                                  f"층 변경({ga.floor}→{gb.floor})이 {dt_min:.0f}분 내 불가 — 최소 {MIN_FLOOR_CHANGE_MIN}분 필요")
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_CROSS_FLOOR, None, None,
                              f"층 변경 {ga.floor}→{gb.floor} ({dt_min:.0f}분)")

    # 동일 건물+층 → 수평 거리
    if not ga.has_coords or not gb.has_coords:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, None, None, "좌표 결측 — 판정 생략")

    dist = float(np.hypot(ga.x - gb.x, ga.y - gb.y))
    speed = dist / max(dt_min, 1e-6)

    if speed <= WALK_SPEED_NORMAL:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, dist, speed,
                              f"정상 이동 {dist:.0f}m / {dt_min:.0f}분 = {speed:.0f}m/분")
    if speed <= WALK_SPEED_FAST:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_OK, dist, speed,
                              f"빠른 걸음 {dist:.0f}m / {dt_min:.0f}분 = {speed:.0f}m/분")
    if speed <= WALK_SPEED_MAX:
        return TransitionEval(prev_locus, curr_locus, dt_min,
                              FEAS_WARN_FAST, dist, speed,
                              f"달리기 수준 {dist:.0f}m / {dt_min:.0f}분 = {speed:.0f}m/분 — 주의")
    return TransitionEval(prev_locus, curr_locus, dt_min,
                          FEAS_IMPOSSIBLE, dist, speed,
                          f"물리적 불가 {dist:.0f}m / {dt_min:.0f}분 = {speed:.0f}m/분 (>{WALK_SPEED_MAX:.0f})")


def annotate_journey(jdf: pd.DataFrame, geo: GatewayGeo) -> pd.DataFrame:
    """
    작업자 단일 journey(1분 단위, 시간순)에 feasibility 컬럼을 부착.

    반환 컬럼 추가:
      - phys_feasibility : FEAS_* 카테고리
      - phys_distance_m  : 동일 층 내 거리
      - phys_speed       : m/분
      - phys_reason      : 사유 문자열
    """
    if jdf.empty or "locus_id" not in jdf.columns:
        return jdf.assign(
            phys_feasibility=pd.Series(dtype="object"),
            phys_distance_m=pd.Series(dtype="float"),
            phys_speed=pd.Series(dtype="float"),
            phys_reason=pd.Series(dtype="object"),
        )

    df = jdf.sort_values("timestamp").copy() if "timestamp" in jdf.columns else jdf.copy()

    prev_locus = df["locus_id"].shift(1)
    if "timestamp" in df.columns:
        dt = df["timestamp"].diff().dt.total_seconds().div(60).fillna(1.0)
    else:
        dt = pd.Series([1.0] * len(df), index=df.index)

    cats:    list[str] = []
    dists:   list[float | None] = []
    speeds:  list[float | None] = []
    reasons: list[str] = []

    for (p, c, t) in zip(prev_locus, df["locus_id"], dt):
        if pd.isna(p):
            cats.append(FEAS_OK); dists.append(None); speeds.append(None)
            reasons.append("시작점")
            continue
        ev = evaluate_transition(str(p), str(c), float(t), geo)
        cats.append(ev.category)
        dists.append(ev.distance_m)
        speeds.append(ev.speed_m_per_min)
        reasons.append(ev.reason)

    df["phys_feasibility"] = cats
    df["phys_distance_m"]  = dists
    df["phys_speed"]       = speeds
    df["phys_reason"]      = reasons
    return df


def summarize_feasibility(df_annotated: pd.DataFrame) -> dict:
    """annotate_journey 결과에서 요약 통계 반환."""
    if df_annotated.empty or "phys_feasibility" not in df_annotated.columns:
        return {"total": 0}
    vc = df_annotated["phys_feasibility"].value_counts()
    total = int(vc.sum())
    def _pct(k): return float(vc.get(k, 0)) / total * 100 if total else 0.0
    return {
        "total":          total,
        "ok":             int(vc.get(FEAS_OK, 0)),
        "warn_fast":      int(vc.get(FEAS_WARN_FAST, 0)),
        "cross_floor":    int(vc.get(FEAS_CROSS_FLOOR, 0)),
        "cross_bldg":     int(vc.get(FEAS_CROSS_BLDG, 0)),
        "impossible":     int(vc.get(FEAS_IMPOSSIBLE, 0)),
        "impossible_pct": _pct(FEAS_IMPOSSIBLE),
        "warn_pct":       _pct(FEAS_WARN_FAST),
    }


# ─── 모듈 수준 캐시 헬퍼 ──────────────────────────────────────────

_GEO_CACHE: dict[str, GatewayGeo] = {}


def get_gateway_geo(sector_id: str, locus_csv_path: Path | str) -> GatewayGeo:
    """sector별 GatewayGeo 인스턴스 캐시 조회."""
    key = f"{sector_id}:{locus_csv_path}"
    if key not in _GEO_CACHE:
        _GEO_CACHE[key] = GatewayGeo.from_csv(locus_csv_path)
    return _GEO_CACHE[key]
