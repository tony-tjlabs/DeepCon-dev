"""
integrity/helpers.py — 정합성 탭 공통 상수 · 순수 헬퍼
=======================================================
컬러 상수, 공간/locus 타입 매핑, 임계값, 순수 계산 유틸을
집중 관리한다. Streamlit·I/O 의존 없는 순수 함수만 포함.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.styles import COLORS, CHART_COLORS, PLOTLY_DARK


# ─── 색상 코드 (styles.CHART_COLORS 단일 소스 참조) ───────────────────
CAT_COLORS = {
    "raw_signal":    CHART_COLORS["raw"],          # 원본 BLE 신호
    "gap_high":      CHART_COLORS["gap_high"],     # 고신뢰 gap-fill
    "gap_medium":    CHART_COLORS["gap_medium"],   # 중신뢰 gap-fill
    "gap_low":       CHART_COLORS["gap_low"],      # 저신뢰 gap-fill
    "invalid":       CHART_COLORS["critical"],     # 비유효 전이
    "deep_inactive": CHART_COLORS["deep_inact"],   # 정지
}

ACTIVITY_COLORS = {
    "HIGH_ACTIVE":   CHART_COLORS["high_active"],
    "ACTIVE":        CHART_COLORS["work_zone"],
    "INACTIVE":      CHART_COLORS["medium"],
    "DEEP_INACTIVE": CHART_COLORS["deep_inact"],
    "ESTIMATED":     CHART_COLORS["sii"],
}

# ─── 활성도 티어 / 헬멧 방치 판정 임계값 ────────────────────────────
# src/pipeline/metrics.py 와 동일한 값을 유지 (단일 소스).
# HIGH: 집중 작업 / LOW: 정지·대기 / MID: 그 사이 / GAP: 음영(gap-fill) 보정
ACTIVE_HIGH_THRESHOLD  = 0.90
ACTIVE_LOW_THRESHOLD   = 0.40
# 헬멧 방치 의심: WORK_AREA + LOW active가 연속 N분 이상 지속
HELMET_SUSPECT_MIN_RUN = 30

# 활성도 티어 색상 (chart_colors 참조 — 하드코딩 금지)
_ACT_TIER_COLORS: dict[str, str] = {
    "HIGH":  CHART_COLORS["high_active"],   # #FF4C4C (집중 작업)
    "MID":   CHART_COLORS["standby"],       # #9AB5D4 (중간)
    "LOW":   CHART_COLORS["low_active"],    # #0078AA (저활성)
    "GAP":   CHART_COLORS["gap"],           # #4A5A6A (음영/보정)
}
_ACT_TIER_KO: dict[str, str] = {
    "HIGH": "고활성", "MID": "중간", "LOW": "저활성", "GAP": "음영(gap-fill)",
}

GAP_COLORS = {
    "none":   COLORS["card_bg"],
    "high":   CHART_COLORS["gap_high"],
    "medium": CHART_COLORS["gap_medium"],
    "low":    CHART_COLORS["gap_low"],
}

# ─── 공간 유형 공통 상수 (단일 팔레트 적용) ───────────────────────────
# ★ 전사 통일: styles.CHART_COLORS 참조. 작업공간=accent, 휴게공간=success.
_SPACE_COLORS: dict[str, str] = {
    "작업공간 (WORK_AREA)": CHART_COLORS["work_area"],   # #00AEEF
    "이동공간 (TRANSIT)":   CHART_COLORS["transit"],     # #9AB5D4
    "휴게공간 (REST_AREA)": CHART_COLORS["rest_area"],   # #00C897
    "출입공간 (GATE)":      CHART_COLORS["gate"],        # #A78BFA
    "미분류":               CHART_COLORS["other"],       # #6A7A95
    "음영 (0-signal)":      CHART_COLORS["gap"],         # #4A5A6A
    "음영 (gap-fill)":      CHART_COLORS["gap"],         # #4A5A6A
    "음영 (미수신)":        COLORS["border"],            # #2A3A4A (더 진한 톤)
}
_LTYPE_TO_SPACE: dict[str, str] = {
    "WORK_AREA": "작업공간 (WORK_AREA)",
    "TRANSIT":   "이동공간 (TRANSIT)",
    "REST_AREA": "휴게공간 (REST_AREA)",
    "GATE":      "출입공간 (GATE)",
}
_SPACE_DISPLAY_ORDER: list[str] = [
    "작업공간 (WORK_AREA)", "이동공간 (TRANSIT)",
    "휴게공간 (REST_AREA)", "출입공간 (GATE)",
    "미분류", "음영 (0-signal)", "음영 (gap-fill)", "음영 (미수신)",
]


def _make_donut(labels, values, colors, total_min: int):
    """공간 유형별 체류 비율 도넛 차트 생성 (모듈 레벨 공용)."""
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.54,
        marker=dict(colors=colors, line=dict(color="#0D1B2A", width=2)),
        textinfo="percent",
        textfont=dict(size=11, color="#D5E5FF"),
        hovertemplate="<b>%{label}</b><br>%{value}분 (%{percent})<extra></extra>",
        sort=False,
    ))
    fig.update_layout(**{
        **PLOTLY_DARK,
        "height": 300,
        "showlegend": True,
        "margin": dict(l=10, r=10, t=10, b=10),
        "legend": dict(font=dict(size=10, color="#9AB5D4"),
                       bgcolor="rgba(0,0,0,0)",
                       orientation="v", x=1.02, y=0.5),
        "annotations": [dict(text=f"<b>{total_min}분</b>",
                              x=0.5, y=0.5,
                              font=dict(size=14, color="#D5E5FF"),
                              showarrow=False)],
    })
    return fig

def _locus_display_name(locus_id: str, locus_meta_name: str,
                         building: str, floor: str) -> str:
    """gap-fill 레코드용 사람이 읽을 수 있는 place 이름 도출.

    우선순위:
      1) locus_meta_name에서 좌표(X##Y##) 제거 후 정제
      2) building + floor → 예: 'FAB 2F'
      3) locus_id 그대로
    """
    import re
    name = str(locus_meta_name or "").strip()
    if name:
        # 좌표 패턴 제거: _X##Y## 또는 _X###Y###
        name = re.sub(r"_X\d+Y\d+", "", name)
        # 앞부분에서 prefix 추출
        # fab → FAB, cub → CUB, bl → BL, wwt → WWT, 본진 등
        prefix_map = {
            "fab_": "FAB", "cub_": "CUB", "bl_": "BL", "wwt_": "WWT",
            "upw_": "UPW", "154kv_": "154kV",
        }
        for pat, rep in prefix_map.items():
            if name.lower().startswith(pat):
                rest = name[len(pat):]
                # 층 번호 추출: 1F, 2F, B1F 등
                floor_match = re.match(r"([Bb]?\d*F)", rest, re.IGNORECASE)
                if floor_match:
                    return f"{rep} {floor_match.group(1).upper()}"
                # 특수 장소명: 호이스트, 휴게실 등
                rest_clean = rest.replace("_", " ").strip()
                return f"{rep} {rest_clean[:20]}"
        # 한글 이름: 언더스코어를 공백으로 치환
        return name.replace("_", " ")
    if building and floor:
        bldg_clean = str(building).upper()
        return f"{bldg_clean} {floor}"
    return str(locus_id or "unknown")

# 공간 유형별 place 정렬 순서 — building 접두사 기반
_PLACE_SORT_PREFIX_ORDER = [
    "FAB", "CUB", "WWT", "BL", "UPW", "지원", "본진", "공사", "154kV", "전진", "포터"
]


def _sort_place_names(places: list[str]) -> list[str]:
    """place 이름을 building 그룹 → 층 번호 순으로 정렬.

    FAB 1F, FAB 2F, ..., CUB 1F, CUB B1F, ... 순서로 그룹핑.
    """
    import re

    def _sort_key(p: str):
        p_upper = str(p).upper()
        # prefix 그룹 인덱스
        grp = len(_PLACE_SORT_PREFIX_ORDER)  # unknown → 마지막
        for i, prefix in enumerate(_PLACE_SORT_PREFIX_ORDER):
            if p_upper.startswith(prefix):
                grp = i
                break
        # 층 번호: B층 → 음수, 1F→1, 2F→2 ...
        floor_match = re.search(r"([Bb]?)(\d+)F", p_upper)
        if floor_match:
            sign = -1 if floor_match.group(1).upper() == "B" else 1
            level = sign * int(floor_match.group(2))
        else:
            level = 999
        return (grp, level, p_upper)

    return sorted(places, key=_sort_key)

# locus_type → 색상·한국어 (styles.CHART_COLORS 단일 소스 참조)
# ★ _SPACE_COLORS와 동일한 매핑 유지 (전사 통일)
LOCUS_TYPE_COLOR: dict[str, str] = {
    "WORK_AREA":  CHART_COLORS["work_area"],   # #00AEEF
    "TRANSIT":    CHART_COLORS["transit"],     # #9AB5D4
    "GATE":       CHART_COLORS["gate"],        # #A78BFA
    "REST_AREA":  CHART_COLORS["rest_area"],   # #00C897
}
LOCUS_TYPE_KO: dict[str, str] = {
    "WORK_AREA": "작업구역",
    "TRANSIT":   "이동",
    "GATE":      "게이트",
    "REST_AREA": "휴게",
}

def _enrich_journey_with_locus(
    user_jdf: pd.DataFrame, locus_meta: pd.DataFrame
) -> pd.DataFrame:
    """journey DataFrame에 locus_v2 메타 컬럼을 left-join해서 반환."""
    if locus_meta.empty or user_jdf.empty:
        return user_jdf
    # 이미 보유한 컬럼 제외
    skip = set(user_jdf.columns) - {"locus_id"}
    add_cols = [c for c in locus_meta.columns if c not in skip and c != "locus_id"]
    if not add_cols:
        return user_jdf
    return user_jdf.merge(
        locus_meta[["locus_id"] + add_cols],
        on="locus_id", how="left",
    )

def _compute_act_tier(df: pd.DataFrame) -> pd.Series:
    """active_ratio + is_gap_filled → 4단계 활성도 티어.

    GAP은 Gap-fill 보정 구간 (activity_level == ESTIMATED 포함).
    """
    ar = df["active_ratio"].fillna(0.0)
    is_gap = df.get("is_gap_filled", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    # activity_level=ESTIMATED는 gap과 동의어로 취급
    if "activity_level" in df.columns:
        is_est = df["activity_level"].fillna("").eq("ESTIMATED")
        is_gap = is_gap | is_est

    tier = pd.Series("MID", index=df.index)
    tier[ar >= ACTIVE_HIGH_THRESHOLD] = "HIGH"
    tier[ar <= ACTIVE_LOW_THRESHOLD]  = "LOW"
    tier[is_gap] = "GAP"
    return tier



def _find_raw_file(raw_dir: Path, prefix: str, date_str: str) -> "Path | None":
    """날짜에 해당하는 raw CSV 파일 경로를 반환 (단일일/범위 파일 모두 지원)."""
    exact = raw_dir / f"{prefix}_{date_str}.csv"
    if exact.exists():
        return exact
    target = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    for f in sorted(raw_dir.glob(f"{prefix}_*.csv")):
        stem = f.stem.replace(f"{prefix}_", "")
        if "~" in stem:
            parts = [p.strip() for p in stem.split("~")]
            if len(parts) == 2:
                try:
                    d0 = parts[0].replace(" ", "")[:8]
                    d1 = parts[1].replace(" ", "")[:8]
                    s2 = pd.Timestamp(f"{d0[:4]}-{d0[4:6]}-{d0[6:8]}")
                    e2 = pd.Timestamp(f"{d1[:4]}-{d1[4:6]}-{d1[6:8]}")
                    if s2 <= target <= e2:
                        return f
                except Exception:
                    pass
    return None
