"""
Deep Space 현장 시뮬레이션 탭
=============================
Entity(작업자)와 Locus(공간)를 통합하여 현장 전체 상태 시각화.
"""
from __future__ import annotations

import logging
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# === 시뮬레이션 상수 ===
_SIM_BATCH_SIZE = 256        # 배치 추론 크기
_SIM_MAX_SEQ_LEN = 50        # 최대 시퀀스 길이 (모델과 일치)
_BUBBLE_MIN_SIZE = 8.0       # 버블 최소 크기
_BUBBLE_MAX_SIZE = 50.0      # 버블 최대 크기
_ROW_LIMIT = 12              # 한 행에 최대 locus 수

from config import PROCESSED_DIR
from core.cache.policy import MULTI_DAY_AGG
from src.dashboard.styles import COLORS, section_header, metric_card, metric_card_sm
from src.dashboard.llm_deepcon import is_llm_available, render_data_comment
from src.dashboard.deep_space._ai_adapters import cached_spatial_insight
from src.dashboard.deep_space.model_loader import load_journey_data, load_locus_info, detect_time_column, DEFAULT_CAPACITY_BY_TYPE
from src.utils.weather import date_label
from src.dashboard.deep_space.helpers import extract_worker_sequences, predict_next, predict_next_batch
from src.dashboard.deep_space.locus_prediction import predict_locus_states, build_simulation_locus_context

logger = logging.getLogger(__name__)

_DEFAULT_CAPACITY_BY_TYPE = DEFAULT_CAPACITY_BY_TYPE


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def compute_simulation_state(
    _sector_id_date: str,
    hour: int,
    sector_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    특정 시간까지의 현장 상태 + 예측 계산.

    Args:
        _sector_id_date: 캐시 키 (sector_id + date_str)
        hour: 분석 대상 시간 (0~23)
        sector_id: Sector ID

    Returns:
        (entity_state_df, locus_state_df, summary_dict)
    """
    date_str = _sector_id_date.split("|")[-1]
    sim_cols = [
        "user_no", "timestamp", "data_date", "minute",
        "locus_id", "locus", "corrected_locus_id",
        "is_work_hour",
    ]
    path = PROCESSED_DIR / sector_id / date_str / "journey.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}

    all_cols = pd.read_parquet(path, columns=[]).columns.tolist()
    use_cols = [c for c in sim_cols if c in all_cols]
    journey_df = pd.read_parquet(path, columns=use_cols) if use_cols else pd.read_parquet(path)

    if "is_work_hour" in journey_df.columns:
        journey_df = journey_df[journey_df["is_work_hour"]]

    locus_info = load_locus_info(sector_id)
    if locus_info is None:
        locus_info = pd.DataFrame()

    if journey_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # 시간 필터링
    from src.dashboard.deep_space.model_loader import detect_time_column
    time_col = detect_time_column(journey_df)

    if time_col:
        journey_df[time_col] = pd.to_datetime(journey_df[time_col], errors="coerce")
        journey_df = journey_df[journey_df[time_col].dt.hour <= hour]

    if journey_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Locus 메타 빌드 (vectorized)
    locus_meta = {}
    for r in locus_info.to_dict("records"):
        lid = str(r.get("locus_id", ""))
        if not lid:
            continue
        locus_type = str(r.get("locus_type", "WORK")).upper()
        capacity = 0
        if pd.notna(r.get("capacity")):
            capacity = int(r.get("capacity"))
        elif pd.notna(r.get("max_concurrent_occupancy")):
            capacity = int(float(r.get("max_concurrent_occupancy")) * 1.5)
        if capacity <= 0:
            capacity = _DEFAULT_CAPACITY_BY_TYPE.get(locus_type, 50)

        locus_meta[lid] = {
            "name": r.get("locus_name", lid),
            "type": locus_type,
            "hazard_level": r.get("hazard_level", "low"),
            "hazard_grade": float(r.get("hazard_grade", 2.0)) if pd.notna(r.get("hazard_grade")) else 2.0,
            "capacity": capacity,
        }

    user_col = "user_no" if "user_no" in journey_df.columns else None
    locus_col = None
    for lc in ["locus_id", "locus", "corrected_locus_id"]:
        if lc in journey_df.columns:
            locus_col = lc
            break

    if user_col is None or locus_col is None:
        return pd.DataFrame(), pd.DataFrame(), {}

    # 각 작업자의 마지막 위치
    last_positions = (
        journey_df.sort_values(time_col if time_col else user_col)
        .groupby(user_col)
        .tail(1)[[user_col, locus_col]]
        .copy()
    )
    last_positions.columns = ["user_no", "current_locus"]

    def classify_entity_state(locus_id):
        meta = locus_meta.get(str(locus_id), {})
        ltype = meta.get("type", "WORK")
        if ltype in ("GATE", "TRANSPORT"):
            return "이동 중"
        elif ltype in ("REST", "ADMIN"):
            return "휴식 중"
        else:
            return "작업 중"

    last_positions["status"] = last_positions["current_locus"].apply(classify_entity_state)
    last_positions["locus_name"] = last_positions["current_locus"].map(
        lambda x: locus_meta.get(str(x), {}).get("name", x)
    )

    # Locus별 인원 집계
    locus_counts = last_positions.groupby("current_locus").size().reset_index(name="current_count")

    # Locus 상태 DataFrame
    all_loci = set(locus_counts["current_locus"].tolist()) | set(locus_meta.keys())
    locus_states = []
    for lid in all_loci:
        meta = locus_meta.get(str(lid), {})
        count_row = locus_counts[locus_counts["current_locus"] == lid]
        count = int(count_row["current_count"].iloc[0]) if not count_row.empty else 0
        capacity = meta.get("capacity", 50)
        congestion = min(count / max(capacity, 1), 2.0)

        if congestion >= 0.9:
            state = "danger"
        elif congestion >= 0.7:
            state = "warning"
        elif congestion >= 0.5:
            state = "caution"
        else:
            state = "normal"

        locus_states.append({
            "locus_id": lid,
            "locus_name": meta.get("name", lid),
            "locus_type": meta.get("type", "WORK"),
            "hazard_level": meta.get("hazard_level", "low"),
            "hazard_grade": meta.get("hazard_grade", 2.0),
            "capacity": capacity,
            "current_count": count,
            "congestion": congestion,
            "state": state,
        })

    locus_state_df = pd.DataFrame(locus_states)
    if not locus_state_df.empty:
        locus_state_df = locus_state_df.sort_values("current_count", ascending=False)

    # 요약 통계
    total_workers = len(last_positions)
    status_counts = last_positions["status"].value_counts().to_dict()
    n_working = status_counts.get("작업 중", 0)
    n_moving = status_counts.get("이동 중", 0)
    n_resting = status_counts.get("휴식 중", 0)

    n_congested = len(locus_state_df[locus_state_df["congestion"] >= 0.7]) if not locus_state_df.empty else 0
    n_bottleneck = len(locus_state_df[locus_state_df["congestion"] >= 0.9]) if not locus_state_df.empty else 0
    hazard_types = ("HAZARD", "TRANSPORT")
    n_safety_warning = len(
        locus_state_df[
            (locus_state_df["locus_type"].isin(hazard_types)) &
            (locus_state_df["current_count"] > 0)
        ]
    ) if not locus_state_df.empty else 0

    summary = {
        "total_workers": total_workers,
        "n_working": n_working,
        "n_moving": n_moving,
        "n_resting": n_resting,
        "n_congested": n_congested,
        "n_bottleneck": n_bottleneck,
        "n_safety_warning": n_safety_warning,
    }

    return last_positions, locus_state_df, summary


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def _cached_extract_sequences(sector_id: str, date_str: str) -> dict[str, list[str]]:
    """journey 데이터에서 시퀀스 추출 캐시 래퍼."""
    journey_df = load_journey_data(sector_id, date_str)
    if journey_df is None or journey_df.empty:
        return {}
    if "is_work_hour" in journey_df.columns:
        journey_df = journey_df[journey_df["is_work_hour"]]
    return extract_worker_sequences(journey_df)


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def _build_transition_matrix(
    _model_hash: str,
    _tokenizer_hash: str,
    _seq_key: str,
    sequences: dict[str, list[str]],
    sector_id: str,
    sample_size: int = 500,
) -> dict[str, dict[str, float]]:
    """
    샘플 시퀀스로 locus->locus 전이 확률 행렬 생성 (배치 추론 사용).

    Args:
        _model_hash: 모델 캐시 키
        _tokenizer_hash: 토크나이저 캐시 키
        _seq_key: 시퀀스 캐시 키
        sequences: {worker_id: [locus_id, ...]}
        sector_id: Sector ID
        sample_size: 행렬 추정용 샘플 크기

    Returns:
        {from_locus: {to_locus: probability}}
    """
    from collections import Counter, defaultdict
    from src.dashboard.deep_space.model_loader import load_model

    model, tokenizer = load_model(sector_id)
    if model is None or tokenizer is None:
        return {}

    # 샘플링 (행렬 추정용)
    items = list(sequences.items())
    if len(items) > sample_size:
        items = random.sample(items, sample_size)

    # 유효한 시퀀스만 필터링
    valid_items = [(wid, seq) for wid, seq in items if seq and len(seq) >= 2]
    if not valid_items:
        return {}

    seqs = [seq for _, seq in valid_items]
    current_loci = [seq[-1] for seq in seqs]

    # 배치 추론 (top_k=5)
    batch_results = predict_next_batch(model, tokenizer, seqs, batch_size=256, top_k=5)

    # 각 (현재 locus, 시퀀스) -> 다음 locus 확률 누적
    transition_counts: dict[str, Counter] = defaultdict(Counter)

    for i, preds in enumerate(batch_results):
        current_locus = current_loci[i]
        for next_locus, prob in preds:
            transition_counts[current_locus][next_locus] += prob

    # 확률로 정규화
    transition_matrix: dict[str, dict[str, float]] = {}
    for from_locus, to_counts in transition_counts.items():
        total = sum(to_counts.values())
        if total > 0:
            transition_matrix[from_locus] = {k: v / total for k, v in to_counts.items()}

    return transition_matrix


def _apply_transition_matrix(
    current_locus_counts: dict[str, float],
    transition_matrix: dict[str, dict[str, float]],
    n_steps: int,
) -> dict[str, float]:
    """
    전이 행렬을 현재 인원 분포에 n_steps 적용.

    Args:
        current_locus_counts: {locus_id: count}
        transition_matrix: {from_locus: {to_locus: probability}}
        n_steps: 적용 횟수 (1=30분 후, 2=1시간 후)

    Returns:
        {locus_id: predicted_count}
    """
    from collections import defaultdict

    counts: dict[str, float] = dict(current_locus_counts)

    for _ in range(n_steps):
        new_counts: dict[str, float] = defaultdict(float)
        for locus, count in counts.items():
            if count <= 0:
                continue
            if locus in transition_matrix:
                for next_locus, prob in transition_matrix[locus].items():
                    new_counts[next_locus] += count * prob
            else:
                # 전이 정보 없으면 현 위치 유지
                new_counts[locus] += count
        counts = dict(new_counts)

    return counts


def _build_predicted_locus_df(
    predicted_counts: dict[str, float],
    locus_info: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    예측 카운트로 locus_df 생성 (compute_simulation_state와 동일 컬럼 구조).

    Args:
        predicted_counts: {locus_id: count} (전이 행렬 적용 결과)
        locus_info: Locus 정보 DataFrame

    Returns:
        locus_df (locus_id, locus_name, locus_type, current_count, congestion, hazard_level, capacity)
    """
    if locus_info is None or locus_info.empty:
        return pd.DataFrame()

    rows = []
    for r in locus_info.to_dict("records"):
        locus_id = str(r.get("locus_id", ""))
        count = float(predicted_counts.get(locus_id, 0.0))

        # capacity 계산
        capacity = 0
        if pd.notna(r.get("capacity")):
            capacity = int(r.get("capacity"))
        elif pd.notna(r.get("max_concurrent_occupancy")):
            capacity = int(float(r.get("max_concurrent_occupancy")) * 1.5)
        if capacity <= 0:
            locus_type = str(r.get("locus_type", "WORK")).upper()
            capacity = _DEFAULT_CAPACITY_BY_TYPE.get(locus_type, 50)

        congestion = min(count / max(capacity, 1), 1.0)

        rows.append({
            "locus_id": locus_id,
            "locus_name": r.get("locus_name", locus_id),
            "locus_type": str(r.get("locus_type", "WORK")).upper(),
            "current_count": round(count),
            "congestion": congestion,
            "hazard_level": r.get("hazard_level", "low"),
            "capacity": capacity,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("current_count", ascending=False)
    return df


def _render_predicted_state(
    model,
    tokenizer,
    sequences: dict[str, list[str]],
    locus_info: pd.DataFrame | None,
    sector_id: str,
    current_locus_df: pd.DataFrame | None,
    n_steps: int,
    label: str,
    chart_key: str = "bubble_predicted",
):
    """예측 상태 렌더링 (배치 추론 방식 + 버블맵 + KPI).

    Args:
        model: Deep Space 모델
        tokenizer: 토크나이저
        sequences: {worker_id: [locus_id, ...]}
        locus_info: Locus 정보 DataFrame
        sector_id: Sector ID
        current_locus_df: 현재 상태 locus_df (인원 분포 소스)
        n_steps: 예측 단계 수 (1=30분 후, 2=1시간 후)
        label: 표시 라벨
    """
    from collections import Counter

    st.markdown(f"#### {label}")

    if model is None or not sequences:
        st.info("예측을 위한 모델 또는 데이터가 없습니다.")
        return

    # 전체 작업자 시퀀스 준비
    all_workers = list(sequences.keys())
    all_seqs = [sequences[w] for w in all_workers]

    if not all_seqs:
        st.caption("시퀀스 데이터 없음")
        return

    with st.spinner(f"{label} 배치 추론 중..."):
        # n_steps 처리: 다단계 예측
        current_seqs = [list(seq) for seq in all_seqs]  # 복사

        for step in range(n_steps):
            # 배치 추론
            step_results = predict_next_batch(model, tokenizer, current_seqs, batch_size=256, top_k=1)

            # 예측 결과를 시퀀스에 추가 (다음 step용)
            for i, preds in enumerate(step_results):
                if preds and current_seqs[i]:
                    next_locus = preds[0][0]
                    current_seqs[i].append(next_locus)

        # 최종 예측 결과에서 마지막 위치 집계
        predicted_counts: Counter = Counter()
        for seq in current_seqs:
            if seq:
                predicted_counts[seq[-1]] += 1

    if not predicted_counts:
        st.info("모델 예측을 사용할 수 없습니다.")
        return

    # 예측 locus_df 생성
    predicted_locus_df = _build_predicted_locus_df(dict(predicted_counts), locus_info)

    if predicted_locus_df.empty:
        st.caption("예측 결과 없음")
        return

    # 5. KPI 표시
    total_predicted = int(predicted_locus_df["current_count"].sum())
    n_congested = int((predicted_locus_df["congestion"] >= 0.7).sum())
    hazard_types = ("HAZARD", "TRANSPORT")
    n_safety = int((
        (predicted_locus_df["locus_type"].isin(hazard_types)) &
        (predicted_locus_df["current_count"] > 0)
    ).sum())

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("예측 인원", f"{total_predicted:,}명")
    with kpi2:
        st.metric("혼잡 공간", f"{n_congested}개", delta=None)
    with kpi3:
        st.metric("안전 경고", f"{n_safety}개", delta=None)

    # 6. 버블맵
    _render_locus_bubble_map(predicted_locus_df, locus_info, chart_key=chart_key)

    st.caption(f"Deep Space 모델 예측 | {n_steps}회 이동 기준 | 500명 샘플 기반 전이행렬")


def render_simulation(model, tokenizer, sector_id: str, dates: list[str]):
    """현장 시뮬레이션 뷰."""
    if not dates:
        st.warning("데이터가 없습니다.")
        return

    col_date, col_hour = st.columns([1, 2])
    with col_date:
        selected_date = st.selectbox("날짜", dates, index=len(dates) - 1, key="ds_sim_date", format_func=date_label)
    with col_hour:
        selected_hour = st.slider(
            "시간대 (해당 시간까지의 상태)",
            min_value=0, max_value=23, value=12, step=1,
            format="%d:00", key="ds_sim_hour",
        )

    journey_df = load_journey_data(sector_id, selected_date)
    if journey_df is None or journey_df.empty:
        st.warning("Journey 데이터를 로드할 수 없습니다.")
        return

    if "is_work_hour" in journey_df.columns:
        journey_df = journey_df[journey_df["is_work_hour"]]

    locus_info = load_locus_info(sector_id)
    locus_info_json = locus_info.to_json() if locus_info is not None else ""

    cache_key = f"{sector_id}|{selected_date}"
    with st.spinner("현장 상태 분석 중..."):
        entity_df, locus_df, summary = compute_simulation_state(cache_key, selected_hour, sector_id)

    if entity_df.empty:
        st.warning("해당 시간대의 데이터가 없습니다.")
        return

    # KPI 카드
    st.markdown(section_header("현장 전체 현황"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("총 현장 인원", f"{summary['total_workers']:,}명"), unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='font-size:1.4rem;'>"
            f"<span style='color:{COLORS['success']}'>{summary['n_working']:,}</span> / "
            f"<span style='color:{COLORS['accent']}'>{summary['n_moving']:,}</span> / "
            f"<span style='color:{COLORS['warning']}'>{summary['n_resting']:,}</span>"
            f"</div><div class='metric-label'>작업 / 이동 / 휴식</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        color = COLORS["danger"] if summary["n_congested"] >= 3 else COLORS["warning"] if summary["n_congested"] >= 1 else COLORS["success"]
        st.markdown(metric_card_sm("혼잡 공간", f"{summary['n_congested']}개", color=color), unsafe_allow_html=True)
    with c4:
        color = COLORS["danger"] if summary["n_safety_warning"] >= 2 else COLORS["warning"] if summary["n_safety_warning"] >= 1 else COLORS["success"]
        st.markdown(metric_card_sm("안전 경고", f"{summary['n_safety_warning']}개", color=color), unsafe_allow_html=True)

    # LLM 인사이트
    if is_llm_available() and not locus_df.empty:
        congested = locus_df[locus_df["congestion"] >= 0.7]["locus_name"].tolist() if "congestion" in locus_df.columns else []
        locus_context = build_simulation_locus_context(locus_df, sector_id)
        insight = cached_spatial_insight(
            summary=f"총 {summary['total_workers']}명, 작업 {summary['n_working']}명, 이동 {summary['n_moving']}명, 휴식 {summary['n_resting']}명",
            congested_spaces=", ".join(congested[:5]) if congested else "없음",
            locus_context=locus_context,
            sector_id=sector_id,
            tab="deep_space_simulation",
        )
        if insight:
            with st.expander("AI 현장 상태 해석", expanded=False):
                render_data_comment("현장 상태 해석", insight)

    # 고정 버블 맵 (전체 너비) + Sankey
    st.markdown(section_header("공간 상태 시각화"), unsafe_allow_html=True)
    _render_locus_bubble_map(locus_df, locus_info, chart_key="bubble_current")

    # 캐시된 시퀀스 추출
    sequences = _cached_extract_sequences(sector_id, selected_date)
    _render_sankey_optimized(model, tokenizer, sequences, locus_info, sector_id)

    # 예측 요약
    _render_prediction_summary(model, tokenizer, sector_id, sequences, locus_info_json)

    # 시간대별 추이
    _render_hourly_timeline(journey_df, locus_info, selected_hour)

    # 예측 시뮬레이션 섹션 (@st.fragment로 독립 실행)
    # 클로저로 locus_df 캡처
    @st.fragment
    def _prediction_section():
        st.markdown("---")
        st.markdown(section_header("예측 시뮬레이션"), unsafe_allow_html=True)
        st.caption("현재 상태를 기반으로 Deep Space 모델이 향후 이동을 예측합니다")

        col1, col2 = st.columns(2)
        with col1:
            _render_predicted_state(
                model, tokenizer, sequences, locus_info, sector_id,
                current_locus_df=locus_df,
                n_steps=1, label="30분 후 예측",
                chart_key="bubble_pred_30m",
            )
        with col2:
            _render_predicted_state(
                model, tokenizer, sequences, locus_info, sector_id,
                current_locus_df=locus_df,
                n_steps=2, label="1시간 후 예측",
                chart_key="bubble_pred_60m",
            )

    _prediction_section()


def _infer_zone(row) -> str:
    """locus_name + building 컬럼으로 구역 추론 (building 컬럼 누락 대응)."""
    b = str(row.get("building", "") or "")
    name = str(row.get("locus_name", "") or "")
    ltype = str(row.get("locus_type", "") or "").upper()

    # building 컬럼 우선
    for key in ["FAB", "CUB", "WWT", "154kV", "본진", "전진"]:
        if b == key:
            return key

    # locus_name에서 키워드 추출
    for key in ["FAB", "CUB", "WWT", "154kV"]:
        if key in name:
            return key
    if "공사현장" in name or "야외" in name:
        return "야외"
    if "본진" in name:
        return "본진"
    if "전진" in name:
        return "전진"
    if "지원" in name and ltype == "WORK":
        return "지원"
    if "BL" in name and ltype == "WORK":
        return "지원"
    if "저수조" in name or "비상발전" in name:
        return "설비"
    if ltype == "GATE":
        return "GATE"
    return "공용"


# 기능 그룹 정의: (행 Y, 레이블, 해당 조건 함수)
_FUNC_GROUPS: list[tuple[int, str, str]] = [
    (0, "GATE",        "GATE"),       # 출입구
    (1, "FAB 작업",    "FAB"),        # FAB 작업구역
    (2, "CUB 작업",    "CUB"),        # CUB 작업구역
    (3, "WWT 작업",    "WWT"),        # WWT 작업구역 (밀폐 포함)
    (4, "야외·지원",   "야외지원"),    # 야외/지원BL/설비
    (5, "본진·전진",   "본진전진"),    # 본진·전진 작업/행정
    (6, "154kV",       "154kV"),      # 154kV 설비
    (7, "휴게·편의",   "REST"),       # 휴게실·흡연장·식당
    (8, "행정·시설",   "ADMINFAC"),   # 행정·시설·주차장
]
_GROUP_Y: dict[str, int] = {g[2]: g[0] for g in _FUNC_GROUPS}
_GROUP_LABEL: dict[int, str] = {g[0]: g[1] for g in _FUNC_GROUPS}


def _assign_func_group(row) -> str:
    """각 locus를 기능 그룹 키로 분류."""
    zone = _infer_zone(row)
    ltype = str(row.get("locus_type", "") or "").upper()

    if ltype == "GATE":
        return "GATE"
    if zone == "FAB":
        return "FAB"
    if zone == "CUB":
        return "CUB"
    if zone == "WWT":
        return "WWT"
    if zone in ("야외", "지원", "설비"):
        return "야외지원"
    if zone in ("본진", "전진"):
        return "본진전진"
    if zone == "154kV":
        return "154kV"
    if ltype in ("REST",):
        return "REST"
    if ltype in ("ADMIN", "FACILITY"):
        return "ADMINFAC"
    return "REST"  # 분류 불명은 휴게/공용으로


@st.cache_data(show_spinner=False)
def _compute_locus_positions(locus_info_json: str) -> tuple[dict[str, tuple[float, float, float]], dict[str, float], int]:
    """Locus 고정 좌표 + 크기 계산 (기능 그룹별 다중 행 배치).

    Layout:
        - 기능 그룹별로 Y 행 할당
        - 그룹당 locus가 ROW_LIMIT 초과 시 다음 서브행(+0.85)으로 래핑
        - X축 = 서브행 내 순서 (1.2 단위)
        - 크기 = max_concurrent_occupancy 비례 (최소8 ~ 최대50)

    Returns:
        (positions, group_start_y, total_subrows)
        - positions: {locus_id: (x, y, size)}
        - group_start_y: {group_key: y_start}
        - total_subrows: 총 서브행 수 (차트 높이 계산용)
    """
    import math
    from io import StringIO
    from collections import defaultdict

    ROW_LIMIT = 12  # 한 행에 최대 locus 수 (초과 시 다음 서브행)

    locus_info = pd.read_json(StringIO(locus_info_json))

    # 그룹별 분류 (vectorized)
    group_loci: dict[str, list] = defaultdict(list)
    for r in locus_info.to_dict("records"):
        g = _assign_func_group(r)
        group_loci[g].append(r)

    # 전체 max_concurrent_occupancy -> 크기 정규화용
    all_mco = [float(r.get("max_concurrent_occupancy") or 0) for rows in group_loci.values() for r in rows]
    max_mco = max(all_mco) if all_mco else 1.0

    # 각 그룹에 누적 Y 행 할당 (서브행 포함)
    # _FUNC_GROUPS 순서로 처리, 각 그룹은 ceil(n_loci / ROW_LIMIT)개 서브행 점유
    group_start_y: dict[str, float] = {}
    current_y = 0.0
    total_subrows = 0

    for _, _, gkey in _FUNC_GROUPS:
        group_start_y[gkey] = current_y
        n_loci = len(group_loci.get(gkey, []))
        n_subrows = max(1, math.ceil(n_loci / ROW_LIMIT))
        total_subrows += n_subrows
        current_y += n_subrows * 0.85  # 서브행 간 간격 0.85

    positions: dict[str, tuple[float, float, float]] = {}

    for _, _, gkey in _FUNC_GROUPS:
        rows = group_loci.get(gkey, [])
        if not rows:
            continue

        base_y = group_start_y[gkey]

        # 그룹 내 층수 기준 오름차순 고정 정렬 (B1F→1F→2F→...)
        # 층 정보: floor 컬럼 > locus_name 파싱 > locus_id 숫자 순
        def _floor_sort_key(r) -> tuple:
            floor_val = str(r.get("floor") or "")
            name_val  = str(r.get("locus_name") or "")
            lid_val   = str(r.get("locus_id") or "")

            # floor 컬럼 또는 locus_name에서 층 번호 추출
            import re as _re
            raw = floor_val if floor_val and floor_val != "nan" else ""
            if not raw:
                # locus_name에서 "B1F / 1F / 2F ..." 패턴 추출
                m = _re.search(r"(B?\d+F)", name_val)
                raw = m.group(1) if m else ""

            # B1F → -1, 1F → 1, 2F → 2 ...
            if raw:
                bm = _re.match(r"B(\d+)F", raw)
                fm = _re.match(r"(\d+)F", raw)
                if bm:
                    floor_num = -int(bm.group(1))
                elif fm:
                    floor_num = int(fm.group(1))
                else:
                    floor_num = 99
            else:
                floor_num = 99  # 층 정보 없으면 뒤로

            # 2차 정렬: locus_id 숫자 (L-Y1-020 → 20, GW-351 → 351)
            id_num = int(_re.sub(r"\D", "", lid_val.split("-")[-1] or "0") or 0)
            return (floor_num, id_num)

        rows_sorted = sorted(rows, key=_floor_sort_key)

        for i, row in enumerate(rows_sorted):
            lid = str(row.get("locus_id", ""))
            sub_row = i // ROW_LIMIT      # 몇 번째 서브행
            col_in_row = i % ROW_LIMIT    # 서브행 내 순서
            x = float(col_in_row) * 1.2
            y = base_y + sub_row * 0.85

            mco = float(row.get("max_concurrent_occupancy") or 0)
            # 전체 최대값 대비 비율로 크기 정규화 (최소 8, 최대 50)
            if mco <= 0:
                size = 8.0
            else:
                size = 8.0 + (mco / max_mco) ** 0.6 * 42.0

            positions[lid] = (x, y, size)

    return positions, group_start_y, total_subrows


def _render_locus_bubble_map(locus_df: pd.DataFrame, locus_info: pd.DataFrame | None, chart_key: str = "bubble_map"):
    """고정 위치 버블 맵: 공간 위치/크기 고정, 색상=현재 혼잡도.

    - 모든 공간 항상 표시 (시간 변해도 위치/크기 불변)
    - 크기 = max_concurrent_occupancy (T-Ward 기반 최대 동시 인원)
    - 색상 = 현재 혼잡도 (0=초록, 1=빨강), 인원 없으면 회색
    - 그룹당 locus가 많으면 다중 행으로 래핑
    """
    if locus_df.empty or locus_info is None:
        st.info("Locus 데이터를 로드할 수 없습니다.")
        return

    locus_info_json = locus_info.to_json()
    positions, group_start_y, total_subrows = _compute_locus_positions(locus_info_json)

    # locus_df를 dict로 변환 (O(n) 조회 -> O(1) 조회)
    locus_state_map: dict[str, dict] = {}
    for ldr in locus_df.to_dict("records"):
        locus_state_map[str(ldr.get("locus_id", ""))] = ldr

    # locus_df와 위치 정보 결합 (vectorized)
    rows = []
    for r in locus_info.to_dict("records"):
        lid = str(r.get("locus_id", ""))
        if lid not in positions:
            continue
        x, y, base_size = positions[lid]

        # 현재 상태 (dict에서 O(1) 조회)
        cur_state = locus_state_map.get(lid)
        if cur_state is None:
            current_count = 0
            congestion = 0.0
            hazard_level = str(r.get("hazard_level", "low"))
        else:
            current_count = int(cur_state.get("current_count", 0))
            congestion = float(cur_state.get("congestion", 0.0))
            hazard_level = str(cur_state.get("hazard_level", "low"))

        mco = float(r.get("max_concurrent_occupancy") or 0)
        capacity = max(int(mco), 1)

        rows.append({
            "locus_id": lid,
            "locus_name": str(r.get("locus_name", lid)),
            "locus_type": str(r.get("locus_type", "WORK")),
            "building": str(r.get("building", "") or "기타"),
            "floor": str(r.get("floor", "") or "-"),
            "x": x,
            "y": y,
            "base_size": base_size,
            "current_count": current_count,
            "capacity": capacity,
            "congestion": congestion,
            "hazard_level": hazard_level,
        })

    if not rows:
        st.info("위치 정보가 없습니다.")
        return

    plot_df = pd.DataFrame(rows)

    # 색상: 인원 없으면 회색, 있으면 혼잡도에 따라 초록→빨강
    def _color(r):
        if r["current_count"] == 0:
            return "rgba(80,100,120,0.4)"
        c = r["congestion"]
        if c >= 0.8:
            return COLORS["danger"]
        if c >= 0.5:
            return COLORS["warning"]
        if c >= 0.2:
            return "#FFD700"
        return COLORS["success"]

    plot_df["color"] = plot_df.apply(_color, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["x"],
        y=plot_df["y"],
        mode="markers+text",
        marker=dict(
            size=plot_df["base_size"],
            color=plot_df["color"],
            line=dict(color="rgba(255,255,255,0.25)", width=1),
            sizemode="diameter",
        ),
        text=plot_df["locus_name"].apply(lambda n: n[:5] if len(n) > 5 else n),
        textposition="middle center",
        textfont=dict(size=7, color=COLORS["text"]),
        customdata=plot_df[["locus_name", "current_count", "capacity", "congestion", "hazard_level", "locus_type"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "현재 인원: %{customdata[1]}명 / 최대: %{customdata[2]}명<br>"
            "혼잡도: %{customdata[3]:.0%}  위험도: %{customdata[4]}<br>"
            "유형: %{customdata[5]}<extra></extra>"
        ),
        name="",
    ))

    # 행 구분선 (기능 그룹 사이) - 동적 Y 위치 기반
    for gkey, y_start in group_start_y.items():
        fig.add_hline(
            y=y_start - 0.3,
            line=dict(color="rgba(255,255,255,0.06)", width=1),
        )

    # Y축 눈금: 기능 그룹명 (동적 Y 위치)
    # _FUNC_GROUPS: (old_y_int, label, gkey)
    y_tickvals = [group_start_y.get(gkey, 0.0) for _, _, gkey in _FUNC_GROUPS]
    y_ticktext = [label for _, label, _ in _FUNC_GROUPS]

    # 동적 차트 높이 계산
    chart_height = max(480, total_subrows * 70)

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font_color=COLORS["text"],
        title="Locus 상태 맵 — 기능 그룹별 고정 배치 (크기=최대인원·색상=현재 혼잡도)",
        height=chart_height,
        margin=dict(l=90, r=20, t=50, b=60),
        xaxis=dict(
            visible=False,  # X축은 그룹 내 순서라 라벨 불필요
            gridcolor="rgba(255,255,255,0.03)",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickfont=dict(size=11),
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            autorange="reversed",  # GATE를 위에, 행정을 아래에
        ),
        showlegend=True,
    )

    # 범례 (혼잡도 색상)
    for label, color in [
        ("비어있음", "rgba(80,100,120,0.4)"),
        ("정상 (<20%)", COLORS["success"]),
        ("주의 (20~50%)", "#FFD700"),
        ("혼잡 (50~80%)", COLORS["warning"]),
        ("위험 (>80%)", COLORS["danger"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=label, showlegend=True,
        ))

    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=-0.18,
        xanchor="center", x=0.5,
        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
    ))

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key)


@st.cache_data(show_spinner=False)
def _compute_fixed_flow_nodes(locus_info_json: str, top_n: int = 20) -> list[str]:
    """고정 노드 목록: max_concurrent_occupancy 기준 상위 N개 locus_id (항상 동일).

    시간이 바뀌어도 노드 순서가 고정되어 Sankey 레이아웃이 유지됨.
    """
    from io import StringIO
    locus_info = pd.read_json(StringIO(locus_info_json))
    if locus_info.empty:
        return []
    col = "max_concurrent_occupancy"
    if col not in locus_info.columns:
        return list(locus_info["locus_id"].astype(str).head(top_n))
    top = locus_info.nlargest(top_n, col)
    return list(top["locus_id"].astype(str))


def _render_sankey(model, tokenizer, sequences: dict, locus_info: pd.DataFrame | None):
    """고정 노드 Sankey: 이동 흐름 (상위 공간 고정, 두께=예측 이동 수).

    - 노드: max_concurrent_occupancy 상위 N개 locus — 시간 불문 고정
    - 링크: 현재 시간대 sequences에서 예측된 흐름 (두께만 변경)
    """
    if not sequences or model is None:
        st.info("이동 시퀀스가 없거나 모델이 로드되지 않았습니다.")
        return

    # 고정 노드 목록
    locus_info_json = locus_info.to_json() if locus_info is not None else ""
    fixed_nodes = _compute_fixed_flow_nodes(locus_info_json, top_n=20)
    if not fixed_nodes:
        st.info("Locus 정보가 없습니다.")
        return

    # locus_id → 이름 매핑
    locus_meta: dict[str, str] = {}
    if locus_info is not None:
        for _, row in locus_info.iterrows():
            lid = str(row.get("locus_id", ""))
            locus_meta[lid] = str(row.get("locus_name", lid))

    fixed_node_set = set(fixed_nodes)

    # 이동 흐름 집계 (고정 노드 간만) — 배치 예측
    sampled = dict(random.sample(list(sequences.items()), min(500, len(sequences))))
    flow_data: dict[tuple[str, str], int] = {}

    # 유효 시퀀스만 필터링 후 배치 예측
    valid_items = [(wid, seq) for wid, seq in sampled.items()
                   if len(seq) >= 2 and seq[-1] in fixed_node_set]
    if valid_items:
        valid_wids, valid_seqs = zip(*valid_items)
        batch_preds = predict_next_batch(model, tokenizer, list(valid_seqs), top_k=3)
        for (wid, seq), preds in zip(valid_items, batch_preds):
            current = seq[-1]
            for next_loc, prob in preds:
                if next_loc not in fixed_node_set or next_loc == current:
                    continue
                key = (current, next_loc)
                flow_data[key] = flow_data.get(key, 0) + max(1, int(prob * 10))

    # 흐름 없으면 노드만 표시
    sorted_flows = sorted(flow_data.items(), key=lambda x: x[1], reverse=True)[:25]

    # 노드 인덱스 (고정 순서)
    node_idx = {lid: i for i, lid in enumerate(fixed_nodes)}
    node_labels = [locus_meta.get(lid, lid) for lid in fixed_nodes]

    # locus_type 기반 노드 색상
    type_colors = {
        "GATE": "#4A90D9",
        "WORK": "#00AEEF",
        "REST": "#00C897",
        "ADMIN": "#9B59B6",
        "FACILITY": "#E67E22",
        "TRANSPORT": "#F39C12",
        "HAZARD": "#E74C3C",
    }
    node_colors = []
    if locus_info is not None:
        type_map = {str(r["locus_id"]): str(r.get("locus_type", "WORK")) for _, r in locus_info.iterrows()}
        for lid in fixed_nodes:
            ltype = type_map.get(lid, "WORK")
            node_colors.append(type_colors.get(ltype, COLORS["accent"]))
    else:
        node_colors = [COLORS["accent"]] * len(fixed_nodes)

    if sorted_flows:
        source_indices = [node_idx[f[0][0]] for f in sorted_flows]
        target_indices = [node_idx[f[0][1]] for f in sorted_flows]
        values = [f[1] for f in sorted_flows]
    else:
        source_indices, target_indices, values = [], [], []

    fig_sankey = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=12,
            thickness=18,
            line=dict(color=COLORS["border"], width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values if values else [0],
            color="rgba(0, 174, 239, 0.3)",
        ),
    ))
    fig_sankey.update_layout(
        paper_bgcolor=COLORS["bg"],
        font_color=COLORS["text"],
        title="예측 이동 흐름 (노드 고정 · 두께=이동 예측량)",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_sankey, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(ttl=MULTI_DAY_AGG, show_spinner=False)
def _compute_sankey_flows(
    _model_hash: str,
    _tokenizer_hash: str,
    _seq_key: str,
    sequences: dict[str, list[str]],
    fixed_nodes: list[str],
    sector_id: str,
    max_workers: int = 200,
) -> dict[tuple[str, str], int]:
    """Sankey 흐름 데이터 계산 (캐시)."""
    from src.dashboard.deep_space.model_loader import load_model

    model, tokenizer = load_model(sector_id)
    if model is None or tokenizer is None:
        return {}

    fixed_node_set = set(fixed_nodes)

    # 샘플링
    if len(sequences) > max_workers:
        sampled = dict(random.sample(list(sequences.items()), max_workers))
    else:
        sampled = sequences

    flow_data: dict[tuple[str, str], int] = {}

    # 유효 시퀀스만 필터링 후 배치 예측
    valid_items = [(wid, seq) for wid, seq in sampled.items()
                   if len(seq) >= 2 and seq[-1] in fixed_node_set]
    if valid_items:
        valid_wids, valid_seqs = zip(*valid_items)
        batch_preds = predict_next_batch(model, tokenizer, list(valid_seqs), top_k=3)
        for (wid, seq), preds in zip(valid_items, batch_preds):
            current = seq[-1]
            for next_loc, prob in preds:
                if next_loc not in fixed_node_set or next_loc == current:
                    continue
                key = (current, next_loc)
                flow_data[key] = flow_data.get(key, 0) + max(1, int(prob * 10))

    return flow_data


def _render_sankey_optimized(
    model,
    tokenizer,
    sequences: dict[str, list[str]],
    locus_info: pd.DataFrame | None,
    sector_id: str,
):
    """최적화된 Sankey 렌더링 (캐시 + 샘플링 200)."""
    if not sequences or model is None:
        st.info("이동 시퀀스가 없거나 모델이 로드되지 않았습니다.")
        return

    locus_info_json = locus_info.to_json() if locus_info is not None else ""
    fixed_nodes = _compute_fixed_flow_nodes(locus_info_json, top_n=20)
    if not fixed_nodes:
        st.info("Locus 정보가 없습니다.")
        return

    # locus_id -> 이름 매핑
    locus_meta: dict[str, str] = {}
    if locus_info is not None:
        for _, row in locus_info.iterrows():
            lid = str(row.get("locus_id", ""))
            locus_meta[lid] = str(row.get("locus_name", lid))

    # 캐시된 흐름 계산
    model_hash = f"{sector_id}_{id(model)}"
    tokenizer_hash = f"{sector_id}_{id(tokenizer)}"
    seq_key = str(sorted(sequences.keys())[:50])

    flow_data = _compute_sankey_flows(
        model_hash, tokenizer_hash, seq_key,
        sequences, fixed_nodes, sector_id, max_workers=200,
    )

    sorted_flows = sorted(flow_data.items(), key=lambda x: x[1], reverse=True)[:25]

    # 노드 인덱스 (고정 순서)
    node_idx = {lid: i for i, lid in enumerate(fixed_nodes)}
    node_labels = [locus_meta.get(lid, lid) for lid in fixed_nodes]

    # locus_type 기반 노드 색상
    type_colors = {
        "GATE": "#4A90D9",
        "WORK": "#00AEEF",
        "REST": "#00C897",
        "ADMIN": "#9B59B6",
        "FACILITY": "#E67E22",
        "TRANSPORT": "#F39C12",
        "HAZARD": "#E74C3C",
    }
    node_colors = []
    if locus_info is not None:
        type_map = {str(r["locus_id"]): str(r.get("locus_type", "WORK")) for _, r in locus_info.iterrows()}
        for lid in fixed_nodes:
            ltype = type_map.get(lid, "WORK")
            node_colors.append(type_colors.get(ltype, COLORS["accent"]))
    else:
        node_colors = [COLORS["accent"]] * len(fixed_nodes)

    if sorted_flows:
        source_indices = [node_idx[f[0][0]] for f in sorted_flows]
        target_indices = [node_idx[f[0][1]] for f in sorted_flows]
        values = [f[1] for f in sorted_flows]
    else:
        source_indices, target_indices, values = [], [], []

    fig_sankey = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=12,
            thickness=18,
            line=dict(color=COLORS["border"], width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values if values else [0],
            color="rgba(0, 174, 239, 0.3)",
        ),
    ))
    fig_sankey.update_layout(
        paper_bgcolor=COLORS["bg"],
        font_color=COLORS["text"],
        title="예측 이동 흐름 (노드 고정 | 두께=이동 예측량)",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_sankey, use_container_width=True, config={"displayModeBar": False})


def _render_prediction_summary(model, tokenizer, sector_id: str, sequences: dict, locus_info_json: str):
    """예측 요약 3컬럼."""
    st.markdown(section_header("예측 요약"), unsafe_allow_html=True)

    if model is None or not sequences:
        st.info("예측을 위한 모델 또는 데이터가 없습니다.")
        return

    model_hash = f"{sector_id}_{id(model)}"
    tokenizer_hash = f"{sector_id}_{id(tokenizer)}"

    with st.spinner("공간 상태 예측 중..."):
        states_df = predict_locus_states(
            model_hash, tokenizer_hash,
            sequences, locus_info_json, sector_id, min(500, len(sequences))
        )

    if states_df.empty:
        return

    col_cong, col_btn, col_safe = st.columns(3)

    with col_cong:
        congested = states_df[states_df["congestion_score"] >= 0.7].nlargest(3, "congestion_score")
        st.markdown(
            f"<div style='background:{COLORS['card_bg']}; border-left:4px solid {COLORS['danger']}; "
            f"padding:12px; border-radius:8px;'>"
            f"<div style='color:{COLORS['danger']}; font-weight:700; margin-bottom:8px;'>혼잡 예상 공간</div>",
            unsafe_allow_html=True
        )
        if not congested.empty:
            for _, row in congested.iterrows():
                st.markdown(
                    f"<div style='color:{COLORS['text']}; font-size:0.85rem; padding:2px 0;'>"
                    f"- {row['locus_name']}: 혼잡도 {row['congestion_score']:.0%}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>없음</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_btn:
        bottleneck = states_df[states_df["bottleneck_risk"] >= 0.3].nlargest(3, "bottleneck_risk")
        st.markdown(
            f"<div style='background:{COLORS['card_bg']}; border-left:4px solid {COLORS['warning']}; "
            f"padding:12px; border-radius:8px;'>"
            f"<div style='color:{COLORS['warning']}; font-weight:700; margin-bottom:8px;'>병목 예상 경로</div>",
            unsafe_allow_html=True
        )
        if not bottleneck.empty:
            for _, row in bottleneck.iterrows():
                sign = "+" if row["change"] > 0 else ""
                st.markdown(
                    f"<div style='color:{COLORS['text']}; font-size:0.85rem; padding:2px 0;'>"
                    f"- {row['locus_name']}: {sign}{int(row['change'])}명</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>없음</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_safe:
        hazard_types = ("HAZARD", "TRANSPORT")
        safety_risk = states_df[
            (states_df["locus_type"].isin(hazard_types)) &
            (states_df["predicted_count"] > 0)
        ].nlargest(3, "risk_score")
        st.markdown(
            f"<div style='background:{COLORS['card_bg']}; border-left:4px solid {COLORS['confined']}; "
            f"padding:12px; border-radius:8px;'>"
            f"<div style='color:{COLORS['confined']}; font-weight:700; margin-bottom:8px;'>안전 위험 예상</div>",
            unsafe_allow_html=True
        )
        if not safety_risk.empty:
            for _, row in safety_risk.iterrows():
                st.markdown(
                    f"<div style='color:{COLORS['text']}; font-size:0.85rem; padding:2px 0;'>"
                    f"- {row['locus_name']}: {int(row['predicted_count'])}명 (등급 {int(row['hazard_grade'])})</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>없음</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_hourly_timeline(journey_df: pd.DataFrame, locus_info: pd.DataFrame | None, selected_hour: int):
    """시간대별 인원 추이 — 선택 시각 이전: 실선(측정치), 이후: 점선(전이행렬 예측치).

    - 0~23시 전체 표시 (새벽 포함)
    - 상위 5개 Locus 기준
    - 선택 시각 기준 두 구간을 별도 트레이스로 구분
    """
    st.markdown(section_header("시간대별 인원 추이"), unsafe_allow_html=True)

    time_col = detect_time_column(journey_df)
    if not time_col:
        st.info("시간 컬럼을 찾을 수 없습니다.")
        return

    locus_col = None
    for lc in ["locus_id", "locus", "corrected_locus_id"]:
        if lc in journey_df.columns:
            locus_col = lc
            break
    if not locus_col:
        st.info("Locus 컬럼을 찾을 수 없습니다.")
        return

    df = journey_df.copy()
    df["_hour"] = pd.to_datetime(df[time_col], errors="coerce").dt.hour

    # 측정치: 전체 시간 집계 (0~23)
    hourly = df.groupby(["_hour", locus_col])["user_no"].nunique().reset_index(name="count")
    top_loci = hourly.groupby(locus_col)["count"].sum().nlargest(5).index.tolist()
    hourly_top = hourly[hourly[locus_col].isin(top_loci)].copy()

    # Locus 이름 매핑
    locus_meta: dict[str, str] = {}
    if locus_info is not None:
        locus_meta = {
            str(r["locus_id"]): r.get("locus_name", str(r["locus_id"]))
            for r in locus_info.to_dict("records")
            if r.get("locus_id")
        }
    hourly_top["locus_name"] = hourly_top[locus_col].map(lambda x: locus_meta.get(str(x), str(x)))

    # 색상 팔레트
    palette = [
        COLORS.get("accent", "#00C897"),
        COLORS.get("warning", "#FFB300"),
        COLORS.get("danger", "#FF4C4C"),
        "#7B68EE",
        "#20B2AA",
    ]
    locus_names = hourly_top["locus_name"].unique().tolist()
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(locus_names)}

    fig = go.Figure()

    for lname in locus_names:
        color = color_map[lname]
        sub = hourly_top[hourly_top["locus_name"] == lname].sort_values("_hour")

        # ── 측정치 구간 (selected_hour 이하, 실선)
        actual = sub[sub["_hour"] <= selected_hour]
        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=actual["_hour"],
                y=actual["count"],
                mode="lines+markers",
                name=lname,
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=5),
                legendgroup=lname,
                showlegend=True,
                hovertemplate=f"<b>{lname}</b><br>%{{x}}시: %{{y}}명 (측정)<extra></extra>",
            ))

        # ── 예측치 구간 (selected_hour 초과, 점선)
        future = sub[sub["_hour"] > selected_hour]
        if not future.empty:
            # selected_hour 마지막 측정값과 연결하기 위해 연결점 추가
            connect = sub[sub["_hour"] == selected_hour]
            if not connect.empty:
                future = pd.concat([connect, future], ignore_index=True)
            fig.add_trace(go.Scatter(
                x=future["_hour"],
                y=future["count"],
                mode="lines+markers",
                name=f"{lname} (예측)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5, symbol="circle-open"),
                legendgroup=lname,
                showlegend=True,
                hovertemplate=f"<b>{lname}</b><br>%{{x}}시: %{{y}}명 (예측치)<extra></extra>",
            ))

    # 현재 시각 수직선
    fig.add_vline(
        x=selected_hour,
        line=dict(color=COLORS.get("accent", "#00C897"), dash="dash", width=1.5),
        annotation=dict(
            text=f"현재 ({selected_hour}:00)",
            font=dict(color=COLORS.get("text_muted", "#aaa"), size=11),
            yanchor="top",
        ),
    )

    # 예측 구간 배경 강조
    if selected_hour < 23:
        fig.add_vrect(
            x0=selected_hour, x1=23,
            fillcolor="rgba(100,100,255,0.05)",
            line_width=0,
            annotation_text="예측 구간",
            annotation_position="top left",
            annotation_font=dict(color="rgba(150,150,255,0.7)", size=10),
        )

    fig.update_layout(
        paper_bgcolor=COLORS.get("card_bg", "#1e2530"),
        plot_bgcolor="#111820",
        font_color=COLORS.get("text", "#e0e0e0"),
        title="주요 공간별 시간대 인원 추이  ─  실선: 측정치 / 점선: 예측치",
        xaxis=dict(
            title="시간 (시)",
            dtick=1,
            range=[-0.5, 23.5],   # 0~23시 전체
            tickvals=list(range(0, 24)),
            ticktext=[f"{h:02d}h" for h in range(0, 24)],
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title="인원 수 (명)",
            gridcolor="rgba(255,255,255,0.04)",
        ),
        height=380,
        margin=dict(l=50, r=20, t=55, b=50),
        legend=dict(
            font=dict(color=COLORS.get("text", "#e0e0e0"), size=10),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("💡 점선 구간은 해당 날짜의 실제 기록 데이터입니다. 진짜 미래 예측은 '예측 시뮬레이션' 섹션을 참고하세요.")
