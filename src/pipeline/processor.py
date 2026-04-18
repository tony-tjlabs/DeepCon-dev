"""
Daily Processor — 일별 데이터 처리 모듈
==========================================
하루치 이동 데이터를 처리하여:
  1. spot_name → locus_id 매핑
  2. 작업자별 Journey 구성
  3. Journey 보정 (JOURNEY_CORRECTION_ENABLED=True 시)
  4. Journey 토큰화 (JOURNEY_TOKENIZATION_ENABLED=True 시)
  5. 기본 지표 + EWI/CRE/SII 계산
  6. 데이터 정합성 검증 (DATA_VALIDATION_ENABLED=True 시)
  7. 공간별·업체별 집계

Q. EWI/CRE는 어떻게 계산하나?
A. src/pipeline/metrics.py 참조.
   add_metrics_to_worker(journey_df, worker_df) 호출로 worker_df에 컬럼 추가.

Q. Journey 토큰화란?
A. 1분 단위 raw Journey를 의미 블록(GATE_IN, WORK, REST, TRANSIT 등)으로 변환.
   src/pipeline/tokenizer.py 참조.

Q. 데이터 검증이란?
A. BLE 커버리지, Journey 연속성, 출퇴근 일관성 검증.
   src/pipeline/validator.py 참조.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─── v2 호환: Locus 속성 기반 동적 분류 ─────────────────────────────
def _load_locus_sets(sector_id: str | None = None) -> dict[str, set[str]]:
    """
    locus CSV에서 confined/high_voltage/work/rest locus 집합을 동적으로 생성.

    v1: 하드코딩된 "L-Y1-XXX" 집합 사용
    v2: locus_v2.csv의 building/locus_name/locus_type/function 기반 분류

    Returns:
        {"confined": set, "high_voltage": set, "work_tokens": set, "rest_tokens": set,
         "gate_tokens": set}
    """
    import config as cfg

    sid = sector_id or cfg.SECTOR_ID
    paths = cfg.get_sector_paths(sid)

    if cfg.LOCUS_VERSION == "v2":
        csv_path = paths.get("locus_v2_csv")
    else:
        csv_path = paths.get("locus_csv")

    result = {
        "confined": set(),
        "high_voltage": set(),
        "work_tokens": set(),
        "rest_tokens": set(),
        "gate_tokens": set(),
    }

    if not csv_path or not csv_path.exists():
        logger.warning("locus CSV 없음: %s — 빈 집합 반환", csv_path)
        return result

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        logger.warning("locus CSV 로드 실패: %s", e)
        return result

    df["locus_id"] = df["locus_id"].astype(str).str.strip()

    if cfg.LOCUS_VERSION == "v2":
        # v2: building/locus_name 기반 분류
        # 고압전 구역: building 이 '154kV' 이거나 locus_name 에 '154kv' 포함
        building = df["building"].fillna("").str.lower()
        name = df["locus_name"].fillna("").str.lower()
        hv_mask = building.str.contains("154kv") | name.str.contains("154kv")
        result["high_voltage"] = set(df.loc[hv_mask, "locus_id"])

        # 밀폐공간: locus_name에 '밀폐', '맨홀', 'confined' 포함
        confined_mask = (
            name.str.contains("밀폐")
            | name.str.contains("맨홀")
            | name.str.contains("confined")
        )
        result["confined"] = set(df.loc[confined_mask, "locus_id"])

        # ★ locus_v2.csv의 locus_type/function 기반 직접 분류
        ltype = df.get("locus_type", pd.Series(dtype=str)).fillna("").str.upper()
        func = df.get("function", pd.Series(dtype=str)).fillna("").str.upper()
        token_col = "locus_id"

        # REST: REST_AREA 타입
        rest_mask = (ltype == "REST_AREA") | (func == "REST")
        result["rest_tokens"] = set(df.loc[rest_mask, token_col].dropna())

        # GATE/TRANSIT: GATE + TRANSIT 타입
        transit_mask = (ltype == "GATE") | (func == "ACCESS") | (ltype == "TRANSIT") | (func == "MOVE")
        result["gate_tokens"] = set(df.loc[transit_mask, token_col].dropna()) - result["rest_tokens"]

        # WORK: WORK_AREA/VERTICAL 중 REST/TRANSIT 아닌 것
        work_mask = ((ltype == "WORK_AREA") | (func == "WORK") | (ltype == "VERTICAL")) & ~rest_mask & ~transit_mask
        result["work_tokens"] = set(df.loc[work_mask, token_col].dropna())
    else:
        # v1: 기존 하드코딩 유지
        result["confined"] = {"L-Y1-055", "L-Y1-056", "L-Y1-057"}
        result["high_voltage"] = {"L-Y1-080", "L-Y1-083"}
        result["work_tokens"] = {"work_zone", "outdoor_work"}
        result["rest_tokens"] = {"breakroom", "smoking_area", "dining_hall"}
        result["gate_tokens"] = {"timeclock", "main_gate", "sub_gate"}

    return result


# ─── Step 1: Locus 매핑 ───────────────────────────────────────────
def apply_locus_mapping(journey_df: pd.DataFrame, spot_map: dict) -> pd.DataFrame:
    """
    [v1] spot_name → locus_id, locus_token 매핑 적용.

    Q. 매핑 안 된 spot_name은?
    A.  locus_id="unmapped", locus_token="unknown" 으로 표시.
        추후 spot_name_map.json 업데이트로 해소.
    """
    df = journey_df.copy()

    def _map(spot_name: str) -> tuple[str, str]:
        entry = spot_map.get(spot_name, {})
        return (
            entry.get("locus_id", "unmapped"),
            entry.get("locus_token", "unknown"),
        )

    mapped = df["spot_name"].apply(_map)
    mapped_df = pd.DataFrame(mapped.tolist(), columns=["locus_id", "locus_token"], index=df.index)
    df["locus_id"] = mapped_df["locus_id"]
    df["locus_token"] = mapped_df["locus_token"]

    unmapped_ratio = (df["locus_id"] == "unmapped").mean() * 100
    if unmapped_ratio > 5:
        logger.info(f"  ⚠ unmapped locus 비율 높음: {unmapped_ratio:.1f}%")

    return df


def apply_locus_mapping_v2(journey_df: pd.DataFrame, gateway_index) -> pd.DataFrame:
    """
    [v2] 좌표 기반 최근접 S-Ward 할당 → locus_id, locus_token 매핑.

    작업자 (building_name, floor_name, x, y) → nearest Gateway → GW-{no}
    GatewayIndex.assign_batch()를 사용하여 벡터화 배치 할당.
    """
    df = journey_df.copy()

    # assign_batch 실행 — building_name, floor_name, x, y 컬럼 사용
    df = gateway_index.assign_batch(
        df,
        building_col="building_name",
        level_col="floor_name",
        x_col="x",
        y_col="y",
        place_col="spot_name",
    )

    # gw_locus_id → locus_id, locus_token, locus_name 매핑
    df["locus_id"] = df["gw_locus_id"].fillna("unmapped")
    # locus_token = locus_id (GW-XXX) — Deep Space 토크나이저와 일치해야 함
    df["locus_token"] = df["locus_id"]
    # locus_name = gateway_name — 표시용 (사람이 읽는 이름)
    gw_meta = gateway_index._gw_meta
    df["locus_name"] = df["gw_gateway_no"].apply(
        lambda no: gw_meta.get(int(no), {}).get("gateway_name", "unknown")
        if pd.notna(no) and no >= 0 else "unknown"
    )

    # 통계 출력
    unmapped = (df["locus_id"] == "unmapped").sum()
    total = len(df)
    if total > 0:
        matched = total - unmapped
        logger.info(f"  ✓ v2 Locus 할당: {matched:,}/{total:,} ({matched/total*100:.1f}%)")
        if unmapped > 0:
            logger.info(f"  ⚠ unmapped: {unmapped:,}건 ({unmapped/total*100:.1f}%)")

    # 임시 gw_* 컬럼 정리 (locus_id, locus_token만 유지)
    drop_cols = [c for c in df.columns if c.startswith("gw_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ─── Step 2: Journey 구성 ─────────────────────────────────────────
def build_worker_journeys(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별로 시간순 정렬 + 이전 위치 정보 추가.

    추가 컬럼:
        seq          : 작업자 내 시간순 번호 (0부터)
        prev_locus   : 직전 locus_id
        locus_changed: 위치 변경 여부 (bool)
        dwell_min    : 현재 locus 체류 시간(분, 연속 체류 기준)
        is_transition: 이동 구간 여부

    Q. 보정 로직 없이 raw를 그대로 사용하면 문제없나?
    A.  BLE 신호 특성상 순간적인 위치 튐(노이즈)이 있음.
        현재는 is_work_hour 필터 + signal_count>=1 만 적용.
        DeepCon_SOIF의 DBSCAN 기반 보정은 추후 JOURNEY_CORRECTION_ENABLED=True 시 적용.
    """
    df = journey_df.copy()

    # 근무 시간 내 + 최소 신호 필터
    df = df[df["signal_count"] >= 1].copy()

    # 시간순 정렬
    df = df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)

    # 작업자 내 순번
    df["seq"] = df.groupby("user_no").cumcount()

    # 직전 locus (같은 작업자 내)
    df["prev_locus"] = df.groupby("user_no")["locus_id"].shift(1).fillna("")

    # 위치 변경 여부
    df["locus_changed"] = df["locus_id"] != df["prev_locus"]
    df.loc[df["seq"] == 0, "locus_changed"] = False  # 첫 기록은 변경 아님

    # 이동 구간 (locus_id가 변경된 시점)
    df["is_transition"] = df["locus_changed"]

    # dwell 블록 ID (연속 체류 구간)
    df["dwell_block"] = (
        df.groupby("user_no")["locus_changed"].cumsum()
    )

    # dwell_min: 각 dwell_block의 체류 시간(분) — 첫/마지막 timestamp 차이
    dwell_times = df.groupby(["user_no", "dwell_block"])["timestamp"].transform(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    df["dwell_min"] = dwell_times

    return df


# ─── Step 3: 기본 지표 계산 ───────────────────────────────────────
def calc_basic_metrics(journey_df: pd.DataFrame, access_df: pd.DataFrame) -> pd.DataFrame:
    """
    작업자별 기본 지표 집계 (벡터화 — 루프 없음).

    반환 컬럼:
        user_no, user_name, company_name,
        in_datetime, out_datetime, work_minutes (출입 기록 기반),
        recorded_minutes    : TWardData 기록 분 수 (gap-fill 포함 전체 journey 행, is_work_hour 무관)
        active_minutes      : active_ratio > 0 인 분 수 — is_work_hour 필터 후
                              (폭넓은 BLE 활성 신호 기준, "발길 통과"까지 포함)
        valid_ble_minutes   : active_count > 0 AND is_work_hour — 더 엄격한 BLE 활성 기준
                              ("실제 근무 중 움직임 있는 BLE" — EWI 분자의 recorded_work_min 정의)
                              ※ active_count>0 ⇒ active_ratio>0 이지만 역은 아님. valid_ble_minutes ≤ active_minutes.
        unique_loci         : 방문한 고유 Locus 수
        locus_sequence      : 방문 Locus 시퀀스 (토큰 문자열, AI 학습용)
        confined_minutes    : 밀폐공간 체류 시간(분) — is_work_hour 필터 후
        high_voltage_minutes: 고압전 구역 체류 시간(분) — is_work_hour 필터 후
        transition_count    : 이동 횟수
        work_zone_minutes   : 작업구역 체류 시간(분)
        rest_minutes        : 휴게 시간(분)

    Note: EWI, CRE, SII는 add_metrics_to_worker() 에서 별도 계산.
    """
    locus_sets = _load_locus_sets()
    CONFINED_LOCI = locus_sets["confined"]
    HIGH_V_LOCI   = locus_sets["high_voltage"]
    WORK_TOKENS   = locus_sets["work_tokens"]
    REST_TOKENS   = locus_sets["rest_tokens"]

    df = journey_df.copy()
    df["_confined"]     = df["locus_id"].isin(CONFINED_LOCI).astype(int)
    df["_high_voltage"] = df["locus_id"].isin(HIGH_V_LOCI).astype(int)
    df["_work_zone"]    = df["locus_token"].isin(WORK_TOKENS).astype(int)
    df["_rest"]         = df["locus_token"].isin(REST_TOKENS).astype(int)
    df["_active"]       = (df["active_ratio"] > 0).astype(int)

    # ★ is_work_hour 필터 적용 — 퇴근 후 헬멧 방치분을 work_zone/rest/active 집계에서 제외
    # calc_ewi_all_workers()와 분모를 맞추기 위한 단일화 조치 (audit P1/H2)
    # recorded_minutes는 전체 journey 행 수 유지 (gap-fill 포함 원시 지표)
    if "is_work_hour" in df.columns:
        _wh = df[df["is_work_hour"]].copy()
    else:
        _wh = df

    # ── 기본 집계 (groupby 한 번) ──────────────────────────────────
    # recorded_minutes는 전체 df 기반, 나머지는 is_work_hour 필터 후 집계
    recorded = df.groupby("user_no").agg(
        user_name        = ("user_name",      "first"),
        company_name     = ("company_name",   "first"),
        recorded_minutes = ("user_no",        "count"),
        unique_loci      = ("locus_id",       "nunique"),
    ).reset_index()
    wh_agg = _wh.groupby("user_no").agg(
        active_minutes   = ("_active",        "sum"),
        confined_minutes = ("_confined",      "sum"),
        high_voltage_minutes = ("_high_voltage", "sum"),
        transition_count = ("is_transition",  "sum"),
        work_zone_minutes = ("_work_zone",    "sum"),
        rest_minutes     = ("_rest",          "sum"),
    ).reset_index()
    agg = recorded.merge(wh_agg, on="user_no", how="left")
    # is_work_hour 바깥 작업자는 NaN → 0으로 채움
    for _c in ("active_minutes", "confined_minutes", "high_voltage_minutes",
               "transition_count", "work_zone_minutes", "rest_minutes"):
        agg[_c] = agg[_c].fillna(0).astype(int)

    # ── valid_ble_minutes: is_work_hour + active_count>0 (헬멧 방치 제외) ──
    # 퇴근 후 헬멧 방치 시 active_count=0 BLE가 계속 잡히므로 이를 제외한
    # "실제 근무 중 움직임 있는 BLE" 기록 수를 별도 집계
    valid_ble = df[df["is_work_hour"] & (df["active_count"] > 0)] if "is_work_hour" in df.columns else df[df["active_count"] > 0]
    valid_agg = (
        valid_ble.groupby("user_no")
        .size()
        .reset_index(name="valid_ble_minutes")
    )
    agg = agg.merge(valid_agg, on="user_no", how="left")
    agg["valid_ble_minutes"] = agg["valid_ble_minutes"].fillna(0).astype(int)

    # ── Locus 시퀀스 (token dedup) — 작업자별 ─────────────────────
    # 정렬된 상태에서 shift 비교로 dedup
    df_sorted = df.sort_values(["user_no", "timestamp"])
    df_sorted["_prev_token"] = df_sorted.groupby("user_no")["locus_token"].shift(1)
    df_sorted["_token_changed"] = df_sorted["locus_token"] != df_sorted["_prev_token"]
    df_sorted.loc[df_sorted.groupby("user_no").head(1).index, "_token_changed"] = True

    seq_df = (
        df_sorted[df_sorted["_token_changed"]]
        .groupby("user_no")["locus_token"]
        .apply(lambda x: " ".join(x.fillna("unknown").astype(str)))
        .reset_index()
        .rename(columns={"locus_token": "locus_sequence"})
    )
    agg = agg.merge(seq_df, on="user_no", how="left")

    # ── AccessLog join (출입 시간, 근무 시간, shift_type) ─────────
    ac_cols = ["user_no", "in_datetime", "out_datetime", "work_minutes",
               "shift_type", "exit_source"]
    ac_cols = [c for c in ac_cols if c in access_df.columns]
    agg = agg.merge(access_df[ac_cols], on="user_no", how="left")

    # shift_type / exit_source 기본값
    if "shift_type" in agg.columns:
        agg["shift_type"]  = agg["shift_type"].fillna("unknown")
        agg["exit_source"] = agg["exit_source"].fillna("missing")

    # ── helmet_abandoned: 헬멧 방치 의심 탐지 ─────────────────────
    # 조건: 퇴근 기록 있음 + 근무시간 < 60분 + valid BLE < 5분 + 전체 BLE > 120분
    # → 실제 작업자 없이 헬멧만 현장에 방치된 케이스
    agg["helmet_abandoned"] = (
        (agg["exit_source"] == "access_log")
        & (agg["work_minutes"] < 60)
        & (agg["valid_ble_minutes"] < 5)
        & (agg["recorded_minutes"] > 120)
    )

    # 타입 정리
    for col in ["confined_minutes", "high_voltage_minutes", "transition_count",
                "work_zone_minutes", "rest_minutes", "active_minutes"]:
        agg[col] = agg[col].fillna(0).astype(int)

    return agg


# ─── Step 4: 공간별 집계 ──────────────────────────────────────────
def calc_space_metrics(journey_df: pd.DataFrame) -> pd.DataFrame:
    """
    공간(locus_id)별 집계.

    반환 컬럼:
        locus_id, locus_token, locus_name,
        total_person_minutes : 총 체류 인분(명×분)
        unique_workers       : 방문 고유 작업자 수
        unique_companies     : 방문 고유 업체 수
        avg_signal_count     : 평균 신호 수
        avg_active_ratio     : 평균 활성 비율
        is_confined          : 밀폐공간 여부
    """
    locus_sets = _load_locus_sets()
    CONFINED_LOCI = locus_sets["confined"]
    HIGH_V_LOCI   = locus_sets["high_voltage"]

    # ★ is_work_hour 필터 적용 — 퇴근 후 헬멧 방치분을 공간 집계에서 제외
    # calc_ewi_all_workers()와 분모를 맞추기 위한 단일화 조치 (audit H2)
    if "is_work_hour" in journey_df.columns:
        _jdf = journey_df[journey_df["is_work_hour"]].copy()
    else:
        _jdf = journey_df

    # 벡터화 집계 (for-loop 제거)
    space = (
        _jdf.groupby("locus_id", sort=False)
        .agg(
            locus_token          = ("locus_token",   "first"),
            total_person_minutes = ("locus_id",      "count"),
            unique_workers       = ("user_no",       "nunique"),
            unique_companies     = ("company_name",  "nunique"),
            avg_signal_count     = ("signal_count",  "mean"),
            avg_active_ratio     = ("active_ratio",  "mean"),
        )
        .reset_index()
    )

    space["avg_signal_count"] = space["avg_signal_count"].round(2)
    space["avg_active_ratio"] = space["avg_active_ratio"].round(3)
    space["is_confined"]      = space["locus_id"].isin(CONFINED_LOCI)
    space["is_high_voltage"]  = space["locus_id"].isin(HIGH_V_LOCI)

    return space.sort_values("total_person_minutes", ascending=False).reset_index(drop=True)


# ─── Step 5: 업체별 집계 ──────────────────────────────────────────
def calc_company_metrics(journey_df: pd.DataFrame, worker_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    업체(company_name)별 집계.

    반환 컬럼:
        company_name,
        worker_count         : 이동 데이터 있는 작업자 수
        total_person_minutes : 총 체류 인분
        avg_work_zone_minutes: 평균 작업구역 체류 시간
        avg_rest_minutes     : 평균 휴게 시간
        confined_workers     : 밀폐공간 진입 작업자 수
        total_confined_minutes: 총 밀폐공간 체류 인분
    """
    agg_journey = journey_df.groupby("company_name").agg(
        worker_count         = ("user_no",       "nunique"),
        total_person_minutes = ("user_no",       "count"),
    ).reset_index()

    agg_worker_cols = {
        "avg_work_zone_minutes":  ("work_zone_minutes",  "mean"),
        "avg_rest_minutes":       ("rest_minutes",       "mean"),
        "confined_workers":       ("confined_minutes",   lambda x: int((x > 0).sum())),
        "total_confined_minutes": ("confined_minutes",   "sum"),
    }
    # EWI/CRE 집계 (존재할 경우)
    if "ewi" in worker_metrics.columns:
        agg_worker_cols["avg_ewi"] = ("ewi", "mean")
    if "cre" in worker_metrics.columns:
        agg_worker_cols["avg_cre"] = ("cre", "mean")
    if "high_active_min" in worker_metrics.columns:
        agg_worker_cols["total_high_active_min"] = ("high_active_min", "sum")

    agg_worker = worker_metrics.groupby("company_name").agg(**agg_worker_cols).reset_index()
    for c in ["avg_work_zone_minutes", "avg_rest_minutes", "avg_ewi", "avg_cre"]:
        if c in agg_worker.columns:
            agg_worker[c] = agg_worker[c].round(3)

    result = agg_journey.merge(agg_worker, on="company_name", how="left")
    return result.sort_values("worker_count", ascending=False).reset_index(drop=True)


# ─── 전체 처리 오케스트레이터 ─────────────────────────────────────
def process_daily(
    journey_df: pd.DataFrame,
    access_df: pd.DataFrame,
    spot_map: dict,
    date_str: str,
    progress_callback=None,
    gateway_index=None,
) -> dict[str, pd.DataFrame]:
    """
    하루치 데이터 전체 처리.

    Args:
        spot_map: v1용 spot_name → locus 매핑 dict
        gateway_index: v2용 GatewayIndex 인스턴스 (None이면 v1 사용)

    반환:
        {
          "journey"  : 작업자별 1분 위치 (locus 포함),
          "worker"   : 작업자별 지표,
          "space"    : 공간별 지표,
          "company"  : 업체별 지표,
        }
    """
    def _log(msg: str, pct: int):
        logger.info(f"  [{pct:3d}%] {msg}")
        if progress_callback:
            progress_callback(pct, msg)

    import config as cfg
    from src.pipeline.corrector import correct_journeys

    # Locus 매핑: v2 (좌표 기반) 또는 v1 (spot_name 기반)
    if cfg.LOCUS_VERSION == "v2" and gateway_index is not None:
        _log(f"Locus v2 매핑 중 (S-Ward 좌표 기반)... ({len(journey_df):,}건)", 10)
        journey_df = apply_locus_mapping_v2(journey_df, gateway_index)
    else:
        _log(f"Locus v1 매핑 중... ({len(journey_df):,}건)", 10)
        journey_df = apply_locus_mapping(journey_df, spot_map)

    _log("Journey 구성 중...", 25)
    journey_df = build_worker_journeys(journey_df)

    # ── Journey 보정 ───────────────────────────────────────────────────────
    correction_stats: dict = {}
    if cfg.JOURNEY_CORRECTION_ENABLED:
        _log("Journey 보정 중 (슬라이딩 윈도우 + DBSCAN)...", 40)
        journey_df, corr_stats = correct_journeys(journey_df, sector_id=cfg.SECTOR_ID)

        # 보정 후 seq / locus_changed / dwell_block 재계산
        journey_df = journey_df.sort_values(["user_no", "timestamp"]).reset_index(drop=True)
        # ★ v2: 보정으로 locus_id가 변경되었으므로 locus_token 재동기화
        if cfg.LOCUS_VERSION == "v2":
            journey_df["locus_token"] = journey_df["locus_id"]
        journey_df["seq"] = journey_df.groupby("user_no").cumcount()  # ★ seq 재생성
        journey_df["prev_locus"]   = journey_df.groupby("user_no")["locus_id"].shift(1).fillna("")
        journey_df["locus_changed"] = journey_df["locus_id"] != journey_df["prev_locus"]
        journey_df.loc[journey_df["seq"] == 0, "locus_changed"] = False
        journey_df["is_transition"] = journey_df["locus_changed"]
        journey_df["dwell_block"]   = journey_df.groupby("user_no")["locus_changed"].cumsum()

        correction_stats = corr_stats
        _log(
            f"  ✓ 보정 완료: {corr_stats['corrected_records']:,}건 "
            f"({corr_stats['correction_ratio']:.2f}%) "
            f"/ 작업자 {corr_stats['workers_corrected']}명",
            50,
        )
    else:
        _log("Journey 보정 미적용 (JOURNEY_CORRECTION_ENABLED=False)", 40)

    # ── Gap Analysis (BLE 음영 탐지 + 채우기 + 활성 레벨 분류) ──────────
    gap_stats: dict = {}
    if cfg.GAP_ANALYSIS_ENABLED:
        _log("Gap 분석 중 (BLE 음영 탐지·채우기)...", 52)
        try:
            from src.pipeline.gap_analyzer import analyze_gaps
            journey_df, gap_stats = analyze_gaps(journey_df, sector_id=cfg.SECTOR_ID)
            _log(
                f"  ✓ Gap 분석 완료: {gap_stats['total_gaps']:,}개 Gap → "
                f"채움 {gap_stats['filled_gaps']:,}개 ({gap_stats['filled_records']:,}건) "
                f"/ 스킵 {gap_stats['skipped_gaps']:,}개",
                55,
            )
        except Exception as e:
            logger.info(f"  ⚠ Gap 분석 실패 (계속 진행): {e}")

    # ── Coverage 계산 (AccessLog 기반 BLE 신뢰도 — Deep Space 훈련 품질) ─
    coverage_df: pd.DataFrame = pd.DataFrame()
    if cfg.JOURNEY_RECONSTRUCTION_ENABLED:
        _log("Coverage 계산 중 (BLE 신뢰도)...", 57)
        try:
            from src.pipeline.journey_reconstructor import calc_coverage_ratio
            worker_access = (
                access_df[["user_no", "in_datetime", "out_datetime", "work_minutes"]]
                .drop_duplicates("user_no")
                .copy()
            )
            coverage_df = calc_coverage_ratio(journey_df, worker_access)
            mean_cov   = coverage_df["coverage_pct"].mean()   if not coverage_df.empty else 0.0
            n_sufficient = int(
                (coverage_df["coverage_ratio"] >= cfg.COVERAGE_THRESHOLD).sum()
            ) if not coverage_df.empty else 0
            n_total_cov  = len(coverage_df)
            _log(
                f"  ✓ Coverage 완료: 평균 {mean_cov:.1f}% "
                f"/ threshold({cfg.COVERAGE_THRESHOLD*100:.0f}%) 이상 "
                f"{n_sufficient}/{n_total_cov}명",
                59,
            )
        except Exception as e:
            logger.info(f"  ⚠ Coverage 계산 실패 (계속 진행): {e}")

    _log("작업자별 기본 지표 계산 중...", 60)
    worker_df = calc_basic_metrics(journey_df, access_df)

    _log("EWI / CRE / SII 계산 중...", 72)
    try:
        from src.pipeline.metrics import add_metrics_to_worker
        # ★ L1: 현재 sector_id 명시 전달 (멀티 섹터 전환 후에도 정확한 토큰 집합 사용)
        _sid = getattr(cfg, "SECTOR_ID", None)
        worker_df = add_metrics_to_worker(journey_df, worker_df, sector_id=_sid)
    except Exception as e:
        logger.info(f"  ⚠ EWI/CRE 계산 실패 (기본 지표만 저장): {e}")

    _log("Sanity check (물리/생리 불가 패턴 탐지) 중...", 76)
    try:
        from src.pipeline.sanity_checker import check_worker_sanity
        worker_df = check_worker_sanity(worker_df)
    except Exception as e:
        logger.info(f"  ⚠ Sanity check 실패 (계속 진행): {e}")

    _log("공간별 집계 중...", 82)
    space_df = calc_space_metrics(journey_df)

    # ── Journey 토큰화 (의미 블록 추출) ──────────────────────────────────
    tokenization_stats: dict = {}
    spatial_graph = None
    if cfg.JOURNEY_TOKENIZATION_ENABLED:
        _log("Journey 토큰화 중 (의미 블록 추출)...", 78)
        try:
            from src.spatial.graph import get_spatial_graph
            from src.pipeline.tokenizer import add_journey_blocks, get_tokenization_stats

            # Sector ID 추출 (date_str에서 못 가져오면 기본값)
            sector_id = cfg.SECTOR_ID  # 기본 Sector
            spatial_graph = get_spatial_graph(sector_id)

            journey_df = add_journey_blocks(journey_df, spatial_graph)
            tokenization_stats = get_tokenization_stats(journey_df)

            block_count = len(journey_df["block_id"].unique()) if "block_id" in journey_df.columns else 0
            _log(f"  ✓ 토큰화 완료: {block_count:,}개 블록 추출", 80)
        except Exception as e:
            logger.info(f"  ⚠ Journey 토큰화 실패 (기본 모드로 계속): {e}")
            tokenization_stats = {"error": str(e)}
    else:
        _log("Journey 토큰화 미적용 (JOURNEY_TOKENIZATION_ENABLED=False)", 78)

    _log("업체별 집계 중...", 85)
    company_df = calc_company_metrics(journey_df, worker_df)

    # ── 데이터 정합성 검증 ─────────────────────────────────────────────
    validation_results: dict = {}
    if cfg.DATA_VALIDATION_ENABLED:
        _log("데이터 정합성 검증 중...", 92)
        try:
            # ★ v3 (T-03): run_validation_suite → 레거시 3검증 + DSL DEFAULT_CHECKS
            #   NaN-safe 처리 내장 → 이전의 "Cannot mask with non-boolean array..." 버그 해소
            from src.pipeline.validator import run_validation_suite

            quality_report = run_validation_suite(
                journey_df=journey_df,
                worker_df=worker_df,
                access_df=access_df,
                spatial_graph=spatial_graph,
            )
            validation_results = quality_report

            _log(
                f"  ✓ 검증 완료: {quality_report['summary']} "
                f"(품질 점수 {quality_report['overall_score']:.2f})",
                95,
            )
        except Exception as e:
            logger.info(f"  ⚠ 데이터 검증 실패 (계속 진행): {e}")
            validation_results = {"error": str(e)}
    else:
        _log("데이터 검증 미적용 (DATA_VALIDATION_ENABLED=False)", 92)

    _log("완료", 100)

    # ── 작업자 이름 마스킹 ────────────────────────────────────────────────
    from src.pipeline.anonymizer import mask_name_series

    if "user_name" in journey_df.columns:
        journey_df["user_name"] = mask_name_series(journey_df["user_name"])
    if "user_name" in worker_df.columns:
        worker_df["user_name"] = mask_name_series(worker_df["user_name"])

    # ── 통계 최종 정리 ────────────────────────────────────────────────
    total_records  = len(journey_df)
    unmapped_cnt   = int((journey_df["locus_id"] == "unmapped").sum())
    unmapped_ratio = round(unmapped_cnt / total_records * 100, 2) if total_records > 0 else 0.0

    stats = {
        # 보정 통계
        "journey_corrected":  cfg.JOURNEY_CORRECTION_ENABLED,
        "corrected_records":  correction_stats.get("corrected_records", 0),
        "correction_ratio":   correction_stats.get("correction_ratio",  0.0),
        "workers_corrected":  correction_stats.get("workers_corrected", 0),
        "unmapped_records":   unmapped_cnt,
        "unmapped_ratio":     unmapped_ratio,
        # Gap 분석 통계 (M15X 이식)
        "gap_analysis_enabled": cfg.GAP_ANALYSIS_ENABLED,
        "gap_total":            gap_stats.get("total_gaps",       0),
        "gap_filled":           gap_stats.get("filled_gaps",      0),
        "gap_filled_records":   gap_stats.get("filled_records",   0),
        "gap_skipped":          gap_stats.get("skipped_gaps",     0),
        "gap_workers":          gap_stats.get("gap_workers",      0),
        "gap_avg_min":          gap_stats.get("avg_gap_min",      0.0),
        "gap_type_dist":        gap_stats.get("gap_type_dist",    {}),
        # Coverage 통계 (M15X 이식)
        "coverage_enabled":     cfg.JOURNEY_RECONSTRUCTION_ENABLED,
        "coverage_threshold":   cfg.COVERAGE_THRESHOLD,
        "coverage_mean_pct":    float(coverage_df["coverage_pct"].mean())   if not coverage_df.empty else 0.0,
        "coverage_median_pct":  float(coverage_df["coverage_pct"].median()) if not coverage_df.empty else 0.0,
        "coverage_sufficient":  int((coverage_df["coverage_ratio"] >= cfg.COVERAGE_THRESHOLD).sum())
                                if not coverage_df.empty else 0,
        # 토큰화 통계
        "tokenization_enabled": cfg.JOURNEY_TOKENIZATION_ENABLED,
        "tokenization_stats":   tokenization_stats,
        # 검증 결과
        "validation_enabled":   cfg.DATA_VALIDATION_ENABLED,
        "validation":           validation_results,
    }

    return {
        "journey":  journey_df,
        "worker":   worker_df,
        "space":    space_df,
        "company":  company_df,
        "coverage": coverage_df,   # ★ BLE Coverage per worker (Deep Space 훈련 품질)
        "stats":    stats,
    }
