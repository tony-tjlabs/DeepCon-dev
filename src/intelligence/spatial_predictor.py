"""
Spatial Predictor — Agentic AI 핵심 엔진
=========================================
Intelligence Engine (EWI/CRE/SII) + Deep Space (Transformer) 통합.

4가지 예측:
  1. 혼잡도 예측 (Congestion Forecast)
  2. 병목 예측 (Bottleneck Prediction)
  3. 안전 위험 예측 (Safety Risk Prediction)
  4. 생산성 예측 (Productivity Prediction)

v1.2 — 리팩토링: 임계값 상수화, severity 매핑 통합, risk 분류 메서드 추출
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 용량 기본값 (config.py 단일 소스) ────────────────────────
from config import DEFAULT_CAPACITY_BY_TYPE as DEFAULT_CAPACITY
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

# ─── 심각도 ↔ 정수 매핑 (모듈 전역 단일 소스) ──────────────────
SEVERITY_ORDER: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# ─── 예측 임계값 (튜닝 포인트 명시) ─────────────────────────────
# 혼잡도: congestion_pct 기준
CONGESTION_THRESHOLDS = {"critical": 1.2, "high": 1.0, "medium": 0.8, "alert": 0.7}
# 병목: bottleneck_score 또는 순유입 인원 기준
BOTTLENECK_THRESHOLDS = {
    "score": {"critical": 0.5, "high": 0.3, "medium": 0.15},
    "count": {"critical": 10, "high": 7, "medium": 5},
    "alert": 3,  # 최소 순유입
}
# 안전: risk_score 기준
SAFETY_THRESHOLDS = {
    "risk_score": {"critical": 0.5, "high": 0.3, "medium": 0.15},
    "prob_min": 0.15,      # 최소 예측 확률
    "cre_alert": 0.4,      # CRE 경고 기준
    "cre_surge": 0.15,     # CRE 급등 판정
    "fatigue_alert": 0.5,  # 피로 경고
    "cre_boost_mult": 2.0, # CRE 급등 가중치
}
# 생산성
PRODUCTIVITY_THRESHOLDS = {
    "fatigue_alert": 0.5,       # 피로 경고 시작
    "fatigue_critical": 0.7,    # 피로 위험
    "ewi_drop_min": 0.1,       # EWI 하락 최소치
    "high_movement_mult": 2.0, # 평균 대비 이동 과다 배수
    "high_movement_abs": 0.04, # 절대 이동 빈도 기준
    "low_util_pct": 0.5,       # 작업공간 비율 하한
    "low_util_ewi_mult": 0.7,  # 평균 EWI 대비 하한
    "company_alert_min": 0.2,  # 업체 경고율 최소
}


def _safe_float(val, default: float = 0.0) -> float:
    """NaN-safe float 변환."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """NaN-safe int 변환."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else int(f)
    except (ValueError, TypeError):
        return default


# ─── 데이터 모델 ─────────────────────────────────────────────

@dataclass
class CongestionAlert:
    """혼잡도 예측 결과."""
    locus_id: str
    locus_name: str
    current_count: int
    predicted_count: int
    capacity: int
    congestion_pct: float
    trend: str                   # "급증", "증가", "유지", "감소"
    severity: str                # "critical", "high", "medium", "low"
    inflow: int
    outflow: int
    recommendation: str = ""


@dataclass
class BottleneckAlert:
    """병목 예측 결과."""
    locus_id: str
    locus_name: str
    inflow: int
    outflow: int
    net_accumulation: int
    bottleneck_score: float
    severity: str
    upstream_sources: list[tuple[str, int]] = field(default_factory=list)
    downstream_targets: list[tuple[str, int]] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class SafetyAlert:
    """안전 위험 예측 결과."""
    user_no: str
    user_name: str
    company_name: str
    current_locus: str
    predicted_locus: str
    predicted_locus_name: str
    prediction_prob: float
    hazard_level: str
    hazard_grade: int
    cre: float
    fatigue_score: float
    risk_score: float
    severity: str
    reason: str
    cre_delta: float = 0.0       # 전일 대비 CRE 변화
    recommendation: str = ""


@dataclass
class ProductivityAlert:
    """생산성 예측 결과."""
    user_no: str
    user_name: str
    company_name: str
    current_ewi: float
    predicted_ewi: float
    ewi_trend: str
    work_zone_pct: float
    transition_rate: float
    fatigue_score: float
    risk_type: str
    severity: str
    recommendation: str = ""


@dataclass
class CompanyProductivitySummary:
    """업체 단위 생산성 요약."""
    company_name: str
    worker_count: int
    alert_count: int
    alert_pct: float             # alert_count / worker_count
    avg_ewi: float
    avg_fatigue: float
    dominant_risk: str           # 가장 많은 risk_type
    severity: str
    recommendation: str = ""


@dataclass
class AgenticReport:
    """Agentic AI 종합 리포트."""
    date: str
    sector_id: str
    total_workers: int

    congestion_alerts: list[CongestionAlert] = field(default_factory=list)
    bottleneck_alerts: list[BottleneckAlert] = field(default_factory=list)
    safety_alerts: list[SafetyAlert] = field(default_factory=list)
    productivity_alerts: list[ProductivityAlert] = field(default_factory=list)
    company_summaries: list[CompanyProductivitySummary] = field(default_factory=list)

    critical_count: int = 0
    high_count: int = 0

    @property
    def total_alerts(self) -> int:
        return (len(self.congestion_alerts) + len(self.bottleneck_alerts)
                + len(self.safety_alerts) + len(self.productivity_alerts))

    def summary_text(self) -> str:
        """LLM 프롬프트용 요약 (개인정보 마스킹)."""
        from src.utils.anonymizer import mask_name

        lines = [f"[Agentic AI 예측 — {self.date}]"]
        if self.congestion_alerts:
            lines.append(f"혼잡 경고 {len(self.congestion_alerts)}건:")
            for a in self.congestion_alerts[:3]:
                lines.append(f"  - {a.locus_name}: {a.congestion_pct:.0%} ({a.trend})")
                if a.recommendation:
                    lines.append(f"    권고: {a.recommendation}")
        if self.bottleneck_alerts:
            lines.append(f"병목 경고 {len(self.bottleneck_alerts)}건:")
            for a in self.bottleneck_alerts[:3]:
                lines.append(f"  - {a.locus_name}: 순유입 {a.net_accumulation}명")
        if self.safety_alerts:
            lines.append(f"안전 경고 {len(self.safety_alerts)}건:")
            for a in self.safety_alerts[:3]:
                lines.append(f"  - {mask_name(a.user_name)}: {a.reason}")
        if self.productivity_alerts:
            lines.append(f"생산성 경고 {len(self.productivity_alerts)}건:")
            for a in self.productivity_alerts[:3]:
                lines.append(f"  - {mask_name(a.user_name)}: {a.risk_type}")
        if self.company_summaries:
            lines.append(f"업체별 생산성 경고 {len(self.company_summaries)}건:")
            for c in self.company_summaries[:3]:
                lines.append(f"  - {c.company_name}: {c.alert_count}/{c.worker_count}명 ({c.dominant_risk})")
        return "\n".join(lines)


# ─── 메인 예측 엔진 ─────────────────────────────────────────

class SpatialPredictor:
    """
    Intelligence Engine + Deep Space 통합 예측기.

    Architecture:
        worker_df (EWI/CRE/SII) ──┐
        journey_df (sequences)  ───┼──→ SpatialPredictor ──→ AgenticReport
        locus_info (capacity)  ────┤
        Deep Space Model        ───┤
        prev_worker_df (추세)   ───┘
    """

    def __init__(
        self,
        model,
        tokenizer,
        worker_df: pd.DataFrame | None,
        journey_df: pd.DataFrame | None,
        locus_info: pd.DataFrame | None,
        sequences: dict[str, list[str]] | None = None,
        prev_worker_df: pd.DataFrame | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.worker_df = worker_df
        self.journey_df = journey_df
        self.locus_info = locus_info
        self.prev_worker_df = prev_worker_df

        # 시퀀스 추출
        if sequences is not None:
            self.sequences = sequences
        elif journey_df is not None:
            from src.dashboard.deep_space.helpers import extract_worker_sequences
            self.sequences = extract_worker_sequences(journey_df)
        else:
            self.sequences = {}

        # 메타 정보
        self._locus_meta = self._build_locus_meta()
        self._worker_metrics = self._build_worker_metrics()
        self._prev_cre = self._build_prev_cre()

    def _build_locus_meta(self) -> dict[str, dict]:
        """locus_info → {locus_id: {name, type, capacity, hazard_level, hazard_grade}}."""
        meta = {}
        if self.locus_info is None or self.locus_info.empty:
            return meta
        for row in self.locus_info.to_dict("records"):
            lid = str(row.get("locus_id", ""))
            if not lid:
                continue
            ltype = str(row.get("locus_type", "WORK")).upper()
            meta[lid] = {
                "name": str(row.get("locus_name", lid)),
                "type": ltype,
                "capacity": _safe_int(row.get("capacity"), DEFAULT_CAPACITY.get(ltype, 50)),
                "hazard_level": str(row.get("hazard_level", "low")),
                "hazard_grade": _safe_int(row.get("hazard_grade"), 1),
            }
        return meta

    def _build_worker_metrics(self) -> dict[str, dict]:
        """worker_df → {user_no: {ewi, cre, fatigue_score, ...}}."""
        metrics = {}
        if self.worker_df is None or self.worker_df.empty:
            return metrics
        for row in self.worker_df.to_dict("records"):
            uid = str(row.get("user_no", ""))
            if not uid:
                continue
            metrics[uid] = {
                "user_name": str(row.get("user_name", "")),
                "company_name": str(row.get("company_name", "")),
                "ewi": _safe_float(row.get("ewi")),
                "cre": _safe_float(row.get("cre")),
                "fatigue_score": _safe_float(row.get("fatigue_score")),
                "work_minutes": _safe_float(row.get("work_minutes")),
                "work_zone_minutes": _safe_float(row.get("work_zone_minutes")),
                "transition_count": _safe_int(row.get("transition_count")),
                "static_risk": _safe_float(row.get("static_norm", row.get("static_risk"))),
            }
        return metrics

    def _build_prev_cre(self) -> dict[str, float]:
        """전일 worker_df → {user_no: cre}."""
        if self.prev_worker_df is None or self.prev_worker_df.empty:
            return {}
        cre_map = {}
        for row in self.prev_worker_df.to_dict("records"):
            uid = str(row.get("user_no", ""))
            cre_val = _safe_float(row.get("cre"))
            if uid:
                cre_map[uid] = cre_val
        return cre_map

    def _get_locus_name(self, lid: str) -> str:
        return self._locus_meta.get(lid, {}).get("name", lid)

    def _get_locus_capacity(self, lid: str) -> int:
        return self._locus_meta.get(lid, {}).get("capacity", 50)

    def _get_locus_type(self, lid: str) -> str:
        return self._locus_meta.get(lid, {}).get("type", "WORK")

    def _get_hazard_info(self, lid: str) -> tuple[str, int]:
        meta = self._locus_meta.get(lid, {})
        return meta.get("hazard_level", "low"), meta.get("hazard_grade", 1)

    # ─── 심각도 / 분류 헬퍼 ────────────────────────────────

    @staticmethod
    def _build_safety_reasons(
        hazard_grade: int, cre: float, cre_delta: float, fatigue: float,
    ) -> list[str]:
        """안전 경고 사유 문자열 생성."""
        st = SAFETY_THRESHOLDS
        reasons: list[str] = []
        if hazard_grade >= 4:
            reasons.append(f"위험등급 {hazard_grade}등급 구역 진입 예측")
        elif hazard_grade >= 3:
            reasons.append(f"위험등급 {hazard_grade}등급 구역 접근")
        if cre >= st["cre_alert"]:
            reasons.append(f"누적 위험도(CRE) {cre:.2f}")
        if cre_delta >= st["cre_surge"]:
            reasons.append(f"CRE 전일 대비 +{cre_delta:.2f} 급등")
        if fatigue >= st["fatigue_alert"]:
            reasons.append(f"피로도 {fatigue:.2f}")
        return reasons

    @staticmethod
    def _classify_safety_severity(
        risk_score: float, hazard_grade: int, cre: float, fatigue: float,
    ) -> str:
        """안전 위험 심각도 분류."""
        rs = SAFETY_THRESHOLDS["risk_score"]
        if risk_score >= rs["critical"] or (hazard_grade >= 4 and cre >= CRE_HIGH):
            return "critical"
        if risk_score >= rs["high"] or (hazard_grade >= 3 and fatigue >= 0.5):
            return "high"
        if risk_score >= rs["medium"]:
            return "medium"
        return "low"

    @staticmethod
    def _classify_productivity_risk(
        fatigue: float, ewi: float, ewi_drop: float,
        trans_rate: float, avg_trans_rate: float,
        work_zone_pct: float, avg_ewi: float,
        current_risk_type: str,
    ) -> tuple[str, str, str]:
        """생산성 위험 유형/심각도/권고 분류. Returns (risk_type, severity, recommendation)."""
        pt = PRODUCTIVITY_THRESHOLDS
        risk_type = current_risk_type
        severity = "low"
        rec = ""

        # 피로 누적
        if fatigue >= pt["fatigue_alert"] and ewi > 0.2:
            if ewi_drop >= pt["ewi_drop_min"] or fatigue >= pt["fatigue_critical"]:
                risk_type = "fatigue_decline"
                severity = "high" if fatigue >= pt["fatigue_critical"] else "medium"
                rec = "피로 누적으로 EWI 하락 예상. 휴식 시간 확보 또는 경작업 전환 권장."

        # 이동 과다
        if trans_rate > avg_trans_rate * pt["high_movement_mult"] and trans_rate > pt["high_movement_abs"]:
            if risk_type == "fatigue_decline":
                severity = "critical"
                rec = "피로 + 과다이동 복합 위험. 즉시 작업 재배치 및 휴식 필요."
            else:
                risk_type = "high_movement"
                severity = "medium"
                rec = "이동 빈도 과다. 작업 동선 최적화 또는 자재 배치 재검토 권장."

        # 저활용
        if risk_type == "normal" and work_zone_pct < pt["low_util_pct"] and ewi < avg_ewi * pt["low_util_ewi_mult"]:
            risk_type = "low_utilization"
            severity = "medium"
            rec = "작업 공간 체류 비율 낮음. 작업 배정 확인 또는 대기 원인 파악 필요."

        return risk_type, severity, rec

    # ─── 배치 예측 ─────────────────────────────────────────

    def _batch_predict(self, top_k: int = 3) -> dict:
        """전체 작업자 배치 예측 (try/except 안전 처리).

        model=None 시 fallback: 각 작업자의 현재 위치를 다음 위치로 유지.
        이를 통해 model 없이도 current_counts 기반 혼잡도/병목 분석 가능.
        """
        from src.dashboard.deep_space.helpers import predict_next_batch

        worker_ids = list(self.sequences.keys())
        seqs = [self.sequences[wid] for wid in worker_ids]

        current_loci = {wid: seq[-1] for wid, seq in zip(worker_ids, seqs) if seq}

        batch_preds = [[] for _ in seqs]
        if self.model is not None and self.tokenizer is not None:
            try:
                batch_preds = predict_next_batch(
                    self.model, self.tokenizer, seqs, top_k=top_k
                )
            except Exception as e:
                logger.warning(f"Batch prediction failed: {e}")

        # model=None 또는 예측 실패 시 fallback: 현재 위치를 예측값으로 사용
        for i, (wid, preds) in enumerate(zip(worker_ids, batch_preds)):
            if not preds and wid in current_loci:
                batch_preds[i] = [(current_loci[wid], 1.0)]

        predictions = {wid: preds for wid, preds in zip(worker_ids, batch_preds)}

        return {
            "worker_ids": worker_ids,
            "current_loci": current_loci,
            "predictions": predictions,
        }

    # ─── 1. 혼잡도 예측 ───────────────────────────────────

    def predict_congestion(
        self,
        pred_data: dict | None = None,
        threshold: float = 0.7,
    ) -> list[CongestionAlert]:
        """혼잡도 예측. congestion_pct = predicted / capacity."""
        if pred_data is None:
            pred_data = self._batch_predict(top_k=1)

        current_loci = pred_data["current_loci"]
        predictions = pred_data["predictions"]

        current_counts = Counter(current_loci.values())

        # 이동 흐름 집계
        movements: dict[str, dict[str, int]] = {}
        for wid, preds in predictions.items():
            if not preds:
                continue
            current = current_loci.get(wid)
            next_loc = preds[0][0]
            if current and next_loc != current:
                movements.setdefault(current, {"inflow": 0, "outflow": 0})["outflow"] += 1
                movements.setdefault(next_loc, {"inflow": 0, "outflow": 0})["inflow"] += 1

        all_loci = set(current_counts.keys()) | set(movements.keys())
        alerts = []

        for lid in all_loci:
            current = current_counts.get(lid, 0)
            moves = movements.get(lid, {"inflow": 0, "outflow": 0})
            predicted = max(current + moves["inflow"] - moves["outflow"], 0)
            capacity = self._get_locus_capacity(lid)
            pct = predicted / max(capacity, 1)

            if pct < threshold:
                continue

            # 트렌드
            delta = predicted - current
            trend = "급증" if delta >= 5 else "증가" if delta >= 2 else "감소" if delta <= -2 else "유지"

            # 심각도
            ct = CONGESTION_THRESHOLDS
            if pct >= ct["critical"]:
                severity = "critical"
            elif pct >= ct["high"]:
                severity = "high"
            elif pct >= ct["medium"]:
                severity = "medium"
            else:
                severity = "low"

            # 권고사항
            name = self._get_locus_name(lid)
            if severity in ("critical", "high"):
                rec = f"{name}에 인원 분산 필요. 유입원 상위 공간에서 우회 동선 안내 권장."
            elif trend == "급증":
                rec = f"{name} 인원 급증 추세. 사전 공간 확보 또는 진입 시차 조정 권장."
            else:
                rec = ""

            alerts.append(CongestionAlert(
                locus_id=lid, locus_name=name,
                current_count=current, predicted_count=predicted,
                capacity=capacity, congestion_pct=pct, trend=trend,
                severity=severity, inflow=moves["inflow"], outflow=moves["outflow"],
                recommendation=rec,
            ))

        alerts.sort(key=lambda a: a.congestion_pct, reverse=True)
        return alerts

    # ─── 2. 병목 예측 ─────────────────────────────────────

    def predict_bottlenecks(
        self,
        pred_data: dict | None = None,
        threshold: int = 3,
    ) -> list[BottleneckAlert]:
        """병목 예측. net_accumulation = inflow - outflow."""
        if pred_data is None:
            pred_data = self._batch_predict(top_k=1)

        current_loci = pred_data["current_loci"]
        predictions = pred_data["predictions"]

        flow: dict[str, dict] = {}
        source_map: dict[str, Counter] = {}
        target_map: dict[str, Counter] = {}

        for wid, preds in predictions.items():
            if not preds:
                continue
            current = current_loci.get(wid)
            next_loc = preds[0][0]
            if current and next_loc != current:
                flow.setdefault(current, {"inflow": 0, "outflow": 0})["outflow"] += 1
                flow.setdefault(next_loc, {"inflow": 0, "outflow": 0})["inflow"] += 1
                source_map.setdefault(next_loc, Counter())[current] += 1
                target_map.setdefault(current, Counter())[next_loc] += 1

        alerts = []
        current_counts = Counter(current_loci.values())

        for lid, moves in flow.items():
            net = moves["inflow"] - moves["outflow"]
            if net < threshold:
                continue

            capacity = self._get_locus_capacity(lid)
            score = min(net / max(capacity, 1), 1.0)

            bt_s, bt_c = BOTTLENECK_THRESHOLDS["score"], BOTTLENECK_THRESHOLDS["count"]
            if score >= bt_s["critical"] or net >= bt_c["critical"]:
                severity = "critical"
            elif score >= bt_s["high"] or net >= bt_c["high"]:
                severity = "high"
            elif score >= bt_s["medium"] or net >= bt_c["medium"]:
                severity = "medium"
            else:
                severity = "low"

            upstream = source_map.get(lid, Counter()).most_common(5)
            name = self._get_locus_name(lid)
            top_src = ", ".join(self._get_locus_name(s) for s, _ in upstream[:2])
            rec = f"{name} 병목 해소: {top_src}에서 우회 경로 안내 또는 진입 시차 조정." if top_src else ""

            alerts.append(BottleneckAlert(
                locus_id=lid, locus_name=name,
                inflow=moves["inflow"], outflow=moves["outflow"],
                net_accumulation=net, bottleneck_score=score, severity=severity,
                upstream_sources=[(self._get_locus_name(s), c) for s, c in upstream],
                downstream_targets=[(self._get_locus_name(t), c)
                                    for t, c in target_map.get(lid, Counter()).most_common(5)],
                recommendation=rec,
            ))

        alerts.sort(key=lambda a: a.net_accumulation, reverse=True)
        return alerts

    # ─── 3. 안전 위험 예측 (CRE 추세 반영) ─────────────────

    def predict_safety_risks(
        self,
        pred_data: dict | None = None,
    ) -> list[SafetyAlert]:
        """안전 위험 예측. CRE 추세 + 위험구역 진입 확률 + 피로도."""
        if pred_data is None:
            pred_data = self._batch_predict(top_k=3)

        st = SAFETY_THRESHOLDS
        current_loci = pred_data["current_loci"]
        predictions = pred_data["predictions"]
        alerts = []

        for wid, preds in predictions.items():
            if not preds:
                continue

            metrics = self._worker_metrics.get(wid, {})
            cre = metrics.get("cre", 0)
            fatigue = metrics.get("fatigue_score", 0)

            # CRE 추세 (전일 대비)
            prev_cre = self._prev_cre.get(wid, cre)
            cre_delta = cre - prev_cre

            for next_loc, prob in preds:
                if prob < st["prob_min"]:
                    continue

                hazard_level, hazard_grade = self._get_hazard_info(next_loc)
                is_hazardous = (
                    hazard_level in ("critical", "high")
                    or hazard_grade >= 3
                    or self._get_locus_type(next_loc) == "HAZARD"
                )
                if not is_hazardous:
                    continue

                # 위험 점수: 예측 확률 x 위험 등급 x (1 + CRE) x CRE 추세 보정
                cre_boost = 1.0 + max(cre_delta, 0) * st["cre_boost_mult"]
                risk_score = prob * (hazard_grade / 5.0) * (1.0 + cre) * cre_boost

                reasons = self._build_safety_reasons(
                    hazard_grade, cre, cre_delta, fatigue,
                )
                if not reasons:
                    continue

                severity = self._classify_safety_severity(
                    risk_score, hazard_grade, cre, fatigue,
                )

                locus_name = self._get_locus_name(next_loc)
                rec = f"{metrics.get('user_name', wid)} 작업자를 {locus_name} 대신 인근 저위험 구역으로 재배치 권장."

                alerts.append(SafetyAlert(
                    user_no=wid,
                    user_name=metrics.get("user_name", wid),
                    company_name=metrics.get("company_name", ""),
                    current_locus=current_loci.get(wid, ""),
                    predicted_locus=next_loc,
                    predicted_locus_name=locus_name,
                    prediction_prob=prob,
                    hazard_level=hazard_level, hazard_grade=hazard_grade,
                    cre=cre, fatigue_score=fatigue,
                    risk_score=risk_score, severity=severity,
                    reason=" + ".join(reasons),
                    cre_delta=cre_delta,
                    recommendation=rec,
                ))

        # 중복 제거 (같은 작업자 → 가장 높은 risk만)
        seen = {}
        for a in alerts:
            if a.user_no not in seen or a.risk_score > seen[a.user_no].risk_score:
                seen[a.user_no] = a
        alerts = list(seen.values())
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return alerts

    # ─── 4. 생산성 예측 (업체 집계 포함) ───────────────────

    def predict_productivity(
        self,
        pred_data: dict | None = None,
        fatigue_decay: float = 0.15,
    ) -> tuple[list[ProductivityAlert], list[CompanyProductivitySummary]]:
        """생산성 예측 + 업체 단위 집계."""
        if not self._worker_metrics:
            return [], []

        if pred_data is None:
            pred_data = self._batch_predict(top_k=1)

        # 전체 평균 (baseline)
        rates = [m["transition_count"] / max(m["work_minutes"], 1)
                 for m in self._worker_metrics.values() if m["work_minutes"] > 0]
        avg_trans_rate = np.mean(rates) if rates else 0.02

        all_ewi = [m["ewi"] for m in self._worker_metrics.values() if m["ewi"] > 0]
        avg_ewi = np.mean(all_ewi) if all_ewi else 0.3

        alerts = []
        company_alerts: dict[str, list[ProductivityAlert]] = {}

        for wid in self.sequences:
            metrics = self._worker_metrics.get(wid)
            if not metrics or metrics["work_minutes"] <= 0:
                continue

            ewi = metrics["ewi"]
            fatigue = metrics["fatigue_score"]
            work_min = metrics["work_minutes"]
            work_zone_min = metrics["work_zone_minutes"]
            trans_count = metrics["transition_count"]

            predicted_ewi = ewi * (1.0 - fatigue_decay * fatigue)
            work_zone_pct = work_zone_min / max(work_min, 1)
            trans_rate = trans_count / max(work_min, 1)
            ewi_drop = ewi - predicted_ewi

            risk_type, severity, rec = self._classify_productivity_risk(
                fatigue, ewi, ewi_drop, trans_rate, avg_trans_rate,
                work_zone_pct, avg_ewi, "normal",
            )
            if risk_type == "normal":
                continue

            if predicted_ewi < ewi * 0.85:
                ewi_trend = "하락 예상"
            elif predicted_ewi > ewi * 1.05:
                ewi_trend = "상승 예상"
            else:
                ewi_trend = "유지"

            alert = ProductivityAlert(
                user_no=wid,
                user_name=metrics.get("user_name", wid),
                company_name=metrics.get("company_name", ""),
                current_ewi=ewi, predicted_ewi=predicted_ewi, ewi_trend=ewi_trend,
                work_zone_pct=work_zone_pct, transition_rate=trans_rate,
                fatigue_score=fatigue, risk_type=risk_type, severity=severity,
                recommendation=rec,
            )
            alerts.append(alert)

            # 업체별 집계
            comp = metrics.get("company_name", "")
            if comp:
                company_alerts.setdefault(comp, []).append(alert)

        alerts.sort(key=lambda a: (
            SEVERITY_ORDER.get(a.severity, 0),
            -a.predicted_ewi,
        ), reverse=True)

        # 업체 단위 요약
        company_summaries = self._build_company_summaries(company_alerts)

        return alerts, company_summaries

    def _build_company_summaries(
        self,
        company_alerts: dict[str, list[ProductivityAlert]],
    ) -> list[CompanyProductivitySummary]:
        """업체별 생산성 경고 집계."""
        summaries = []

        # 업체별 전체 작업자 수
        company_workers: Counter = Counter()
        for m in self._worker_metrics.values():
            comp = m.get("company_name", "")
            if comp:
                company_workers[comp] += 1

        for comp, comp_alerts in company_alerts.items():
            total = company_workers.get(comp, len(comp_alerts))
            alert_count = len(comp_alerts)
            alert_pct = alert_count / max(total, 1)

            if alert_pct < 0.2:  # 20% 미만이면 스킵
                continue

            avg_ewi = np.mean([a.current_ewi for a in comp_alerts]) if comp_alerts else 0
            avg_fatigue = np.mean([a.fatigue_score for a in comp_alerts]) if comp_alerts else 0

            # 가장 많은 risk_type
            risk_counts = Counter(a.risk_type for a in comp_alerts)
            dominant_risk = risk_counts.most_common(1)[0][0] if risk_counts else "unknown"

            severity = "critical" if alert_pct >= 0.5 else "high" if alert_pct >= 0.35 else "medium"
            rec = f"{comp} 작업자 {alert_pct:.0%}가 생산성 경고. 업체 단위 피로 관리 및 작업 재배치 필요."

            summaries.append(CompanyProductivitySummary(
                company_name=comp, worker_count=total,
                alert_count=alert_count, alert_pct=alert_pct,
                avg_ewi=avg_ewi, avg_fatigue=avg_fatigue,
                dominant_risk=dominant_risk, severity=severity,
                recommendation=rec,
            ))

        summaries.sort(key=lambda s: s.alert_pct, reverse=True)
        return summaries

    # ─── 종합 리포트 ──────────────────────────────────────

    def generate_report(self, date_str: str = "", sector_id: str = "") -> AgenticReport:
        """전체 예측 수행 → AgenticReport 반환."""
        pred_data = self._batch_predict(top_k=3)

        congestion = self.predict_congestion(pred_data)
        bottlenecks = self.predict_bottlenecks(pred_data)
        safety = self.predict_safety_risks(pred_data)
        productivity_alerts, company_summaries = self.predict_productivity(pred_data)

        all_severities = (
            [a.severity for a in congestion]
            + [a.severity for a in bottlenecks]
            + [a.severity for a in safety]
            + [a.severity for a in productivity_alerts]
        )

        return AgenticReport(
            date=date_str,
            sector_id=sector_id,
            total_workers=len(self.sequences),
            congestion_alerts=congestion,
            bottleneck_alerts=bottlenecks,
            safety_alerts=safety,
            productivity_alerts=productivity_alerts,
            company_summaries=company_summaries,
            critical_count=sum(1 for s in all_severities if s == "critical"),
            high_count=sum(1 for s in all_severities if s == "high"),
        )


# ─── M1: AgenticReport → InsightReport 브릿지 ───────────────

def agentic_to_insights(report: AgenticReport) -> list:
    """
    AgenticReport의 경고들을 Insight 객체 리스트로 변환.

    기존 InsightReport 파이프라인(anomaly/journey/trend)에
    source="agentic" 인사이트를 추가하여 통합 뷰를 제공한다.
    """
    from src.intelligence.models import Insight, Severity
    from src.utils.anonymizer import mask_name

    insights: list[Insight] = []

    # 1. 혼잡도 → space 카테고리
    for a in report.congestion_alerts:
        if a.severity not in ("critical", "high"):
            continue
        insights.append(Insight(
            category="space",
            severity=Severity(SEVERITY_ORDER[a.severity]),
            title=f"{a.locus_name} 혼잡도 {a.congestion_pct:.0%} ({a.trend})",
            description=(
                f"현재 {a.current_count}명 → 예측 {a.predicted_count}명 "
                f"(수용력 {a.capacity}명). 유입 {a.inflow} / 유출 {a.outflow}."
            ),
            evidence={
                "congestion_pct": round(a.congestion_pct, 3),
                "current": a.current_count,
                "predicted": a.predicted_count,
                "capacity": a.capacity,
            },
            affected=[a.locus_name],
            recommendation=a.recommendation,
            source="agentic",
        ))

    # 2. 병목 → space 카테고리
    for a in report.bottleneck_alerts:
        if a.severity not in ("critical", "high"):
            continue
        sources_str = ", ".join(s for s, _ in a.upstream_sources[:3])
        insights.append(Insight(
            category="space",
            severity=Severity(SEVERITY_ORDER[a.severity]),
            title=f"{a.locus_name} 병목 (순유입 +{a.net_accumulation}명)",
            description=(
                f"유입 {a.inflow} / 유출 {a.outflow}. "
                f"주요 유입원: {sources_str or '불명'}."
            ),
            evidence={
                "net_accumulation": a.net_accumulation,
                "bottleneck_score": round(a.bottleneck_score, 3),
            },
            affected=[a.locus_name],
            recommendation=a.recommendation,
            source="agentic",
        ))

    # 3. 안전 → safety 카테고리
    for a in report.safety_alerts:
        if a.severity not in ("critical", "high"):
            continue
        masked = mask_name(a.user_name) if a.user_name else a.user_no
        insights.append(Insight(
            category="safety",
            severity=Severity(SEVERITY_ORDER[a.severity]),
            title=f"{masked} 위험구역 진입 예측 ({a.predicted_locus_name})",
            description=a.reason,
            evidence={
                "risk_score": round(a.risk_score, 3),
                "cre": round(a.cre, 3),
                "fatigue": round(a.fatigue_score, 3),
                "hazard_grade": a.hazard_grade,
                "cre_delta": round(a.cre_delta, 3),
            },
            affected=[masked, a.predicted_locus_name],
            recommendation=a.recommendation,
            source="agentic",
        ))

    # 4. 생산성 → productivity 카테고리
    for a in report.productivity_alerts:
        if a.severity not in ("critical", "high"):
            continue
        masked = mask_name(a.user_name) if a.user_name else a.user_no
        type_label = {
            "fatigue_decline": "피로 누적",
            "high_movement": "과다 이동",
            "low_utilization": "저활용",
        }.get(a.risk_type, a.risk_type)
        insights.append(Insight(
            category="productivity",
            severity=Severity(SEVERITY_ORDER[a.severity]),
            title=f"{masked} 생산성 하락 예측 ({type_label})",
            description=(
                f"현재 EWI {a.current_ewi:.2f} → 예측 {a.predicted_ewi:.2f}. "
                f"피로도 {a.fatigue_score:.2f}, 작업공간 {a.work_zone_pct:.0%}."
            ),
            evidence={
                "current_ewi": round(a.current_ewi, 3),
                "predicted_ewi": round(a.predicted_ewi, 3),
                "fatigue": round(a.fatigue_score, 3),
                "risk_type": a.risk_type,
            },
            affected=[masked, a.company_name],
            recommendation=a.recommendation,
            source="agentic",
        ))

    # 5. 업체 단위 → productivity 카테고리
    for c in report.company_summaries:
        insights.append(Insight(
            category="productivity",
            severity=Severity(SEVERITY_ORDER.get(c.severity, 2)),
            title=f"{c.company_name} 작업자 {c.alert_pct:.0%} 생산성 경고",
            description=(
                f"전체 {c.worker_count}명 중 {c.alert_count}명 경고. "
                f"평균 EWI {c.avg_ewi:.2f}, 평균 피로도 {c.avg_fatigue:.2f}."
            ),
            evidence={
                "alert_pct": round(c.alert_pct, 3),
                "avg_ewi": round(c.avg_ewi, 3),
                "dominant_risk": c.dominant_risk,
            },
            affected=[c.company_name],
            recommendation=c.recommendation,
            source="agentic",
        ))

    return insights
