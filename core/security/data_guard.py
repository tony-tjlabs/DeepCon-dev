"""
DataGuard — LLM 전송 전 데이터 보안 게이트
=============================================
LLM에 전송되는 모든 데이터에서 민감 정보를 제거한다.

보안 원칙:
  1. 실제 좌표(X,Y)는 LLM에 절대 전달하지 않는다
  2. 작업자 실명, 회사명, 사원번호는 마스킹한다
  3. Locus ID는 일반화된 설명으로 변환한다
  4. 핵심 알고리즘 파라미터는 추상화한다
  5. 모든 LLM 전송에 감사 로그를 남긴다

사용:
    from core.security.data_guard import DataGuard

    guard = DataGuard()
    safe_data = guard.sanitize_for_llm({"worker_name": "홍길동", "ewi": 0.65})
    safe_context = guard.sanitize_locus_context(registry, ["L-Y1-020"])
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataGuard:
    """LLM 전송 전 데이터 보안 게이트.

    모든 LLM 호출 체인에서 이 클래스를 거쳐야 한다:
      사용자 요청 → DataGuard.sanitize → LLM → 응답 (원본 데이터 미포함)

    Attributes:
        audit_enabled: 감사 로그 활성화 여부
    """

    # LLM에 절대 전달하면 안 되는 필드명
    BLOCKED_FIELDS: set[str] = {
        "real_name", "phone", "employee_id", "company_name",
        "user_name", "worker_name", "company_no", "user_no",
        "twardid", "mac_address", "ip_address", "email",
        "coordinates", "x", "y", "xy",
        "password", "api_key", "secret", "token",
        "service_account", "client_secret",
    }

    # 좌표 관련 키 패턴 (x, y, lat, lon, coord 등)
    _COORD_PATTERN = re.compile(
        r"(^x$|^y$|^lat|^lon|^coord|^position|_x$|_y$|_lat$|_lon$)",
        re.IGNORECASE,
    )

    # 이름 패턴 (한글 2~4자, 영문 이름)
    _NAME_PATTERN = re.compile(r"^[가-힣]{2,4}$|^[A-Z][a-z]+ [A-Z][a-z]+$")

    def __init__(self, audit_enabled: bool = True) -> None:
        """DataGuard를 초기화한다.

        Args:
            audit_enabled: 감사 로그 활성화 여부
        """
        self.audit_enabled = audit_enabled
        self._audit_log: list[dict] = []

    # ─── 핵심 보안 메서드 ──────────────────────────────────────────

    def sanitize_for_llm(self, data: dict[str, Any]) -> dict[str, Any]:
        """LLM에 전송할 데이터에서 민감 정보를 제거한다.

        차단 대상:
          - BLOCKED_FIELDS에 해당하는 키
          - 좌표 관련 키 (x, y, lat, lon 등)
          - 이름으로 보이는 문자열 값 (한글 2~4자)

        Args:
            data: 원본 데이터 딕셔너리

        Returns:
            민감 정보가 제거된 안전한 딕셔너리
        """
        safe_data: dict[str, Any] = {}

        for key, value in data.items():
            key_lower = key.lower()

            # 차단 필드 체크
            if key_lower in self.BLOCKED_FIELDS:
                safe_data[key] = "[REDACTED]"
                continue

            # 좌표 패턴 체크
            if self._COORD_PATTERN.search(key_lower):
                safe_data[key] = "[COORDINATES_REMOVED]"
                continue

            # 문자열 값에서 이름 패턴 체크
            if isinstance(value, str) and self._NAME_PATTERN.match(value):
                safe_data[key] = self._mask_name(value)
                continue

            # 중첩 딕셔너리 재귀 처리
            if isinstance(value, dict):
                safe_data[key] = self.sanitize_for_llm(value)
                continue

            # 리스트 내 딕셔너리 처리
            if isinstance(value, list):
                safe_data[key] = [
                    self.sanitize_for_llm(item) if isinstance(item, dict) else item
                    for item in value
                ]
                continue

            safe_data[key] = value

        if self.audit_enabled:
            removed_keys = [k for k in data if k not in safe_data or safe_data.get(k) in ("[REDACTED]", "[COORDINATES_REMOVED]")]
            if removed_keys:
                self.audit_log("sanitize_for_llm", f"제거된 필드: {removed_keys}")

        return safe_data

    def sanitize_locus_context(
        self,
        registry: Any,
        locus_ids: list[str],
    ) -> str:
        """Locus 정보를 LLM용으로 안전하게 변환한다.

        좌표(X,Y)는 제거하고 일반적인 공간 설명만 전달한다.
        실제 Locus ID 대신 "구역 A", "구역 B" 형태로 일반화한다.

        Args:
            registry: LocusRegistry 인스턴스
            locus_ids: 변환할 Locus ID 목록

        Returns:
            LLM용 안전한 공간 컨텍스트 문자열
        """
        lines: list[str] = []
        label_map: dict[str, str] = {}

        for idx, locus_id in enumerate(locus_ids):
            # Locus ID를 일반화된 라벨로 변환
            label = f"구역_{chr(65 + idx % 26)}"
            if idx >= 26:
                label = f"구역_{chr(65 + idx // 26 - 1)}{chr(65 + idx % 26)}"
            label_map[locus_id] = label

            locus = registry.get(locus_id) if registry else None
            if locus is None:
                lines.append(f"- {label}: 알 수 없는 구역")
                continue

            # 안전한 속성만 포함 (좌표 제외)
            desc_parts = [f"- {label}: {locus.locus_type.value} 구역"]

            if locus.building:
                desc_parts.append(f"건물={locus.building}")
            if locus.floor:
                desc_parts.append(f"층={locus.floor}")
            if locus.hazard_grade is not None and locus.hazard_grade > 0:
                desc_parts.append(f"위험등급={int(locus.hazard_grade)}")
            if locus.is_confined_space:
                desc_parts.append("밀폐공간")
            if locus.requires_team:
                desc_parts.append("2인이상필수")

            dwell_names = {
                "TRANSIT": "통과지점",
                "SHORT_STAY": "단시간체류",
                "LONG_STAY": "장시간체류",
                "HAZARD_ZONE": "위험구역",
                "ADMIN": "사무구역",
            }
            dwell_name = dwell_names.get(locus.dwell_category.value, "")
            if dwell_name:
                desc_parts.append(dwell_name)

            lines.append(", ".join(desc_parts))

        if self.audit_enabled:
            self.audit_log(
                "sanitize_locus_context",
                f"{len(locus_ids)}개 Locus 컨텍스트 생성 (ID 일반화 적용)",
            )

        return "\n".join(lines)

    def sanitize_journey(
        self,
        journey: list[str],
        registry: Any,
    ) -> str:
        """Journey 시퀀스를 LLM용으로 변환한다.

        Locus ID 대신 일반화된 공간 설명으로 변환한다.
        예: ["L-Y1-001", "L-Y1-070"] → "출입구 → 야외작업구역"

        Args:
            journey: Locus ID 시퀀스
            registry: LocusRegistry 인스턴스

        Returns:
            LLM용 안전한 Journey 설명 문자열
        """
        descriptions: list[str] = []

        for locus_id in journey:
            locus = registry.get(locus_id) if registry else None
            if locus is None:
                descriptions.append("미확인구역")
                continue

            # 타입 기반 일반 설명 (구체적 위치명 숨김)
            type_descriptions = {
                "GATE": "출입구",
                "WORK": "작업구역",
                "REST": "휴게구역",
                "FACILITY": "편의시설",
                "ADMIN": "사무구역",
                "TRANSPORT": "이동설비",
                "HAZARD": "위험구역",
            }
            desc = type_descriptions.get(locus.locus_type.value, "기타구역")

            # 건물 정보는 유지 (위치 특정 어려움)
            if locus.building:
                desc = f"{locus.building} {desc}"
            if locus.floor:
                desc = f"{desc}({locus.floor})"

            descriptions.append(desc)

        result = " -> ".join(descriptions)

        if self.audit_enabled:
            self.audit_log(
                "sanitize_journey",
                f"{len(journey)}개 이동 시퀀스 변환 (Locus ID 제거)",
            )

        return result

    # ─── 감사 로그 ──────────────────────────────────────────────────

    def audit_log(self, action: str, data_summary: str) -> None:
        """LLM 전송 감사 로그를 기록한다.

        Args:
            action: 수행된 보안 작업 (sanitize_for_llm 등)
            data_summary: 처리 요약
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "summary": data_summary,
        }
        self._audit_log.append(entry)
        logger.info("DataGuard 감사: [%s] %s", action, data_summary)

    def get_audit_log(self) -> list[dict]:
        """감사 로그를 반환한다.

        Returns:
            감사 로그 엔트리 리스트
        """
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        """감사 로그를 초기화한다."""
        self._audit_log.clear()

    # ─── 내부 헬퍼 ──────────────────────────────────────────────────

    @staticmethod
    def _mask_name(name: str) -> str:
        """이름을 마스킹한다 (성만 남김).

        예: "홍길동" → "홍**", "Kim" → "K**"

        Args:
            name: 원본 이름

        Returns:
            마스킹된 이름
        """
        if not name:
            return name
        return f"{name[0]}**"
