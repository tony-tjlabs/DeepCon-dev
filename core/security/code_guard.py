"""
CodeGuard — 핵심 로직 보호
============================
핵심 알고리즘(EWI/CRE 공식, Journey 보정 로직 등)이
대시보드 UI나 LLM 응답에 노출되지 않도록 보호한다.

보안 원칙:
  1. theory_tab에는 일반 설명만 표시 (SAFE_DESCRIPTIONS)
  2. Release 빌드에서 핵심 로직 파일 포함 여부 검증
  3. .gitignore에 시크릿 파일 목록 관리
  4. 배포 전 자동 보안 스캔

사용:
    from core.security.code_guard import CodeGuard

    desc = CodeGuard.get_safe_description("ewi")
    is_safe = CodeGuard.is_deployable(Path("config.py"))
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeGuard:
    """핵심 로직 보호 게이트.

    대시보드 UI(특히 방법론/이론 탭)에 노출 가능한 설명과
    배포 가능 파일을 관리한다.
    """

    # 외부 노출 안전한 일반 설명 (theory_tab 등에서 사용)
    SAFE_DESCRIPTIONS: dict[str, str] = {
        "ewi": "BLE 신호의 활성 비율을 기반으로 작업 강도를 측정합니다.",
        "cre": "공간 위험도와 체류 패턴을 종합하여 위험 노출을 평가합니다.",
        "sii": "단독 작업 상황을 공간 정보와 결합하여 안전 위험을 감지합니다.",
        "deep_space": "이동 시퀀스를 토큰화하여 Foundation Model로 학습합니다.",
        "journey_correction": "BLE 신호의 노이즈를 통계적 방법으로 보정합니다.",
        "locus_tokenization": "물리적 공간을 의미 단위로 분류하여 토큰으로 변환합니다.",
        "anomaly_detection": "이동 패턴의 통계적 이상을 감지하여 경보를 생성합니다.",
        "congestion": "공간별 동시 체류 인원을 분석하여 혼잡도를 평가합니다.",
        "embedding": "이동 경로를 벡터 공간에 매핑하여 패턴 유사도를 계산합니다.",
        "shift_detection": "출퇴근 시간과 활동 패턴으로 근무 시프트를 자동 감지합니다.",
    }

    # 배포에 포함하면 안 되는 파일 패턴
    _BLOCKED_FILE_PATTERNS: set[str] = {
        ".env",
        "service_account_key.json",
        "client_secret.json",
        "token.json",
        "drive_config.json",
        "*_secret*",
        "*_key.json",
        "*.pem",
        "*.p12",
    }

    # 배포에 포함하면 안 되는 디렉토리
    _BLOCKED_DIRS: set[str] = {
        "__pycache__",
        ".git",
        ".env",
        "raw",  # Raw 데이터는 배포 불가
    }

    # 핵심 로직 파일 (Release 빌드에서 보호 검증 대상)
    _CORE_LOGIC_FILES: set[str] = {
        "metrics.py",          # EWI/CRE 공식
        "corrector.py",        # Journey 보정 알고리즘
        "transformer.py",      # Deep Space 모델 아키텍처
        "anomaly_detector.py", # 이상 탐지 알고리즘
        "congestion.py",       # 혼잡도 분석 알고리즘
    }

    @classmethod
    def get_safe_description(cls, metric: str) -> str:
        """외부 노출 안전한 일반 설명을 반환한다.

        theory_tab.py 등 사용자 대면 UI에서 알고리즘 설명 시 사용.
        핵심 공식이나 파라미터가 포함되지 않은 일반적인 설명만 반환.

        Args:
            metric: 지표/알고리즘 키 (ewi, cre, deep_space 등)

        Returns:
            안전한 설명 문자열 (없으면 기본 메시지)
        """
        return cls.SAFE_DESCRIPTIONS.get(
            metric,
            "상세 알고리즘 설명은 보안 정책에 따라 제공되지 않습니다.",
        )

    @classmethod
    def is_deployable(cls, file_path: Path) -> bool:
        """배포 가능 파일인지 검사한다.

        시크릿 파일, 핵심 알고리즘 소스는 배포 불가.

        Args:
            file_path: 검사할 파일 경로

        Returns:
            배포 가능 여부
        """
        file_name = file_path.name.lower()

        # 차단 파일 패턴 체크
        for pattern in cls._BLOCKED_FILE_PATTERNS:
            if pattern.startswith("*"):
                if file_name.endswith(pattern[1:]):
                    return False
            elif file_name == pattern.lower():
                return False

        # 차단 디렉토리 체크
        for part in file_path.parts:
            if part.lower() in cls._BLOCKED_DIRS:
                return False

        return True

    @classmethod
    def scan_directory(cls, directory: Path) -> dict[str, list[str]]:
        """디렉토리를 스캔하여 보안 이슈를 보고한다.

        Args:
            directory: 스캔할 디렉토리

        Returns:
            {"blocked_files": [...], "secret_files": [...],
             "core_logic_exposed": [...]}
        """
        result: dict[str, list[str]] = {
            "blocked_files": [],
            "secret_files": [],
            "core_logic_exposed": [],
        }

        if not directory.exists():
            return result

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            rel_path = str(file_path.relative_to(directory))

            # 배포 불가 파일
            if not cls.is_deployable(file_path):
                result["blocked_files"].append(rel_path)

            # 시크릿 파일 탐지
            if cls._is_secret_file(file_path):
                result["secret_files"].append(rel_path)

            # 핵심 로직 노출 여부
            if file_path.name in cls._CORE_LOGIC_FILES:
                result["core_logic_exposed"].append(rel_path)

        return result

    @classmethod
    def validate_gitignore(cls, project_dir: Path) -> dict[str, bool]:
        """프로젝트 .gitignore에 필수 항목이 포함되어 있는지 검증한다.

        Args:
            project_dir: 프로젝트 루트 디렉토리

        Returns:
            {패턴: 포함여부} 딕셔너리
        """
        gitignore_path = project_dir / ".gitignore"
        if not gitignore_path.exists():
            return {p: False for p in cls._BLOCKED_FILE_PATTERNS}

        with open(gitignore_path, encoding="utf-8") as f:
            gitignore_content = f.read().lower()

        results: dict[str, bool] = {}
        for pattern in sorted(cls._BLOCKED_FILE_PATTERNS):
            # 간단한 포함 여부 체크 (완벽한 glob 매칭은 아님)
            check_pattern = pattern.replace("*", "").lower()
            results[pattern] = check_pattern in gitignore_content

        return results

    @classmethod
    def get_deployment_checklist(cls) -> list[dict[str, str]]:
        """배포 전 보안 체크리스트를 반환한다.

        Returns:
            [{"item": str, "description": str, "severity": str}, ...]
        """
        return [
            {
                "item": "시크릿 파일 제외",
                "description": ".env, *_key.json, *_secret.json이 .gitignore에 포함",
                "severity": "Critical",
            },
            {
                "item": "API 키 환경변수화",
                "description": "ANTHROPIC_API_KEY 등이 st.secrets 또는 환경변수로만 로드",
                "severity": "Critical",
            },
            {
                "item": "LLM 데이터 익명화",
                "description": "DataGuard.sanitize_for_llm()이 모든 LLM 호출 체인에 적용",
                "severity": "Critical",
            },
            {
                "item": "좌표 제거 확인",
                "description": "LLM 전송 데이터에 X,Y 좌표가 포함되지 않음",
                "severity": "High",
            },
            {
                "item": "핵심 로직 보호",
                "description": "theory_tab에 EWI/CRE 공식이 노출되지 않음 (SAFE_DESCRIPTIONS 사용)",
                "severity": "High",
            },
            {
                "item": "Raw 데이터 제외",
                "description": "data/raw/ 디렉토리가 배포에 포함되지 않음",
                "severity": "Medium",
            },
            {
                "item": "CLOUD_MODE 검증",
                "description": "CLOUD_MODE=True 시 파이프라인 탭 숨김, 캐시만 사용",
                "severity": "Medium",
            },
            {
                "item": "감사 로그 확인",
                "description": "DataGuard 감사 로그가 활성화되어 있음",
                "severity": "Low",
            },
        ]

    # ─── 내부 헬퍼 ──────────────────────────────────────────────────

    @staticmethod
    def _is_secret_file(file_path: Path) -> bool:
        """시크릿 파일 여부를 판단한다."""
        name_lower = file_path.name.lower()
        secret_indicators = [
            "secret", "key.json", "token.json", "credential",
            ".pem", ".p12", ".pfx", "password",
        ]
        return any(ind in name_lower for ind in secret_indicators)
