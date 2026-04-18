"""
core.ai.audit_log — 영속 LLM 감사 로그 (M2-A T-12)
====================================================
모든 LLM 호출을 append-only JSONL 로 영속 기록.
설계 참조: upgrade_v3_03_security.md §3.C-4 (V-02 CRITICAL)

저장 포맷:
  data/audit/{sector_id}/YYYY-MM.jsonl
  1줄 1호출 (JSON)

보안 원칙:
  1. **원문 저장 금지** — prompt / response 는 SHA256 해시만.
  2. 월 1회 자동 로테이션 (달이 바뀌면 새 파일).
  3. `.gitignore` 에 `data/audit/` 반드시 포함.
  4. 파일 권한은 OS 기본 — 추후 chmod 0600 추가 검토 (Phase 2).

스키마(고정):
{
  "timestamp":        "2026-04-18T19:23:45Z",       # ISO8601 UTC
  "request_id":       "uuid4",
  "user_role":        "Y1_SKHynix" | "administrator",
  "sector_id":        "Y1_SKHynix",
  "tab":              "overview" | "zone_time" | ...,
  "role":             "overview_commentator",
  "date_str":         "20260409" | null,
  "model":            "claude-haiku-4-5",
  "tokens_in":        1234,
  "tokens_out":       567,
  "cache_read_tokens":  0,
  "cache_write_tokens": 0,
  "latency_ms":       892,
  "prompt_hash":      "sha256:...",
  "response_hash":    "sha256:...",
  "pii_leak_detected":  false,
  "pii_warnings":     [...],    # 최대 10건, kind 만
  "anonymization_version": "1.0",
  "blocked":          false,
  "error":            null | "..."
}
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 현재 익명화 규칙 버전 (core/security/anonymizer/core.py 와 연동).
ANONYMIZATION_VERSION = "1.0"

# 한 쓰기당 lock (append race 조건 방어)
_WRITE_LOCK = threading.Lock()


# ─── 엔트리 dataclass ──────────────────────────────────────
@dataclass
class AuditEntry:
    timestamp: str
    request_id: str
    user_role: str
    sector_id: str
    tab: str
    role: str
    date_str: str | None
    model: str
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int
    prompt_hash: str
    response_hash: str
    pii_leak_detected: bool
    pii_warnings: list[dict] = field(default_factory=list)
    anonymization_version: str = ANONYMIZATION_VERSION
    blocked: bool = False
    error: str | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ─── 해시 유틸 ───────────────────────────────────────────
def _sha256_prefix(text: str, prefix: str = "sha256:") -> str:
    if text is None:
        return prefix + "null"
    h = hashlib.sha256(str(text).encode("utf-8", errors="replace")).hexdigest()
    return f"{prefix}{h}"


def _utcnow_iso() -> str:
    # "2026-04-18T19:23:45Z" 형식 (microseconds 제거)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _yyyymm(ts_iso: str) -> str:
    # "2026-04-18T19:23:45Z" → "2026-04"
    return ts_iso[:7]


# ─── Logger 본체 ─────────────────────────────────────────
class AuditLogger:
    """
    영속 감사 로그 게이트.

    사용:
        audit = AuditLogger(base_dir=Path("data/audit"))
        audit.log(
            user_role="administrator", sector_id="Y1_SKHynix",
            tab="overview", role="overview_commentator",
            date_str="20260409", model="claude-haiku-4-5",
            prompt="...", response="...",
            tokens_in=1234, tokens_out=567,
            cache_read_tokens=0, cache_write_tokens=0,
            latency_ms=892, pii_warnings=[],
        )
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        # 디렉토리는 첫 쓰기에서 생성 (인스턴스 생성은 부수효과 최소화)

    # ─── 파일 경로 ──────────────────────────────
    def _path_for(self, sector_id: str, ts_iso: str) -> Path:
        ym = _yyyymm(ts_iso)
        return self.base_dir / sector_id / f"{ym}.jsonl"

    # ─── 쓰기 ──────────────────────────────────
    def log(
        self,
        *,
        user_role: str,
        sector_id: str,
        tab: str,
        role: str,
        date_str: str | None,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        latency_ms: int = 0,
        pii_warnings: list[dict] | None = None,
        blocked: bool = False,
        error: str | None = None,
        request_id: str | None = None,
    ) -> AuditEntry:
        """
        1줄 append. 실패해도 호출자에 예외를 던지지 않는다(감사 실패로 본 기능 중단 방지).
        """
        entry = AuditEntry(
            timestamp=_utcnow_iso(),
            request_id=request_id or str(uuid.uuid4()),
            user_role=user_role or "unknown",
            sector_id=sector_id or "unknown",
            tab=tab or "unknown",
            role=role or "unknown",
            date_str=date_str,
            model=model or "unknown",
            tokens_in=int(tokens_in or 0),
            tokens_out=int(tokens_out or 0),
            cache_read_tokens=int(cache_read_tokens or 0),
            cache_write_tokens=int(cache_write_tokens or 0),
            latency_ms=int(latency_ms or 0),
            prompt_hash=_sha256_prefix(prompt or ""),
            response_hash=_sha256_prefix(response or ""),
            pii_leak_detected=bool(pii_warnings),
            # 최대 10건만, kind만 노출(원문 저장 금지)
            pii_warnings=[
                {"kind": w.get("kind"), "span": w.get("span")}
                for w in (pii_warnings or [])[:10]
            ],
            anonymization_version=ANONYMIZATION_VERSION,
            blocked=bool(blocked),
            error=error,
        )

        try:
            path = self._path_for(entry.sector_id, entry.timestamp)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK:
                with path.open("a", encoding="utf-8") as f:
                    f.write(entry.to_json_line() + "\n")
        except Exception as e:  # pragma: no cover (감사 실패는 로그만)
            log.warning("[audit_log] write failed: %s", e)

        return entry


# ─── 읽기 유틸 (admin 탭용) ─────────────────────────────
def read_audit_log(
    sector_id: str,
    yyyymm: str,
    *,
    base_dir: Path | str = "data/audit",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    월별 감사 로그를 읽어 dict 리스트로 반환.

    Args:
        sector_id: "Y1_SKHynix" 등
        yyyymm:    "2026-04"
        base_dir:  감사 로그 루트
        limit:     최근 N건만 (None = 전체)

    Returns:
        오래된 순 정렬 dict 리스트. 파일 없으면 빈 리스트.
    """
    path = Path(base_dir) / sector_id / f"{yyyymm}.jsonl"
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("[audit_log.read] skip malformed line: %s", e)
                continue

    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def list_audit_months(
    sector_id: str | None = None,
    *,
    base_dir: Path | str = "data/audit",
) -> list[tuple[str, str]]:
    """
    사용 가능한 (sector, yyyymm) 목록 반환.

    Returns:
        [("Y1_SKHynix", "2026-04"), ...] — sector / yyyymm 순 정렬.
    """
    root = Path(base_dir)
    if not root.exists():
        return []
    out: list[tuple[str, str]] = []
    sector_dirs = [root / sector_id] if sector_id else sorted(root.iterdir())
    for d in sector_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.jsonl")):
            out.append((d.name, f.stem))
    return out


__all__ = [
    "ANONYMIZATION_VERSION",
    "AuditEntry",
    "AuditLogger",
    "read_audit_log",
    "list_audit_months",
]
