"""
core.security.anonymizer.core — PII 마스킹 규칙
================================================
단일 소스 오브 트루스. 다른 곳에서 자체 마스킹 구현 금지.

규칙 (upgrade_v3_03_security.md §3.C-2 기준):

| 필드          | 마스킹                                    | 예시                  |
|---------------|-------------------------------------------|-----------------------|
| user_name     | 성(첫 글자) + ** + 사번 앞 5자 해시 alias | 홍길동 → 홍** (A-7f3e1) |
|               | 사번 미지정 시 성 + ** 만 사용            | 홍길동 → 홍**         |
| user_no       | sha256 앞 5자 (일관성 유지)               | 32763  → a3f8e        |
| twardid       | T-XX-**** (마지막 4자리 마스킹)           | T-41-12345678 → T-41-**** |
| company_name  | 유지 (사업 맥락 필수, 경쟁 민감도 존재)   | 유지                   |
| 자유텍스트    | 위 규칙 일괄 적용                          | —                     |

설계 원칙:
  - 순수 함수. side-effect 없음.
  - NaN/None/빈 문자열은 안전하게 패스스루.
  - 동일 입력 → 동일 출력 (재현 가능).
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


# ─── 단일 이름 마스킹 ─────────────────────────────────────────
def mask_name(name, user_no: str | int | None = None) -> str:
    """작업자 이름을 비식별화.

    Args:
        name:    원본 이름 (한글/영문/NaN/None)
        user_no: 있으면 alias 형태 "성** (A-xxxxx)"로 확장

    Examples:
        >>> mask_name("홍길동")
        '홍**'
        >>> mask_name("홍길동", user_no="32763")
        '홍** (A-a3f8e)'
        >>> mask_name("John Kim")
        'J** K**'
        >>> mask_name("")
        ''
        >>> mask_name(None)
        ''
    """
    if name is None:
        return ""
    if _PANDAS_AVAILABLE and pd.isna(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""

    # 이미 마스킹된 것(** 포함) → 그대로 반환 (중복 마스킹 방지)
    if "**" in s:
        base_masked = s
    else:
        parts = s.split()
        if len(parts) >= 2:
            # 공백 분리 (영문 복수 단어)
            masked = [p if len(p) <= 1 else p[0] + "**" for p in parts]
            base_masked = " ".join(masked)
        else:
            base_masked = s if len(s) <= 1 else s[0] + "**"

    # user_no 있으면 alias 뒤에 붙임
    if user_no is not None and str(user_no).strip():
        alias = mask_user_no(user_no)
        if alias:
            return f"{base_masked} ({alias})"
    return base_masked


def mask_name_series(series, user_no_series=None):
    """pandas Series 전체를 마스킹 (user_no 병렬 옵션)."""
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas 필요")
    if user_no_series is None:
        return series.apply(mask_name)
    # index 기반 zip
    return pd.Series(
        [mask_name(n, u) for n, u in zip(series, user_no_series)],
        index=series.index,
    )


def mask_names_in_df(df, column: str = "user_name", user_no_col: str | None = None):
    """
    DataFrame 의 이름 컬럼에 마스킹을 일괄 적용. 원본 불변.

    Args:
        column:       마스킹할 이름 컬럼 (기본 "user_name")
        user_no_col:  alias 를 덧붙이려면 사번 컬럼명 지정 (기본 None — UI 호환).
                      예: mask_names_in_df(df, "user_name", user_no_col="user_no")
                      → "홍** (A-xxxxx)" 형태로 렌더링.

    M1 Security 주의: alias 형태는 LLM 전송/감사용으로만 쓰고, 기존 UI
    (daily/weekly 탭 등) 는 기본값 (None) 으로 두어 변동 없게 한다.
    """
    if not _PANDAS_AVAILABLE:
        raise RuntimeError("pandas 필요")
    if column not in df.columns:
        return df
    result = df.copy()
    if user_no_col and user_no_col in df.columns:
        result[column] = mask_name_series(df[column], df[user_no_col])
    else:
        result[column] = df[column].apply(mask_name)
    return result


# ─── user_no 마스킹 ───────────────────────────────────────────
# Salt 는 환경변수 DEEPCON_USER_NO_SALT 로 주입한다.
# 기본값은 개발/테스트용이며, 운영 배포 전 반드시 랜덤 32자로 교체해야 한다.
# salt 가 바뀌면 기존 alias (A-xxxxx) 가 깨지므로 로그/감사 연속성 주의.
#
# 지연 해석: .env 로딩이 config.py import 시점에 일어나므로,
# 모듈 import 타이밍에 getenv 하면 fallback 이 박힐 수 있다.
# → 호출 시마다 getenv (결과 캐싱).
import os as _os

_USER_NO_SALT_FALLBACK = "DEEPCON_V3_USER_NO_SALT_DEV_ONLY"
_USER_NO_SALT_CACHE: str | None = None


def _get_user_no_salt() -> str:
    """env 에서 salt 를 조회 (한 번 읽으면 프로세스 내 재사용)."""
    global _USER_NO_SALT_CACHE
    if _USER_NO_SALT_CACHE is None:
        _USER_NO_SALT_CACHE = _os.getenv(
            "DEEPCON_USER_NO_SALT", _USER_NO_SALT_FALLBACK
        )
    return _USER_NO_SALT_CACHE


# 기존 코드 호환용 module 상수 (현재 env 상태로 해석)
# 주의: import 타이밍에 따라 fallback 일 수 있음. mask_user_no() 는
# _get_user_no_salt() 를 직접 사용하므로 env 가 나중에 설정돼도 반영된다.
_USER_NO_SALT = _get_user_no_salt()


def mask_user_no(user_no, prefix: str = "A-") -> str:
    """
    user_no → `A-<sha256 앞 5자>`

    일관성: 동일 user_no → 항상 동일 alias.
    salt는 프로세스 전역 상수 (변경하면 기존 alias 깨짐).

    >>> mask_user_no("32763")
    'A-a3f8e'
    >>> mask_user_no(32763) == mask_user_no("32763")
    True
    >>> mask_user_no("")
    ''
    """
    if user_no is None:
        return ""
    if _PANDAS_AVAILABLE and pd.isna(user_no):
        return ""
    s = str(user_no).strip()
    if not s:
        return ""
    # 이미 마스킹된 형태면 그대로
    if s.startswith(prefix):
        return s
    digest = hashlib.sha256((_get_user_no_salt() + s).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:5]}"


# ─── twardid 마스킹 ───────────────────────────────────────────
_TWARDID_RE = re.compile(r"^(T-\d{2})-([A-Za-z0-9]+)$")


def mask_twardid(twardid) -> str:
    """
    T-Ward ID 마지막 세그먼트를 마스킹.

    >>> mask_twardid("T-41-12345678")
    'T-41-****'
    >>> mask_twardid("T-10-ABCD")
    'T-10-****'
    >>> mask_twardid("")
    ''
    """
    if twardid is None:
        return ""
    if _PANDAS_AVAILABLE and pd.isna(twardid):
        return ""
    s = str(twardid).strip()
    if not s:
        return ""
    m = _TWARDID_RE.match(s)
    if not m:
        return s  # 포맷 불일치 → 원본 (이미 마스킹됐거나 비규격)
    return f"{m.group(1)}-****"


# ─── 자유 텍스트 마스킹 ─────────────────────────────────────────
# 한글 이름 2~4자 + 선택적 공백 + (님/씨) 패턴
_KR_NAME_HONORIFIC_RE = re.compile(r"([가-힣]{2,4})\s?(씨|님|과장|대리|부장|팀장|소장)")
# 한국 사번 형태 (5자리 숫자) — 보수적으로 공백 또는 문장 경계 사이
_USER_NO_IN_TEXT_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
# twardid 본문 내 패턴
_TWARDID_IN_TEXT_RE = re.compile(r"\bT-\d{2}-[A-Za-z0-9]{4,}\b")


def mask_free_text(
    text: str,
    *,
    known_names: Iterable[str] | None = None,
    known_user_nos: Iterable[str] | None = None,
) -> str:
    """
    자유 텍스트에 마스킹 규칙을 적용.

    순서가 중요:
      1) known_names 의 실명을 먼저 치환 (긴 이름부터)
      2) known_user_nos 를 alias 로 치환
      3) 정규식 기반 잔여 패턴 마스킹 (한글 이름+호칭, 5자리 숫자, twardid)

    >>> mask_free_text("홍길동 씨가 작업")
    '홍** 씨가 작업'
    >>> mask_free_text("작업자 32763 이동")
    '작업자 A-a3f8e 이동'
    >>> mask_free_text("T-41-12345678 신호 수신")
    'T-41-**** 신호 수신'
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s

    # 1) 실명 치환 (긴 이름부터)
    if known_names:
        names = sorted({n.strip() for n in known_names if n and str(n).strip()},
                       key=len, reverse=True)
        for n in names:
            s = s.replace(n, mask_name(n))

    # 2) 사번 치환 (긴 것 우선)
    if known_user_nos:
        nos = sorted({str(u).strip() for u in known_user_nos if u and str(u).strip()},
                     key=len, reverse=True)
        for u in nos:
            s = s.replace(u, mask_user_no(u))

    # 3) 잔여 패턴 마스킹
    # 3-1. twardid
    s = _TWARDID_IN_TEXT_RE.sub(lambda m: mask_twardid(m.group(0)), s)
    # 3-2. 한글 이름 + 호칭 → "X** 호칭"
    def _kr_name_sub(m: re.Match) -> str:
        name, honor = m.group(1), m.group(2)
        return f"{mask_name(name)} {honor}"
    s = _KR_NAME_HONORIFIC_RE.sub(_kr_name_sub, s)
    # 3-3. 5자리 숫자 → user_no alias
    #     주의: 좌표/시간 등 수치와 충돌 가능성 있음 — 보수적 패턴
    s = _USER_NO_IN_TEXT_RE.sub(lambda m: mask_user_no(m.group(1)), s)
    return s


# ─── 잔여 PII 탐지 (검증용) ───────────────────────────────────
_LEFTOVER_PATTERNS: dict[str, re.Pattern[str]] = {
    "kr_name_honorific": _KR_NAME_HONORIFIC_RE,
    "user_no_5digit":    _USER_NO_IN_TEXT_RE,
    "raw_twardid":       _TWARDID_IN_TEXT_RE,
}


def detect_leftover_pii(text: str) -> list[dict]:
    """
    텍스트에 남아 있는 PII 의심 패턴을 찾는다 (WARNING 용).

    Returns:
        [{"kind": str, "match": str, "span": (start, end)}, ...]
    """
    if not text:
        return []
    findings: list[dict] = []
    for kind, pat in _LEFTOVER_PATTERNS.items():
        for m in pat.finditer(text):
            # 이미 마스킹된 형태 (** 포함 / A-xxxxx 형식) 는 건너뜀
            matched = m.group(0)
            if "**" in matched or matched.startswith("A-"):
                continue
            findings.append({
                "kind": kind,
                "match": matched,
                "span": (m.start(), m.end()),
            })
    return findings
