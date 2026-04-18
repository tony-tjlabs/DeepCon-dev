"""
DeepCon 로그인 & 인증 시스템 (M1 Security — 2026-04-18)
=======================================================
Sector 기반 접근 제어:
  - Administrator → 모든 Sector 접근
  - Client        → 지정된 Sector만 접근

★ 보안 (M1 업그레이드):
  - 비밀번호: argon2id 해시 저장 (`.env`의 DEEPCON_*_PW_HASH), 평문 저장 금지
  - argon2.PasswordHasher.verify() → constant-time + 내부 delay (timing attack 방지)
  - 세션 관리: Idle 30분 / Absolute 8시간 타임아웃
  - Lockout: 로그인 실패 5회 → 15분 잠금 (data/security/lockout.json 영속화,
             동일 브라우저/여러 탭/재시작에도 유효)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import argon2
    from argon2 import PasswordHasher
    from argon2.exceptions import (
        InvalidHashError,
        VerifyMismatchError,
        VerificationError,
    )
    _ARGON2_AVAILABLE = True
except ImportError:
    _ARGON2_AVAILABLE = False

import config as cfg

log = logging.getLogger(__name__)


# ─── 보안 상수 ───────────────────────────────────────────────
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60           # 15분 (이전 30초 → 900초로 강화)
IDLE_TIMEOUT_SEC = 30 * 60          # 30분 무입력
ABSOLUTE_TIMEOUT_SEC = 8 * 3600     # 로그인 후 절대 8시간

# Lockout 영속 저장소
_LOCKOUT_DIR = cfg.BASE_DIR / "data" / "security"
_LOCKOUT_FILE = _LOCKOUT_DIR / "lockout.json"


# ─── user_id → argon2 해시 env 키 매핑 ─────────────────────────
# USER_REGISTRY 의 password 필드는 사용하지 않는다 (평문 저장 금지).
# 새 비밀번호 추가 시 이 테이블에만 항목을 넣는다.
_USER_PWHASH_ENV: dict[str, str] = {
    "administrator": "DEEPCON_ADMIN_PW_HASH",
    "Y1_SKHynix":    "DEEPCON_Y1_PW_HASH",
    "M15X_SKHynix":  "DEEPCON_M15X_PW_HASH",
}


def _get_pw_hash(user_id: str) -> str:
    """user_id 의 argon2 해시를 `.env` / Streamlit secrets 에서 로드."""
    env_key = _USER_PWHASH_ENV.get(user_id)
    if not env_key:
        return ""
    # config._get_secret: os.getenv → st.secrets → default
    return cfg._get_secret(env_key, "")


# argon2 해셔 (전역 싱글턴)
_PH: PasswordHasher | None = None


def _get_hasher() -> PasswordHasher:
    global _PH
    if _PH is None:
        if not _ARGON2_AVAILABLE:
            raise RuntimeError(
                "argon2-cffi 미설치 — `pip install argon2-cffi>=23.1.0` 필요"
            )
        _PH = PasswordHasher(
            time_cost=3, memory_cost=65536, parallelism=2, hash_len=32
        )
    return _PH


def verify_password(user_id: str, submitted: str) -> bool:
    """
    argon2id 해시로 비밀번호를 검증.
    - constant-time 비교 + argon2 내부 delay → timing attack 방지
    - 해시 미설정 / 해시 포맷 오류 시 False
    - 예외는 내부에서 흡수 (평문/해시가 로그/에러에 새지 않도록)
    """
    if not submitted:
        return False
    hashed = _get_pw_hash(user_id)
    if not hashed:
        log.warning("verify_password: %s 해시 미설정", user_id)
        return False
    try:
        _get_hasher().verify(hashed, submitted)
        return True
    except VerifyMismatchError:
        return False
    except (InvalidHashError, VerificationError) as exc:
        log.error("verify_password: 해시 포맷/검증 오류 user=%s type=%s",
                  user_id, type(exc).__name__)
        return False
    except Exception:  # noqa: BLE001
        log.exception("verify_password: 예기치 않은 오류 user=%s", user_id)
        return False


# ─── Lockout 영속 저장소 ───────────────────────────────────────
def _load_lockout_state() -> dict[str, dict[str, Any]]:
    """lockout.json 로드. 파일 없으면 빈 dict."""
    try:
        if _LOCKOUT_FILE.exists():
            return json.loads(_LOCKOUT_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        log.warning("lockout.json 로드 실패 — 빈 상태로 진행")
    return {}


def _save_lockout_state(state: dict[str, dict[str, Any]]) -> None:
    """lockout.json 저장. 쓰기 실패해도 로그인 흐름은 유지."""
    try:
        _LOCKOUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LOCKOUT_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(_LOCKOUT_FILE)
        try:
            os.chmod(_LOCKOUT_FILE, 0o600)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        log.exception("lockout.json 저장 실패")


def _get_lockout_info(user_id: str) -> dict[str, Any]:
    """특정 user_id 의 잠금 정보. 기본: attempts=0, locked_until=0."""
    state = _load_lockout_state()
    return state.get(user_id, {"attempts": 0, "locked_until": 0.0})


def _update_lockout_info(user_id: str, info: dict[str, Any]) -> None:
    state = _load_lockout_state()
    state[user_id] = info
    _save_lockout_state(state)


def _clear_lockout(user_id: str) -> None:
    state = _load_lockout_state()
    if user_id in state:
        del state[user_id]
        _save_lockout_state(state)


def _is_locked(user_id: str) -> tuple[bool, int]:
    """
    user_id 가 잠금 상태인지 확인. (is_locked, remaining_seconds)
    """
    info = _get_lockout_info(user_id)
    locked_until = float(info.get("locked_until", 0.0))
    now = time.time()
    if now < locked_until:
        return True, int(locked_until - now)
    # 잠금 만료 → 카운터 리셋
    if locked_until > 0 and info.get("attempts", 0) >= MAX_LOGIN_ATTEMPTS:
        _clear_lockout(user_id)
    return False, 0


# ─── 세션 상태 키 ─────────────────────────────────────────────────
_KEY_LOGGED_IN      = "deepcon_logged_in"
_KEY_USER_ID        = "deepcon_user_id"
_KEY_USER_ROLE      = "deepcon_user_role"
_KEY_USER_LABEL     = "deepcon_user_label"
_KEY_USER_ICON      = "deepcon_user_icon"
_KEY_CURRENT_SECTOR = "deepcon_current_sector"
_KEY_LOGIN_ERROR    = "deepcon_login_error"  # 에러 피드백용

# 세션 타임아웃 추적 키
_KEY_LAST_ACTIVITY  = "deepcon_last_activity"
_KEY_LOGIN_AT       = "deepcon_login_at"


# ─── Public API ───────────────────────────────────────────────────
def is_logged_in() -> bool:
    return st.session_state.get(_KEY_LOGGED_IN, False)


def get_current_user() -> dict:
    """현재 로그인 사용자 정보 반환."""
    return {
        "user_id": st.session_state.get(_KEY_USER_ID, ""),
        "role":    st.session_state.get(_KEY_USER_ROLE, ""),
        "label":   st.session_state.get(_KEY_USER_LABEL, ""),
        "icon":    st.session_state.get(_KEY_USER_ICON, ""),
    }


def get_current_sector() -> str | None:
    return st.session_state.get(_KEY_CURRENT_SECTOR)


def set_current_sector(sector_id: str):
    st.session_state[_KEY_CURRENT_SECTOR] = sector_id


def get_allowed_sectors() -> list[str]:
    """현재 사용자의 접근 가능 Sector 목록."""
    user_id = st.session_state.get(_KEY_USER_ID, "")
    return cfg.get_allowed_sectors_for_user(user_id)


def is_admin() -> bool:
    return st.session_state.get(_KEY_USER_ROLE) == "admin"


def logout():
    """로그아웃 — 세션 초기화."""
    for key in [_KEY_LOGGED_IN, _KEY_USER_ID, _KEY_USER_ROLE,
                _KEY_USER_LABEL, _KEY_USER_ICON, _KEY_CURRENT_SECTOR,
                _KEY_LAST_ACTIVITY, _KEY_LOGIN_AT]:
        st.session_state.pop(key, None)
    st.rerun()


def require_login():
    """로그인 게이트. 미인증 시 로그인 페이지 표시 후 실행 중단."""
    # 먼저 타임아웃 검사 (이미 로그인된 세션이어도 만료면 강제 로그아웃)
    enforce_timeout()
    if not is_logged_in():
        _render_login_page()
        st.stop()


def check_session() -> bool:
    """
    현재 세션이 유효한지 검사. 만료되면 False + 세션 정리.
    각 페이지 로드 시 호출할 수 있다.
    """
    if not is_logged_in():
        return False
    now = time.time()
    last = float(st.session_state.get(_KEY_LAST_ACTIVITY, 0))
    login_at = float(st.session_state.get(_KEY_LOGIN_AT, 0))

    # idle timeout
    if last > 0 and (now - last) > IDLE_TIMEOUT_SEC:
        _force_logout("idle")
        return False
    # absolute timeout
    if login_at > 0 and (now - login_at) > ABSOLUTE_TIMEOUT_SEC:
        _force_logout("absolute")
        return False

    # 활동 시간 갱신
    st.session_state[_KEY_LAST_ACTIVITY] = now
    return True


def enforce_timeout() -> None:
    """
    main.py / 라우터 상단에서 호출. 만료된 세션이면 정리 + 메시지.
    미로그인 상태는 그대로 통과 (로그인 페이지는 require_login 이 책임).
    """
    if not is_logged_in():
        return
    if not check_session():
        # _force_logout 내부에서 세션 클리어 + 에러 플래그 설정
        # rerun은 _force_logout 에서 호출됨
        return


def _force_logout(reason: str) -> None:
    """세션 만료 시 강제 로그아웃. reason: 'idle' | 'absolute'."""
    for key in [_KEY_LOGGED_IN, _KEY_USER_ID, _KEY_USER_ROLE,
                _KEY_USER_LABEL, _KEY_USER_ICON, _KEY_CURRENT_SECTOR,
                _KEY_LAST_ACTIVITY, _KEY_LOGIN_AT]:
        st.session_state.pop(key, None)
    st.session_state[_KEY_LOGIN_ERROR] = {
        "type": "account_error",
        "message": (
            "세션이 만료되었습니다 (30분 무입력). 다시 로그인하세요."
            if reason == "idle"
            else "세션이 만료되었습니다 (8시간 경과). 다시 로그인하세요."
        ),
    }
    st.rerun()


# ─── 에러 피드백 렌더링 ────────────────────────────────────────────
def _render_login_feedback(is_locked: bool, remaining_lockout: int):
    """
    로그인 에러 피드백 카드 렌더링.

    케이스별 시각 구분:
      - 케이스 A (비밀번호 오류 1~4회): danger 색상, 프로그레스 바
      - 케이스 B (잠금 상태 5회+): warning 색상, 남은 초 표시
      - 케이스 C (계정없음/설정오류): muted 색상, 관리자 문의 안내
    """
    error_info = st.session_state.get(_KEY_LOGIN_ERROR)
    if not error_info and not is_locked:
        return

    # 에러 표시 후 클리어 (다음 입력 시 사라짐)
    if error_info:
        st.session_state.pop(_KEY_LOGIN_ERROR, None)

    error_type = error_info.get("type", "") if error_info else ""

    # 케이스 B: 잠금 상태
    if is_locked or error_type == "lockout":
        remaining = remaining_lockout if remaining_lockout > 0 else error_info.get("remaining_seconds", LOCKOUT_SECONDS)
        mins = remaining // 60
        secs = remaining % 60
        human = f"{mins}분 {secs}초" if mins > 0 else f"{secs}초"
        st.markdown(
            f"""
            <div style='background:rgba(255,179,0,0.08); border-left:3px solid #FFB300;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#FFB300; font-size:0.88rem; font-weight:600;'>
                    X 로그인이 잠금되었습니다.
                </div>
                <div style='color:#9AB5D4; font-size:0.84rem; margin-top:6px;'>
                    잠금 해제까지 <span style='color:#FFB300; font-weight:700; font-size:1.1rem;'>{human}</span> 남음
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 케이스 A: 비밀번호 오류
    if error_type == "password_error":
        remaining = error_info.get("remaining_attempts", 0)
        attempts = error_info.get("attempts", 0)
        message = error_info.get("message", "비밀번호가 일치하지 않습니다.")

        # 프로그레스 바 생성 (채워진 칸 = 실패 횟수, 빈 칸 = 남은 시도)
        filled_slots = attempts
        empty_slots = remaining
        filled_bar = "".join(
            f"<div style='width:{100/MAX_LOGIN_ATTEMPTS - 1}%; height:6px; "
            f"background:#FF4C4C; border-radius:3px; margin-right:2px; display:inline-block;'></div>"
            for _ in range(filled_slots)
        )
        empty_bar = "".join(
            f"<div style='width:{100/MAX_LOGIN_ATTEMPTS - 1}%; height:6px; "
            f"background:#3A4A5A; border-radius:3px; margin-right:2px; display:inline-block;'></div>"
            for _ in range(empty_slots)
        )

        st.markdown(
            f"""
            <div style='background:rgba(255,76,76,0.08); border-left:3px solid #FF4C4C;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#FF4C4C; font-size:0.88rem; font-weight:600;'>
                    ! {message}
                </div>
                <div style='margin-top:10px;'>
                    {filled_bar}{empty_bar}
                </div>
                <div style='color:#7A8FA6; font-size:0.78rem; margin-top:4px;'>
                    남은 시도: {remaining}/{MAX_LOGIN_ATTEMPTS}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 케이스 C: 계정 없음 / 설정 오류 / 세션 만료
    if error_type == "account_error":
        message = error_info.get("message", "계정 정보를 확인할 수 없습니다.")
        st.markdown(
            f"""
            <div style='background:rgba(154,181,212,0.08); border-left:3px solid #9AB5D4;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#9AB5D4; font-size:0.88rem; font-weight:600;'>
                    ? {message}
                </div>
                <div style='color:#5A6A7A; font-size:0.82rem; margin-top:4px;'>
                    관리자에게 문의하세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return


# ─── 로그인 페이지 ────────────────────────────────────────────────
def _render_login_page():
    """전체 화면 로그인 UI."""
    # 로그인 페이지 전용 CSS — 사이드바 숨김 + form/vertical block 기본 배경 제거
    st.markdown(
        """
        <style>
            [data-testid='stSidebar']{display:none !important;}
            /* 로그인 페이지 전체 form/column 컨테이너 투명화 */
            [data-testid='stForm'] {
                background: transparent !important;
                border: none !important;
                padding: 0 !important;
            }
            [data-testid='stVerticalBlockBorderWrapper'] {
                background: transparent !important;
                border: none !important;
            }
            /* 혹시 있을 container 배경도 제거 */
            div[data-testid="column"] > div > div > div > div {
                background: transparent !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        # ── 로고 ────────────────────────────────────────────────────
        st.markdown(
            """
            <div style='text-align:center; padding: 48px 0 32px 0;'>
                <div style='font-size: 3.8rem; margin-bottom: 10px;'>🌐</div>
                <div style='font-size: 2.2rem; font-weight: 800; color: #00AEEF;
                            letter-spacing: 3px; font-family: monospace;'>
                    DeepCon
                </div>
                <div style='font-size: 0.88rem; color: #7A8FA6; margin-top: 8px;
                            letter-spacing: 1px;'>
                    Agentic AI based on Spatial Data
                </div>
                <div style='margin-top: 10px; font-size: 0.75rem; color: #3A4A5A;'>
                    TJLABS Research
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 로그인 제목 ─────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:1.1rem; font-weight:600; color:#C8D6E8;"
            "text-align:center; margin:24px 0 16px 0; letter-spacing:1px;'>"
            "Sign In"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── 사용자 선택 ─────────────────────────────────────────────
        user_options = {
            uid: f"{info['icon']}  {info['label']}"
            for uid, info in cfg.USER_REGISTRY.items()
        }
        selected_uid = st.selectbox(
            "계정 선택",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x],
            key="login_user_select",
        )

        # ── 비밀번호 + 엔터키 로그인 (st.form) ────────────────────
        # 잠금 상태 확인 (영속 저장소 + 계정별)
        is_locked, remaining_lockout = _is_locked(selected_uid)

        with st.form("login_form", clear_on_submit=False):
            password = st.text_input(
                "비밀번호", type="password", key="login_password",
                placeholder="Password"
            )

            # 에러 피드백 영역 (폼 내부, 버튼 위)
            _render_login_feedback(is_locked, remaining_lockout)

            submitted = st.form_submit_button(
                "로그인", use_container_width=True, type="primary",
                disabled=is_locked,
            )
            if submitted and not is_locked:
                _do_login(selected_uid, password)

        # ── Sector 미리보기 ─────────────────────────────────────────
        _render_sector_preview(selected_uid)

        # ── 버전 ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='text-align:center; color:#2A3A4A; font-size:0.72rem;"
            f"margin-top:20px;'>v{cfg.APP_VERSION}</div>",
            unsafe_allow_html=True,
        )


def _do_login(user_id: str, password: str):
    """로그인 검증 및 세션 설정. argon2id + 영속 lockout."""
    # ── 잠금 상태 재확인 ────────────────────────────────
    locked, _remaining = _is_locked(user_id)
    if locked:
        st.rerun()
        return

    user_info = cfg.USER_REGISTRY.get(user_id, {})
    if not user_info:
        # 케이스 C: 계정 없음
        st.session_state[_KEY_LOGIN_ERROR] = {
            "type": "account_error",
            "message": "계정 정보를 확인할 수 없습니다.",
        }
        st.rerun()
        return

    # ── 해시 미설정 체크 ─────────────────────────────────
    if not _get_pw_hash(user_id):
        st.session_state[_KEY_LOGIN_ERROR] = {
            "type": "account_error",
            "message": "비밀번호가 설정되지 않았습니다.",
        }
        st.rerun()
        return

    # ── argon2id 검증 ───────────────────────────────────
    if not verify_password(user_id, password):
        info = _get_lockout_info(user_id)
        attempts = int(info.get("attempts", 0)) + 1
        info["attempts"] = attempts

        if attempts >= MAX_LOGIN_ATTEMPTS:
            # 케이스 B: 잠금
            info["locked_until"] = time.time() + LOCKOUT_SECONDS
            _update_lockout_info(user_id, info)
            st.session_state[_KEY_LOGIN_ERROR] = {
                "type": "lockout",
                "message": "로그인이 잠금되었습니다.",
                "remaining_seconds": LOCKOUT_SECONDS,
            }
            log.warning("Login LOCKOUT user=%s attempts=%d duration=%ds",
                        user_id, attempts, LOCKOUT_SECONDS)
        else:
            # 케이스 A: 비밀번호 오류
            info["locked_until"] = 0.0
            _update_lockout_info(user_id, info)
            remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts
            st.session_state[_KEY_LOGIN_ERROR] = {
                "type": "password_error",
                "message": "비밀번호가 일치하지 않습니다.",
                "remaining_attempts": remaining_attempts,
                "attempts": attempts,
            }
            log.info("Login FAIL user=%s attempts=%d", user_id, attempts)
        st.rerun()
        return

    # ── 로그인 성공 — 잠금 상태 클리어 ───────────────────
    _clear_lockout(user_id)
    st.session_state.pop(_KEY_LOGIN_ERROR, None)

    # 세션 설정
    now = time.time()
    st.session_state[_KEY_LOGGED_IN]     = True
    st.session_state[_KEY_USER_ID]       = user_id
    st.session_state[_KEY_USER_ROLE]     = user_info["role"]
    st.session_state[_KEY_USER_LABEL]    = user_info["label"]
    st.session_state[_KEY_USER_ICON]     = user_info["icon"]
    st.session_state[_KEY_LOGIN_AT]      = now
    st.session_state[_KEY_LAST_ACTIVITY] = now

    # 기본 Sector: 허용 목록의 첫 번째 활성 Sector
    allowed = cfg.get_allowed_sectors_for_user(user_id)
    st.session_state[_KEY_CURRENT_SECTOR] = allowed[0] if allowed else None

    log.info("Login SUCCESS user=%s role=%s", user_id, user_info.get("role"))

    # 성공 피드백
    st.toast(f"환영합니다, {user_info['label']}님!", icon="✅")
    st.rerun()


def _render_sector_preview(user_id: str):
    """선택된 계정의 접근 가능 Sector 미리보기.

    ★ 단일 st.markdown 호출로 HTML 조립 — Streamlit이 각 markdown을 별도 컨테이너로
       감싸므로 여는/닫는 태그가 분리되면 빈 박스가 생성된다 (bug fix).
    """
    allowed = cfg.get_allowed_sectors_for_user(user_id)
    all_sectors = list(cfg.SECTOR_REGISTRY.keys())

    # Sector 항목 HTML 조립
    items_html = []
    for sid in all_sectors:
        info       = cfg.SECTOR_REGISTRY[sid]
        has_access = sid in allowed
        is_active  = info.get("status") == "active"
        color  = "#00C897" if (has_access and is_active) else "#3A4A5A"
        prefix = "✓" if (has_access and is_active) else "✕" if has_access else "—"
        note   = "" if is_active else " (준비중)"
        items_html.append(
            f"<div style='font-size:0.82rem; color:{color}; padding:2px 0;'>"
            f"{prefix}  {info['icon']} {info['label']}{note}</div>"
        )

    # 전체 카드를 단일 markdown으로 렌더링 (빈 박스 방지)
    st.markdown(
        "<div style='margin-top:16px; padding:12px 16px; background:#111820;"
        "border-radius:10px; border:1px solid #1A2A3A;'>"
        "<div style='font-size:0.78rem; color:#7A8FA6; margin-bottom:8px;'>"
        "접근 가능 Sector</div>"
        + "".join(items_html)
        + "</div>",
        unsafe_allow_html=True,
    )
