"""
rotate_passwords.py — DeepCon 계정 비밀번호 해시 생성 도구
=============================================================
argon2id 해시를 대화형으로 생성하고 `.env` 에 주입할 수 있는
운영자용 CLI. 평문 비밀번호는 **절대** 터미널 스크롤에 남지 않으며,
해시 출력은 stdout 으로만 표시된다 (로그 파일 저장 금지).

사용법:
    $ python -m scripts.rotate_passwords
    계정 ID (administrator/Y1_SKHynix/M15X_SKHynix): Y1_SKHynix
    새 비밀번호 (getpass, 입력 안 보임): ********
    확인 비밀번호: ********

    결과 (복사하여 .env 에 반영):
    DEEPCON_Y1_PW_HASH=$argon2id$v=19$m=65536,t=3,p=2$...

여러 계정을 연속으로 처리할 수도 있다:
    $ python -m scripts.rotate_passwords --all

보안:
  - 평문 입력은 getpass.getpass 로 비가시 처리
  - 해시는 stdout 에만, 파일/로그에 자동 저장하지 않음
  - OWASP 2024 argon2id 기본값 (t=3, m=65536 KiB, p=2)
"""
from __future__ import annotations

import argparse
import getpass
import sys

try:
    from argon2 import PasswordHasher
except ImportError:
    print(
        "ERROR: argon2-cffi 가 설치되어 있지 않습니다. "
        "`pip install argon2-cffi` 또는 requirements.txt 재설치 후 다시 실행하세요.",
        file=sys.stderr,
    )
    sys.exit(2)


# DeepCon 계정 → .env 환경변수명 매핑 (src/dashboard/auth.py._USER_PWHASH_ENV 와 동기화)
_ACCOUNT_ENV = {
    "administrator": "DEEPCON_ADMIN_PW_HASH",
    "Y1_SKHynix":    "DEEPCON_Y1_PW_HASH",
    "M15X_SKHynix":  "DEEPCON_M15X_PW_HASH",
}


def _read_new_password(account_id: str) -> str:
    """getpass 2회 확인으로 비밀번호 획득."""
    while True:
        pw1 = getpass.getpass(f"[{account_id}] 새 비밀번호 (입력 안 보임): ")
        if len(pw1) < 8:
            print("⚠ 비밀번호는 최소 8자 이상이어야 합니다. 다시 입력하세요.",
                  file=sys.stderr)
            continue
        pw2 = getpass.getpass(f"[{account_id}] 확인 비밀번호: ")
        if pw1 != pw2:
            print("⚠ 두 비밀번호가 일치하지 않습니다. 다시 입력하세요.",
                  file=sys.stderr)
            continue
        return pw1


def _hash_one(account_id: str) -> str:
    """단일 계정에 대해 해시 생성 후 `.env` 항목을 출력."""
    if account_id not in _ACCOUNT_ENV:
        print(f"ERROR: 알 수 없는 계정 ID '{account_id}'. "
              f"사용 가능: {list(_ACCOUNT_ENV)}", file=sys.stderr)
        sys.exit(3)
    env_key = _ACCOUNT_ENV[account_id]
    pw = _read_new_password(account_id)
    hasher = PasswordHasher()  # OWASP 2024 기본값
    hash_str = hasher.hash(pw)
    # 평문은 지역 변수에서 즉시 폐기 (GC 신뢰)
    del pw
    return f"{env_key}={hash_str}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepCon 계정 비밀번호 argon2id 해시 생성기"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="3개 계정 전부 순차 처리",
    )
    parser.add_argument(
        "--account",
        choices=list(_ACCOUNT_ENV),
        help="특정 계정만 처리 (미지정 시 대화형 프롬프트)",
    )
    args = parser.parse_args()

    if args.all:
        accounts = list(_ACCOUNT_ENV)
    elif args.account:
        accounts = [args.account]
    else:
        default_hint = " / ".join(_ACCOUNT_ENV.keys())
        acc = input(f"계정 ID ({default_hint}): ").strip()
        accounts = [acc]

    print("\n=== 해시 결과 (아래 라인을 `.env` 에 복사) ===\n")
    for acc in accounts:
        line = _hash_one(acc)
        print(line)
    print(
        "\n"
        "⚠ 중요: 위 해시를 `.env` 에 반영한 뒤 이 터미널의 스크롤 버퍼를 "
        "지우거나 창을 닫으세요. 해시 자체는 복원 불가능하지만 해시 문자열이 "
        "노출되면 공격자가 오프라인 공격에 활용할 수 있습니다.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
