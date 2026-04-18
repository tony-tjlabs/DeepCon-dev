"""
Google Drive 업로드 모듈 (DeepCon Pipeline용).

전처리된 데이터를 Google Drive에 업로드한다.
Streamlit Cloud 배포 시, 앱은 Drive에서 이 데이터를 다운로드하여 사용.

Drive 폴더 구조:
  GDRIVE_FOLDER_ID/
    ├── Y1_SKHynix_20260316_journey.parquet
    ├── Y1_SKHynix_20260316_worker.parquet
    ├── Y1_SKHynix_20260316_company.parquet
    ├── Y1_SKHynix_20260316_space.parquet
    ├── Y1_SKHynix_20260316_meta.json
    ├── model__Y1_SKHynix__w2v_model.pkl        ← 모델 파일
    ├── model__Y1_SKHynix__kmeans_model.pkl
    ├── model__Y1_SKHynix__cluster_labels.json
    ├── model__Y1_SKHynix__embedder_meta.json
    └── ...
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 프로젝트 루트
_ROOT = Path(__file__).resolve().parent.parent.parent
TOKEN_PATH = _ROOT / "token.json"
CLIENT_SECRET_PATH = _ROOT / "client_secret.json"
DRIVE_CONFIG_PATH = _ROOT / "drive_config.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]


def is_drive_configured() -> bool:
    """Google Drive 업로드 설정이 완료되었는지 확인."""
    if not DRIVE_CONFIG_PATH.exists():
        return False
    try:
        config = json.loads(DRIVE_CONFIG_PATH.read_text(encoding="utf-8"))
        # v2: sector별 폴더 구조
        if "sectors" in config:
            return any(
                s.get("folder_id") for s in config["sectors"].values()
            )
        # v1: 단일 폴더 (하위호환)
        folder_id = config.get("folder_id", "")
        return bool(folder_id) and not folder_id.startswith("여기에")
    except Exception:
        return False


def _get_drive_service():
    """OAuth2로 Google Drive API 서비스 생성 (토큰 재사용)."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except (ValueError, KeyError) as e:
            import logging
            logging.getLogger(__name__).warning(f"토큰 파일 손상, 재인증 필요: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        else:
            raise RuntimeError(
                "Google Drive 인증 토큰이 없거나 만료되었습니다.\n"
                "터미널에서 `python upload_to_drive.py` 를 실행하여 인증해 주세요."
            )

    return build("drive", "v3", credentials=creds)


def _load_drive_config(sector_id: str | None = None) -> dict:
    """
    drive_config.json 로드.

    sector_id 지정 시 → 해당 sector의 {folder_id, folder_name} 반환.
    미지정 시 → 전체 config 반환.
    """
    config = json.loads(DRIVE_CONFIG_PATH.read_text(encoding="utf-8"))

    # v2: sector별 폴더 구조
    if "sectors" in config and sector_id:
        sector_cfg = config["sectors"].get(sector_id, {})
        if not sector_cfg:
            raise ValueError(f"sector {sector_id} not in drive_config.json")
        return sector_cfg

    # v1: 단일 폴더 (하위호환)
    return config


def _list_drive_files(service, folder_id: str) -> dict[str, dict]:
    """Drive 폴더의 파일 목록 → {name: {id, size, modifiedTime}}."""
    files = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, size, modifiedTime)",
            pageToken=page_token, pageSize=1000,
        ).execute()
        for f in resp.get("files", []):
            files[f["name"]] = f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def check_upload_status(
    date_list: list[str],
    sector_id: str,
) -> dict:
    """
    업로드 전 사전 검사: 신규/변경/동일 파일 수를 미리 계산.

    Returns:
        {"new": [...], "changed": [...], "synced": [...], "total_new_size": int}
    """
    import config as cfg

    config = _load_drive_config(sector_id)
    service = _get_drive_service()
    drive_files = _list_drive_files(service, config["folder_id"])

    proc_dir = cfg.PROCESSED_DIR / sector_id
    result = {"new": [], "changed": [], "synced": [], "total_new_size": 0}

    for date_str in date_list:
        date_dir = proc_dir / date_str
        if not date_dir.exists():
            continue
        for f in sorted(date_dir.iterdir()):
            if f.suffix not in (".parquet", ".json"):
                continue
            drive_name = f"{sector_id}_{date_str}_{f.name}"
            local_size = f.stat().st_size
            existing = drive_files.get(drive_name)

            if not existing:
                result["new"].append(drive_name)
                result["total_new_size"] += local_size
            else:
                # 수정 시간 비교
                from datetime import datetime, timezone
                local_mtime = datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc
                )
                drive_mtime_str = existing.get("modifiedTime", "")
                try:
                    drive_mtime = datetime.fromisoformat(
                        drive_mtime_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    drive_mtime = datetime.min.replace(tzinfo=timezone.utc)

                if local_mtime > drive_mtime:
                    result["changed"].append(drive_name)
                    result["total_new_size"] += local_size
                else:
                    result["synced"].append(drive_name)

    return result


def upload_processed_dates(
    date_list: list[str],
    sector_id: str,
    progress_callback=None,
    force: bool = False,
) -> dict:
    """
    전처리된 날짜 데이터를 Google Drive에 업로드.

    Args:
        date_list: 업로드할 날짜 목록 (YYYYMMDD)
        sector_id: Sector ID (e.g., "Y1_SKHynix")
        progress_callback: (current, total, message) → None

    Returns:
        {"uploaded": int, "skipped": int, "errors": int, "details": [...]}
    """
    import config as cfg
    from googleapiclient.http import MediaFileUpload

    config = _load_drive_config(sector_id)
    folder_id = config["folder_id"]
    service = _get_drive_service()

    # Drive 기존 파일 목록
    drive_files = _list_drive_files(service, folder_id)

    # 업로드 대상 수집
    proc_dir = cfg.PROCESSED_DIR / sector_id
    files_to_upload: list[tuple[Path, str]] = []  # (local_path, drive_name)

    for date_str in date_list:
        date_dir = proc_dir / date_str
        if not date_dir.exists():
            continue
        for f in sorted(date_dir.iterdir()):
            if f.suffix in (".parquet", ".json"):
                drive_name = f"{sector_id}_{date_str}_{f.name}"
                files_to_upload.append((f, drive_name))

    total = len(files_to_upload)
    result = {"uploaded": 0, "skipped": 0, "errors": 0, "details": []}

    for i, (local_path, drive_name) in enumerate(files_to_upload):
        if progress_callback:
            progress_callback(i, total, f"업로드 중: {drive_name}")

        existing = drive_files.get(drive_name)

        # 수정 시간 비교 → 로컬이 더 새로우면 업로드 (force=True면 무조건)
        if existing and not force:
            from datetime import datetime, timezone
            local_mtime = datetime.fromtimestamp(
                local_path.stat().st_mtime, tz=timezone.utc
            )
            drive_mtime_str = existing.get("modifiedTime", "")
            try:
                drive_mtime = datetime.fromisoformat(
                    drive_mtime_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                drive_mtime = datetime.min.replace(tzinfo=timezone.utc)

            if local_mtime <= drive_mtime:
                result["skipped"] += 1
                continue

        try:
            mime = "application/json" if local_path.suffix == ".json" else "application/octet-stream"
            media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)

            if existing:
                service.files().update(fileId=existing["id"], media_body=media).execute()
            else:
                meta = {"name": drive_name, "parents": [folder_id]}
                service.files().create(body=meta, media_body=media, fields="id").execute()

            result["uploaded"] += 1
            result["details"].append(f"✅ {drive_name}")
        except Exception as e:
            result["errors"] += 1
            result["details"].append(f"❌ {drive_name}: {e}")
            logger.error(f"Drive 업로드 실패: {drive_name} — {e}")

    if progress_callback:
        progress_callback(total, total, "업로드 완료!")

    return result


def upload_model_files(
    sector_id: str,
    progress_callback=None,
) -> dict:
    """
    Journey 임베딩 모델 파일을 Google Drive에 업로드.

    Drive 파일명 규칙: model__{sector_id}__{filename}
    예: model__Y1_SKHynix__w2v_model.pkl

    Returns:
        {"uploaded": int, "skipped": int, "errors": int, "details": [...]}
    """
    import config as cfg
    from googleapiclient.http import MediaFileUpload

    model_dir = cfg.MODEL_DIR / sector_id
    if not model_dir.exists():
        return {"uploaded": 0, "skipped": 0, "errors": 0, "details": ["모델 디렉토리 없음"]}

    config = _load_drive_config(sector_id)
    folder_id = config["folder_id"]
    service = _get_drive_service()
    drive_files = _list_drive_files(service, folder_id)

    # 모델 파일 수집
    MODEL_EXTENSIONS = {".pkl", ".json"}
    files_to_upload: list[tuple[Path, str]] = []
    for f in sorted(model_dir.iterdir()):
        if f.suffix in MODEL_EXTENSIONS:
            drive_name = f"model__{sector_id}__{f.name}"
            files_to_upload.append((f, drive_name))

    total = len(files_to_upload)
    result = {"uploaded": 0, "skipped": 0, "errors": 0, "details": []}

    for i, (local_path, drive_name) in enumerate(files_to_upload):
        if progress_callback:
            progress_callback(i, total, f"모델 업로드: {drive_name}")

        existing = drive_files.get(drive_name)
        if existing:
            from datetime import datetime, timezone
            local_mtime = datetime.fromtimestamp(
                local_path.stat().st_mtime, tz=timezone.utc
            )
            drive_mtime_str = existing.get("modifiedTime", "")
            try:
                drive_mtime = datetime.fromisoformat(
                    drive_mtime_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                drive_mtime = datetime.min.replace(tzinfo=timezone.utc)

            if local_mtime <= drive_mtime:
                result["skipped"] += 1
                continue

        try:
            mime = "application/json" if local_path.suffix == ".json" else "application/octet-stream"
            media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)

            if existing:
                service.files().update(fileId=existing["id"], media_body=media).execute()
            else:
                meta = {"name": drive_name, "parents": [folder_id]}
                service.files().create(body=meta, media_body=media, fields="id").execute()

            result["uploaded"] += 1
            result["details"].append(f"✅ {drive_name}")
        except Exception as e:
            result["errors"] += 1
            result["details"].append(f"❌ {drive_name}: {e}")
            logger.error(f"모델 업로드 실패: {drive_name} — {e}")

    if progress_callback:
        progress_callback(total, total, "모델 업로드 완료!")

    return result


def get_drive_file_count(sector_id: str = None) -> int:
    """Drive에 업로드된 파일 수 조회."""
    try:
        config = _load_drive_config(sector_id)
        service = _get_drive_service()
        files = _list_drive_files(service, config["folder_id"])
        return len(files)
    except Exception:
        return -1
