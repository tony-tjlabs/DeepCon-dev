"""
Google Drive 다운로드 모듈 (CLOUD_MODE용).

Streamlit Cloud에서 Google Drive에 저장된 전처리 데이터를
sector별 폴더에서 다운로드하여 로컬에 복원한다.

★ v2: sector별 독립 폴더 구조
  Drive:
    DeepCon_Y1_SKHynix/     ← Y1 전용 폴더
      Y1_SKHynix_20260316_worker.parquet
      model__Y1_SKHynix__w2v_model.pkl
    DeepCon_M15X_SKHynix/   ← M15X 전용 폴더
      M15X_SKHynix_20260309_worker.parquet
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveStorage:
    """Google Drive에서 DeepCon 캐시를 다운로드."""

    def __init__(self, credentials_info: dict, folder_id: str) -> None:
        self._folder_id = folder_id
        self._credentials_info = credentials_info
        self._service = None
        self._local_dir: Path | None = None
        self._file_list_cache: list[dict] | None = None

    @property
    def local_cache_dir(self) -> Path | None:
        return self._local_dir

    def _get_service(self):
        if self._service is not None:
            return self._service
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_service_account_info(
            self._credentials_info, scopes=_SCOPES
        )
        self._service = build("drive", "v3", credentials=creds)
        return self._service

    def list_files(self, force_refresh: bool = False) -> list[dict]:
        if self._file_list_cache is not None and not force_refresh:
            return self._file_list_cache
        service = self._get_service()
        files, page_token = [], None
        while True:
            resp = service.files().list(
                q=f"'{self._folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, size)",
                pageToken=page_token, pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        self._file_list_cache = files
        return files

    @staticmethod
    def _parse_drive_name(name: str) -> tuple[str, str, str] | None:
        """Drive 파일명 파싱: {sector}_{date}_{file}.ext"""
        m = re.match(r"^(.+?)_(\d{8})_(.+)$", name)
        return (m.group(1), m.group(2), m.group(3)) if m else None

    @staticmethod
    def _parse_model_name(name: str) -> tuple[str, str] | None:
        """모델 파일명 파싱: model__{sector}__{file}"""
        m = re.match(r"^model__(.+?)__(.+)$", name)
        return (m.group(1), m.group(2)) if m else None

    # Cloud 시작 시 다운로드 제외할 파일 (용량/시간 절약)
    # journey_slim.parquet 는 온디맨드(ensure_journey_slim)로만 다운로드
    _SKIP_PATTERNS = {"journey.parquet", "journey_slim.parquet"}

    def sync_all(self) -> int:
        """Drive 파일을 로컬에 다운로드 (journey.parquet 제외, 모델 포함)."""
        from googleapiclient.http import MediaIoBaseDownload

        if self._local_dir is None:
            logger.warning("DriveStorage: local_dir 미설정")
            return 0

        files = self.list_files(force_refresh=True)
        service = self._get_service()
        new_count = 0
        skip_count = 0

        for f in files:
            # ── 모델 파일 (model__{sector}__{file}) ──
            model_parsed = self._parse_model_name(f["name"])
            if model_parsed:
                sector_id, filename = model_parsed
                import config as cfg
                local_dir = cfg.MODEL_DIR / sector_id
                local_path = local_dir / filename

                drive_size = int(f.get("size", 0))
                if local_path.exists() and abs(local_path.stat().st_size - drive_size) < 100:
                    continue

                local_dir.mkdir(parents=True, exist_ok=True)
                try:
                    request = service.files().get_media(
                        fileId=f["id"], supportsAllDrives=True
                    )
                    buf = io.BytesIO()
                    downloader = MediaIoBaseDownload(buf, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    buf.seek(0)
                    local_path.write_bytes(buf.read())
                    new_count += 1
                except Exception as e:
                    logger.warning(f"Drive model download failed: {f['name']} — {e}")
                continue

            # ── 데이터 파일 ({sector}_{date}_{file}) ──
            parsed = self._parse_drive_name(f["name"])
            if not parsed:
                continue
            sector_id, date_str, filename = parsed

            if filename in self._SKIP_PATTERNS:
                skip_count += 1
                continue

            local_dir = self._local_dir / sector_id / date_str
            local_path = local_dir / filename

            drive_size = int(f.get("size", 0))
            if local_path.exists() and abs(local_path.stat().st_size - drive_size) < 100:
                continue

            local_dir.mkdir(parents=True, exist_ok=True)
            try:
                request = service.files().get_media(
                    fileId=f["id"], supportsAllDrives=True
                )
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                buf.seek(0)
                local_path.write_bytes(buf.read())
                new_count += 1
                logger.info(f"Drive updated: {f['name']} ({drive_size:,} bytes)")
            except Exception as e:
                logger.warning(f"Drive download failed: {f['name']} — {e}")

        logger.info(f"Drive sync: {new_count} new, {skip_count} skipped / {len(files)} total")
        return new_count

    def ensure_journey_slim(self, sector_id: str, date_str: str, local_processed_dir: "Path") -> bool:
        """
        특정 날짜의 journey_slim.parquet 를 온디맨드로 다운로드.

        - 이미 로컬에 있으면 즉시 True 반환 (재다운로드 없음)
        - Drive에서 {sector_id}_{date_str}_journey_slim.parquet 를 찾아 다운로드
        - 실패 시 False 반환 (탭에서 graceful degradation)

        Args:
            sector_id: 섹터 ID (예: Y1_SKHynix)
            date_str:  날짜 문자열 (YYYYMMDD)
            local_processed_dir: 로컬 processed 루트 경로 (cfg.PROCESSED_DIR)

        Returns:
            True: 다운로드 성공 또는 이미 존재
            False: Drive에 파일 없거나 다운로드 실패
        """
        local_path = local_processed_dir / sector_id / date_str / "journey_slim.parquet"

        # 이미 있으면 스킵
        if local_path.exists():
            return True

        drive_name = f"{sector_id}_{date_str}_journey_slim.parquet"
        files      = self.list_files()
        target     = next((f for f in files if f["name"] == drive_name), None)

        if target is None:
            logger.info(f"Drive에 {drive_name} 없음 (아직 업로드 안 됐거나 해당 날짜 없음)")
            return False

        try:
            from googleapiclient.http import MediaIoBaseDownload
            service = self._get_service()
            request = service.files().get_media(
                fileId=target["id"], supportsAllDrives=True
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            local_path.write_bytes(buf.read())
            size_mb = round(local_path.stat().st_size / 1024 / 1024, 1)
            logger.info(f"journey_slim 다운로드 완료: {drive_name} ({size_mb}MB)")
            return True
        except Exception as e:
            logger.warning(f"journey_slim 다운로드 실패 ({drive_name}): {e}")
            return False


# ── sector별 폴더 ID 기본값 ──────────────────────────────────────
_DEFAULT_SECTOR_FOLDERS: dict[str, str] = {
    "Y1_SKHynix":   "1d50bz14seBR-GyNpo5NWUt2E6di5eWDj",
    "M15X_SKHynix": "1GvrljfFQNxPMqXb8S-Tsu4hccufYyRjw",
}


def init_drive_storage_from_secrets(sector_id: str | None = None) -> dict[str, DriveStorage] | DriveStorage | None:
    """
    Streamlit secrets에서 DriveStorage 초기화.

    sector_id=None → 전체 sector dict 반환 {sector_id: DriveStorage}
    sector_id=str  → 해당 sector의 DriveStorage 반환
    """
    try:
        import streamlit as st

        gcp_info = (
            st.secrets.get("google_drive_sa")
            or st.secrets.get("gcp_service_account")
        )
        if not gcp_info:
            logger.info("DriveStorage: SA 키 없음 (google_drive_sa)")
            return None

        gcp_dict = dict(gcp_info)

        # sector별 폴더 ID (secrets > 기본값)
        folder_overrides = {}
        try:
            folder_overrides = dict(st.secrets.get("google_drive_folders", {}))
        except Exception:
            pass

        def _get_folder_id(sid: str) -> str | None:
            return folder_overrides.get(sid) or _DEFAULT_SECTOR_FOLDERS.get(sid)

        if sector_id is not None:
            fid = _get_folder_id(sector_id)
            if not fid:
                logger.warning(f"DriveStorage: sector {sector_id} 폴더 ID 없음")
                return None
            return DriveStorage(gcp_dict, fid)

        # 전체 sector
        result: dict[str, DriveStorage] = {}
        for sid, default_fid in _DEFAULT_SECTOR_FOLDERS.items():
            fid = _get_folder_id(sid)
            if fid:
                result[sid] = DriveStorage(gcp_dict, fid)
        return result if result else None

    except Exception as e:
        logger.warning(f"DriveStorage init failed: {e}")
        return None
