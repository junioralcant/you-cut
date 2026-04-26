import logging
import socket
import time
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .auth import get_token, save_token
from .base import ClipMetadata, UploadResult, Uploader
from .metadata import apply_platform_limits

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_PLATFORM = "youtube"
_RETRY_STATUS_CODES = frozenset({500, 502, 503, 504})
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0
_CHUNKSIZE = 256 * 1024  # 256 KB — minimum resumable chunk


def _credentials_from_token(token: dict) -> Credentials:
    return Credentials(
        token=token.get("token"),
        refresh_token=token.get("refresh_token"),
        token_uri=token.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token.get("client_id"),
        client_secret=token.get("client_secret"),
        scopes=token.get("scopes", _SCOPES),
    )


def _credentials_to_token(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or _SCOPES),
    }


class YouTubeUploader(Uploader):
    def __init__(self, token_dir: Path, client_secrets_file: Path | None = None) -> None:
        super().__init__(token_dir)
        self._client_secrets_file = client_secrets_file
        self._credentials: Credentials | None = None

    @property
    def platform_name(self) -> str:
        return _PLATFORM

    def authenticate(self) -> None:
        token = get_token(_PLATFORM, self.token_dir)
        if token is not None:
            logger.debug("Reusing saved YouTube credentials.")
            self._credentials = _credentials_from_token(token)
            return

        if self._client_secrets_file is None:
            raise RuntimeError(
                "No YouTube OAuth credentials found and no client_secrets_file provided. "
                "Run `youcut auth login --platform youtube` first."
            )

        logger.info("Starting YouTube OAuth flow — browser will open.")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self._client_secrets_file), scopes=_SCOPES
        )
        creds = flow.run_local_server(port=0)
        self._credentials = creds
        save_token(_PLATFORM, _credentials_to_token(creds), self.token_dir)
        logger.info("YouTube credentials saved.")

    def upload(
        self,
        video_path: Path,
        metadata: ClipMetadata,
        clip_index: int = 0,
        privacy: str = "public",
        thumbnail_path: Path | None = None,
        cut_mode: str = "youtube",
    ) -> UploadResult:
        if self._credentials is None:
            self.authenticate()

        meta = apply_platform_limits(metadata, _PLATFORM)

        description = meta.caption
        if cut_mode == "social":
            description = f"{description}\n\n#Shorts"

        body = {
            "snippet": {
                "title": meta.title,
                "description": description,
                "tags": meta.hashtags,
                "categoryId": "22",  # People & Blogs
            },
            "status": {
                "privacyStatus": privacy,
            },
        }

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/*",
            resumable=True,
            chunksize=_CHUNKSIZE,
        )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                youtube = build("youtube", "v3", credentials=self._credentials)
                request = youtube.videos().insert(
                    part="snippet,status",
                    body=body,
                    media_body=media,
                )

                response = None
                while response is None:
                    _, response = request.next_chunk()

                video_id = response["id"]
                url = f"https://youtu.be/{video_id}"
                logger.info("YouTube upload succeeded: %s", url)

                thumbnail_status: str | None = None
                if thumbnail_path is not None:
                    # thumbnails.set costs 50 quota units per call; limit is 10 uploads/channel/24h via API
                    try:
                        thumb_mime = "image/png" if thumbnail_path.suffix.lower() == ".png" else "image/jpeg"
                        thumb_media = MediaFileUpload(
                            str(thumbnail_path),
                            mimetype=thumb_mime,
                            resumable=False,
                        )
                        youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=thumb_media,
                            media_mime_type=thumb_mime,
                        ).execute()
                        logger.info("thumbnail_status=uploaded video_id=%s", video_id)
                        thumbnail_status = "uploaded"
                    except Exception as exc:
                        logger.warning("thumbnail_status=failed video_id=%s: %s", video_id, exc)
                        thumbnail_status = "failed"
                else:
                    thumbnail_status = "skipped"

                return UploadResult(
                    platform=_PLATFORM,
                    clip_index=clip_index,
                    status="success",
                    url=url,
                    video_id=video_id,
                    thumbnail_status=thumbnail_status,
                )

            except HttpError as exc:
                status_code = exc.resp.status
                error_content = exc.content.decode("utf-8", errors="replace") if exc.content else ""

                if status_code == 403 and "quotaExceeded" in error_content:
                    msg = (
                        "YouTube quota exceeded (~6 uploads/day). "
                        "Increase quota at https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas"
                    )
                    logger.error(msg)
                    return UploadResult(
                        platform=_PLATFORM,
                        clip_index=clip_index,
                        status="failed",
                        error=msg,
                    )

                if status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES:
                    logger.warning(
                        "YouTube HTTP %d on attempt %d/%d; retrying in %.1fs.",
                        status_code, attempt, _MAX_RETRIES, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    continue

                msg = f"YouTube HTTP {status_code}: {error_content}"
                logger.error(msg)
                return UploadResult(
                    platform=_PLATFORM,
                    clip_index=clip_index,
                    status="failed",
                    error=msg,
                )

            except socket.timeout as exc:
                msg = f"YouTube upload timed out: {exc}"
                logger.error(msg)
                return UploadResult(
                    platform=_PLATFORM,
                    clip_index=clip_index,
                    status="failed",
                    error=msg,
                )
