import yt_dlp

from youcut.models import VideoMetadata
from youcut.url_utils import normalize_video_url
from youcut.yt_dlp_auth import YtDlpAuthConfig, apply_yt_dlp_auth, append_yt_auth_hint


class VideoMetadataError(Exception):
    pass


def fetch_metadata(url: str, auth_config: YtDlpAuthConfig | None = None) -> VideoMetadata:
    normalized_url = normalize_video_url(url)
    ydl_opts = {"quiet": True}
    apply_yt_dlp_auth(ydl_opts, auth_config)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=False)
    except (yt_dlp.utils.DownloadError, OSError) as e:
        message = append_yt_auth_hint(str(e))
        raise VideoMetadataError(f"Could not access video: {message}") from e

    if not info:
        raise VideoMetadataError(f"No metadata returned for URL: {normalized_url}")

    title = info.get("title")
    duration = info.get("duration")

    if not title:
        raise VideoMetadataError("Video metadata is missing title")
    if duration is None:
        raise VideoMetadataError("Video metadata is missing duration")

    return VideoMetadata(title=title, duration_seconds=float(duration), url=normalized_url)
