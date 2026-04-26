import yt_dlp

from youcut.models import VideoMetadata
from youcut.url_utils import normalize_video_url


class VideoMetadataError(Exception):
    pass


def fetch_metadata(url: str) -> VideoMetadata:
    normalized_url = normalize_video_url(url)
    ydl_opts = {"quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=False)
    except (yt_dlp.utils.DownloadError, OSError) as e:
        raise VideoMetadataError(f"Could not access video: {e}") from e

    if not info:
        raise VideoMetadataError(f"No metadata returned for URL: {normalized_url}")

    title = info.get("title")
    duration = info.get("duration")

    if not title:
        raise VideoMetadataError("Video metadata is missing title")
    if duration is None:
        raise VideoMetadataError("Video metadata is missing duration")

    return VideoMetadata(title=title, duration_seconds=float(duration), url=normalized_url)
