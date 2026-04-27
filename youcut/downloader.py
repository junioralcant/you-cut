from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

from youcut.url_utils import normalize_video_url
from youcut.yt_dlp_auth import YtDlpAuthConfig, apply_yt_dlp_auth, append_yt_auth_hint


class VideoDownloadError(Exception):
    pass


def download_video(
    source: str,
    output_dir: Path,
    auth_config: YtDlpAuthConfig | None = None,
) -> Path:
    if source.startswith("http://") or source.startswith("https://"):
        return _download_from_url(source, output_dir, auth_config)
    return _resolve_local_file(source)


def _download_from_url(
    url: str,
    output_dir: Path,
    auth_config: YtDlpAuthConfig | None,
) -> Path:
    url = normalize_video_url(url)
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    apply_yt_dlp_auth(ydl_opts, auth_config)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            path = Path(filename)
            if not path.suffix or path.suffix not in (".mp4", ".mkv", ".webm", ".mov"):
                path = path.with_suffix(".mp4")
            return path
    except DownloadError as e:
        message = append_yt_auth_hint(str(e))
        raise VideoDownloadError(f"Falha ao baixar o vídeo: {message}") from e


def _resolve_local_file(source: str) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source}")
    return path
