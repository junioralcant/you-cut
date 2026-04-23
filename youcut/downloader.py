from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError


class VideoDownloadError(Exception):
    pass


def download_video(source: str, output_dir: Path) -> Path:
    if source.startswith("http://") or source.startswith("https://"):
        return _download_from_url(source, output_dir)
    return _resolve_local_file(source)


def _download_from_url(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            path = Path(filename)
            if not path.suffix or path.suffix not in (".mp4", ".mkv", ".webm", ".mov"):
                path = path.with_suffix(".mp4")
            return path
    except DownloadError as e:
        raise VideoDownloadError(f"Falha ao baixar o vídeo: {e}") from e


def _resolve_local_file(source: str) -> Path:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source}")
    return path
