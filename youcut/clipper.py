import logging
import shutil
import subprocess
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import ViralClip

logger = logging.getLogger(__name__)

PADDING = 0.1

_BLACK_PAD_FILTER = (
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
)

_BLUR_BG_FILTER = (
    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
    "crop=1080:1920,boxblur=20:5[bg];"
    "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg não encontrado. Instale o FFmpeg e adicione ao PATH."
        )


def cut_clip(
    video_path: Path, clip: ViralClip, index: int, config: PipelineConfig
) -> Path:
    check_ffmpeg()

    start = max(0.0, clip.start_time - PADDING)
    end = clip.end_time + PADDING
    duration = end - start

    output_dir = config.output_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"clip_{index + 1:02d}.mp4"

    if config.blur_background:
        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-filter_complex", _BLUR_BG_FILTER,
            "-map", "[v]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", _BLACK_PAD_FILTER,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            str(output_path),
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou (código %d): %s", e.returncode, stderr)
        raise

    return output_path
