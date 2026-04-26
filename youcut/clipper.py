import logging
import shutil
import subprocess
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import ViralClip

logger = logging.getLogger(__name__)

PADDING = 0.1

_BLUR_BG_FILTER = (
    "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
    "crop=1080:1920,boxblur=20:5[bg];"
    "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
)


def build_vertical_fill_filter(width: int = 1080, height: int = 1920) -> str:
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg não encontrado. Instale o FFmpeg e adicione ao PATH."
        )


def cut_clip(
    video_path: Path, clip: ViralClip, index: int, config: PipelineConfig
) -> Path:
    check_ffmpeg()

    output_dir = config.output_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"clip_{index + 1:02d}.mp4"

    if clip.cut_mode == "youtube":
        cmd = _build_youtube_cmd(video_path, clip, output_path)
    else:
        cmd = _build_social_cmd(video_path, clip, config, output_path)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou (código %d): %s", e.returncode, stderr)
        raise

    return output_path


def _build_youtube_cmd(video_path: Path, clip: ViralClip, output_path: Path) -> list[str]:
    duration = clip.end_time - clip.start_time
    return [
        "ffmpeg",
        "-ss", str(clip.start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        "-y",
        str(output_path),
    ]


def _build_social_cmd(
    video_path: Path, clip: ViralClip, config: PipelineConfig, output_path: Path
) -> list[str]:
    start = max(0.0, clip.start_time - PADDING)
    end = clip.end_time + PADDING
    duration = end - start

    use_blur = config.blur_background or config.vertical_fill_mode == "blur_background"
    if config.blur_background:
        logger.warning(
            "O campo 'blur_background' está depreciado. Use vertical_fill_mode='blur_background'."
        )
    strategy = "blur_background" if use_blur else "fill_crop"
    logger.info("Estratégia de enquadramento vertical: %s", strategy)
    if use_blur:
        return [
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
    return [
        "ffmpeg",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", build_vertical_fill_filter(),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output_path),
    ]
