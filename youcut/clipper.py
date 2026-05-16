import logging
import os
import shutil
import subprocess
from pathlib import Path

from youcut.color_filter import get_filter_chain
from youcut.config import PipelineConfig
from youcut.decoupage import remove_silences
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
    """Corta o clipe e aplica decoupage. NÃO queima legendas — isso é
    responsabilidade do orquestrador em `cli.py`, executado **após** o
    tratamento visual no caminho social/classic (ver Tech Spec da feature
    'Tratamento Visual Padrão dos Cortes Sociais').
    """
    check_ffmpeg()

    output_dir = config.output_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"clip_{index + 1:02d}.mp4"

    if clip.cut_mode == "youtube":
        cmd = _build_youtube_cmd(video_path, clip, output_path)
    elif clip.cut_mode == "social" and config.social_layout_mode == "speaker_bottom_ai_top":
        cmd = _build_social_raw_cmd(video_path, clip, config, output_path)
    else:
        cmd = _build_social_cmd(video_path, clip, config, output_path)

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou (código %d): %s", e.returncode, stderr)
        raise

    if config.decoupage_enabled:
        try:
            remove_silences(
                output_path,
                noise_db=config.decoupage_noise_db,
                min_silence_gap=config.decoupage_min_silence_gap,
                keep_padding=config.decoupage_keep_padding,
            )
        except Exception as exc:
            logger.warning(
                "Decoupage falhou em %s: %s — mantendo clipe original.",
                output_path.name, exc,
            )

    return output_path


def _build_social_raw_cmd(
    video_path: Path, clip: ViralClip, config: PipelineConfig, output_path: Path,
) -> list[str]:
    """Time-trim the source preserving original aspect ratio.

    The editorial layout (speaker_bottom_ai_top) defers framing to a
    post-cut face-aware step; pre-cropping here would discard the horizontal
    context needed to centre on a single speaker or zoom out for two.

    Quando ``social_filter_preset`` está ativo, aplicamos o color grade
    aqui para que o composer (que monta o canvas final) já receba o speaker
    com o look desejado.
    """
    start = max(0.0, clip.start_time - PADDING)
    end = clip.end_time + PADDING
    duration = end - start
    color_chain = get_filter_chain(config.social_filter_preset)
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
    ]
    if color_chain:
        cmd.extend(["-vf", color_chain])
    cmd.extend([
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output_path),
    ])
    return cmd


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

    color_chain = get_filter_chain(config.social_filter_preset)
    color_suffix = f",{color_chain}" if color_chain else ""

    # Face-zoom opcional via env MOTIVACAO_FACE_ZOOM (ex.: 1.6).
    # Aplicado depois do crop 9:16 e antes da queima de legendas em ASS
    # com coordenadas absolutas 1080×1920, então a legenda permanece em
    # tamanho original.
    try:
        face_zoom = float(os.environ.get("MOTIVACAO_FACE_ZOOM", "1.0"))
    except ValueError:
        face_zoom = 1.0
    zoom_suffix = ""
    if face_zoom > 1.001:
        crop_w = int(1080 / face_zoom)
        crop_h = int(1920 / face_zoom)
        # Crop centered horizontally; bias upward to keep face framed.
        crop_x = (1080 - crop_w) // 2
        crop_y = max(0, (1920 - crop_h) // 3)
        zoom_suffix = f",crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920"
        logger.info("Face zoom %.2fx aplicado (crop %dx%d @ %d,%d).", face_zoom, crop_w, crop_h, crop_x, crop_y)

    if use_blur:
        filter_complex = (
            _BLUR_BG_FILTER.replace(
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]",
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2{color_suffix}{zoom_suffix}[v]",
            )
        )
        return [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-filter_complex", filter_complex,
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
        "-vf", build_vertical_fill_filter() + color_suffix + zoom_suffix,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output_path),
    ]
