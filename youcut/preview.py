import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from youcut.clipper import build_vertical_fill_filter
from youcut.config import PipelineConfig
from youcut.models import ViralClip

logger = logging.getLogger(__name__)


@dataclass
class PreviewArtifact:
    path: Path
    source_clip_index: int
    width: int
    height: int


def generate_clip_preview(
    video_path: Path,
    clip: ViralClip,
    index: int,
    config: PipelineConfig,
    width: int = 1080,
    height: int = 1920,
) -> Optional[PreviewArtifact]:
    output_dir = config.output_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"clip_{index + 1:02d}_preview.jpg"

    midpoint = (clip.start_time + clip.end_time) / 2
    vf = build_vertical_fill_filter(width, height)

    cmd = [
        "ffmpeg",
        "-ss", str(midpoint),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", vf,
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        logger.warning(
            "Falha ao gerar preview do clipe %d (código %d): %s",
            index + 1,
            e.returncode,
            stderr,
        )
        return None

    logger.info("Preview gerado: %s", output_path)
    return PreviewArtifact(
        path=output_path,
        source_clip_index=index,
        width=width,
        height=height,
    )
