"""Testes de integração para generate_thumbnail() com vídeo sintético.

Usa FFmpeg para gerar um .mp4 de cor sólida e verifica que a thumbnail resultante
atende às especificações do YouTube: 1280×720 px, PNG, ≤ 2 MB.
"""

import subprocess
from pathlib import Path

import pytest

from youcut.models import ViralClip


def _generate_synthetic_video(output: Path, duration: float = 10.0) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:size=1280x720:rate=25:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output


def _make_clip() -> ViralClip:
    return ViralClip(
        title="Momento Viral Incrível",
        reason="High energy",
        viral_score=9.0,
        start_time=0.0,
        end_time=10.0,
        description="Descrição do clipe.",
        hashtags=["#youtube"],
        thumbnail_idea="Host explaining excitedly",
        thumbnail_text="MOMENTO IMPACTANTE",
        cut_mode="youtube",
    )


@pytest.fixture
def synthetic_video(tmp_path):
    video = tmp_path / "synthetic.mp4"
    _generate_synthetic_video(video, duration=10.0)
    yield video
    video.unlink(missing_ok=True)


@pytest.mark.integration
def test_generate_thumbnail_end_to_end(synthetic_video, tmp_path):
    from PIL import Image

    from youcut.thumbnail_generator import generate_thumbnail

    clip = _make_clip()
    output_dir = tmp_path / "output"
    result = generate_thumbnail(clip, output_dir, clip_index=1, clip_path=synthetic_video)

    assert result.exists(), f"Thumbnail não foi criada em {result}"
    assert result.suffix == ".png"
    assert result.name == "clip_01.png"

    with Image.open(result) as img:
        assert img.size == (1280, 720), f"Dimensões incorretas: {img.size}"
        assert img.format == "PNG"

    size_bytes = result.stat().st_size
    assert size_bytes <= 2 * 1024 * 1024, f"Thumbnail muito grande: {size_bytes} bytes"


@pytest.mark.integration
def test_generate_thumbnail_output_path(synthetic_video, tmp_path):
    from youcut.thumbnail_generator import generate_thumbnail

    clip = _make_clip()
    output_dir = tmp_path / "output"
    result = generate_thumbnail(clip, output_dir, clip_index=3, clip_path=synthetic_video)

    assert result == output_dir / "thumbnails" / "clip_03.png"


@pytest.mark.integration
def test_generate_thumbnail_creates_thumbnails_dir(synthetic_video, tmp_path):
    from youcut.thumbnail_generator import generate_thumbnail

    clip = _make_clip()
    output_dir = tmp_path / "new_output"
    assert not output_dir.exists()

    generate_thumbnail(clip, output_dir, clip_index=0, clip_path=synthetic_video)

    assert (output_dir / "thumbnails").is_dir()
