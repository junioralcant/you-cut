import io
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from youcut.models import ViralClip
from youcut.thumbnail_generator import _build_thumbnail_result, generate_thumbnail


def _generate_synthetic_video(output: Path, duration: float = 10.0) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:size=1280x720:rate=25:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output


def _make_clip(thumbnail_text: str = "") -> ViralClip:
    return ViralClip(
        title="Momento Viral Incrível",
        reason="High energy",
        viral_score=9.0,
        start_time=0.0,
        end_time=10.0,
        description="Descrição do clipe.",
        hashtags=["#youtube"],
        thumbnail_idea="Host explaining excitedly",
        thumbnail_text=thumbnail_text,
        cut_mode="youtube",
    )


def _make_png(width: int = 1536, height: int = 1024, color: tuple[int, int, int] = (0, 90, 255)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_video(tmp_path):
    video = tmp_path / "synthetic.mp4"
    _generate_synthetic_video(video, duration=10.0)
    yield video
    video.unlink(missing_ok=True)


@pytest.mark.integration
def test_generate_thumbnail_local_path_produces_valid_png(synthetic_video, tmp_path):
    clip = _make_clip("")
    output_path = tmp_path / "local.png"

    result = _build_thumbnail_result(clip, synthetic_video, output_path, None)

    assert result.selection_method == "local"
    assert result.generation_method == "local"
    with Image.open(output_path) as img:
        assert img.size == (1280, 720)
        assert img.format == "PNG"


@pytest.mark.integration
def test_generate_thumbnail_ai_path_produces_valid_png(synthetic_video, tmp_path):
    clip = _make_clip("")
    output_path = tmp_path / "ai.png"
    anth_response = SimpleNamespace(content=[SimpleNamespace(text='{"selected_index": 0, "reason": "bright"}')])
    anthropic_client = MagicMock()
    anthropic_client.with_options.return_value = anthropic_client
    anthropic_client.messages.create.return_value = anth_response

    openai_client = SimpleNamespace(api_key="test-key")

    with (
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(anthropic_client, openai_client)),
        patch("youcut.thumbnail_generator._run_thumbnail_skill_script", return_value=_make_png(1536, 864)),
    ):
        result = _build_thumbnail_result(clip, synthetic_video, output_path, SimpleNamespace())

    assert result.selection_method == "ai"
    assert result.generation_method == "ai"
    with Image.open(output_path) as img:
        assert img.size == (1280, 720)


@pytest.mark.integration
def test_generate_thumbnail_ai_selection_fallback(synthetic_video, tmp_path):
    clip = _make_clip("")
    output_path = tmp_path / "selection_fallback.png"
    anthropic_client = MagicMock()
    anthropic_client.with_options.return_value = anthropic_client
    anthropic_client.messages.create.side_effect = RuntimeError("boom")

    openai_client = SimpleNamespace(api_key="test-key")

    with (
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(anthropic_client, openai_client)),
        patch("youcut.thumbnail_generator._run_thumbnail_skill_script", return_value=_make_png(1536, 864)),
    ):
        result = _build_thumbnail_result(clip, synthetic_video, output_path, SimpleNamespace())

    assert result.selection_method == "local"
    assert result.generation_method == "ai"


@pytest.mark.integration
def test_generate_thumbnail_ai_generation_fallback(synthetic_video, tmp_path):
    clip = _make_clip("")
    output_path = tmp_path / "generation_fallback.png"
    anth_response = SimpleNamespace(content=[SimpleNamespace(text='{"selected_index": 0, "reason": "bright"}')])
    anthropic_client = MagicMock()
    anthropic_client.with_options.return_value = anthropic_client
    anthropic_client.messages.create.return_value = anth_response

    openai_client = SimpleNamespace(api_key="test-key")

    with (
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(anthropic_client, openai_client)),
        patch("youcut.thumbnail_generator._run_thumbnail_skill_script", side_effect=RuntimeError("429")),
    ):
        result = _build_thumbnail_result(clip, synthetic_video, output_path, SimpleNamespace())

    assert result.selection_method == "ai"
    assert result.generation_method == "local"
    with Image.open(output_path) as img:
        assert img.size == (1280, 720)


@pytest.mark.integration
def test_generate_thumbnail_no_text_by_default(synthetic_video, tmp_path):
    clip = _make_clip("")
    output_dir = tmp_path / "output"

    result = generate_thumbnail(clip, output_dir, clip_index=1, clip_path=synthetic_video)

    with Image.open(result) as img:
        assert img.size == (1280, 720)
        assert img.getpixel((0, 0)) == img.getpixel((640, 360)) == img.getpixel((1279, 719))
