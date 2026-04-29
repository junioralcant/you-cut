import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from youcut.models import ViralClip
from youcut.social_composer import (
    _render_title_band_image,
    compose_social_clip,
    resolve_title_band_colors,
)
from youcut.thumbnail_generator import generate_social_top_image


def _make_clip() -> ViralClip:
    return ViralClip(
        title="Crise no debate",
        reason="Gancho forte",
        viral_score=9.0,
        start_time=0.0,
        end_time=30.0,
        description="Descrição",
        hashtags=["#teste"],
        thumbnail_idea="Debate acalorado em estúdio",
        thumbnail_text="MOMENTO IMPACTANTE",
        social_hook_title="ALERTA TOTAL",
        social_image_prompt="Cena editorial de tensão política sem texto",
        social_visual_style="claro e vivo",
    )


def test_resolve_title_band_colors_default_is_yellow(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig()
    assert resolve_title_band_colors(config) == ("#F4C400", "#111111")


def test_resolve_title_band_colors_orange(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(social_layout_title_color_mode="orange")
    assert resolve_title_band_colors(config) == ("#FF8A00", "#111111")


def test_compose_social_clip_builds_social_output(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake-video")
    top_image = tmp_path / "social_images" / "clip_top.png"
    top_image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1080, 860), color=(240, 200, 0)).save(top_image)

    config = PipelineConfig(
        social_layout_mode="speaker_bottom_ai_top",
        social_layout_title_color_mode="engagement_default",
    )
    clip = _make_clip()

    with (
        patch("youcut.social_composer._render_social_header_image", return_value=top_image),
        patch("youcut.social_composer.subprocess.run", return_value=subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)) as mock_run,
    ):
        output_path = compose_social_clip(clip_path, clip, config)

    cmd = mock_run.call_args[0][0]
    assert output_path.name == "clip_social.mp4"
    assert "-filter_complex" in cmd
    assert "overlay=0:1040" in cmd[cmd.index("-filter_complex") + 1]


def test_render_title_band_image_prefers_ai_output(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    ai_band = tmp_path / "ai_band.png"
    Image.new("RGB", (1080, 180), color=(255, 140, 0)).save(ai_band)
    config = PipelineConfig(
        social_layout_mode="speaker_bottom_ai_top",
        social_layout_title_color_mode="orange",
        openai_api_key="test-openai-key",
    )

    with (
        patch("youcut.social_composer._build_ai_clients", return_value=(object(), object())),
        patch("youcut.social_composer._resolve_openai_api_key", return_value="test-openai-key"),
        patch("youcut.social_composer._run_thumbnail_skill_script", return_value=ai_band.read_bytes()),
    ):
        output = _render_title_band_image(
            "ESTELIONATO ELEITORAL",
            config,
            width=1080,
            height=180,
            suggested_color_mode="orange",
        )

    assert output.exists()
    with Image.open(output) as rendered:
        assert rendered.size == (1080, 180)


def test_generate_social_top_image_falls_back_to_local_frame(tmp_path):
    clip = _make_clip()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake-video")
    frame_bytes_path = tmp_path / "frame.png"
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(frame_bytes_path)
    frame_bytes = frame_bytes_path.read_bytes()

    with (
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(None, None)),
        patch("youcut.thumbnail_generator._extract_frames_candidates", return_value=[(1.0, frame_bytes)]),
        patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(1.0, frame_bytes, 0.8)),
    ):
        output = generate_social_top_image(clip, tmp_path, clip_path, config=SimpleNamespace(social_layout_image_provider="local", social_layout_top_image_height=860))

    assert output.exists()
