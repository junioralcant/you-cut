import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from youcut.models import ViralClip
from youcut.social_composer import (
    _build_bottom_crop_filter,
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
        patch("youcut.social_composer._probe_video_dimensions", return_value=(1080, 1920)),
        patch("youcut.social_composer._detect_face_y_norm", return_value=0.30),
        patch("youcut.social_composer.subprocess.run", return_value=subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)) as mock_run,
    ):
        output_path = compose_social_clip(clip_path, clip, config)

    cmd = mock_run.call_args[0][0]
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert output_path.name == "clip_social.mp4"
    assert "-filter_complex" in cmd
    assert "overlay=0:740" in filter_complex
    # Bottom crop must use a face-aware Y offset (not the default centre crop).
    assert "crop=1080:1180:0:" in filter_complex


def test_build_bottom_crop_filter_anchors_on_face_when_known():
    # Source 1080x1920 portrait → bottom panel 1080x880.
    # Face center at y_norm=0.30 → y_scaled=576. Target ratio 0.45 → desired
    # offset = 576 - 880*0.45 = 180, clamped to [0, 1040].
    result = _build_bottom_crop_filter(
        src_w=1080, src_h=1920, target_w=1080, target_h=880, face_y_norm=0.30,
    )
    assert result == "scale=1080:1920,crop=1080:880:0:180"


def test_build_bottom_crop_filter_falls_back_to_center_when_no_face():
    # No face detected → centre crop: offset = (1920-880)/2 = 520.
    result = _build_bottom_crop_filter(
        src_w=1080, src_h=1920, target_w=1080, target_h=880, face_y_norm=None,
    )
    assert result == "scale=1080:1920,crop=1080:880:0:520"


def test_build_bottom_crop_filter_clamps_offset_to_frame():
    # Face very near the bottom (y_norm=0.95 → y_scaled=1824) would push the
    # crop past the source height; must clamp to scaled_h - target_h = 1040.
    result = _build_bottom_crop_filter(
        src_w=1080, src_h=1920, target_w=1080, target_h=880, face_y_norm=0.95,
    )
    assert result == "scale=1080:1920,crop=1080:880:0:1040"


def test_build_bottom_crop_filter_face_near_top_clamps_to_zero():
    # Face right at the top (y_norm=0.05 → y_scaled=96) would want a negative
    # offset; must clamp to 0.
    result = _build_bottom_crop_filter(
        src_w=1080, src_h=1920, target_w=1080, target_h=880, face_y_norm=0.05,
    )
    assert result == "scale=1080:1920,crop=1080:880:0:0"


def test_build_bottom_crop_filter_landscape_source_uses_x_axis():
    # Source 1920x1080 (landscape) into bottom panel 1080x880: scale-to-cover
    # locks height (target_h), crops horizontally; y stays 0.
    result = _build_bottom_crop_filter(
        src_w=1920, src_h=1080, target_w=1080, target_h=880, face_y_norm=0.30,
    )
    # scaled_w = round(880 * 1920/1080) = 1564, x_offset = (1564-1080)/2 = 242
    assert result == "scale=1564:880,crop=1080:880:242:0"


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


# ────────────────────────────────────────────────────────────────────────────────
# Task 2.0 — propagação de config nos callers AI do social_composer
# ────────────────────────────────────────────────────────────────────────────────


def test_render_title_band_via_ai_propagates_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    ai_band = tmp_path / "ai_band.png"
    Image.new("RGB", (1080, 180), color=(255, 140, 0)).save(ai_band)
    config = PipelineConfig(
        social_layout_mode="speaker_bottom_ai_top",
        social_layout_title_color_mode="orange",
        openai_api_key="test-openai-key",
        cut_mode="social",
    )

    captured: dict = {}

    def fake_skill(*, prompt, reference_frames, openai_api_key, timeout, config=None):
        captured["config"] = config
        return ai_band.read_bytes()

    with (
        patch("youcut.social_composer._build_ai_clients", return_value=(object(), object())),
        patch("youcut.social_composer._resolve_openai_api_key", return_value="test-openai-key"),
        patch("youcut.social_composer._run_thumbnail_skill_script", side_effect=fake_skill),
    ):
        _render_title_band_image(
            "ALERTA",
            config,
            width=1080,
            height=180,
            suggested_color_mode="orange",
        )

    # `_render_title_band_image` faz `model_copy` para ajustar a paleta,
    # então o objeto chega como cópia preservando `cut_mode`.
    assert captured["config"] is not None
    assert getattr(captured["config"], "cut_mode", None) == "social"


def test_render_social_header_via_ai_propagates_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    from youcut.social_composer import _render_social_header_image_via_ai

    top_path = tmp_path / "top.png"
    fallback_path = tmp_path / "fallback.png"
    Image.new("RGB", (1080, 600), color=(0, 200, 200)).save(top_path)
    Image.new("RGB", (1080, 740), color=(0, 100, 100)).save(fallback_path)
    out_bytes_path = tmp_path / "ai_header.png"
    Image.new("RGB", (1080, 740), color=(255, 200, 0)).save(out_bytes_path)

    config = PipelineConfig(
        openai_api_key="test-openai-key",
        cut_mode="social",
    )
    captured: dict = {}

    def fake_skill(*, prompt, reference_frames, openai_api_key, timeout, config=None):
        captured["config"] = config
        return out_bytes_path.read_bytes()

    with (
        patch("youcut.social_composer._build_ai_clients", return_value=(object(), object())),
        patch("youcut.social_composer._resolve_openai_api_key", return_value="test-openai-key"),
        patch("youcut.social_composer._run_thumbnail_skill_script", side_effect=fake_skill),
    ):
        _render_social_header_image_via_ai(
            top_image_path=top_path,
            title="hello",
            config=config,
            width=1080,
            height=740,
            top_height=600,
            band_height=140,
            suggested_color_mode="yellow",
            fallback_path=fallback_path,
        )

    assert captured["config"] is config
    assert getattr(captured["config"], "cut_mode", None) == "social"
