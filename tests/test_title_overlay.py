"""Unit tests for youcut/title_overlay.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from youcut.title_overlay import (
    add_title_overlay,
    compute_text_color,
    extract_dominant_color,
    get_video_dimensions,
    render_card_image,
)


# ---------------------------------------------------------------------------
# compute_text_color
# ---------------------------------------------------------------------------

def test_compute_text_color_dark_background_returns_white():
    result = compute_text_color((20, 20, 20))
    assert result == (255, 255, 255)


def test_compute_text_color_light_background_returns_black():
    result = compute_text_color((230, 230, 230))
    assert result == (0, 0, 0)


def test_compute_text_color_contrast_always_sufficient():
    """Chosen color must always achieve contrast >= 4.5:1."""
    from youcut.title_overlay import _relative_luminance

    test_backgrounds = [
        (0, 0, 0),
        (255, 255, 255),
        (128, 64, 192),
        (50, 150, 80),
        (200, 100, 50),
    ]
    for bg in test_backgrounds:
        text = compute_text_color(bg)
        bg_lum = _relative_luminance(bg)
        txt_lum = _relative_luminance(text)
        lighter = max(bg_lum, txt_lum)
        darker = min(bg_lum, txt_lum)
        contrast = (lighter + 0.05) / (darker + 0.05)
        assert contrast >= 4.5, f"Contrast {contrast:.2f} < 4.5 for bg={bg}, text={text}"


# ---------------------------------------------------------------------------
# extract_dominant_color
# ---------------------------------------------------------------------------

def _make_fake_image() -> MagicMock:
    """Return a mock that behaves like a small PIL image after quantize."""
    palette = [255, 0, 0] + [0] * (768 - 3)
    quantized = MagicMock()
    quantized.getpalette.return_value = palette

    img_mock = MagicMock()
    img_mock.resize.return_value = img_mock
    img_mock.quantize.return_value = quantized
    return img_mock


def test_extract_dominant_color_returns_rgb_tuple(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()

    with (
        patch("youcut.title_overlay.subprocess.run") as mock_run,
        patch("youcut.title_overlay.Image.open") as mock_open,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        img_mock = _make_fake_image()
        img_mock.convert.return_value = img_mock
        mock_open.return_value = img_mock

        result = extract_dominant_color(fake_video)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(c, int) for c in result)


# ---------------------------------------------------------------------------
# render_card_image
# ---------------------------------------------------------------------------

def test_render_card_image_creates_png_with_rgba_mode(tmp_path):
    font_path = Path(__file__).parent.parent / "youcut" / "assets" / "Roboto-Regular.ttf"
    card_path = render_card_image(
        title="Título do Clipe",
        bg_color=(30, 30, 200),
        text_color=(255, 255, 255),
        video_width=1080,
        font_path=font_path,
    )
    try:
        assert card_path.exists()
        img = Image.open(card_path)
        assert img.mode == "RGBA"
        assert img.width > 0
        assert img.height > 0
    finally:
        card_path.unlink(missing_ok=True)


def test_render_card_image_fallback_font(tmp_path):
    """When font_path doesn't exist, falls back to default font without raising."""
    missing_font = tmp_path / "nonexistent.ttf"
    card_path = render_card_image(
        title="Fallback",
        bg_color=(100, 100, 100),
        text_color=(255, 255, 255),
        video_width=1080,
        font_path=missing_font,
    )
    try:
        assert card_path.exists()
    finally:
        card_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# get_video_dimensions
# ---------------------------------------------------------------------------

def test_get_video_dimensions_parses_ffprobe_json(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()

    ffprobe_output = json.dumps({
        "streams": [{"width": 1080, "height": 1920}]
    })

    with patch("youcut.title_overlay.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=ffprobe_output, returncode=0)
        w, h = get_video_dimensions(fake_video)

    assert w == 1080
    assert h == 1920


# ---------------------------------------------------------------------------
# add_title_overlay — guard conditions
# ---------------------------------------------------------------------------

def _make_config(title_overlay: bool = True):
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.title_overlay = title_overlay
    return cfg


def _make_clip(title: str = "Meu Título"):
    from unittest.mock import MagicMock
    clip = MagicMock()
    clip.title = title
    return clip


def test_add_title_overlay_disabled_no_subprocess(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()

    with patch("youcut.title_overlay.subprocess.run") as mock_run:
        result = add_title_overlay(fake_video, _make_clip(), _make_config(title_overlay=False))
        mock_run.assert_not_called()

    assert result == fake_video


def test_add_title_overlay_empty_title_returns_unchanged(tmp_path, caplog):
    import logging
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()

    with patch("youcut.title_overlay.subprocess.run") as mock_run:
        with caplog.at_level(logging.WARNING, logger="youcut.title_overlay"):
            result = add_title_overlay(fake_video, _make_clip(title="   "), _make_config())
        mock_run.assert_not_called()

    assert result == fake_video
    assert any("vazio" in record.message.lower() or "overlay ignorado" in record.message.lower()
               for record in caplog.records)
