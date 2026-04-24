"""Tests for bundled assets — font file existence and loadability."""

from pathlib import Path

import pytest
from PIL import ImageFont


ASSETS_DIR = Path(__file__).parent.parent / "youcut" / "assets"
FONT_PATH = ASSETS_DIR / "Roboto-Regular.ttf"


def test_font_file_exists():
    assert FONT_PATH.exists(), f"Font file not found at {FONT_PATH}"


def test_font_file_is_ttf():
    assert FONT_PATH.suffix.lower() == ".ttf"
    # TTF files start with bytes 0x00 0x01 0x00 0x00 or 'true'
    header = FONT_PATH.read_bytes()[:4]
    assert header in (b"\x00\x01\x00\x00", b"true", b"OTTO"), (
        f"File does not appear to be a valid TTF (header: {header!r})"
    )


def test_font_loadable_via_truetype():
    font = ImageFont.truetype(str(FONT_PATH), size=32)
    assert font is not None


def test_font_loadable_various_sizes():
    for size in (16, 32, 48, 64):
        font = ImageFont.truetype(str(FONT_PATH), size=size)
        assert font is not None, f"Failed to load font at size {size}"
