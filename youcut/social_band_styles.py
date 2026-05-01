"""Stylistic presets for the social-clip title band / header.

Each preset bundles a font + visual treatment + an AI prompt directive, so the
same logical "yellow/orange editorial" tarja can render in radically different
visual personalities across clips. Preset selection is deterministic by seed
(clip stem), so re-running yields the same look per clip.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ASSETS_DIR = Path(__file__).parent / "assets"
_FONTS_DIR = _ASSETS_DIR / "fonts"
_LEGACY_FONT = _ASSETS_DIR / "Roboto-Regular.ttf"


BgTreatment = Literal[
    "gradient_smooth",
    "solid_flat",
    "paper_grain",
    "burst_rays",
    "split_blocks",
]

AccentStyle = Literal[
    "chevrons",
    "stripe_top_bottom",
    "side_block",
    "stamp_border",
    "torn_edges",
    "marker_underline",
    "burst_polygon",
    "corner_squares",
    "none",
]

TextShadow = Literal["soft", "hard_offset", "no_shadow"]

BorderStyle = Literal[
    "rounded_rect",
    "sharp_rect",
    "double_inset",
    "torn_paper",
    "polygon_burst",
    "no_border",
]


@dataclass(frozen=True)
class BandStyle:
    name: str
    font_path: Path
    font_size_start: int
    font_size_min: int
    bg_treatment: BgTreatment
    accent_style: AccentStyle
    text_shadow: TextShadow
    border_style: BorderStyle
    letter_spacing_px: int
    ai_directive: str


PRESETS: tuple[BandStyle, ...] = (
    BandStyle(
        name="editorial_clean",
        font_path=_LEGACY_FONT,
        font_size_start=84,
        font_size_min=42,
        bg_treatment="gradient_smooth",
        accent_style="chevrons",
        text_shadow="soft",
        border_style="rounded_rect",
        letter_spacing_px=0,
        ai_directive=(
            "Style mood: premium editorial band — smooth vertical gradient background, "
            "soft rounded inner border inset from the edges, two pairs of solid triangular "
            "chevron accents flanking the centered title, clean medium-weight sans-serif uppercase, "
            "subtle soft shadow on text. Polished, restrained, magazine-like."
        ),
    ),
    BandStyle(
        name="tabloid_bold",
        font_path=_FONTS_DIR / "Anton-Regular.ttf",
        font_size_start=160,
        font_size_min=70,
        bg_treatment="solid_flat",
        accent_style="stripe_top_bottom",
        text_shadow="hard_offset",
        border_style="sharp_rect",
        letter_spacing_px=-2,
        ai_directive=(
            "Style mood: tabloid manchete — solid flat background with NO gradient; "
            "title in a massive ultra-condensed bold all-caps display face filling the band edge to edge; "
            "thick hard offset drop shadow on the title (no soft blur); "
            "thick solid horizontal stripes hugging the very top and very bottom edges of the band. "
            "Sharp corners, no rounded edges, aggressive loud headline energy."
        ),
    ),
    BandStyle(
        name="news_ticker",
        font_path=_FONTS_DIR / "Oswald-Bold.ttf",
        font_size_start=72,
        font_size_min=44,
        bg_treatment="solid_flat",
        accent_style="side_block",
        text_shadow="no_shadow",
        border_style="no_border",
        letter_spacing_px=2,
        ai_directive=(
            "Style mood: broadcast news ticker — clean horizontal bar; on the left side a small "
            "contrasting darker accent block (about 18 percent of the width) holding a single geometric "
            "mark (a filled circle or square); title sits to the right of that block in a tall "
            "medium-condensed bold sans-serif uppercase with letter-spacing; absolutely no shadow on text; "
            "sharp rectangular layout, professional newsroom chyron feel."
        ),
    ),
    BandStyle(
        name="magazine_cover",
        font_path=_FONTS_DIR / "Montserrat-Black.ttf",
        font_size_start=98,
        font_size_min=52,
        bg_treatment="paper_grain",
        accent_style="stamp_border",
        text_shadow="no_shadow",
        border_style="double_inset",
        letter_spacing_px=4,
        ai_directive=(
            "Style mood: vintage magazine cover badge — solid background with very subtle paper texture grain; "
            "two thin rectangular outlined frames inset from the edges (one outer, one inner just inside it); "
            "title in heavy geometric black sans-serif uppercase centered inside the inner frame with generous "
            "letter-spacing; no shadow on text; evokes the title-strip of a premium print magazine cover."
        ),
    ),
    BandStyle(
        name="sticker_punch",
        font_path=_FONTS_DIR / "ArchivoBlack-Regular.ttf",
        font_size_start=96,
        font_size_min=52,
        bg_treatment="solid_flat",
        accent_style="torn_edges",
        text_shadow="no_shadow",
        border_style="torn_paper",
        letter_spacing_px=0,
        ai_directive=(
            "Style mood: pasted sticker — the title block looks like a printed adhesive sticker stuck onto the canvas; "
            "slight rotation (3 to 5 degrees off horizontal); thick chunky boxy black sans-serif typeface in uppercase; "
            "hard chunky offset drop shadow underneath the sticker giving real cutout depth; "
            "the edges of the sticker shape are slightly irregular / ripped / torn paper, not perfectly straight; "
            "playful, punchy, hand-applied energy."
        ),
    ),
    BandStyle(
        name="marker_scrawl",
        font_path=_FONTS_DIR / "PermanentMarker-Regular.ttf",
        font_size_start=110,
        font_size_min=58,
        bg_treatment="solid_flat",
        accent_style="marker_underline",
        text_shadow="no_shadow",
        border_style="no_border",
        letter_spacing_px=0,
        ai_directive=(
            "Style mood: hand-marker scrawl — solid flat background; title written in chunky permanent-marker "
            "handwriting style with slightly irregular weight per stroke; one rough hand-drawn underline scribble "
            "under the title (or a loose circle around a key word); raw, spontaneous, personal energy; "
            "absolutely no rectangular borders, no shadows, no decorative chevrons — feels written by hand on a wall."
        ),
    ),
    BandStyle(
        name="burst_attention",
        font_path=_FONTS_DIR / "Anton-Regular.ttf",
        font_size_start=130,
        font_size_min=64,
        bg_treatment="burst_rays",
        accent_style="burst_polygon",
        text_shadow="hard_offset",
        border_style="polygon_burst",
        letter_spacing_px=-1,
        ai_directive=(
            "Style mood: comic-book sunburst — title sits inside an irregular jagged starburst polygon shape "
            "with about 10 sharp points, like a Pow! callout; radial diverging lines emanate from behind the polygon "
            "filling the band background; ultra-condensed heavy bold sans-serif uppercase title rotated slightly off-axis; "
            "hard offset drop shadow on text; dramatic comic-book attention-grabbing energy."
        ),
    ),
    BandStyle(
        name="split_blocks",
        font_path=_FONTS_DIR / "BebasNeue-Regular.ttf",
        font_size_start=126,
        font_size_min=66,
        bg_treatment="split_blocks",
        accent_style="corner_squares",
        text_shadow="no_shadow",
        border_style="no_border",
        letter_spacing_px=3,
        ai_directive=(
            "Style mood: Bauhaus split-blocks — background divided into two horizontal color blocks "
            "(yellow on the top half, orange on the bottom half, or vice versa, both staying inside the editorial palette); "
            "title in a tall narrow display sans-serif uppercase laid across the band so its baseline crosses the color boundary; "
            "small geometric primitive shapes (filled squares, dots, or circles) decorating two of the corners; "
            "no shadows, no borders, modernist editorial."
        ),
    ),
)


def select_band_style(seed: str | int) -> BandStyle:
    """Pick a deterministic preset based on *seed* (e.g. clip path stem).

    Same seed -> same preset across runs. Falls back to the default
    ``editorial_clean`` if any unexpected error happens.
    """
    if not PRESETS:
        raise RuntimeError("PRESETS is empty")
    if isinstance(seed, int):
        idx = seed % len(PRESETS)
    else:
        digest = hashlib.md5(str(seed).encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(PRESETS)
    return PRESETS[idx]


def get_style_by_name(name: str) -> BandStyle | None:
    for preset in PRESETS:
        if preset.name == name:
            return preset
    return None
