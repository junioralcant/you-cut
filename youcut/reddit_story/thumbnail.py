"""Gera thumbnails 1280×720 estilo canais Reddit revenge.

Pipeline por variante: Flux Schnell base (composição com negative space à
esquerda, sem texto) + Pillow overlay (Anton condensed bold branco + amarelo,
banner vermelho top-right, tag r/subreddit bottom-left).

API:
- ``generate_thumbnail(...)`` — 1 thumbnail (backward compat)
- ``generate_thumbnail_set(variants, ...)`` — N thumbnails (cada uma com
  scene_brief + headlines próprios). Salva versionado em ``thumbnails/``.

Sempre preserva arquivos; nunca sobrescreve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from youcut.reddit_story.providers import flux_schnell_image


THUMB_W = 1280
THUMB_H = 720


_FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_FONT_HEADLINE = _FONT_DIR / "Anton-Regular.ttf"
_FONT_BANNER = _FONT_DIR / "Inter-Bold.ttf"


_DEFAULT_BASE_PROMPT = (
    "Cinematic dramatic thumbnail composition for a YouTube revenge story video. "
    "<<SCENE>> "
    "Strong negative space and darker vignette on the LEFT THIRD of the frame "
    "(for text overlay later). Composition pushes visual elements to the RIGHT "
    "TWO-THIRDS. Dramatic chiaroscuro lighting, deep saturated palette, 35mm "
    "cinematic, shallow depth of field. Pure visual: ABSOLUTELY NO TEXT, NO "
    "LETTERS, NO WORDS, NO LOGOS, NO SIGNS WITH WRITING. Horizontal 16:9 framing."
)


@dataclass
class ThumbnailRender:
    """1 thumb gerada: path do PNG final + path do base do Flux."""

    name: str
    path: Path
    base_path: Path


def _draw_with_stroke(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple | str,
    stroke_width: int,
    stroke_fill: tuple | str = "black",
) -> None:
    # Drop shadow (offset 4,4)
    draw.text(
        (pos[0] + 4, pos[1] + 4),
        text,
        font=font,
        fill=(0, 0, 0, 180),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 180),
    )
    # Main text com stroke
    draw.text(
        pos, text, font=font, fill=fill,
        stroke_width=stroke_width, stroke_fill=stroke_fill,
    )


def _compose_overlay(
    base_path: Path,
    out_path: Path,
    *,
    headline_line1: str,
    headline_line2: str,
    accent_banner: str,
    subreddit_tag: str,
) -> None:
    """Aplica gradient + headlines + banner + tag em cima do base."""
    img = Image.open(base_path).convert("RGBA")
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    # Gradient escurecendo o terço esquerdo
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    grad_w = int(THUMB_W * 0.55)
    for x in range(grad_w):
        alpha = int(180 * (1 - x / grad_w))
        od.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    # Headline 2 linhas
    headline_font = ImageFont.truetype(str(_FONT_HEADLINE), 130)
    _draw_with_stroke(
        draw, (50, 220), headline_line1, headline_font,
        fill="white", stroke_width=8,
    )
    _draw_with_stroke(
        draw, (50, 360), headline_line2, headline_font,
        fill=(255, 215, 0), stroke_width=8,
    )

    # Banner vermelho top-right
    banner_font = ImageFont.truetype(str(_FONT_BANNER), 56)
    bbox = draw.textbbox((0, 0), accent_banner, font=banner_font)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 30, 18
    bx = THUMB_W - bw - pad_x * 2 - 30
    by = 30
    draw.rectangle(
        [(bx, by), (bx + bw + pad_x * 2, by + bh + pad_y * 2)],
        fill=(220, 30, 30, 240),
    )
    draw.text(
        (bx + pad_x, by + pad_y - 8),
        accent_banner, font=banner_font, fill="white",
    )

    # Tag r/<sub> bottom-left
    tag_font = ImageFont.truetype(str(_FONT_BANNER), 26)
    _draw_with_stroke(
        draw, (50, THUMB_H - 60), subreddit_tag, tag_font,
        fill=(255, 255, 255, 220), stroke_width=3,
    )

    img.convert("RGB").save(out_path, "PNG", optimize=True)


def generate_thumbnail(
    *,
    scene_brief: str,
    headline_line1: str,
    headline_line2: str,
    accent_banner: str,
    subreddit_tag: str,
    out_path: Path,
    base_out_path: Path,
) -> None:
    """Gera 1 thumbnail (backward compat com versão antiga)."""
    prompt = _DEFAULT_BASE_PROMPT.replace("<<SCENE>>", scene_brief)
    flux_schnell_image(prompt, aspect_ratio="16:9", out_path=base_out_path)
    _compose_overlay(
        base_out_path, out_path,
        headline_line1=headline_line1,
        headline_line2=headline_line2,
        accent_banner=accent_banner,
        subreddit_tag=subreddit_tag,
    )


def generate_thumbnail_set(
    variants,  # list[ThumbVariant] de metadata.py — evita import circular
    *,
    subreddit_tag: str,
    out_dir: Path,
    on_progress: Callable[[int, int, str, float], None] | None = None,
) -> list[ThumbnailRender]:
    """Gera N thumbnails (1 por variante). Output em ``out_dir/`` com nomes
    ``thumb_<letter>_<name>.png`` (e ``thumb_<letter>_<name>_base.png`` pro
    base do Flux). Letters a/b/c/d/... seguem a ordem da lista.

    ``on_progress(idx, total, name, seconds_elapsed)`` é chamado depois de
    cada variante pra UI/log.
    """
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    letters = "abcdefghijklmnop"
    renders: list[ThumbnailRender] = []
    for i, v in enumerate(variants):
        letter = letters[i] if i < len(letters) else f"x{i}"
        slug = v.name.lower().replace(" ", "_")
        base_path = out_dir / f"thumb_{letter}_{slug}_base.png"
        final_path = out_dir / f"thumb_{letter}_{slug}.png"
        t0 = time.time()
        prompt = _DEFAULT_BASE_PROMPT.replace("<<SCENE>>", v.scene_brief)
        flux_schnell_image(prompt, aspect_ratio="16:9", out_path=base_path)
        _compose_overlay(
            base_path, final_path,
            headline_line1=v.headline_line1,
            headline_line2=v.headline_line2,
            accent_banner=v.accent_banner,
            subreddit_tag=subreddit_tag,
        )
        renders.append(
            ThumbnailRender(name=slug, path=final_path, base_path=base_path)
        )
        if on_progress is not None:
            on_progress(i + 1, len(variants), slug, time.time() - t0)
    return renders
