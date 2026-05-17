"""Gera 2 variantes de logo pro canal THREAD COURT (long-form Reddit revenge).

Paleta: legal-drama deep navy + burgundy + gold (oposta ao orange/cyan do
HOLLOWIRE). Tipografia condensada serif/stencil pra evocar courthouse.

Outputs versionados em poc/branding/threadcourt/_v<N>/.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont

from poc.hfy_poc import load_env, log
from youcut.reddit_story.providers import flux_schnell_image


_FONT_DIR = REPO_ROOT / "youcut" / "assets" / "fonts"
_FONT_HEADLINE = _FONT_DIR / "Anton-Regular.ttf"
_FONT_SUBLINE = _FONT_DIR / "Inter-Bold.ttf"


VARIANTS = {
    "courtroom_gavel": (
        "Square 1:1 channel logo design. Pure flat graphic illustration (not "
        "photographic). A stylized golden gavel resting on a burgundy court bench, "
        "set against a deep midnight-navy background. Above the gavel, a single "
        "thin golden thread arcs upward as if pulling the scene together. Sharp "
        "geometric vector lines, no gradients, no photorealism. Legal drama / Mass "
        "Effect codex aesthetic. Vibrant saturated palette: deep navy blue, rich "
        "burgundy red, antique gold accents, NOT pastel. Strong negative space at "
        "the CENTER and BOTTOM half for text overlay (do not put any object "
        "in the lower half). Background absolutely empty in the lower half. "
        "Absolutely no text, no letters, no words, no logos, no symbols with "
        "writing anywhere in the frame."
    ),
    "scales_thread": (
        "Square 1:1 channel logo. Vintage courthouse aesthetic, flat vector "
        "illustration. A pair of antique golden scales of justice in the center, "
        "with a glowing red thread woven through both pans connecting them. "
        "Background: deep matte navy with subtle classical column silhouettes "
        "fading into the corners. Sharp clean vector lines, vibrant saturated "
        "colors (deep navy + crimson + antique gold), NOT pastel. Strong "
        "negative space on the LEFT and RIGHT thirds for text. Composition "
        "tight in the upper-center. Bottom third must be EMPTY background. "
        "No text, no letters, no words, no logos."
    ),
}


def next_version_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(m.group(1))
        for p in base.iterdir()
        if (m := re.match(r"_v(\d+)$", p.name))
    )
    next_n = (existing[-1] + 1) if existing else 1
    out = base / f"_v{next_n}"
    out.mkdir()
    return out


def overlay_text(base_path: Path, out_path: Path) -> None:
    """Sobrepõe 'THREAD COURT' em 2 linhas + tagline.

    Layout: 'THREAD' branco grande em cima do centro, 'COURT' em ouro logo
    abaixo, tagline 'where Reddit gets its verdict' menor no rodapé.
    """
    img = Image.open(base_path).convert("RGBA")
    # Garantir quadrado 1024x1024
    img = img.resize((1024, 1024), Image.LANCZOS)

    # Vinheta sutil no rodapé pra dar contraste pro texto
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(380, 1024):
        alpha = int(120 * ((y - 380) / (1024 - 380)))
        od.line([(0, y), (1024, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    headline_font = ImageFont.truetype(str(_FONT_HEADLINE), 180)

    def _draw_centered(text: str, y: int, fill: tuple | str) -> None:
        bbox = draw.textbbox((0, 0), text, font=headline_font)
        w = bbox[2] - bbox[0]
        x = (1024 - w) // 2
        # Drop shadow + stroke
        draw.text(
            (x + 5, y + 5), text, font=headline_font,
            fill=(0, 0, 0, 200), stroke_width=6, stroke_fill=(0, 0, 0, 200),
        )
        draw.text(
            (x, y), text, font=headline_font,
            fill=fill, stroke_width=6, stroke_fill="black",
        )

    _draw_centered("THREAD", y=560, fill="white")
    _draw_centered("COURT", y=720, fill=(212, 175, 55))  # antique gold

    # Tagline rodapé — "Internet" em vez de "Reddit" pra evitar trademark
    # collision em brand-line (ver feedback_marca_reddit em memória).
    tag_font = ImageFont.truetype(str(_FONT_SUBLINE), 32)
    tag = "WHERE THE INTERNET GETS ITS VERDICT"
    bbox = draw.textbbox((0, 0), tag, font=tag_font)
    tw = bbox[2] - bbox[0]
    tx = (1024 - tw) // 2
    draw.text(
        (tx + 2, 920 + 2), tag, font=tag_font,
        fill=(0, 0, 0, 200), stroke_width=2, stroke_fill=(0, 0, 0, 200),
    )
    draw.text(
        (tx, 920), tag, font=tag_font,
        fill=(255, 255, 255, 220), stroke_width=2, stroke_fill="black",
    )

    img.convert("RGB").save(out_path, "PNG", optimize=True)


def main() -> None:
    load_env()
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN missing")

    branding_dir = REPO_ROOT / "poc" / "branding" / "threadcourt"
    out_dir = next_version_dir(branding_dir)
    log(f"Branding dir (versionado): {out_dir}")

    for name, prompt in VARIANTS.items():
        base_path = out_dir / f"base_{name}.png"
        final_path = out_dir / f"logo_{name}.png"
        log(f"  generating {name}...")
        t0 = time.time()
        flux_schnell_image(prompt, aspect_ratio="1:1", out_path=base_path)
        overlay_text(base_path, final_path)
        log(
            f"    done in {time.time()-t0:.1f}s — "
            f"{final_path.stat().st_size/1024:.0f} KB → {final_path.name}"
        )

    log(f"DONE → {out_dir}")


if __name__ == "__main__":
    main()
