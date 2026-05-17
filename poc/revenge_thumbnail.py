"""Gera thumbnail 1280x720 pra vídeo Reddit revenge long-form.

Pipeline: Flux Schnell (Replicate) gera base 16:9 com composição reservando
espaço pra texto → Pillow sobrepõe text overlay tipo "EmKay/Reddit On Tap"
(Anton condensed bold, white com black stroke 8px + drop shadow + red corner
banner).

Saves to: poc/revenge/<latest_session>/thumbnail.png
Versiona se já existir: thumbnail_v2.png, _v3.png etc.

Run:
    .venv/bin/python poc/revenge_thumbnail.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx
import replicate
from PIL import Image, ImageDraw, ImageFont

from poc.hfy_poc import load_env, log


FONT_DIR = REPO_ROOT / "youcut" / "assets" / "fonts"
ANTON = FONT_DIR / "Anton-Regular.ttf"
INTER_BOLD = FONT_DIR / "Inter-Bold.ttf"

THUMB_W = 1280
THUMB_H = 720


FLUX_PROMPT = (
    "Cinematic dramatic thumbnail composition for a YouTube revenge story video. "
    "Exclusive suburban gated community at golden hour, with a damaged kidney-shaped "
    "paddling pool in the foreground, scattered legal documents and torn HOA "
    "violation notices floating in the air mid-frame, a wrought-iron entrance gate "
    "in the background partially open. Strong negative space and darker vignette "
    "on the LEFT THIRD of the frame (for text overlay later). Composition pushes "
    "the visual elements to the RIGHT TWO-THIRDS. Dramatic chiaroscuro lighting, "
    "deep orange sunset, vibrant saturated palette, 35mm cinematic, shallow depth "
    "of field. Pure visual: ABSOLUTELY NO TEXT, NO LETTERS, NO WORDS, NO LOGOS, "
    "NO SIGNS WITH WRITING. Horizontal 16:9 framing."
)

# Main headline + accent. Bold and dramatic.
MAIN_TEXT_LINE1 = "HOA TRIED TO"
MAIN_TEXT_LINE2 = "TAKE MY LAND"
ACCENT_BANNER = "BIG MISTAKE"


def generate_base_image(out_path: Path) -> None:
    """Flux Schnell 16:9 ~1408x768 base image."""
    log("  Flux Schnell base image...")
    t0 = time.time()
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": FLUX_PROMPT,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "num_outputs": 1,
            "num_inference_steps": 4,
            "go_fast": True,
            "megapixels": "1",
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=120).content)
    log(f"    done in {time.time()-t0:.1f}s — {out_path.stat().st_size/1024:.0f} KB")


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    stroke_fill: str,
    stroke_width: int,
) -> None:
    """Drop shadow + stroke + fill."""
    # Drop shadow (offset 4,4, semi-transparent black)
    draw.text(
        (pos[0] + 4, pos[1] + 4),
        text,
        font=font,
        fill=(0, 0, 0, 180),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 180),
    )
    # Main with stroke
    draw.text(
        pos, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill
    )


def compose_thumbnail(base_path: Path, out_path: Path) -> None:
    """Pillow overlay: title text (left), accent banner (top-right corner)."""
    log("  Pillow overlay (Anton text + red banner)...")
    t0 = time.time()
    img = Image.open(base_path).convert("RGBA")
    # Resize/crop to exact 1280x720
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    # Darken left third for text legibility
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Vertical gradient gradient from left (darker) to right (transparent)
    for x in range(int(THUMB_W * 0.55)):
        alpha = int(180 * (1 - x / (THUMB_W * 0.55)))
        od.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    # Main headline — Anton condensed bold, white with thick black stroke
    main_font = ImageFont.truetype(str(ANTON), 130)
    _draw_text_with_stroke(
        draw,
        (50, 220),
        MAIN_TEXT_LINE1,
        main_font,
        fill="white",
        stroke_fill="black",
        stroke_width=8,
    )
    # Line 2 — yellow accent
    _draw_text_with_stroke(
        draw,
        (50, 360),
        MAIN_TEXT_LINE2,
        main_font,
        fill=(255, 215, 0),  # YT-style yellow
        stroke_fill="black",
        stroke_width=8,
    )

    # Red accent banner top-right corner
    banner_font = ImageFont.truetype(str(INTER_BOLD), 56)
    banner_bbox = draw.textbbox((0, 0), ACCENT_BANNER, font=banner_font)
    banner_w = banner_bbox[2] - banner_bbox[0]
    banner_h = banner_bbox[3] - banner_bbox[1]
    pad_x, pad_y = 30, 18
    banner_x = THUMB_W - banner_w - pad_x * 2 - 30
    banner_y = 30
    # Red rectangle background
    draw.rectangle(
        [
            (banner_x, banner_y),
            (banner_x + banner_w + pad_x * 2, banner_y + banner_h + pad_y * 2),
        ],
        fill=(220, 30, 30, 240),
    )
    # White text on red
    draw.text(
        (banner_x + pad_x, banner_y + pad_y - 8),
        ACCENT_BANNER,
        font=banner_font,
        fill="white",
    )

    # Subtle "r/MaliciousCompliance" tag bottom-left (small, for niche signal)
    tag_font = ImageFont.truetype(str(INTER_BOLD), 26)
    tag_text = "r/MaliciousCompliance"
    _draw_text_with_stroke(
        draw,
        (50, THUMB_H - 60),
        tag_text,
        tag_font,
        fill=(255, 255, 255, 220),
        stroke_fill="black",
        stroke_width=3,
    )

    img.convert("RGB").save(out_path, "PNG", optimize=True)
    log(f"    done in {time.time()-t0:.1f}s — {out_path.stat().st_size/1024:.0f} KB")


def next_versioned_path(base: Path, stem: str) -> Path:
    """thumbnail.png → thumbnail_v2.png → thumbnail_v3.png ..."""
    p = base / f"{stem}.png"
    if not p.exists():
        return p
    n = 2
    while True:
        p = base / f"{stem}_v{n}.png"
        if not p.exists():
            return p
        n += 1


def main() -> None:
    load_env()
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN missing")

    # Find latest revenge session
    revenge_root = REPO_ROOT / "poc" / "revenge"
    sessions = [p for p in revenge_root.iterdir() if p.is_dir() and p.name != "_source"]
    if not sessions:
        sys.exit("No revenge POC sessions found")
    session = max(sessions, key=lambda p: p.stat().st_mtime)
    log(f"Session: {session}")

    base_path = next_versioned_path(session, "thumbnail_base")
    thumb_path = next_versioned_path(session, "thumbnail")
    log(f"Output: {thumb_path}")

    log("[1/2] Generating base image via Flux Schnell...")
    generate_base_image(base_path)

    log("[2/2] Compositing text overlay via Pillow...")
    compose_thumbnail(base_path, thumb_path)

    log(f"DONE → {thumb_path}")


if __name__ == "__main__":
    main()
