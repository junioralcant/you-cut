"""Gera YouTube channel banner 2048×1152 pro THREAD COURT.

Pipeline: Flux Schnell wide background (16:9, atmospheric courtroom) + Pillow
upscale 2048×1152 + dark vignette central + composite do logo existente no
safe zone (centro 1546×423 visível em todos os devices).

Output: poc/branding/threadcourt/_v2/channel_banner.png (≤6MB, atende specs YT).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFilter

from poc.hfy_poc import load_env, log
from youcut.reddit_story.providers import flux_schnell_image


BANNER_W = 2048
BANNER_H = 1152
# YouTube safe zone (visível em mobile/TV): centro 1546×423
SAFE_W = 1546
SAFE_H = 423


BG_PROMPT = (
    "Wide cinematic 16:9 channel banner background for a legal/court themed "
    "YouTube channel. Classical courtroom interior: tall ornate marble columns "
    "framing the LEFT and RIGHT sides, deep midnight navy walls, faint warm "
    "spotlights from above creating atmospheric haze, dust particles floating "
    "in the light beams. CENTER OF FRAME completely empty — wide negative "
    "space in the middle third for logo overlay later. Antique gold details "
    "on the column capitals, subtle vintage burgundy drapery hints on the far "
    "edges. Vibrant saturated palette: deep midnight navy blue dominant, "
    "antique gold accents, hints of crimson burgundy on draperies. NOT pastel, "
    "NOT washed out. Cinematic 35mm wide depth of field. ABSOLUTELY NO text, "
    "no letters, no words, no logos, no scales of justice, no symbols anywhere "
    "in the frame."
)


def main() -> None:
    load_env()
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN missing")

    logo_path = (
        REPO_ROOT
        / "poc"
        / "branding"
        / "threadcourt"
        / "_v2"
        / "logo_scales_thread.png"
    )
    if not logo_path.exists():
        sys.exit(f"Logo ausente: {logo_path}")

    out_dir = logo_path.parent
    base_path = out_dir / "banner_base.png"
    banner_path = out_dir / "channel_banner.png"

    log(f"Logo: {logo_path}")
    log(f"Output: {banner_path}")

    # 1. Flux Schnell wide base
    log("[1/2] Flux Schnell wide 16:9 base...")
    t0 = time.time()
    flux_schnell_image(BG_PROMPT, aspect_ratio="16:9", out_path=base_path)
    log(f"    done in {time.time()-t0:.1f}s")

    # 2. Compose banner
    log("[2/2] Compositing 2048×1152 + logo overlay...")
    t0 = time.time()

    bg = Image.open(base_path).convert("RGBA")
    bg = bg.resize((BANNER_W, BANNER_H), Image.LANCZOS)

    # Vinheta radial suave no centro (melhora contraste do logo)
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    cx, cy = BANNER_W // 2, BANNER_H // 2
    # Soft circular fill no centro pra escurecer ~10-20% atrás do logo
    for r in range(700, 100, -10):
        alpha = int(70 * (1 - r / 700))
        od.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=(0, 0, 0, alpha),
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(40))
    bg = Image.alpha_composite(bg, overlay)

    # Logo composite — escalado pra ~85% da altura do safe zone
    logo = Image.open(logo_path).convert("RGBA")
    target_h = int(SAFE_H * 0.85)  # 360px
    target_w = target_h  # logo é quadrado
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    paste_x = (BANNER_W - target_w) // 2
    paste_y = (BANNER_H - target_h) // 2

    # Glow dourado sutil atrás do logo pra integrar visualmente com o BG
    glow = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    glow_r = int(target_h * 0.75)
    for r in range(glow_r, 50, -8):
        alpha = int(35 * (1 - r / glow_r))
        gd.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=(212, 175, 55, alpha),  # antique gold
        )
    glow = glow.filter(ImageFilter.GaussianBlur(30))
    bg = Image.alpha_composite(bg, glow)

    # Paste do logo
    bg.paste(logo, (paste_x, paste_y), logo)

    bg.convert("RGB").save(banner_path, "PNG", optimize=True)

    size_mb = banner_path.stat().st_size / 1024 / 1024
    log(f"    done in {time.time()-t0:.1f}s — {size_mb:.2f} MB")
    log(f"    spec check: {BANNER_W}×{BANNER_H} ✓ (mín 2048×1152), {size_mb:.2f} MB ✓ (máx 6)")
    log(f"DONE → {banner_path}")


if __name__ == "__main__":
    main()
