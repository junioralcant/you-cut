"""Generate HOLLOWIRE channel logo variants via gpt-image-1.

Outputs go to poc/branding/hollowire/_v<N>/ — never overwrites prior runs
(per memory rule: preserve generated images, version them).
"""

from __future__ import annotations

import base64
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from poc.hfy_poc import load_env, log


VARIANTS = {
    "stencil_military": (
        "Square 1:1 channel logo. Pure flat graphic design (not photographic). "
        "Bold condensed military-stencil typography reading 'HOLLOWIRE' across "
        "the center on two lines: 'HOLLO' in burnt vibrant orange on top, "
        "'WIRE' in saturated electric cyan below. A single broken copper wire "
        "frays from the right edge of the W and arcs above the text with a "
        "small spark at its tip. Background: deep matte black with subtle "
        "warning-stripe corners in faded orange. Crisp vector look, sharp "
        "edges, no gradients, no photorealism, no shadows. Modern military "
        "insignia / Helldivers / Halo UDF chevron aesthetic. Vibrant saturated "
        "colors, NOT pastel. The text 'HOLLOWIRE' must be perfectly spelled, "
        "uppercase, legible, centered. No other text anywhere."
    ),
    "signal_glitch": (
        "Square 1:1 channel logo. Cinematic poster-style design. Wordmark "
        "'HOLLOWIRE' centered in massive condensed serif-sans hybrid, "
        "uppercase, with a horizontal RGB signal-glitch tear ripping through "
        "the middle: top half of the letters in vibrant burnt orange, bottom "
        "half displaced in electric cyan. Scanline noise across the whole "
        "frame. Background: deep black with a faint orange glow rising from "
        "below as if an explosion just ended. Tiny static dots scattered. "
        "Vibrant saturated colors, NOT pastel. The text 'HOLLOWIRE' must be "
        "perfectly spelled, uppercase, legible, centered. No other text "
        "anywhere — no tagline, no version number, no logos."
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


def generate_logo(client: OpenAI, prompt: str, out_path: Path) -> None:
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        n=1,
        quality="high",
    )
    img_b64 = result.data[0].b64_json
    out_path.write_bytes(base64.b64decode(img_b64))


def main() -> None:
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing")

    branding_dir = REPO_ROOT / "poc" / "branding" / "hollowire"
    out_dir = next_version_dir(branding_dir)
    log(f"Branding dir (versioned): {out_dir}")

    oai = OpenAI()

    for name, prompt in VARIANTS.items():
        out_path = out_dir / f"logo_{name}.png"
        log(f"  generating {name}...")
        t0 = time.time()
        generate_logo(oai, prompt, out_path)
        log(
            f"    done in {time.time()-t0:.1f}s — "
            f"{out_path.stat().st_size/1024:.0f} KB → {out_path.name}"
        )

    log(f"DONE → {out_dir}")


if __name__ == "__main__":
    main()
