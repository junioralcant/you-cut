"""One-off: aplica restyle "pôster cinematográfico Copa do Mundo / Brasil" nas
5 primeiras fotos de jogadores_brasil_2026/ usando google/nano-banana via Replicate.

Output: jogadores_brasil_2026/_v1_poster/<slug>.png
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

import httpx
import replicate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "jogadores_brasil_2026"
OUT_DIR = SRC_DIR / "_v1_poster"
OUT_DIR.mkdir(exist_ok=True)

MODEL = "google/nano-banana"

PROMPT = (
    "Cinematic World Cup poster portrait. Brazilian football player wearing the "
    "classic yellow CBF national team jersey. High-contrast dramatic lighting "
    "with warm golden rim light hitting the face, deep shadows on the opposite "
    "side. Subject looks intensely directly into the camera. Background: out-of-"
    "focus stadium with vibrant Brazilian flag colors — saturated yellow as the "
    "dominant tone, deep emerald green and royal blue secondary accents, subtle "
    "stadium floodlight bokeh. Sports magazine cover aesthetic. Skin texture "
    "preserved, sharp eyes, hero shot framing, square 1:1 composition, photo-"
    "realistic, no text, no watermark, no logos, no FIFA branding, no broadcaster "
    "badges."
)


def crop_to_square(path: Path) -> bytes:
    """Center-crop to 1:1 e converte para PNG bytes."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    sq = img.crop((left, top, left + side, top + side))
    if sq.size[0] > 1024:
        sq = sq.resize((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    sq.save(buf, format="PNG")
    return buf.getvalue()


def generate(client: replicate.Client, src: Path) -> bytes:
    ref_bytes = crop_to_square(src)
    handle = io.BytesIO(ref_bytes)
    handle.name = "ref.png"
    output = client.run(
        MODEL,
        input={
            "prompt": PROMPT,
            "image_input": [handle],
            "output_format": "png",
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        return item.read()
    return httpx.get(str(item), timeout=180).content


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        print("ERRO: REPLICATE_API_TOKEN ausente no .env", file=sys.stderr)
        return 1

    photos = sorted(SRC_DIR.glob("*.jpg")) + sorted(SRC_DIR.glob("*.jpeg"))
    photos = [p for p in photos if not p.name.startswith("_")][:5]
    if not photos:
        print("ERRO: nenhuma foto encontrada", file=sys.stderr)
        return 1

    print(f"[INFO] processando {len(photos)} fotos -> {OUT_DIR}")
    for p in photos:
        print(f"  - {p.name}")
    print()

    client = replicate.Client(api_token=token)
    cost_estimate_usd = 0.039 * len(photos)
    print(f"[INFO] custo estimado: ~${cost_estimate_usd:.3f} USD")
    print()

    for i, src in enumerate(photos, 1):
        out_path = OUT_DIR / f"{src.stem}.png"
        if out_path.exists():
            print(f"[{i}/{len(photos)}] {src.name} ja existe -> skip")
            continue
        t0 = time.time()
        try:
            img_bytes = generate(client, src)
            out_path.write_bytes(img_bytes)
            dt = time.time() - t0
            print(f"[{i}/{len(photos)}] {src.name} -> {out_path.name} ({dt:.1f}s, {len(img_bytes)//1024} KB)")
        except Exception as exc:
            print(f"[{i}/{len(photos)}] {src.name} FALHOU: {exc}", file=sys.stderr)

    print()
    print(f"[DONE] outputs em {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
