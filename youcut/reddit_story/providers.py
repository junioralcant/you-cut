"""Wrappers Replicate: Kokoro 82M TTS + Flux Schnell text-to-image.

Mantém os version hashes pinned pra reprodutibilidade. Atualizar
KOKORO_VERSION/FLUX_MODEL exige passar por validação manual antes
(canal pode pagar ou perder qualidade silenciosamente)."""

from __future__ import annotations

from pathlib import Path

import httpx
import replicate


# Kokoro-82M é community model — exige version pin
# (replicate.run("owner/name") sem version retorna 404 pra community)
KOKORO_MODEL = (
    "jaaari/kokoro-82m:"
    "f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
)

# Flux Schnell é official (black-forest-labs) — funciona sem version pin
FLUX_MODEL = "black-forest-labs/flux-schnell"


def kokoro_tts(text: str, *, voice: str, speed: float, out_path: Path) -> None:
    """Narra ``text`` via Kokoro 82M. Kokoro split textos longos internamente
    e devolve um único arquivo .wav."""
    output = replicate.run(
        KOKORO_MODEL,
        input={"text": text, "voice": voice, "speed": speed},
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=300).content)


def flux_schnell_image(
    prompt: str,
    *,
    aspect_ratio: str,
    out_path: Path,
    megapixels: str = "1",
    steps: int = 4,
) -> None:
    """Gera 1 imagem via Flux Schnell. aspect_ratio ∈ {16:9, 9:16, 1:1, ...}."""
    output = replicate.run(
        FLUX_MODEL,
        input={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "num_outputs": 1,
            "num_inference_steps": steps,
            "go_fast": True,
            "megapixels": megapixels,
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=120).content)
