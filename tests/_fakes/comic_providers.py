"""Fakes determinísticos para os providers do `youcut comic`.

Usados por testes que precisam de artefatos reais (PNG e MP4) sem chamar
APIs pagas. ``FakeImageProvider`` gera PNGs sintéticos via Pillow;
``FakeI2VProvider`` gera mp4s sintéticos via FFmpeg color source.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


class FakeImageProvider:
    """Gera PNGs determinísticos a partir do prompt + nº de chamadas."""

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1024, 1024),
        color: tuple[int, int, int] = (200, 220, 240),
    ) -> None:
        self.size = size
        self.color = color
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[Path] | None = None,
        size: str = "1024x1024",
        input_fidelity: str = "high",
    ) -> bytes:
        try:
            w, h = (int(p) for p in size.lower().split("x"))
        except (ValueError, AttributeError):
            w, h = self.size

        self.calls.append(
            {
                "prompt": prompt,
                "reference_images": [Path(r) for r in (reference_images or [])],
                "size": (w, h),
                "input_fidelity": input_fidelity,
            }
        )

        img = Image.new("RGB", (w, h), self.color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        snippet = (prompt or "fake")[:40]
        try:
            draw.text((20, 20), snippet, fill=(20, 20, 30), font=font)
        except Exception:
            draw.rectangle([0, 0, 100, 100], fill=(20, 20, 30))

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


class FakeI2VProvider:
    """Gera mp4s sintéticos via FFmpeg `color` source.

    Pula testes (via ``ffmpeg_available``) quando FFmpeg não está disponível.
    """

    def __init__(self, *, default_color: str = "blue", size: tuple[int, int] = (720, 1280)) -> None:
        self.default_color = default_color
        self.size = size
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()

    @property
    def ffmpeg_available(self) -> bool:
        return _has_ffmpeg()

    def image_to_video(
        self,
        prompt_image: Path,
        prompt_text: str,
        reference_images: list[Path],
        duration_seconds: float = 3.0,
        ratio: str = "720:1280",
    ) -> bytes:
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg não disponível; FakeI2VProvider não pode operar.")

        with self._lock:
            self.calls.append(
                {
                    "prompt_image": Path(prompt_image),
                    "prompt_text": prompt_text,
                    "reference_images": [Path(r) for r in reference_images],
                    "duration": duration_seconds,
                    "ratio": ratio,
                }
            )

        try:
            w, h = (int(p) for p in ratio.split(":"))
        except (ValueError, AttributeError):
            w, h = self.size

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={self.default_color}:size={w}x{h}:duration={duration_seconds}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-t",
                f"{duration_seconds:.2f}",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return out_path.read_bytes()
        finally:
            out_path.unlink(missing_ok=True)


def has_ffmpeg() -> bool:
    return _has_ffmpeg()


def reset_fakes(*fakes: Iterable) -> None:
    for fake in fakes:
        if hasattr(fake, "calls"):
            fake.calls.clear()
