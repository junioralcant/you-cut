"""Validador de entrada para o pipeline `youcut comic` (RF-01, RF-02)."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".webm"})
MAX_DURATION_SECONDS: float = 120.0


class VideoSpec(BaseModel):
    path: Path
    duration_seconds: float
    width: int
    height: int
    codec: str


class VideoValidationError(Exception):
    """Erro de validação de vídeo de entrada do `youcut comic`."""


def _ffprobe_video_info(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoValidationError(
            "ffprobe não encontrado no PATH. Instale o FFmpeg para usar `youcut comic`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise VideoValidationError(
            f"Falha ao inspecionar o vídeo {path.name}: {stderr or 'erro do ffprobe'}"
        ) from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoValidationError(
            f"Saída inválida do ffprobe ao inspecionar {path.name}."
        ) from exc


def validate_video(path: Path | str) -> VideoSpec:
    """Valida o vídeo de entrada (formato e duração) e retorna um `VideoSpec`.

    Erros são lançados como `VideoValidationError` em pt-BR.
    """
    video_path = Path(path)

    if not video_path.exists():
        raise VideoValidationError(
            f"Arquivo não encontrado: {video_path}. Informe o caminho de um vídeo local."
        )
    if not video_path.is_file():
        raise VideoValidationError(
            f"O caminho informado não é um arquivo: {video_path}."
        )

    extension = video_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise VideoValidationError(
            f"Formato de vídeo não suportado ({extension or 'sem extensão'}). "
            f"Use um destes: {supported}."
        )

    info = _ffprobe_video_info(video_path)
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    if not streams:
        raise VideoValidationError(
            f"Não foi possível encontrar uma faixa de vídeo em {video_path.name}."
        )

    stream = streams[0]
    try:
        duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    except (TypeError, ValueError) as exc:
        raise VideoValidationError(
            f"Duração inválida reportada pelo ffprobe para {video_path.name}."
        ) from exc

    if duration <= 0:
        raise VideoValidationError(
            f"Duração inválida (≤0s) para {video_path.name}."
        )
    if duration > MAX_DURATION_SECONDS:
        raise VideoValidationError(
            f"Vídeo muito longo ({duration:.1f}s). "
            f"O `youcut comic` aceita no máximo {MAX_DURATION_SECONDS:.0f}s."
        )

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    codec = str(stream.get("codec_name") or "unknown")

    spec = VideoSpec(
        path=video_path,
        duration_seconds=duration,
        width=width,
        height=height,
        codec=codec,
    )
    logger.info(
        "comic.validator: %s aceito (%.2fs, %dx%d, codec=%s)",
        video_path.name,
        duration,
        width,
        height,
        codec,
    )
    return spec
