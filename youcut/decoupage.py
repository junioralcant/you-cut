"""Remoção automática de silêncios (decupagem) em um clipe.

Pipeline:
1. ``ffmpeg -af silencedetect`` localiza intervalos silenciosos.
2. O complemento desses intervalos vira a lista de trechos a manter,
   com uma pequena margem de padding nas bordas.
3. Um único ``ffmpeg`` re-encoda o clipe usando ``select`` + ``aselect``
   para descartar os silêncios em uma só passagem.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")

_MIN_KEEP_DURATION = 0.05  # ranges menores que isso são descartados (ruído)


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def detect_silences(
    video_path: Path,
    *,
    noise_db: float = -30.0,
    min_silence_gap: float = 0.4,
) -> list[tuple[float, float]]:
    """Roda ``silencedetect`` e devolve a lista de janelas silenciosas."""
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-af", f"silencedetect=noise={noise_db}dB:duration={min_silence_gap}",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_silencedetect(proc.stderr)


def parse_silencedetect(stderr: str) -> list[tuple[float, float]]:
    """Extrai pares (start, end) das mensagens emitidas pelo silencedetect."""
    starts: list[float] = []
    ends: list[float] = []
    for line in stderr.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = _SILENCE_END_RE.search(line)
        if m:
            ends.append(float(m.group(1)))
    pairs: list[tuple[float, float]] = []
    for start, end in zip(starts, ends):
        if end > start:
            pairs.append((start, end))
    return pairs


def compute_keep_ranges(
    silences: list[tuple[float, float]],
    duration: float,
    keep_padding: float,
) -> list[tuple[float, float]]:
    """Inverte a lista de silêncios em janelas a manter, aplicando padding."""
    if duration <= 0:
        return []

    if not silences:
        return [(0.0, duration)]

    silences_sorted = sorted(silences, key=lambda x: x[0])
    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences_sorted:
        s_start = max(0.0, s_start)
        s_end = min(duration, s_end)
        if s_start > cursor:
            keeps.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if cursor < duration:
        keeps.append((cursor, duration))

    padded: list[tuple[float, float]] = []
    for start, end in keeps:
        new_start = max(0.0, start - keep_padding)
        new_end = min(duration, end + keep_padding)
        if new_end - new_start < _MIN_KEEP_DURATION:
            continue
        if padded and new_start <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], new_end))
        else:
            padded.append((new_start, new_end))
    return padded


def build_select_expr(keep_ranges: list[tuple[float, float]]) -> str:
    """Constrói a expressão ``between(t,a,b)+between(t,c,d)+...`` para select."""
    if not keep_ranges:
        return "0"
    parts = [f"between(t,{a:.3f},{b:.3f})" for a, b in keep_ranges]
    return "+".join(parts)


def remove_silences(
    video_path: Path,
    *,
    noise_db: float = -30.0,
    min_silence_gap: float = 0.4,
    keep_padding: float = 0.05,
) -> Path:
    """Remove silêncios de ``video_path`` (sobrescrevendo o arquivo original).

    Devolve o próprio caminho. Em caso de detecção sem silêncios significativos,
    o arquivo é mantido inalterado. Falhas internas re-levantam — o caller
    decide se trata como warning não-fatal.
    """
    duration = _probe_duration(video_path)
    silences = detect_silences(
        video_path, noise_db=noise_db, min_silence_gap=min_silence_gap
    )
    if not silences:
        logger.info("Decoupage: nenhum silêncio detectado em %s.", video_path.name)
        return video_path

    keeps = compute_keep_ranges(silences, duration, keep_padding)
    if not keeps:
        logger.warning(
            "Decoupage: clipe %s ficou sem trechos audíveis após análise — mantendo original.",
            video_path.name,
        )
        return video_path

    total_kept = sum(b - a for a, b in keeps)
    if total_kept >= duration - 0.01:
        logger.info("Decoupage: silêncios irrelevantes em %s, mantendo original.", video_path.name)
        return video_path

    expr = build_select_expr(keeps)
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{expr}',asetpts=N/SR/TB"

    with tempfile.NamedTemporaryFile(
        suffix=video_path.suffix, delete=False, dir=video_path.parent
    ) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tmp_path.unlink(missing_ok=True)
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        logger.error(
            "Decoupage: ffmpeg falhou em %s (código %d): %s",
            video_path.name, e.returncode, stderr,
        )
        raise

    tmp_path.replace(video_path)
    logger.info(
        "Decoupage: %s — %.2fs → %.2fs (removidos %d trechos silenciosos).",
        video_path.name, duration, total_kept, len(silences),
    )
    return video_path
