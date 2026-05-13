"""Helpers do preset visual 'motivacao' — badge canto inferior esquerdo e
outro fade-to-black.

A geração da legenda serif itálica fica em :mod:`youcut.captioner`; aqui
moram apenas as etapas adicionais que rodam ANTES do caption burn
(overlay) e DEPOIS (outro concatenado).

Ver tasks/prd-preset-motivacao/analise-video-referencia.md.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INTER_BOLD_FONT = ASSETS_DIR / "fonts" / "Inter-Bold.ttf"
SOCIAL_ICON = ASSETS_DIR / "overlays" / "social_icon.png"

# Coordenadas no canvas 1080×1920. Padding 60 px da borda esquerda, badge
# alinhado ~140 px da borda inferior (mesma região do referência).
_BADGE_X = 60
_BADGE_Y_FROM_BOTTOM = 220
_BADGE_ICON_SIZE = 64  # ícone redimensionado a partir do PNG 96×96
_BADGE_TEXT_SIZE = 38
_BADGE_TEXT_GAP = 18  # espaço horizontal entre ícone e texto

_OUTRO_DURATION_S = 3.0
_OUTRO_ICON_SIZE = 220
_OUTRO_TEXT_SIZE = 72


def _escape_drawtext(value: str) -> str:
    """Escapa caracteres especiais do drawtext (`:`, `'`, `\\`)."""
    return value.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def build_badge_filtergraph(handle: str) -> str:
    """Filter-complex que recebe input [0:v] e emite [vbadge] com:

    - ícone PNG (social_icon.png) escalado pra 64×64
    - texto `@HANDLE` (uppercase) ao lado, branco, Inter Bold
    - ambos ancorados em (60 px da borda esquerda, 220 px da borda inferior)

    A cadeia retornada NÃO inclui o nó de input do PNG: o orquestrador
    monta a chamada ffmpeg passando ``-i social_icon.png`` como input [1].
    """
    handle_text = _escape_drawtext(f"@{handle.upper()}")
    icon_x = _BADGE_X
    text_x = _BADGE_X + _BADGE_ICON_SIZE + _BADGE_TEXT_GAP
    icon_y = f"H-h-{_BADGE_Y_FROM_BOTTOM}"
    text_y = (
        f"H-{_BADGE_Y_FROM_BOTTOM}-({_BADGE_ICON_SIZE}+text_h)/2"
    )
    return (
        f"[1:v]scale={_BADGE_ICON_SIZE}:{_BADGE_ICON_SIZE}[icon];"
        f"[0:v][icon]overlay={icon_x}:{icon_y}[vico];"
        f"[vico]drawtext=fontfile='{INTER_BOLD_FONT}':"
        f"text='{handle_text}':fontcolor=white:fontsize={_BADGE_TEXT_SIZE}:"
        f"x={text_x}:y={text_y}[vbadge]"
    )


def apply_badge(input_path: Path, handle: str, output_path: Path) -> None:
    """Queima o badge motivacao em ``input_path`` e escreve em ``output_path``.

    Falha alto se algum asset não está presente — não temos fallback porque
    o preset depende dos arquivos.
    """
    if not INTER_BOLD_FONT.exists():
        raise FileNotFoundError(f"Fonte Inter-Bold ausente: {INTER_BOLD_FONT}")
    if not SOCIAL_ICON.exists():
        raise FileNotFoundError(f"Ícone social ausente: {SOCIAL_ICON}")
    filter_complex = build_badge_filtergraph(handle)
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-i", str(SOCIAL_ICON),
        "-filter_complex", filter_complex,
        "-map", "[vbadge]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou ao aplicar badge motivacao: %s", stderr)
        raise


def render_outro(
    handle: str,
    duration_s: float,
    output_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
) -> None:
    """Gera um clipe estático de fade-to-black com badge centralizado.

    Usado como cartela final do preset motivacao. Áudio é silêncio AAC pra
    casar com a faixa do clipe principal no concat.
    """
    if not INTER_BOLD_FONT.exists():
        raise FileNotFoundError(f"Fonte Inter-Bold ausente: {INTER_BOLD_FONT}")
    if not SOCIAL_ICON.exists():
        raise FileNotFoundError(f"Ícone social ausente: {SOCIAL_ICON}")
    handle_text = _escape_drawtext(f"@{handle.upper()}")
    # Ícone centralizado no canvas + texto 30 px abaixo.
    icon_x = f"(W-{_OUTRO_ICON_SIZE})/2"
    icon_y = f"(H-{_OUTRO_ICON_SIZE})/2"
    text_y = f"(H+{_OUTRO_ICON_SIZE})/2+40"
    filter_complex = (
        f"[1:v]scale={_OUTRO_ICON_SIZE}:{_OUTRO_ICON_SIZE}[icon];"
        f"[0:v][icon]overlay={icon_x}:{icon_y}[vico];"
        f"[vico]drawtext=fontfile='{INTER_BOLD_FONT}':"
        f"text='{handle_text}':fontcolor=white:fontsize={_OUTRO_TEXT_SIZE}:"
        f"x=(w-text_w)/2:y={text_y},"
        f"fade=t=in:st=0:d=0.5[v]"
    )
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"color=black:s={width}x{height}:d={duration_s}:r=30",
        "-i", str(SOCIAL_ICON),
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(duration_s),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "2:a",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou ao renderizar outro motivacao: %s", stderr)
        raise


def append_outro(clip_path: Path, outro_path: Path) -> None:
    """Concatena ``outro_path`` ao final de ``clip_path`` (in-place).

    Usa o concat demuxer (sem re-encode quando codecs casam). O clipe
    original é substituído atomicamente pelo resultado.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado no PATH")
    list_file = clip_path.with_suffix(".concat.txt")
    list_file.write_text(
        f"file '{clip_path.resolve()}'\nfile '{outro_path.resolve()}'\n",
        encoding="utf-8",
    )
    tmp_out = clip_path.with_name(clip_path.stem + "_with_outro.mp4")
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp_out.replace(clip_path)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou ao concatenar outro motivacao: %s", stderr)
        raise
    finally:
        list_file.unlink(missing_ok=True)


def apply_motivacao_postprocess(
    clip_path: Path, handle: str, *, with_overlay: bool, with_outro: bool
) -> None:
    """Aplica overlay (badge) e/ou outro em sequência.

    Roda APÓS o caption burn — assim o badge fica sobre as legendas e o
    outro herda os codecs do clipe final.
    """
    if with_overlay:
        tmp = clip_path.with_name(clip_path.stem + "_badge.mp4")
        try:
            apply_badge(clip_path, handle, tmp)
            tmp.replace(clip_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    if with_outro:
        outro = clip_path.with_name(clip_path.stem + "_outro.mp4")
        try:
            render_outro(handle, _OUTRO_DURATION_S, outro)
            append_outro(clip_path, outro)
        finally:
            outro.unlink(missing_ok=True)
