"""Composer Final — concat de mini-clipes + áudio original + legendas (RF-26..RF-31)."""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from youcut.captioner import build_ass_for_words
from youcut.config import PipelineConfig
from youcut.models import (
    Panel,
    PanelRenderResult,
    TranscriptionResult,
    WordTimestamp,
)

logger = logging.getLogger(__name__)


OUTPUT_WIDTH: int = 1080
OUTPUT_HEIGHT: int = 1920
DURATION_TOLERANCE_SECONDS: float = 0.2
DEFAULT_OUTPUT_NAME: str = "motion_comic.mp4"


class ComposerError(Exception):
    """Erro do composer (FFmpeg ausente, painel sem clipe, etc.)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ComposerError("ffmpeg não encontrado no PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
        raise ComposerError(
            f"FFmpeg falhou em '{shlex.join(cmd[:3])} …': {stderr.strip() or 'erro desconhecido'}"
        ) from exc


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ComposerError(f"ffprobe falhou em {path}: {exc}") from exc
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise ComposerError(f"Saída inesperada do ffprobe: {result.stdout!r}") from exc


def _filter_words_in_range(
    transcription: TranscriptionResult,
    start: float,
    end: float,
) -> list[WordTimestamp]:
    out: list[WordTimestamp] = []
    for seg in transcription.segments:
        for w in seg.words:
            if w.start >= start and w.start < end:
                out.append(w)
    return out


# ---------------------------------------------------------------------------
# Extend / trim
# ---------------------------------------------------------------------------


def _extend_or_trim(
    clip_path: Path,
    target_seconds: float,
    *,
    out_path: Path,
    mode: str = "hold",
) -> Path:
    """Ajusta ``clip_path`` para ``target_seconds``.

    Em ``mode="hold"`` (default), estende com freeze do último frame; em
    ``mode="loop"``, faz looping; em ``mode="ping_pong"`` reflete e concatena.
    Quando o clipe é mais longo que o alvo, sempre trima.
    """

    actual = _ffprobe_duration(clip_path)
    delta = target_seconds - actual

    if abs(delta) <= 1e-3:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(clip_path),
            "-c",
            "copy",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
        return out_path

    if delta < 0:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(clip_path),
            "-t",
            f"{target_seconds:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
        return out_path

    if mode == "hold":
        vf = f"tpad=stop_mode=clone:stop_duration={delta:.3f}"
    elif mode == "loop":
        vf = f"loop=loop=-1:size=1:start=0,trim=duration={target_seconds:.3f}"
    else:
        raise ComposerError(f"modo de extensão desconhecido: {mode}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
        "-vf",
        vf,
        "-t",
        f"{target_seconds:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-an",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Concat / mux / burn
# ---------------------------------------------------------------------------


def _concat_clips(clip_paths: Iterable[Path], out_path: Path) -> Path:
    paths = [Path(p) for p in clip_paths]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for p in paths:
            tmp.write(f"file '{p.resolve()}'\n")
        list_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-an",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
    finally:
        list_path.unlink(missing_ok=True)
    return out_path


def _mux_audio(video_path: Path, audio_source: Path, out_path: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return out_path


def _burn_subtitles(
    video_path: Path,
    ass_path: Path,
    out_path: Path,
    *,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
) -> Path:
    """Queima legendas e normaliza dimensões pra ``output_width × output_height``.

    O filter chain faz scale + crop centralizado pra garantir que o vídeo
    final fique exatamente em 9:16 (default 1080×1920) — compatível com
    Reels/TikTok/Shorts. Vídeos com aspect-ratio diferente são preenchidos
    cortando as bordas (sem letterbox preto), e o conteúdo central é
    preservado.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, dir=tempfile.gettempdir()
    ) as safe_tmp:
        safe_path = Path(safe_tmp.name)
    safe_path.write_bytes(Path(ass_path).read_bytes())
    try:
        scale_filter = (
            f"scale={output_width}:{output_height}"
            f":force_original_aspect_ratio=increase:flags=lanczos"
        )
        crop_filter = f"crop={output_width}:{output_height}"
        ass_filter = f"ass={safe_path}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"{scale_filter},{crop_filter},{ass_filter}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
    finally:
        safe_path.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose(
    panels: list[Panel],
    panel_results: list[PanelRenderResult],
    transcription: TranscriptionResult,
    video_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    *,
    extend_mode: str = "hold",
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Monta o vídeo final a partir dos painéis renderizados (RF-26..RF-31)."""

    if not panels:
        raise ComposerError("Lista de painéis vazia: nada para compor.")
    if not panel_results:
        raise ComposerError("Lista de PanelRenderResult vazia.")

    by_index = {r.panel_index: r for r in panel_results}
    missing = [p.index for p in panels if p.index not in by_index]
    if missing:
        raise ComposerError(f"Faltam mini-clipes para os painéis: {missing}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "comic" / "_compose"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extend/trim por painel para casar com a janela de áudio.
    adjusted_paths: list[Path] = []
    for panel in sorted(panels, key=lambda p: p.start_time):
        result = by_index[panel.index]
        target = float(panel.end_time - panel.start_time)
        out = work_dir / f"adj_{panel.index:02d}.mp4"
        _extend_or_trim(result.clip_path, target_seconds=target, out_path=out, mode=extend_mode)
        adjusted_paths.append(out)

    # 2) Concat com cortes secos.
    concat_path = work_dir / "concat.mp4"
    _concat_clips(adjusted_paths, concat_path)

    # 3) Mux áudio original.
    muxed_path = work_dir / "muxed.mp4"
    _mux_audio(concat_path, video_path, muxed_path)

    # 4) Legendas palavra-a-palavra.
    output_width = getattr(config, "comic_output_width", OUTPUT_WIDTH)
    output_height = getattr(config, "comic_output_height", OUTPUT_HEIGHT)
    audio_duration = transcription.segments[-1].end if transcription.segments else 0.0
    words = _filter_words_in_range(transcription, 0.0, audio_duration + 1.0)
    ass_doc = build_ass_for_words(
        words, output_size=(output_width, output_height), offset=0.0
    )
    ass_path = work_dir / "captions.ass"
    ass_path.write_text(ass_doc, encoding="utf-8")

    final_path = output_dir / output_name
    _burn_subtitles(
        muxed_path,
        ass_path,
        final_path,
        output_width=output_width,
        output_height=output_height,
    )

    final_duration = _ffprobe_duration(final_path)
    if abs(final_duration - audio_duration) > DURATION_TOLERANCE_SECONDS:
        logger.warning(
            "comic.composer: duração final %.2fs difere do áudio (%.2fs) em %.2fs (tol %.2f)",
            final_duration,
            audio_duration,
            abs(final_duration - audio_duration),
            DURATION_TOLERANCE_SECONDS,
        )

    logger.info("comic.composer: vídeo final em %s (%.2fs)", final_path, final_duration)
    return final_path


def compose_single_video(
    raw_video_path: Path,
    audio_source: Path,
    transcription: TranscriptionResult,
    output_dir: Path,
    config: PipelineConfig,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Composer simplificado para o modo prunaai (single-video).

    Pula o concat de painéis (não há painéis aqui) e aplica diretamente:
    1. mux do áudio original (substitui o áudio do prunaai pelo source);
    2. queima de legendas word-by-word a partir da transcrição;
    3. scale + crop pra 1080×1920 (config.comic_output_*).

    Retorna o path do vídeo final em ``output_dir/output_name``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "comic" / "_compose"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mux áudio original — o prunaai output já tem áudio mas re-mux
    # garante fidelidade ao source.
    muxed_path = work_dir / "muxed.mp4"
    _mux_audio(raw_video_path, audio_source, muxed_path)

    # 2) Legendas palavra-a-palavra.
    output_width = getattr(config, "comic_output_width", OUTPUT_WIDTH)
    output_height = getattr(config, "comic_output_height", OUTPUT_HEIGHT)
    audio_duration = transcription.segments[-1].end if transcription.segments else 0.0
    words = _filter_words_in_range(transcription, 0.0, audio_duration + 1.0)
    ass_doc = build_ass_for_words(
        words, output_size=(output_width, output_height), offset=0.0
    )
    ass_path = work_dir / "captions.ass"
    ass_path.write_text(ass_doc, encoding="utf-8")

    # 3) Burn subtitles + scale 1080×1920.
    final_path = output_dir / output_name
    _burn_subtitles(
        muxed_path,
        ass_path,
        final_path,
        output_width=output_width,
        output_height=output_height,
    )

    final_duration = _ffprobe_duration(final_path)
    if abs(final_duration - audio_duration) > DURATION_TOLERANCE_SECONDS:
        logger.warning(
            "comic.composer.single: duração final %.2fs difere do áudio (%.2fs)",
            final_duration,
            audio_duration,
        )

    logger.info(
        "comic.composer.single: vídeo final em %s (%.2fs)", final_path, final_duration
    )
    return final_path
