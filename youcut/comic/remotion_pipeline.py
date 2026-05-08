"""Orquestrador do engine `youcut comic --engine remotion`.

Pipeline:
    validate → transcribe → diarize → cast (detect/invent) → plan_panels
    → estimate_cost → preflight → build_cast_anchors → build_mouth_sheets
    → derive_lipsync (per scene) → RemotionInputProps
    → preview (open_studio) ? → render (subprocess Node) → compose

A composição final reusa `composer.compose_from_single_clip` que queima
legendas + watermark e emite duas versões (`motion_comic.mp4`,
`motion_comic_no_subs.mp4`).

Em ambiente sem TTY/DISPLAY o pipeline cai automaticamente em modo
headless (RF-18) com warning em pt-BR.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from youcut.comic.cast_builder import build_cast
from youcut.comic.cast_inventor import invent_cast
from youcut.comic.composer import compose_from_single_clip
from youcut.comic.cost_estimator import (
    CostCapExceededError,
    estimate_cost,
    preflight,
)
from youcut.comic.mouth_shapes import build_mouth_sheet
from youcut.comic.pipeline import (
    ComicPipelineError,
    PipelineCallbacks,
    _build_default_image_provider,
    _new_session_id,
    _video_output_dir,
)
from youcut.comic.providers.images import ImageProvider
from youcut.comic.providers.remotion_renderer import RemotionRenderer
from youcut.comic.script_planner import plan_panels
from youcut.comic.session import (
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.comic.syllable_mapper import derive_lipsync_track
from youcut.comic.validator import validate_video
from youcut.comic.visual_analyzer import detect_cast
from youcut.config import PipelineConfig
from youcut.diarizer import diarize
from youcut.models import (
    CastMember,
    MotionComicSession,
    MouthEvent,
    MouthShape,
    Panel,
    RemotionInputProps,
    RemotionScene,
    SpeakerSegment,
    TranscriptionResult,
    WordTimestamp,
)
from youcut.transcriber import transcribe

logger = logging.getLogger(__name__)


REMOTION_PROJECT_DIR: Path = Path(__file__).resolve().parent / "remotion_project"


# ── Helpers ────────────────────────────────────────────────────────────────


def _is_interactive_environment() -> bool:
    """True quando há terminal interativo *e* algum display gráfico viável."""
    try:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
    except (ValueError, OSError):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    # macOS / Windows assumem que o desktop está disponível quando há TTY.
    return True


def _resolve_speaker_for_panel(
    panel: Panel,
    cast: list[CastMember],
    speakers: list[SpeakerSegment],
) -> str | None:
    """Tenta inferir qual character_id está falando no painel."""
    if not panel.participants:
        return None
    participants = panel.participants
    speaker_diarization_id: str | None = None
    if speakers:
        overlap = max(
            speakers,
            key=lambda s: max(0.0, min(s.end, panel.end_time) - max(s.start, panel.start_time)),
            default=None,
        )
        if overlap is not None:
            speaker_diarization_id = overlap.speaker_id
    if speaker_diarization_id:
        for member in cast:
            if member.speaker_id == speaker_diarization_id and member.character_id in participants:
                return member.character_id
    return participants[0]


def _words_in_panel(transcription: TranscriptionResult, panel: Panel) -> list[WordTimestamp]:
    out: list[WordTimestamp] = []
    for seg in transcription.segments:
        for w in seg.words:
            if w.start >= panel.start_time and w.start < panel.end_time:
                out.append(w)
    return out


def _shift_word(word: WordTimestamp, offset: float) -> WordTimestamp:
    return WordTimestamp(
        word=word.word,
        start=max(0.0, word.start - offset),
        end=max(0.0, word.end - offset),
    )


def _extract_audio(video_path: Path, dest: Path) -> Path:
    """Extrai a trilha de áudio do vídeo de input para um AAC standalone.

    Garante que o `<Audio src=...>` do Remotion receba um arquivo "puro"
    de áudio (alguns players Remotion têm dificuldade com MP4 contendo vídeo).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dest),
    ]
    completed = subprocess.run(cmd, capture_output=True)
    if completed.returncode != 0:
        raise ComicPipelineError(
            "Falha ao extrair áudio do vídeo de entrada: "
            f"{completed.stderr.decode('utf-8', 'replace')}"
        )
    return dest


def _build_remotion_scenes(
    panels: list[Panel],
    cast: list[CastMember],
    speakers: list[SpeakerSegment],
    transcription: TranscriptionResult,
    *,
    config: PipelineConfig,
) -> list[RemotionScene]:
    """Converte `panels` em `RemotionScene`s com lip-sync derivado por panel."""
    scenes: list[RemotionScene] = []
    locale = (transcription.language or "pt").lower()
    if locale.startswith("pt"):
        primary_locale = "pt_BR"
    elif locale.startswith("en"):
        primary_locale = "en_US"
    else:
        primary_locale = config.comic_remotion_pyphen_locale_fallback

    for idx, panel in enumerate(sorted(panels, key=lambda p: p.start_time)):
        speaker_id = _resolve_speaker_for_panel(panel, cast, speakers)
        words = _words_in_panel(transcription, panel)
        relative_words = [_shift_word(w, panel.start_time) for w in words]

        if speaker_id and relative_words:
            lipsync_events = derive_lipsync_track(
                relative_words,
                character_id=speaker_id,
                locale=primary_locale,
                fallback_locale=config.comic_remotion_pyphen_locale_fallback,
            )
        else:
            lipsync_events = []

        scenes.append(
            RemotionScene(
                index=idx,
                start_sec=panel.start_time,
                end_sec=panel.end_time,
                character_ids=list(panel.participants),
                speaker_id=speaker_id,
                ken_burns={
                    "scale_from": 1.0,
                    "scale_to": config.comic_remotion_kenburns_default_scale,
                    "from": [0, 0],
                    "to": [0, 0],
                },
                transition_in="cut" if idx == 0 else "crossfade",
                shakes=[],
                lip_sync=lipsync_events,
            )
        )
    return scenes


def _build_input_props(
    *,
    audio_path: Path,
    duration_sec: float,
    cast: list[CastMember],
    mouth_sheets: dict,
    scenes: list[RemotionScene],
    config: PipelineConfig,
) -> RemotionInputProps:
    characters: dict[str, dict] = {}
    for member in cast:
        sheet = mouth_sheets.get(member.character_id)
        if sheet is None:
            continue
        characters[member.character_id] = {
            "anchor_path": str(member.anchor_image_path) if member.anchor_image_path else None,
            "mouth_sheet_path": str(sheet.sheet_path),
            "cells": {
                shape.value: list(box) for shape, box in sheet.cells.items()
            },
        }
    return RemotionInputProps(
        audio_path=str(audio_path),
        duration_sec=duration_sec,
        fps=config.comic_remotion_fps,
        width=config.comic_output_width,
        height=config.comic_output_height,
        characters=characters,
        scenes=scenes,
        background_color="#000000",
    )


# ── API pública ────────────────────────────────────────────────────────────


def run_remotion_pipeline(
    video_path: Path,
    config: PipelineConfig,
    *,
    session_id: str | None = None,
    callbacks: PipelineCallbacks | None = None,
    image_provider: ImageProvider | None = None,
    renderer: RemotionRenderer | None = None,
    preview: bool = True,
    dry_run: bool = False,
) -> MotionComicSession:
    """Executa o pipeline ponta-a-ponta do engine remotion."""
    callbacks = callbacks or PipelineCallbacks()

    callbacks.on_stage("validate", {"video_path": str(video_path)})
    video_spec = validate_video(video_path)

    output_dir = _video_output_dir(video_spec.path, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: MotionComicSession | None = None
    if session_id:
        try:
            existing = load_motion_comic_session(session_id)
        except FileNotFoundError as exc:
            raise ComicPipelineError(str(exc)) from exc

    callbacks.on_stage("transcribe", {})
    transcription: TranscriptionResult = transcribe(video_spec.path, config)

    callbacks.on_stage("diarize", {})
    speakers: list[SpeakerSegment] = diarize(video_spec.path, config)

    if existing and existing.cast:
        cast = existing.cast
        cast_already_built = all(
            m.anchor_image_path is not None
            and Path(m.anchor_image_path).exists()
            and Path(m.anchor_image_path).stat().st_size > 0
            for m in cast
        )
        callbacks.on_stage("cast_reused", {"n": len(cast)})
    else:
        if config.comic_invent_cast:
            callbacks.on_stage("cast_invent", {})
            cast = invent_cast(transcription, speakers, config)
        else:
            callbacks.on_stage("visual_analyzer", {})
            cast = detect_cast(
                video_spec.path, transcription, speakers, config, output_dir=output_dir
            )
        cast_already_built = False
        if not callbacks.confirm_cast(cast):
            raise ComicPipelineError(
                "Pipeline abortado pelo usuário durante revisão do cast."
            )

    if existing and existing.panels:
        panels = existing.panels
        callbacks.on_stage("script_reused", {"n": len(panels)})
    else:
        callbacks.on_stage("script_planner", {})
        panels = plan_panels(transcription, cast, speakers, config)

    breakdown = estimate_cost(cast, panels, config)
    callbacks.on_stage(
        "cost_estimate",
        {
            "engine": "remotion",
            "n_cast": breakdown.n_cast,
            "total_usd": breakdown.total_usd,
        },
    )

    if dry_run:
        session = MotionComicSession(
            session_id=session_id or _new_session_id(),
            video_path=video_spec.path,
            created_at=datetime.now(timezone.utc),
            cast=cast,
            panels=panels,
            panel_results=[],
            total_cost_usd=0.0,
            output_path=None,
        )
        save_motion_comic_session(session)
        callbacks.on_stage("dry_run_done", {"total_usd": breakdown.total_usd})
        return session

    if not callbacks.confirm_cost(breakdown):
        raise ComicPipelineError(
            "Pipeline abortado pelo usuário ao revisar o custo estimado."
        )

    try:
        preflight(cast, panels, config)
    except CostCapExceededError as exc:
        raise ComicPipelineError(str(exc)) from exc

    image_provider = image_provider or _build_default_image_provider(config)

    callbacks.on_stage("cast_anchors", {"n": len(cast)})
    if not cast_already_built:
        cast = build_cast(cast, output_dir, config, image_provider=image_provider)

    callbacks.on_stage("mouth_sheets", {"n": len(cast)})
    mouth_sheets: dict = {}
    for member in cast:
        if not member.anchor_image_path:
            logger.warning(
                "comic.remotion_pipeline: cast %s sem âncora — pulando mouth sheet",
                member.character_id,
            )
            continue
        sheet = build_mouth_sheet(
            member,
            Path(member.anchor_image_path),
            output_dir,
            image_provider=image_provider,
        )
        mouth_sheets[member.character_id] = sheet

    callbacks.on_stage("lipsync_derive", {})
    scenes = _build_remotion_scenes(
        panels, cast, speakers, transcription, config=config
    )

    audio_path = _extract_audio(
        video_spec.path,
        output_dir / "comic" / "_compose" / f"{video_spec.path.stem}_audio.aac",
    )

    props = _build_input_props(
        audio_path=audio_path,
        duration_sec=video_spec.duration_seconds,
        cast=cast,
        mouth_sheets=mouth_sheets,
        scenes=scenes,
        config=config,
    )

    renderer = renderer or RemotionRenderer(
        REMOTION_PROJECT_DIR, node_bin=config.comic_remotion_node_bin
    )

    effective_preview = preview
    if preview and not _is_interactive_environment():
        logger.warning(
            "comic.remotion_pipeline: ambiente sem TTY/DISPLAY detectado — "
            "caindo em modo headless (preview desligado)."
        )
        callbacks.on_stage("preview_fallback_headless", {})
        effective_preview = False

    if effective_preview:
        callbacks.on_stage("preview_open", {"port": config.comic_remotion_studio_port})
        renderer.open_studio(props, port=config.comic_remotion_studio_port)

    callbacks.on_stage("render", {"n_scenes": len(scenes)})
    raw_render = output_dir / "comic" / "_compose" / "remotion_raw.mp4"
    raw_render.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(
        props,
        raw_render,
        on_progress=lambda p: callbacks.on_stage("render_progress", {"progress": p}),
    )

    callbacks.on_stage("compose", {})
    no_subs_path, with_subs_path = compose_from_single_clip(
        raw_render, transcription, output_dir, config
    )

    total_cost = breakdown.total_usd
    session = MotionComicSession(
        session_id=session_id or _new_session_id(),
        video_path=video_spec.path,
        created_at=datetime.now(timezone.utc),
        cast=cast,
        panels=panels,
        panel_results=[],
        total_cost_usd=total_cost,
        output_path=with_subs_path,
    )
    save_motion_comic_session(session)
    callbacks.on_stage(
        "done",
        {
            "path": str(with_subs_path),
            "no_subs_path": str(no_subs_path),
            "cost": total_cost,
        },
    )
    return session
