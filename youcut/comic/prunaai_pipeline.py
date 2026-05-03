"""Orquestrador alternativo `prunaai` — animação completa em 1 chamada.

Diferente do orquestrador clássico (`pipeline.run_comic_pipeline`) que gera
N painéis individualmente via Hailuo i2v e depois costura, este modo:

1. valida vídeo
2. transcreve áudio
3. diariza falantes
4. extrai (ou inventa) cast
5. gera âncoras 1024×1024 dos personagens
6. **monta uma master composition** (cenário + personagens posicionados)
7. **chama prunaai/p-video-avatar UMA VEZ** com (master, áudio, prompt)
   → recebe o vídeo final inteiro, com lip-sync e camera-punches gerados
   pela própria IA
8. faz mux do áudio original + queima legendas + força 1080×1920
9. gera metadata por plataforma

Custo médio: **~$0.05/vídeo** (vs ~$2 do modo painéis).
Tempo médio: **~70s** (vs ~20min sequenciais do modo painéis).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from youcut.comic.cast_builder import build_cast
from youcut.comic.cast_inventor import invent_cast
from youcut.comic.composer import compose_single_video
from youcut.comic.composition_builder import build_master_composition
from youcut.comic.providers.images import ImageProvider, OpenAIImageProvider
from youcut.comic.providers.prunaai import (
    PrunaaiAnimationError,
    PrunaaiAnimationProvider,
)
from youcut.comic.run_report import write_run_report
from youcut.comic.session import (
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.comic.validator import validate_video
from youcut.comic.visual_analyzer import detect_cast
from youcut.config import PipelineConfig
from youcut.diarizer import diarize
from youcut.models import (
    MotionComicSession,
    SpeakerSegment,
    TranscriptionResult,
)
from youcut.transcriber import transcribe

logger = logging.getLogger(__name__)


class PrunaaiPipelineError(Exception):
    """Erro fatal do pipeline `youcut comic --engine prunaai`."""


def _video_output_dir(video_path: Path, output_root: Path) -> Path:
    return output_root / video_path.stem


def _new_session_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def _extract_audio_for_prunaai(
    source_video: Path, work_dir: Path
) -> Path:
    """Extrai o áudio do vídeo source em MP3 — formato aceito pelo prunaai."""

    import subprocess

    audio_path = work_dir / "_prunaai_audio.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "128k",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def _build_video_prompt(
    cast_descriptions: list[str], scene_seed: str, dialogue_mode: bool = True
) -> str:
    """Constrói o prompt enviado ao prunaai com cast + cenário + diretrizes."""

    cast_block = "\n".join(f"- {d}" for d in cast_descriptions) or "(cast vazio)"

    base = (
        f"Animação cartoon flat 2D pastel: {scene_seed}\n\n"
        f"PERSONAGENS EM CENA (mantenha as posições da imagem master):\n"
        f"{cast_block}\n\n"
        "Estilo OBRIGATÓRIO: contorno preto único espesso (~3px), cores "
        "chapadas, OLHOS ENORMES E REDONDOS com pupila preta marcada "
        "(super-deformed big-eyes), cabeças grandes com corpos pequenos. "
    )

    if dialogue_mode:
        base += (
            "DIÁLOGO: o personagem que está falando articula a boca de forma "
            "sincronizada com o áudio; o(s) outro(s) reage(m) com expressão "
            "facial coerente (espanto, riso, deboche, confusão). Use camera-"
            "punches (close-ups dramáticos) nos beats fortes do diálogo, "
            "alternando com plano médio. "
        )

    base += (
        "PROIBIDO incluir QUALQUER texto, letras, palavras, balões de fala, "
        "legendas, números ou estampas com palavras dentro da imagem. "
        "Sem multidão, sem marcas/logos visíveis."
    )
    return base


def _format_cast_for_prompt(cast: list) -> list[str]:
    descriptions: list[str] = []
    for member in cast:
        snippet = member.text_card or member.narrative_role or member.character_id
        descriptions.append(f"`{member.character_id}`: {snippet[:240]}")
    return descriptions


def run_prunaai_pipeline(
    video_path: Path,
    config: PipelineConfig,
    *,
    session_id: str | None = None,
    callbacks=None,
    image_provider: ImageProvider | None = None,
    animation_provider: PrunaaiAnimationProvider | None = None,
) -> MotionComicSession:
    """Executa o pipeline em modo prunaai.

    Reaproveita ``MotionComicSession`` existente quando ``session_id`` é
    passado (reusa cast e suas âncoras se já materializadas). Master
    composition é gerada idempotentemente — se já existe no path, reusa.
    """

    # Import lazy de PipelineCallbacks pra evitar ciclo
    from youcut.comic.pipeline import PipelineCallbacks

    callbacks = callbacks or PipelineCallbacks()
    callbacks.on_stage("validate", {"video_path": str(video_path)})
    video_spec = validate_video(video_path)

    output_dir = _video_output_dir(video_spec.path, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "comic" / "_prunaai"
    work_dir.mkdir(parents=True, exist_ok=True)

    existing: MotionComicSession | None = None
    if session_id:
        try:
            existing = load_motion_comic_session(session_id)
        except FileNotFoundError as exc:
            raise PrunaaiPipelineError(str(exc)) from exc

    # 1) Transcrição
    callbacks.on_stage("transcribe", {})
    transcription: TranscriptionResult = transcribe(video_spec.path, config)

    # 2) Diarização
    callbacks.on_stage("diarize", {})
    speakers: list[SpeakerSegment] = diarize(video_spec.path, config)

    # 3) Cast (reuso de sessão se cast + anchors estiverem prontos)
    cast_already_built = False
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
        if not callbacks.confirm_cast(cast):
            raise PrunaaiPipelineError(
                "Pipeline abortado pelo usuário durante revisão do cast."
            )

    # 4) Anchors (gpt-image-1 — pago)
    image_provider = image_provider or OpenAIImageProvider(
        api_key=config.openai_api_key,
        max_retries=config.comic_image_retries,
    )
    if not cast_already_built:
        callbacks.on_stage("cast_anchors", {"n": len(cast)})
        cast = build_cast(cast, output_dir, config, image_provider=image_provider)

    # 5) Master composition (idempotente)
    callbacks.on_stage("composition_master", {})
    master_path, scene_seed = build_master_composition(
        cast,
        transcription,
        config,
        output_dir,
        image_provider=image_provider,
    )
    callbacks.on_stage(
        "composition_master_done",
        {"path": str(master_path), "scene": scene_seed[:140]},
    )

    # 6) Prunaai — animação completa em 1 chamada
    callbacks.on_stage("prunaai", {})
    animation_provider = animation_provider or PrunaaiAnimationProvider(
        api_token=config.replicate_api_token,
        max_retries=config.comic_i2v_retries,
    )

    audio_path = _extract_audio_for_prunaai(video_spec.path, work_dir)
    cast_descriptions = _format_cast_for_prompt(cast)
    video_prompt = _build_video_prompt(
        cast_descriptions,
        scene_seed,
        dialogue_mode=config.comic_dialogue_mode,
    )

    raw_video_path = work_dir / "_prunaai_raw.mp4"
    try:
        video_bytes = animation_provider.animate(
            master_path,
            audio_path,
            video_prompt=video_prompt,
            voice_prompt=None,
        )
    except PrunaaiAnimationError as exc:
        raise PrunaaiPipelineError(str(exc)) from exc
    raw_video_path.write_bytes(video_bytes)
    callbacks.on_stage(
        "prunaai_done",
        {"path": str(raw_video_path), "size_kb": len(video_bytes) // 1024},
    )

    # 7) Composer (mux áudio + legendas + scale 1080×1920)
    callbacks.on_stage("compose", {})
    final_path = compose_single_video(
        raw_video_path,
        video_spec.path,
        transcription,
        output_dir,
        config,
    )

    # 8) Metadados editoriais por plataforma
    if config.comic_generate_metadata:
        try:
            from youcut.comic.metadata_generator import (
                MetadataGenerationError,
                generate_metadata,
                write_metadata_files,
            )

            callbacks.on_stage("metadata", {})
            metadata = generate_metadata(
                transcription, cast, config, scene_seed=scene_seed
            )
            json_path, txt_path = write_metadata_files(metadata, output_dir)
            callbacks.on_stage(
                "metadata_done",
                {"json": str(json_path), "txt": str(txt_path)},
            )
        except MetadataGenerationError as exc:
            logger.warning("comic.prunaai_pipeline: falha ao gerar metadados (%s)", exc)
            callbacks.on_stage("metadata_failed", {"error": str(exc)})

    # 9) Persiste sessão (sem painéis — modo prunaai não usa)
    # Custo aproximado: image (anchors+master) + prunaai animation.
    n_anchors_built = sum(
        1
        for m in cast
        if m.anchor_image_path and Path(m.anchor_image_path).exists()
    )
    estimated_cost = round(0.04 * (n_anchors_built + 1) + 0.01, 4)  # +1 = master
    session = MotionComicSession(
        session_id=session_id or _new_session_id(),
        video_path=video_spec.path,
        created_at=datetime.now(timezone.utc),
        cast=cast,
        panels=[],
        panel_results=[],
        total_cost_usd=estimated_cost,
        output_path=final_path,
    )
    save_motion_comic_session(session)
    write_run_report(session, output_dir)
    callbacks.on_stage(
        "done", {"path": str(final_path), "cost": estimated_cost}
    )
    return session
