"""Orquestrador do pipeline ``youcut reddit-story``.

Ordem das etapas:
  1. Fetch Reddit thread (httpx + UA descritivo)
  2. Format script (Claude long-form)
  3. TTS (Replicate Kokoro am_adam @ 1.05x)
  4. Whisper word-level timestamps
  5. Plan visual beats (Claude, N=config.reddit_story_scene_count)
  6. Gera N imagens 16:9 (Replicate Flux Schnell)
  7. Render Ken Burns por cena + ASS subs + mux final
  8. (opcional) Thumbnail 1280×720 (Flux + Pillow overlay)
  9. (opcional) Metadata YouTube (title + description + chapters)
  10. Persiste sessão em ~/.youcut/sessions/<id>.json

Persistência intermediária garante que falhas no fim não desperdiçam custo
da IA já consumido (todos os artefatos ficam em ``output/reddit_story/<id>/``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic

from youcut.config import PipelineConfig
from youcut.reddit_story.compositor import (
    build_ass_16x9,
    build_scene_clip,
    compose_long_form,
)
from youcut.reddit_story.metadata import generate_metadata_pack, save_metadata
from youcut.reddit_story.models import RedditStorySession
from youcut.reddit_story.providers import flux_schnell_image, kokoro_tts
from youcut.reddit_story.published_log import PublishedEntry, PublishedLog
from youcut.reddit_story.reddit_fetcher import extract_thread_id, fetch_reddit_thread
from youcut.reddit_story.scene_planner import plan_scenes
from youcut.reddit_story.script_formatter import format_script_with_claude
from youcut.reddit_story.transcriber import transcribe_words

logger = logging.getLogger(__name__)


StageCallback = Callable[[str, str], None]
"""Hook ``(stage, message) -> None`` chamado em cada etapa pra TUI."""


class RedditStoryPipelineError(Exception):
    """Falha fatal em qualquer etapa do pipeline."""


@dataclass
class RedditStoryResult:
    session: RedditStorySession
    final_video: Path
    thumbnails: list[Path]  # lista das N thumbs geradas (1 por variante)
    metadata_json: Path | None


def _noop(stage: str, message: str) -> None:
    logger.info("[%s] %s", stage, message)


_FLUX_PROMPT_SUFFIX = (
    " Horizontal 16:9 framing. Cinematic 35mm, vibrant saturated colors, "
    "dramatic lighting, ultra-detailed. Pure visual, absolutely no text, "
    "no letters, no captions, no logos."
)


def run_reddit_story_pipeline(
    url: str,
    *,
    config: PipelineConfig,
    output_root: Path,
    on_stage: StageCallback | None = None,
    generate_thumbnail: bool = True,
    generate_metadata: bool = True,
    force: bool = False,
    channel: str = "ThreadCourt",
) -> RedditStoryResult:
    """Executa o pipeline end-to-end. Retorna paths dos artefatos finais.

    Dedup: aborta se ``reddit_thread_id`` já foi processado (ver
    ``published_log.json``), a menos que ``force=True``.
    """
    on_stage = on_stage or _noop
    if not config.anthropic_api_key:
        raise RedditStoryPipelineError("ANTHROPIC_API_KEY ausente em PipelineConfig.")
    # Replicate auth vem direto da env var REPLICATE_API_TOKEN

    # 0. Dedup check ---------------------------------------------------
    thread_id = extract_thread_id(url)
    log = PublishedLog()
    existing = log.is_published(thread_id)
    if existing and not force:
        raise RedditStoryPipelineError(
            f"Thread {thread_id} já foi processada anteriormente:\n"
            f"  session: {existing.session_id}\n"
            f"  title: {existing.title!r}\n"
            f"  status: {existing.status}\n"
            f"  video: {existing.video_path}\n"
            f"Use --force pra reprocessar mesmo assim (custo ~$0.63)."
        )
    if existing:
        on_stage(
            "dedup",
            f"⚠ thread {thread_id} já existe (session {existing.session_id}, "
            f"status={existing.status}) — reprocessando por --force",
        )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_root / session_id
    images_dir = work_dir / "images"
    scenes_dir = work_dir / "scene_clips"
    images_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    on_stage("init", f"Session {session_id} → {work_dir}")
    anth = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # 1. Reddit fetch ---------------------------------------------------
    on_stage("fetch", f"Buscando thread {url}")
    t0 = time.time()
    source = fetch_reddit_thread(url, user_agent=config.reddit_story_user_agent)
    (work_dir / "source.json").write_text(source.model_dump_json(indent=2))
    on_stage(
        "fetch",
        f"r/{source.subreddit} · {source.word_count} palavras · {source.ups:,} ups "
        f"({time.time()-t0:.1f}s)",
    )

    session = RedditStorySession(
        session_id=session_id, source=source, work_dir=work_dir
    )

    # 2. Format script --------------------------------------------------
    on_stage(
        "script",
        f"Claude formatando (target {config.reddit_story_target_words} palavras)…",
    )
    t0 = time.time()
    script = format_script_with_claude(
        anth, source,
        model=config.claude_model,
        target_words=config.reddit_story_target_words,
    )
    (work_dir / "script.txt").write_text(script, encoding="utf-8")
    session.script = script
    on_stage(
        "script",
        f"{len(script.split())} palavras geradas ({time.time()-t0:.1f}s)",
    )

    # 3. TTS Kokoro -----------------------------------------------------
    est_min = len(script.split()) / 165
    on_stage(
        "tts",
        f"Kokoro {config.reddit_story_voice} @ {config.reddit_story_speed}x "
        f"narrando ~{est_min:.0f} min…",
    )
    t0 = time.time()
    narration_path = work_dir / "narration.wav"
    kokoro_tts(
        script,
        voice=config.reddit_story_voice,
        speed=config.reddit_story_speed,
        out_path=narration_path,
    )
    session.narration_path = narration_path
    on_stage(
        "tts",
        f"{narration_path.stat().st_size/1024/1024:.1f} MB ({time.time()-t0:.1f}s)",
    )

    # 4. Whisper --------------------------------------------------------
    on_stage("whisper", f"Whisper {config.reddit_story_whisper_model} (English)…")
    t0 = time.time()
    words, duration = transcribe_words(
        narration_path, model_name=config.reddit_story_whisper_model
    )
    session.words = words
    session.narration_duration_s = duration
    (work_dir / "transcript.json").write_text(
        json.dumps([w.model_dump() for w in words], indent=2)
    )
    on_stage(
        "whisper",
        f"{len(words)} words · {duration:.1f}s ({duration/60:.1f} min) "
        f"({time.time()-t0:.1f}s)",
    )

    # 5. Scene plan -----------------------------------------------------
    on_stage(
        "scenes",
        f"Claude dividindo em {config.reddit_story_scene_count} beats visuais…",
    )
    t0 = time.time()
    scenes = plan_scenes(
        anth, script,
        model=config.claude_model,
        count=config.reddit_story_scene_count,
    )
    session.scenes = scenes
    (work_dir / "scenes.json").write_text(
        json.dumps([s.model_dump() for s in scenes], indent=2)
    )
    on_stage("scenes", f"{len(scenes)} cenas planejadas ({time.time()-t0:.1f}s)")
    for i, sc in enumerate(scenes, 1):
        on_stage("scenes", f"  scene {i}: {sc.beat[:80]}")

    # 6. Flux Schnell images --------------------------------------------
    on_stage("images", f"Gerando {len(scenes)} imagens 16:9 via Flux Schnell…")
    t0 = time.time()
    image_paths: list[Path] = []
    for i, sc in enumerate(scenes, 1):
        img_path = images_dir / f"scene_{i:02d}.png"
        ti = time.time()
        flux_schnell_image(
            sc.prompt + _FLUX_PROMPT_SUFFIX,
            aspect_ratio="16:9",
            out_path=img_path,
        )
        on_stage(
            "images",
            f"  [{i}/{len(scenes)}] {time.time()-ti:.1f}s — "
            f"{img_path.stat().st_size/1024:.0f} KB",
        )
        image_paths.append(img_path)
    session.image_paths = image_paths
    on_stage("images", f"{len(scenes)} imagens em {time.time()-t0:.1f}s")

    # 7. Ken Burns + ASS + mux ------------------------------------------
    on_stage("compose", f"Renderizando vídeo {config.reddit_story_resolution_w}×{config.reddit_story_resolution_h}…")
    t0 = time.time()
    scene_duration = duration / len(scenes)
    scene_clip_paths: list[Path] = []
    for i, img in enumerate(image_paths):
        clip = scenes_dir / f"scene_{i+1:02d}.mp4"
        build_scene_clip(
            img, scene_duration, clip,
            idx=i,
            out_w=config.reddit_story_resolution_w,
            out_h=config.reddit_story_resolution_h,
        )
        scene_clip_paths.append(clip)

    ass_path = work_dir / "captions.ass"
    ass_path.write_text(
        build_ass_16x9(
            words,
            res_x=config.reddit_story_resolution_w,
            res_y=config.reddit_story_resolution_h,
        ),
        encoding="utf-8",
    )
    final_video = work_dir / "final.mp4"
    compose_long_form(scene_clip_paths, narration_path, ass_path, final_video, work_dir)
    session.final_video_path = final_video
    on_stage("compose", f"final.mp4 pronto ({time.time()-t0:.1f}s)")

    # 8. Metadata pack (Claude: title + alts + logline + 4 thumb briefs) ─
    metadata_json: Path | None = None
    meta = None
    if generate_metadata or generate_thumbnail:
        on_stage(
            "metadata",
            "Claude gerando title + 3 alt titles + 4 thumb briefs…",
        )
        t0 = time.time()
        meta = generate_metadata_pack(anth, session=session, model=config.claude_model)
        on_stage(
            "metadata",
            f"title='{meta.main_title[:60]}…' · {len(meta.thumb_variants)} thumb briefs ({time.time()-t0:.1f}s)",
        )
        if generate_metadata:
            metadata_json = save_metadata(meta, work_dir)
            session.metadata_path = metadata_json
            on_stage("metadata", f"metadata.json salvo")

    # 9. Thumbnails (4 variantes paralelas) -----------------------------
    thumb_paths: list[Path] = []
    if generate_thumbnail and meta is not None:
        from youcut.reddit_story.thumbnail import generate_thumbnail_set

        thumbs_dir = work_dir / "thumbnails"
        on_stage(
            "thumbnails",
            f"Gerando {len(meta.thumb_variants)} variantes 16:9 via Flux Schnell…",
        )
        t0 = time.time()

        def _thumb_progress(idx: int, total: int, name: str, secs: float) -> None:
            on_stage(
                "thumbnails",
                f"  [{idx}/{total}] {name} ({secs:.1f}s)",
            )

        renders = generate_thumbnail_set(
            meta.thumb_variants,
            subreddit_tag=f"r/{source.subreddit}",
            out_dir=thumbs_dir,
            on_progress=_thumb_progress,
        )
        thumb_paths = [r.path for r in renders]
        # Mantém thumbnail_path apontando pra primeira variante (backward compat)
        session.thumbnail_path = thumb_paths[0] if thumb_paths else None
        on_stage(
            "thumbnails",
            f"{len(thumb_paths)} thumbs prontas em {time.time()-t0:.1f}s",
        )

    # 10. Persiste sessão ───────────────────────────────────────────────
    (work_dir / "session.json").write_text(session.model_dump_json(indent=2))

    # 11. Registra no published_log (dedup futuro) ──────────────────────
    title_used = meta.main_title if meta is not None else source.title
    log.register(
        PublishedEntry(
            session_id=session_id,
            reddit_thread_id=thread_id,
            reddit_url=url,
            subreddit=source.subreddit,
            title=title_used,
            video_path=str(final_video.resolve()),
            channel=channel,
            status="generated",
        )
    )
    on_stage("log", f"registrado em ~/.youcut/published_videos.json (status=generated)")

    on_stage("done", f"final → {final_video}")
    return RedditStoryResult(
        session=session,
        final_video=final_video,
        thumbnails=thumb_paths,
        metadata_json=metadata_json,
    )
