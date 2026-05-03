"""Orquestrador do pipeline `youcut comic`.

Combina validador → transcrição → diarização → visual analyzer → cast
builder → script planner → cost estimator → panel renderer → composer.

Suporta retomada via ``MotionComicSession`` e regeneração granular de
painéis.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from youcut.comic.cast_builder import build_cast
from youcut.comic.cast_inventor import invent_cast
from youcut.comic.composer import compose
from youcut.comic.run_report import write_run_report
from youcut.comic.cost_estimator import (
    CostBreakdown,
    CostCapExceededError,
    estimate_cost,
    preflight,
)
from youcut.comic.panel_renderer import render_all
from youcut.comic.providers.i2v import ImageToVideoProvider, RunwayProvider
from youcut.comic.providers.images import ImageProvider, OpenAIImageProvider
from youcut.comic.script_planner import ScriptPlanError, plan_panels
from youcut.comic.session import (
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.comic.validator import VideoSpec, validate_video
from youcut.comic.visual_analyzer import detect_cast
from youcut.config import PipelineConfig
from youcut.diarizer import diarize
from youcut.models import (
    CastMember,
    MotionComicSession,
    Panel,
    PanelRenderResult,
    SpeakerSegment,
    TranscriptionResult,
)
from youcut.transcriber import transcribe

logger = logging.getLogger(__name__)


class ComicPipelineError(Exception):
    """Erro fatal do pipeline `youcut comic` (mensagens em pt-BR)."""


@dataclass
class PipelineCallbacks:
    """Hooks de UX para o CLI customizar comportamento.

    Todos os callbacks são opcionais; defaults são no-op ou seguem o
    contrato simples (confirmação automática, dry-run preserva apenas o
    JSON de roteiro).
    """

    confirm_cast: Callable[[list[CastMember]], bool] = field(
        default_factory=lambda: lambda _cast: True
    )
    confirm_cost: Callable[[CostBreakdown], bool] = field(
        default_factory=lambda: lambda _b: True
    )
    on_stage: Callable[[str, dict], None] = field(default_factory=lambda: lambda *_args, **_kw: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_output_dir(video_path: Path, output_root: Path) -> Path:
    name = video_path.stem
    return output_root / name


def _build_default_image_provider(config: PipelineConfig) -> ImageProvider:
    return OpenAIImageProvider(
        api_key=config.openai_api_key,
        max_retries=config.comic_image_retries,
    )


def _build_default_i2v_provider(config: PipelineConfig) -> ImageToVideoProvider:
    if config.comic_i2v_provider == "fal":
        from youcut.comic.providers.i2v_fal import FalImageToVideoProvider

        return FalImageToVideoProvider(
            api_key=config.fal_api_key,
            model=config.comic_i2v_fal_model,
            max_retries=config.comic_i2v_retries,
        )
    if config.comic_i2v_provider == "replicate":
        from youcut.comic.providers.i2v_replicate import ReplicateImageToVideoProvider

        return ReplicateImageToVideoProvider(
            api_token=config.replicate_api_token,
            model=config.comic_i2v_replicate_model,
            max_retries=config.comic_i2v_retries,
        )
    return RunwayProvider(
        api_key=config.runway_api_key,
        max_retries=config.comic_i2v_retries,
        max_poll_time=config.comic_i2v_max_poll_seconds,
    )


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _write_dry_run_json(
    output_dir: Path,
    *,
    video_spec: VideoSpec,
    cast: list[CastMember],
    panels: list[Panel],
    breakdown: CostBreakdown,
) -> Path:
    dry_dir = output_dir / "comic"
    dry_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": {
            "path": str(video_spec.path),
            "duration_seconds": video_spec.duration_seconds,
            "size": [video_spec.width, video_spec.height],
        },
        "cast": [c.model_dump(mode="json") for c in cast],
        "panels": [p.model_dump(mode="json") for p in panels],
        "estimate": {
            "n_cast": breakdown.n_cast,
            "n_panels": breakdown.n_panels,
            "total_usd": breakdown.total_usd,
            "anchor_cost_usd": breakdown.anchor_cost_usd,
            "base_image_cost_usd": breakdown.base_image_cost_usd,
            "i2v_cost_usd": breakdown.i2v_cost_usd,
        },
    }
    out = dry_dir / "dry_run.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _filter_panels_for_regenerate(
    panels: list[Panel],
    panel_results: list[PanelRenderResult],
    indices: Iterable[int],
) -> tuple[list[Panel], list[PanelRenderResult]]:
    target = set(int(i) for i in indices)
    keep_existing = [r for r in panel_results if r.panel_index not in target]
    panels_to_regenerate = [p for p in panels if p.index in target]
    return panels_to_regenerate, keep_existing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_comic_pipeline(
    video_path: Path,
    config: PipelineConfig,
    *,
    session_id: str | None = None,
    regenerate_panels: list[int] | None = None,
    dry_run: bool = False,
    callbacks: PipelineCallbacks | None = None,
    image_provider: ImageProvider | None = None,
    i2v_provider: ImageToVideoProvider | None = None,
) -> MotionComicSession:
    """Executa o pipeline completo (ou trecho parcial via flags) e retorna a sessão final.

    - ``dry_run=True``: roda até script_planner e grava ``dry_run.json``.
    - ``session_id`` carregado: reaproveita cast e painéis prontos; regenera apenas o que faltar.
    - ``regenerate_panels=[i,...]``: força regeneração somente desses índices.

    Quando ``config.comic_animation_engine == "prunaai"`` (default), delega
    ao orquestrador alternativo em :mod:`youcut.comic.prunaai_pipeline` —
    que gera o vídeo final em 1 chamada à IA (mais barato e mais rápido).
    O modo ``"panels"`` mantém o pipeline clássico (Hailuo i2v por painel).
    """

    if config.comic_animation_engine == "scenes" and not dry_run and not regenerate_panels:
        from youcut.comic.scenes_pipeline import run_scenes_pipeline

        return run_scenes_pipeline(
            video_path,
            config,
            session_id=session_id,
            callbacks=callbacks,
            image_provider=image_provider,
        )

    if config.comic_animation_engine == "prunaai" and not dry_run and not regenerate_panels:
        from youcut.comic.prunaai_pipeline import run_prunaai_pipeline

        return run_prunaai_pipeline(
            video_path,
            config,
            session_id=session_id,
            callbacks=callbacks,
            image_provider=image_provider,
        )

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

    # 1) Transcrição
    callbacks.on_stage("transcribe", {})
    transcription: TranscriptionResult = transcribe(video_spec.path, config)

    # 2) Diarização
    callbacks.on_stage("diarize", {})
    speakers: list[SpeakerSegment] = diarize(video_spec.path, config)

    # 3) Detecção de cast (sem gerar âncoras ainda — gratuito além do Claude vision)
    if existing and existing.cast and not regenerate_panels:
        cast = existing.cast
        # Só pula `build_cast` se TODAS as âncoras realmente existirem no disco.
        # Sessões de dry-run preservam o cast com `anchor_image_path=None` e,
        # nesse caso, o build precisa rodar pra materializar as fichas.
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
            raise ComicPipelineError("Pipeline abortado pelo usuário durante revisão do cast.")

    # 4) Roteiro (gratuito além do Claude texto)
    if existing and existing.panels and not regenerate_panels:
        panels = existing.panels
        callbacks.on_stage("script_reused", {"n": len(panels)})
    else:
        callbacks.on_stage("script_planner", {})
        panels = plan_panels(transcription, cast, speakers, config)

    # 5) Custo + dry-run
    breakdown = estimate_cost(cast, panels, config)
    callbacks.on_stage("cost_estimate", {"total_usd": breakdown.total_usd, "n_panels": breakdown.n_panels})

    if dry_run:
        dry_path = _write_dry_run_json(
            output_dir, video_spec=video_spec, cast=cast, panels=panels, breakdown=breakdown
        )
        session = MotionComicSession(
            session_id=session_id or _new_session_id(),
            video_path=video_spec.path,
            created_at=datetime.now(timezone.utc),
            cast=cast,
            panels=panels,
            panel_results=[],
            total_cost_usd=0.0,
            output_path=dry_path,
        )
        save_motion_comic_session(session)
        callbacks.on_stage("dry_run_done", {"path": str(dry_path)})
        return session

    if not callbacks.confirm_cost(breakdown):
        raise ComicPipelineError("Pipeline abortado pelo usuário ao revisar o custo estimado.")

    try:
        preflight(cast, panels, config)
    except CostCapExceededError as exc:
        raise ComicPipelineError(str(exc)) from exc

    # 5b) Gerar âncoras (pago — apenas se cast ainda não foi materializado).
    image_provider = image_provider or _build_default_image_provider(config)
    if not cast_already_built:
        cast = build_cast(cast, output_dir, config, image_provider=image_provider)

    # 6) Render
    i2v_provider = i2v_provider or _build_default_i2v_provider(config)

    if regenerate_panels and existing:
        target_panels, kept = _filter_panels_for_regenerate(
            panels, existing.panel_results, regenerate_panels
        )
        callbacks.on_stage("render_partial", {"regen": len(target_panels), "kept": len(kept)})
        new_results = render_all(
            target_panels,
            cast,
            output_dir,
            config,
            image_provider=image_provider,
            i2v_provider=i2v_provider,
            transcription=transcription,
            speakers=speakers,
        )
        panel_results = sorted(kept + new_results, key=lambda r: r.panel_index)
    else:
        callbacks.on_stage("render_all", {"n": len(panels)})
        panel_results = render_all(
            panels,
            cast,
            output_dir,
            config,
            image_provider=image_provider,
            i2v_provider=i2v_provider,
            transcription=transcription,
            speakers=speakers,
        )

    # 7) Composer
    callbacks.on_stage("compose", {})
    final_path = compose(
        panels,
        panel_results,
        transcription,
        video_spec.path,
        output_dir,
        config,
    )

    # 8) Metadados editoriais por plataforma (TikTok, Reels, Shorts).
    if config.comic_generate_metadata:
        try:
            callbacks.on_stage("metadata", {})
            from youcut.comic.metadata_generator import (
                MetadataGenerationError,
                generate_metadata,
                write_metadata_files,
            )

            metadata = generate_metadata(
                transcription,
                cast,
                config,
                scene_seed=config.comic_scene_seed,
            )
            json_path, txt_path = write_metadata_files(metadata, output_dir)
            callbacks.on_stage(
                "metadata_done",
                {"json": str(json_path), "txt": str(txt_path)},
            )
        except MetadataGenerationError as exc:
            logger.warning("comic.pipeline: falha ao gerar metadados (%s)", exc)
            callbacks.on_stage("metadata_failed", {"error": str(exc)})

    total_cost = round(sum(r.cost_usd for r in panel_results), 4)
    session = MotionComicSession(
        session_id=session_id or _new_session_id(),
        video_path=video_spec.path,
        created_at=datetime.now(timezone.utc),
        cast=cast,
        panels=panels,
        panel_results=panel_results,
        total_cost_usd=total_cost,
        output_path=final_path,
    )
    save_motion_comic_session(session)
    write_run_report(session, output_dir)
    callbacks.on_stage("done", {"path": str(final_path), "cost": total_cost})
    return session
