"""Subcomando `youcut comic`."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from youcut.comic.cost_estimator import CostBreakdown
from youcut.comic.pipeline import (
    ComicPipelineError,
    PipelineCallbacks,
    run_comic_pipeline,
)
from youcut.config import PipelineConfig
from youcut.models import CastMember

logger = logging.getLogger(__name__)


_console = Console()
_err_console = Console(stderr=True)


def _parse_panel_indices(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise typer.BadParameter(
                f"Índice de painel inválido: {part!r}. Use números separados por vírgula."
            ) from exc
    return out


def _show_cast_table(cast: list[CastMember]) -> None:
    table = Table(title="Cast detectado")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Tipo")
    table.add_column("Papel")
    table.add_column("Ficha", overflow="fold")
    for member in cast:
        table.add_row(
            member.character_id,
            member.kind,
            member.narrative_role or "—",
            member.text_card or "—",
        )
    _console.print(table)


def _confirm_cast_interactive(cast: list[CastMember]) -> bool:
    if not cast:
        _err_console.print("[red]Nenhum personagem detectado — abortando.[/red]")
        return False
    _show_cast_table(cast)
    return typer.confirm("Aprovar cast e prosseguir?", default=True)


def _confirm_cost_interactive(breakdown: CostBreakdown) -> bool:
    _console.print(
        f"[yellow]Pré-IA:[/yellow] {breakdown.n_panels} painéis · "
        f"[bold]US$ {breakdown.total_usd:.2f}[/bold] estimados "
        f"(âncoras US$ {breakdown.anchor_cost_usd:.2f} · "
        f"imagens US$ {breakdown.base_image_cost_usd:.2f} · "
        f"i2v US$ {breakdown.i2v_cost_usd:.2f})."
    )
    return typer.confirm("Confirmar e iniciar a geração paga?", default=True)


def _make_stage_logger(progress: bool):
    def _on_stage(name: str, payload: dict) -> None:
        if not progress:
            return
        if name == "validate":
            _console.print(f"[blue]» validando vídeo:[/blue] {payload.get('video_path')}")
        elif name == "transcribe":
            _console.print("[blue]» transcrevendo áudio…[/blue]")
        elif name == "diarize":
            _console.print("[blue]» diarizando falantes…[/blue]")
        elif name == "visual_analyzer":
            _console.print("[blue]» detectando cast com Claude vision…[/blue]")
        elif name == "cast_invent":
            _console.print("[blue]» inventando cast a partir da transcrição…[/blue]")
        elif name == "cast_reused":
            _console.print(f"[green]✓ cast reaproveitado da sessão ({payload.get('n')} membros)[/green]")
        elif name == "script_planner":
            _console.print("[blue]» planejando roteiro com Claude…[/blue]")
        elif name == "script_reused":
            _console.print(f"[green]✓ roteiro reaproveitado ({payload.get('n')} painéis)[/green]")
        elif name == "cost_estimate":
            _console.print(
                f"[yellow]» custo estimado:[/yellow] {payload.get('n_panels')} painéis · "
                f"US$ {payload.get('total_usd', 0):.2f}"
            )
        elif name == "render_all":
            _console.print(f"[blue]» renderizando {payload.get('n')} painéis…[/blue]")
        elif name == "render_partial":
            _console.print(
                f"[blue]» regenerando {payload.get('regen')} painéis "
                f"(mantidos {payload.get('kept')})[/blue]"
            )
        elif name == "compose":
            _console.print("[blue]» compondo vídeo final (FFmpeg)…[/blue]")
        elif name == "metadata":
            _console.print("[blue]» gerando metadados por plataforma…[/blue]")
        elif name == "metadata_done":
            _console.print(
                f"[green]✓ metadados:[/green] {payload.get('txt')}"
            )
        elif name == "metadata_failed":
            _console.print(
                f"[yellow]AVISO: falha ao gerar metadados ({payload.get('error')})[/yellow]"
            )
        elif name == "cast_anchors":
            _console.print(
                f"[blue]» gerando âncoras dos {payload.get('n')} personagens…[/blue]"
            )
        elif name == "composition_master":
            _console.print("[blue]» montando composição master da cena…[/blue]")
        elif name == "composition_master_done":
            _console.print(
                f"[green]✓ master:[/green] {payload.get('path')}"
            )
        elif name == "prunaai":
            _console.print(
                "[blue]» gerando animação completa via Prunaai (1 chamada)…[/blue]"
            )
        elif name == "scenes_plan":
            _console.print("[blue]» planejando cenas narrativas (Claude)…[/blue]")
        elif name == "scenes_reused":
            _console.print(f"[green]✓ scenes.json reusado ({payload.get('n')} cenas)[/green]")
        elif name == "scenes_anchor":
            _console.print("[blue]» gerando visual anchor canônico…[/blue]")
        elif name == "scenes_masters":
            _console.print(f"[blue]» gerando masters de {payload.get('n')} cenas…[/blue]")
        elif name == "scenes_attribution":
            _console.print("[blue]» word-level visual attribution (Claude vision)…[/blue]")
        elif name == "scenes_attribution_reused":
            _console.print(f"[green]✓ attribution reusada ({payload.get('n')} palavras)[/green]")
        elif name == "scenes_render":
            _console.print(f"[blue]» renderizando {payload.get('n')} chunks via Prunaai…[/blue]")
        elif name == "scenes_compose":
            _console.print("[blue]» concat com crossfades + mux + finals…[/blue]")
        elif name == "scenes_done_with_subs":
            _console.print(f"[green]✓ vídeo COM legendas:[/green] {payload.get('path')}")
        elif name == "scenes_done_no_subs":
            _console.print(f"[green]✓ vídeo SEM legendas:[/green] {payload.get('path')}")
        elif name == "prunaai_done":
            _console.print(
                f"[green]✓ Prunaai:[/green] {payload.get('size_kb')} KB"
            )
        elif name == "dry_run_done":
            _console.print(f"[green]✓ dry-run concluído:[/green] {payload.get('path')}")
        elif name == "done":
            _console.print(
                f"[bold green]✓ vídeo final:[/bold green] {payload.get('path')} "
                f"(custo total US$ {payload.get('cost', 0):.2f})"
            )

    return _on_stage


def comic_command(
    video: Path = typer.Argument(
        ...,
        help="Caminho local do vídeo (≤120s) — formatos mp4/mov/mkv/webm.",
    ),
    max_panels: int | None = typer.Option(
        None,
        "--max-panels",
        "-n",
        help="Limite máximo de painéis (sobrescreve `comic_max_panels`).",
    ),
    cost_cap: float | None = typer.Option(
        None,
        "--cost-cap",
        help="Teto duro de custo em USD (sobrescreve `comic_cost_cap_usd`).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Apenas transcrição + cast + roteiro; grava `dry_run.json` sem custo de IA.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session",
        help="ID de sessão a retomar (reaproveita cast e painéis prontos).",
    ),
    regenerate_panel: str | None = typer.Option(
        None,
        "--regenerate-panel",
        help="Lista de índices de painel para regenerar (ex.: `2,5`).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Aceita cast e custo automaticamente (não interativo).",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Suprime mensagens de progresso por etapa.",
    ),
    invent_cast: bool = typer.Option(
        False,
        "--invent-cast",
        help=(
            "Inventa personagens fictícios a partir do áudio "
            "(ignora frames do vídeo — não usa rosto real como referência)."
        ),
    ),
    multi_participant: bool = typer.Option(
        False,
        "--multi-participant",
        help=(
            "Exige ≥2 personagens interagindo em todo painel não-narrativo "
            "(e ≥2 narrative_elements em painéis narrativos)."
        ),
    ),
    narrative_only: bool = typer.Option(
        False,
        "--narrative-only",
        help=(
            "Força narrative_mode=true em TODOS os painéis: a animação "
            "encena visualmente o que o áudio narra (cenários e personagens "
            "fictícios construídos), sem mostrar o falante. Ideal combinar "
            "com --invent-cast."
        ),
    ),
    dialogue_mode: bool = typer.Option(
        False,
        "--dialogue-mode",
        help=(
            "Modo diálogo (formato MOTO ANIMADA): cena única com personagens "
            "fixos do cast conversando entre si, com punches de câmera nos "
            "beats cômicos. Mutuamente exclusivo com --narrative-only."
        ),
    ),
    scene: str | None = typer.Option(
        None,
        "--scene",
        help=(
            "Cenário fixo aplicado a TODOS os painéis (ex.: \"dois personagens "
            "em uma motocicleta enferrujada atravessando um deserto pastel "
            "com cactos\"). Sobrescreve o `scene` do script_planner."
        ),
    ),
    composition_image: Path | None = typer.Option(
        None,
        "--composition-image",
        help=(
            "PNG/JPG da composição master da cena: define quem fica onde "
            "(piloto/garupa), direção da moto, framing canônico. Será passada "
            "como 1ª reference_image em todo painel não-narrativo (gpt-image-1 "
            "com input_fidelity=high copia a composição) — fixa o layout."
        ),
    ),
    no_metadata: bool = typer.Option(
        False,
        "--no-metadata",
        help=(
            "Desliga a geração automática de título/descrição/hashtags por "
            "plataforma (TikTok, Reels, Shorts) ao final do pipeline."
        ),
    ),
    engine: str = typer.Option(
        "scenes",
        "--engine",
        help=(
            "Engine de animação. `scenes` (default) divide em N cenas "
            "narrativas com word-level lip-sync via Claude vision (recomendado). "
            "`prunaai` gera o vídeo final em 1 chamada à IA (mais barato mas "
            "sem controle de narrativa). `panels` usa o modo clássico Hailuo "
            "i2v com N painéis individuais. `remotion` renderiza local via "
            "Node + React com lip-sync sílaba-level (custo ≤ $1, determinístico)."
        ),
    ),
    no_preview: bool = typer.Option(
        False,
        "--no-preview",
        help=(
            "Pula o modo preview do engine `remotion` e vai direto para o "
            "render headless. `--yes`/`-y` também implica `--no-preview`."
        ),
    ),
) -> None:
    """Gera um motion comic 9:16 a partir de um vídeo local (≤120s)."""

    config_overrides: dict = {}
    if max_panels is not None:
        config_overrides["comic_max_panels"] = max_panels
    if cost_cap is not None:
        config_overrides["comic_cost_cap_usd"] = float(cost_cap)
    if invent_cast:
        config_overrides["comic_invent_cast"] = True
    if multi_participant:
        config_overrides["comic_enforce_multi_participant"] = True
    if narrative_only and dialogue_mode:
        _err_console.print(
            "[red]--narrative-only e --dialogue-mode são mutuamente exclusivos.[/red]"
        )
        raise typer.Exit(code=2)
    if narrative_only:
        config_overrides["comic_force_narrative_mode"] = True
    if dialogue_mode:
        config_overrides["comic_dialogue_mode"] = True
    if scene:
        config_overrides["comic_scene_seed"] = scene
    if composition_image:
        if not composition_image.exists():
            _err_console.print(
                f"[red]--composition-image não encontrada: {composition_image}[/red]"
            )
            raise typer.Exit(code=2)
        config_overrides["comic_composition_seed_image"] = composition_image
    if no_metadata:
        config_overrides["comic_generate_metadata"] = False
    if engine not in ("prunaai", "panels", "scenes", "remotion"):
        _err_console.print(
            f"[red]--engine inválido: {engine!r} "
            "(use 'scenes', 'prunaai', 'panels' ou 'remotion')[/red]"
        )
        raise typer.Exit(code=2)
    config_overrides["comic_animation_engine"] = engine

    try:
        config = PipelineConfig(**config_overrides)
    except Exception as exc:
        _err_console.print(f"[red]Configuração inválida:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    panel_indices = _parse_panel_indices(regenerate_panel)

    callbacks = PipelineCallbacks(
        confirm_cast=(lambda _c: True) if yes else _confirm_cast_interactive,
        confirm_cost=(lambda _b: True) if yes else _confirm_cost_interactive,
        on_stage=_make_stage_logger(progress=not no_progress),
    )

    # `--yes` implica `--no-preview` para preservar compatibilidade com automação (RF-17).
    effective_preview = not (no_preview or yes)

    try:
        if engine == "remotion":
            from youcut.comic.remotion_pipeline import run_remotion_pipeline

            session = run_remotion_pipeline(
                video,
                config,
                session_id=session_id,
                callbacks=callbacks,
                preview=effective_preview,
                dry_run=dry_run,
            )
        else:
            session = run_comic_pipeline(
                video,
                config,
                session_id=session_id,
                regenerate_panels=panel_indices or None,
                dry_run=dry_run,
                callbacks=callbacks,
            )
    except ComicPipelineError as exc:
        _err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        _err_console.print(
            "[yellow]Interrompido pelo usuário. Use `--session <id>` para retomar.[/yellow]"
        )
        raise typer.Exit(code=130)
    except Exception as exc:
        _err_console.print(f"[red]Erro inesperado:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _console.print(
        f"[bold]Sessão:[/bold] {session.session_id} · "
        f"saída: {session.output_path}"
    )


# Typer standalone para testes via CliRunner. O subcomando real é registrado
# em `youcut/cli.py` via `app.command(name="comic")(comic_command)`.
comic_app = typer.Typer(
    name="comic",
    help="Gera motion comics 9:16 a partir de vídeos curtos (≤120s).",
)
comic_app.command(name="comic")(comic_command)
