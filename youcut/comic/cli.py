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
) -> None:
    """Gera um motion comic 9:16 a partir de um vídeo local (≤120s)."""

    config_overrides: dict = {}
    if max_panels is not None:
        config_overrides["comic_max_panels"] = max_panels
    if cost_cap is not None:
        config_overrides["comic_cost_cap_usd"] = float(cost_cap)

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

    try:
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
