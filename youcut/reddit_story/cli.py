"""Subcomando ``youcut reddit-story`` — agora um Typer subapp com:

- ``generate <url>`` — roda o pipeline completo (era o command default antes).
- ``list`` — lista entradas do ``published_videos.json``.
- ``mark-uploaded <session_id>`` — marca uma session como publicada no YouTube.

Comportamento padrão de retrocompat: ``youcut reddit-story <url>`` ainda funciona
graças ao callback default que aceita URL como argumento direto.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from youcut.config import PipelineConfig
from youcut.reddit_story.pipeline import (
    RedditStoryPipelineError,
    run_reddit_story_pipeline,
)
from youcut.reddit_story.published_log import PublishedLog
from youcut.reddit_story.reddit_fetcher import RedditFetchError


logger = logging.getLogger(__name__)
_console = Console()
_err_console = Console(stderr=True)


reddit_story_app = typer.Typer(
    name="reddit-story",
    help="Pipeline long-form 16:9 narrado a partir de threads do Reddit.",
    no_args_is_help=True,
)


@reddit_story_app.command("generate")
def generate(
    url: str = typer.Argument(
        ..., help="URL completa de uma thread do Reddit (text post)."
    ),
    output_dir: Path = typer.Option(
        Path("output/reddit_story"), "--output-dir", "-o",
        help="Diretório base. Subdir por timestamp é criado automaticamente.",
    ),
    skip_thumbnail: bool = typer.Option(
        False, "--skip-thumbnail",
        help="Pula geração das 4 thumbs (economiza ~$0.012).",
    ),
    skip_metadata: bool = typer.Option(
        False, "--skip-metadata",
        help="Pula geração do metadata pack (title + alts + briefs).",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-processa thread mesmo já registrada em published_videos.json.",
    ),
    channel: str = typer.Option(
        "ThreadCourt", "--channel",
        help="Nome do canal alvo (gravado no published_log p/ rastreamento).",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="DEBUG | INFO | WARNING | ERROR"
    ),
) -> None:
    """Gera vídeo long-form a partir de uma URL do Reddit."""
    logging.basicConfig(level=log_level.upper(), format="%(message)s")

    if not os.environ.get("REPLICATE_API_TOKEN"):
        _err_console.print(
            "[red]REPLICATE_API_TOKEN ausente.[/red] Configure no .env "
            "(Kokoro TTS + Flux Schnell)."
        )
        raise typer.Exit(code=2)

    try:
        config = PipelineConfig()
    except Exception as exc:
        _err_console.print(f"[red]Erro carregando PipelineConfig:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    def on_stage(stage: str, message: str) -> None:
        _console.print(f"[cyan]{stage:>10}[/cyan] · {message}")

    try:
        result = run_reddit_story_pipeline(
            url,
            config=config,
            output_root=output_dir,
            on_stage=on_stage,
            generate_thumbnail=not skip_thumbnail,
            generate_metadata=not skip_metadata,
            force=force,
            channel=channel,
        )
    except RedditFetchError as exc:
        _err_console.print(f"[red]Reddit fetch falhou:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    except RedditStoryPipelineError as exc:
        _err_console.print(f"[red]Pipeline abortou:[/red] {exc}")
        raise typer.Exit(code=4) from exc
    except KeyboardInterrupt:
        _err_console.print("[yellow]Interrompido pelo usuário.[/yellow]")
        raise typer.Exit(code=130) from None

    thumb_block = ""
    if result.thumbnails:
        thumb_block = "[green]thumbnails[/green] (escolha 1 pra upload):\n" + "\n".join(
            f"  · {p}" for p in result.thumbnails
        ) + "\n"
    panel = Panel(
        f"[green]final.mp4[/green] → {result.final_video}\n"
        + thumb_block
        + (
            f"[green]metadata.json[/green] → {result.metadata_json}\n"
            if result.metadata_json else ""
        )
        + f"[dim]session_id[/dim] = {result.session.session_id}\n"
        f"[dim]channel[/dim] = {channel}\n"
        f"[dim]registered in[/dim] = ~/.youcut/published_videos.json",
        title="reddit-story · DONE",
        border_style="green",
    )
    _console.print(panel)


@reddit_story_app.command("list")
def list_entries(
    channel: str | None = typer.Option(
        None, "--channel", help="Filtra entries por canal."
    ),
    status: str | None = typer.Option(
        None, "--status",
        help="Filtra por status: 'generated' ou 'uploaded'."
    ),
) -> None:
    """Lista vídeos já gerados/publicados em ~/.youcut/published_videos.json."""
    log = PublishedLog()
    entries = log.list_entries()
    if channel:
        entries = [e for e in entries if e.channel == channel]
    if status:
        entries = [e for e in entries if e.status == status]

    if not entries:
        _console.print("[dim]nenhum vídeo registrado ainda.[/dim]")
        return

    table = Table(title=f"published_videos.json ({len(entries)} entries)")
    table.add_column("session_id", style="cyan", no_wrap=True)
    table.add_column("thread", style="yellow", no_wrap=True)
    table.add_column("subreddit")
    table.add_column("channel", style="magenta")
    table.add_column("status")
    table.add_column("title", overflow="fold")
    for e in entries:
        status_style = "green" if e.status == "uploaded" else "yellow"
        table.add_row(
            e.session_id,
            e.reddit_thread_id,
            f"r/{e.subreddit}",
            e.channel,
            f"[{status_style}]{e.status}[/{status_style}]",
            e.title[:80] + ("…" if len(e.title) > 80 else ""),
        )
    _console.print(table)


@reddit_story_app.command("mark-uploaded")
def mark_uploaded(
    session_id: str = typer.Argument(
        ..., help="session_id do vídeo (timestamp tipo 20260517_141846)."
    ),
    youtube_url: str | None = typer.Option(
        None, "--url",
        help="URL pública do vídeo no YouTube (opcional).",
    ),
) -> None:
    """Marca uma session como published_status=uploaded."""
    log = PublishedLog()
    updated = log.mark_uploaded(session_id, youtube_url=youtube_url)
    if not updated:
        _err_console.print(
            f"[red]Session {session_id!r} não encontrada em published_videos.json[/red]"
        )
        raise typer.Exit(code=1)
    _console.print(
        f"[green]✓[/green] {session_id} marcada como [green]uploaded[/green]"
        + (f" · {youtube_url}" if youtube_url else "")
    )


# Backward compat: o command default antigo era `youcut reddit-story <url>`.
# Mantemos como entry-point principal — quem omite subcommand recebe help.
reddit_story_command = reddit_story_app
