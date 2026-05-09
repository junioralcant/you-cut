"""Subcomando Typer `youcut music` — sincronização da playlist YouTube com o acervo local."""
from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from youcut.config import PipelineConfig
from youcut.models import SyncReport
from youcut.music.classifier import TrackMoodClassifier
from youcut.music.library import MusicLibrary
from youcut.music.sync import PlaylistSyncer

logger = logging.getLogger("youcut.music.cli")
console = Console()

app_music = typer.Typer(
    name="music",
    help="Gerencia o acervo local de trilha sonora (sincronização da playlist YouTube).",
)


@app_music.command("sync", help="Sincroniza a playlist do YouTube com o acervo local.")
def sync(
    playlist: str | None = typer.Option(
        None,
        "--playlist",
        help="URL da playlist do YouTube. Default: youtube_music_playlist_url do .env.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Apenas lista o que seria sincronizado sem baixar nada.",
    ),
) -> None:
    config = PipelineConfig()
    playlist_url = (playlist or config.youtube_music_playlist_url or "").strip()
    if not playlist_url:
        console.print(
            "[red]Nenhuma URL de playlist informada e youtube_music_playlist_url não está configurada.[/red]"
        )
        raise typer.Exit(code=2)

    library = MusicLibrary()
    library.load()

    if dry_run:
        console.print(f"[yellow]Dry-run: seria sincronizada a playlist {playlist_url}[/yellow]")
        console.print(
            f"Acervo atual: {len(library.all_tracks())} faixa(s) em {library.root}."
        )
        raise typer.Exit(code=0)

    classifier = TrackMoodClassifier(config)
    syncer = PlaylistSyncer(library, classifier)
    report = syncer.sync(playlist_url)

    _render_report(report, playlist_url)


def _render_report(report: SyncReport, playlist_url: str) -> None:
    table = Table(title=f"Sync: {playlist_url}")
    table.add_column("Novas", justify="right", style="green")
    table.add_column("Em cache", justify="right", style="cyan")
    table.add_column("Falhas", justify="right", style="red")
    table.add_row(
        str(report.new_tracks),
        str(report.cached_tracks),
        str(report.failed_tracks),
    )
    console.print(table)

    if report.failed_details:
        details = Table(title="Falhas detalhadas")
        details.add_column("video_id", style="red")
        details.add_column("motivo")
        for vid, reason in report.failed_details:
            details.add_row(vid, reason)
        console.print(details)
