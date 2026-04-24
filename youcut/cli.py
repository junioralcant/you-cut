import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from youcut.analyzer import analyze
from youcut.captioner import add_captions
from youcut.clipper import cut_clip
from youcut.config import PipelineConfig
from youcut.downloader import VideoDownloadError, download_video
from youcut.exporter import export_metadata
from youcut.preview import generate_clip_preview
from youcut.transcriber import transcribe

app = typer.Typer(name="youcut", help="Gerador automático de clipes virais a partir de vídeos longos")


@app.callback()
def _main() -> None:
    """YouCut — Gerador automático de clipes virais."""

_console = Console()
_err_console = Console(stderr=True)


def _configure_logging(log_level: str, log_file: Optional[Path]) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        _err_console.print(
            Panel(
                "FFmpeg não encontrado. Instale o FFmpeg e adicione ao PATH antes de continuar.\n"
                "Veja: https://ffmpeg.org/download.html",
                title="[red]Erro: FFmpeg ausente[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


def _show_clips_table(
    clips: list,
    clip_paths: Optional[list[Path]] = None,
    preview_paths: Optional[list[Optional[Path]]] = None,
    dry_run: bool = False,
) -> None:
    title = "Clipes Identificados (Dry Run — sem arquivos gerados)" if dry_run else "Clipes Gerados"
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Título")
    table.add_column("Score", width=8, justify="center")
    if not dry_run:
        table.add_column("Arquivo")
        table.add_column("Preview")

    for i, clip in enumerate(clips):
        row: list[str] = [str(i + 1), clip.title, f"{clip.viral_score:.1f}/10"]
        if not dry_run:
            path = str(clip_paths[i]) if clip_paths and i < len(clip_paths) else "N/A"
            preview = (
                str(preview_paths[i])
                if preview_paths and i < len(preview_paths) and preview_paths[i]
                else "N/A"
            )
            row.append(path)
            row.append(preview)
        table.add_row(*row)

    _console.print(table)


@app.command()
def run(
    source: str = typer.Argument(..., help="URL do YouTube ou caminho de arquivo de vídeo local"),
    clips: int = typer.Option(5, "--clips", "-n", help="Número de clipes a gerar", min=1),
    style: str = typer.Option(
        "word",
        "--style",
        "-s",
        help="Estilo das legendas: 'word' (palavra por palavra) ou 'phrase' (frase completa)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Exibir análise dos trechos sem gerar arquivos de vídeo"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Nível de log: DEBUG, INFO, WARNING, ERROR"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Caminho para salvar o arquivo de log"),
) -> None:
    """Processa um vídeo longo e gera clipes virais prontos para publicação."""
    _configure_logging(log_level, log_file)

    _check_ffmpeg()

    if style not in ("word", "phrase"):
        _err_console.print(
            Panel(
                f"Estilo inválido: '{style}'. Use 'word' ou 'phrase'.",
                title="[red]Erro de Parâmetro[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    try:
        config = PipelineConfig(
            clip_count=clips,
            subtitle_style=style,  # type: ignore[arg-type]
            dry_run=dry_run,
        )
    except Exception as e:
        _err_console.print(
            Panel(
                str(e),
                title="[red]Erro de Configuração[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )

    with progress:
        # Step 1: Download
        task_dl = progress.add_task("Baixando vídeo...", total=None)
        try:
            video_path = download_video(source, config.output_dir / "downloads")
        except VideoDownloadError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Download[/red]", border_style="red"))
            raise typer.Exit(code=1)
        except FileNotFoundError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Arquivo não encontrado[/red]", border_style="red"))
            raise typer.Exit(code=1)
        progress.update(task_dl, description="[green]Vídeo pronto[/green]", completed=1, total=1)

        # Step 2: Transcribe
        task_tr = progress.add_task("Transcrevendo áudio...", total=None)
        try:
            transcription = transcribe(video_path, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Transcrição[/red]", border_style="red"))
            raise typer.Exit(code=1)
        progress.update(task_tr, description="[green]Transcrição concluída[/green]", completed=1, total=1)

        # Step 3: Analyze with AI
        task_ai = progress.add_task("Analisando com IA (Claude)...", total=None)
        try:
            viral_clips = analyze(transcription, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro na Análise IA[/red]", border_style="red"))
            raise typer.Exit(code=1)
        viral_clips = viral_clips[: config.clip_count]
        progress.update(task_ai, description="[green]Análise concluída[/green]", completed=1, total=1)

        if dry_run:
            progress.stop()
            _show_clips_table(viral_clips, dry_run=True)
            return

        # Step 4: Cut clips and generate previews
        clip_paths: list[Path] = []
        preview_paths: list[Optional[Path]] = []
        task_cut = progress.add_task("Cortando clipes...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            try:
                clip_path = cut_clip(video_path, clip, i, config)
                clip_paths.append(clip_path)
            except Exception as e:
                progress.stop()
                _err_console.print(
                    Panel(str(e), title="[red]Erro ao cortar clipe[/red]", border_style="red")
                )
                raise typer.Exit(code=1)
            preview = generate_clip_preview(video_path, clip, i, config)
            preview_paths.append(preview.path if preview else None)
            progress.advance(task_cut)
        progress.update(task_cut, description="[green]Clipes cortados[/green]")

        # Step 5: Add captions
        task_cap = progress.add_task("Adicionando legendas...", total=len(clip_paths))
        for clip_path, clip in zip(clip_paths, viral_clips):
            try:
                add_captions(clip_path, transcription, clip, config)
            except Exception as e:
                progress.stop()
                _err_console.print(
                    Panel(str(e), title="[red]Erro ao adicionar legendas[/red]", border_style="red")
                )
                raise typer.Exit(code=1)
            progress.advance(task_cap)
        progress.update(task_cap, description="[green]Legendas adicionadas[/green]")

        # Step 6: Export metadata
        output_dir = config.output_dir / video_path.stem
        task_exp = progress.add_task("Exportando metadados...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            export_metadata(clip, i, output_dir)
            progress.advance(task_exp)
        progress.update(task_exp, description="[green]Metadados exportados[/green]")

    _show_clips_table(viral_clips, clip_paths=clip_paths, preview_paths=preview_paths)
