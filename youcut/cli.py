import logging
import os
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
from youcut.selector import prompt_clip_selection
from youcut.title_overlay import add_title_overlay
from youcut.transcriber import transcribe
from youcut.uploader import upload_clips
from youcut.uploader.auth import get_token, revoke_token
from youcut.uploader.instagram import InstagramUploader
from youcut.uploader.tiktok import TikTokUploader
from youcut.uploader.youtube import YouTubeUploader

logger = logging.getLogger(__name__)

app = typer.Typer(name="youcut", help="Gerador automático de clipes virais a partir de vídeos longos")
app_auth = typer.Typer(help="Gerencia autenticação das plataformas de upload")
app.add_typer(app_auth, name="auth")


@app.callback()
def _main() -> None:
    """YouCut — Gerador automático de clipes virais."""
    _load_env_file()

_console = Console()
_err_console = Console(stderr=True)
_SUPPORTED_PLATFORMS = ("youtube", "instagram", "tiktok")


def _default_token_dir() -> Path:
    return Path.home() / ".youcut" / "credentials"


def _load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


def _parse_platforms(platforms_raw: str) -> list[str]:
    normalized = platforms_raw.strip().lower()
    if not normalized:
        raise ValueError("`--platforms` não pode ser vazio. Use youtube, instagram, tiktok ou all.")
    if normalized == "all":
        return list(_SUPPORTED_PLATFORMS)

    platforms = sorted({part.strip() for part in normalized.split(",") if part.strip()})
    invalid = [platform for platform in platforms if platform not in _SUPPORTED_PLATFORMS]
    if invalid:
        invalid_str = ", ".join(invalid)
        valid_str = ", ".join(_SUPPORTED_PLATFORMS)
        raise ValueError(f"Plataformas inválidas em `--platforms`: {invalid_str}. Valores aceitos: {valid_str} ou all.")
    return platforms


def _parse_upload_clips(clips_raw: str) -> list[int] | None:
    normalized = clips_raw.strip().lower()
    if not normalized or normalized == "all":
        return None

    values: list[int] = []
    invalid: list[str] = []
    for part in clips_raw.split(","):
        item = part.strip()
        if not item:
            invalid.append("<vazio>")
            continue
        if not item.isdigit():
            invalid.append(item)
            continue
        values.append(int(item))

    if invalid:
        invalid_str = ", ".join(invalid)
        raise ValueError(
            f"Valores inválidos em `--clips`: {invalid_str}. Use `all` ou uma lista de índices numéricos como `1,3,5`."
        )

    return sorted(set(values))


def _resolve_run_clip_options(
    *,
    upload: bool,
    clip_count: int,
    clips_raw: str | None,
) -> tuple[int, list[int] | None]:
    if upload:
        return clip_count, _parse_upload_clips(clips_raw or "all")

    if clips_raw is None:
        return clip_count, None

    legacy_value = clips_raw.strip()
    if not legacy_value:
        raise ValueError("`--clips` não pode ser vazio.")
    if not legacy_value.isdigit():
        raise ValueError(
            "Sem `--upload`, `--clips` mantém o comportamento legado e aceita apenas a quantidade de clipes a gerar."
        )

    return int(legacy_value), None


def _build_uploader(platform: str, token_dir: Path):
    if platform == "youtube":
        client_secrets_file = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE")
        return YouTubeUploader(
            token_dir=token_dir,
            client_secrets_file=Path(client_secrets_file) if client_secrets_file else None,
        )
    if platform == "instagram":
        return InstagramUploader(token_dir=token_dir)
    if platform == "tiktok":
        return TikTokUploader(token_dir=token_dir)
    raise ValueError(f"Unsupported platform: {platform}")


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
    clip_count: int = typer.Option(5, "--clip-count", "--count", "-n", help="Número de clipes a gerar", min=1),
    style: str = typer.Option(
        "word",
        "--style",
        "-s",
        help="Estilo das legendas: 'word' (palavra por palavra) ou 'phrase' (frase completa)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Exibir análise dos trechos sem gerar arquivos de vídeo"
    ),
    title_overlay: bool = typer.Option(
        False, "--title-overlay", help="Queimar título do clipe nos primeiros 5 segundos do vídeo"
    ),
    upload: bool = typer.Option(False, "--upload", help="Faz upload dos clipes ao final do pipeline"),
    platforms_raw: str = typer.Option(
        "all",
        "--platforms",
        help="Plataformas de upload: youtube, instagram, tiktok ou all",
    ),
    upload_clips_raw: Optional[str] = typer.Option(
        None,
        "--clips",
        help="Sem --upload: quantidade legada de clipes. Com --upload: all ou lista separada por vírgula, ex: 1,3,5",
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
        selected_platforms = _parse_platforms(platforms_raw) if upload else list(_SUPPORTED_PLATFORMS)
        resolved_clip_count, clips_filter = _resolve_run_clip_options(
            upload=upload,
            clip_count=clip_count,
            clips_raw=upload_clips_raw,
        )
    except ValueError as e:
        _err_console.print(
            Panel(
                str(e),
                title="[red]Erro de Parâmetro[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    try:
        config = PipelineConfig(
            clip_count=resolved_clip_count,
            subtitle_style=style,  # type: ignore[arg-type]
            dry_run=dry_run,
            title_overlay=title_overlay,
            upload=upload,
            platforms=selected_platforms,
            clips=clips_filter,
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
        progress.update(task_dl, description="[green]Download concluído[/green]", completed=1, total=1)

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

        # Step 5.5: Add title overlay (opt-in)
        if config.title_overlay:
            task_ovl = progress.add_task("Adicionando sobreposição de título...", total=len(clip_paths))
            for clip_path, clip in zip(clip_paths, viral_clips):
                try:
                    add_title_overlay(clip_path, clip, config)
                except Exception as e:
                    progress.stop()
                    _err_console.print(
                        Panel(str(e), title="[red]Erro ao adicionar title overlay[/red]", border_style="red")
                    )
                    raise typer.Exit(code=1)
                progress.advance(task_ovl)
            progress.update(task_ovl, description="[green]Sobreposição de título adicionada[/green]")

        # Step 6: Export metadata
        output_dir = config.output_dir / video_path.stem
        task_exp = progress.add_task("Exportando metadados...", total=len(viral_clips))
        metadata_paths: list[Path] = []
        for i, clip in enumerate(viral_clips):
            metadata_path = export_metadata(clip, i, output_dir)
            metadata_paths.append(metadata_path)
            progress.advance(task_exp)
        progress.update(task_exp, description="[green]Metadados exportados[/green]")

    if upload and upload_clips_raw and upload_clips_raw.strip().lower() not in ("", "all"):
        logger.warning(
            "O valor de --clips (%s) é ignorado quando o seletor interativo está ativo.", upload_clips_raw
        )

    clips_filter = prompt_clip_selection(viral_clips, clip_paths) if config.upload else None

    if config.upload:
        try:
            upload_clips(
                clips=list(zip(clip_paths, metadata_paths)),
                platforms=config.platforms,
                token_dir=_default_token_dir(),
                clips_filter=clips_filter,
            )
        except Exception as e:
            _err_console.print(
                Panel(str(e), title="[red]Erro de Upload[/red]", border_style="red")
            )
            raise typer.Exit(code=1)

    _show_clips_table(viral_clips, clip_paths=clip_paths, preview_paths=preview_paths)


@app_auth.command("login")
def auth_login(
    platform: str = typer.Option(..., "--platform", help="Plataforma: youtube, instagram ou tiktok"),
) -> None:
    """Inicia autenticação da plataforma informada."""
    try:
        normalized_platforms = _parse_platforms(platform)
    except ValueError as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Parâmetro[/red]", border_style="red"))
        raise typer.Exit(code=1)

    if len(normalized_platforms) != 1:
        _err_console.print(
            Panel(
                "`youcut auth login` aceita apenas uma plataforma por vez.",
                title="[red]Erro de Parâmetro[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    selected_platform = normalized_platforms[0]

    try:
        _build_uploader(selected_platform, _default_token_dir()).authenticate()
    except Exception as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Autenticação[/red]", border_style="red"))
        raise typer.Exit(code=1)

    _console.print(f"Autenticação concluída para {selected_platform}.")


@app_auth.command("revoke")
def auth_revoke(
    platform: str = typer.Option(..., "--platform", help="Plataforma: youtube, instagram ou tiktok"),
) -> None:
    """Revoga o token salvo da plataforma informada."""
    try:
        normalized_platforms = _parse_platforms(platform)
    except ValueError as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Parâmetro[/red]", border_style="red"))
        raise typer.Exit(code=1)

    if len(normalized_platforms) != 1:
        _err_console.print(
            Panel(
                "`youcut auth revoke` aceita apenas uma plataforma por vez.",
                title="[red]Erro de Parâmetro[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    selected_platform = normalized_platforms[0]
    revoke_token(selected_platform, _default_token_dir())
    _console.print(f"Token revogado para {selected_platform}.")


@app_auth.command("status")
def auth_status() -> None:
    """Exibe o status de autenticação das plataformas suportadas."""
    token_dir = _default_token_dir()
    table = Table(title="Status de Autenticação", show_header=True, header_style="bold cyan")
    table.add_column("Plataforma")
    table.add_column("Status")

    for platform in _SUPPORTED_PLATFORMS:
        status = "autenticado" if get_token(platform, token_dir) is not None else "não autenticado"
        table.add_row(platform, status)

    _console.print(table)
