import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from youcut.analyzer import YOUTUBE_MIN_DURATION, analyze
from youcut.caption_burner import CaptionBurner
from youcut.captioner import add_captions
from youcut.clipper import cut_clip
from youcut.face_tracker import apply_face_tracking, frame_for_panel
from youcut.config import PipelineConfig
from youcut.downloader import VideoDownloadError, download_video
from youcut.exporter import export_metadata
from youcut.models import CaptionBurnResult, ClipRecord, CutMode, MusicTrack, SessionData, TranscriptionResult, ViralClip
from youcut.music_mixer import MusicMixer
from youcut.music_provider import PixabayMusicProvider
from youcut.preview import generate_clip_preview
from youcut.reviewer import review_clips
from youcut.selector import prompt_clip_selection
from youcut.session_store import list_sessions, save_session
from youcut.social_composer import compose_social_clip
from youcut.thumbnail_generator import generate_thumbnail
from youcut.title_overlay import add_title_overlay
from youcut.transcriber import transcribe
from youcut.uploader import upload_clips
from youcut.uploader.auth import get_token, revoke_token
from youcut.uploader.base import ClipMetadata
from youcut.uploader.instagram import InstagramUploader
from youcut.uploader.tiktok import TikTokUploader
from youcut.uploader.youtube import YouTubeUploader
from youcut.url_utils import normalize_video_url
from youcut.video_metadata import VideoMetadata, VideoMetadataError, fetch_metadata
from youcut.yt_dlp_auth import YtDlpAuthConfig, YtDlpAuthConfigError, resolve_yt_dlp_auth_config

logger = logging.getLogger(__name__)

app = typer.Typer(name="youcut", help="Gerador automático de clipes virais a partir de vídeos longos")
app_auth = typer.Typer(help="Gerencia autenticação das plataformas de upload")
app.add_typer(app_auth, name="auth")

from youcut.comic.cli import comic_command  # noqa: E402

app.command(name="comic", help="Gera motion comics 9:16 a partir de vídeos curtos (≤120s).")(comic_command)


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


def _resolve_run_clip_options(
    *,
    clip_count: int,
    clips_raw: str | None,
) -> tuple[int, list[int] | None]:
    if clips_raw is None:
        return clip_count, None

    clip_count_value = clips_raw.strip()
    if not clip_count_value:
        raise ValueError("`--clips` não pode ser vazio.")
    if not clip_count_value.isdigit():
        raise ValueError("`--clips` deve ser um número inteiro com a quantidade de clipes a gerar.")

    return int(clip_count_value), None


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


def _resolve_cli_yt_dlp_auth_config() -> YtDlpAuthConfig | None:
    try:
        return resolve_yt_dlp_auth_config()
    except YtDlpAuthConfigError as exc:
        _err_console.print(
            Panel(str(exc), title="[red]Erro de Configuração yt-dlp[/red]", border_style="red")
        )
        raise typer.Exit(code=1) from exc


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


class _PipelineError(Exception):
    """Sinaliza falha interna do pipeline; capturada por chamadores para ação adequada."""


def _run_single_source_pipeline(
    source: str,
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
    auth_config: YtDlpAuthConfig | None = None,
) -> list[ClipRecord]:
    """Executa o pipeline completo do `run` para uma fonte (URL ou path local)."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )

    viral_clips: list[ViralClip] = []
    clip_paths: list[Path] = []
    preview_paths: list[Optional[Path]] = []
    metadata_paths: list[Path] = []
    captions_applied_flags: list[bool] = []
    caption_warnings: list[str | None] = []
    captions_already_burned: list[bool] = []

    with progress:
        task_dl = progress.add_task("Baixando vídeo...", total=None)
        try:
            if auth_config is None:
                video_path = download_video(source, config.output_dir / "downloads")
            else:
                video_path = download_video(source, config.output_dir / "downloads", auth_config=auth_config)
        except VideoDownloadError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Download[/red]", border_style="red"))
            raise _PipelineError(str(e)) from e
        except FileNotFoundError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Arquivo não encontrado[/red]", border_style="red"))
            raise _PipelineError(str(e)) from e
        progress.update(task_dl, description="[green]Download concluído[/green]", completed=1, total=1)

        task_tr = progress.add_task("Transcrevendo áudio...", total=None)
        try:
            transcription = transcribe(video_path, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Transcrição[/red]", border_style="red"))
            raise _PipelineError(str(e)) from e
        progress.update(task_tr, description="[green]Transcrição concluída[/green]", completed=1, total=1)

        task_ai = progress.add_task("Analisando com IA (Claude)...", total=None)
        try:
            viral_clips = analyze(transcription, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro na Análise IA[/red]", border_style="red"))
            raise _PipelineError(str(e)) from e
        viral_clips = viral_clips[: config.clip_count]
        progress.update(task_ai, description="[green]Análise concluída[/green]", completed=1, total=1)

        if not viral_clips:
            progress.stop()
            _console.print("[yellow]Nenhum trecho relevante identificado.[/yellow]")
            return []
        if config.cut_mode == "youtube":
            _show_youtube_duration_guidance(viral_clips)

        if config.dry_run:
            progress.stop()
            _show_clips_table(viral_clips, dry_run=True)
            return []

        task_cut = progress.add_task("Cortando clipes...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            try:
                cut_result = cut_clip(video_path, clip, i, config)
                clip_path, captions_burned, _ = _extract_cut_result(cut_result)
                clip_paths.append(clip_path)
                captions_already_burned.append(
                    captions_burned and isinstance(cut_result, CaptionBurnResult)
                )
            except Exception as e:
                progress.stop()
                _err_console.print(
                    Panel(str(e), title="[red]Erro ao cortar clipe[/red]", border_style="red")
                )
                raise _PipelineError(str(e)) from e
            preview = generate_clip_preview(video_path, clip, i, config)
            preview_paths.append(preview.path if preview else None)
            progress.advance(task_cut)
        progress.update(task_cut, description="[green]Clipes cortados[/green]")

        task_cap = progress.add_task("Adicionando legendas...", total=len(clip_paths))
        for index, (clip_path, clip) in enumerate(zip(clip_paths, viral_clips)):
            try:
                if _is_editorial_social_layout(config):
                    final_path, captions_applied, caption_warning = _finalize_editorial_social_clip(
                        clip_path,
                        clip,
                        config,
                    )
                    clip_paths[index] = final_path
                    captions_applied_flags.append(captions_applied)
                    caption_warnings.append(caption_warning)
                elif index < len(captions_already_burned) and captions_already_burned[index]:
                    # CaptionBurner já queimou legendas (pós-decoupage) em cut_clip.
                    # Pular o segundo burn evita legendas duplicadas e desalinhadas.
                    captions_applied_flags.append(True)
                    caption_warnings.append(None)
                else:
                    add_captions(clip_path, transcription, clip, config)
                    captions_applied_flags.append(True)
                    caption_warnings.append(None)
            except Exception as e:
                progress.stop()
                _err_console.print(
                    Panel(str(e), title="[red]Erro ao adicionar legendas[/red]", border_style="red")
                )
                raise _PipelineError(str(e)) from e
            progress.advance(task_cap)
        progress.update(task_cap, description="[green]Legendas adicionadas[/green]")

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
                    raise _PipelineError(str(e)) from e
                progress.advance(task_ovl)
            progress.update(task_ovl, description="[green]Sobreposição de título adicionada[/green]")

        output_dir = config.output_dir / video_path.stem
        task_exp = progress.add_task("Exportando metadados...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            metadata_path = export_metadata(clip, i, output_dir)
            metadata_paths.append(metadata_path)
            progress.advance(task_exp)
        progress.update(task_exp, description="[green]Metadados exportados[/green]")

    clip_records: list[ClipRecord] = []
    for clip, clip_path, captions_applied, caption_warning in zip(
        viral_clips,
        clip_paths,
        captions_applied_flags,
        caption_warnings,
    ):
        clip_records.append(ClipRecord(
            title=clip.title,
            start_time=clip.start_time,
            end_time=clip.end_time,
            clip_path=clip_path,
            thumbnail_path=None,
            approved=True,
            description=clip.description,
            hashtags=clip.hashtags,
            captions_applied=captions_applied,
            caption_warning=caption_warning,
        ))

    if not skip_review:
        clip_records = review_clips(viral_clips, clip_records, config.cut_mode)

    if upload:
        try:
            upload_clips(
                clips=list(zip(clip_paths, metadata_paths)),
                platforms=platforms or list(_SUPPORTED_PLATFORMS),
                token_dir=_default_token_dir(),
                clips_filter=config.clips,
            )
        except Exception as e:
            _err_console.print(
                Panel(str(e), title="[red]Erro de Upload[/red]", border_style="red")
            )

    return clip_records


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
    thumbnail_text: str = typer.Option(
        "",
        "--thumbnail-text",
        help="Texto opcional para sobrescrever o texto padrão sugerido pela IA na thumbnail",
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
        help="Quantidade de clipes a gerar",
    ),
    decoupage: bool = typer.Option(
        True,
        "--decoupage/--no-decoupage",
        help="Remover silêncios/respirações dentro de cada clipe (ligado por padrão).",
    ),
    color_preset: str = typer.Option(
        "none",
        "--filter",
        help="Preset de cor: none, warm, cool, vintage, punchy.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Nível de log: DEBUG, INFO, WARNING, ERROR"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Caminho para salvar o arquivo de log"),
) -> None:
    """Processa um vídeo longo e gera clipes virais prontos para publicação."""
    _configure_logging(log_level, log_file)
    thumbnail_text_value = thumbnail_text if isinstance(thumbnail_text, str) else ""

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
        auth_config = _resolve_cli_yt_dlp_auth_config()
        config = PipelineConfig(
            clip_count=resolved_clip_count,
            subtitle_style=style,  # type: ignore[arg-type]
            dry_run=dry_run,
            title_overlay=title_overlay,
            thumbnail_text=thumbnail_text_value,
            upload=upload,
            platforms=selected_platforms,
            clips=clips_filter,
            social_layout_mode="speaker_bottom_ai_top",
            decoupage_enabled=decoupage,
            social_filter_preset=color_preset,
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

    try:
        clip_records = _run_single_source_pipeline(
            source,
            config,
            skip_review=True,
            upload=upload,
            platforms=selected_platforms,
            auth_config=auth_config,
        )
    except _PipelineError:
        raise typer.Exit(code=1)

    if clip_records:
        _show_records_table(clip_records)


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


# ============================================================
# Smart Cuts — Fluxos A, B e C
# ============================================================


def _select_cut_mode() -> CutMode:
    mode = questionary.select(
        "Selecione o modo de corte:",
        choices=[
            questionary.Choice(
                "Vídeos longos para YouTube (paisagem 16:9, 15–25 min por clipe)",
                value="youtube",
            ),
            questionary.Choice(
                "Vídeos curtos para redes sociais (vertical 9:16, até ~3 min — TikTok, Reels, Shorts)",
                value="social",
            ),
        ],
    ).ask()
    if mode is None:
        raise typer.Exit(code=0)
    return mode


def _select_execution_mode() -> Literal["manual", "auto"]:
    mode = questionary.select(
        "Como deseja executar os cortes?",
        choices=[
            questionary.Choice(
                "Manual — Revisão de clipes, oferta de Social com timer, seleção de plataformas",
                value="manual",
            ),
            questionary.Choice(
                "Automático (YouTube + Social) — Pipeline completo sem interrupções: "
                "download → cortes → thumbnails → upload YouTube → clipes sociais → upload Social",
                value="auto",
            ),
        ],
    ).ask()
    if mode is None:
        raise typer.Exit(code=0)
    return mode


def _load_transcription_from_cache(cache_path: Path) -> TranscriptionResult:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return TranscriptionResult.model_validate(data["result"])


def _show_records_table(records: list[ClipRecord]) -> None:
    table = Table(title="Clipes Gerados", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Título")
    table.add_column("Duração", width=10, justify="center")
    table.add_column("Status", width=10, justify="center")
    for i, r in enumerate(records):
        duration = f"{r.end_time - r.start_time:.0f}s"
        status = "[green]aprovado[/green]" if r.approved else "[red]rejeitado[/red]"
        table.add_row(str(i + 1), r.title, duration, status)
    _console.print(table)


def _show_consolidated_summary(
    yt_records: list[ClipRecord],
    social_records: list[ClipRecord],
) -> None:
    table = Table(
        title="Resumo Consolidado — YouTube + Social",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tipo", width=8, justify="center")
    table.add_column("Título")
    table.add_column("Duração", width=10, justify="center")
    table.add_column("Plataforma", width=12, justify="center")
    table.add_column("Status", width=10, justify="center")

    _STATUS_DISPLAY: dict[str, tuple[str, str | None]] = {
        "success": ("[green]success[/green]", "success"),
        "failed":  ("[red]failed[/red]",   "failed"),
        "pending": ("[yellow]pending[/yellow]", "failed"),
        "skipped": ("[dim]skipped[/dim]",   None),
    }

    counters: dict[str, dict[str, int]] = {}

    def _add_rows(records: list[ClipRecord], tipo: str) -> None:
        for r in records:
            duration = f"{r.end_time - r.start_time:.0f}s"
            if r.upload_status:
                for platform, status in r.upload_status.items():
                    status_fmt, counter_key = _STATUS_DISPLAY.get(
                        status, ("[red]failed[/red]", "failed")
                    )
                    if counter_key is not None:
                        if platform not in counters:
                            counters[platform] = {"success": 0, "failed": 0}
                        counters[platform][counter_key] += 1
                    table.add_row(tipo, r.title, duration, platform, status_fmt)
            else:
                table.add_row(tipo, r.title, duration, "—", "[dim]sem upload[/dim]")

    _add_rows(yt_records, "[bold blue]YouTube[/bold blue]")
    _add_rows(social_records, "[bold magenta]Social[/bold magenta]")

    _console.print(table)

    if counters:
        summary_parts = []
        for platform, counts in sorted(counters.items()):
            summary_parts.append(
                f"[bold]{platform}[/bold]: [green]{counts['success']} ok[/green] / [red]{counts['failed']} falhou[/red]"
            )
        _console.print(Panel(
            "  |  ".join(summary_parts),
            title="[bold]Uploads por Plataforma[/bold]",
            border_style="cyan",
        ))


def _show_sessions_table(sessions: list[SessionData]) -> None:
    table = Table(title="Sessões Anteriores", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("URL")
    table.add_column("Modo", width=10, justify="center")
    table.add_column("Clipes", width=7, justify="center")
    table.add_column("Data", width=20)
    for i, s in enumerate(sessions):
        table.add_row(
            str(i + 1),
            s.source_url[:60],
            s.cut_mode,
            str(len(s.clips)),
            s.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    _console.print(table)


def _publish_clips_with_status(
    records: list[ClipRecord],
    platforms: list[str],
    token_dir: Path,
    session: SessionData | None = None,
    cut_mode: str = "youtube",
) -> None:
    client_secrets_file = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE")
    uploader_map = {
        "youtube": YouTubeUploader(
            token_dir,
            client_secrets_file=Path(client_secrets_file) if client_secrets_file else None,
        ),
        "instagram": InstagramUploader(token_dir),
        "tiktok": TikTokUploader(token_dir),
    }

    # Authenticate once per platform before uploading
    auth_failures: dict[str, str] = {}
    for platform in platforms:
        if platform not in uploader_map:
            continue
        try:
            uploader_map[platform].authenticate()
        except Exception as exc:
            auth_failures[platform] = str(exc)
            logger.error("Falha na autenticação para %s: %s", platform, exc)

    for i, record in enumerate(records):
        for platform in platforms:
            if platform not in uploader_map:
                continue
            if platform in auth_failures:
                _console.print(f"[red]✗ Clipe {i + 1} — {platform}: falha de autenticação: {auth_failures[platform]}[/red]")
                record.upload_status[platform] = "failed"
                continue
            _console.print(f"[dim]Publicando clipe {i + 1}/{len(records)} em {platform}...[/dim]")
            uploader = uploader_map[platform]
            try:
                metadata = ClipMetadata(
                    title=record.title,
                    description=record.description,
                    hashtags=record.hashtags,
                    caption=record.description,
                )
                if platform == "youtube":
                    result = uploader.upload(
                        record.clip_path,
                        metadata,
                        clip_index=i + 1,
                        thumbnail_path=record.thumbnail_path,
                        cut_mode=cut_mode,
                    )
                else:
                    result = uploader.upload(record.clip_path, metadata, clip_index=i + 1)
            except Exception as exc:
                _console.print(f"[red]✗ Clipe {i + 1} — {platform}: erro inesperado: {exc}[/red]")
                logger.exception("Upload falhou para clipe %d em %s", i + 1, platform)
                record.upload_status[platform] = "failed"
                continue

            record.upload_status[platform] = result.status
            if platform == "youtube" and result.status == "success":
                record.youtube_video_id = result.video_id
                record.youtube_url = result.url

            if result.status == "success":
                if platform == "youtube" and result.warning:
                    _console.print(
                        f"[yellow]! Clipe {i + 1} publicado em {platform}, mas a thumbnail falhou: "
                        f"{result.warning} URL: {result.url}[/yellow]"
                    )
                elif platform == "youtube" and result.thumbnail_status == "uploaded":
                    _console.print(
                        f"[green]✓ Clipe {i + 1} publicado em {platform} com thumbnail aplicada: "
                        f"{result.url}[/green]"
                    )
                else:
                    _console.print(f"[green]✓ Clipe {i + 1} publicado em {platform}: {result.url}[/green]")
            else:
                _console.print(f"[red]✗ Clipe {i + 1} falhou em {platform}: {result.error}[/red]")

    if session is not None:
        save_session(session)


def _can_prompt_interactively() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _build_social_pipeline_config(
    base_config: PipelineConfig | None = None,
    *,
    output_dir: Path | None = None,
) -> PipelineConfig:
    """Normaliza o contrato de entrada do fluxo social pós-upload."""
    return PipelineConfig(
        anthropic_api_key=base_config.anthropic_api_key if base_config else None,
        openai_api_key=base_config.openai_api_key if base_config else None,
        cut_mode="social",
        subtitle_style="word",
        max_clips=base_config.max_clips if base_config else None,
        output_dir=output_dir or (base_config.output_dir if base_config else Path("output")),
        face_tracking=base_config.face_tracking if base_config else False,
        social_layout_mode=(
            base_config.social_layout_mode if base_config and base_config.social_layout_mode != "classic"
            else "speaker_bottom_ai_top"
        ),
        social_layout_title_enabled=base_config.social_layout_title_enabled if base_config else True,
        social_layout_image_provider=base_config.social_layout_image_provider if base_config else "openai",
        social_layout_apply_face_tracking=base_config.social_layout_apply_face_tracking if base_config else True,
        social_layout_top_image_height=base_config.social_layout_top_image_height if base_config else 600,
        social_layout_title_band_height=base_config.social_layout_title_band_height if base_config else 140,
        social_layout_title_color_mode=base_config.social_layout_title_color_mode if base_config else "engagement_default",
        social_layout_title_bg_color=base_config.social_layout_title_bg_color if base_config else "#F4C400",
        social_layout_title_text_color=base_config.social_layout_title_text_color if base_config else "#111111",
        huggingface_token=base_config.huggingface_token if base_config else None,
        face_detection_confidence=(
            base_config.face_detection_confidence if base_config else 0.5
        ),
    )


def _resolve_upload_platforms(
    *,
    upload: bool,
    approved_records: list[ClipRecord],
    default_platforms: list[str],
) -> list[str]:
    if not approved_records:
        return []

    if upload:
        return default_platforms

    if not _can_prompt_interactively():
        return []

    should_upload = questionary.confirm(
        "Deseja publicar agora os clipes aprovados?",
        default=False,
    ).ask()
    if should_upload is not True:
        return []

    selected_platforms = questionary.checkbox(
        "Selecione as plataformas de upload:",
        choices=[
            questionary.Choice(platform, value=platform, checked=platform in default_platforms)
            for platform in _SUPPORTED_PLATFORMS
        ],
    ).ask()

    if not selected_platforms:
        _console.print("[yellow]Upload cancelado: nenhuma plataforma selecionada.[/yellow]")
        return []

    return list(selected_platforms)


def _extract_cut_result(cut_result: Path | CaptionBurnResult) -> tuple[Path, bool, str | None]:
    if isinstance(cut_result, CaptionBurnResult):
        return cut_result.output_path, cut_result.captions_applied, cut_result.warning
    return cut_result, True, None


def _is_editorial_social_layout(config: PipelineConfig) -> bool:
    return config.cut_mode == "social" and config.social_layout_mode == "speaker_bottom_ai_top"


def _finalize_editorial_social_clip(
    clip_path: Path,
    clip: ViralClip,
    config: PipelineConfig,
) -> tuple[Path, bool, str | None]:
    composed_path = clip_path
    try:
        band_h = config.social_layout_title_band_height if config.social_layout_title_enabled else 0
        bottom_h = 1920 - config.social_layout_top_image_height - band_h
        if bottom_h <= 0:
            raise ValueError(f"bottom_h inválido: {bottom_h}")
        framed_path = frame_for_panel(
            clip_path, target_w=1080, target_h=bottom_h, config=config,
        )
        logger.info("Social composer: generating top image for clip %s", framed_path.name)
        composed_path = compose_social_clip(framed_path, clip, config)
    except Exception as exc:
        logger.warning("Social composer falhou; usando clipe base %s: %s", clip_path.name, exc)

    result = CaptionBurner().burn(composed_path, style="word", layout_mode="bottom_panel")
    return result.output_path, result.captions_applied, result.warning


def _show_youtube_duration_guidance(clips: list[ViralClip]) -> None:
    ideal_count = sum(1 for clip in clips if (clip.end_time - clip.start_time) >= YOUTUBE_MIN_DURATION)
    fallback_count = len(clips) - ideal_count

    if fallback_count <= 0:
        return

    if ideal_count == 0:
        _console.print(
            (
                "[yellow]Não encontrei clipes na faixa prioritária de 15 a 25 min. "
                f"Vou seguir com {fallback_count} sugestão(ões) menores que 15 min que também "
                "podem ser aprovadas e enviadas.[/yellow]"
            )
        )
        return

    _console.print(
        (
            "[cyan]Além dos clipes prioritários de 15 a 25 min, encontrei "
            f"{fallback_count} sugestão(ões) menores que 15 min que também podem ser aprovadas "
            "e enviadas.[/cyan]"
        )
    )


def run_flow_a(
    source: str,
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
    auth_config: YtDlpAuthConfig | None = None,
) -> SessionData | None:
    """Fluxo A: cortes longos para YouTube (paisagem 16:9)."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )

    with progress:
        task_dl = progress.add_task("Baixando vídeo...", total=None)
        try:
            if auth_config is None:
                video_path = download_video(source, config.output_dir / "downloads")
            else:
                video_path = download_video(source, config.output_dir / "downloads", auth_config=auth_config)
        except (VideoDownloadError, FileNotFoundError) as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Download[/red]", border_style="red"))
            return None
        progress.update(task_dl, description="[green]Download concluído[/green]", completed=1, total=1)

        task_tr = progress.add_task("Transcrevendo áudio...", total=None)
        try:
            transcription = transcribe(video_path, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Transcrição[/red]", border_style="red"))
            return None
        transcription_cache_path = (
            video_path.resolve().parent / f"{video_path.resolve().stem}_transcript.json"
        )
        progress.update(task_tr, description="[green]Transcrição concluída[/green]", completed=1, total=1)

        task_ai = progress.add_task("Identificando momentos relevantes (IA)...", total=None)
        try:
            viral_clips = analyze(transcription, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro na Análise IA[/red]", border_style="red"))
            return None
        progress.update(task_ai, description="[green]Análise concluída[/green]", completed=1, total=1)

        if not viral_clips:
            progress.stop()
            _console.print("[yellow]Nenhum trecho relevante identificado.[/yellow]")
            return None
        if config.cut_mode == "youtube":
            _show_youtube_duration_guidance(viral_clips)

        clip_paths: list[Path] = []
        task_cut = progress.add_task("Cortando clipes (stream copy 16:9)...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            try:
                clip_path = cut_clip(video_path, clip, i, config)
                clip_path, _, _ = _extract_cut_result(clip_path)
                clip_paths.append(clip_path)
            except Exception as e:
                progress.stop()
                _err_console.print(Panel(str(e), title="[red]Erro ao cortar clipe[/red]", border_style="red"))
                return None
            progress.advance(task_cut)
        progress.update(task_cut, description="[green]Clipes cortados[/green]")

        thumbnail_paths: list[Path | None] = []
        output_dir = config.output_dir / video_path.stem
        task_th = progress.add_task("Gerando thumbnails...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            try:
                if config.thumbnail_text.strip():
                    clip.thumbnail_text = config.thumbnail_text
                clip_path_for_thumb = clip_paths[i] if i < len(clip_paths) else None
                thumb = generate_thumbnail(clip, output_dir, i, clip_path=clip_path_for_thumb, config=config)
                thumbnail_paths.append(thumb)
            except Exception as e:
                logger.warning("Falha ao gerar thumbnail %d: %s", i + 1, e)
                thumbnail_paths.append(None)
            progress.advance(task_th)
        progress.update(task_th, description="[green]Thumbnails geradas[/green]")

    clip_records: list[ClipRecord] = []
    for i, (clip, clip_path) in enumerate(zip(viral_clips, clip_paths)):
        thumb = thumbnail_paths[i] if i < len(thumbnail_paths) else None
        clip_records.append(ClipRecord(
            title=clip.title,
            start_time=clip.start_time,
            end_time=clip.end_time,
            clip_path=clip_path,
            thumbnail_path=thumb,
            approved=True,
            description=clip.description,
            hashtags=clip.hashtags,
        ))

    if not skip_review:
        clip_records = review_clips(
            viral_clips, clip_records, config.cut_mode, api_key=config.openai_api_key
        )

    approved = [r for r in clip_records if r.approved]

    session = SessionData(
        session_id=str(uuid.uuid4()),
        source_url=source,
        cut_mode=config.cut_mode,
        transcription_cache_path=transcription_cache_path,
        clips=clip_records,
        created_at=datetime.now(),
        output_dir=config.output_dir / video_path.stem,
    )

    selected_upload_platforms = _resolve_upload_platforms(
        upload=upload,
        approved_records=approved,
        default_platforms=platforms if upload and platforms else ["youtube"],
    )
    if selected_upload_platforms:
        _publish_clips_with_status(
            approved, selected_upload_platforms, _default_token_dir(), session=session
        )

    _show_records_table(clip_records)

    session_path = save_session(session)
    logger.info("Sessão salva: %s", session_path)
    _console.print(f"[green]Sessão salva em {session_path}[/green]")

    return session


def run_flow_b(
    session: SessionData,
    selected_clips: list[ClipRecord],
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
) -> list[ClipRecord]:
    """Fluxo B: vídeos curtos a partir de cortes longos existentes (sem reprocessar áudio)."""
    _console.print(Panel(
        "[cyan]Fonte: cortes locais já gerados — o áudio não será reprocessado.[/cyan]",
        title="[bold]Fluxo B — Vídeos Curtos para Redes Sociais[/bold]",
        border_style="cyan",
    ))

    if not session.transcription_cache_path.exists():
        _err_console.print(Panel(
            f"Cache de transcrição não encontrado: {session.transcription_cache_path}",
            title="[red]Erro[/red]",
            border_style="red",
        ))
        return []

    try:
        transcription = _load_transcription_from_cache(session.transcription_cache_path)
    except Exception as e:
        _err_console.print(Panel(str(e), title="[red]Erro ao carregar transcrição[/red]", border_style="red"))
        return []

    logger.info("Fluxo B: transcrição reutilizada de %s (transcribe() não chamado)", session.transcription_cache_path)

    try:
        social_config = _build_social_pipeline_config(
            config,
            output_dir=session.output_dir,
        )
    except Exception as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Configuração[/red]", border_style="red"))
        return []

    all_short_clips: list[ViralClip] = []
    all_clip_sources: list[Path] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )

    with progress:
        for clip_record in selected_clips:
            clip_segments = [
                s for s in transcription.segments
                if s.start >= clip_record.start_time and s.end <= clip_record.end_time
            ]

            if not clip_segments:
                logger.warning("Sem segmentos de transcrição para '%s'", clip_record.title)
                continue

            sub_transcription = TranscriptionResult(
                segments=clip_segments,
                language=transcription.language,
                source_path=clip_record.clip_path,
            )

            task_ai = progress.add_task(f"Analisando '{clip_record.title}'...", total=None)
            try:
                short_clips = analyze(sub_transcription, social_config)
            except RuntimeError as e:
                progress.update(
                    task_ai,
                    description=f"[yellow]Erro ao analisar '{clip_record.title}'[/yellow]",
                    completed=1, total=1,
                )
                logger.warning("Erro na análise de '%s': %s", clip_record.title, e)
                continue

            for sc in short_clips:
                adj_start = max(0.0, sc.start_time - clip_record.start_time)
                adj_end = max(0.0, sc.end_time - clip_record.start_time)
                if adj_end <= adj_start:
                    continue
                all_short_clips.append(sc.model_copy(update={"start_time": adj_start, "end_time": adj_end}))
                all_clip_sources.append(clip_record.clip_path)

            progress.update(
                task_ai,
                description=f"[green]'{clip_record.title}' analisado[/green]",
                completed=1, total=1,
            )

        if not all_short_clips:
            _console.print("[yellow]Nenhum trecho curto identificado nos cortes selecionados.[/yellow]")
            return []

        short_clip_results: list[tuple[Path, bool, str | None] | None] = []
        task_cut = progress.add_task("Cortando vídeos curtos (9:16)...", total=len(all_short_clips))
        for i, (sc, src) in enumerate(zip(all_short_clips, all_clip_sources)):
            try:
                cp = cut_clip(src, sc, i, social_config)
                clip_path, captions_applied, caption_warning = _extract_cut_result(cp)
                if not captions_applied and caption_warning:
                    warning_message = (
                        f"Clipe '{sc.title}' gerado sem legenda: {caption_warning}"
                    )
                    logger.warning(warning_message)
                    _console.print(f"[yellow]! {warning_message}[/yellow]")
                if (
                    social_config.cut_mode == "social"
                    and social_config.face_tracking
                    and social_config.social_layout_apply_face_tracking
                ):
                    logger.info("Aplicando face tracking ao clipe %s...", clip_path.name)
                    try:
                        tracked = apply_face_tracking(clip_path, social_config)
                        if tracked == clip_path:
                            logger.warning(
                                "Face tracking não aplicado para %s — usando enquadramento original.", clip_path.name
                            )
                        clip_path = tracked
                    except Exception as ft_exc:
                        logger.warning(
                            "Face tracking falhou com erro: %s — exportando clipe original.", ft_exc
                        )
                if _is_editorial_social_layout(social_config):
                    clip_path, captions_applied, caption_warning = _finalize_editorial_social_clip(
                        clip_path, sc, social_config
                    )
                short_clip_results.append((clip_path, captions_applied, caption_warning))
            except Exception as e:
                logger.warning("Erro ao cortar clipe %d: %s", i, e)
                short_clip_results.append(None)
            progress.advance(task_cut)
        progress.update(task_cut, description="[green]Vídeos curtos cortados[/green]")

    records: list[ClipRecord] = []
    valid_clips: list[ViralClip] = []
    for sc, cut_result in zip(all_short_clips, short_clip_results):
        if cut_result is None:
            continue
        clip_path, captions_applied, caption_warning = cut_result
        records.append(ClipRecord(
            title=sc.title,
            start_time=sc.start_time,
            end_time=sc.end_time,
            clip_path=clip_path,
            thumbnail_path=None,
            approved=True,
            description=sc.description,
            hashtags=sc.hashtags,
            captions_applied=captions_applied,
            caption_warning=caption_warning,
        ))
        valid_clips.append(sc)

    if not records:
        _console.print("[yellow]Nenhum vídeo curto gerado com sucesso.[/yellow]")
        return []

    if not skip_review:
        records = review_clips(valid_clips, records, "social")

    approved = [r for r in records if r.approved]

    selected_upload_platforms = _resolve_upload_platforms(
        upload=upload,
        approved_records=approved,
        default_platforms=platforms or list(_SUPPORTED_PLATFORMS),
    )
    if selected_upload_platforms:
        _publish_clips_with_status(
            approved, selected_upload_platforms, _default_token_dir(), cut_mode="social"
        )

    _show_records_table(records)
    return records


def run_flow_c(
    source: str,
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
    auth_config: YtDlpAuthConfig | None = None,
    music: bool = False,
    music_mood: Optional[str] = None,
) -> None:
    """Fluxo C: vídeos curtos diretamente do vídeo original (9:16 para redes sociais)."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=_console,
    )

    with progress:
        task_dl = progress.add_task("Baixando vídeo...", total=None)
        try:
            if auth_config is None:
                video_path = download_video(source, config.output_dir / "downloads")
            else:
                video_path = download_video(source, config.output_dir / "downloads", auth_config=auth_config)
        except (VideoDownloadError, FileNotFoundError) as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Download[/red]", border_style="red"))
            return
        progress.update(task_dl, description="[green]Download concluído[/green]", completed=1, total=1)

        task_tr = progress.add_task("Transcrevendo áudio...", total=None)
        try:
            transcription = transcribe(video_path, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro de Transcrição[/red]", border_style="red"))
            return
        progress.update(task_tr, description="[green]Transcrição concluída[/green]", completed=1, total=1)

        task_ai = progress.add_task("Identificando momentos relevantes (IA)...", total=None)
        try:
            viral_clips = analyze(transcription, config)
        except RuntimeError as e:
            progress.stop()
            _err_console.print(Panel(str(e), title="[red]Erro na Análise IA[/red]", border_style="red"))
            return
        progress.update(task_ai, description="[green]Análise concluída[/green]", completed=1, total=1)

        if not viral_clips:
            progress.stop()
            _console.print("[yellow]Nenhum trecho relevante identificado.[/yellow]")
            return
        if config.cut_mode == "youtube":
            _show_youtube_duration_guidance(viral_clips)

        music_provider = PixabayMusicProvider(config) if (music and config.cut_mode == "social") else None
        music_mixer = MusicMixer() if (music and config.cut_mode == "social") else None
        music_tracks: list[MusicTrack | None] = []

        clip_paths: list[Path] = []
        caption_statuses: list[tuple[bool, str | None]] = []
        task_cut = progress.add_task("Cortando clipes (9:16)...", total=len(viral_clips))
        for i, clip in enumerate(viral_clips):
            try:
                clip_path = cut_clip(video_path, clip, i, config)
                clip_path, captions_applied, caption_warning = _extract_cut_result(clip_path)
                if (
                    config.cut_mode == "social"
                    and config.face_tracking
                    and config.social_layout_apply_face_tracking
                ):
                    logger.info("Aplicando face tracking ao clipe %s...", clip_path.name)
                    try:
                        tracked = apply_face_tracking(clip_path, config)
                        if tracked == clip_path:
                            logger.warning(
                                "Face tracking não aplicado para %s — usando enquadramento original.",
                                clip_path.name,
                            )
                        clip_path = tracked
                    except Exception as ft_exc:
                        logger.warning(
                            "Face tracking falhou com erro: %s — exportando clipe original.", ft_exc
                        )
                if _is_editorial_social_layout(config):
                    clip_path, captions_applied, caption_warning = _finalize_editorial_social_clip(
                        clip_path, clip, config
                    )

                # Etapa de trilha sonora (apenas modo social)
                track: MusicTrack | None = None
                if music and music_provider and music_mixer and config.cut_mode == "social":
                    progress.stop()
                    try:
                        mood = music_mood if music_mood else music_provider.classify_mood(clip)
                        _console.print(f"🎵 Detectando mood: {mood}...")
                        clip_dur = clip.end_time - clip.start_time
                        track = music_provider.fetch_track(mood, clip_dur)
                        if track:
                            _console.print(f'🎵 Trilha: "{track.name}" — Pixabay Music')
                            clip_path = music_mixer.mix(clip_path, track, config)
                        else:
                            _console.print(
                                f'[yellow]⚠️  Nenhuma música encontrada para o mood "{mood}". Clipe gerado sem trilha.[/yellow]'
                            )
                    finally:
                        progress.start()

                music_tracks.append(track)
                clip_paths.append(clip_path)
                caption_statuses.append((captions_applied, caption_warning))
            except Exception as e:
                progress.stop()
                _err_console.print(Panel(str(e), title="[red]Erro ao cortar clipe[/red]", border_style="red"))
                return
            progress.advance(task_cut)
        progress.update(task_cut, description="[green]Clipes cortados[/green]")

    output_dir = config.output_dir / video_path.stem
    clip_records: list[ClipRecord] = []
    for i, (clip, clip_path) in enumerate(zip(viral_clips, clip_paths)):
        captions_applied, caption_warning = caption_statuses[i] if i < len(caption_statuses) else (True, None)
        track = music_tracks[i] if i < len(music_tracks) else None
        export_metadata(clip, i, output_dir, music_track=track)
        clip_records.append(ClipRecord(
            title=clip.title,
            start_time=clip.start_time,
            end_time=clip.end_time,
            clip_path=clip_path,
            thumbnail_path=None,
            approved=True,
            description=clip.description,
            hashtags=clip.hashtags,
            captions_applied=captions_applied,
            caption_warning=caption_warning,
            music_track=track,
        ))

    if not skip_review:
        clip_records = review_clips(viral_clips, clip_records, config.cut_mode)

    approved = [r for r in clip_records if r.approved]

    selected_upload_platforms = _resolve_upload_platforms(
        upload=upload,
        approved_records=approved,
        default_platforms=platforms or list(_SUPPORTED_PLATFORMS),
    )
    if selected_upload_platforms:
        _publish_clips_with_status(
            approved, selected_upload_platforms, _default_token_dir(), cut_mode="social"
        )

    _show_records_table(clip_records)


def offer_flow_b(
    session: SessionData,
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
) -> None:
    """Exibe oferta do Fluxo B com timer; processa todos automaticamente após timeout."""
    approved_clips = [c for c in session.clips if c.approved]
    if not approved_clips:
        _console.print("[yellow]Nenhum clipe aprovado disponível para o Fluxo B.[/yellow]")
        return

    # Gate: Flow B only available after YouTube uploads complete (when applicable)
    if upload and platforms and "youtube" in platforms:
        not_uploaded = [c for c in approved_clips if c.upload_status.get("youtube") != "success"]
        if not_uploaded:
            _console.print(Panel(
                f"[yellow]{len(not_uploaded)} clipe(s) ainda não foram publicados no YouTube. "
                f"O Fluxo B ficará disponível após a conclusão dos uploads.[/yellow]",
                title="[yellow]Gate Flow B — Aguardando YouTube[/yellow]",
                border_style="yellow",
            ))
            return

    timeout_seconds = config.session_timeout_minutes * 60

    _console.print(Panel(
        f"[bold]Gere vídeos curtos para redes sociais a partir dos cortes recém-gerados.[/bold]\n\n"
        f"Fonte: cortes locais (sem reprocessar o vídeo original)\n\n"
        f"Você tem [bold cyan]{config.session_timeout_minutes} minutos[/bold cyan] para selecionar os cortes.\n"
        f"Após o timeout, todos os {len(approved_clips)} corte(s) serão processados automaticamente.",
        title="[cyan]Fluxo B — Oferta Automática[/cyan]",
        border_style="cyan",
    ))

    user_responded = threading.Event()
    auto_process_triggered = threading.Event()
    selected_holder: list[list[ClipRecord] | None] = [None]

    choices = [
        f"{i + 1}. {r.title} ({r.end_time - r.start_time:.0f}s)"
        for i, r in enumerate(approved_clips)
    ]

    def _run_user_selection() -> None:
        try:
            want = questionary.confirm(
                "Deseja selecionar cortes específicos para o Fluxo B? (Não = processar todos)",
                default=True,
            ).ask()

            if want is None:
                # Cancelled (e.g. Ctrl+C via questionary)
                selected_holder[0] = None
            elif not want:
                # User chose "No" → process all clips without manual selection
                selected_holder[0] = approved_clips
            else:
                selected_labels = questionary.checkbox(
                    "Selecione os cortes (todos marcados por padrão; Enter para confirmar):",
                    choices=choices,
                ).ask()

                if selected_labels:
                    indices = [int(s.split(".")[0]) - 1 for s in selected_labels]
                    selected_holder[0] = [
                        approved_clips[i] for i in indices if 0 <= i < len(approved_clips)
                    ]
                else:
                    selected_holder[0] = approved_clips
        except (KeyboardInterrupt, EOFError):
            selected_holder[0] = None
        finally:
            user_responded.set()

    def _timeout_callback() -> None:
        if not user_responded.is_set():
            auto_process_triggered.set()

    timer = threading.Timer(timeout_seconds, _timeout_callback)
    timer.daemon = True
    timer.start()

    selection_thread = threading.Thread(target=_run_user_selection, daemon=True)
    selection_thread.start()

    while not user_responded.is_set() and not auto_process_triggered.is_set():
        time.sleep(0.5)

    if auto_process_triggered.is_set() and not user_responded.is_set():
        timer.cancel()
        _console.print(
            "\n[yellow]Tempo esgotado! Iniciando Fluxo B automaticamente com todos os cortes...[/yellow]"
        )
        _console.print("[dim]Pressione Ctrl+C para cancelar antes do processamento efetivo.[/dim]")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            _console.print("[yellow]Fluxo B cancelado.[/yellow]")
            return
        _run_social_from_clips(
            approved_clips,
            config,
            skip_review=skip_review,
            upload=upload,
            platforms=platforms,
        )
        return

    timer.cancel()
    selection_thread.join(timeout=1)

    selected = selected_holder[0]
    if selected is None:
        _console.print("[dim]Fluxo B cancelado.[/dim]")
        return
    if not selected:
        _console.print("[dim]Nenhum corte selecionado para o Fluxo B.[/dim]")
        return

    _run_social_from_clips(
        selected,
        config,
        skip_review=skip_review,
        upload=upload,
        platforms=platforms,
    )


def _handle_history(*, skip_review: bool, upload: bool, platforms_raw: str) -> None:
    sessions = list_sessions()
    if not sessions:
        _console.print("[yellow]Nenhuma sessão anterior encontrada em ~/.youcut/sessions/[/yellow]")
        return

    _show_sessions_table(sessions)

    session_choices = [
        f"{i + 1}. {s.source_url[:55]} ({s.cut_mode}, {len(s.clips)} clipes) — {s.created_at.strftime('%Y-%m-%d %H:%M')}"
        for i, s in enumerate(sessions)
    ]

    selected_label = questionary.select(
        "Selecione uma sessão para iniciar o Fluxo B:",
        choices=session_choices + ["Cancelar"],
    ).ask()

    if not selected_label or selected_label == "Cancelar":
        return

    idx = int(selected_label.split(".")[0]) - 1
    session = sessions[idx]

    try:
        plats = _parse_platforms(platforms_raw) if upload else list(_SUPPORTED_PLATFORMS)
        config = _build_social_pipeline_config(output_dir=session.output_dir)
    except Exception as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Configuração[/red]", border_style="red"))
        return

    approved_clips = [c for c in session.clips if c.approved]
    if not approved_clips:
        _console.print("[yellow]Nenhum clipe aprovado nesta sessão.[/yellow]")
        return

    choices = [
        f"{i + 1}. {r.title} ({r.end_time - r.start_time:.0f}s)"
        for i, r in enumerate(approved_clips)
    ]

    selected_labels = questionary.checkbox(
        "Selecione os cortes para gerar vídeos curtos (todos por padrão):",
        choices=choices,
    ).ask()

    if not selected_labels:
        selected = approved_clips
    else:
        indices = [int(s.split(".")[0]) - 1 for s in selected_labels]
        selected = [approved_clips[i] for i in indices if 0 <= i < len(approved_clips)]

    run_flow_b(
        session=session,
        selected_clips=selected,
        config=config,
        skip_review=skip_review,
        upload=upload,
        platforms=plats,
    )


def _run_social_from_clips(
    clip_records: list[ClipRecord],
    config: PipelineConfig,
    *,
    skip_review: bool = False,
    upload: bool = False,
    platforms: list[str] | None = None,
) -> list[ClipRecord]:
    """Processa cada clipe aprovado do Flow A como fonte local no pipeline social."""
    social_config = _build_social_pipeline_config(config)
    all_records: list[ClipRecord] = []

    for clip_record in clip_records:
        logger.info("Pipeline social iniciado para clipe local: %s", clip_record.clip_path.name)
        try:
            records = _run_single_source_pipeline(
                str(clip_record.clip_path),
                social_config,
                skip_review=skip_review,
                upload=upload,
                platforms=platforms,
            )
            all_records.extend(records)
        except _PipelineError as e:
            logger.warning("Falha no pipeline social para '%s': %s", clip_record.clip_path.name, e)

    return all_records


def _run_auto_pipeline(
    source: str,
    config: PipelineConfig,
    auth_config: YtDlpAuthConfig | None = None,
) -> None:
    """Orquestrador do modo automático: Fluxo A → Fluxo B → resumo consolidado."""
    logger.info("Pipeline automático iniciado: %s", source)
    _console.print(Panel(
        "[bold]O modo Automático sempre faz upload para as plataformas configuradas, "
        "independente do flag --upload.[/bold]\n"
        "Acompanhe o progresso abaixo. Nenhuma interação será solicitada.",
        title="[cyan]Modo Automático — YouTube + Social[/cyan]",
        border_style="cyan",
    ))

    session = run_flow_a(
        source,
        config,
        skip_review=True,
        upload=True,
        platforms=["youtube"],
        auth_config=auth_config,
    )
    if session is None:
        logger.warning("Pipeline automático encerrado: run_flow_a retornou None")
        return

    yt_records = list(session.clips)
    approved = [c for c in session.clips if c.approved]

    social_records = _run_social_from_clips(
        approved,
        config,
        skip_review=True,
        upload=True,
        platforms=["tiktok", "instagram"],
    )

    _show_consolidated_summary(yt_records, social_records)
    logger.info("Pipeline automático concluído")


@app.command()
def cuts(
    source: Optional[str] = typer.Argument(None, help="URL do YouTube (deixe em branco para digitar)"),
    history: bool = typer.Option(False, "--history", "-H", help="Listar sessões anteriores e iniciar Fluxo B"),
    max_clips: Optional[int] = typer.Option(None, "--max-clips", "-n", help="Número máximo de clipes"),
    skip_review: bool = typer.Option(False, "--skip-review", help="Publicar diretamente sem revisão"),
    upload: bool = typer.Option(False, "--upload", help="Fazer upload dos clipes ao final"),
    thumbnail_text: str = typer.Option(
        "",
        "--thumbnail-text",
        help="Texto opcional para sobrescrever o texto padrão sugerido pela IA na thumbnail do Flow A",
    ),
    platforms_raw: str = typer.Option(
        "all", "--platforms", help="Plataformas de upload: youtube, instagram, tiktok ou all"
    ),
    decoupage: bool = typer.Option(
        True,
        "--decoupage/--no-decoupage",
        help="Remover silêncios/respirações dentro dos clipes (modo social, ligado por padrão).",
    ),
    color_preset: str = typer.Option(
        "none",
        "--filter",
        help="Preset de cor para clipes social: none, warm, cool, vintage, punchy.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Nível de log"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Arquivo de log"),
    music: bool = typer.Option(False, "--music", help="Adicionar trilha sonora automática (apenas modo social)"),
    music_mood: Optional[str] = typer.Option(None, "--music-mood", help="Override manual do mood musical (ex: reflexivo, energico)"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Modo de corte: social ou youtube (pula seleção interativa)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirmar automaticamente todas as perguntas interativas"),
) -> None:
    """Gera cortes inteligentes (YouTube 16:9 ou redes sociais 9:16) com revisão e publicação."""
    _configure_logging(log_level, log_file)
    thumbnail_text_value = thumbnail_text if isinstance(thumbnail_text, str) else ""
    _check_ffmpeg()

    if history:
        _handle_history(skip_review=skip_review, upload=upload, platforms_raw=platforms_raw)
        return

    auth_config = _resolve_cli_yt_dlp_auth_config()

    # RF-01: seleção do modo como primeira etapa
    if mode in ("social", "youtube"):
        cut_mode: CutMode = mode  # type: ignore[assignment]
    else:
        cut_mode = _select_cut_mode()
    mode = cut_mode

    # RF-05/06: validar URL e exibir metadados
    url = source
    if not url:
        url = questionary.text(
            "Informe a URL do vídeo no YouTube:",
            validate=lambda x: len(x.strip()) > 0 or "URL não pode ser vazia",
        ).ask()
        if not url:
            raise typer.Exit(code=0)

    local_path = Path(url) if Path(url).exists() else None
    if local_path is None:
        url = normalize_video_url(url)

    _console.print(f"\nBuscando metadados: [bold]{url}[/bold]")
    if local_path is not None:
        import subprocess as _sp
        dur_probe = _sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(local_path)],
            capture_output=True, text=True,
        )
        try:
            dur_s = float(dur_probe.stdout.strip())
        except ValueError:
            dur_s = 0.0
        video_meta = VideoMetadata(title=local_path.stem, duration_seconds=dur_s, url=str(local_path))
    else:
        try:
            if auth_config is None:
                video_meta = fetch_metadata(url)
            else:
                video_meta = fetch_metadata(url, auth_config=auth_config)
        except VideoMetadataError as e:
            _err_console.print(Panel(str(e), title="[red]Erro ao acessar vídeo[/red]", border_style="red"))
            raise typer.Exit(code=1)

    dur_min = int(video_meta.duration_seconds // 60)
    dur_sec = int(video_meta.duration_seconds % 60)
    _console.print(Panel(
        f"[bold]{video_meta.title}[/bold]\nDuração: {dur_min}min {dur_sec}s",
        title="[green]Vídeo encontrado[/green]",
        border_style="green",
    ))

    if not yes and not questionary.confirm("Prosseguir com este vídeo?", default=True).ask():
        _console.print("Operação cancelada.")
        raise typer.Exit(code=0)

    # RF-10/11: campo opcional de max_clips
    resolved_max_clips = max_clips
    if resolved_max_clips is None and not yes:
        mc_input = questionary.text(
            "Número máximo de clipes (Enter para deixar a IA decidir):",
            default="",
        ).ask()
        if mc_input and mc_input.strip().isdigit():
            resolved_max_clips = int(mc_input.strip())

    try:
        selected_platforms = _parse_platforms(platforms_raw) if upload else list(_SUPPORTED_PLATFORMS)
        # O preset de cor só faz sentido no layout clássico (o editorial é montado
        # depois pelo social_composer). Se o usuário pediu --filter no modo social,
        # desce para o classic e avisa.
        layout_mode: Literal["classic", "speaker_bottom_ai_top"]
        if mode == "social":
            if color_preset != "none":
                layout_mode = "classic"
                _console.print(
                    f"[yellow]--filter={color_preset} aplicado no layout 'classic' "
                    f"(o layout editorial não suporta filtro de cor nesta versão).[/yellow]"
                )
            else:
                layout_mode = "speaker_bottom_ai_top"
        else:
            layout_mode = "classic"

        config = PipelineConfig(
            cut_mode=mode,
            max_clips=resolved_max_clips,
            thumbnail_text=thumbnail_text_value,
            social_layout_mode=layout_mode,
            decoupage_enabled=decoupage,
            social_filter_preset=color_preset,
        )
    except Exception as e:
        _err_console.print(Panel(str(e), title="[red]Erro de Configuração[/red]", border_style="red"))
        raise typer.Exit(code=1)

    if mode == "youtube":
        execution_mode = _select_execution_mode() if _can_prompt_interactively() else "manual"

        if execution_mode == "auto":
            _run_auto_pipeline(url, config, auth_config=auth_config)
        else:
            # Fluxo A + oferta de Fluxo B (path manual existente)
            yt_only = [p for p in selected_platforms if p == "youtube"]
            session = run_flow_a(
                url,
                config,
                skip_review=skip_review,
                upload=upload,
                platforms=yt_only,
                auth_config=auth_config,
            )
            if session:
                offer_flow_b(
                    session=session,
                    config=config,
                    skip_review=skip_review,
                    upload=upload,
                    platforms=selected_platforms,
                )
    else:
        # Fluxo C
        run_flow_c(
            url,
            config,
            skip_review=skip_review,
            upload=upload,
            platforms=selected_platforms,
            auth_config=auth_config,
            music=music,
            music_mood=music_mood,
        )
