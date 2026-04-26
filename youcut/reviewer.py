import logging

import questionary
from rich.console import Console

from youcut import thumbnail_generator
from youcut.models import ClipRecord, CutMode, ViralClip

logger = logging.getLogger(__name__)

_console = Console()


def review_clips(
    clips: list[ViralClip],
    records: list[ClipRecord],
    cut_mode: CutMode,
    *,
    api_key: str | None = None,
) -> list[ClipRecord]:
    updated = [r.model_copy() for r in records]

    for i, (clip, record) in enumerate(zip(clips, updated)):
        _display_clip_info(i, len(clips), record, cut_mode)

        while True:
            choices = ["Aprovar", "Rejeitar", "Editar título"]
            if cut_mode == "youtube":
                choices.append("Regenerar thumbnail")

            action = questionary.select(
                f"Clipe {i + 1}/{len(clips)} — O que deseja fazer?",
                choices=choices,
            ).ask()

            if action is None or action == "Rejeitar":
                record.approved = False
                logger.info("Clipe %d/%d rejeitado: %s", i + 1, len(clips), record.title)
                break
            elif action == "Aprovar":
                record.approved = True
                logger.info("Clipe %d/%d aprovado: %s", i + 1, len(clips), record.title)
                break
            elif action == "Editar título":
                old_title = record.title
                new_title = questionary.text("Novo título:", default=record.title).ask()
                if new_title:
                    record.title = new_title
                    logger.info("Título editado: '%s' → '%s'", old_title, record.title)
            elif action == "Regenerar thumbnail":
                if api_key:
                    try:
                        new_path = thumbnail_generator.regenerate_thumbnail(clip, record, api_key)
                        record.thumbnail_path = new_path
                        _console.print(f"[green]Thumbnail regenerada: {new_path}[/green]")
                        logger.info("Thumbnail regenerada para clipe %d: %s", i + 1, new_path)
                    except Exception as e:
                        _console.print(f"[red]Erro ao regenerar thumbnail: {e}[/red]")
                        logger.warning("Falha ao regenerar thumbnail: %s", e, exc_info=True)
                else:
                    _console.print("[yellow]API key OpenAI não disponível.[/yellow]")
                    logger.warning("Regeneração de thumbnail solicitada sem api_key configurada.")

    return updated


def _display_clip_info(
    index: int,
    total: int,
    record: ClipRecord,
    cut_mode: CutMode,
) -> None:
    duration = record.end_time - record.start_time
    _console.print(f"\n[bold]Clipe {index + 1}/{total}[/bold]: {record.title}")
    _console.print(f"  Duração: {duration:.0f}s")
    if cut_mode == "youtube" and record.thumbnail_path:
        _console.print(f"  Thumbnail: {record.thumbnail_path}")
