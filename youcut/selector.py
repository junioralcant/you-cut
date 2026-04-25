import logging
import os
import sys
import threading
import time
from pathlib import Path

import questionary
from rich.console import Console
from rich.live import Live
from rich.text import Text

from youcut.models import ViralClip

logger = logging.getLogger(__name__)

SELECTION_TIMEOUT_SECONDS = 600  # 10 minutes


def _is_interactive() -> bool:
    return sys.stdin.isatty() and os.environ.get("TERM", "") != "dumb"


def prompt_clip_selection(
    clips: list[ViralClip],
    clip_paths: list[Path],
    *,
    timeout: int = SELECTION_TIMEOUT_SECONDS,
    console: Console | None = None,
) -> list[int] | None:
    """
    Exibe checkbox interativo com countdown de 'timeout' segundos.
    Retorna None se: sem TTY, Enter sem seleção, timeout expirado ou Ctrl+C.
    None sinaliza 'enviar tudo' para upload_clips().
    """
    if console is None:
        console = Console()

    if not _is_interactive():
        console.print(
            "[yellow]Modo automático: TTY não detectado, todos os cortes serão enviados.[/yellow]"
        )
        logger.info("Modo automático: TTY não detectado, todos os cortes serão enviados.")
        return None

    choices = [
        f"{clip.title} — {clip.end_time - clip.start_time:.0f}s"
        for clip in clips
    ]

    result_holder: list = [None]
    result_lock = threading.Lock()

    def run_checkbox() -> None:
        try:
            answer = questionary.checkbox(
                "Selecione os cortes para upload (Enter sem seleção = enviar tudo):",
                choices=choices,
            ).ask()
        except KeyboardInterrupt:
            answer = None
        with result_lock:
            result_holder[0] = answer

    checkbox_thread = threading.Thread(target=run_checkbox, daemon=True)
    checkbox_thread.start()

    deadline = time.monotonic() + timeout

    # Show countdown ticking once per second while user interacts.
    # questionary and Live conflict for cursor control, so Live is stopped
    # before questionary fully renders — leaving the last countdown line static
    # above the checkbox UI (acceptable for MVP per tech spec).
    with Live(console=console, refresh_per_second=1, transient=False) as live:
        while checkbox_thread.is_alive():
            remaining = max(0, int(deadline - time.monotonic()))
            live.update(Text(_format_countdown(remaining)))
            if remaining <= 0:
                break
            checkbox_thread.join(timeout=1)

    if checkbox_thread.is_alive():
        console.print(
            "[yellow]Tempo de seleção expirado. Todos os cortes serão enviados.[/yellow]"
        )
        logger.info("Timeout de seleção expirado. Todos os cortes serão enviados.")
        return None

    with result_lock:
        answer = result_holder[0]

    if not answer:
        return None

    selected_indices = []
    for label in answer:
        for idx, choice in enumerate(choices):
            if choice == label:
                selected_indices.append(idx + 1)
                break

    logger.debug("Seleção do usuário: clips %s", selected_indices)
    return selected_indices if selected_indices else None


def _format_countdown(seconds: int) -> str:
    mm, ss = divmod(seconds, 60)
    return f"Tempo restante para seleção: {mm:02d}:{ss:02d}"
