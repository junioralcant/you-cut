import io
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from youcut.models import ViralClip
from youcut.selector import SELECTION_TIMEOUT_SECONDS, _format_countdown, prompt_clip_selection


def make_clips() -> tuple[list[ViralClip], list[Path]]:
    clips = [
        ViralClip(
            title="Clip A",
            reason="funny",
            viral_score=8.0,
            start_time=0.0,
            end_time=30.0,
            description="desc",
            hashtags=["#a"],
            thumbnail_idea="idea",
            thumbnail_text="MOMENTO IMPACTANTE",
        ),
        ViralClip(
            title="Clip B",
            reason="engaging",
            viral_score=7.5,
            start_time=60.0,
            end_time=90.0,
            description="desc",
            hashtags=["#b"],
            thumbnail_idea="idea",
            thumbnail_text="MOMENTO IMPACTANTE",
        ),
        ViralClip(
            title="Clip C",
            reason="viral",
            viral_score=9.0,
            start_time=120.0,
            end_time=165.0,
            description="desc",
            hashtags=["#c"],
            thumbnail_idea="idea",
            thumbnail_text="MOMENTO IMPACTANTE",
        ),
    ]
    paths = [Path(f"clip_{i}.mp4") for i in range(len(clips))]
    return clips, paths


def buf_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, highlight=False), buf


def test_no_tty_returns_none_and_prints_auto_mode_message():
    clips, paths = make_clips()
    console, buf = buf_console()
    with patch("youcut.selector.sys.stdin") as mock_stdin, \
         patch("youcut.selector.os.environ.get", return_value="xterm"):
        mock_stdin.isatty.return_value = False
        result = prompt_clip_selection(clips, paths, console=console)

    assert result is None
    assert "Modo automático" in buf.getvalue()


def test_empty_selection_returns_none():
    clips, paths = make_clips()
    console, _ = buf_console()

    mock_checkbox = MagicMock()
    mock_checkbox.return_value.ask.return_value = []

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", mock_checkbox), \
         patch("youcut.selector.Live"):
        result = prompt_clip_selection(clips, paths, console=console)

    assert result is None


def test_subset_selection_returns_1based_indices():
    clips, paths = make_clips()
    console, _ = buf_console()

    # User selects clips A and C (indices 1 and 3)
    selected_labels = [
        "Clip A — 30s",
        "Clip C — 45s",
    ]
    mock_checkbox = MagicMock()
    mock_checkbox.return_value.ask.return_value = selected_labels

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", mock_checkbox), \
         patch("youcut.selector.Live"):
        result = prompt_clip_selection(clips, paths, console=console)

    assert result == [1, 3]


def test_ctrl_c_returns_none():
    clips, paths = make_clips()
    console, _ = buf_console()

    mock_checkbox = MagicMock()
    mock_checkbox.return_value.ask.return_value = None

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", mock_checkbox), \
         patch("youcut.selector.Live"):
        result = prompt_clip_selection(clips, paths, console=console)

    assert result is None


def test_label_format():
    clips, paths = make_clips()
    console, _ = buf_console()

    captured_choices: list[str] = []

    def fake_checkbox(question, choices):
        captured_choices.extend(choices)
        mock = MagicMock()
        mock.ask.return_value = []
        return mock

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", fake_checkbox), \
         patch("youcut.selector.Live"):
        prompt_clip_selection(clips, paths, console=console)

    assert captured_choices[0] == "Clip A — 30s"
    assert captured_choices[1] == "Clip B — 30s"
    assert captured_choices[2] == "Clip C — 45s"


def test_timeout_expired_returns_none_and_prints_message():
    clips, paths = make_clips()
    console, buf = buf_console()

    slow_event = threading.Event()

    def slow_checkbox(question, choices):
        mock = MagicMock()
        def blocking_ask():
            slow_event.wait(timeout=60)
            return []
        mock.ask = blocking_ask
        return mock

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", slow_checkbox), \
         patch("youcut.selector.Live"):
        result = prompt_clip_selection(clips, paths, timeout=1, console=console)

    slow_event.set()
    assert result is None
    assert "expirado" in buf.getvalue()


def test_short_timeout_behaves_as_expired():
    clips, paths = make_clips()
    console, _ = buf_console()

    slow_event = threading.Event()

    def slow_checkbox(question, choices):
        mock = MagicMock()
        def blocking_ask():
            slow_event.wait(timeout=60)
            return []
        mock.ask = blocking_ask
        return mock

    with patch("youcut.selector._is_interactive", return_value=True), \
         patch("youcut.selector.questionary.checkbox", slow_checkbox), \
         patch("youcut.selector.Live"):
        result = prompt_clip_selection(clips, paths, timeout=0, console=console)

    slow_event.set()
    assert result is None


def test_format_countdown_boundaries():
    assert _format_countdown(0) == "Tempo restante para seleção: 00:00"
    assert _format_countdown(59) == "Tempo restante para seleção: 00:59"
    assert _format_countdown(60) == "Tempo restante para seleção: 01:00"
    assert _format_countdown(600) == "Tempo restante para seleção: 10:00"
    assert _format_countdown(599) == "Tempo restante para seleção: 09:59"
