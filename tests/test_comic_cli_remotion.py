"""Testes CLI da integração `--engine remotion` (Task 10.0)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from youcut.comic.cli import comic_app
from youcut.models import MotionComicSession


runner = CliRunner()


@pytest.fixture
def synthetic_video(tmp_path):
    """MP4 mínimo (apenas placeholder; o run é mockado)."""
    p = tmp_path / "video.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))


def _fake_session(tmp_path: Path) -> MotionComicSession:
    from datetime import datetime, timezone

    return MotionComicSession(
        session_id="abc123",
        video_path=tmp_path / "video.mp4",
        created_at=datetime.now(timezone.utc),
        cast=[],
        panels=[],
        panel_results=[],
        total_cost_usd=0.5,
        output_path=tmp_path / "motion_comic.mp4",
    )


def test_engine_remotion_invokes_remotion_pipeline(env, synthetic_video, tmp_path):
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch(
        "youcut.comic.remotion_pipeline.run_remotion_pipeline", fake
    ):
        with patch("youcut.comic.cli._confirm_cast_interactive", return_value=True):
            result = runner.invoke(
                comic_app,
                [str(synthetic_video), "--engine", "remotion", "--no-preview"],
            )
    assert result.exit_code == 0, result.output
    assert fake.called
    kwargs = fake.call_args.kwargs
    assert kwargs["preview"] is False


def test_engine_remotion_with_yes_implies_no_preview(env, synthetic_video, tmp_path):
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch("youcut.comic.remotion_pipeline.run_remotion_pipeline", fake):
        result = runner.invoke(
            comic_app,
            [str(synthetic_video), "--engine", "remotion", "-y"],
        )
    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["preview"] is False


def test_engine_remotion_default_keeps_preview_on(env, synthetic_video, tmp_path):
    """Sem --no-preview e sem --yes, preview ON (mas testes rodam sem TTY → fallback no orquestrador)."""
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch("youcut.comic.remotion_pipeline.run_remotion_pipeline", fake):
        with patch("youcut.comic.cli._confirm_cast_interactive", return_value=True):
            with patch("youcut.comic.cli._confirm_cost_interactive", return_value=True):
                result = runner.invoke(
                    comic_app,
                    [str(synthetic_video), "--engine", "remotion"],
                )
    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["preview"] is True


def test_engine_invalid_value_rejected(env, synthetic_video):
    result = runner.invoke(
        comic_app,
        [str(synthetic_video), "--engine", "xyz"],
    )
    assert result.exit_code != 0
    assert "engine inválido" in result.output.lower() or "engine" in result.output.lower()


def test_engine_scenes_still_works(env, synthetic_video, tmp_path):
    """Regressão: --engine scenes continua dispatcheando para run_comic_pipeline."""
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch("youcut.comic.cli.run_comic_pipeline", fake):
        result = runner.invoke(
            comic_app,
            [str(synthetic_video), "--engine", "scenes", "-y"],
        )
    assert result.exit_code == 0, result.output
    assert fake.called
    # `preview` não deve estar no kwargs do path antigo
    assert "preview" not in fake.call_args.kwargs


def test_engine_panels_still_works(env, synthetic_video, tmp_path):
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch("youcut.comic.cli.run_comic_pipeline", fake):
        result = runner.invoke(
            comic_app,
            [str(synthetic_video), "--engine", "panels", "-y"],
        )
    assert result.exit_code == 0, result.output
    assert fake.called


def test_help_lists_remotion_engine(env):
    result = runner.invoke(comic_app, ["--help"])
    assert result.exit_code == 0
    assert "remotion" in result.output
    assert "--no-preview" in result.output


def test_engine_remotion_dry_run(env, synthetic_video, tmp_path):
    fake = MagicMock(return_value=_fake_session(tmp_path))
    with patch("youcut.comic.remotion_pipeline.run_remotion_pipeline", fake):
        result = runner.invoke(
            comic_app,
            [
                str(synthetic_video),
                "--engine",
                "remotion",
                "--dry-run",
                "--no-preview",
            ],
        )
    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs["dry_run"] is True
