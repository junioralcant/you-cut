"""Testes do helper `compose_from_single_clip` (Task 8.0)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from youcut.comic.composer import (
    DEFAULT_NO_SUBS_NAME,
    DEFAULT_OUTPUT_NAME,
    _build_watermark_filter,
    compose_from_single_clip,
)
from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def transcription() -> TranscriptionResult:
    return TranscriptionResult(
        language="pt",
        source_path=Path("/tmp/x.mp4"),
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=2.0,
                text="olá mundo",
                words=[
                    WordTimestamp(word="olá", start=0.0, end=0.4),
                    WordTimestamp(word="mundo", start=0.5, end=1.0),
                ],
            )
        ],
    )


@pytest.fixture
def remotion_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    return PipelineConfig(
        comic_animation_engine="remotion",
        comic_scenes_watermark_text="@youcut",
        comic_scenes_watermark_opacity=0.4,
        comic_scenes_watermark_y_from_bottom=280,
    )


@pytest.fixture
def remotion_config_no_watermark(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    return PipelineConfig(
        comic_animation_engine="remotion",
        comic_scenes_watermark_text=None,
    )


# ── _build_watermark_filter ─────────────────────────────────────────────────


def test_build_watermark_filter_format():
    f = _build_watermark_filter("@hello", opacity=0.5, y_from_bottom=100)
    assert "drawtext=" in f
    assert "text='@hello'" in f
    assert "fontcolor=white@0.50" in f
    assert "y=h-100" in f


# ── compose_from_single_clip — mocked ───────────────────────────────────────


def _make_fake_video(path: Path) -> None:
    """Cria placeholder file representando um MP4."""
    path.write_bytes(b"\x00" * 256)


def test_invokes_ffmpeg_with_ass_filter(tmp_path, transcription, remotion_config):
    input_video = tmp_path / "input.mp4"
    _make_fake_video(input_video)

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        # cria os outputs esperados
        if "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy":
            out = Path(cmd[-1])
            _make_fake_video(out)
        else:
            out = Path(cmd[-1])
            _make_fake_video(out)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with patch.object(subprocess, "run", side_effect=fake_run):
        with patch(
            "youcut.comic.composer._ffprobe_duration", return_value=2.0
        ):
            no_subs, with_subs = compose_from_single_clip(
                input_video, transcription, tmp_path, remotion_config
            )

    # 1ª chamada: re-mux do input para no_subs (stream copy)
    assert any(
        "-c" in c and "copy" in c and DEFAULT_NO_SUBS_NAME in c[-1]
        for c in captured_cmds
    )
    # 2ª chamada: burn com `-vf` contendo `ass=...`
    burn_cmd = next(c for c in captured_cmds if "-vf" in c)
    vf_idx = burn_cmd.index("-vf")
    vf = burn_cmd[vf_idx + 1]
    assert "ass=" in vf
    assert "drawtext=" in vf
    assert "@youcut" in vf

    assert no_subs.name == DEFAULT_NO_SUBS_NAME
    assert with_subs.name == DEFAULT_OUTPUT_NAME
    assert no_subs.parent.name == "comic"


def test_no_watermark_when_text_is_none(tmp_path, transcription, remotion_config_no_watermark):
    input_video = tmp_path / "input.mp4"
    _make_fake_video(input_video)
    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x00" * 256)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with patch.object(subprocess, "run", side_effect=fake_run):
        with patch("youcut.comic.composer._ffprobe_duration", return_value=2.0):
            compose_from_single_clip(
                input_video, transcription, tmp_path, remotion_config_no_watermark
            )

    burn_cmd = next(c for c in captured_cmds if "-vf" in c)
    vf = burn_cmd[burn_cmd.index("-vf") + 1]
    assert "ass=" in vf
    assert "drawtext" not in vf


def test_output_paths_under_comic_dir(tmp_path, transcription, remotion_config):
    input_video = tmp_path / "src.mp4"
    _make_fake_video(input_video)

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"\x00" * 256)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with patch.object(subprocess, "run", side_effect=fake_run):
        with patch("youcut.comic.composer._ffprobe_duration", return_value=2.0):
            no_subs, with_subs = compose_from_single_clip(
                input_video, transcription, tmp_path, remotion_config
            )

    assert no_subs == tmp_path / "comic" / DEFAULT_NO_SUBS_NAME
    assert with_subs == tmp_path / "comic" / DEFAULT_OUTPUT_NAME
    # Workspace temporário criado
    assert (tmp_path / "comic" / "_compose" / "captions.ass").exists()


def test_duration_warning_when_drift_exceeds_tolerance(
    tmp_path, transcription, remotion_config, caplog
):
    input_video = tmp_path / "input.mp4"
    _make_fake_video(input_video)

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"\x00" * 256)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    # input=2.0s, no_subs=2.0s OK, with_subs=2.5s → drift 0.5s > 0.2s tolerância
    durations = iter([2.0, 2.0, 2.5])

    with patch.object(subprocess, "run", side_effect=fake_run):
        with patch(
            "youcut.comic.composer._ffprobe_duration", side_effect=lambda p: next(durations)
        ):
            with caplog.at_level("WARNING", logger="youcut.comic.composer"):
                compose_from_single_clip(
                    input_video, transcription, tmp_path, remotion_config
                )
    assert any("difere do input" in rec.message for rec in caplog.records)


# ── Regressão: compose() existente ──────────────────────────────────────────


def test_existing_compose_still_imports():
    """`compose()` original do composer não deve ter sido removida/quebrada."""
    from youcut.comic.composer import compose, compose_single_video

    assert callable(compose)
    assert callable(compose_single_video)


# ── Integração ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_compose_from_single_clip_end_to_end(tmp_path, transcription, remotion_config):
    """Integration: vídeo sintético via FFmpeg → ambas as versões válidas."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe ausentes")

    # Gera vídeo sintético 2s 1080×1920 com áudio
    input_video = tmp_path / "input.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1080x1920:d=2:r=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(input_video),
        ],
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")

    no_subs, with_subs = compose_from_single_clip(
        input_video, transcription, tmp_path, remotion_config
    )

    assert no_subs.exists() and no_subs.stat().st_size > 0
    assert with_subs.exists() and with_subs.stat().st_size > 0

    # Validar duração via ffprobe (drift < 200ms)
    def _duration(p: Path) -> float:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(out.stdout.strip())

    src = _duration(input_video)
    assert abs(_duration(no_subs) - src) < 0.2
    assert abs(_duration(with_subs) - src) < 0.2
