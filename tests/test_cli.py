from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from youcut.cli import app
from youcut.models import (
    TranscriptionResult,
    TranscriptionSegment,
    ViralClip,
    WordTimestamp,
)

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_ENV = {"ANTHROPIC_API_KEY": "test-key"}


def _make_clip(title: str = "Clipe Teste", score: float = 8.0) -> ViralClip:
    return ViralClip(
        title=title,
        reason="Gancho forte",
        viral_score=score,
        start_time=10.0,
        end_time=40.0,
        description="Descrição",
        hashtags=["#teste"],
        thumbnail_idea="Frame impactante",
    )


def _make_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=50.0,
                text="Conteúdo de teste",
                words=[WordTimestamp(word="Conteúdo", start=0.0, end=0.5)],
            )
        ],
        language="pt",
        source_path=Path("video.mp4"),
    )


# ---------------------------------------------------------------------------
# Helper: build a fully-mocked pipeline context
# ---------------------------------------------------------------------------

def _mock_pipeline(
    video_path: Path = Path("video.mp4"),
    clips: list[ViralClip] | None = None,
    transcription: TranscriptionResult | None = None,
    cut_clip_side_effect=None,
    add_captions_side_effect=None,
):
    """Return a dict of patches that simulate a successful pipeline run."""
    if clips is None:
        clips = [_make_clip()]
    if transcription is None:
        transcription = _make_transcription()

    mock_download = MagicMock(return_value=video_path)
    mock_transcribe = MagicMock(return_value=transcription)
    mock_analyze = MagicMock(return_value=clips)
    mock_cut = MagicMock(return_value=video_path)
    if cut_clip_side_effect is not None:
        mock_cut.side_effect = cut_clip_side_effect
    mock_caption = MagicMock(return_value=video_path)
    if add_captions_side_effect is not None:
        mock_caption.side_effect = add_captions_side_effect
    mock_export = MagicMock(return_value=Path("clip_01.txt"))

    return {
        "download": mock_download,
        "transcribe": mock_transcribe,
        "analyze": mock_analyze,
        "cut_clip": mock_cut,
        "add_captions": mock_caption,
        "export_metadata": mock_export,
    }


def _run_with_mocks(mocks: dict, extra_args: list[str] | None = None) -> "Result":
    args = ["run", "video.mp4"] + (extra_args or [])
    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("youcut.cli.download_video", mocks["download"]),
        patch("youcut.cli.transcribe", mocks["transcribe"]),
        patch("youcut.cli.analyze", mocks["analyze"]),
        patch("youcut.cli.cut_clip", mocks["cut_clip"]),
        patch("youcut.cli.add_captions", mocks["add_captions"]),
        patch("youcut.cli.export_metadata", mocks["export_metadata"]),
    ):
        return runner.invoke(app, args, env=API_ENV)


# ---------------------------------------------------------------------------
# 8.1 — CLI structure and --help
# ---------------------------------------------------------------------------

class TestCliHelp:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

    def test_help_shows_source_param(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "source" in result.output.lower() or "SOURCE" in result.output

    def test_help_shows_clips_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--clips" in result.output

    def test_help_shows_style_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--style" in result.output

    def test_help_shows_dry_run_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--dry-run" in result.output

    def test_help_shows_log_level_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--log-level" in result.output

    def test_help_shows_log_file_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--log-file" in result.output


# ---------------------------------------------------------------------------
# 8.2 — FFmpeg check
# ---------------------------------------------------------------------------

class TestFFmpegCheck:
    def test_missing_ffmpeg_exits_nonzero(self):
        with patch("shutil.which", return_value=None):
            result = runner.invoke(app, ["run", "video.mp4"], env=API_ENV)
        assert result.exit_code != 0

    def test_missing_ffmpeg_friendly_message(self):
        with patch("shutil.which", return_value=None):
            result = runner.invoke(app, ["run", "video.mp4"], env=API_ENV)
        assert "FFmpeg" in result.output or "ffmpeg" in result.output.lower()

    def test_ffmpeg_present_does_not_abort_early(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)
        # pipeline started: download was called
        mocks["download"].assert_called_once()


# ---------------------------------------------------------------------------
# 8.3 — API key validation
# ---------------------------------------------------------------------------

class TestAPIKeyValidation:
    def test_missing_api_key_exits_nonzero(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = runner.invoke(app, ["run", "video.mp4"])
        assert result.exit_code != 0

    def test_missing_api_key_friendly_message(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = runner.invoke(app, ["run", "video.mp4"])
        output = result.output
        assert "ANTHROPIC_API_KEY" in output or "configuração" in output.lower() or "api" in output.lower()

    def test_missing_api_key_no_pipeline_started(self):
        mock_download = MagicMock()
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("youcut.cli.download_video", mock_download),
        ):
            runner.invoke(app, ["run", "video.mp4"])
        mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# 8.4 & 8.3 — Pipeline orchestration order
# ---------------------------------------------------------------------------

class TestPipelineOrder:
    def test_pipeline_runs_in_correct_order(self):
        call_order: list[str] = []

        mocks = _mock_pipeline()
        mocks["download"].side_effect = lambda *a, **kw: (call_order.append("download"), Path("video.mp4"))[1]
        mocks["transcribe"].side_effect = lambda *a, **kw: (call_order.append("transcribe"), _make_transcription())[1]
        mocks["analyze"].side_effect = lambda *a, **kw: (call_order.append("analyze"), [_make_clip()])[1]
        mocks["cut_clip"].side_effect = lambda *a, **kw: (call_order.append("cut_clip"), Path("video.mp4"))[1]
        mocks["add_captions"].side_effect = lambda *a, **kw: (call_order.append("add_captions"), Path("video.mp4"))[1]
        mocks["export_metadata"].side_effect = lambda *a, **kw: (call_order.append("export_metadata"), Path("clip_01.txt"))[1]

        result = _run_with_mocks(mocks)

        assert result.exit_code == 0
        assert call_order == [
            "download",
            "transcribe",
            "analyze",
            "cut_clip",
            "add_captions",
            "export_metadata",
        ]

    def test_all_pipeline_steps_called(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)

        assert result.exit_code == 0
        mocks["download"].assert_called_once()
        mocks["transcribe"].assert_called_once()
        mocks["analyze"].assert_called_once()
        mocks["cut_clip"].assert_called_once()
        mocks["add_captions"].assert_called_once()
        mocks["export_metadata"].assert_called_once()

    def test_clips_option_passed_to_config(self):
        clips = [_make_clip(f"Clipe {i}") for i in range(7)]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(mocks, extra_args=["--clips", "3"])

        assert result.exit_code == 0
        # cut_clip should be called 3 times (top 3 clips)
        assert mocks["cut_clip"].call_count == 3


# ---------------------------------------------------------------------------
# 8.5 — Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_exits_zero(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--dry-run"])
        assert result.exit_code == 0

    def test_dry_run_does_not_call_clipper(self):
        mocks = _mock_pipeline()
        _run_with_mocks(mocks, extra_args=["--dry-run"])
        mocks["cut_clip"].assert_not_called()

    def test_dry_run_does_not_call_captioner(self):
        mocks = _mock_pipeline()
        _run_with_mocks(mocks, extra_args=["--dry-run"])
        mocks["add_captions"].assert_not_called()

    def test_dry_run_does_not_call_exporter(self):
        mocks = _mock_pipeline()
        _run_with_mocks(mocks, extra_args=["--dry-run"])
        mocks["export_metadata"].assert_not_called()

    def test_dry_run_calls_download_transcribe_analyze(self):
        mocks = _mock_pipeline()
        _run_with_mocks(mocks, extra_args=["--dry-run"])
        mocks["download"].assert_called_once()
        mocks["transcribe"].assert_called_once()
        mocks["analyze"].assert_called_once()

    def test_dry_run_shows_table_output(self):
        mocks = _mock_pipeline(clips=[_make_clip("Clipe Viral")])
        result = _run_with_mocks(mocks, extra_args=["--dry-run"])
        assert "Clipe Viral" in result.output or "Dry Run" in result.output or "8.0" in result.output


# ---------------------------------------------------------------------------
# 8.6 — Summary table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def test_full_run_shows_clip_title_in_table(self):
        mocks = _mock_pipeline(clips=[_make_clip("Título Incrível")])
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        assert "Título Incrível" in result.output

    def test_full_run_shows_score_in_table(self):
        mocks = _mock_pipeline(clips=[_make_clip(score=9.5)])
        result = _run_with_mocks(mocks)
        assert "9.5" in result.output

    def test_dry_run_shows_clip_info(self):
        mocks = _mock_pipeline(clips=[_make_clip("Dry Run Clip", score=7.3)])
        result = _run_with_mocks(mocks, extra_args=["--dry-run"])
        assert "Dry Run Clip" in result.output or "7.3" in result.output


# ---------------------------------------------------------------------------
# 8.7 — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_download_error_exits_nonzero(self):
        from youcut.downloader import VideoDownloadError
        mocks = _mock_pipeline()
        mocks["download"].side_effect = VideoDownloadError("Link inválido")
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0

    def test_download_error_shows_friendly_message(self):
        from youcut.downloader import VideoDownloadError
        mocks = _mock_pipeline()
        mocks["download"].side_effect = VideoDownloadError("Link inválido")
        result = _run_with_mocks(mocks)
        assert "Link inválido" in result.output or "Download" in result.output

    def test_file_not_found_exits_nonzero(self):
        mocks = _mock_pipeline()
        mocks["download"].side_effect = FileNotFoundError("Arquivo não encontrado: video.mp4")
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0

    def test_file_not_found_shows_friendly_message(self):
        mocks = _mock_pipeline()
        mocks["download"].side_effect = FileNotFoundError("Arquivo não encontrado: video.mp4")
        result = _run_with_mocks(mocks)
        assert "video.mp4" in result.output or "encontrado" in result.output

    def test_api_error_exits_nonzero(self):
        mocks = _mock_pipeline()
        mocks["analyze"].side_effect = RuntimeError("Erro na API do Claude")
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0

    def test_api_error_shows_friendly_message(self):
        mocks = _mock_pipeline()
        mocks["analyze"].side_effect = RuntimeError("Erro na API do Claude")
        result = _run_with_mocks(mocks)
        assert "Claude" in result.output or "API" in result.output or "IA" in result.output

    def test_transcription_error_exits_nonzero(self):
        mocks = _mock_pipeline()
        mocks["transcribe"].side_effect = RuntimeError("Whisper não disponível")
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0

    def test_invalid_style_exits_nonzero(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--style", "invalid"])
        assert result.exit_code != 0

    def test_cut_clip_error_exits_nonzero(self):
        mocks = _mock_pipeline()
        mocks["cut_clip"].side_effect = RuntimeError("FFmpeg falhou")
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0

    def test_add_captions_error_exits_nonzero(self):
        mocks = _mock_pipeline(add_captions_side_effect=RuntimeError("Legendas falharam"))
        result = _run_with_mocks(mocks)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 8.8 — Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_file_option_accepted(self, tmp_path):
        log_file = tmp_path / "run.log"
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--log-file", str(log_file)])
        # The run itself should succeed; log file creation is best-effort checked via no crash
        assert result.exit_code == 0

    def test_log_level_option_accepted(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--log-level", "DEBUG"])
        assert result.exit_code == 0
