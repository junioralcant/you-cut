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
        thumbnail_text="MOMENTO IMPACTANTE",
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
    mock_preview = MagicMock(return_value=MagicMock(path=Path("clip_01_preview.jpg")))
    mock_title_overlay = MagicMock(return_value=video_path)
    mock_upload = MagicMock(return_value=[])
    mock_prompt_clip_selection = MagicMock(return_value=None)

    return {
        "download": mock_download,
        "transcribe": mock_transcribe,
        "analyze": mock_analyze,
        "cut_clip": mock_cut,
        "add_captions": mock_caption,
        "export_metadata": mock_export,
        "generate_clip_preview": mock_preview,
        "add_title_overlay": mock_title_overlay,
        "upload_clips": mock_upload,
        "prompt_clip_selection": mock_prompt_clip_selection,
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
        patch("youcut.cli.generate_clip_preview", mocks["generate_clip_preview"]),
        patch("youcut.cli.add_title_overlay", mocks["add_title_overlay"]),
        patch("youcut.cli.upload_clips", mocks["upload_clips"]),
        patch("youcut.cli.prompt_clip_selection", mocks["prompt_clip_selection"]),
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
        assert "--clip-count" in result.output

    def test_help_shows_upload_clips_option(self):
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

    def test_help_shows_title_overlay_option(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--title-overlay" in result.output


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

    def test_missing_api_key_friendly_message(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = runner.invoke(app, ["run", "video.mp4"])
        output = result.output
        assert "ANTHROPIC_API_KEY" in output or "configuração" in output.lower() or "api" in output.lower()

    def test_missing_api_key_no_pipeline_started(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
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

        mock_preview_artifact = MagicMock(path=Path("clip_01_preview.jpg"))
        mocks = _mock_pipeline()
        mocks["download"].side_effect = lambda *a, **kw: (call_order.append("download"), Path("video.mp4"))[1]
        mocks["transcribe"].side_effect = lambda *a, **kw: (call_order.append("transcribe"), _make_transcription())[1]
        mocks["analyze"].side_effect = lambda *a, **kw: (call_order.append("analyze"), [_make_clip()])[1]
        mocks["cut_clip"].side_effect = lambda *a, **kw: (call_order.append("cut_clip"), Path("video.mp4"))[1]
        mocks["generate_clip_preview"].side_effect = lambda *a, **kw: (call_order.append("generate_clip_preview"), mock_preview_artifact)[1]
        mocks["add_captions"].side_effect = lambda *a, **kw: (call_order.append("add_captions"), Path("video.mp4"))[1]
        mocks["export_metadata"].side_effect = lambda *a, **kw: (call_order.append("export_metadata"), Path("clip_01.txt"))[1]

        result = _run_with_mocks(mocks)

        assert result.exit_code == 0
        assert call_order == [
            "download",
            "transcribe",
            "analyze",
            "cut_clip",
            "generate_clip_preview",
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
        mocks["generate_clip_preview"].assert_called_once()
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

    def test_full_run_shows_clip_title_again(self):
        mocks = _mock_pipeline(clips=[_make_clip("Clipe Score Teste")])
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        assert "Clipe Score Teste" in result.output

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


# ---------------------------------------------------------------------------
# 8.9 — Preview integration
# ---------------------------------------------------------------------------

class TestPreviewIntegration:
    def test_cli_requests_preview_in_normal_flow(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        mocks["generate_clip_preview"].assert_called_once()

    def test_cli_dry_run_no_preview(self):
        mocks = _mock_pipeline()
        _run_with_mocks(mocks, extra_args=["--dry-run"])
        mocks["generate_clip_preview"].assert_not_called()

    def test_cli_generate_preview_called_in_normal_flow(self):
        preview_path = Path("clip_01_preview.jpg")
        mock_artifact = MagicMock(path=preview_path)
        mocks = _mock_pipeline()
        mocks["generate_clip_preview"].return_value = mock_artifact
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        mocks["generate_clip_preview"].assert_called_once()


# ---------------------------------------------------------------------------
# 8.10 — Title overlay flag
# ---------------------------------------------------------------------------

class TestTitleOverlay:
    def test_without_flag_title_overlay_not_called(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        mocks["add_title_overlay"].assert_not_called()

    def test_with_flag_title_overlay_called_for_each_clip(self):
        clips = [_make_clip("Clipe A"), _make_clip("Clipe B")]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(mocks, extra_args=["--title-overlay"])
        assert result.exit_code == 0
        assert mocks["add_title_overlay"].call_count == len(clips)

    def test_with_flag_title_overlay_called_with_correct_args(self):
        clip = _make_clip("Meu Título")
        video_path = Path("video.mp4")
        mocks = _mock_pipeline(clips=[clip], video_path=video_path)
        result = _run_with_mocks(mocks, extra_args=["--title-overlay"])
        assert result.exit_code == 0
        call_args = mocks["add_title_overlay"].call_args
        _, called_clip, called_config = call_args.args
        assert called_clip is clip
        assert called_config.title_overlay is True

    def test_with_flag_exits_zero(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--title-overlay"])
        assert result.exit_code == 0

    def test_title_overlay_error_exits_nonzero(self):
        mocks = _mock_pipeline()
        mocks["add_title_overlay"].side_effect = RuntimeError("Overlay falhou")
        result = _run_with_mocks(mocks, extra_args=["--title-overlay"])
        assert result.exit_code != 0

    def test_dry_run_with_title_overlay_does_not_call_overlay(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--dry-run", "--title-overlay"])
        assert result.exit_code == 0
        mocks["add_title_overlay"].assert_not_called()

    def test_pipeline_order_with_title_overlay(self):
        call_order: list[str] = []

        mock_preview_artifact = MagicMock(path=Path("clip_01_preview.jpg"))
        mocks = _mock_pipeline()
        mocks["download"].side_effect = lambda *a, **kw: (call_order.append("download"), Path("video.mp4"))[1]
        mocks["transcribe"].side_effect = lambda *a, **kw: (call_order.append("transcribe"), _make_transcription())[1]
        mocks["analyze"].side_effect = lambda *a, **kw: (call_order.append("analyze"), [_make_clip()])[1]
        mocks["cut_clip"].side_effect = lambda *a, **kw: (call_order.append("cut_clip"), Path("video.mp4"))[1]
        mocks["generate_clip_preview"].side_effect = lambda *a, **kw: (call_order.append("generate_clip_preview"), mock_preview_artifact)[1]
        mocks["add_captions"].side_effect = lambda *a, **kw: (call_order.append("add_captions"), Path("video.mp4"))[1]
        mocks["add_title_overlay"].side_effect = lambda *a, **kw: (call_order.append("add_title_overlay"), Path("video.mp4"))[1]
        mocks["export_metadata"].side_effect = lambda *a, **kw: (call_order.append("export_metadata"), Path("clip_01.txt"))[1]

        result = _run_with_mocks(mocks, extra_args=["--title-overlay"])

        assert result.exit_code == 0
        assert call_order == [
            "download",
            "transcribe",
            "analyze",
            "cut_clip",
            "generate_clip_preview",
            "add_captions",
            "add_title_overlay",
            "export_metadata",
        ]


# ---------------------------------------------------------------------------
# 8.11 — Upload integration
# ---------------------------------------------------------------------------

class TestUploadIntegration:
    def test_run_without_upload_does_not_call_upload_clips(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)
        assert result.exit_code == 0
        mocks["upload_clips"].assert_not_called()

    def test_run_with_upload_calls_upload_clips_for_all_platforms(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--upload", "--platforms", "all"])

        assert result.exit_code == 0
        kwargs = mocks["upload_clips"].call_args.kwargs
        assert kwargs["platforms"] == ["youtube", "instagram", "tiktok"]
        assert kwargs["clips_filter"] is None

    def test_run_with_upload_uses_clips_flag_as_clip_count(self):
        clips = [_make_clip(f"Clipe {i}") for i in range(6)]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(
            mocks,
            extra_args=["--upload", "--platforms", "youtube", "--clips", "2"],
        )

        assert result.exit_code == 0
        assert mocks["cut_clip"].call_count == 2
        kwargs = mocks["upload_clips"].call_args.kwargs
        assert kwargs["platforms"] == ["youtube"]
        assert kwargs["clips_filter"] is None

    def test_run_with_upload_uses_clip_count_from_explicit_flag(self):
        clips = [_make_clip(f"Clipe {i}") for i in range(6)]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(
            mocks,
            extra_args=["--upload", "--clip-count", "2"],
        )

        assert result.exit_code == 0
        assert mocks["cut_clip"].call_count == 2

    def test_invalid_platforms_exit_nonzero_before_processing(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--upload", "--platforms", "xyz"])

        assert result.exit_code != 0
        assert "platform" in result.output.lower() or "plataforma" in result.output.lower()
        mocks["download"].assert_not_called()

    def test_invalid_upload_clips_exit_nonzero_before_processing(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--upload", "--clips", "abc"])

        assert result.exit_code != 0
        assert "--clips" in result.output or "número inteiro" in result.output.lower()
        mocks["download"].assert_not_called()

    def test_invalid_non_upload_clips_exit_nonzero_before_processing(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--clips", "1,3"])

        assert result.exit_code != 0
        assert "número inteiro" in result.output.lower() or "--clips" in result.output.lower()
        mocks["download"].assert_not_called()

    def test_run_with_upload_and_no_clips_flag_uploads_all(self):
        clips = [_make_clip("A"), _make_clip("B"), _make_clip("C")]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(
            mocks,
            extra_args=["--upload", "--platforms", "youtube"],
        )

        assert result.exit_code == 0
        kwargs = mocks["upload_clips"].call_args.kwargs
        assert kwargs["clips_filter"] is None


# ---------------------------------------------------------------------------
# 8.11.1 — prompt_clip_selection integration
# ---------------------------------------------------------------------------

class TestPromptClipSelection:
    def test_prompt_clip_selection_not_called_with_upload_flag(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks, extra_args=["--upload", "--platforms", "youtube"])

        assert result.exit_code == 0
        mocks["prompt_clip_selection"].assert_not_called()

    def test_upload_clips_receives_all_generated_clips_when_using_clips_flag(self):
        clips = [_make_clip(f"Clipe {i}") for i in range(5)]
        mocks = _mock_pipeline(clips=clips)
        result = _run_with_mocks(mocks, extra_args=["--upload", "--platforms", "youtube", "--clips", "3"])

        assert result.exit_code == 0
        assert mocks["cut_clip"].call_count == 3
        kwargs = mocks["upload_clips"].call_args.kwargs
        assert kwargs["clips_filter"] is None

    def test_prompt_clip_selection_not_called_without_upload_flag(self):
        mocks = _mock_pipeline()
        result = _run_with_mocks(mocks)

        assert result.exit_code == 0
        mocks["prompt_clip_selection"].assert_not_called()

    def test_none_selection_passes_none_as_clips_filter(self):
        clips = [_make_clip("A"), _make_clip("B")]
        mocks = _mock_pipeline(clips=clips)
        mocks["prompt_clip_selection"].return_value = None
        result = _run_with_mocks(mocks, extra_args=["--upload", "--platforms", "youtube"])

        assert result.exit_code == 0
        kwargs = mocks["upload_clips"].call_args.kwargs
        assert kwargs["clips_filter"] is None


# ---------------------------------------------------------------------------
# 8.12 — Auth subcommands
# ---------------------------------------------------------------------------

class TestAuthCommands:
    def test_auth_login_youtube_calls_authenticate(self):
        fake_uploader = MagicMock()
        with patch("youcut.cli.YouTubeUploader", return_value=fake_uploader):
            result = runner.invoke(app, ["auth", "login", "--platform", "youtube"], env=API_ENV)

        assert result.exit_code == 0
        fake_uploader.authenticate.assert_called_once()

    def test_auth_login_youtube_reads_client_secrets_from_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("YOUTUBE_CLIENT_SECRETS_FILE", raising=False)
        (tmp_path / ".env").write_text(
            "YOUTUBE_CLIENT_SECRETS_FILE=oauth/client_secret.json\n",
            encoding="utf-8",
        )

        fake_uploader = MagicMock()
        with patch("youcut.cli.YouTubeUploader", return_value=fake_uploader) as mock_youtube_uploader:
            result = runner.invoke(
                app,
                ["auth", "login", "--platform", "youtube"],
                env=API_ENV,
            )

        assert result.exit_code == 0
        mock_youtube_uploader.assert_called_once_with(
            token_dir=Path.home() / ".youcut" / "credentials",
            client_secrets_file=Path("oauth/client_secret.json"),
        )
        fake_uploader.authenticate.assert_called_once()

    def test_auth_login_instagram_calls_authenticate(self):
        fake_uploader = MagicMock()
        with patch("youcut.cli.InstagramUploader", return_value=fake_uploader):
            result = runner.invoke(app, ["auth", "login", "--platform", "instagram"], env=API_ENV)

        assert result.exit_code == 0
        fake_uploader.authenticate.assert_called_once()

    def test_auth_revoke_instagram_calls_revoke_token(self):
        with patch("youcut.cli.revoke_token") as mock_revoke:
            result = runner.invoke(app, ["auth", "revoke", "--platform", "instagram"], env=API_ENV)

        assert result.exit_code == 0
        mock_revoke.assert_called_once()
        assert mock_revoke.call_args.args[0] == "instagram"

    def test_auth_status_shows_three_platforms(self):
        with patch(
            "youcut.cli.get_token",
            side_effect=lambda platform, _: {"token": "x"} if platform in {"youtube", "tiktok"} else None,
        ):
            result = runner.invoke(app, ["auth", "status"], env=API_ENV)

        assert result.exit_code == 0
        assert "youtube" in result.output
        assert "instagram" in result.output
        assert "tiktok" in result.output


def test_cli_thumbnail_text_flag_passed_to_flow_a_config():
    fake_session = MagicMock()
    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("youcut.cli._resolve_cli_yt_dlp_auth_config", return_value=None),
        patch("youcut.cli._select_cut_mode", return_value="youtube"),
        patch("youcut.cli.normalize_video_url", return_value="video.mp4"),
        patch("youcut.cli.fetch_metadata", return_value=MagicMock(title="Video", duration_seconds=120.0)),
        patch("youcut.cli.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=True))),
        patch("youcut.cli._can_prompt_interactively", return_value=False),
        patch("youcut.cli.run_flow_a", return_value=fake_session) as mock_flow_a,
        patch("youcut.cli.offer_flow_b"),
    ):
        result = runner.invoke(app, ["cuts", "video.mp4", "--max-clips", "1", "--thumbnail-text", "TITLE"], env=API_ENV)

    assert result.exit_code == 0
    config = mock_flow_a.call_args.args[1]
    assert config.thumbnail_text == "TITLE"


def test_cli_without_thumbnail_text_uses_empty_default():
    fake_session = MagicMock()
    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("youcut.cli._resolve_cli_yt_dlp_auth_config", return_value=None),
        patch("youcut.cli._select_cut_mode", return_value="youtube"),
        patch("youcut.cli.normalize_video_url", return_value="video.mp4"),
        patch("youcut.cli.fetch_metadata", return_value=MagicMock(title="Video", duration_seconds=120.0)),
        patch("youcut.cli.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=True))),
        patch("youcut.cli._can_prompt_interactively", return_value=False),
        patch("youcut.cli.run_flow_a", return_value=fake_session) as mock_flow_a,
        patch("youcut.cli.offer_flow_b"),
    ):
        result = runner.invoke(app, ["cuts", "video.mp4", "--max-clips", "1"], env=API_ENV)

    assert result.exit_code == 0
    config = mock_flow_a.call_args.args[1]
    assert config.thumbnail_text == ""
