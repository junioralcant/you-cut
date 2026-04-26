"""
Integration tests for the full YouCut pipeline.

Full integration tests (TestPipelineIntegration) require:
  - FFmpeg in PATH
  - tests/fixtures/sample.mp4 (run tests/create_fixtures.py to generate)

Error-path tests (TestPipelineErrorPaths) do not require FFmpeg or the fixture.
"""
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic as real_anthropic
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"
_FIXTURE_MIN_SIZE = 10_000

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None


def _fixture_valid() -> bool:
    return FIXTURE.exists() and FIXTURE.stat().st_size > _FIXTURE_MIN_SIZE


def _run_pipeline(integration_video: Path, integration_config):
    from youcut.analyzer import analyze
    from youcut.captioner import add_captions
    from youcut.clipper import cut_clip
    from youcut.exporter import export_metadata
    from youcut.transcriber import transcribe

    transcription = transcribe(integration_video, integration_config)
    with patch("youcut.analyzer.anthropic.Anthropic", return_value=_make_mock_client()):
        clips = analyze(transcription, integration_config)

    assert clips, "Analyzer mock should return at least one valid clip"

    clip = clips[0]
    clip_path = cut_clip(integration_video, clip, 0, integration_config)
    captioned_path = add_captions(clip_path, transcription, clip, integration_config)
    output_dir = integration_config.output_dir / integration_video.stem
    metadata_path = export_metadata(clip, 0, output_dir)

    return transcription, clip, captioned_path, metadata_path


_needs_integration = pytest.mark.skipif(
    not _HAS_FFMPEG or not _fixture_valid(),
    reason=(
        "Integration tests require FFmpeg and a valid tests/fixtures/sample.mp4. "
        "Run: python tests/create_fixtures.py"
    ),
)

_MOCK_CLIP_DATA = {
    "title": "Teste de Clipe",
    "reason": "Teste de integração",
    "viral_score": 8.0,
    "start_time": 5.0,
    "end_time": 25.0,
    "description": "Descrição de teste",
    "hashtags": ["#teste"],
    "thumbnail_idea": "Frame de teste",
}


def _make_mock_client() -> MagicMock:
    """Return an Anthropic client mock yielding one fixed ViralClip."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "identify_viral_clips"
    block.input = {"clips": [_MOCK_CLIP_DATA]}

    response = MagicMock()
    response.content = [block]

    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Integration test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-integration")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("CLIP_COUNT", raising=False)
    monkeypatch.delenv("SUBTITLE_STYLE", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    from youcut.config import PipelineConfig
    return PipelineConfig(whisper_model="tiny", output_dir=tmp_path / "output")


@pytest.fixture
def integration_video(tmp_path) -> Path:
    """Copy sample.mp4 fixture to an isolated tmp directory."""
    video = tmp_path / "sample.mp4"
    shutil.copy(FIXTURE, video)
    return video


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

@_needs_integration
class TestPipelineIntegration:
    """End-to-end pipeline: real Whisper tiny + mocked Claude."""

    def test_full_pipeline_generates_expected_outputs(
        self, integration_config, integration_video
    ):
        """Pipeline runs end-to-end and generates the final video and metadata outputs."""
        transcription, clip, clip_path, txt_path = _run_pipeline(
            integration_video, integration_config
        )

        assert transcription.segments, "Real Whisper transcription should contain segments"
        assert clip.title == "Teste de Clipe"
        assert clip_path.exists()
        assert clip_path.stem.startswith("clip_01")
        assert clip_path.stat().st_size > 0
        assert txt_path.exists()
        assert txt_path.name == "clip_01.txt"
        assert txt_path.stat().st_size > 0

        content = txt_path.read_text(encoding="utf-8")
        assert "TÍTULO" in content
        assert "DESCRIÇÃO" in content
        assert "HASHTAGS" in content
        assert "SUGESTÃO DE THUMBNAIL" in content
        assert "NOTA DE VIRALIDADE" in content
        assert "MOTIVO DA SELEÇÃO" in content
        assert clip.title in content

    @pytest.mark.skipif(not _HAS_FFPROBE, reason="ffprobe required for resolution check")
    def test_clip_resolution_is_1080x1920(self, integration_config, integration_video):
        """Output clip has 1080×1920 resolution (9:16 vertical format)."""
        _, _, clip_path, _ = _run_pipeline(integration_video, integration_config)

        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(clip_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        video_stream = next(s for s in streams if s["codec_type"] == "video")
        assert video_stream["width"] == 1080
        assert video_stream["height"] == 1920

    def test_ass_file_cleaned_up_after_captioning(self, integration_config, integration_video):
        """.ass subtitle file is deleted after burn-in; clip_01.mp4 still exists."""
        _, _, clip_path, _ = _run_pipeline(integration_video, integration_config)

        assert clip_path.exists(), "clip_01.mp4 must exist after captioning"
        assert clip_path.stat().st_size > 0
        assert not clip_path.with_suffix(".ass").exists(), ".ass file must be cleaned up"

    def test_transcription_cache_created_after_first_run(
        self, integration_config, integration_video
    ):
        """First transcription creates a JSON cache file next to the video."""
        from youcut.transcriber import transcribe, _cache_path

        transcribe(integration_video, integration_config)

        cache_file = _cache_path(integration_video)
        assert cache_file.exists(), "Cache JSON must be written after first transcription"

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "md5" in data
        assert "result" in data

    def test_transcription_cache_reused_on_second_run(
        self, integration_config, integration_video
    ):
        """Second transcription of the same file skips Whisper (cache reused)."""
        from youcut.transcriber import transcribe

        result1 = transcribe(integration_video, integration_config)

        with patch("youcut.transcriber._transcribe_faster_whisper") as mock_fw, \
             patch("youcut.transcriber._transcribe_openai_whisper") as mock_ow:
            result2 = transcribe(integration_video, integration_config)
            mock_fw.assert_not_called()
            mock_ow.assert_not_called()

        assert result2.segments == result1.segments
        assert result2.language == result1.language


# ---------------------------------------------------------------------------
# Vertical fill pipeline tests (mock-based, no FFmpeg required)
# ---------------------------------------------------------------------------

class TestPipelineVerticalFill:
    """Validate vertical fill behavior via command inspection — no FFmpeg required."""

    @pytest.fixture
    def vf_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        return PipelineConfig(output_dir=tmp_path / "output")

    @pytest.fixture
    def video_path(self, tmp_path):
        p = tmp_path / "testvideo.mp4"
        p.write_bytes(b"\x00" * 100)
        return p

    @pytest.fixture
    def mock_clip(self):
        from youcut.models import ViralClip
        return ViralClip(
            title="Test Clip",
            reason="test",
            viral_score=8.0,
            start_time=5.0,
            end_time=25.0,
            description="desc",
            hashtags=["#test"],
            thumbnail_idea="thumb",
        )

    def test_pipeline_clip_resolution_1080x1920(self, vf_config, video_path, mock_clip):
        """Default fill_crop command targets 1080x1920 with scale+crop filter (no ffprobe needed)."""
        from youcut.clipper import cut_clip

        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("youcut.clipper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cut_clip(video_path, mock_clip, 0, vf_config)

        full_cmd = " ".join(mock_run.call_args[0][0])
        assert "1080" in full_cmd
        assert "1920" in full_cmd
        assert "crop=1080:1920" in full_cmd

    def test_pipeline_no_black_pad_in_default(self, vf_config, video_path, mock_clip):
        """Default fill_crop mode must not include pad=1080:1920 in the FFmpeg command."""
        from youcut.clipper import cut_clip

        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("youcut.clipper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cut_clip(video_path, mock_clip, 0, vf_config)

        full_cmd = " ".join(mock_run.call_args[0][0])
        assert "pad=1080:1920" not in full_cmd

    def test_pipeline_preview_artifact_exists(self, vf_config, video_path, mock_clip):
        """Preview artifact is created in the same directory as the clip."""
        from youcut.preview import generate_clip_preview

        def _fake_ffmpeg(cmd, **kwargs):
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\xff\xd8\xff")
            return MagicMock(returncode=0)

        with patch("youcut.preview.subprocess.run", side_effect=_fake_ffmpeg):
            artifact = generate_clip_preview(video_path, mock_clip, 0, vf_config)

        expected_dir = vf_config.output_dir / video_path.stem
        assert artifact is not None
        assert artifact.path.exists()
        assert artifact.path.parent == expected_dir
        assert artifact.path.name == "clip_01_preview.jpg"

    def test_pipeline_preview_filter_parity(self, vf_config, video_path, mock_clip):
        """Preview uses exactly the same filter as the export (same builder)."""
        from youcut.clipper import build_vertical_fill_filter
        from youcut.preview import generate_clip_preview

        with patch("youcut.preview.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            generate_clip_preview(video_path, mock_clip, 0, vf_config)

        cmd = mock_run.call_args[0][0]
        vf_index = cmd.index("-vf") + 1
        preview_filter = cmd[vf_index]

        assert preview_filter == build_vertical_fill_filter()

    def test_pipeline_blur_background_still_works(self, vf_config, video_path, mock_clip):
        """Legacy blur_background mode completes without error and emits boxblur filter."""
        from youcut.clipper import cut_clip

        blur_config = vf_config.model_copy(update={"vertical_fill_mode": "blur_background"})

        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("youcut.clipper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = cut_clip(video_path, mock_clip, 0, blur_config)

        assert result is not None
        full_cmd = " ".join(mock_run.call_args[0][0])
        assert "boxblur" in full_cmd


# ---------------------------------------------------------------------------
# Error path tests (no FFmpeg or real fixture required)
# ---------------------------------------------------------------------------

class TestPipelineErrorPaths:
    """Error paths tests — mock heavy dependencies, no FFmpeg or video fixture needed."""

    def test_ffmpeg_missing_raises_runtime_error(self):
        """check_ffmpeg() raises RuntimeError with a user-friendly message."""
        from youcut.clipper import check_ffmpeg

        with patch("youcut.clipper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="FFmpeg não encontrado"):
                check_ffmpeg()

    def test_missing_api_key_fails_before_processing(self, monkeypatch, tmp_path):
        """PipelineConfig raises ValidationError immediately when ANTHROPIC_API_KEY is absent."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        from pydantic import ValidationError
        from youcut.config import PipelineConfig

        with pytest.raises(ValidationError):
            PipelineConfig()

    def test_local_file_not_found_raises_file_not_found_error(self, tmp_path):
        """download_video() raises FileNotFoundError for a non-existent local path."""
        from youcut.downloader import download_video

        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError, match="does_not_exist.mp4"):
            download_video(str(missing), tmp_path / "downloads")

    def test_download_error_wrapped_as_video_download_error(self, tmp_path):
        """yt-dlp DownloadError is wrapped as VideoDownloadError with a friendly message."""
        import yt_dlp

        from youcut.downloader import VideoDownloadError, download_video

        mock_ydl = MagicMock()
        mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("connection refused")

        with patch("youcut.downloader.yt_dlp.YoutubeDL", return_value=mock_ydl):
            with pytest.raises(VideoDownloadError):
                download_video("https://youtube.com/watch?v=test", tmp_path / "downloads")

    def test_api_error_raises_runtime_error_with_friendly_message(self, monkeypatch):
        """Claude APIError is re-raised as RuntimeError with a user-friendly message."""
        from youcut.analyzer import analyze
        from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig

        config = PipelineConfig()
        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=30.0,
                    text="Teste de conteúdo",
                    words=[WordTimestamp(word="Teste", start=0.0, end=1.0)],
                )
            ],
            language="pt",
            source_path=Path("test.mp4"),
        )

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.messages.create.side_effect = real_anthropic.APIConnectionError(
            request=MagicMock()
        )

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Erro na API do Claude"):
                analyze(transcription, config)
