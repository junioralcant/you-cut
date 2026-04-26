"""Integration tests for the Flow A (YouTube mode) pipeline."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.models import (
    ClipRecord,
    SessionData,
    TranscriptionResult,
    TranscriptionSegment,
    ViralClip,
    WordTimestamp,
)


@pytest.fixture
def youtube_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from youcut.config import PipelineConfig
    return PipelineConfig(
        cut_mode="youtube",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def mock_transcription(tmp_path):
    tr = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=400.0,
                text="Long discussion about technology and innovation.",
                words=[WordTimestamp(word="Long", start=0.0, end=0.5)],
            )
        ],
        language="pt",
        source_path=tmp_path / "downloads" / "test_video.mp4",
    )
    return tr


@pytest.fixture
def mock_viral_clip_youtube():
    return ViralClip(
        title="Tech Innovation Discussion",
        reason="Informative content with clear narrative",
        viral_score=8.5,
        start_time=10.0,
        end_time=370.0,  # 360s = 6 min (within youtube 5–20 min range)
        description="Deep dive into tech innovation",
        hashtags=["#tech", "#innovation"],
        thumbnail_idea="Speaker with confident expression discussing technology",
        cut_mode="youtube",
    )


@pytest.fixture
def video_path(tmp_path):
    p = tmp_path / "downloads" / "test_video.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


@pytest.fixture
def clip_path(tmp_path):
    p = tmp_path / "output" / "test_video" / "clip_01.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _make_cache(tmp_path, transcription: TranscriptionResult) -> Path:
    cache_data = {
        "md5": "abc123",
        "result": json.loads(transcription.model_dump_json()),
    }
    cache_path = tmp_path / "downloads" / "test_video_transcript.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
    return cache_path


class TestFlowASavesSession:
    def test_session_saved_after_flow_a(
        self, tmp_path, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        saved_sessions = []
        cache_path = _make_cache(tmp_path, mock_transcription)

        def side_effect_transcribe(vp, cfg):
            # Also write the cache file (mirroring real transcriber behavior)
            return mock_transcription

        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", side_effect=side_effect_transcribe),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", side_effect=lambda s: saved_sessions.append(s) or Path("/tmp/s.json")),
        ):
            from youcut.cli import run_flow_a
            result = run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        assert result is not None, "run_flow_a should return a SessionData"
        assert len(saved_sessions) == 1

        session = saved_sessions[0]
        assert session.cut_mode == "youtube"
        assert session.source_url == "https://youtube.com/watch?v=test"
        assert len(session.clips) == 1
        assert session.clips[0].title == "Tech Innovation Discussion"

    def test_transcription_cache_path_in_session(
        self, tmp_path, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        saved_sessions = []

        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", side_effect=lambda s: saved_sessions.append(s) or Path("/tmp/s.json")),
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        assert len(saved_sessions) == 1
        session = saved_sessions[0]
        # Path must follow <parent>/<stem>_transcript.json convention
        assert "_transcript.json" in session.transcription_cache_path.name
        assert "test_video" in session.transcription_cache_path.name

    def test_transcribe_called_exactly_once(
        self, tmp_path, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription) as mock_tr,
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        mock_tr.assert_called_once()


class TestFlowAYouTubeClipFormat:
    def test_clipper_uses_stream_copy_for_youtube(
        self, tmp_path, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """Verifies that cut_clip is called with a youtube-mode ViralClip (cut_mode='youtube')."""
        cut_calls = []

        def capture_cut(vp, clip, idx, cfg):
            cut_calls.append(clip)
            return clip_path

        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", side_effect=capture_cut),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        assert len(cut_calls) == 1
        # The clip passed to cut_clip must have cut_mode="youtube"
        assert cut_calls[0].cut_mode == "youtube"

    def test_no_thumbnail_generated_without_openai_key(
        self, tmp_path, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """When openai_api_key is absent, no thumbnails are generated."""
        assert youtube_config.openai_api_key is None

        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
            patch("youcut.cli.generate_thumbnail") as mock_thumb,
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        mock_thumb.assert_not_called()

    def test_thumbnail_generated_when_openai_key_present(
        self, tmp_path, monkeypatch, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """When openai_api_key is set, generate_thumbnail is called for each clip."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")
        assert config.openai_api_key == "openai-test-key"

        thumb_path = tmp_path / "output" / "test_video" / "thumbnails" / "clip_00.png"
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb_path.touch()

        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
            patch("youcut.cli.generate_thumbnail", return_value=thumb_path) as mock_thumb,
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                config,
                skip_review=True,
                upload=False,
            )

        mock_thumb.assert_called_once()
        # Session should have the thumbnail path set


class TestFlowAUploadPrompt:
    def test_manual_run_can_offer_upload_after_review(
        self, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
            patch("youcut.cli.sys.stdin.isatty", return_value=True),
            patch("youcut.cli.sys.stdout.isatty", return_value=True),
            patch("youcut.cli.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=True))),
            patch(
                "youcut.cli.questionary.checkbox",
                return_value=MagicMock(ask=MagicMock(return_value=["youtube"])),
            ),
            patch("youcut.cli._publish_clips_with_status") as mock_publish,
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        mock_publish.assert_called_once()
        approved_records, selected_platforms, _ = mock_publish.call_args.args
        assert len(approved_records) == 1
        assert selected_platforms == ["youtube"]

    def test_non_interactive_run_does_not_prompt_or_upload(
        self, youtube_config, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        with (
            patch("youcut.cli.download_video", return_value=video_path),
            patch("youcut.cli.transcribe", return_value=mock_transcription),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip_youtube]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli.save_session", return_value=Path("/tmp/s.json")),
            patch("youcut.cli.sys.stdin.isatty", return_value=False),
            patch("youcut.cli.questionary.confirm") as mock_confirm,
            patch("youcut.cli._publish_clips_with_status") as mock_publish,
        ):
            from youcut.cli import run_flow_a
            run_flow_a(
                "https://youtube.com/watch?v=test",
                youtube_config,
                skip_review=True,
                upload=False,
            )

        mock_confirm.assert_not_called()
        mock_publish.assert_not_called()
