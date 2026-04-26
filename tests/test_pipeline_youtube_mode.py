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
        self, tmp_path, monkeypatch, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """When openai_api_key is absent, no thumbnails are generated."""
        # chdir to tmp_path so pydantic-settings doesn't pick up the project's .env
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")
        assert config.openai_api_key is None

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
                config,
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


# ---------------------------------------------------------------------------
# Task 5.0 — _publish_clips_with_status: real metadata + persistence
# ---------------------------------------------------------------------------

class TestPublishClipsWithStatus:
    def _make_record(self, tmp_path, *, thumbnail: bool = False) -> "ClipRecord":
        clip = tmp_path / "clip_01.mp4"
        clip.write_bytes(b"\x00" * 10)
        thumb = None
        if thumbnail:
            thumb = tmp_path / "thumb.jpg"
            thumb.write_bytes(b"\xff\xd8\xff")
        return ClipRecord(
            title="Test Clip",
            start_time=0.0,
            end_time=60.0,
            clip_path=clip,
            thumbnail_path=thumb,
        )

    def test_upload_status_set_on_success(self, tmp_path, monkeypatch):
        """upload_status['youtube'] is set to 'success' after a successful upload."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success", url="https://youtu.be/abc", video_id="abc", error=None
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        assert record.upload_status["youtube"] == "success"

    def test_youtube_video_id_stored_in_record(self, tmp_path, monkeypatch):
        """youtube_video_id and youtube_url are stored in ClipRecord after YouTube success."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success", url="https://youtu.be/vid999", video_id="vid999", error=None
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        assert record.youtube_video_id == "vid999"
        assert record.youtube_url == "https://youtu.be/vid999"

    def test_thumbnail_path_passed_to_youtube_uploader(self, tmp_path, monkeypatch):
        """thumbnail_path from ClipRecord is forwarded to YouTubeUploader.upload()."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path, thumbnail=True)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success", url="https://youtu.be/t1", video_id="t1", error=None
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        call_kwargs = mock_uploader.upload.call_args.kwargs
        assert call_kwargs.get("thumbnail_path") == record.thumbnail_path

    def test_upload_status_failed_on_upload_error(self, tmp_path, monkeypatch):
        """upload_status['youtube'] is set to 'failed' when upload returns failed."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="failed", url=None, video_id=None, error="quota exceeded"
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        assert record.upload_status["youtube"] == "failed"
        assert record.youtube_video_id is None

    def test_session_saved_after_upload(self, tmp_path, monkeypatch):
        """save_session is called with the session after uploads complete."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.models import SessionData
        from datetime import datetime
        record = self._make_record(tmp_path)
        session = SessionData(
            session_id="sess-1",
            source_url="https://youtube.com/watch?v=test",
            cut_mode="youtube",
            transcription_cache_path=tmp_path / "cache.json",
            clips=[record],
            created_at=datetime.now(),
            output_dir=tmp_path / "out",
        )

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success", url="https://youtu.be/s1", video_id="s1", error=None
        )

        saved = []
        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()), \
             patch("youcut.cli.save_session", side_effect=lambda s: saved.append(s) or tmp_path / "s.json"):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds", session=session)

        assert len(saved) == 1
        assert saved[0] is session

    def test_multiple_platforms_one_fails_other_continues(self, tmp_path, monkeypatch):
        """When one platform fails, the other still gets its status recorded."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path)

        mock_yt = MagicMock()
        mock_yt.upload.return_value = MagicMock(
            status="failed", url=None, video_id=None, error="quota"
        )
        mock_ig = MagicMock()
        mock_ig.upload.return_value = MagicMock(
            status="success", url="https://instagram.com/p/abc", video_id=None, error=None
        )

        def build_uploader(cls_mock):
            return cls_mock

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_yt), \
             patch("youcut.cli.InstagramUploader", return_value=mock_ig), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube", "instagram"], tmp_path / "creds")

        assert record.upload_status["youtube"] == "failed"
        assert record.upload_status["instagram"] == "success"

    def test_real_title_used_in_metadata(self, tmp_path, monkeypatch):
        """ClipRecord.title is passed as metadata title to the uploader."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path)
        record_title = record.title

        captured_metadata = []
        mock_uploader = MagicMock()
        mock_uploader.upload.side_effect = lambda path, meta, **kw: (
            captured_metadata.append(meta) or
            MagicMock(status="success", url="https://youtu.be/x", video_id="x", error=None)
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        assert len(captured_metadata) == 1
        assert captured_metadata[0].title == record_title


# ---------------------------------------------------------------------------
# Task 7.0 — Auto mode menu
# ---------------------------------------------------------------------------

class TestAutoMode:
    def test_auto_mode_confirmed_sets_skip_review_true(self, tmp_path, monkeypatch):
        """When user confirms auto mode, run_flow_a is called with skip_review=True."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        run_flow_a_calls = []

        def fake_run_flow_a(source, cfg, *, skip_review, **kw):
            run_flow_a_calls.append({"skip_review": skip_review})
            return None

        with (
            patch("youcut.cli.sys.stdin.isatty", return_value=True),
            patch("youcut.cli.sys.stdout.isatty", return_value=True),
            patch("youcut.cli._select_cut_mode", return_value="youtube"),
            patch("youcut.cli.questionary.text", return_value=MagicMock(ask=MagicMock(return_value=""))),
            patch("youcut.cli.normalize_video_url", return_value="https://youtube.com/watch?v=test"),
            patch("youcut.cli.fetch_metadata", return_value=MagicMock(title="Test", duration_seconds=600)),
            patch("youcut.cli.questionary.confirm", side_effect=[
                MagicMock(ask=MagicMock(return_value=True)),   # "Prosseguir com este vídeo?"
                MagicMock(ask=MagicMock(return_value=True)),   # "Executar tudo automaticamente?"
            ]),
            patch("youcut.cli._parse_platforms", return_value=["youtube"]),
            patch("youcut.cli.run_flow_a", side_effect=fake_run_flow_a),
            patch("youcut.cli._check_ffmpeg"),
            patch("youcut.cli._configure_logging"),
        ):
            from youcut.cli import cuts
            cuts(
                source="https://youtube.com/watch?v=test",
                history=False,
                max_clips=None,
                skip_review=False,
                upload=False,
                platforms_raw="youtube",
                log_level="INFO",
                log_file=None,
            )

        assert len(run_flow_a_calls) == 1
        assert run_flow_a_calls[0]["skip_review"] is True

    def test_auto_mode_declined_preserves_original_skip_review(self, tmp_path, monkeypatch):
        """When user declines auto mode, run_flow_a uses original skip_review value (False)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        run_flow_a_calls = []

        def fake_run_flow_a(source, cfg, *, skip_review, **kw):
            run_flow_a_calls.append({"skip_review": skip_review})
            return None

        with (
            patch("youcut.cli.sys.stdin.isatty", return_value=True),
            patch("youcut.cli.sys.stdout.isatty", return_value=True),
            patch("youcut.cli._select_cut_mode", return_value="youtube"),
            patch("youcut.cli.questionary.text", return_value=MagicMock(ask=MagicMock(return_value=""))),
            patch("youcut.cli.normalize_video_url", return_value="https://youtube.com/watch?v=test"),
            patch("youcut.cli.fetch_metadata", return_value=MagicMock(title="Test", duration_seconds=600)),
            patch("youcut.cli.questionary.confirm", side_effect=[
                MagicMock(ask=MagicMock(return_value=True)),    # "Prosseguir?"
                MagicMock(ask=MagicMock(return_value=False)),   # "Automático?" — declined
            ]),
            patch("youcut.cli._parse_platforms", return_value=["youtube"]),
            patch("youcut.cli.run_flow_a", side_effect=fake_run_flow_a),
            patch("youcut.cli._check_ffmpeg"),
            patch("youcut.cli._configure_logging"),
        ):
            from youcut.cli import cuts
            cuts(
                source="https://youtube.com/watch?v=test",
                history=False,
                max_clips=None,
                skip_review=False,
                upload=False,
                platforms_raw="youtube",
                log_level="INFO",
                log_file=None,
            )

        assert len(run_flow_a_calls) == 1
        assert run_flow_a_calls[0]["skip_review"] is False
