"""Integration tests for the Flow A (YouTube mode) pipeline."""

import json
import uuid
from contextlib import ExitStack
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
        thumbnail_text="MOMENTO IMPACTANTE",
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

    def test_thumbnail_always_generated(
        self, tmp_path, monkeypatch, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """generate_thumbnail is always called, regardless of openai_api_key."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

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

    def test_thumbnail_generated_when_openai_key_present(
        self, tmp_path, monkeypatch, mock_transcription, mock_viral_clip_youtube, video_path, clip_path
    ):
        """generate_thumbnail is called for each clip (OpenAI key no longer required)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

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
            status="success",
            url="https://youtu.be/abc",
            video_id="abc",
            error=None,
            warning=None,
            thumbnail_status="uploaded",
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
            status="success",
            url="https://youtu.be/vid999",
            video_id="vid999",
            error=None,
            warning=None,
            thumbnail_status="uploaded",
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
            status="success",
            url="https://youtu.be/t1",
            video_id="t1",
            error=None,
            warning=None,
            thumbnail_status="uploaded",
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
            status="success",
            url="https://youtu.be/s1",
            video_id="s1",
            error=None,
            warning=None,
            thumbnail_status="uploaded",
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
            MagicMock(
                status="success",
                url="https://youtu.be/x",
                video_id="x",
                error=None,
                warning=None,
                thumbnail_status="uploaded",
            )
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()):
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        assert len(captured_metadata) == 1
        assert captured_metadata[0].title == record_title

    def test_partial_youtube_success_prints_warning_and_preserves_video_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path, thumbnail=True)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success",
            url="https://youtu.be/partial1",
            video_id="partial1",
            error=None,
            warning="Video publicado, mas a thumbnail nao foi aplicada.",
            thumbnail_status="failed",
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()), \
             patch("youcut.cli._console.print") as mock_print:
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "thumbnail falhou" in printed
        assert "https://youtu.be/partial1" in printed
        assert record.upload_status["youtube"] == "success"
        assert record.youtube_video_id == "partial1"
        assert record.youtube_url == "https://youtu.be/partial1"

    def test_complete_youtube_success_mentions_thumbnail_applied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        record = self._make_record(tmp_path, thumbnail=True)

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = MagicMock(
            status="success",
            url="https://youtu.be/full1",
            video_id="full1",
            error=None,
            warning=None,
            thumbnail_status="uploaded",
        )

        from youcut.cli import _publish_clips_with_status
        with patch("youcut.cli.YouTubeUploader", return_value=mock_uploader), \
             patch("youcut.cli.InstagramUploader", return_value=MagicMock()), \
             patch("youcut.cli.TikTokUploader", return_value=MagicMock()), \
             patch("youcut.cli._console.print") as mock_print:
            _publish_clips_with_status([record], ["youtube"], tmp_path / "creds")

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "thumbnail aplicada" in printed


# ---------------------------------------------------------------------------
# Task 7.0 — Auto mode menu
# ---------------------------------------------------------------------------

class TestAutoMode:
    def _cuts_kwargs(self, **overrides):
        base = dict(
            source="https://youtube.com/watch?v=test",
            history=False,
            max_clips=None,
            skip_review=False,
            upload=False,
            platforms_raw="youtube",
            log_level="INFO",
            log_file=None,
        )
        base.update(overrides)
        return base

    def _base_patches(self):
        return [
            patch("youcut.cli.sys.stdin.isatty", return_value=True),
            patch("youcut.cli.sys.stdout.isatty", return_value=True),
            patch("youcut.cli._select_cut_mode", return_value="youtube"),
            patch("youcut.cli.questionary.text", return_value=MagicMock(ask=MagicMock(return_value=""))),
            patch("youcut.cli.normalize_video_url", return_value="https://youtube.com/watch?v=test"),
            patch("youcut.cli.fetch_metadata", return_value=MagicMock(title="Test", duration_seconds=600)),
            patch("youcut.cli.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=True))),
            patch("youcut.cli._parse_platforms", return_value=["youtube"]),
            patch("youcut.cli._check_ffmpeg"),
            patch("youcut.cli._configure_logging"),
        ]

    def test_auto_mode_selected_calls_run_auto_pipeline(self, tmp_path, monkeypatch):
        """When user selects 'auto', _run_auto_pipeline is called (not run_flow_a directly)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        auto_pipeline_calls = []

        with ExitStack() as stack:
            for p in self._base_patches():
                stack.enter_context(p)
            stack.enter_context(patch("youcut.cli._select_execution_mode", return_value="auto"))
            stack.enter_context(patch("youcut.cli._run_auto_pipeline", side_effect=lambda *a, **kw: auto_pipeline_calls.append(a)))
            from youcut.cli import cuts
            cuts(**self._cuts_kwargs())

        assert len(auto_pipeline_calls) == 1

    def test_manual_mode_selected_calls_run_flow_a_with_original_skip_review(self, tmp_path, monkeypatch):
        """When user selects 'manual', run_flow_a uses the original CLI skip_review value."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        run_flow_a_calls = []

        def fake_run_flow_a(source, cfg, *, skip_review, **kw):
            run_flow_a_calls.append({"skip_review": skip_review})
            return None

        with ExitStack() as stack:
            for p in self._base_patches():
                stack.enter_context(p)
            stack.enter_context(patch("youcut.cli._select_execution_mode", return_value="manual"))
            stack.enter_context(patch("youcut.cli.run_flow_a", side_effect=fake_run_flow_a))
            from youcut.cli import cuts
            cuts(**self._cuts_kwargs(skip_review=False))

        assert len(run_flow_a_calls) == 1
        assert run_flow_a_calls[0]["skip_review"] is False

    def test_manual_mode_does_not_call_run_auto_pipeline(self, tmp_path, monkeypatch):
        """When user selects 'manual', _run_auto_pipeline is never called."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        auto_pipeline_calls = []

        with ExitStack() as stack:
            for p in self._base_patches():
                stack.enter_context(p)
            stack.enter_context(patch("youcut.cli._select_execution_mode", return_value="manual"))
            stack.enter_context(patch("youcut.cli.run_flow_a", return_value=None))
            stack.enter_context(patch("youcut.cli._run_auto_pipeline", side_effect=lambda *a, **kw: auto_pipeline_calls.append(a)))
            from youcut.cli import cuts
            cuts(**self._cuts_kwargs())

        assert len(auto_pipeline_calls) == 0


class TestAutoModePipeline:
    """Tests for _run_auto_pipeline() — the automatic pipeline orchestrator."""

    @pytest.fixture
    def auto_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from youcut.config import PipelineConfig
        return PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

    @pytest.fixture
    def fake_session(self, tmp_path):
        import uuid
        from datetime import datetime
        p = tmp_path / "clip.mp4"
        p.touch()
        cache = tmp_path / "transcript.json"
        cache.write_text("{}", encoding="utf-8")
        return SessionData(
            session_id=str(uuid.uuid4()),
            source_url="https://youtube.com/watch?v=test",
            cut_mode="youtube",
            transcription_cache_path=cache,
            clips=[
                ClipRecord(
                    title="YT Clip",
                    start_time=0.0,
                    end_time=300.0,
                    clip_path=p,
                    thumbnail_path=None,
                    approved=True,
                )
            ],
            created_at=datetime.now(),
            output_dir=tmp_path,
        )

    def test_run_flow_a_called_with_skip_review_and_upload(self, auto_config, fake_session):
        """run_flow_a is called with skip_review=True, upload=True, platforms=['youtube']."""
        flow_a_calls = []

        def fake_flow_a(source, cfg, *, skip_review, upload, platforms, **kw):
            flow_a_calls.append(dict(skip_review=skip_review, upload=upload, platforms=platforms))
            return fake_session

        with (
            patch("youcut.cli.run_flow_a", side_effect=fake_flow_a),
            patch("youcut.cli._run_social_from_clips", return_value=[]),
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        assert len(flow_a_calls) == 1
        assert flow_a_calls[0]["skip_review"] is True
        assert flow_a_calls[0]["upload"] is True
        assert flow_a_calls[0]["platforms"] == ["youtube"]

    def test_social_pipeline_called_with_skip_review_and_social_platforms(self, auto_config, fake_session):
        """_run_social_from_clips is called with skip_review=True, upload=True, platforms=['tiktok', 'instagram']."""
        social_calls = []

        def fake_run_social(clip_records, config, *, skip_review, upload, platforms, **kw):
            social_calls.append(
                dict(
                    skip_review=skip_review,
                    upload=upload,
                    platforms=platforms,
                )
            )
            return []

        with (
            patch("youcut.cli.run_flow_a", return_value=fake_session),
            patch("youcut.cli._run_social_from_clips", side_effect=fake_run_social),
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        assert len(social_calls) == 1
        assert social_calls[0]["skip_review"] is True
        assert social_calls[0]["upload"] is True
        assert set(social_calls[0]["platforms"]) == {"tiktok", "instagram"}

    def test_run_flow_b_not_called_when_flow_a_returns_none(self, auto_config):
        """If run_flow_a returns None, _run_social_from_clips is never called."""
        social_calls = []

        with (
            patch("youcut.cli.run_flow_a", return_value=None),
            patch("youcut.cli._run_social_from_clips", side_effect=lambda *a, **kw: social_calls.append(True)),
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        assert len(social_calls) == 0

    def test_show_consolidated_summary_called_even_when_flow_b_empty(self, auto_config, fake_session):
        """_show_consolidated_summary is called even when _run_social_from_clips returns []."""
        summary_calls = []

        with (
            patch("youcut.cli.run_flow_a", return_value=fake_session),
            patch("youcut.cli._run_social_from_clips", return_value=[]),
            patch("youcut.cli._show_consolidated_summary", side_effect=lambda yt, soc: summary_calls.append((yt, soc))),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        assert len(summary_calls) == 1
        yt_records, social_records = summary_calls[0]
        assert social_records == []

    def test_show_consolidated_summary_receives_yt_records_from_session(self, auto_config, fake_session):
        """_show_consolidated_summary receives session.clips as yt_records."""
        summary_calls = []

        with (
            patch("youcut.cli.run_flow_a", return_value=fake_session),
            patch("youcut.cli._run_social_from_clips", return_value=[]),
            patch("youcut.cli._show_consolidated_summary", side_effect=lambda yt, soc: summary_calls.append((yt, soc))),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        yt_records, _ = summary_calls[0]
        assert yt_records == fake_session.clips

    def test_social_pipeline_receives_only_approved_clips(self, auto_config, fake_session):
        """_run_social_from_clips receives only approved clips from the session."""
        rejected = ClipRecord(
            title="Rejected",
            start_time=300.0,
            end_time=400.0,
            clip_path=fake_session.clips[0].clip_path,
            thumbnail_path=None,
            approved=False,
        )
        fake_session.clips.append(rejected)

        social_clips = []

        def fake_run_social(clip_records, config, **kw):
            social_clips.extend(clip_records)
            return []

        with (
            patch("youcut.cli.run_flow_a", return_value=fake_session),
            patch("youcut.cli._run_social_from_clips", side_effect=fake_run_social),
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=test", auto_config)

        assert all(c.approved for c in social_clips)
        assert len(social_clips) == 1
