"""Testes de integração do face tracking no pipeline CLI (Task 5.0)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from youcut.models import ClipRecord, SessionData, TranscriptionResult, ViralClip


# ---------------------------------------------------------------------------
# Fixtures comuns
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_clip_path(tmp_path):
    p = tmp_path / "clip_00.mp4"
    p.write_bytes(b"fake-video-data")
    return p


@pytest.fixture
def tracked_clip_path(tmp_path):
    p = tmp_path / "clip_00_tracked.mp4"
    p.write_bytes(b"tracked-video-data")
    return p


@pytest.fixture
def social_config_face_tracking(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="social", face_tracking=True)


@pytest.fixture
def social_config_no_face_tracking(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="social", face_tracking=False)


@pytest.fixture
def mock_viral_clip():
    return ViralClip(
        title="Test Clip",
        reason="test",
        viral_score=8.0,
        start_time=0.0,
        end_time=30.0,
        description="Test",
        hashtags=["#test"],
        thumbnail_idea="test",
        cut_mode="social",
    )


@pytest.fixture
def youtube_session(tmp_path, fake_clip_path):
    transcript_path = tmp_path / "transcript.json"
    transcript_data = {
        "result": {
            "segments": [{"start": 0.0, "end": 30.0, "text": "hello", "words": []}],
            "language": "pt",
            "source_path": str(fake_clip_path),
        }
    }
    transcript_path.write_text(json.dumps(transcript_data), encoding="utf-8")

    clip_record = ClipRecord(
        title="Long clip",
        start_time=0.0,
        end_time=300.0,
        clip_path=fake_clip_path,
        thumbnail_path=None,
        approved=True,
    )
    return SessionData(
        session_id="test-session",
        source_url="https://youtube.com/watch?v=test",
        cut_mode="youtube",
        transcription_cache_path=transcript_path,
        clips=[clip_record],
        created_at=__import__("datetime").datetime.now(),
        output_dir=tmp_path / "output",
    )


# ---------------------------------------------------------------------------
# Fluxo B — face tracking ativo
# ---------------------------------------------------------------------------

class TestFlowBFaceTracking:
    def test_apply_face_tracking_called_once_per_clip_when_enabled(
        self, social_config_face_tracking, youtube_session, fake_clip_path, tracked_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking", return_value=tracked_clip_path) as mock_ft,
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config_face_tracking,
                skip_review=True,
                upload=False,
            )

        mock_ft.assert_called_once()
        call_path, call_cfg = mock_ft.call_args[0]
        assert call_path == fake_clip_path
        assert call_cfg.face_tracking is True
        assert call_cfg.cut_mode == "social"

    def test_apply_face_tracking_not_called_when_disabled(
        self, social_config_no_face_tracking, youtube_session, fake_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking") as mock_ft,
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config_no_face_tracking,
                skip_review=True,
                upload=False,
            )

        mock_ft.assert_not_called()

    def test_face_tracking_exception_does_not_break_pipeline(
        self, social_config_face_tracking, youtube_session, fake_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking", side_effect=RuntimeError("face tracking failed")),
        ):
            from youcut.cli import run_flow_b
            result = run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config_face_tracking,
                skip_review=True,
                upload=False,
            )

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_tracked_path_used_in_records_when_face_tracking_enabled(
        self, social_config_face_tracking, youtube_session, fake_clip_path, tracked_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking", return_value=tracked_clip_path),
        ):
            from youcut.cli import run_flow_b
            records = run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config_face_tracking,
                skip_review=True,
                upload=False,
            )

        assert any(r.clip_path == tracked_clip_path for r in records)


# ---------------------------------------------------------------------------
# Fluxo C — face tracking ativo
# ---------------------------------------------------------------------------

class TestFlowCFaceTracking:
    def test_apply_face_tracking_called_once_per_clip_when_enabled(
        self, social_config_face_tracking, fake_clip_path, tracked_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.download_video", return_value=fake_clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[], language="pt", source_path=fake_clip_path)),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking", return_value=tracked_clip_path) as mock_ft,
        ):
            from youcut.cli import run_flow_c
            run_flow_c(
                "https://youtube.com/watch?v=test",
                social_config_face_tracking,
                skip_review=True,
                upload=False,
            )

        mock_ft.assert_called_once_with(fake_clip_path, social_config_face_tracking)

    def test_apply_face_tracking_not_called_when_disabled(
        self, social_config_no_face_tracking, fake_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.download_video", return_value=fake_clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[], language="pt", source_path=fake_clip_path)),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking") as mock_ft,
        ):
            from youcut.cli import run_flow_c
            run_flow_c(
                "https://youtube.com/watch?v=test",
                social_config_no_face_tracking,
                skip_review=True,
                upload=False,
            )

        mock_ft.assert_not_called()

    def test_face_tracking_exception_does_not_abort_flow_c(
        self, social_config_face_tracking, fake_clip_path, mock_viral_clip
    ):
        with (
            patch("youcut.cli.download_video", return_value=fake_clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[], language="pt", source_path=fake_clip_path)),
            patch("youcut.cli.analyze", return_value=[mock_viral_clip]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.apply_face_tracking", side_effect=RuntimeError("ft error")),
            patch("youcut.cli._show_records_table"),
        ):
            from youcut.cli import run_flow_c
            # Should NOT raise
            run_flow_c(
                "https://youtube.com/watch?v=test",
                social_config_face_tracking,
                skip_review=True,
                upload=False,
            )
