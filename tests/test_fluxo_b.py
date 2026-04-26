"""Integration tests for Flow B (short videos from existing long clips)."""

import json
import threading
import time
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
def social_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")


@pytest.fixture
def transcription_cache_path(tmp_path):
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=10.0,
                end=50.0,
                text="Intro segment outside long clip range.",
                words=[WordTimestamp(word="Intro", start=10.0, end=10.5)],
            ),
            TranscriptionSegment(
                start=300.0,
                end=500.0,
                text="This is the key discussion within the long clip.",
                words=[WordTimestamp(word="This", start=300.0, end=300.5)],
            ),
            TranscriptionSegment(
                start=500.0,
                end=700.0,
                text="Second part of the long clip content.",
                words=[WordTimestamp(word="Second", start=500.0, end=500.5)],
            ),
        ],
        language="pt",
        source_path=Path("original_video.mp4"),
    )

    cache_data = {
        "md5": "test-md5-abc123",
        "result": json.loads(transcription.model_dump_json()),
    }

    cache_path = tmp_path / "original_video_transcript.json"
    cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
    return cache_path


@pytest.fixture
def long_clip_path(tmp_path):
    p = tmp_path / "clips" / "clip_01.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


@pytest.fixture
def youtube_session(tmp_path, transcription_cache_path, long_clip_path):
    """SessionData fixture representing a completed Flow A session."""
    return SessionData(
        session_id=str(uuid.uuid4()),
        source_url="https://youtube.com/watch?v=test",
        cut_mode="youtube",
        transcription_cache_path=transcription_cache_path,
        clips=[
            ClipRecord(
                title="Tech Discussion Segment",
                start_time=300.0,
                end_time=700.0,
                clip_path=long_clip_path,
                thumbnail_path=None,
                approved=True,
            )
        ],
        created_at=datetime.now(),
        output_dir=tmp_path / "clips",
    )


@pytest.fixture
def short_clip_path(tmp_path):
    p = tmp_path / "output" / "flow_b" / "clip_01" / "clip_01.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


@pytest.fixture
def mock_short_clip():
    return ViralClip(
        title="Quick Insight",
        reason="High engagement moment",
        viral_score=9.0,
        start_time=350.0,  # relative to original video (within 300–700 range)
        end_time=410.0,    # 60s duration — within social 15–180s range
        description="A key insight from the discussion",
        hashtags=["#viral", "#insight"],
        thumbnail_idea="Speaker making a key point",
        cut_mode="social",
    )


class TestFlowBDoesNotCallTranscribe:
    def test_transcribe_not_called(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """Core requirement: Flow B must never invoke transcribe()."""
        with (
            patch("youcut.cli.transcribe") as mock_tr,
            patch("youcut.cli.analyze", return_value=[mock_short_clip]),
            patch("youcut.cli.cut_clip", return_value=short_clip_path),
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        mock_tr.assert_not_called()

    def test_transcription_loaded_from_cache(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """Transcription is loaded from the cache file, not from a fresh transcription."""
        analyze_calls = []

        def capture_analyze(transcription, config):
            analyze_calls.append(transcription)
            return [mock_short_clip]

        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", side_effect=capture_analyze),
            patch("youcut.cli.cut_clip", return_value=short_clip_path),
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        # analyze() must be called with a TranscriptionResult (loaded from cache)
        assert len(analyze_calls) >= 1
        cached_tr = analyze_calls[0]
        assert isinstance(cached_tr, TranscriptionResult)
        # Segments should be filtered to the clip's time range (300–700s)
        for seg in cached_tr.segments:
            assert seg.start >= 300.0
            assert seg.end <= 700.0


class TestFlowBGeneratesSocialClips:
    def test_clips_produced_from_selected_long_clips(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """Flow B produces ClipRecords from the selected long clips."""
        cut_calls = []

        def capture_cut(src, clip, idx, cfg):
            cut_calls.append({"src": src, "clip": clip, "cfg": cfg})
            return short_clip_path

        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_short_clip]),
            patch("youcut.cli.cut_clip", side_effect=capture_cut),
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        # cut_clip must be called with the long clip as source
        assert len(cut_calls) == 1
        assert cut_calls[0]["src"] == youtube_session.clips[0].clip_path
        # Config must be social mode
        assert cut_calls[0]["cfg"].cut_mode == "social"

    def test_timestamps_adjusted_relative_to_clip(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """Clip timestamps passed to cut_clip are relative to the long clip, not the original video."""
        captured_clips = []

        def capture_cut(src, clip, idx, cfg):
            captured_clips.append(clip)
            return short_clip_path

        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_short_clip]),
            patch("youcut.cli.cut_clip", side_effect=capture_cut),
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        assert len(captured_clips) == 1
        adj_clip = captured_clips[0]
        # Original: start=350, end=410; clip starts at 300 → adjusted: start=50, end=110
        assert abs(adj_clip.start_time - 50.0) < 0.01
        assert abs(adj_clip.end_time - 110.0) < 0.01

    def test_no_clips_when_cache_missing(self, tmp_path, social_config, youtube_session):
        """Flow B exits gracefully when the transcription cache is missing."""
        # Point to a non-existent cache
        missing_session = youtube_session.model_copy(
            update={"transcription_cache_path": tmp_path / "nonexistent_transcript.json"}
        )

        with patch("youcut.cli.transcribe") as mock_tr:
            from youcut.cli import run_flow_b
            run_flow_b(
                session=missing_session,
                selected_clips=missing_session.clips,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        # transcribe should still not be called even on error
        mock_tr.assert_not_called()


class TestFlowBTimerTimeout:
    def test_timeout_processes_all_approved_clips(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """When threading.Timer fires before user interaction, all approved clips are processed."""
        import time as _time_mod

        # Capture the original sleep BEFORE any patching so we can call it directly
        _real_sleep = _time_mod.sleep

        approved_clips = [c for c in youtube_session.clips if c.approved]
        run_flow_b_calls: list[list[ClipRecord]] = []

        class ImmediateTimer:
            """Fires the callback after 50 ms to simulate a timeout."""
            def __init__(self, interval, callback):
                self._callback = callback
                self.daemon = False

            def start(self):
                t = threading.Thread(target=self._fire_soon, daemon=True)
                t.start()

            def _fire_soon(self):
                _real_sleep(0.05)   # original sleep — not affected by patch
                self._callback()

            def cancel(self):
                pass

        def fake_run_flow_b(session, selected_clips, config, **kwargs):
            run_flow_b_calls.append(list(selected_clips))

        # Block the selection thread long enough for the timer to fire (50 ms)
        # but let it time-out on its own so the daemon thread exits cleanly.
        _selection_unblock = threading.Event()

        def fake_confirm(*args, **kwargs):
            m = MagicMock()
            m.ask.side_effect = lambda: _selection_unblock.wait(timeout=1.0)
            return m

        # Patch time.sleep in cli so the while-loop and cancellation window are fast.
        # Use _real_sleep inside the lambda to avoid infinite recursion.
        def _capped_sleep(s):
            _real_sleep(min(s, 0.05))

        with (
            patch("threading.Timer", ImmediateTimer),
            patch("youcut.cli.run_flow_b", side_effect=fake_run_flow_b),
            patch("youcut.cli.time.sleep", side_effect=_capped_sleep),
            patch("youcut.cli.questionary.confirm", side_effect=fake_confirm),
        ):
            from youcut.cli import offer_flow_b
            offer_flow_b(
                session=youtube_session,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        # After offer_flow_b returns, unblock the daemon selection thread so it exits cleanly
        _selection_unblock.set()

        # run_flow_b must have been called with ALL approved clips
        assert len(run_flow_b_calls) == 1
        assert run_flow_b_calls[0] == approved_clips

    def test_user_selection_cancels_timer(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """When user responds before timeout, timer is cancelled and only selected clips are processed."""
        run_flow_b_calls: list[list[ClipRecord]] = []
        cancelled_timers: list[bool] = []

        class TrackingTimer:
            def __init__(self, interval, callback):
                self.interval = interval
                self._callback = callback
                self.daemon = False
                self._cancelled = False

            def start(self):
                pass

            def cancel(self):
                self._cancelled = True
                cancelled_timers.append(True)

        def fake_run_flow_b(session, selected_clips, config, **kwargs):
            run_flow_b_calls.append(list(selected_clips))

        approved_clips = [c for c in youtube_session.clips if c.approved]

        def fake_confirm(*args, **kwargs):
            m = MagicMock()
            m.ask.return_value = True
            return m

        def fake_checkbox(*args, **kwargs):
            m = MagicMock()
            # User selects the first clip (label format: "1. <title> (Xs)")
            m.ask.return_value = [f"1. {approved_clips[0].title} ({approved_clips[0].end_time - approved_clips[0].start_time:.0f}s)"]
            return m

        with (
            patch("youcut.cli.threading.Timer", TrackingTimer),
            patch("youcut.cli.run_flow_b", side_effect=fake_run_flow_b),
            patch("youcut.cli.time.sleep"),
            patch("youcut.cli.questionary.confirm", side_effect=fake_confirm),
            patch("youcut.cli.questionary.checkbox", side_effect=fake_checkbox),
        ):
            from youcut.cli import offer_flow_b
            offer_flow_b(
                session=youtube_session,
                config=social_config,
                skip_review=True,
                upload=False,
            )

        # Timer was cancelled (user responded before timeout)
        assert len(cancelled_timers) >= 1
        # run_flow_b was called with the user's selection
        assert len(run_flow_b_calls) == 1
        assert len(run_flow_b_calls[0]) == 1
        assert run_flow_b_calls[0][0].title == approved_clips[0].title


class TestFlowBHistoryIntegration:
    def test_list_sessions_returns_existing_sessions(self, tmp_path, youtube_session, monkeypatch):
        """list_sessions() returns sessions stored in ~/.youcut/sessions/."""
        sessions_dir = tmp_path / ".youcut" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_file = sessions_dir / f"{youtube_session.session_id}.json"
        session_file.write_text(youtube_session.model_dump_json(), encoding="utf-8")

        with patch("youcut.session_store._SESSIONS_DIR", sessions_dir):
            from youcut.session_store import list_sessions
            sessions = list_sessions()

        assert len(sessions) == 1
        assert sessions[0].session_id == youtube_session.session_id

    def test_run_flow_b_processes_selected_clips_from_history(
        self, social_config, youtube_session, short_clip_path, mock_short_clip
    ):
        """Selecting clips from history and running Flow B generates short clips."""
        cut_calls = []

        with (
            patch("youcut.cli.transcribe"),
            patch("youcut.cli.analyze", return_value=[mock_short_clip]),
            patch("youcut.cli.cut_clip", side_effect=lambda *a, **kw: (cut_calls.append(a), short_clip_path)[1]),
        ):
            from youcut.cli import run_flow_b
            run_flow_b(
                session=youtube_session,
                selected_clips=youtube_session.clips,  # As if selected from history
                config=social_config,
                skip_review=True,
                upload=False,
            )

        assert len(cut_calls) >= 1
