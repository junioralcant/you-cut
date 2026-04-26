"""Testes unitários para youcut/diarizer.py (Task 2.0)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.models import SpeakerSegment


@pytest.fixture
def config_no_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    from youcut.config import PipelineConfig
    return PipelineConfig(huggingface_token=None)


@pytest.fixture
def config_with_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(huggingface_token="hf-test-token")


@pytest.fixture
def fake_clip(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake")
    return p


def _mock_ffprobe(duration: float):
    result = MagicMock()
    result.stdout = f"{duration}\n"
    return result


class TestDiarizeNoToken:
    def test_returns_single_speaker_segment_when_no_token(self, config_no_token, fake_clip):
        with patch("youcut.diarizer.subprocess.run", return_value=_mock_ffprobe(10.0)):
            from youcut.diarizer import diarize
            segments = diarize(fake_clip, config_no_token)

        assert len(segments) == 1
        assert segments[0].speaker_id == "SPEAKER_00"
        assert segments[0].start == 0.0
        assert segments[0].end == 10.0

    def test_fallback_uses_clip_duration(self, config_no_token, fake_clip):
        with patch("youcut.diarizer.subprocess.run", return_value=_mock_ffprobe(42.5)):
            from youcut.diarizer import diarize
            segments = diarize(fake_clip, config_no_token)

        assert segments[0].end == pytest.approx(42.5)

    def test_does_not_raise_when_no_token(self, config_no_token, fake_clip):
        with patch("youcut.diarizer.subprocess.run", return_value=_mock_ffprobe(5.0)):
            from youcut.diarizer import diarize
            segments = diarize(fake_clip, config_no_token)

        assert isinstance(segments, list)


class TestDiarizeWithMockedPyannote:
    def _make_diarization_mock(self, speaker_turns: list[tuple[float, float, str]]):
        """Build a mock pyannote diarization object with given (start, end, speaker) turns."""
        diarization = MagicMock()
        tracks = []
        for start, end, speaker in speaker_turns:
            turn = MagicMock()
            turn.start = start
            turn.end = end
            tracks.append((turn, None, speaker))
        diarization.itertracks.return_value = iter(tracks)
        return diarization

    def test_three_segments_two_speakers(self, config_with_token, fake_clip):
        turns = [(0.0, 2.0, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01"), (5.5, 8.0, "SPEAKER_00")]
        diarization_mock = self._make_diarization_mock(turns)
        pipeline_instance = MagicMock(return_value=diarization_mock)
        pipeline_class = MagicMock(from_pretrained=MagicMock(return_value=pipeline_instance))
        pyannote_audio_mock = MagicMock(Pipeline=pipeline_class)

        import sys
        import importlib
        original = sys.modules.get("pyannote.audio")
        sys.modules["pyannote.audio"] = pyannote_audio_mock
        try:
            import youcut.diarizer
            importlib.reload(youcut.diarizer)
            segments = youcut.diarizer.diarize(fake_clip, config_with_token)
        finally:
            if original is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = original
            importlib.reload(youcut.diarizer)

        assert len(segments) == 3
        assert segments[0] == SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=2.0)
        assert segments[1] == SpeakerSegment(speaker_id="SPEAKER_01", start=2.5, end=5.0)
        assert segments[2] == SpeakerSegment(speaker_id="SPEAKER_00", start=5.5, end=8.0)

    def test_two_speakers_via_direct_pyannote_mock(self, config_with_token, fake_clip):
        turns = [(0.0, 3.0, "SPEAKER_00"), (3.5, 7.0, "SPEAKER_01")]
        diarization_mock = self._make_diarization_mock(turns)
        pipeline_instance = MagicMock(return_value=diarization_mock)
        pipeline_class = MagicMock(from_pretrained=MagicMock(return_value=pipeline_instance))
        pyannote_audio_mock = MagicMock(Pipeline=pipeline_class)

        import sys
        original = sys.modules.get("pyannote.audio")
        sys.modules["pyannote.audio"] = pyannote_audio_mock
        try:
            import importlib
            import youcut.diarizer
            importlib.reload(youcut.diarizer)
            segments = youcut.diarizer.diarize(fake_clip, config_with_token)
        finally:
            if original is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = original
            import youcut.diarizer
            importlib.reload(youcut.diarizer)

        assert len(segments) == 2
        speaker_ids = {s.speaker_id for s in segments}
        assert "SPEAKER_00" in speaker_ids
        assert "SPEAKER_01" in speaker_ids

    def test_all_segments_have_start_less_than_end(self, config_with_token, fake_clip):
        turns = [(0.0, 2.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")]
        diarization_mock = self._make_diarization_mock(turns)
        pipeline_instance = MagicMock(return_value=diarization_mock)
        pipeline_class = MagicMock(from_pretrained=MagicMock(return_value=pipeline_instance))
        pyannote_audio_mock = MagicMock(Pipeline=pipeline_class)

        import sys
        import importlib
        original = sys.modules.get("pyannote.audio")
        sys.modules["pyannote.audio"] = pyannote_audio_mock
        try:
            import youcut.diarizer
            importlib.reload(youcut.diarizer)
            segments = youcut.diarizer.diarize(fake_clip, config_with_token)
        finally:
            if original is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = original
            importlib.reload(youcut.diarizer)

        for seg in segments:
            assert seg.start < seg.end, f"Segment {seg} violates start < end"


class TestDiarizeExceptionFallback:
    def test_pyannote_runtime_error_returns_fallback(self, config_with_token, fake_clip):
        pipeline_instance = MagicMock(side_effect=RuntimeError("pyannote failed"))
        pipeline_class = MagicMock(from_pretrained=MagicMock(return_value=pipeline_instance))
        pyannote_audio_mock = MagicMock(Pipeline=pipeline_class)

        import sys
        import importlib
        original = sys.modules.get("pyannote.audio")
        sys.modules["pyannote.audio"] = pyannote_audio_mock
        try:
            import youcut.diarizer
            importlib.reload(youcut.diarizer)
            with patch("youcut.diarizer.subprocess.run", return_value=_mock_ffprobe(15.0)):
                segments = youcut.diarizer.diarize(fake_clip, config_with_token)
        finally:
            if original is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = original
            importlib.reload(youcut.diarizer)

        assert len(segments) == 1
        assert segments[0].speaker_id == "SPEAKER_00"
        assert segments[0].start == 0.0

    def test_pyannote_exception_does_not_propagate(self, config_with_token, fake_clip):
        pipeline_instance = MagicMock(side_effect=Exception("unexpected error"))
        pipeline_class = MagicMock(from_pretrained=MagicMock(return_value=pipeline_instance))
        pyannote_audio_mock = MagicMock(Pipeline=pipeline_class)

        import sys
        import importlib
        original = sys.modules.get("pyannote.audio")
        sys.modules["pyannote.audio"] = pyannote_audio_mock
        try:
            import youcut.diarizer
            importlib.reload(youcut.diarizer)
            with patch("youcut.diarizer.subprocess.run", return_value=_mock_ffprobe(5.0)):
                segments = youcut.diarizer.diarize(fake_clip, config_with_token)
        finally:
            if original is None:
                sys.modules.pop("pyannote.audio", None)
            else:
                sys.modules["pyannote.audio"] = original
            importlib.reload(youcut.diarizer)

        assert isinstance(segments, list)
