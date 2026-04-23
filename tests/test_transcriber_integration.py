"""
Integration tests for transcriber.py — require faster-whisper installed.
Uses the real sample.mp4 fixture to validate end-to-end transcription and cache.
"""
import json
from pathlib import Path

import pytest

from youcut.transcriber import _cache_path, _md5, transcribe

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(whisper_model="tiny")


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample fixture not found")
class TestTranscriberIntegration:
    def test_transcription_returns_result_with_segments(self, config, tmp_path):
        import shutil
        video = tmp_path / "sample.mp4"
        shutil.copy(FIXTURE, video)

        result = transcribe(video, config)

        assert result is not None
        assert result.language is not None
        assert isinstance(result.segments, list)

    def test_cache_json_is_persisted_to_disk(self, config, tmp_path):
        import shutil
        video = tmp_path / "sample.mp4"
        shutil.copy(FIXTURE, video)

        transcribe(video, config)

        cache_file = _cache_path(video)
        assert cache_file.exists(), "Cache JSON must be written after first transcription"

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "md5" in data
        assert "result" in data
        assert data["md5"] == _md5(video)

    def test_cache_is_reloaded_on_second_call(self, config, tmp_path):
        import shutil
        video = tmp_path / "sample.mp4"
        shutil.copy(FIXTURE, video)

        result1 = transcribe(video, config)
        result2 = transcribe(video, config)

        assert result1.segments == result2.segments
        assert result1.language == result2.language

    def test_cached_result_has_valid_timestamps(self, config, tmp_path):
        import shutil
        video = tmp_path / "sample.mp4"
        shutil.copy(FIXTURE, video)

        transcribe(video, config)
        cache_file = _cache_path(video)
        data = json.loads(cache_file.read_text(encoding="utf-8"))

        for seg in data["result"]["segments"]:
            assert seg["start"] >= 0
            assert seg["end"] > seg["start"]
            for w in seg.get("words", []):
                assert w["start"] >= 0
                assert w["end"] >= w["start"]
