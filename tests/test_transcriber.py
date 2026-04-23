import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp
from youcut.transcriber import _cache_path, _load_cache, _md5, _save_cache, transcribe


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(whisper_model="tiny")


@pytest.fixture
def sample_video(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-video-content")
    return video


@pytest.fixture
def sample_result(sample_video):
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=2.0,
                text="Olá mundo",
                words=[
                    WordTimestamp(word="Olá", start=0.0, end=1.0),
                    WordTimestamp(word="mundo", start=1.0, end=2.0),
                ],
            )
        ],
        language="pt",
        source_path=sample_video,
    )


def _make_faster_whisper_mock(result: TranscriptionResult):
    mock_word = MagicMock()
    mock_word.word = "Olá"
    mock_word.start = 0.0
    mock_word.end = 1.0

    mock_word2 = MagicMock()
    mock_word2.word = "mundo"
    mock_word2.start = 1.0
    mock_word2.end = 2.0

    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 2.0
    mock_seg.text = "Olá mundo"
    mock_seg.words = [mock_word, mock_word2]

    mock_info = MagicMock()
    mock_info.language = "pt"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)

    mock_cls = MagicMock(return_value=mock_model)
    return mock_cls, mock_model


class TestMd5:
    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello")
        result = _md5(f)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content-a")
        b.write_bytes(b"content-b")
        assert _md5(a) != _md5(b)

    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"same")
        b.write_bytes(b"same")
        assert _md5(a) == _md5(b)


class TestCache:
    def test_save_and_load_cache(self, sample_video, sample_result):
        file_hash = _md5(sample_video)
        cache_file = _cache_path(sample_video)

        _save_cache(cache_file, sample_result, file_hash)

        assert cache_file.exists()
        loaded = _load_cache(cache_file, file_hash)
        assert loaded is not None
        assert len(loaded.segments) == 1
        assert loaded.segments[0].text == "Olá mundo"
        assert len(loaded.segments[0].words) == 2

    def test_load_cache_wrong_hash_returns_none(self, sample_video, sample_result):
        file_hash = _md5(sample_video)
        cache_file = _cache_path(sample_video)

        _save_cache(cache_file, sample_result, file_hash)

        loaded = _load_cache(cache_file, "different-hash")
        assert loaded is None

    def test_load_cache_missing_file_returns_none(self, tmp_path):
        missing = tmp_path / "nonexistent_transcript.json"
        assert _load_cache(missing, "any-hash") is None

    def test_cache_path_naming(self, sample_video):
        cache = _cache_path(sample_video)
        assert cache.name == "sample_transcript.json"
        assert cache.parent == sample_video.parent

    def test_cache_json_contains_md5(self, sample_video, sample_result):
        file_hash = _md5(sample_video)
        cache_file = _cache_path(sample_video)
        _save_cache(cache_file, sample_result, file_hash)

        data = json.loads(cache_file.read_text())
        assert data["md5"] == file_hash
        assert "result" in data


class TestTranscribe:
    def test_calls_faster_whisper_and_maps_result(self, sample_video, config):
        mock_cls, mock_model = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            result = transcribe(sample_video, config)

        mock_cls.assert_called_once_with("tiny", device="auto")
        mock_model.transcribe.assert_called_once()
        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert result.segments[0].text == "Olá mundo"
        assert result.language == "pt"

    def test_cache_created_after_first_call(self, sample_video, config):
        mock_cls, _ = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            transcribe(sample_video, config)

        cache_file = _cache_path(sample_video)
        assert cache_file.exists()

    def test_second_call_uses_cache_not_whisper(self, sample_video, config):
        mock_cls, mock_model = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            transcribe(sample_video, config)
            mock_model.transcribe.reset_mock()
            mock_cls.reset_mock()
            result2 = transcribe(sample_video, config)

        mock_model.transcribe.assert_not_called()
        assert isinstance(result2, TranscriptionResult)

    def test_cache_invalidated_when_file_changes(self, sample_video, config):
        mock_cls, mock_model = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            transcribe(sample_video, config)

            # Modify the file so hash changes
            sample_video.write_bytes(b"modified-video-content")

            mock_model.transcribe.reset_mock()
            mock_cls.reset_mock()
            transcribe(sample_video, config)

        # Whisper called again because hash changed
        mock_model.transcribe.assert_called_once()

    def test_fallback_to_openai_whisper_on_import_error(self, sample_video, config):
        mock_openai_word = {"word": "Oi", "start": 0.0, "end": 0.5}
        mock_openai_seg = {"start": 0.0, "end": 0.5, "text": "Oi", "words": [mock_openai_word]}
        mock_openai_result = {"segments": [mock_openai_seg], "language": "pt"}

        mock_openai_model = MagicMock()
        mock_openai_model.transcribe.return_value = mock_openai_result
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_openai_model

        # Make faster_whisper raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("faster_whisper not available")
            if name == "whisper":
                return mock_whisper
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = transcribe(sample_video, config)

        assert isinstance(result, TranscriptionResult)
        assert result.segments[0].text == "Oi"

    def test_word_timestamps_mapped_correctly(self, sample_video, config):
        mock_cls, _ = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            result = transcribe(sample_video, config)

        words = result.segments[0].words
        assert len(words) == 2
        assert words[0].word == "Olá"
        assert words[0].start == 0.0
        assert words[0].end == 1.0
        assert words[1].word == "mundo"

    def test_raises_runtime_error_when_both_backends_absent(self, sample_video, config):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("faster_whisper", "whisper"):
                raise ImportError(f"{name} not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="Nenhum backend de transcrição disponível"):
                transcribe(sample_video, config)

    def test_corrupted_cache_is_ignored_and_retranscribed(self, sample_video, config):
        cache_file = _cache_path(sample_video)
        cache_file.write_text("{ invalid json {{", encoding="utf-8")

        mock_cls, mock_model = _make_faster_whisper_mock(None)

        with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
            result = transcribe(sample_video, config)

        mock_model.transcribe.assert_called_once()
        assert isinstance(result, TranscriptionResult)
