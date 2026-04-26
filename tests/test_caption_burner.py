"""Unit tests for CaptionBurner."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.caption_burner import CaptionBurner, _format_srt_time
from youcut.models import CaptionBurnResult


# ---------------------------------------------------------------------------
# _format_srt_time helpers
# ---------------------------------------------------------------------------

def test_format_srt_time_zero():
    assert _format_srt_time(0.0) == "00:00:00,000"


def test_format_srt_time_basic():
    assert _format_srt_time(1.5) == "00:00:01,500"


def test_format_srt_time_with_hours():
    assert _format_srt_time(3661.123) == "01:01:01,123"


def test_format_srt_time_negative_clamps_to_zero():
    assert _format_srt_time(-1.0) == "00:00:00,000"


# ---------------------------------------------------------------------------
# _write_word_srt
# ---------------------------------------------------------------------------

class TestWriteWordSrt:
    def test_generates_srt_with_one_word_per_cue(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.touch()

        words = [
            {"word": "Olá", "start": 0.0, "end": 0.35},
            {"word": "mundo", "start": 0.35, "end": 0.70},
        ]

        srt_path = burner._write_word_srt(words, video)

        content = srt_path.read_text(encoding="utf-8")
        assert "1\n00:00:00,000 --> 00:00:00,350\nOlá" in content
        assert "2\n00:00:00,350 --> 00:00:00,700\nmundo" in content

    def test_srt_cue_count_matches_word_count(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.touch()

        words = [{"word": f"word{i}", "start": float(i), "end": float(i) + 0.5} for i in range(5)]
        srt_path = burner._write_word_srt(words, video)

        content = srt_path.read_text(encoding="utf-8")
        # SRT indices 1..5 should all be present
        for i in range(1, 6):
            assert f"\n{i}\n" in content or content.startswith(f"{i}\n")

    def test_srt_written_to_same_dir_as_video(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "subdir" / "clip.mp4"
        video.parent.mkdir()
        video.touch()

        words = [{"word": "Hi", "start": 0.0, "end": 0.5}]
        srt_path = burner._write_word_srt(words, video)

        assert srt_path.parent == video.parent
        assert srt_path.suffix == ".srt"


# ---------------------------------------------------------------------------
# burn() — success path
# ---------------------------------------------------------------------------

class TestBurnSuccess:
    @patch("youcut.caption_burner.subprocess.run")
    def test_burn_returns_captioned_path(self, mock_run, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        mock_words = [{"word": "Hello", "start": 0.0, "end": 0.5}]

        with patch.object(burner, "_transcribe_words", return_value=mock_words):
            mock_run.return_value = MagicMock(returncode=0)
            # Create the output file to simulate FFmpeg success
            out = tmp_path / "clip_captioned.mp4"
            out.write_bytes(b"\x00" * 10)

            result = burner.burn(video)

        assert isinstance(result, CaptionBurnResult)
        assert result.output_path.name == "clip_captioned.mp4"
        assert result.output_path != video
        assert result.captions_applied is True
        assert result.warning is None

    @patch("youcut.caption_burner.subprocess.run")
    def test_burn_output_has_captioned_suffix(self, mock_run, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "my_clip.mp4"
        video.write_bytes(b"\x00" * 10)

        mock_words = [{"word": "Word", "start": 0.1, "end": 0.4}]

        with patch.object(burner, "_transcribe_words", return_value=mock_words):
            mock_run.return_value = MagicMock(returncode=0)
            (tmp_path / "my_clip_captioned.mp4").write_bytes(b"\x00")

            result = burner.burn(video)

        assert "_captioned" in result.output_path.stem


# ---------------------------------------------------------------------------
# burn() — fallback on Whisper failure
# ---------------------------------------------------------------------------

class TestBurnWhisperFallback:
    def test_returns_original_when_whisper_fails(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        with patch.object(burner, "_transcribe_words", side_effect=RuntimeError("whisper unavailable")):
            result = burner.burn(video)

        assert result.output_path == video
        assert result.captions_applied is False
        assert "Transcrição falhou" in result.warning

    def test_no_exception_raised_when_whisper_fails(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        with patch.object(burner, "_transcribe_words", side_effect=Exception("model error")):
            # Should not raise
            result = burner.burn(video)

        assert result.output_path == video

    def test_warning_logged_when_whisper_fails(self, tmp_path, caplog):
        import logging
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        with patch.object(burner, "_transcribe_words", side_effect=RuntimeError("no model")):
            with caplog.at_level(logging.WARNING, logger="youcut.caption_burner"):
                burner.burn(video)

        assert any("transcrição falhou" in record.message for record in caplog.records)

    def test_returns_structured_fallback_when_srt_generation_fails(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        with (
            patch.object(burner, "_transcribe_words", return_value=[{"word": "oi", "start": 0.0, "end": 0.5}]),
            patch.object(burner, "_write_word_srt", side_effect=RuntimeError("disk full")),
        ):
            result = burner.burn(video)

        assert result.output_path == video
        assert result.captions_applied is False
        assert "Geração de SRT falhou" in result.warning


# ---------------------------------------------------------------------------
# burn() — fallback on FFmpeg failure
# ---------------------------------------------------------------------------

class TestBurnFFmpegFallback:
    def test_returns_original_when_ffmpeg_fails(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        mock_words = [{"word": "Hi", "start": 0.0, "end": 0.5}]

        with patch.object(burner, "_transcribe_words", return_value=mock_words):
            with patch.object(burner, "_ffmpeg_burn", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                result = burner.burn(video)

        assert result.output_path == video
        assert result.captions_applied is False
        assert "FFmpeg falhou" in result.warning

    def test_no_exception_raised_when_ffmpeg_fails(self, tmp_path):
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        mock_words = [{"word": "Test", "start": 0.0, "end": 0.3}]

        with patch.object(burner, "_transcribe_words", return_value=mock_words):
            with patch.object(burner, "_ffmpeg_burn", side_effect=Exception("ffmpeg not found")):
                result = burner.burn(video)

        assert result.output_path == video

    def test_warning_logged_when_ffmpeg_fails(self, tmp_path, caplog):
        import logging
        burner = CaptionBurner()
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 10)

        mock_words = [{"word": "Test", "start": 0.0, "end": 0.3}]

        with patch.object(burner, "_transcribe_words", return_value=mock_words):
            with patch.object(burner, "_ffmpeg_burn", side_effect=Exception("ffmpeg error")):
                with caplog.at_level(logging.WARNING, logger="youcut.caption_burner"):
                    burner.burn(video)

        assert any("FFmpeg falhou" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------

class TestModuleImportability:
    def test_caption_burner_importable(self):
        from youcut.caption_burner import CaptionBurner  # noqa: F401

    def test_instantiable_without_arguments(self):
        burner = CaptionBurner()
        assert burner is not None

    def test_burn_method_exists(self):
        burner = CaptionBurner()
        assert callable(burner.burn)
