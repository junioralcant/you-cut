import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.captioner import (
    _escape_ass,
    _filter_words,
    _format_ass_time,
    _generate_phrase_events,
    _generate_word_events,
    add_captions,
)
from youcut.config import PipelineConfig
from youcut.models import TranscriptionResult, TranscriptionSegment, ViralClip, WordTimestamp


@pytest.fixture
def config_word(tmp_path):
    return PipelineConfig(
        anthropic_api_key="test-key",
        subtitle_style="word",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def config_phrase(tmp_path):
    return PipelineConfig(
        anthropic_api_key="test-key",
        subtitle_style="phrase",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def clip():
    return ViralClip(
        title="Test",
        reason="Test",
        viral_score=8.0,
        start_time=10.0,
        end_time=20.0,
        description="Test",
        hashtags=[],
        thumbnail_idea="Test",
        thumbnail_text="MOMENTO IMPACTANTE",
    )


@pytest.fixture
def transcription():
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=10.0,
                end=14.0,
                text="Olá mundo",
                words=[
                    WordTimestamp(word="Olá", start=10.0, end=10.5),
                    WordTimestamp(word="mundo", start=10.6, end=11.0),
                ],
            ),
            TranscriptionSegment(
                start=15.0,
                end=19.0,
                text="Como vai você",
                words=[
                    WordTimestamp(word="Como", start=15.0, end=15.4),
                    WordTimestamp(word="vai", start=15.5, end=15.8),
                    WordTimestamp(word="você", start=15.9, end=16.3),
                ],
            ),
        ],
        language="pt",
        source_path=Path("video.mp4"),
    )


class TestFormatAssTime:
    def test_zero(self):
        assert _format_ass_time(0.0) == "0:00:00.00"

    def test_one_second_half(self):
        assert _format_ass_time(1.5) == "0:00:01.50"

    def test_one_minute(self):
        assert _format_ass_time(60.0) == "0:01:00.00"

    def test_one_hour(self):
        assert _format_ass_time(3600.0) == "1:00:00.00"

    def test_complex(self):
        assert _format_ass_time(3723.25) == "1:02:03.25"

    def test_negative_clamped_to_zero(self):
        assert _format_ass_time(-1.0) == "0:00:00.00"

    def test_centiseconds_rounding(self):
        assert _format_ass_time(1.999) == "0:00:02.00"


class TestEscapeAss:
    def test_opening_brace(self):
        assert _escape_ass("hello {world}") == "hello \\{world\\}"

    def test_closing_brace(self):
        assert _escape_ass("a}b") == "a\\}b"

    def test_backslash(self):
        assert _escape_ass("a\\b") == "a\\\\b"

    def test_no_special_chars(self):
        assert _escape_ass("hello world") == "hello world"

    def test_comma_unchanged(self):
        assert _escape_ass("a,b") == "a,b"

    def test_multiple_special_chars(self):
        result = _escape_ass("{bold}")
        assert result == "\\{bold\\}"


class TestFilterWords:
    def test_returns_words_in_interval(self, transcription):
        words = _filter_words(transcription, 10.0, 20.0)
        assert len(words) == 5

    def test_excludes_words_before_interval(self, transcription):
        words = _filter_words(transcription, 15.0, 20.0)
        assert all(w.start >= 15.0 for w in words)
        assert len(words) == 3

    def test_excludes_words_at_or_after_clip_end(self, transcription):
        words = _filter_words(transcription, 10.0, 15.5)
        word_texts = [w.word for w in words]
        assert "vai" not in word_texts
        assert "você" not in word_texts

    def test_empty_when_no_overlap(self, transcription):
        words = _filter_words(transcription, 50.0, 60.0)
        assert words == []

    def test_preserves_word_timestamps(self, transcription):
        words = _filter_words(transcription, 10.0, 20.0)
        assert words[0].start == 10.0
        assert words[0].end == 10.5
        assert words[0].word == "Olá"


class TestGenerateWordEvents:
    def test_event_count_matches_word_count(self, transcription):
        words = _filter_words(transcription, 10.0, 20.0)
        events = _generate_word_events(words, offset=10.0)
        lines = [l for l in events.strip().split("\n") if l.startswith("Dialogue")]
        assert len(lines) == 5

    def test_timestamps_are_offset_adjusted(self, transcription):
        words = _filter_words(transcription, 10.0, 20.0)
        events = _generate_word_events(words, offset=10.0)
        # First word: start=10.0 - 10.0 = 0.0 → "0:00:00.00"
        assert "0:00:00.00" in events
        # Second word: start=10.6 - 10.0 = 0.6 → "0:00:00.60"
        assert "0:00:00.60" in events

    def test_each_dialogue_contains_word_text(self, transcription):
        words = _filter_words(transcription, 10.0, 20.0)
        events = _generate_word_events(words, offset=10.0)
        assert "Olá" in events
        assert "mundo" in events
        assert "Como" in events

    def test_dialogue_format(self, transcription):
        words = [WordTimestamp(word="Teste", start=10.0, end=10.5)]
        events = _generate_word_events(words, offset=10.0)
        assert events == "Dialogue: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,Teste"

    def test_special_chars_escaped_in_word(self):
        words = [WordTimestamp(word="{bold}", start=1.0, end=1.5)]
        events = _generate_word_events(words, offset=0.0)
        assert "\\{bold\\}" in events


class TestGeneratePhraseEvents:
    def test_event_count_matches_overlapping_segments(self, transcription):
        events = _generate_phrase_events(transcription, 10.0, 20.0, offset=10.0)
        lines = [l for l in events.strip().split("\n") if l.startswith("Dialogue")]
        assert len(lines) == 2

    def test_segment_text_present(self, transcription):
        events = _generate_phrase_events(transcription, 10.0, 20.0, offset=10.0)
        assert "Olá mundo" in events
        assert "Como vai você" in events

    def test_timestamps_offset_adjusted(self, transcription):
        events = _generate_phrase_events(transcription, 10.0, 20.0, offset=10.0)
        # segment 1: start=10.0 - 10.0 = 0.0
        assert "0:00:00.00" in events
        # segment 2: start=15.0 - 10.0 = 5.0
        assert "0:00:05.00" in events

    def test_excludes_non_overlapping_segments(self, transcription):
        events = _generate_phrase_events(transcription, 50.0, 60.0, offset=50.0)
        assert events == ""

    def test_segment_partially_before_clip_start_clamped(self):
        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=8.0,
                    end=12.0,
                    text="Partial segment",
                    words=[],
                )
            ],
            language="pt",
            source_path=Path("v.mp4"),
        )
        events = _generate_phrase_events(transcription, 10.0, 20.0, offset=10.0)
        # clamped start: max(8.0, 10.0) - 10.0 = 0.0
        assert "0:00:00.00" in events
        assert "Partial segment" in events

    def test_special_chars_escaped_in_phrase(self):
        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=10.0,
                    end=15.0,
                    text="Curly {braces}",
                    words=[],
                )
            ],
            language="pt",
            source_path=Path("v.mp4"),
        )
        events = _generate_phrase_events(transcription, 10.0, 20.0, offset=10.0)
        assert "\\{braces\\}" in events


class TestAddCaptions:
    @staticmethod
    def _extract_ass_path(cmd: list[str]) -> Path:
        vf_value = cmd[cmd.index("-vf") + 1]
        return Path(vf_value.removeprefix("ass=filename=").removeprefix("ass="))

    def _run_add_captions(self, clip_path, transcription, clip, config):
        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = add_captions(clip_path, transcription, clip, config)
            return result, mock_run.call_args[0][0]

    def test_returns_clip_path(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        result, _ = self._run_add_captions(clip_path, transcription, clip, config_word)
        assert result == clip_path

    def test_ffmpeg_command_uses_ass_filter(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        _, cmd = self._run_add_captions(clip_path, transcription, clip, config_word)
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1].startswith("ass=")

    def test_ffmpeg_input_is_clip_path(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        _, cmd = self._run_add_captions(clip_path, transcription, clip, config_word)
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == str(clip_path)

    def test_ass_file_deleted_after_success(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        self._run_add_captions(clip_path, transcription, clip, config_word)
        ass_path = clip_path.with_suffix(".ass")
        assert not ass_path.exists()

    def test_ass_file_deleted_on_ffmpeg_failure(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        error = subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"error")
        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run", side_effect=error):
            with pytest.raises(subprocess.CalledProcessError):
                add_captions(clip_path, transcription, clip, config_word)
        assert not clip_path.with_suffix(".ass").exists()

    def test_raises_on_ffmpeg_failure(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        error = subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"error")
        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run", side_effect=error):
            with pytest.raises(subprocess.CalledProcessError):
                add_captions(clip_path, transcription, clip, config_word)

    def test_tmp_mp4_deleted_on_ffmpeg_failure(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        error = subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"error")
        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run", side_effect=error):
            with pytest.raises(subprocess.CalledProcessError):
                add_captions(clip_path, transcription, clip, config_word)
        mp4_files = list(tmp_path.glob("*.mp4"))
        assert len(mp4_files) == 1
        assert mp4_files[0] == clip_path

    def test_ffmpeg_ass_uses_safe_temp_path(self, tmp_path, config_word, transcription, clip):
        # Original clip sits in a directory with spaces and special chars; FFmpeg
        # must receive a safe temp path (no special chars) to avoid filter-parse errors.
        spaced_dir = tmp_path / 'my output, "special"'
        spaced_dir.mkdir()
        clip_path = spaced_dir / "clip_01.mp4"
        clip_path.touch()
        _, cmd = self._run_add_captions(clip_path, transcription, clip, config_word)
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert vf_value.startswith("ass=")
        # Safe temp path must not contain the original problematic directory name
        assert "my output" not in vf_value
        assert ',"special"' not in vf_value

    def test_word_style_generates_event_per_word(self, tmp_path, config_word, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        written_content = []

        def capture_run(cmd, **kwargs):
            ass_path = self._extract_ass_path(cmd)
            written_content.append(ass_path.read_text(encoding="utf-8"))
            # Simulate creating output (tmp file)
            Path(cmd[-1]).touch()
            return MagicMock(returncode=0)

        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run", side_effect=capture_run):
            add_captions(clip_path, transcription, clip, config_word)

        content = written_content[0]
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue")]
        assert len(dialogue_lines) == 5  # 5 words total in clip interval

    def test_phrase_style_generates_event_per_segment(self, tmp_path, config_phrase, transcription, clip):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()
        written_content = []

        def capture_run(cmd, **kwargs):
            ass_path = self._extract_ass_path(cmd)
            written_content.append(ass_path.read_text(encoding="utf-8"))
            Path(cmd[-1]).touch()
            return MagicMock(returncode=0)

        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=True), \
             patch("youcut.captioner.subprocess.run", side_effect=capture_run):
            add_captions(clip_path, transcription, clip, config_phrase)

        content = written_content[0]
        dialogue_lines = [l for l in content.split("\n") if l.startswith("Dialogue")]
        assert len(dialogue_lines) == 2  # 2 segments in clip interval

    def test_skips_burn_in_when_ass_filter_is_unavailable(
        self, tmp_path, config_word, transcription, clip
    ):
        clip_path = tmp_path / "clip_01.mp4"
        clip_path.touch()

        with patch("youcut.captioner._ffmpeg_supports_ass", return_value=False), \
             patch("youcut.captioner.subprocess.run") as mock_run:
            result = add_captions(clip_path, transcription, clip, config_word)

        assert result == clip_path
        assert not clip_path.with_suffix(".ass").exists()
        mock_run.assert_not_called()
