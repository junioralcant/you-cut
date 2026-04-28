from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.analyzer import (
    SOCIAL_MAX_DURATION,
    SOCIAL_MIN_DURATION,
    YOUTUBE_TITLE_IDEAL_MAX_CHARS,
    YOUTUBE_TITLE_MAX_WORDS,
    YOUTUBE_TITLE_MIN_WORDS,
    YOUTUBE_MAX_DURATION,
    YOUTUBE_MIN_DURATION,
    analyze,
)
from youcut.models import TranscriptionResult, TranscriptionSegment, ViralClip, WordTimestamp


@pytest.fixture
def social_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="social")


@pytest.fixture
def youtube_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="youtube")


@pytest.fixture
def transcription():
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=600.0,
                text="Conteúdo longo de 10 minutos",
                words=[WordTimestamp(word="Conteúdo", start=0.0, end=0.5)],
            ),
            TranscriptionSegment(
                start=600.0,
                end=1200.0,
                text="Mais 10 minutos de conteúdo relevante",
                words=[WordTimestamp(word="Mais", start=600.0, end=600.5)],
            ),
        ],
        language="pt",
        source_path=Path("test.mp4"),
    )


def _make_mock_client(clips_data: list[dict]) -> MagicMock:
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "identify_viral_clips"
    mock_block.input = {"clips": clips_data}

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.with_options.return_value = mock_client
    mock_client.messages.create.return_value = mock_response
    return mock_client


def _clip(title: str, start: float, end: float, score: float = 8.0) -> dict:
    return {
        "title": title,
        "reason": "R",
        "viral_score": score,
        "start_time": start,
        "end_time": end,
        "description": "D",
        "hashtags": [],
        "thumbnail_idea": "T",
        "thumbnail_text": "MOMENTO IMPACTANTE",
    }


class TestSocialModeLimits:
    def test_social_clips_within_max_duration_accepted(self, social_config, transcription):
        clips_data = [_clip("C1", 0.0, float(SOCIAL_MAX_DURATION))]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 1
        assert result[0].end_time - result[0].start_time <= SOCIAL_MAX_DURATION

    def test_social_clips_above_max_duration_filtered(self, social_config, transcription):
        clips_data = [
            _clip("Too Long", 0.0, SOCIAL_MAX_DURATION + 1, score=9.0),
            _clip("Valid", 0.0, 60.0, score=8.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 1
        assert result[0].title == "Valid"

    def test_social_clips_below_min_duration_filtered(self, social_config, transcription):
        clips_data = [
            _clip("Too Short", 0.0, SOCIAL_MIN_DURATION - 1, score=9.0),
            _clip("Valid", 0.0, 60.0, score=8.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 1
        assert result[0].title == "Valid"


class TestYouTubeModeLimits:
    def test_youtube_clips_within_bounds_accepted(self, youtube_config, transcription):
        clips_data = [_clip("C1", 0.0, float(YOUTUBE_MIN_DURATION))]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, youtube_config)

        assert len(result) == 1
        dur = result[0].end_time - result[0].start_time
        assert YOUTUBE_MIN_DURATION <= dur <= YOUTUBE_MAX_DURATION

    def test_youtube_clips_below_min_filtered(self, youtube_config, transcription):
        clips_data = [
            _clip("Short", 0.0, YOUTUBE_MIN_DURATION - 1, score=9.0),
            _clip("Valid", 0.0, float(YOUTUBE_MIN_DURATION), score=8.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, youtube_config)

        assert len(result) == 1
        assert result[0].title == "Valid"

    def test_youtube_clips_above_max_filtered(self, youtube_config, transcription):
        clips_data = [
            _clip("Too Long", 0.0, YOUTUBE_MAX_DURATION + 1, score=9.0),
            _clip("Valid", 0.0, float(YOUTUBE_MAX_DURATION), score=8.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, youtube_config)

        assert len(result) == 1
        assert result[0].title == "Valid"


class TestPromptContainsModeSpecificLimits:
    def _get_system_prompt_text(self, mock_client: MagicMock) -> str:
        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_blocks = call_kwargs.get("system", [])
        return " ".join(b["text"] for b in system_blocks if b.get("type") == "text")

    def test_social_prompt_contains_social_limits(self, social_config, transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(transcription, social_config)

        system_text = self._get_system_prompt_text(mock_client)
        assert str(SOCIAL_MIN_DURATION) in system_text
        assert str(SOCIAL_MAX_DURATION) in system_text

    def test_youtube_prompt_contains_youtube_limits(self, youtube_config, transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(transcription, youtube_config)

        system_text = self._get_system_prompt_text(mock_client)
        assert str(YOUTUBE_MIN_DURATION) in system_text
        assert str(YOUTUBE_MAX_DURATION) in system_text

    def test_youtube_prompt_contains_title_guidance(self, youtube_config, transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(transcription, youtube_config)

        system_text = self._get_system_prompt_text(mock_client)
        assert f"{YOUTUBE_TITLE_MIN_WORDS}" in system_text
        assert f"{YOUTUBE_TITLE_MAX_WORDS}" in system_text
        assert f"{YOUTUBE_TITLE_IDEAL_MAX_CHARS}" in system_text
        assert "pode ultrapassar" in system_text


class TestMaxClips:
    def test_max_clips_limits_results(self, social_config, transcription):
        monkeypatch_config = social_config.model_copy(update={"max_clips": 2})
        clips_data = [
            _clip("C1", 0.0, 60.0, score=9.0),
            _clip("C2", 60.0, 120.0, score=8.0),
            _clip("C3", 120.0, 180.0, score=7.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, monkeypatch_config)

        assert len(result) <= 2

    def test_max_clips_in_prompt(self, social_config, transcription):
        monkeypatch_config = social_config.model_copy(update={"max_clips": 3})
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(transcription, monkeypatch_config)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        user_text = messages[0]["content"][0]["text"]
        assert "3" in user_text

    def test_no_max_clips_returns_all(self, social_config, transcription):
        clips_data = [_clip(f"C{i}", i * 30.0, (i + 1) * 30.0) for i in range(5)]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 5


class TestCutModeOnClips:
    def test_cut_mode_set_on_social_clips(self, social_config, transcription):
        clips_data = [_clip("C1", 0.0, 60.0)]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 1
        assert result[0].cut_mode == "social"

    def test_cut_mode_set_on_youtube_clips(self, youtube_config, transcription):
        clips_data = [_clip("C1", 0.0, float(YOUTUBE_MIN_DURATION))]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, youtube_config)

        assert len(result) == 1
        assert result[0].cut_mode == "youtube"

    def test_youtube_title_above_ideal_limit_is_still_accepted(self, youtube_config, transcription):
        long_title = "Esse titulo passa um pouco do limite ideal"
        clips_data = [_clip(long_title, 0.0, float(YOUTUBE_MIN_DURATION))]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, youtube_config)

        assert len(result) == 1
        assert result[0].title == long_title

    def test_all_returned_clips_have_cut_mode(self, social_config, transcription):
        clips_data = [_clip(f"C{i}", i * 30.0, (i + 1) * 30.0) for i in range(3)]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert all(isinstance(c, ViralClip) and c.cut_mode == "social" for c in result)


class TestOverlapRemoval:
    def test_overlapping_clips_from_model_are_removed(self, social_config, transcription):
        # C1 (score 9) wins; C2 overlaps C1 → removed; C3 doesn't overlap → kept
        clips_data = [
            _clip("C1", 0.0, 60.0, score=9.0),
            _clip("C2", 30.0, 90.0, score=7.0),   # overlaps C1
            _clip("C3", 90.0, 150.0, score=8.0),  # no overlap
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        titles = [c.title for c in result]
        assert "C2" not in titles
        assert "C1" in titles
        assert "C3" in titles

    def test_overlapping_clips_lower_score_removed(self, social_config, transcription):
        clips_data = [
            _clip("High", 0.0, 60.0, score=9.0),
            _clip("Low", 0.0, 60.0, score=5.0),  # same interval, lower score
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 1
        assert result[0].title == "High"


class TestNoOverlap:
    def test_clips_sorted_by_viral_score(self, social_config, transcription):
        clips_data = [
            _clip("Low", 0.0, 30.0, score=4.0),
            _clip("High", 30.0, 60.0, score=9.0),
            _clip("Mid", 60.0, 90.0, score=6.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        scores = [c.viral_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_non_overlapping_clips_all_returned(self, social_config, transcription):
        clips_data = [
            _clip("C1", 0.0, 30.0),
            _clip("C2", 30.0, 60.0),
            _clip("C3", 60.0, 90.0),
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(transcription, social_config)

        assert len(result) == 3
