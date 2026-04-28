from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from youcut.models import ClipRecord, ViralClip
from youcut.reviewer import review_clips


def _make_clip(title: str = "Clip") -> ViralClip:
    return ViralClip(
        title=title,
        reason="test",
        viral_score=8.0,
        start_time=0.0,
        end_time=60.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="idea",
        thumbnail_text="MOMENTO IMPACTANTE",
        cut_mode="social",
    )


def _make_record(title: str = "Clip", thumbnail: Path | None = None) -> ClipRecord:
    return ClipRecord(
        title=title,
        start_time=0.0,
        end_time=60.0,
        clip_path=Path("/tmp/clip.mp4"),
        thumbnail_path=thumbnail,
        approved=True,
    )


def _mock_select(answers: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.return_value.ask.side_effect = answers
    return mock


class TestReviewClipsApproval:
    def test_approve_all_clips(self):
        clips = [_make_clip("A"), _make_clip("B")]
        records = [_make_record("A"), _make_record("B")]

        with patch("youcut.reviewer.questionary.select", _mock_select(["Aprovar", "Aprovar"])):
            result = review_clips(clips, records, "social")

        assert all(r.approved for r in result)

    def test_reject_second_clip(self):
        clips = [_make_clip("A"), _make_clip("B")]
        records = [_make_record("A"), _make_record("B")]

        with patch("youcut.reviewer.questionary.select", _mock_select(["Aprovar", "Rejeitar"])):
            result = review_clips(clips, records, "social")

        assert result[0].approved is True
        assert result[1].approved is False

    def test_none_action_rejects_clip(self):
        clips = [_make_clip()]
        records = [_make_record()]

        with patch("youcut.reviewer.questionary.select", _mock_select([None])):
            result = review_clips(clips, records, "social")

        assert result[0].approved is False

    def test_original_records_not_mutated(self):
        clips = [_make_clip()]
        records = [_make_record()]

        with patch("youcut.reviewer.questionary.select", _mock_select(["Rejeitar"])):
            review_clips(clips, records, "social")

        assert records[0].approved is True


class TestTitleEditing:
    def test_edit_title_updates_record(self):
        clips = [_make_clip("Original")]
        records = [_make_record("Original")]

        mock_select = _mock_select(["Editar título", "Aprovar"])
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = "Novo Título"

        with patch("youcut.reviewer.questionary.select", mock_select), \
             patch("youcut.reviewer.questionary.text", mock_text):
            result = review_clips(clips, records, "social")

        assert result[0].title == "Novo Título"

    def test_empty_title_keeps_original(self):
        clips = [_make_clip("Original")]
        records = [_make_record("Original")]

        mock_select = _mock_select(["Editar título", "Aprovar"])
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = ""

        with patch("youcut.reviewer.questionary.select", mock_select), \
             patch("youcut.reviewer.questionary.text", mock_text):
            result = review_clips(clips, records, "social")

        assert result[0].title == "Original"

    def test_youtube_title_prompt_shows_guidance(self):
        clips = [_make_clip("Original")]
        records = [_make_record("Original")]

        mock_select = _mock_select(["Editar título", "Aprovar"])
        mock_text = MagicMock()
        mock_text.return_value.ask.return_value = "Titulo revisado para YouTube"

        with patch("youcut.reviewer.questionary.select", mock_select), \
             patch("youcut.reviewer.questionary.text", mock_text):
            review_clips(clips, records, "youtube")

        prompt = mock_text.call_args.args[0]
        assert "5-9 palavras" in prompt
        assert "30 caracteres" in prompt
        assert "pode passar" in prompt


class TestYoutubeMode:
    def test_regenerate_thumbnail_called(self):
        thumbnail = Path("/tmp/thumb.png")
        clips = [ViralClip(
            title="T",
            reason="r",
            viral_score=7.0,
            start_time=0.0,
            end_time=300.0,
            description="d",
            hashtags=[],
            thumbnail_idea="idea",
            thumbnail_text="MOMENTO IMPACTANTE",
            cut_mode="youtube",
        )]
        records = [ClipRecord(
            title="T",
            start_time=0.0,
            end_time=300.0,
            clip_path=Path("/tmp/clip.mp4"),
            thumbnail_path=thumbnail,
            approved=True,
        )]

        new_thumb = Path("/tmp/new_thumb.png")
        mock_select = _mock_select(["Regenerar thumbnail", "Aprovar"])

        with patch("youcut.reviewer.questionary.select", mock_select), \
             patch("youcut.reviewer.thumbnail_generator.regenerate_thumbnail", return_value=new_thumb) as mock_regen:
            result = review_clips(clips, records, "youtube", api_key="test-key")

        mock_regen.assert_called_once_with(clips[0], result[0], "test-key")
        assert result[0].thumbnail_path == new_thumb

    def test_regenerate_thumbnail_not_available_in_social_mode(self):
        clips = [_make_clip()]
        records = [_make_record()]

        captured_choices: list[list[str]] = []

        def mock_select_fn(prompt, choices):
            captured_choices.append(list(choices))
            mock = MagicMock()
            mock.ask.return_value = "Aprovar"
            return mock

        with patch("youcut.reviewer.questionary.select", mock_select_fn):
            review_clips(clips, records, "social")

        assert "Regenerar thumbnail" not in captured_choices[0]

    def test_regenerate_thumbnail_available_in_youtube_mode(self):
        clips = [ViralClip(
            title="T",
            reason="r",
            viral_score=7.0,
            start_time=0.0,
            end_time=300.0,
            description="d",
            hashtags=[],
            thumbnail_idea="idea",
            thumbnail_text="MOMENTO IMPACTANTE",
            cut_mode="youtube",
        )]
        records = [ClipRecord(
            title="T",
            start_time=0.0,
            end_time=300.0,
            clip_path=Path("/tmp/clip.mp4"),
            thumbnail_path=None,
            approved=True,
        )]

        captured_choices: list[list[str]] = []

        def mock_select_fn(prompt, choices):
            captured_choices.append(list(choices))
            mock = MagicMock()
            mock.ask.return_value = "Aprovar"
            return mock

        with patch("youcut.reviewer.questionary.select", mock_select_fn):
            review_clips(clips, records, "youtube")

        assert "Regenerar thumbnail" in captured_choices[0]
