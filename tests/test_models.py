from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import (
    TranscriptionResult,
    TranscriptionSegment,
    ViralClip,
    WordTimestamp,
)


def test_viral_clip_creation():
    clip = ViralClip(
        title="Momento Incrível",
        reason="Gancho forte no início com curiosidade",
        viral_score=9.2,
        start_time=30.0,
        end_time=75.0,
        description="Descrição completa do clipe para redes sociais",
        hashtags=["#viral", "#shorts", "#youcut"],
        thumbnail_idea="Frame com expressão de surpresa no segundo 35",
    )
    assert clip.title == "Momento Incrível"
    assert clip.viral_score == 9.2
    assert clip.start_time == 30.0
    assert clip.end_time == 75.0
    assert len(clip.hashtags) == 3


def test_viral_clip_score_too_high_raises():
    with pytest.raises(ValidationError):
        ViralClip(
            title="Clip",
            reason="x",
            viral_score=11.0,
            start_time=0.0,
            end_time=30.0,
            description="desc",
            hashtags=[],
            thumbnail_idea="thumb",
        )


def test_viral_clip_score_negative_raises():
    with pytest.raises(ValidationError):
        ViralClip(
            title="Clip",
            reason="x",
            viral_score=-1.0,
            start_time=0.0,
            end_time=30.0,
            description="desc",
            hashtags=[],
            thumbnail_idea="thumb",
        )


def test_viral_clip_score_boundary_values():
    clip_zero = ViralClip(
        title="Clip",
        reason="x",
        viral_score=0.0,
        start_time=0.0,
        end_time=30.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="thumb",
    )
    assert clip_zero.viral_score == 0.0

    clip_ten = ViralClip(
        title="Clip",
        reason="x",
        viral_score=10.0,
        start_time=0.0,
        end_time=30.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="thumb",
    )
    assert clip_ten.viral_score == 10.0


def test_transcription_result_with_segments():
    result = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Olá mundo",
                words=[
                    WordTimestamp(word="Olá", start=0.0, end=0.5),
                    WordTimestamp(word="mundo", start=0.6, end=1.0),
                ],
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Teste de transcrição",
                words=[
                    WordTimestamp(word="Teste", start=5.0, end=5.4),
                ],
            ),
        ],
        language="pt",
        source_path=Path("video.mp4"),
    )
    assert len(result.segments) == 2
    assert result.language == "pt"
    assert result.source_path == Path("video.mp4")
    assert len(result.segments[0].words) == 2


def test_word_timestamp():
    word = WordTimestamp(word="Python", start=1.2, end=1.8)
    assert word.word == "Python"
    assert word.start == 1.2
    assert word.end == 1.8
