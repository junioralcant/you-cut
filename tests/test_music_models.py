"""Testes unitários para MusicTrack em youcut/models.py."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import MusicTrack


def test_music_track_instantiation():
    track = MusicTrack(
        name="Upbeat Morning",
        pixabay_url="https://pixabay.com/music/upbeat-morning-123/",
        local_path=Path("/tmp/music.mp3"),
        mood="motivacional",
        duration_s=120.5,
    )
    assert track.name == "Upbeat Morning"
    assert track.pixabay_url == "https://pixabay.com/music/upbeat-morning-123/"
    assert track.local_path == Path("/tmp/music.mp3")
    assert track.mood == "motivacional"
    assert track.duration_s == 120.5


def test_music_track_missing_required_field_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            name="Upbeat Morning",
            pixabay_url="https://pixabay.com/music/test/",
            local_path=Path("/tmp/music.mp3"),
            # mood ausente
            duration_s=60.0,
        )


def test_music_track_missing_name_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            pixabay_url="https://pixabay.com/music/test/",
            local_path=Path("/tmp/music.mp3"),
            mood="feliz",
            duration_s=60.0,
        )


def test_music_track_missing_duration_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            name="Test",
            pixabay_url="https://pixabay.com/music/test/",
            local_path=Path("/tmp/music.mp3"),
            mood="feliz",
        )


def test_music_track_missing_pixabay_url_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            name="Test",
            local_path=Path("/tmp/music.mp3"),
            mood="feliz",
            duration_s=60.0,
        )


def test_music_track_missing_local_path_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            name="Test",
            pixabay_url="https://pixabay.com/music/test/",
            mood="feliz",
            duration_s=60.0,
        )
