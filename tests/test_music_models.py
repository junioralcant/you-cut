"""Testes unitários para MusicTrack e SyncReport em youcut/models.py."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import MusicTrack, SyncReport


def test_music_track_instantiation():
    track = MusicTrack(
        video_id="dQw4w9WgXcQ",
        name="Upbeat Morning",
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        local_path=Path("/tmp/dQw4w9WgXcQ.m4a"),
        mood="motivacional",
        duration_s=120.5,
    )
    assert track.video_id == "dQw4w9WgXcQ"
    assert track.name == "Upbeat Morning"
    assert track.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert track.local_path == Path("/tmp/dQw4w9WgXcQ.m4a")
    assert track.mood == "motivacional"
    assert track.duration_s == 120.5


def test_music_track_missing_video_id_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            name="Test",
            source_url="https://www.youtube.com/watch?v=abc",
            local_path=Path("/tmp/abc.m4a"),
            mood="feliz",
            duration_s=60.0,
        )


def test_music_track_missing_source_url_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            video_id="abc",
            name="Test",
            local_path=Path("/tmp/abc.m4a"),
            mood="feliz",
            duration_s=60.0,
        )


def test_music_track_missing_name_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            video_id="abc",
            source_url="https://www.youtube.com/watch?v=abc",
            local_path=Path("/tmp/abc.m4a"),
            mood="feliz",
            duration_s=60.0,
        )


def test_music_track_missing_mood_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            video_id="abc",
            name="Test",
            source_url="https://www.youtube.com/watch?v=abc",
            local_path=Path("/tmp/abc.m4a"),
            duration_s=60.0,
        )


def test_music_track_missing_duration_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            video_id="abc",
            name="Test",
            source_url="https://www.youtube.com/watch?v=abc",
            local_path=Path("/tmp/abc.m4a"),
            mood="feliz",
        )


def test_music_track_missing_local_path_raises():
    with pytest.raises(ValidationError):
        MusicTrack(
            video_id="abc",
            name="Test",
            source_url="https://www.youtube.com/watch?v=abc",
            mood="feliz",
            duration_s=60.0,
        )


def test_sync_report_defaults():
    report = SyncReport()
    assert report.new_tracks == 0
    assert report.cached_tracks == 0
    assert report.failed_tracks == 0
    assert report.failed_details == []


def test_sync_report_with_values():
    report = SyncReport(
        new_tracks=3,
        cached_tracks=10,
        failed_tracks=1,
        failed_details=[("vid_failed", "download timeout")],
    )
    assert report.new_tracks == 3
    assert report.cached_tracks == 10
    assert report.failed_tracks == 1
    assert report.failed_details == [("vid_failed", "download timeout")]
