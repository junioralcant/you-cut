"""Teste de integração para PlaylistSyncer (yt_dlp + Anthropic mockados).

Reusa um diretório temporário como `~/.youcut/music`, mocka `yt_dlp.YoutubeDL`
e `anthropic.Anthropic` end-to-end, e executa duas syncs consecutivas para
verificar idempotência (RF-04) + contagens do `SyncReport` (RF-05).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.integration


class _FakeYDL:
    def __init__(
        self,
        *,
        playlist_entries,
        per_video_info,
        tracks_dir: Path,
    ) -> None:
        self._playlist_entries = playlist_entries
        self._per_video_info = per_video_info
        self._tracks_dir = tracks_dir

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        if "playlist" in url:
            return {"entries": list(self._playlist_entries)}
        for vid, info in self._per_video_info.items():
            if vid in url:
                return info
        return {}

    def download(self, urls):
        for url in urls:
            for vid in self._per_video_info:
                if vid in url:
                    dest = self._tracks_dir / f"{vid}.m4a"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b"fake-audio")


def _build_anthropic_mock(mood: str) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_track_mood"
    block.input = {"mood": mood}

    response = MagicMock()
    response.content = [block]

    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response
    return client


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def test_two_consecutive_syncs_are_idempotent(tmp_path, config):
    from youcut.music.classifier import TrackMoodClassifier
    from youcut.music.library import MusicLibrary
    from youcut.music.sync import PlaylistSyncer

    root = tmp_path / "music"
    entries = [
        {"id": "abc", "title": "Cinematic Drama"},
        {"id": "xyz", "title": "Happy Pop"},
        {"id": "def", "title": "Workout Beat"},
    ]
    per_video_info = {
        "abc": {"title": "Cinematic Drama", "description": "tense strings", "tags": ["cinematic"], "duration": 200.0},
        "xyz": {"title": "Happy Pop", "description": "summer vibe", "tags": ["pop"], "duration": 180.0},
        "def": {"title": "Workout Beat", "description": "intense edm", "tags": ["edm"], "duration": 220.0},
    }

    fake_ydl_factory = lambda tracks_dir: _FakeYDL(
        playlist_entries=entries,
        per_video_info=per_video_info,
        tracks_dir=tracks_dir,
    )

    # Primeira sync
    lib1 = MusicLibrary(root=root)
    fake1 = fake_ydl_factory(lib1.tracks_dir)
    with (
        patch("youcut.music.sync.yt_dlp.YoutubeDL", fake1),
        patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_build_anthropic_mock("dramatico"),
        ),
    ):
        classifier1 = TrackMoodClassifier(config)
        syncer1 = PlaylistSyncer(lib1, classifier1, auth_config=None)
        report1 = syncer1.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report1.new_tracks == 3
    assert report1.cached_tracks == 0
    assert report1.failed_tracks == 0
    # Arquivos m4a criados
    assert (lib1.tracks_dir / "abc.m4a").exists()
    assert (lib1.tracks_dir / "xyz.m4a").exists()
    assert (lib1.tracks_dir / "def.m4a").exists()

    # Segunda sync (mesma playlist, lib reaberta)
    lib2 = MusicLibrary(root=root)
    fake2 = fake_ydl_factory(lib2.tracks_dir)
    with (
        patch("youcut.music.sync.yt_dlp.YoutubeDL", fake2),
        patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_build_anthropic_mock("dramatico"),
        ),
    ):
        classifier2 = TrackMoodClassifier(config)
        syncer2 = PlaylistSyncer(lib2, classifier2, auth_config=None)
        report2 = syncer2.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report2.new_tracks == 0
    assert report2.cached_tracks == 3
    assert report2.failed_tracks == 0
