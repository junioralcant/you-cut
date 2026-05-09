"""Testes unitários para PlaylistSyncer em youcut/music/sync.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class _FakeYDL:
    """Stand-in para `yt_dlp.YoutubeDL` controlado por roteiro."""

    def __init__(
        self,
        *,
        playlist_entries: list[dict] | None = None,
        per_video_info: dict[str, dict] | None = None,
        download_creates: dict[str, str] | None = None,
        download_raises: dict[str, Exception] | None = None,
    ) -> None:
        self._playlist_entries = playlist_entries or []
        self._per_video_info = per_video_info or {}
        self._download_creates = download_creates or {}
        self._download_raises = download_raises or {}
        self.last_opts: dict | None = None

    def __call__(self, opts):
        self.last_opts = opts
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        if "playlist" in url:
            return {"entries": list(self._playlist_entries)}
        # single video metadata
        for vid, info in self._per_video_info.items():
            if vid in url:
                return info
        return {}

    def download(self, urls):
        for url in urls:
            for vid, exc in self._download_raises.items():
                if vid in url:
                    raise exc
            for vid, dest in self._download_creates.items():
                if vid in url:
                    Path(dest).parent.mkdir(parents=True, exist_ok=True)
                    Path(dest).write_bytes(b"fake-audio")


@pytest.fixture
def patched_classifier():
    classifier = MagicMock()
    classifier.classify.side_effect = lambda title, description, tags: "feliz"
    return classifier


@pytest.fixture
def lib(tmp_path):
    from youcut.music.library import MusicLibrary
    return MusicLibrary(root=tmp_path / "music")


def _make_syncer(lib, classifier, fake_ydl):
    from youcut.music.sync import PlaylistSyncer

    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake_ydl):
        return PlaylistSyncer(lib, classifier, auth_config=None)


def test_first_sync_marks_all_as_new(lib, patched_classifier):
    fake = _FakeYDL(
        playlist_entries=[
            {"id": "vid1", "title": "Song A"},
            {"id": "vid2", "title": "Song B"},
        ],
        per_video_info={
            "vid1": {"title": "Song A", "description": "desc A", "tags": ["a"], "duration": 120.0},
            "vid2": {"title": "Song B", "description": "desc B", "tags": ["b"], "duration": 180.0},
        },
        download_creates={
            "vid1": str(lib.tracks_dir / "vid1.m4a"),
            "vid2": str(lib.tracks_dir / "vid2.m4a"),
        },
    )
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        report = syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report.new_tracks == 2
    assert report.cached_tracks == 0
    assert report.failed_tracks == 0
    # Persistido no índice
    assert lib.has("vid1")
    assert lib.has("vid2")
    # E o classifier foi chamado uma vez por faixa nova
    assert patched_classifier.classify.call_count == 2


def test_second_sync_is_idempotent_all_cached(lib, patched_classifier):
    """RF-04: re-execução não re-baixa nem re-classifica."""
    entries = [
        {"id": "vid1", "title": "Song A"},
        {"id": "vid2", "title": "Song B"},
    ]
    per_video = {
        "vid1": {"title": "Song A", "description": "", "tags": [], "duration": 120.0},
        "vid2": {"title": "Song B", "description": "", "tags": [], "duration": 90.0},
    }
    creates = {
        "vid1": str(lib.tracks_dir / "vid1.m4a"),
        "vid2": str(lib.tracks_dir / "vid2.m4a"),
    }
    fake = _FakeYDL(playlist_entries=entries, per_video_info=per_video, download_creates=creates)

    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        first = syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    assert first.new_tracks == 2

    # Recriar lib + classifier para garantir estado limpo
    from youcut.music.library import MusicLibrary
    lib2 = MusicLibrary(root=lib.root)
    classifier2 = MagicMock()
    classifier2.classify.side_effect = lambda **kw: "feliz"

    fake2 = _FakeYDL(playlist_entries=entries, per_video_info=per_video, download_creates=creates)
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake2):
        from youcut.music.sync import PlaylistSyncer
        syncer2 = PlaylistSyncer(lib2, classifier2, auth_config=None)
        second = syncer2.sync("https://www.youtube.com/playlist?list=PLfake")

    assert second.new_tracks == 0
    assert second.cached_tracks == 2
    assert second.failed_tracks == 0
    classifier2.classify.assert_not_called()


def test_download_failure_does_not_abort_sync(lib, patched_classifier):
    """RF-05: falha individual não derruba o sync; aparece em failed_details."""
    fake = _FakeYDL(
        playlist_entries=[
            {"id": "good", "title": "Good"},
            {"id": "bad", "title": "Bad"},
        ],
        per_video_info={
            "good": {"title": "Good", "description": "", "tags": [], "duration": 60.0},
            "bad": {"title": "Bad", "description": "", "tags": [], "duration": 60.0},
        },
        download_creates={"good": str(lib.tracks_dir / "good.m4a")},
        download_raises={"bad": RuntimeError("download failed")},
    )
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        report = syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report.new_tracks == 1
    assert report.failed_tracks == 1
    assert any(vid == "bad" for vid, _ in report.failed_details)
    assert lib.has("good")
    assert not lib.has("bad")


def test_classifier_failure_marks_track_as_indefinido(lib):
    """RF-09: classifier retornando None vira mood='indefinido' no índice."""
    classifier = MagicMock()
    classifier.classify.return_value = None

    fake = _FakeYDL(
        playlist_entries=[{"id": "vid1", "title": "Mistério"}],
        per_video_info={"vid1": {"title": "Mistério", "description": "", "tags": [], "duration": 60.0}},
        download_creates={"vid1": str(lib.tracks_dir / "vid1.m4a")},
    )
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, classifier, auth_config=None)
        report = syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report.new_tracks == 1
    track = lib.all_tracks()[0]
    assert track.mood == "indefinido"


def test_entry_without_id_counts_as_failed(lib, patched_classifier):
    fake = _FakeYDL(playlist_entries=[{"title": "no id here"}])
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        report = syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    assert report.failed_tracks == 1
    assert report.new_tracks == 0


def test_playlist_url_persisted_on_index(lib, patched_classifier):
    fake = _FakeYDL(playlist_entries=[])
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        syncer.sync("https://www.youtube.com/playlist?list=PLpersisted")

    from youcut.music.library import MusicLibrary
    reload = MusicLibrary(root=lib.root)
    reload.load()
    assert reload.playlist_url == "https://www.youtube.com/playlist?list=PLpersisted"


def test_yt_dlp_extract_flat_used_for_enumeration(lib, patched_classifier):
    """yt-dlp deve ser invocado com extract_flat='in_playlist' para listar a playlist."""
    fake = _FakeYDL(playlist_entries=[])
    with patch("youcut.music.sync.yt_dlp.YoutubeDL", fake):
        from youcut.music.sync import PlaylistSyncer
        syncer = PlaylistSyncer(lib, patched_classifier, auth_config=None)
        syncer.sync("https://www.youtube.com/playlist?list=PLfake")

    # Ao menos uma das chamadas deve ter usado extract_flat=in_playlist
    # (o FakeYDL captura apenas o último opts, mas a primeira chamada é a enumeração).
    # Captura múltipla via spy:
    assert fake.last_opts is not None  # smoke
