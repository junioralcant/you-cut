"""Testes unitários para MusicLibrary em youcut/music/library.py."""
from __future__ import annotations

import json
from pathlib import Path

from youcut.models import MusicTrack
from youcut.music.library import SCHEMA_VERSION, MusicLibrary


def _track(
    *,
    video_id: str,
    name: str = "Sample",
    mood: str = "feliz",
    duration_s: float = 90.0,
    root: Path | None = None,
) -> MusicTrack:
    base = root if root is not None else Path("/tmp")
    return MusicTrack(
        video_id=video_id,
        name=name,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        local_path=base / "tracks" / f"{video_id}.m4a",
        mood=mood,
        duration_s=duration_s,
    )


def test_is_empty_when_index_does_not_exist(tmp_path: Path):
    lib = MusicLibrary(root=tmp_path / "music")
    assert lib.is_empty() is True


def test_round_trip_save_and_load(tmp_path: Path):
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    lib.set_playlist_url("https://www.youtube.com/playlist?list=PLtest")
    lib.add(_track(video_id="abc123", mood="feliz", root=root))
    lib.add(_track(video_id="xyz789", name="Other", mood="energico", duration_s=200.0, root=root))
    lib.save()

    assert (root / "index.json").exists()
    raw = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["playlist_url"] == "https://www.youtube.com/playlist?list=PLtest"
    assert set(raw["tracks"].keys()) == {"abc123", "xyz789"}

    lib2 = MusicLibrary(root=root)
    lib2.load()
    tracks = {t.video_id: t for t in lib2.all_tracks()}
    assert set(tracks.keys()) == {"abc123", "xyz789"}
    assert tracks["abc123"].mood == "feliz"
    assert tracks["xyz789"].mood == "energico"
    assert tracks["xyz789"].duration_s == 200.0
    assert tracks["abc123"].source_url.endswith("v=abc123")


def test_add_idempotent_same_video_id(tmp_path: Path):
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    lib.add(_track(video_id="dup", mood="feliz", root=root))
    lib.add(_track(video_id="dup", name="Updated Name", mood="dramatico", root=root))
    assert len(lib.all_tracks()) == 1
    only = lib.all_tracks()[0]
    assert only.video_id == "dup"
    # Última escrita vence (idempotente — não duplica)
    assert only.name == "Updated Name"
    assert only.mood == "dramatico"


def test_has_video_id(tmp_path: Path):
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    lib.add(_track(video_id="present", root=root))
    assert lib.has("present") is True
    assert lib.has("absent") is False


def test_candidates_for_filters_by_mood(tmp_path: Path):
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    lib.add(_track(video_id="a", mood="feliz", root=root))
    lib.add(_track(video_id="b", mood="dramatico", root=root))
    lib.add(_track(video_id="c", mood="feliz", root=root))

    feliz = lib.candidates_for("feliz")
    assert {t.video_id for t in feliz} == {"a", "c"}
    dramatico = lib.candidates_for("dramatico")
    assert {t.video_id for t in dramatico} == {"b"}
    energico = lib.candidates_for("energico")
    assert energico == []


def test_candidates_for_returns_sorted_by_video_id(tmp_path: Path):
    """Determinismo: a ordem de retorno deve ser estável (sort por video_id)."""
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    for vid in ["zzz", "aaa", "mmm"]:
        lib.add(_track(video_id=vid, mood="feliz", root=root))
    ordered = [t.video_id for t in lib.candidates_for("feliz")]
    assert ordered == ["aaa", "mmm", "zzz"]


def test_is_empty_after_load_with_empty_tracks(tmp_path: Path):
    root = tmp_path / "music"
    root.mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "playlist_url": "", "tracks": {}}),
        encoding="utf-8",
    )
    lib = MusicLibrary(root=root)
    assert lib.is_empty() is True


def test_save_is_atomic_via_tmp_replace(tmp_path: Path, monkeypatch):
    """Garante que save() escreve via arquivo temporário + os.replace, sem deixar lixo."""
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    lib.add(_track(video_id="vid1", root=root))
    lib.save()
    # Não deve ter lixo (.tmp) sobrando no diretório
    leftovers = [p for p in root.iterdir() if p.name != "index.json"]
    assert leftovers == [], f"esperado apenas index.json, encontrado {leftovers}"


def test_local_path_relative_storage(tmp_path: Path):
    """O `file` persistido no índice deve ser relativo à `root`."""
    root = tmp_path / "music"
    lib = MusicLibrary(root=root)
    track_path = root / "tracks" / "vid42.m4a"
    track_path.parent.mkdir(parents=True)
    track_path.write_bytes(b"")
    lib.add(
        MusicTrack(
            video_id="vid42",
            name="N",
            source_url="https://www.youtube.com/watch?v=vid42",
            local_path=track_path,
            mood="feliz",
            duration_s=10.0,
        )
    )
    lib.save()
    raw = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert raw["tracks"]["vid42"]["file"] == "tracks/vid42.m4a"

    # E a leitura reconstrói local_path absoluto baseado em root
    lib2 = MusicLibrary(root=root)
    lib2.load()
    [t] = lib2.all_tracks()
    assert t.local_path == track_path
