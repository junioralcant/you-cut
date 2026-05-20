"""Testes de youcut.players.timeline."""

from __future__ import annotations

from pathlib import Path

from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp
from youcut.players.catalog import load_catalog
from youcut.players.timeline import PlayerSegment, build_player_timeline


def _populate(tmp_path: Path, slugs: list[str]) -> Path:
    for slug in slugs:
        (tmp_path / f"{slug}.jpg").write_bytes(b"fake")
    return tmp_path


def _make_transcript(words: list[tuple[str, float, float]]) -> TranscriptionResult:
    ws = [WordTimestamp(word=w, start=s, end=e) for w, s, e in words]
    seg = TranscriptionSegment(
        start=ws[0].start if ws else 0.0,
        end=ws[-1].end if ws else 0.0,
        text=" ".join(w for w, _, _ in words),
        words=ws,
    )
    return TranscriptionResult(language="pt", segments=[seg], source_path=Path("/tmp/fake.mp4"))


def test_empty_when_no_transcription(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    assert build_player_timeline(None, 0.0, 10.0, catalog) == []


def test_empty_when_no_mentions(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([("o", 0.0, 0.3), ("jogo", 0.3, 0.8)])
    assert build_player_timeline(transcript, 0.0, 5.0, catalog) == []


def test_single_player_covers_full_clip(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("o", 0.0, 0.3),
        ("Neymar", 2.0, 2.4),
        ("driblou", 2.5, 3.0),
    ])
    segs = build_player_timeline(transcript, 0.0, 10.0, catalog)
    assert len(segs) == 1
    assert segs[0].profile.slug == "neymar"
    assert segs[0].start == 0.0  # estende pra trás
    assert segs[0].end == 10.0  # estende pra frente até o fim


def test_two_players_alternated_split_at_second_start(tmp_path: Path):
    _populate(tmp_path, ["neymar", "vinicius_junior"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("o", 0.0, 0.3),
        ("Neymar", 1.0, 1.5),
        ("e", 1.5, 1.7),
        ("o", 1.7, 1.8),
        ("Vinicius", 5.0, 5.5),
        ("Junior", 5.5, 5.9),
    ])
    segs = build_player_timeline(transcript, 0.0, 10.0, catalog)
    assert len(segs) == 2
    assert segs[0].profile.slug == "neymar"
    assert segs[0].start == 0.0  # estende pra trás
    assert segs[0].end == 5.0  # corta no início do próximo
    assert segs[1].profile.slug == "vinicius_junior"
    assert segs[1].start == 5.0
    assert segs[1].end == 10.0  # estende pra frente


def test_unknown_player_between_two_known_does_not_break_segment(tmp_path: Path):
    """Apresentador cita 'Endrick' (no catálogo) → 'Rodrygo' (FORA) → 'Casemiro'
    (no catálogo). A foto do Endrick deve permanecer até o Casemiro."""
    _populate(tmp_path, ["endrick", "casemiro"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("o", 0.0, 0.3),
        ("Endrick", 1.0, 1.5),
        ("e", 1.5, 1.7),
        ("Rodrygo", 3.0, 3.5),  # FORA do catálogo → sem efeito
        ("e", 3.5, 3.7),
        ("Casemiro", 6.0, 6.5),
    ])
    segs = build_player_timeline(transcript, 0.0, 10.0, catalog)
    assert len(segs) == 2
    assert segs[0].profile.slug == "endrick"
    assert segs[0].start == 0.0
    assert segs[0].end == 6.0  # corta no Casemiro, não no Rodrygo
    assert segs[1].profile.slug == "casemiro"
    assert segs[1].start == 6.0


def test_repeated_mentions_same_player_merged(tmp_path: Path):
    """'Neymar ... Neymar ... Vini' → 2 segmentos (Neymar mesclado, depois Vini)."""
    _populate(tmp_path, ["neymar", "vinicius_junior"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("Neymar", 1.0, 1.5),
        ("driblou", 1.5, 2.0),
        ("e", 2.0, 2.1),
        ("Neymar", 3.0, 3.4),
        ("chutou", 3.4, 3.8),
        ("Vinicius", 6.0, 6.5),
    ])
    segs = build_player_timeline(transcript, 0.0, 10.0, catalog)
    assert len(segs) == 2
    assert segs[0].profile.slug == "neymar"
    assert segs[0].start == 0.0
    assert segs[0].end == 6.0  # vai até o início do Vinicius
    assert segs[1].profile.slug == "vinicius_junior"


def test_clip_offset_translates_to_relative_timestamps(tmp_path: Path):
    """Menções na faixa [120, 130] do vídeo viram [0, 10] no clipe."""
    _populate(tmp_path, ["neymar", "vinicius_junior"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("Neymar", 121.0, 121.5),  # rel = 1.0
        ("e", 121.5, 121.7),
        ("Vinicius", 125.0, 125.5),  # rel = 5.0
    ])
    segs = build_player_timeline(transcript, 120.0, 130.0, catalog)
    assert len(segs) == 2
    assert segs[0].profile.slug == "neymar"
    assert segs[0].start == 0.0
    assert segs[0].end == 5.0
    assert segs[1].profile.slug == "vinicius_junior"
    assert segs[1].start == 5.0
    assert segs[1].end == 10.0


def test_empty_catalog_returns_empty(tmp_path: Path):
    catalog = load_catalog(tmp_path)  # vazio
    transcript = _make_transcript([("Neymar", 1.0, 1.5)])
    assert build_player_timeline(transcript, 0.0, 5.0, catalog) == []


def test_ambiguous_without_claude_is_dropped(tmp_path: Path):
    """Sem cliente Claude, menção ambígua é descartada (política conservadora)."""
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo", "neymar"])
    catalog = load_catalog(tmp_path)
    transcript = _make_transcript([
        ("Danilo", 1.0, 1.4),  # ambíguo → descartado
        ("e", 1.4, 1.5),
        ("Neymar", 3.0, 3.5),
    ])
    segs = build_player_timeline(transcript, 0.0, 8.0, catalog)
    assert len(segs) == 1
    assert segs[0].profile.slug == "neymar"
    assert segs[0].start == 0.0  # estende pra trás
    assert segs[0].end == 8.0
