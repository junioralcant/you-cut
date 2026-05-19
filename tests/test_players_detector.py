"""Testes de youcut.players.detector."""

from __future__ import annotations

from pathlib import Path

from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp
from youcut.players.catalog import load_catalog
from youcut.players.detector import (
    _normalize_word,
    detect_players,
    slice_transcript_for_clip,
)


def _make_words(pairs: list[tuple[str, float, float]]) -> list[WordTimestamp]:
    return [WordTimestamp(word=w, start=s, end=e) for w, s, e in pairs]


def _populate(tmp_path: Path, slugs: list[str]) -> Path:
    for slug in slugs:
        (tmp_path / f"{slug}.jpg").write_bytes(b"fake")
    return tmp_path


def test_normalize_word_strips_accents_and_punctuation():
    assert _normalize_word("Vinícius,") == "vinicius"
    assert _normalize_word("Casemiro!") == "casemiro"
    assert _normalize_word("") == ""


def test_detect_players_finds_single_token(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    words = _make_words([("o", 0, 0.1), ("Neymar", 0.1, 0.5), ("driblou", 0.5, 1.0)])
    mentions = detect_players(words, catalog)
    assert len(mentions) == 1
    assert mentions[0].profile.slug == "neymar"
    assert mentions[0].alias_hit == "neymar"
    assert mentions[0].start == 0.1
    assert mentions[0].end == 0.5


def test_detect_players_greedy_multi_token(tmp_path: Path):
    _populate(tmp_path, ["vinicius_junior"])
    catalog = load_catalog(tmp_path)
    # "Vinicius Junior" deve casar como bloco (alias de 2 tokens), e NÃO virar
    # duas menções separadas.
    words = _make_words([
        ("o", 0.0, 0.1),
        ("Vinicius", 0.1, 0.5),
        ("Junior", 0.5, 0.9),
        ("marcou", 0.9, 1.3),
    ])
    mentions = detect_players(words, catalog)
    assert len(mentions) == 1
    assert mentions[0].alias_hit == "vinicius junior"
    assert mentions[0].start == 0.1
    assert mentions[0].end == 0.9


def test_detect_players_handles_punctuation(tmp_path: Path):
    _populate(tmp_path, ["casemiro"])
    catalog = load_catalog(tmp_path)
    words = _make_words([("passou", 0, 0.4), ("pro", 0.4, 0.5), ("Casemiro,", 0.5, 1.0)])
    mentions = detect_players(words, catalog)
    assert len(mentions) == 1
    assert mentions[0].profile.slug == "casemiro"


def test_detect_players_ambiguous_returns_all_candidates(tmp_path: Path):
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo"])
    catalog = load_catalog(tmp_path)
    words = _make_words([("e", 0, 0.1), ("Danilo", 0.1, 0.5), ("cobrou", 0.5, 1.0)])
    mentions = detect_players(words, catalog)
    assert len(mentions) == 2
    slugs = sorted(m.profile.slug for m in mentions)
    assert slugs == ["danilo_botafogo", "danilo_flamengo"]
    # Ambos compartilham o mesmo intervalo de tempo (mesma menção,
    # candidatos múltiplos — disambiguator decide depois).
    starts = {m.start for m in mentions}
    ends = {m.end for m in mentions}
    assert starts == {0.1}
    assert ends == {0.5}


def test_detect_players_no_match(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    words = _make_words([("o", 0, 0.1), ("juiz", 0.1, 0.5), ("apitou", 0.5, 1.0)])
    assert detect_players(words, catalog) == []


def test_detect_players_empty_inputs(tmp_path: Path):
    catalog = load_catalog(tmp_path)  # catálogo vazio
    assert detect_players([], catalog) == []


def test_detect_players_accent_insensitive(tmp_path: Path):
    _populate(tmp_path, ["vinicius_junior"])
    catalog = load_catalog(tmp_path)
    # Transcrição com acento bate com alias sem acento (e vice-versa).
    words = _make_words([("Vinícius", 0, 0.5), ("Júnior", 0.5, 1.0)])
    mentions = detect_players(words, catalog)
    assert len(mentions) == 1
    assert mentions[0].profile.slug == "vinicius_junior"


def test_slice_transcript_filters_words_by_clip_range(tmp_path: Path):
    transcript = TranscriptionResult(
        language="pt",
        source_path=tmp_path / "fake.mp4",
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=3.0,
                text="palavra1 palavra2 palavra3",
                words=[
                    WordTimestamp(word="palavra1", start=0.0, end=1.0),
                    WordTimestamp(word="palavra2", start=1.0, end=2.0),
                    WordTimestamp(word="palavra3", start=2.0, end=3.0),
                ],
            ),
            TranscriptionSegment(
                start=10.0,
                end=11.0,
                text="fora",
                words=[WordTimestamp(word="fora", start=10.0, end=11.0)],
            ),
        ],
    )
    words = slice_transcript_for_clip(transcript, start_time=0.5, end_time=2.5)
    assert [w.word for w in words] == ["palavra1", "palavra2", "palavra3"]


def test_slice_transcript_invalid_range_returns_empty(tmp_path: Path):
    transcript = TranscriptionResult(
        language="pt", source_path=tmp_path / "x.mp4", segments=[],
    )
    assert slice_transcript_for_clip(transcript, 5.0, 5.0) == []
    assert slice_transcript_for_clip(transcript, 5.0, 3.0) == []
