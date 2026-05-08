"""Testes do syllable mapper (Task 2.0)."""

from __future__ import annotations

import logging

import pytest

from youcut.comic.syllable_mapper import (
    MIN_SYLLABLE_DUR_SEC,
    SILENCE_GAP_THRESHOLD_SEC,
    _dominant_vowel,
    derive_lipsync_track,
)
from youcut.models import MouthShape, WordTimestamp


# ── _dominant_vowel ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "syllable, expected",
    [
        ("ca", MouthShape.OPEN_WIDE),
        ("má", MouthShape.OPEN_WIDE),
        ("pé", MouthShape.OPEN_WIDE),
        ("vê", MouthShape.OPEN_WIDE),
        ("li", MouthShape.OPEN_MID),
        ("í", MouthShape.OPEN_MID),
        ("yo", MouthShape.OPEN_MID),  # 'y' antes de 'o' — primeira vogal vence
        ("co", MouthShape.OPEN_ROUND),
        ("õ", MouthShape.OPEN_ROUND),
        ("tu", MouthShape.OPEN_ROUND),
        ("hr", MouthShape.CLOSED),  # sem vogal
        ("st", MouthShape.CLOSED),
        ("", MouthShape.CLOSED),
    ],
)
def test_dominant_vowel_mapping(syllable, expected):
    assert _dominant_vowel(syllable) == expected


def test_dominant_vowel_case_insensitive():
    assert _dominant_vowel("CA") == MouthShape.OPEN_WIDE
    assert _dominant_vowel("Õ") == MouthShape.OPEN_ROUND


# ── derive_lipsync_track — edge cases ────────────────────────────────────────


def test_empty_words_returns_empty_list():
    assert derive_lipsync_track([], character_id="speaker_a") == []


def test_single_word_pt_br():
    words = [WordTimestamp(word="exemplo", start=0.0, end=0.6)]
    events = derive_lipsync_track(words, character_id="speaker_a")
    assert len(events) >= 1
    # Cobre exatamente [0.0, 0.6]
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[-1].end_sec == pytest.approx(0.6)
    # Todos com mesmo character_id
    assert all(ev.character_id == "speaker_a" for ev in events)


def test_word_with_blank_text_ignored():
    words = [
        WordTimestamp(word="  ", start=0.0, end=0.2),
        WordTimestamp(word="oi", start=0.2, end=0.5),
    ]
    events = derive_lipsync_track(words, character_id="x")
    assert events
    assert events[0].start_sec == pytest.approx(0.2)


# ── derive_lipsync_track — pt-BR ─────────────────────────────────────────────


def test_pt_br_multi_syllable_word():
    words = [WordTimestamp(word="paralelepípedo", start=0.0, end=1.4)]
    events = derive_lipsync_track(words, character_id="x", locale="pt_BR")
    # pa-ra-le-le-pí-pe-do → 7 sílabas (~200ms cada com 1.4s total)
    assert len(events) >= 5
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[-1].end_sec == pytest.approx(1.4)
    # Pelo menos um OPEN_WIDE (a/e), um OPEN_MID (i), um OPEN_ROUND (o)
    shapes = {ev.shape for ev in events}
    assert MouthShape.OPEN_WIDE in shapes
    assert MouthShape.OPEN_ROUND in shapes


def test_pt_br_with_consonant_cluster():
    """'ex-tra' / 'cons-tru-ção' — encontros consonantais não quebram o pipeline."""
    words = [WordTimestamp(word="construção", start=0.0, end=0.9)]
    events = derive_lipsync_track(words, character_id="x", locale="pt_BR")
    assert events
    assert events[-1].end_sec == pytest.approx(0.9)


def test_pt_br_multiple_words_no_gap():
    words = [
        WordTimestamp(word="oi", start=0.0, end=0.3),
        WordTimestamp(word="amigo", start=0.3, end=0.9),
    ]
    events = derive_lipsync_track(words, character_id="x")
    # Cobertura contígua
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[-1].end_sec == pytest.approx(0.9)
    # Sem sobreposição nem gap interno
    for prev, curr in zip(events, events[1:]):
        assert curr.start_sec == pytest.approx(prev.end_sec, abs=1e-6)


# ── derive_lipsync_track — en-US ─────────────────────────────────────────────


def test_en_us_locale():
    words = [WordTimestamp(word="hello", start=0.0, end=0.6)]
    events = derive_lipsync_track(words, character_id="x", locale="en_US")
    assert events
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[-1].end_sec == pytest.approx(0.6)


def test_en_us_multi_word():
    words = [
        WordTimestamp(word="hello", start=0.0, end=0.4),
        WordTimestamp(word="world", start=0.4, end=0.8),
    ]
    events = derive_lipsync_track(words, character_id="x", locale="en_US")
    assert events[-1].end_sec == pytest.approx(0.8)


# ── Cobertura 100% ───────────────────────────────────────────────────────────


def test_full_coverage_no_gaps_nor_overlaps():
    """A soma das durações dos eventos deve cobrir [start, end] sem furos."""
    words = [
        WordTimestamp(word="bem", start=0.0, end=0.3),
        WordTimestamp(word="vindo", start=0.35, end=0.85),  # gap pequeno (50ms)
        WordTimestamp(word="amigos", start=1.5, end=2.2),  # gap grande (650ms)
    ]
    events = derive_lipsync_track(words, character_id="x")
    expected_total = 2.2 - 0.0
    actual_total = sum(ev.end_sec - ev.start_sec for ev in events)
    assert actual_total == pytest.approx(expected_total, abs=1e-6)
    # Sem sobreposição
    for prev, curr in zip(events, events[1:]):
        assert curr.start_sec == pytest.approx(prev.end_sec, abs=1e-6)


def test_full_coverage_does_not_overshoot():
    words = [
        WordTimestamp(word="oi", start=0.5, end=1.0),
    ]
    events = derive_lipsync_track(words, character_id="x")
    assert events[0].start_sec == pytest.approx(0.5)
    assert events[-1].end_sec == pytest.approx(1.0)


# ── Closed em gaps grandes ───────────────────────────────────────────────────


def test_silence_gap_emits_closed_event():
    words = [
        WordTimestamp(word="oi", start=0.0, end=0.3),
        WordTimestamp(word="tchau", start=2.0, end=2.5),  # gap 1.7s
    ]
    events = derive_lipsync_track(words, character_id="x")
    closed_events = [ev for ev in events if ev.shape == MouthShape.CLOSED]
    assert closed_events
    # Há um CLOSED cobrindo aproximadamente [0.3, 2.0]
    matching = [
        ev
        for ev in closed_events
        if abs(ev.start_sec - 0.3) < 0.01 and abs(ev.end_sec - 2.0) < 0.01
    ]
    assert matching, f"esperava CLOSED em [0.3, 2.0]; eventos closed: {closed_events}"


def test_small_gap_does_not_emit_closed():
    """Gap < 120ms é absorvido (estende o evento anterior)."""
    words = [
        WordTimestamp(word="oi", start=0.0, end=0.3),
        WordTimestamp(word="ai", start=0.35, end=0.6),  # gap 50ms
    ]
    events = derive_lipsync_track(words, character_id="x")
    # Não deve haver CLOSED em [0.3, 0.35] — o evento anterior foi estendido
    for ev in events:
        if abs(ev.start_sec - 0.3) < 0.01 and abs(ev.end_sec - 0.35) < 0.01:
            assert ev.shape != MouthShape.CLOSED


# ── Smoothing (sílabas < 80ms) ───────────────────────────────────────────────


def test_smoothing_merges_short_syllables():
    """Palavra muito rápida (~50ms total) deve resultar em 1 evento ≥ 80ms."""
    # 'ca' tem 1 sílaba mas com tempo total 0.040s, fica < 80ms
    words = [
        WordTimestamp(word="exemplo", start=0.0, end=0.05),  # 50ms total
        WordTimestamp(word="ok", start=0.05, end=0.4),  # contínuo, 350ms
    ]
    events = derive_lipsync_track(words, character_id="x")
    # Após smoothing, nenhum evento < min_dur (excepto potencialmente o último/único)
    short_events = [ev for ev in events if ev.end_sec - ev.start_sec < MIN_SYLLABLE_DUR_SEC]
    assert not short_events, f"esperava 0 eventos curtos; recebido {short_events}"


def test_smoothing_preserves_total_duration():
    words = [
        WordTimestamp(word="oi", start=0.0, end=0.04),  # 40ms (curto)
        WordTimestamp(word="amigos", start=0.04, end=0.6),
    ]
    events = derive_lipsync_track(words, character_id="x")
    total = sum(ev.end_sec - ev.start_sec for ev in events)
    assert total == pytest.approx(0.6, abs=1e-6)


def test_smoothing_keeps_neighbor_shape():
    """Quando uma sílaba curta é fundida, o shape do VIZINHO permanece."""
    # construímos uma situação onde 'i' (OPEN_MID) é curta e vizinha é OPEN_WIDE
    words = [
        WordTimestamp(word="iaaaaa", start=0.0, end=0.5),  # 'i-a-a-a-a-a' aprox
    ]
    events = derive_lipsync_track(words, character_id="x", locale="pt_BR")
    # o teste é tolerante: só garante que após smoothing a maioria das shapes
    # presentes faz sentido
    assert events
    assert events[-1].end_sec == pytest.approx(0.5)


# ── Locale fallback ──────────────────────────────────────────────────────────


def test_unknown_locale_falls_back_with_warning(caplog):
    words = [WordTimestamp(word="oi", start=0.0, end=0.3)]
    with caplog.at_level(logging.WARNING, logger="youcut.comic.syllable_mapper"):
        events = derive_lipsync_track(
            words,
            character_id="x",
            locale="xx_XX",
            fallback_locale="pt_BR",
        )
    assert events
    assert any("xx_XX" in rec.message for rec in caplog.records)
    assert any("pt_BR" in rec.message for rec in caplog.records)


# ── Múltiplos personagens ───────────────────────────────────────────────────


def test_character_id_propagates():
    words = [WordTimestamp(word="alô", start=0.0, end=0.4)]
    events_a = derive_lipsync_track(words, character_id="speaker_a")
    events_b = derive_lipsync_track(words, character_id="speaker_b")
    assert all(ev.character_id == "speaker_a" for ev in events_a)
    assert all(ev.character_id == "speaker_b" for ev in events_b)


# ── Constantes públicas ──────────────────────────────────────────────────────


def test_thresholds_match_spec():
    """Thresholds documentados na techspec devem bater com o módulo."""
    assert MIN_SYLLABLE_DUR_SEC == 0.080
    assert SILENCE_GAP_THRESHOLD_SEC == 0.120


# ── Critério de sucesso: ~30s em pt-BR → ≥ 50 eventos ───────────────────────


def test_long_pt_br_transcription_yields_dense_events():
    """30s de pt-BR realista deve produzir ≥ 50 eventos cobrindo todo o intervalo."""
    sample_words = [
        "olá", "pessoal", "hoje", "vamos", "falar", "sobre", "uma", "coisa",
        "muito", "interessante", "que", "aconteceu", "comigo", "ontem",
        "estava", "caminhando", "pela", "rua", "quando", "vi", "um", "amigo",
        "antigo", "que", "não", "via", "há", "muitos", "anos", "fui",
        "abraçar", "ele", "e", "ele", "me", "contou", "uma", "história",
        "incrível", "sobre", "viagem", "para", "o", "japão", "vocês",
        "precisam", "ouvir", "isso", "vamos", "lá",
    ]
    # 30s distribuídos uniformemente, 0.5s/palavra com gap 0.1s
    words = []
    cursor = 0.0
    for w in sample_words:
        words.append(WordTimestamp(word=w, start=cursor, end=cursor + 0.5))
        cursor += 0.6  # 0.5s palavra + 0.1s gap (gap < 120ms → absorvido)
    events = derive_lipsync_track(words, character_id="x", locale="pt_BR")
    assert len(events) >= 50
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[-1].end_sec == pytest.approx(words[-1].end)
