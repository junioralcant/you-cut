"""Testes do snap de transições de painel para pausas naturais da fala."""

from pathlib import Path

import pytest

from youcut.comic.script_planner import (
    _extract_speech_gaps,
    _snap_panel_boundaries,
    _snap_to_nearest_gap,
)
from youcut.models import (
    Panel,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


def _t(words: list[tuple[str, float, float]]) -> TranscriptionResult:
    """Helper: cria TranscriptionResult com 1 segment cobrindo todos os words."""
    word_objs = [WordTimestamp(word=w, start=s, end=e) for w, s, e in words]
    seg_text = " ".join(w for w, _, _ in words)
    seg_start = words[0][1] if words else 0.0
    seg_end = words[-1][2] if words else 0.0
    seg = TranscriptionSegment(
        start=seg_start, end=seg_end, text=seg_text, words=word_objs
    )
    return TranscriptionResult(segments=[seg], language="pt", source_path=Path("/tmp/x.mp4"))


def _panel(idx: int, start: float, end: float, participants=None) -> Panel:
    return Panel(
        index=idx,
        start_time=start,
        end_time=end,
        participants=participants or ["x"],
        framing="medium",
        scene="cena",
        pose_description="pose",
        panel_seconds_target=end - start,
    )


# ---------------------------------------------------------------------------
# _extract_speech_gaps
# ---------------------------------------------------------------------------


def test_extract_gaps_returns_empty_for_continuous_speech():
    transcription = _t([("ola", 0.0, 0.5), ("mundo", 0.55, 1.0)])
    gaps = _extract_speech_gaps(transcription)
    assert gaps == []  # gap de 0.05s < 0.15s default


def test_extract_gaps_finds_pauses():
    transcription = _t([
        ("primeira", 0.0, 0.5),
        ("frase", 0.55, 1.0),
        ("nova", 1.5, 2.0),  # gap de 0.5s
        ("frase", 2.05, 2.5),
    ])
    gaps = _extract_speech_gaps(transcription)
    assert gaps == [(1.0, 1.5)]


def test_extract_gaps_respects_min_threshold():
    transcription = _t([
        ("a", 0.0, 0.3),
        ("b", 0.35, 0.7),  # gap 0.05s (abaixo do threshold default 0.10s)
        ("c", 1.0, 1.3),   # gap 0.3s
    ])
    gaps = _extract_speech_gaps(transcription)
    assert gaps == [(0.7, 1.0)]


# ---------------------------------------------------------------------------
# _snap_to_nearest_gap
# ---------------------------------------------------------------------------


def test_snap_returns_midpoint_when_target_inside_gap():
    gaps = [(1.0, 2.0)]
    # target dentro do gap → midpoint = 1.5
    assert _snap_to_nearest_gap(1.5, gaps) == 1.5


def test_snap_to_midpoint_when_target_after_gap():
    gaps = [(1.0, 2.0)]
    # target 2.3 → distância 0.3 da edge 2.0; snap pro midpoint 1.5
    assert _snap_to_nearest_gap(2.3, gaps, window=0.4) == 1.5


def test_snap_to_midpoint_when_target_before_gap():
    gaps = [(1.0, 2.0)]
    # target 0.7 → distância 0.3 da edge 1.0; snap pro midpoint 1.5
    assert _snap_to_nearest_gap(0.7, gaps, window=0.4) == 1.5


def test_snap_returns_none_when_no_gap_in_window():
    gaps = [(1.0, 2.0)]
    # target 5.0 → mais próxima edge é 2.0 a 3.0s, fora da janela
    assert _snap_to_nearest_gap(5.0, gaps, window=0.4) is None


def test_snap_picks_closest_of_multiple_gaps():
    gaps = [(1.0, 1.2), (3.0, 3.2)]
    # target 1.4 → gap1 mais próximo (0.2 vs 1.6). Midpoint do gap1 = 1.1
    assert _snap_to_nearest_gap(1.4, gaps, window=0.4) == 1.1


def test_snap_prefers_larger_gap_on_distance_tie():
    # Dois gaps equidistantes: target 2.0, gap pequeno (1.5,1.7) e grande (2.3,2.9)
    # Distância à edge: gap pequeno 0.3, gap grande 0.3 (empate)
    # Vence o maior → midpoint do gap grande = 2.6
    gaps = [(1.5, 1.7), (2.3, 2.9)]
    assert _snap_to_nearest_gap(2.0, gaps, window=0.5) == pytest.approx(2.6)


# ---------------------------------------------------------------------------
# _snap_panel_boundaries
# ---------------------------------------------------------------------------


def test_snap_boundaries_aligns_to_pause_midpoint():
    transcription = _t([
        ("a", 0.0, 1.0),
        ("b", 1.2, 2.0),
        ("c", 2.5, 3.0),  # gap de 0.5s entre 2.0 e 2.5 → midpoint 2.25
        ("d", 3.0, 4.0),
        ("e", 4.0, 5.0),
    ])
    panels = [_panel(0, 0.0, 2.2), _panel(1, 2.2, 5.0)]
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=1.5, max_panel=5.0, audio_duration=5.0
    )
    # 2.2 cai dentro do gap (2.0, 2.5) → snap pro midpoint 2.25
    assert snapped[0].end_time == 2.25
    assert snapped[1].start_time == 2.25


def test_snap_boundaries_moves_when_target_just_outside_gap():
    transcription = _t([
        ("a", 0.0, 1.0),
        ("b", 1.2, 2.0),
        ("c", 2.5, 3.0),  # gap (2.0, 2.5) → midpoint 2.25
        ("d", 3.0, 4.0),
    ])
    panels = [_panel(0, 0.0, 2.7), _panel(1, 2.7, 4.0)]
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=1.0, max_panel=5.0, audio_duration=4.0
    )
    # 2.7 → distância 0.2 da edge 2.5; snap pro midpoint 2.25
    assert snapped[0].end_time == 2.25
    assert snapped[1].start_time == 2.25


def test_snap_skips_when_resulting_duration_violates_constraints():
    transcription = _t([
        ("a", 0.0, 0.5),
        ("b", 0.6, 1.0),
        ("c", 3.0, 3.5),  # gap enorme (1.0, 3.0)
    ])
    panels = [_panel(0, 0.0, 2.0), _panel(1, 2.0, 3.5)]
    # Snap pra 1.0 deixaria painel 0 com 1.0s (< min_panel=2.0)
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=2.0, max_panel=5.0, audio_duration=3.5
    )
    assert snapped[0].end_time == 2.0  # mantém original
    assert snapped[1].start_time == 2.0


def test_snap_preserves_first_start_and_last_end():
    transcription = _t([
        ("a", 0.0, 1.0),
        ("b", 1.5, 2.0),
        ("c", 2.5, 3.0),
    ])
    panels = [_panel(0, 0.0, 2.2), _panel(1, 2.2, 3.0)]
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=0.5, max_panel=5.0, audio_duration=3.0
    )
    assert snapped[0].start_time == 0.0  # 1º painel sempre começa em 0
    assert snapped[-1].end_time == 3.0  # último painel sempre termina em audio_dur


def test_snap_with_single_panel_is_noop():
    transcription = _t([("a", 0.0, 1.0)])
    panels = [_panel(0, 0.0, 1.0)]
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=0.5, max_panel=5.0, audio_duration=1.0
    )
    assert snapped == panels


def test_snap_no_gaps_returns_original():
    transcription = _t([("a", 0.0, 1.0), ("b", 1.0, 2.0)])  # sem pausas
    panels = [_panel(0, 0.0, 1.0), _panel(1, 1.0, 2.0)]
    snapped = _snap_panel_boundaries(
        panels, transcription, min_panel=0.5, max_panel=5.0, audio_duration=2.0
    )
    assert [(p.start_time, p.end_time) for p in snapped] == [(0.0, 1.0), (1.0, 2.0)]
