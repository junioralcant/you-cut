from youcut.decoupage import (
    build_select_expr,
    compute_keep_ranges,
    parse_silencedetect,
)


def test_parse_silencedetect_pairs_starts_and_ends():
    stderr = """\
[silencedetect @ 0x1] silence_start: 1.5
[silencedetect @ 0x1] silence_end: 2.0 | silence_duration: 0.5
[silencedetect @ 0x1] silence_start: 4.2
[silencedetect @ 0x1] silence_end: 4.8 | silence_duration: 0.6
"""
    assert parse_silencedetect(stderr) == [(1.5, 2.0), (4.2, 4.8)]


def test_parse_silencedetect_drops_invalid_pairs():
    # end <= start deve ser ignorado
    stderr = """\
silence_start: 5.0
silence_end: 3.0
silence_start: 7.0
silence_end: 8.0
"""
    assert parse_silencedetect(stderr) == [(7.0, 8.0)]


def test_parse_silencedetect_empty():
    assert parse_silencedetect("") == []


def test_compute_keep_ranges_no_silence_returns_full_clip():
    assert compute_keep_ranges([], duration=10.0, keep_padding=0.05) == [(0.0, 10.0)]


def test_compute_keep_ranges_inverts_silences_with_padding():
    silences = [(1.0, 2.0), (5.0, 6.0)]
    keeps = compute_keep_ranges(silences, duration=10.0, keep_padding=0.1)
    # cada keep recebe +0.1 nas bordas (clamped no início/fim)
    assert keeps == [(0.0, 1.1), (1.9, 5.1), (5.9, 10.0)]


def test_compute_keep_ranges_merges_overlap_after_padding():
    # padding grande faz keeps consecutivos colidirem -> merge
    silences = [(1.0, 1.05)]
    keeps = compute_keep_ranges(silences, duration=10.0, keep_padding=0.5)
    assert keeps == [(0.0, 10.0)]


def test_compute_keep_ranges_drops_micro_segments():
    # silêncio cobre quase todo o clipe — só sobra um pedacinho < 50ms
    silences = [(0.0, 4.99), (5.0, 10.0)]
    keeps = compute_keep_ranges(silences, duration=10.0, keep_padding=0.0)
    assert keeps == []


def test_build_select_expr_concatenates_betweens():
    expr = build_select_expr([(0.0, 1.0), (2.0, 3.5)])
    assert expr == "between(t,0.000,1.000)+between(t,2.000,3.500)"


def test_build_select_expr_empty_returns_zero():
    assert build_select_expr([]) == "0"
