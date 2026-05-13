"""Smoke tests do preset visual 'motivacao'.

Cobre:
- captioner: estilo word_serif_italic e geração de eventos com/sem handle
- color_filter: presença do preset motivacao_lilac
- motivacao: builders de filter-graph para badge

Integração ffmpeg (badge real + outro real) fica em testes com marker
`integration` — esses só validam contrato/strings.
"""
from __future__ import annotations

import os

import pytest

from youcut.captioner import (
    MOTIVACAO_ITALIC_FONT_FILE,
    MOTIVACAO_ITALIC_FONT_NAME,
    _generate_word_serif_italic_events,
    _word_serif_italic_styles,
)
from youcut.color_filter import VALID_PRESETS, get_filter_chain
from youcut.models import WordTimestamp
from youcut.motivacao import build_badge_filtergraph


os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")


def _words(*pairs: tuple[str, float, float]) -> list[WordTimestamp]:
    return [WordTimestamp(word=w, start=s, end=e) for w, s, e in pairs]


# ── captioner ──────────────────────────────────────────────────────────────


def test_word_serif_italic_styles_emits_two_named_styles():
    block = _word_serif_italic_styles()
    assert "Style: Default," in block
    assert "Style: Handle," in block
    assert block.count("\n") == 1  # exatamente 2 linhas, 1 quebra


def test_word_serif_italic_styles_uses_motivacao_font_with_italic_on():
    block = _word_serif_italic_styles()
    assert MOTIVACAO_ITALIC_FONT_NAME in block
    # Bold=0, Italic=-1 (campos 8 e 9 do format ASS V4+)
    for line in block.splitlines():
        # Confirmamos só que a sequência ",0,-1," aparece (Bold,Italic).
        assert ",0,-1," in line, f"Italic não está habilitado: {line}"


def test_motivacao_italic_font_file_exists():
    assert MOTIVACAO_ITALIC_FONT_FILE.exists(), (
        f"Esperado em {MOTIVACAO_ITALIC_FONT_FILE}. "
        "Baixar via Google Fonts e versionar em youcut/assets/fonts/."
    )


def test_events_group_close_words_into_chunks():
    # 3 palavras coladas (gaps pequenos) -> 1 chunk com as 3
    words = _words(("ola", 1.0, 1.2), ("mundo", 1.25, 1.5), ("legal", 1.55, 1.8))
    out = _generate_word_serif_italic_events(words, offset=0.0, handle=None)
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 1, f"esperava 1 chunk, vi {len(lines)}: {lines}"
    assert "ola mundo legal" in lines[0]
    assert "\\pos(540,964)" in lines[0]


def test_events_break_chunk_on_long_gap():
    # gap >0.35s entre palavras força quebra
    words = _words(("primeira", 0.0, 0.4), ("segunda", 1.0, 1.4))
    out = _generate_word_serif_italic_events(words, offset=0.0, handle=None)
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 2
    assert any("primeira" in ln and "segunda" not in ln for ln in lines)


def test_events_with_handle_emit_single_persistent_handle():
    """Handle deve ser ÚNICO e persistente (não um por chunk)."""
    words = _words(("foco", 0.5, 0.7), ("total", 0.75, 0.95))
    out = _generate_word_serif_italic_events(words, offset=0.0, handle="meucanal")
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    default_lines = [ln for ln in lines if ",Default," in ln]
    handle_lines = [ln for ln in lines if ",Handle," in ln]
    assert len(default_lines) == 1, default_lines
    assert len(handle_lines) == 1, handle_lines
    # handle cobre toda a duração possível (start=0, end sentinela longa)
    assert "0:00:00.00,9:59:59.99" in handle_lines[0]
    assert "\\pos(540,1044)" in handle_lines[0]
    assert "@meucanal" in handle_lines[0]


def test_handle_persistent_independent_of_chunk_count():
    """N chunks → ainda 1 só Handle."""
    words = _words(
        ("um", 0.0, 0.2),
        ("dois", 1.0, 1.2),   # gap > 0.35 -> quebra
        ("três", 2.0, 2.2),
        ("quatro", 3.0, 3.2),
    )
    out = _generate_word_serif_italic_events(words, offset=0.0, handle="meucanal")
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    default_lines = [ln for ln in lines if ",Default," in ln]
    handle_lines = [ln for ln in lines if ",Handle," in ln]
    assert len(default_lines) == 4
    assert len(handle_lines) == 1


def test_events_max_3_words_per_chunk():
    # 5 palavras coladas (gaps zero) -> 2 chunks (3 + 2)
    words = _words(
        ("a", 0.0, 0.1), ("b", 0.1, 0.2), ("c", 0.2, 0.3),
        ("d", 0.3, 0.4), ("e", 0.4, 0.5),
    )
    out = _generate_word_serif_italic_events(words, offset=0.0, handle=None)
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 2
    assert "a b c" in lines[0]
    assert "d e" in lines[1]


def test_handle_with_leading_at_is_normalized():
    words = _words(("hi", 0.0, 0.3))
    out = _generate_word_serif_italic_events(words, offset=0.0, handle="@meucanal")
    assert "@meucanal" in out
    assert "@@meucanal" not in out


def test_offset_subtracted_from_timestamps():
    words = _words(("a", 10.0, 10.4))
    out = _generate_word_serif_italic_events(words, offset=9.5, handle=None)
    # 10.0 - 9.5 = 0.5s -> 0:00:00.50
    assert "0:00:00.50" in out


# ── color filter ───────────────────────────────────────────────────────────


def test_motivacao_lilac_preset_registered():
    assert "motivacao_lilac" in VALID_PRESETS


def test_motivacao_lilac_chain_has_signature_components():
    chain = get_filter_chain("motivacao_lilac")
    # componentes da assinatura: eq + curves + colorbalance + vignette
    for marker in ("eq=", "curves=", "colorbalance=", "vignette="):
        assert marker in chain


# ── motivacao filter graphs ────────────────────────────────────────────────


def test_badge_filtergraph_uppercases_handle():
    fg = build_badge_filtergraph("meucanal")
    assert "@MEUCANAL" in fg
    assert "@meucanal" not in fg


def test_badge_filtergraph_overlays_icon_then_drawtext():
    fg = build_badge_filtergraph("x")
    # icon vem antes do drawtext (overlay precede drawtext)
    assert fg.index("overlay=") < fg.index("drawtext=")
    # ambos ancorados na borda esquerda + bottom
    assert "[vbadge]" in fg


def test_badge_filtergraph_escapes_colons_in_drawtext():
    # handles não devem ter ':' mas se vierem, escape evita quebrar parser
    fg = build_badge_filtergraph("user:foo")
    assert r"\:" in fg
