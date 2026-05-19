"""Testes de youcut.players.disambiguator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from youcut.models import WordTimestamp
from youcut.players.catalog import load_catalog
from youcut.players.disambiguator import disambiguate_mentions
from youcut.players.models import PlayerMention


def _populate(tmp_path: Path, slugs: list[str]) -> Path:
    for slug in slugs:
        (tmp_path / f"{slug}.jpg").write_bytes(b"fake")
    return tmp_path


def _make_words(words: list[str]) -> list[WordTimestamp]:
    return [WordTimestamp(word=w, start=float(i), end=float(i + 1)) for i, w in enumerate(words)]


class _FakeClaude:
    """Stub do anthropic.Anthropic client. Retorna a slug configurada.

    Útil pra evitar chamadas reais à API nos testes.
    """

    def __init__(self, slug: str | None):
        self._slug = slug
        self.messages = SimpleNamespace(create=self._create)
        self.call_count = 0

    def _create(self, model, max_tokens, messages):
        self.call_count += 1
        payload = "{\"slug\": null}" if self._slug is None else f'{{"slug": "{self._slug}"}}'
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])


def test_disambiguate_no_conflict_passthrough(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    neymar = catalog.get("neymar")
    assert neymar is not None
    mention = PlayerMention(profile=neymar, alias_hit="neymar", start=1.0, end=2.0)
    result = disambiguate_mentions([mention], _make_words(["o", "Neymar", "marcou"]), None, "claude")
    assert len(result) == 1
    assert result[0].profile.slug == "neymar"


def test_disambiguate_ambiguous_without_claude_drops_mention(tmp_path: Path):
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo"])
    catalog = load_catalog(tmp_path)
    pa = catalog.get("danilo_botafogo")
    pb = catalog.get("danilo_flamengo")
    assert pa and pb
    mentions = [
        PlayerMention(profile=pa, alias_hit="danilo", start=2.0, end=3.0),
        PlayerMention(profile=pb, alias_hit="danilo", start=2.0, end=3.0),
    ]
    result = disambiguate_mentions(mentions, _make_words(["e", "Danilo", "cobrou"]), None, "claude")
    # Política conservadora: sem Claude, descarta o ambíguo (preferimos não
    # exibir a foto a exibir a errada).
    assert result == []


def test_disambiguate_ambiguous_with_claude_picks_chosen(tmp_path: Path):
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo"])
    catalog = load_catalog(tmp_path)
    pa = catalog.get("danilo_botafogo")
    pb = catalog.get("danilo_flamengo")
    assert pa and pb
    mentions = [
        PlayerMention(profile=pa, alias_hit="danilo", start=2.0, end=3.0),
        PlayerMention(profile=pb, alias_hit="danilo", start=2.0, end=3.0),
    ]
    fake_claude = _FakeClaude("danilo_flamengo")
    result = disambiguate_mentions(mentions, _make_words(["e", "Danilo", "cobrou"]), fake_claude, "claude-x")
    assert len(result) == 1
    assert result[0].profile.slug == "danilo_flamengo"
    assert fake_claude.call_count == 1


def test_disambiguate_claude_returns_null_drops_mention(tmp_path: Path):
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo"])
    catalog = load_catalog(tmp_path)
    pa = catalog.get("danilo_botafogo")
    pb = catalog.get("danilo_flamengo")
    assert pa and pb
    mentions = [
        PlayerMention(profile=pa, alias_hit="danilo", start=2.0, end=3.0),
        PlayerMention(profile=pb, alias_hit="danilo", start=2.0, end=3.0),
    ]
    fake_claude = _FakeClaude(None)
    result = disambiguate_mentions(mentions, _make_words(["e", "Danilo", "cobrou"]), fake_claude, "claude")
    assert result == []


def test_disambiguate_deduplicates_by_slug(tmp_path: Path):
    _populate(tmp_path, ["neymar"])
    catalog = load_catalog(tmp_path)
    neymar = catalog.get("neymar")
    assert neymar
    # Mesmo jogador citado 3x em momentos diferentes → 1 menção (a primeira).
    mentions = [
        PlayerMention(profile=neymar, alias_hit="neymar", start=10.0, end=11.0),
        PlayerMention(profile=neymar, alias_hit="neymar", start=2.0, end=3.0),
        PlayerMention(profile=neymar, alias_hit="neymar", start=20.0, end=21.0),
    ]
    result = disambiguate_mentions(mentions, _make_words(["x"]), None, "claude")
    assert len(result) == 1
    assert result[0].start == 2.0  # ordenação temporal: pega a primeira


def test_disambiguate_claude_invalid_slug_drops(tmp_path: Path):
    _populate(tmp_path, ["danilo_botafogo", "danilo_flamengo"])
    catalog = load_catalog(tmp_path)
    pa = catalog.get("danilo_botafogo")
    pb = catalog.get("danilo_flamengo")
    assert pa and pb
    mentions = [
        PlayerMention(profile=pa, alias_hit="danilo", start=2.0, end=3.0),
        PlayerMention(profile=pb, alias_hit="danilo", start=2.0, end=3.0),
    ]
    # Claude responde slug que NÃO está nos candidatos — descarta.
    fake_claude = _FakeClaude("messi")
    result = disambiguate_mentions(mentions, _make_words(["x"]), fake_claude, "claude")
    assert result == []
