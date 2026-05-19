"""Testes de youcut.players.catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from youcut.players.catalog import (
    _infer_aliases_from_slug,
    _normalize,
    _slug_to_display_name,
    load_catalog,
)


def test_normalize_strips_accents_and_lowercases():
    assert _normalize("Vinícius Júnior") == "vinicius junior"
    assert _normalize("  NEYMAR  Jr  ") == "neymar jr"
    assert _normalize("") == ""


def test_slug_to_display_name_capitalizes_tokens():
    assert _slug_to_display_name("vinicius_junior") == "Vinicius Junior"
    assert _slug_to_display_name("neymar") == "Neymar"


def test_slug_to_display_name_handles_club_hints():
    # Sufixo de clube vai pra entre parênteses.
    assert _slug_to_display_name("danilo_botafogo") == "Danilo (Botafogo)"
    assert _slug_to_display_name("danilo_flamengo") == "Danilo (Flamengo)"


def test_infer_aliases_basic():
    aliases = _infer_aliases_from_slug("vinicius_junior")
    assert "vinicius junior" in aliases
    assert "vinicius" in aliases


def test_infer_aliases_strips_club_suffix():
    # Quando há sufixo de clube, ele é removido — só sobra o nome curto.
    aliases = _infer_aliases_from_slug("danilo_botafogo")
    assert aliases == ["danilo"]


def test_infer_aliases_single_token():
    assert _infer_aliases_from_slug("casemiro") == ["casemiro"]


def test_load_catalog_empty_dir_returns_empty(tmp_path: Path):
    catalog = load_catalog(tmp_path)
    assert catalog.profiles == []
    assert catalog.alias_index == {}


def test_load_catalog_nonexistent_dir_returns_empty(tmp_path: Path):
    catalog = load_catalog(tmp_path / "does-not-exist")
    assert catalog.profiles == []


def test_load_catalog_ignores_non_image_files(tmp_path: Path):
    (tmp_path / "neymar.jpg").write_bytes(b"fake")
    (tmp_path / "README.txt").write_text("not an image")
    (tmp_path / "_download.py").write_text("# script")
    catalog = load_catalog(tmp_path)
    slugs = [p.slug for p in catalog.profiles]
    assert slugs == ["neymar"]


def test_load_catalog_picks_up_supported_extensions(tmp_path: Path):
    for name in ("a.jpg", "b.jpeg", "c.png", "d.webp"):
        (tmp_path / name).write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    assert {p.slug for p in catalog.profiles} == {"a", "b", "c", "d"}


def test_load_catalog_uses_aliases_override(tmp_path: Path):
    (tmp_path / "vinicius_junior.jpg").write_bytes(b"fake")
    overrides = {"vinicius_junior": ["vini", "vini jr"]}
    (tmp_path / "aliases.json").write_text(json.dumps(overrides))
    catalog = load_catalog(tmp_path)
    profile = catalog.get("vinicius_junior")
    assert profile is not None
    assert "vini" in profile.aliases
    assert "vini jr" in profile.aliases


def test_load_catalog_alias_index_groups_ambiguous(tmp_path: Path):
    (tmp_path / "danilo_botafogo.jpg").write_bytes(b"fake")
    (tmp_path / "danilo_flamengo.jpg").write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    # Ambos têm alias "danilo" (sufixo de clube descartado)
    matches = catalog.alias_index.get("danilo", [])
    slugs = sorted(p.slug for p in matches)
    assert slugs == ["danilo_botafogo", "danilo_flamengo"]


def test_load_catalog_max_alias_tokens_reflects_longest(tmp_path: Path):
    (tmp_path / "neymar.jpg").write_bytes(b"fake")
    (tmp_path / "vinicius_junior.jpg").write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    assert catalog.max_alias_tokens == 2


def test_load_catalog_invalid_aliases_json_is_ignored(tmp_path: Path):
    (tmp_path / "neymar.jpg").write_bytes(b"fake")
    (tmp_path / "aliases.json").write_text("not valid json {")
    catalog = load_catalog(tmp_path)
    # Mesmo com aliases.json quebrado, o profile entra com aliases inferidos.
    assert "neymar" in catalog.alias_index
