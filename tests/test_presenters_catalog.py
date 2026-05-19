"""Testes de youcut.presenters.catalog."""

from __future__ import annotations

from pathlib import Path

from youcut.presenters.catalog import (
    _slug_to_display_name,
    load_catalog,
)


def test_slug_to_display_name():
    assert _slug_to_display_name("tiago_leifert") == "Tiago Leifert"
    assert _slug_to_display_name("ana_paula_padrao") == "Ana Paula Padrao"


def test_load_catalog_empty_dir(tmp_path: Path):
    catalog = load_catalog(tmp_path)
    assert len(catalog) == 0
    assert catalog.profiles == []


def test_load_catalog_nonexistent_dir(tmp_path: Path):
    catalog = load_catalog(tmp_path / "missing")
    assert len(catalog) == 0


def test_load_catalog_ignores_non_image(tmp_path: Path):
    (tmp_path / "tiago_leifert.webp").write_bytes(b"fake")
    (tmp_path / "README.md").write_text("not an image")
    catalog = load_catalog(tmp_path)
    assert len(catalog) == 1
    assert catalog.profiles[0].slug == "tiago_leifert"


def test_load_catalog_supports_all_image_extensions(tmp_path: Path):
    for name in ("a.jpg", "b.jpeg", "c.png", "d.webp"):
        (tmp_path / name).write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    assert {p.slug for p in catalog.profiles} == {"a", "b", "c", "d"}


def test_catalog_get_is_case_insensitive(tmp_path: Path):
    (tmp_path / "Tiago_Leifert.webp").write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    # `get` aceita qualquer caixa — armazena lowercase.
    assert catalog.get("tiago_leifert") is not None
    assert catalog.get("TIAGO_LEIFERT") is not None
    assert catalog.get("nao_existe") is None


def test_load_catalog_sets_display_name(tmp_path: Path):
    (tmp_path / "tiago_leifert.webp").write_bytes(b"fake")
    catalog = load_catalog(tmp_path)
    profile = catalog.get("tiago_leifert")
    assert profile is not None
    assert profile.display_name == "Tiago Leifert"
