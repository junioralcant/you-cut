from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._fakes.comic_providers import FakeImageProvider
from youcut.comic.cast_builder import (
    _build_anchor_prompt,
    _build_text_card,
    build_cast,
)
from youcut.comic.providers.images import ImageGenerationError
from youcut.models import CastMember


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def _person(**overrides) -> CastMember:
    base = dict(
        character_id="person_1",
        kind="person",
        gender_apparent="feminino",
        age_apparent="adulta jovem",
        hair="cabelo castanho ondulado",
        facial_hair="",
        skin="pele clara",
        clothing="camiseta branca e jaqueta jeans",
        accessories=["brincos pequenos"],
        narrative_role="narradora",
    )
    base.update(overrides)
    return CastMember(**base)


def test_text_card_includes_descriptors():
    card = _build_text_card(_person())
    assert "feminino" in card
    assert "cabelo: cabelo castanho" in card
    assert "roupa: camiseta branca" in card
    assert "acessórios: brincos pequenos" in card
    assert "papel narrativo: narradora" in card


def test_text_card_omits_empty_fields():
    card = _build_text_card(
        CastMember(character_id="x", kind="person", narrative_role="figurante")
    )
    assert "papel narrativo: figurante" in card
    assert "roupa" not in card


def test_anchor_prompt_for_person_includes_descriptors():
    prompt = _build_anchor_prompt(_person())
    assert "Personagem único em pose neutra" in prompt
    assert "feminino" in prompt
    assert "cabelo castanho" in prompt
    assert "vestindo camiseta branca" in prompt
    assert "caricatura editorial" in prompt
    assert "Sem fotorrealismo, sem texto embutido" in prompt


def test_anchor_prompt_for_animal_uses_animal_template():
    member = CastMember(character_id="dog", kind="animal", narrative_role="cachorro")
    prompt = _build_anchor_prompt(member)
    assert "Animal ilustrado" in prompt
    assert "cachorro" in prompt


def test_anchor_prompt_for_object_uses_object_template():
    member = CastMember(character_id="moto", kind="object", narrative_role="moto vermelha")
    prompt = _build_anchor_prompt(member)
    assert "Objeto narrativo" in prompt
    assert "moto vermelha" in prompt


def test_build_cast_writes_one_png_per_person(tmp_path, config):
    fake = FakeImageProvider()
    cast = [_person(), _person(character_id="person_2", narrative_role="amigo")]

    out = build_cast(cast, tmp_path, config, image_provider=fake)

    assert len(out) == 2
    cast_dir = tmp_path / "comic" / "cast"
    for member in out:
        assert member.anchor_image_path is not None
        assert member.anchor_image_path.exists()
        assert member.anchor_image_path.parent == cast_dir
        assert member.anchor_image_path.read_bytes().startswith(b"\x89PNG")
    assert len(fake.calls) == 2


def test_build_cast_is_idempotent_on_existing_files(tmp_path, config):
    fake = FakeImageProvider()
    cast = [_person()]
    first = build_cast(cast, tmp_path, config, image_provider=fake)
    assert len(fake.calls) == 1

    fake2 = FakeImageProvider()
    second = build_cast(first, tmp_path, config, image_provider=fake2)

    assert len(fake2.calls) == 0
    assert second[0].anchor_image_path == first[0].anchor_image_path


def test_build_cast_overrides_existing_text_card_only_when_empty(tmp_path, config):
    fake = FakeImageProvider()
    member = _person()
    member = member.model_copy(update={"text_card": "ficha pré-existente"})

    out = build_cast([member], tmp_path, config, image_provider=fake)

    assert out[0].text_card == "ficha pré-existente"


def test_build_cast_supports_animal_and_object(tmp_path, config):
    fake = FakeImageProvider()
    cast = [
        CastMember(character_id="dog", kind="animal", narrative_role="cachorro"),
        CastMember(character_id="moto", kind="object", narrative_role="moto vermelha"),
    ]

    out = build_cast(cast, tmp_path, config, image_provider=fake)

    assert len(out) == 2
    assert all(m.anchor_image_path is not None for m in out)
    assert {m.kind for m in out} == {"animal", "object"}
    prompts = [c["prompt"] for c in fake.calls]
    assert any("Animal" in p for p in prompts)
    assert any("Objeto" in p for p in prompts)


def test_build_cast_raises_on_provider_failure(tmp_path, config):
    bad_provider = MagicMock()
    bad_provider.generate.side_effect = ImageGenerationError("api down")

    with pytest.raises(ImageGenerationError, match=r"api down"):
        build_cast([_person()], tmp_path, config, image_provider=bad_provider)


def test_build_cast_raises_on_empty_bytes(tmp_path, config):
    bad_provider = MagicMock()
    bad_provider.generate.return_value = b""

    with pytest.raises(ImageGenerationError, match=r"bytes vazios"):
        build_cast([_person()], tmp_path, config, image_provider=bad_provider)


def test_build_cast_wraps_unexpected_exception(tmp_path, config):
    bad_provider = MagicMock()
    bad_provider.generate.side_effect = RuntimeError("network glitch")

    with pytest.raises(ImageGenerationError, match=r"Falha ao gerar ficha-âncora"):
        build_cast([_person()], tmp_path, config, image_provider=bad_provider)


def test_anchor_image_path_filename_uses_character_id(tmp_path, config):
    fake = FakeImageProvider()
    member = _person(character_id="meu_slug_unico")
    out = build_cast([member], tmp_path, config, image_provider=fake)
    assert out[0].anchor_image_path.name == "meu_slug_unico.png"


def test_build_cast_uses_default_size_1024_square(tmp_path, config):
    fake = FakeImageProvider()
    build_cast([_person()], tmp_path, config, image_provider=fake)
    assert fake.calls[0]["size"] == (1024, 1024)
