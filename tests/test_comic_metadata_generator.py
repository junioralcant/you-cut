"""Testes do gerador de metadados por plataforma."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from youcut.comic.metadata_generator import (
    MetadataGenerationError,
    generate_metadata,
    write_metadata_files,
)
from youcut.models import (
    CastMember,
    ComicMetadata,
    PlatformMetadata,
    TranscriptionResult,
    TranscriptionSegment,
)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    return PipelineConfig()


@pytest.fixture
def transcription():
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0, end=2.0, text="Tudo bem, mano?", words=[]
            ),
            TranscriptionSegment(
                start=2.0, end=4.0, text="Vai comer o quê?", words=[]
            ),
        ],
        language="pt",
        source_path=Path("/tmp/fake.mp4"),
    )


@pytest.fixture
def cast():
    return [
        CastMember(
            character_id="piloto",
            kind="person",
            text_card="motoqueiro de óculos",
            narrative_role="questionador",
        ),
        CastMember(
            character_id="toin",
            kind="person",
            text_card="passageiro distraído",
            narrative_role="responde fora do contexto",
        ),
    ]


def _mock_claude_response(payload: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "generate_platform_metadata"
    block.input = payload
    response = MagicMock()
    response.content = [block]
    return response


def _valid_payload() -> dict:
    return {
        "summary": "Diálogo cômico entre dois amigos numa moto.",
        "tiktok": {
            "title": "Quando o amigo NÃO presta atenção 😂",
            "description": "Marca aquele amigo que sempre viaja na conversa 👇",
            "hashtags": [
                "fyp",
                "foryou",
                "viral",
                "humor",
                "brasileiro",
                "cartoon",
                "moto",
                "comedia",
                "amigos",
                "tiktokbr",
                "engracado",
                "humorbr",
            ],
        },
        "instagram_reels": {
            "title": "O amigo distraído 😂",
            "description": "Tem amigo que viaja em outra dimensão 🛵💨",
            "hashtags": ["reels", "humor", "cartoon", "brasileiro", "comedia"],
        },
        "youtube_shorts": {
            "title": "Quando o amigo não presta atenção - Cartoon",
            "description": (
                "Animação cômica curta com 2 amigos numa moto. Inscreva-se "
                "para mais conteúdo!"
            ),
            "hashtags": ["shorts", "humor", "cartoon", "brasileiro", "comedia"],
        },
    }


def test_generate_metadata_parses_valid_response(config, transcription, cast):
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = _mock_claude_response(_valid_payload())

    metadata = generate_metadata(
        transcription, cast, config, scene_seed="moto no deserto", client=client
    )

    assert isinstance(metadata, ComicMetadata)
    assert metadata.summary.startswith("Diálogo")
    assert metadata.tiktok.title.startswith("Quando")
    assert "fyp" in metadata.tiktok.hashtags
    assert "shorts" in metadata.youtube_shorts.hashtags
    assert len(metadata.tiktok.hashtags) >= 12


def test_generate_metadata_strips_hash_prefix_and_lowercases(
    config, transcription, cast
):
    payload = _valid_payload()
    payload["tiktok"]["hashtags"] = [
        "#FYP",
        "#ForYou",
        "  Humor  ",
        "BRASILEIRO",
        "humor",
        "tiktokbr",
        "foryou",
        "viral",
        "moto",
        "comedia",
        "engracado",
        "humorbr",
    ]
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = _mock_claude_response(payload)

    metadata = generate_metadata(transcription, cast, config, client=client)
    assert "fyp" in metadata.tiktok.hashtags
    assert "humor" in metadata.tiktok.hashtags
    assert "foryou" in metadata.tiktok.hashtags
    assert all(not h.startswith("#") for h in metadata.tiktok.hashtags)
    assert all(h == h.lower() for h in metadata.tiktok.hashtags)


def test_generate_metadata_raises_on_empty_title(config, transcription, cast):
    payload = _valid_payload()
    payload["tiktok"]["title"] = ""
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = _mock_claude_response(payload)

    with pytest.raises(MetadataGenerationError, match=r"título vazio"):
        generate_metadata(transcription, cast, config, client=client)


def test_generate_metadata_raises_when_no_tool_use(config, transcription, cast):
    response = MagicMock()
    response.content = []
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    with pytest.raises(MetadataGenerationError, match=r"chamada à ferramenta"):
        generate_metadata(transcription, cast, config, client=client)


def test_write_metadata_files_creates_json_and_txt(tmp_path):
    metadata = ComicMetadata(
        summary="Resumo curto.",
        tiktok=PlatformMetadata(
            platform="tiktok",
            title="Título TikTok 😂",
            description="Descrição TikTok",
            hashtags=["fyp", "humor"],
        ),
        instagram_reels=PlatformMetadata(
            platform="instagram_reels",
            title="Título IG",
            description="Descrição IG",
            hashtags=["reels", "humor"],
        ),
        youtube_shorts=PlatformMetadata(
            platform="youtube_shorts",
            title="Título YT Shorts",
            description="Descrição YT",
            hashtags=["shorts", "humor"],
        ),
    )

    json_path, txt_path = write_metadata_files(metadata, tmp_path)

    assert json_path.exists()
    assert txt_path.exists()
    assert json_path.name == "metadata.json"
    assert txt_path.name == "metadata.txt"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["tiktok"]["title"] == "Título TikTok 😂"

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "## Tiktok" in txt_content
    assert "## Instagram Reels" in txt_content
    assert "## Youtube Shorts" in txt_content
    assert "#fyp" in txt_content
    assert "#shorts" in txt_content
