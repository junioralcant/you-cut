"""Testes do pipeline `youcut comic --engine prunaai`.

Cobertura:
- ``PrunaaiAnimationProvider``: extração de bytes (FileOutput, URL, list).
- ``composition_builder``: idempotência (master existe → reusa) e geração.
- ``prunaai_pipeline``: orquestração ponta-a-ponta com providers mockados.
- Branch em ``run_comic_pipeline``: prunaai default vs ``--engine panels``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.comic.composition_builder import (
    build_master_composition,
    infer_scene_seed,
)
from youcut.comic.providers.prunaai import (
    PrunaaiAnimationError,
    PrunaaiAnimationProvider,
)
from youcut.models import (
    CastMember,
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
            TranscriptionSegment(start=0.0, end=2.0, text="oi", words=[]),
        ],
        language="pt",
        source_path=Path("/tmp/fake.mp4"),
    )


@pytest.fixture
def cast_with_anchors(tmp_path):
    a1 = tmp_path / "a.png"
    a2 = tmp_path / "b.png"
    a1.write_bytes(b"\x89PNG\r\n\x1a\nfake1")
    a2.write_bytes(b"\x89PNG\r\n\x1a\nfake2")
    return [
        CastMember(
            character_id="alice",
            kind="person",
            text_card="alice descrição",
            anchor_image_path=a1,
        ),
        CastMember(
            character_id="bob",
            kind="person",
            text_card="bob descrição",
            anchor_image_path=a2,
        ),
    ]


# ---------------------------------------------------------------------------
# PrunaaiAnimationProvider
# ---------------------------------------------------------------------------


def test_prunaai_provider_requires_api_token():
    with pytest.raises(PrunaaiAnimationError, match=r"REPLICATE_API_TOKEN"):
        PrunaaiAnimationProvider()


def test_prunaai_provider_extracts_url_from_string():
    url = PrunaaiAnimationProvider._extract_url("https://example.com/video.mp4")
    assert url == "https://example.com/video.mp4"


def test_prunaai_provider_extracts_url_from_list_with_url_attr():
    item = MagicMock()
    item.url = "https://cdn.example.com/v.mp4"
    url = PrunaaiAnimationProvider._extract_url([item])
    assert url == "https://cdn.example.com/v.mp4"


def test_prunaai_provider_extracts_url_from_object_url():
    obj = MagicMock()
    obj.url = "https://x.test/v.mp4"
    url = PrunaaiAnimationProvider._extract_url(obj)
    assert url == "https://x.test/v.mp4"


def test_prunaai_provider_returns_none_for_non_http():
    assert PrunaaiAnimationProvider._extract_url("not a url") is None
    assert PrunaaiAnimationProvider._extract_url(None) is None


def test_prunaai_provider_animate_calls_client_run(tmp_path):
    image = tmp_path / "img.png"
    audio = tmp_path / "aud.mp3"
    image.write_bytes(b"fake-png")
    audio.write_bytes(b"fake-mp3")

    fake_output = MagicMock()
    fake_output.read = MagicMock(return_value=b"fake-mp4-bytes")
    fake_output.url = None  # força fallback pra .read()

    client = MagicMock()
    client.run = MagicMock(return_value=fake_output)

    provider = PrunaaiAnimationProvider(client=client)
    data = provider.animate(
        image,
        audio,
        video_prompt="prompt teste",
        voice_prompt="tom teste",
    )
    assert data == b"fake-mp4-bytes"
    client.run.assert_called_once()
    args, kwargs = client.run.call_args
    assert args[0] == "prunaai/p-video-avatar"
    assert kwargs["input"]["video_prompt"] == "prompt teste"
    assert kwargs["input"]["voice_prompt"] == "tom teste"


def test_prunaai_provider_retries_then_fails(tmp_path):
    image = tmp_path / "img.png"
    audio = tmp_path / "aud.mp3"
    image.write_bytes(b"fake")
    audio.write_bytes(b"fake")

    client = MagicMock()
    client.run = MagicMock(side_effect=RuntimeError("boom"))

    provider = PrunaaiAnimationProvider(client=client, max_retries=1)
    with pytest.raises(PrunaaiAnimationError, match=r"após 2 tentativas"):
        provider.animate(image, audio, video_prompt="x")
    assert client.run.call_count == 2


# ---------------------------------------------------------------------------
# composition_builder
# ---------------------------------------------------------------------------


def test_build_master_reuses_existing_file(tmp_path, cast_with_anchors, transcription, config):
    config.comic_scene_seed = "rua urbana neutra"
    output_dir = tmp_path / "out"
    master_dir = output_dir / "comic" / "cast"
    master_dir.mkdir(parents=True)
    existing = master_dir / "_composition_master.png"
    existing.write_bytes(b"existing-master")

    provider = MagicMock()
    provider.generate = MagicMock()  # NÃO deve ser chamado

    path, scene = build_master_composition(
        cast_with_anchors, transcription, config, output_dir, image_provider=provider
    )
    assert path == existing
    assert scene == "rua urbana neutra"
    provider.generate.assert_not_called()


def test_build_master_calls_provider_when_missing(
    tmp_path, cast_with_anchors, transcription, config
):
    config.comic_scene_seed = "deserto pastel"
    output_dir = tmp_path / "out"

    provider = MagicMock()
    provider.generate = MagicMock(return_value=b"new-master-bytes")

    path, scene = build_master_composition(
        cast_with_anchors, transcription, config, output_dir, image_provider=provider
    )
    assert path.exists()
    assert path.read_bytes() == b"new-master-bytes"
    assert scene == "deserto pastel"
    provider.generate.assert_called_once()


def test_build_master_regenerates_when_flag_true(
    tmp_path, cast_with_anchors, transcription, config
):
    config.comic_scene_seed = "x"
    output_dir = tmp_path / "out"
    master_dir = output_dir / "comic" / "cast"
    master_dir.mkdir(parents=True)
    existing = master_dir / "_composition_master.png"
    existing.write_bytes(b"old")

    provider = MagicMock()
    provider.generate = MagicMock(return_value=b"new")

    path, _ = build_master_composition(
        cast_with_anchors,
        transcription,
        config,
        output_dir,
        image_provider=provider,
        regenerate=True,
    )
    assert path.read_bytes() == b"new"
    provider.generate.assert_called_once()


def test_infer_scene_seed_returns_claude_response(
    cast_with_anchors, transcription, config
):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "infer_scene"
    block.input = {"scene": "praça arborizada"}
    response = MagicMock()
    response.content = [block]

    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    seed = infer_scene_seed(transcription, cast_with_anchors, config, client=client)
    assert seed == "praça arborizada"


# ---------------------------------------------------------------------------
# Pipeline branch
# ---------------------------------------------------------------------------


def test_run_comic_pipeline_branches_to_prunaai_by_default(monkeypatch):
    """Quando engine='prunaai' e não há flags exclusivas de panels, delega."""
    from youcut.comic import pipeline as p

    fake_session = object()

    captured: dict = {}

    def fake_runner(video_path, config, **kwargs):
        captured["video"] = video_path
        captured["config"] = config
        return fake_session

    monkeypatch.setattr(
        "youcut.comic.prunaai_pipeline.run_prunaai_pipeline", fake_runner
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine="prunaai")
    out = p.run_comic_pipeline(Path("/tmp/x.mp4"), config)
    assert out is fake_session
    assert captured["video"] == Path("/tmp/x.mp4")


def test_run_comic_pipeline_uses_panels_when_dry_run(monkeypatch):
    """dry_run mantém modo panels mesmo com engine='prunaai'."""
    from youcut.comic import pipeline as p

    monkeypatch.setattr(
        "youcut.comic.prunaai_pipeline.run_prunaai_pipeline",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("não deveria entrar")),
    )

    # Mockar validate_video pra evitar acesso a arquivo real
    fake_spec = MagicMock()
    fake_spec.path = Path("/tmp/x.mp4")
    fake_spec.duration_seconds = 5.0
    fake_spec.width = 720
    fake_spec.height = 1280
    monkeypatch.setattr("youcut.comic.pipeline.validate_video", lambda _p: fake_spec)
    # Mockar transcribe/diarize pra não tentar processar arquivo
    transcription = TranscriptionResult(
        segments=[TranscriptionSegment(start=0.0, end=2.0, text="oi", words=[])],
        language="pt",
        source_path=Path("/tmp/x.mp4"),
    )
    monkeypatch.setattr("youcut.comic.pipeline.transcribe", lambda *a, **kw: transcription)
    monkeypatch.setattr("youcut.comic.pipeline.diarize", lambda *a, **kw: [])
    monkeypatch.setattr(
        "youcut.comic.pipeline.detect_cast",
        lambda *a, **kw: [CastMember(character_id="x", kind="person")],
    )
    monkeypatch.setattr(
        "youcut.comic.pipeline.plan_panels", lambda *a, **kw: []
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine="prunaai")
    # Não deve estourar — dry_run faz fallback ao panels mode no branch
    session = p.run_comic_pipeline(Path("/tmp/x.mp4"), config, dry_run=True)
    assert session is not None
