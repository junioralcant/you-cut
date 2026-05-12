import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._fakes.comic_providers import FakeImageProvider
from youcut.comic.providers.images import (
    ImageGenerationError,
    ImageProvider,
    OpenAIImageProvider,
)


def _png_b64(size: tuple[int, int] = (1024, 1024)) -> str:
    img = Image.new("RGB", size, (10, 20, 30))
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _success_response() -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])


def test_provider_requires_api_key_or_client():
    with pytest.raises(ImageGenerationError, match=r"OPENAI_API_KEY"):
        OpenAIImageProvider()


def test_provider_accepts_injected_client():
    fake_client = MagicMock()
    fake_client.images.generate.return_value = _success_response()
    provider = OpenAIImageProvider(client=fake_client)
    out = provider.generate("hello world", size="1024x1024")
    assert isinstance(out, bytes)
    assert out.startswith(b"\x89PNG")


def test_provider_uses_edit_when_reference_images_present(tmp_path):
    fake_client = MagicMock()
    fake_client.images.edit.return_value = _success_response()

    ref = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), (100, 100, 100)).save(ref)

    provider = OpenAIImageProvider(client=fake_client)
    out = provider.generate("desenha", reference_images=[ref], size="1024x1024", input_fidelity="high")

    assert out.startswith(b"\x89PNG")
    fake_client.images.generate.assert_not_called()
    fake_client.images.edit.assert_called_once()
    kwargs = fake_client.images.edit.call_args.kwargs
    assert kwargs["model"] == "gpt-image-1.5"
    assert kwargs["prompt"] == "desenha"
    assert kwargs["size"] == "1024x1024"
    assert kwargs["input_fidelity"] == "high"
    assert kwargs["quality"] == "low"
    assert isinstance(kwargs["image"], list)


def test_provider_retries_then_succeeds(monkeypatch):
    fake_client = MagicMock()
    fake_client.images.generate.side_effect = [
        RuntimeError("transient"),
        _success_response(),
    ]

    sleep_calls: list[float] = []
    monkeypatch.setattr("youcut.comic.providers.images.time.sleep", lambda d: sleep_calls.append(d))

    provider = OpenAIImageProvider(client=fake_client, max_retries=2)
    out = provider.generate("retry-me")

    assert out.startswith(b"\x89PNG")
    assert fake_client.images.generate.call_count == 2
    assert len(sleep_calls) == 1


def test_provider_exhausts_retries_and_raises(monkeypatch):
    fake_client = MagicMock()
    fake_client.images.generate.side_effect = RuntimeError("always-fails")

    monkeypatch.setattr("youcut.comic.providers.images.time.sleep", lambda d: None)

    provider = OpenAIImageProvider(client=fake_client, max_retries=2)
    with pytest.raises(ImageGenerationError, match=r"Falha ao gerar imagem após 3 tentativas"):
        provider.generate("never-works")
    assert fake_client.images.generate.call_count == 3


def test_provider_rejects_empty_prompt():
    fake_client = MagicMock()
    provider = OpenAIImageProvider(client=fake_client)
    with pytest.raises(ImageGenerationError, match=r"prompt vazio"):
        provider.generate("")


def test_provider_extracts_b64_from_dict_payload():
    fake_client = MagicMock()
    fake_client.images.generate.return_value = SimpleNamespace(data=[{"b64_json": _png_b64()}])
    provider = OpenAIImageProvider(client=fake_client)
    out = provider.generate("prompt")
    assert out.startswith(b"\x89PNG")


def test_provider_raises_on_invalid_response_shape():
    fake_client = MagicMock()
    fake_client.images.generate.return_value = SimpleNamespace(data=[])
    provider = OpenAIImageProvider(client=fake_client, max_retries=0)
    with pytest.raises(ImageGenerationError):
        provider.generate("prompt")


def test_provider_raises_when_reference_image_missing(tmp_path):
    fake_client = MagicMock()
    provider = OpenAIImageProvider(client=fake_client, max_retries=0)
    missing_ref = tmp_path / "nope.png"
    with pytest.raises(ImageGenerationError, match=r"referência não encontrada"):
        provider.generate("p", reference_images=[missing_ref])


def test_fake_image_provider_implements_protocol():
    fake = FakeImageProvider()
    assert isinstance(fake, ImageProvider)
    out = fake.generate("hello", size="512x512")
    assert out.startswith(b"\x89PNG")
    img = Image.open(__import__("io").BytesIO(out))
    assert img.size == (512, 512)
    assert len(fake.calls) == 1


def test_fake_image_provider_default_size_is_1024_square():
    fake = FakeImageProvider()
    out = fake.generate("p")
    img = Image.open(__import__("io").BytesIO(out))
    assert img.size == (1024, 1024)


def test_openai_image_provider_implements_protocol():
    fake_client = MagicMock()
    fake_client.images.generate.return_value = _success_response()
    provider = OpenAIImageProvider(client=fake_client)
    assert isinstance(provider, ImageProvider)
