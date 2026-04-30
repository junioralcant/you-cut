from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from youcut.comic.providers.i2v import I2VGenerationError
from youcut.comic.providers.i2v_replicate import (
    DEFAULT_REPLICATE_MODEL,
    ReplicateImageToVideoProvider,
)


@pytest.fixture
def prompt_image(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.png"
    Image.new("RGB", (320, 568), (10, 30, 60)).save(p)
    return p


def test_provider_requires_api_token_or_client():
    with pytest.raises(I2VGenerationError, match=r"REPLICATE_API_TOKEN"):
        ReplicateImageToVideoProvider()


def test_provider_runs_with_default_model(prompt_image, monkeypatch):
    expected = b"FAKE-MP4"
    fake_client = MagicMock()
    fake_client.run.return_value = "https://cdn.replicate.com/output.mp4"

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return expected

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    provider = ReplicateImageToVideoProvider(client=fake_client)
    out = provider.image_to_video(
        prompt_image=prompt_image,
        prompt_text="movimento expressivo",
        reference_images=[],
        duration_seconds=5.0,
        ratio="720:1280",
    )
    assert out == expected

    args = fake_client.run.call_args
    assert args.args[0] == DEFAULT_REPLICATE_MODEL
    payload = args.kwargs["input"]
    assert payload["prompt"] == "movimento expressivo"
    assert payload["duration"] == 5
    assert payload["aspect_ratio"] == "9:16"
    assert "start_image" in payload  # arquivo aberto


def test_provider_handles_list_output(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.return_value = ["https://cdn.replicate.com/a.mp4"]

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"x"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    provider = ReplicateImageToVideoProvider(client=fake_client)
    out = provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert out == b"x"


def test_provider_handles_file_output_with_read(prompt_image, monkeypatch):
    """Replicate v1+ retorna FileOutput com método read() em alguns modelos."""
    fake_file_output = MagicMock()
    fake_file_output.read = MagicMock(return_value=b"raw-mp4-bytes")
    fake_client = MagicMock()
    fake_client.run.return_value = fake_file_output

    provider = ReplicateImageToVideoProvider(client=fake_client)
    out = provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert out == b"raw-mp4-bytes"


def test_provider_uses_custom_model(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.return_value = "https://cdn.replicate.com/x.mp4"

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"x"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    custom = "kwaivgi/kling-v1.6-pro"
    provider = ReplicateImageToVideoProvider(client=fake_client, model=custom)
    provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert fake_client.run.call_args.args[0] == custom


def test_provider_uses_hailuo_schema_for_minimax_models(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.return_value = "https://cdn.replicate.com/x.mp4"

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"x"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    provider = ReplicateImageToVideoProvider(client=fake_client, model="minimax/hailuo-02")
    provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")

    payload = fake_client.run.call_args.kwargs["input"]
    # Hailuo usa first_frame_image em vez de start_image
    assert "first_frame_image" in payload
    assert "start_image" not in payload
    # Hailuo aceita duration 6 ou 10 (3.0s -> 6)
    assert payload["duration"] == 6
    assert payload["resolution"] == "768p"
    assert payload["prompt_optimizer"] is True
    assert "aspect_ratio" not in payload  # Hailuo não aceita


def test_provider_maps_ratio_correctly(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.return_value = "https://cdn.replicate.com/x.mp4"

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"x"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    provider = ReplicateImageToVideoProvider(client=fake_client)
    provider.image_to_video(prompt_image, "p", [], duration_seconds=5.0, ratio="1280:720")
    payload = fake_client.run.call_args.kwargs["input"]
    assert payload["aspect_ratio"] == "16:9"


def test_provider_raises_on_invalid_output(prompt_image):
    fake_client = MagicMock()
    fake_client.run.return_value = {"foo": "bar"}  # sem URL nem read()
    provider = ReplicateImageToVideoProvider(client=fake_client, max_retries=0)
    with pytest.raises(I2VGenerationError, match=r"URL ausente"):
        provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")


def test_provider_retries_on_transient_error(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.side_effect = [
        RuntimeError("transient"),
        "https://cdn.replicate.com/ok.mp4",
    ]

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"ok"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_replicate.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )
    monkeypatch.setattr("youcut.comic.providers.i2v_replicate.time.sleep", lambda d: None)

    provider = ReplicateImageToVideoProvider(client=fake_client, max_retries=2)
    out = provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert out == b"ok"
    assert fake_client.run.call_count == 2


def test_provider_exhausts_retries(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.run.side_effect = RuntimeError("always")
    monkeypatch.setattr("youcut.comic.providers.i2v_replicate.time.sleep", lambda d: None)

    provider = ReplicateImageToVideoProvider(client=fake_client, max_retries=2)
    with pytest.raises(I2VGenerationError, match=r"3 tentativas"):
        provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert fake_client.run.call_count == 3


def test_provider_raises_when_prompt_image_missing(tmp_path):
    fake_client = MagicMock()
    provider = ReplicateImageToVideoProvider(client=fake_client)
    with pytest.raises(I2VGenerationError, match=r"prompt_image não encontrada"):
        provider.image_to_video(tmp_path / "missing.png", "p", [], duration_seconds=3.0, ratio="720:1280")
