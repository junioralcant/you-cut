from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from youcut.comic.providers.i2v import I2VGenerationError
from youcut.comic.providers.i2v_fal import (
    DEFAULT_FAL_MODEL,
    FalImageToVideoProvider,
)


@pytest.fixture
def prompt_image(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.png"
    Image.new("RGB", (320, 568), (10, 30, 60)).save(p)
    return p


def _ok_result(video_url: str = "https://cdn.fal.ai/result.mp4") -> dict:
    return {"video": {"url": video_url}}


def test_provider_requires_api_key_or_client():
    with pytest.raises(I2VGenerationError, match=r"FAL_KEY"):
        FalImageToVideoProvider()


def test_provider_subscribes_with_data_url(prompt_image, monkeypatch):
    expected = b"FAKE-MP4-BYTES"
    fake_client = MagicMock()
    fake_client.subscribe.return_value = _ok_result()

    # Mock urlopen para download
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return expected

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_fal.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    provider = FalImageToVideoProvider(client=fake_client)
    out = provider.image_to_video(
        prompt_image=prompt_image,
        prompt_text="movimento expressivo",
        reference_images=[],
        duration_seconds=5.0,
        ratio="720:1280",
    )
    assert out == expected

    args = fake_client.subscribe.call_args
    assert args.args[0] == DEFAULT_FAL_MODEL
    arguments = args.kwargs["arguments"]
    assert arguments["image_url"].startswith("data:image/png;base64,")
    assert arguments["prompt"] == "movimento expressivo"
    assert arguments["duration"] == "5"
    assert arguments["aspect_ratio"] == "9:16"


def test_provider_maps_ratio_correctly(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.subscribe.return_value = _ok_result()
    monkeypatch.setattr(
        "youcut.comic.providers.i2v_fal.urllib.request.urlopen",
        lambda url, timeout: SimpleNamespace(__enter__=lambda *_: SimpleNamespace(read=lambda: b"x"), __exit__=lambda *_: None),
    )

    # ratio que está no map
    provider = FalImageToVideoProvider(client=fake_client)
    try:
        provider.image_to_video(prompt_image, "p", [], duration_seconds=5.0, ratio="1280:720")
    except Exception:
        pass
    arguments = fake_client.subscribe.call_args.kwargs["arguments"]
    assert arguments["aspect_ratio"] == "16:9"


def test_provider_uses_custom_model(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.subscribe.return_value = _ok_result()

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"x"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_fal.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )

    custom = "fal-ai/luma-dream-machine/ray-2/image-to-video"
    provider = FalImageToVideoProvider(client=fake_client, model=custom)
    provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert fake_client.subscribe.call_args.args[0] == custom


def test_provider_raises_on_missing_video_url(prompt_image):
    fake_client = MagicMock()
    fake_client.subscribe.return_value = {"foo": "bar"}  # sem video.url
    provider = FalImageToVideoProvider(client=fake_client, max_retries=0)
    with pytest.raises(I2VGenerationError, match=r"sem video.url"):
        provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")


def test_provider_retries_on_transient_error(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.subscribe.side_effect = [
        RuntimeError("transient"),
        _ok_result(),
    ]

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"ok"

    monkeypatch.setattr(
        "youcut.comic.providers.i2v_fal.urllib.request.urlopen",
        lambda url, timeout: _Resp(),
    )
    monkeypatch.setattr("youcut.comic.providers.i2v_fal.time.sleep", lambda d: None)

    provider = FalImageToVideoProvider(client=fake_client, max_retries=2)
    out = provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert out == b"ok"
    assert fake_client.subscribe.call_count == 2


def test_provider_exhausts_retries(prompt_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.subscribe.side_effect = RuntimeError("always")
    monkeypatch.setattr("youcut.comic.providers.i2v_fal.time.sleep", lambda d: None)

    provider = FalImageToVideoProvider(client=fake_client, max_retries=2)
    with pytest.raises(I2VGenerationError, match=r"3 tentativas"):
        provider.image_to_video(prompt_image, "p", [], duration_seconds=3.0, ratio="720:1280")
    assert fake_client.subscribe.call_count == 3


def test_provider_raises_when_prompt_image_missing(tmp_path):
    fake_client = MagicMock()
    provider = FalImageToVideoProvider(client=fake_client)
    with pytest.raises(I2VGenerationError, match=r"prompt_image não encontrada"):
        provider.image_to_video(
            tmp_path / "missing.png", "p", [], duration_seconds=3.0, ratio="720:1280"
        )
