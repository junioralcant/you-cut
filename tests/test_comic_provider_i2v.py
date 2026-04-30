import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._fakes.comic_providers import FakeI2VProvider
from youcut.comic.providers.i2v import (
    I2VGenerationError,
    ImageToVideoProvider,
    RunwayProvider,
)


@pytest.fixture
def prompt_image(tmp_path):
    p = tmp_path / "prompt.png"
    Image.new("RGB", (1024, 1792), (10, 30, 60)).save(p)
    return p


@pytest.fixture
def reference_image(tmp_path):
    p = tmp_path / "ref.png"
    Image.new("RGB", (1024, 1024), (200, 150, 80)).save(p)
    return p


def _succeeded_task(video_url: str) -> SimpleNamespace:
    return SimpleNamespace(id="t-001", status="SUCCEEDED", output=[video_url])


def _data_url_mp4(payload: bytes = b"\x00fakeMP4") -> str:
    import base64

    return f"data:video/mp4;base64,{base64.b64encode(payload).decode('ascii')}"


def test_provider_requires_api_key_or_client():
    with pytest.raises(I2VGenerationError, match=r"RUNWAY_API_KEY"):
        RunwayProvider()


def test_provider_round_trip_with_data_url(prompt_image, reference_image, monkeypatch):
    expected = b"\x00\x01\x02FAKE-MP4"
    fake_client = MagicMock()
    fake_client.image_to_video.create.return_value = SimpleNamespace(id="t-1")
    fake_client.tasks.retrieve.return_value = _succeeded_task(_data_url_mp4(expected))

    provider = RunwayProvider(client=fake_client, poll_interval=0, max_poll_time=10)
    out = provider.image_to_video(
        prompt_image=prompt_image,
        prompt_text="movimento sutil",
        reference_images=[reference_image],
        duration_seconds=3.0,
        ratio="720:1280",
    )

    assert out == expected
    create_kwargs = fake_client.image_to_video.create.call_args.kwargs
    assert create_kwargs["model"] == "gen4_turbo"
    assert create_kwargs["ratio"] == "720:1280"
    assert create_kwargs["duration"] == 3
    assert create_kwargs["prompt_image"].startswith("data:image/png;base64,")
    # SDK runwayml>=4 removeu reference_images do gen4_turbo (RunwayProvider
    # ignora silenciosamente — consistência vem da imagem-base via gpt-image-1).
    assert "reference_images" not in create_kwargs


def test_provider_polls_until_succeeded(prompt_image, reference_image, monkeypatch):
    expected = b"DONE"
    fake_client = MagicMock()
    fake_client.image_to_video.create.return_value = SimpleNamespace(id="t-2")
    fake_client.tasks.retrieve.side_effect = [
        SimpleNamespace(id="t-2", status="PENDING"),
        SimpleNamespace(id="t-2", status="RUNNING"),
        _succeeded_task(_data_url_mp4(expected)),
    ]

    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: None)

    provider = RunwayProvider(client=fake_client, poll_interval=0, max_poll_time=30)
    out = provider.image_to_video(prompt_image, "p", [reference_image])
    assert out == expected
    assert fake_client.tasks.retrieve.call_count == 3


def test_provider_raises_when_task_fails(prompt_image, reference_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.image_to_video.create.return_value = SimpleNamespace(id="t-3")
    fake_client.tasks.retrieve.return_value = SimpleNamespace(
        id="t-3", status="FAILED", failure_reason="rate-limited"
    )

    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: None)

    provider = RunwayProvider(client=fake_client, max_retries=0, poll_interval=0)
    with pytest.raises(I2VGenerationError, match=r"FAILED|rate-limited"):
        provider.image_to_video(prompt_image, "p", [reference_image])


def test_provider_retries_after_create_error(prompt_image, reference_image, monkeypatch):
    expected = b"second-try"
    fake_client = MagicMock()
    fake_client.image_to_video.create.side_effect = [
        RuntimeError("transient"),
        SimpleNamespace(id="t-4"),
    ]
    fake_client.tasks.retrieve.return_value = _succeeded_task(_data_url_mp4(expected))

    sleep_calls: list[float] = []
    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: sleep_calls.append(d))

    provider = RunwayProvider(client=fake_client, max_retries=2, poll_interval=0)
    out = provider.image_to_video(prompt_image, "p", [reference_image])

    assert out == expected
    assert fake_client.image_to_video.create.call_count == 2
    assert len(sleep_calls) >= 1


def test_provider_exhausts_retries(prompt_image, reference_image, monkeypatch):
    fake_client = MagicMock()
    fake_client.image_to_video.create.side_effect = RuntimeError("always")
    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: None)

    provider = RunwayProvider(client=fake_client, max_retries=2, poll_interval=0)
    with pytest.raises(I2VGenerationError, match=r"3 tentativas"):
        provider.image_to_video(prompt_image, "p", [reference_image])
    assert fake_client.image_to_video.create.call_count == 3


def test_provider_raises_when_prompt_image_missing(tmp_path, reference_image):
    fake_client = MagicMock()
    provider = RunwayProvider(client=fake_client)
    with pytest.raises(I2VGenerationError, match=r"prompt_image não encontrada"):
        provider.image_to_video(tmp_path / "missing.png", "p", [reference_image])


def test_provider_raises_when_reference_missing(tmp_path, prompt_image):
    fake_client = MagicMock()
    provider = RunwayProvider(client=fake_client)
    with pytest.raises(I2VGenerationError, match=r"reference_image não encontrada"):
        provider.image_to_video(prompt_image, "p", [tmp_path / "missing.png"])


def test_provider_drops_references_for_v4_sdk(prompt_image, tmp_path, monkeypatch):
    """SDK runwayml>=4 removeu reference_images do gen4_turbo: extras são
    aceitos pela interface mas silenciosamente descartados antes do create."""
    refs = []
    for i in range(5):
        p = tmp_path / f"ref{i}.png"
        Image.new("RGB", (64, 64), (i * 30, 100, 100)).save(p)
        refs.append(p)

    fake_client = MagicMock()
    fake_client.image_to_video.create.return_value = SimpleNamespace(id="t-5")
    fake_client.tasks.retrieve.return_value = _succeeded_task(_data_url_mp4(b"x"))

    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: None)

    provider = RunwayProvider(client=fake_client, poll_interval=0)
    provider.image_to_video(prompt_image, "p", refs)

    create_kwargs = fake_client.image_to_video.create.call_args.kwargs
    assert "reference_images" not in create_kwargs


def test_provider_handles_dict_task_payload(prompt_image, reference_image, monkeypatch):
    expected = b"DICT"
    fake_client = MagicMock()
    fake_client.image_to_video.create.return_value = {"id": "t-6"}
    fake_client.tasks.retrieve.return_value = {
        "id": "t-6",
        "status": "SUCCEEDED",
        "output": [_data_url_mp4(expected)],
    }
    monkeypatch.setattr("youcut.comic.providers.i2v.time.sleep", lambda d: None)

    provider = RunwayProvider(client=fake_client, poll_interval=0)
    out = provider.image_to_video(prompt_image, "p", [reference_image])
    assert out == expected


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_fake_i2v_provider_generates_valid_mp4(prompt_image, reference_image):
    fake = FakeI2VProvider()
    assert isinstance(fake, ImageToVideoProvider)
    out = fake.image_to_video(
        prompt_image=prompt_image,
        prompt_text="movimento",
        reference_images=[reference_image],
        duration_seconds=2.0,
        ratio="720:1280",
    )
    assert out[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00 ", b"\x00\x00\x00\x1c", b"\x00\x00\x00\x14")
    assert len(fake.calls) == 1
    assert fake.calls[0]["duration"] == 2.0


def test_runway_provider_implements_protocol():
    fake_client = MagicMock()
    provider = RunwayProvider(client=fake_client)
    assert isinstance(provider, ImageToVideoProvider)
