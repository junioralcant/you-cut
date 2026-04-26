from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from youcut.models import ClipRecord, ViralClip
from youcut.thumbnail_generator import generate_thumbnail, regenerate_thumbnail

_API_KEY = "test-api-key"
_IMAGE_URL = "https://oaidalleapiprodscus.blob.core.windows.net/thumb.png"
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG bytes


def _make_clip() -> ViralClip:
    return ViralClip(
        title="Top Moment",
        reason="High energy",
        viral_score=9.0,
        start_time=60.0,
        end_time=660.0,
        description="The host explains the main topic.",
        hashtags=["#youtube"],
        thumbnail_idea="Host explaining excitedly with charts in background",
        cut_mode="youtube",
    )


def _mock_openai_response(url: str = _IMAGE_URL) -> MagicMock:
    img = MagicMock()
    img.url = url
    response = MagicMock()
    response.data = [img]
    return response


def _make_httpx_side_effect(content: bytes = _FAKE_PNG):
    def _get(url, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.content = content
        resp.raise_for_status = MagicMock()
        return resp
    return _get


@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_generate_calls_dalle_with_correct_params(mock_openai_cls, mock_download, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    generate_thumbnail(_make_clip(), "John Doe, brown hair", tmp_path, 1, _API_KEY)

    mock_client.images.generate.assert_called_once()
    call_kwargs = mock_client.images.generate.call_args.kwargs
    assert call_kwargs["model"] == "dall-e-3"
    assert call_kwargs["size"] == "1280x720"
    assert call_kwargs["quality"] == "standard"


@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_prompt_contains_thumbnail_idea(mock_openai_cls, mock_download, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    clip = _make_clip()
    generate_thumbnail(clip, "face context here", tmp_path, 1, _API_KEY)

    prompt = mock_client.images.generate.call_args.kwargs["prompt"]
    assert clip.thumbnail_idea in prompt
    assert "face context here" in prompt


@patch("youcut.thumbnail_generator.httpx.get")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_image_downloaded_and_saved(mock_openai_cls, mock_httpx_get, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()
    mock_httpx_get.side_effect = _make_httpx_side_effect()

    result = generate_thumbnail(_make_clip(), "", tmp_path, 3, _API_KEY)

    expected = tmp_path / "thumbnails" / "clip_03.png"
    assert result == expected
    assert expected.exists()
    assert expected.read_bytes() == _FAKE_PNG
    mock_httpx_get.assert_called_once_with(_IMAGE_URL, follow_redirects=True, timeout=30.0)


@patch("youcut.thumbnail_generator.httpx.get")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_regenerate_calls_generate_thumbnail(mock_openai_cls, mock_httpx_get, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()
    mock_httpx_get.side_effect = _make_httpx_side_effect()

    clip = _make_clip()
    expected_thumb = tmp_path / "thumbnails" / "clip_02.png"
    clip_record = ClipRecord(
        title="Clip 2",
        start_time=120.0,
        end_time=420.0,
        clip_path=tmp_path / "clip_02.mp4",
        thumbnail_path=expected_thumb,
        approved=True,
    )

    result = regenerate_thumbnail(clip, clip_record, _API_KEY)

    mock_client.images.generate.assert_called_once()
    assert result == expected_thumb
    assert expected_thumb.exists()


@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_authentication_error_raises_runtime_error(mock_openai_cls, mock_download, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    req = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
    response = httpx.Response(401, request=req, content=b"unauthorized")
    mock_client.images.generate.side_effect = openai.AuthenticationError(
        "invalid api key", response=response, body=None
    )

    with pytest.raises(RuntimeError, match="Invalid OpenAI API key"):
        generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY)


@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_rate_limit_error_raises_runtime_error(mock_openai_cls, mock_download, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    req = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
    response = httpx.Response(429, request=req, content=b"rate limit")
    mock_client.images.generate.side_effect = openai.RateLimitError(
        "rate limit", response=response, body=None
    )

    with pytest.raises(RuntimeError, match="rate limit"):
        generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY)


@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_thumbnails_dir_created_automatically(mock_openai_cls, mock_download, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    output_dir = tmp_path / "video_stem"
    assert not (output_dir / "thumbnails").exists()

    generate_thumbnail(_make_clip(), "", output_dir, 1, _API_KEY)

    assert (output_dir / "thumbnails").exists()
