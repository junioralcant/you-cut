from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from youcut.models import ClipRecord, ViralClip
from youcut.thumbnail_generator import (
    generate_thumbnail,
    regenerate_thumbnail,
    _build_prompt,
    _select_best_face_frame,
    _resize_to_youtube_format,
)

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


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_generate_calls_dalle_with_correct_params(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    generate_thumbnail(_make_clip(), "João Silva, cabelo castanho", tmp_path, 1, _API_KEY)

    mock_client.images.generate.assert_called_once()
    call_kwargs = mock_client.images.generate.call_args.kwargs
    assert call_kwargs["model"] == "dall-e-3"
    assert call_kwargs["size"] == "1792x1024"
    assert call_kwargs["quality"] == "standard"


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_prompt_contains_thumbnail_idea(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    clip = _make_clip()
    generate_thumbnail(clip, "contexto de rosto aqui", tmp_path, 1, _API_KEY)

    prompt = mock_client.images.generate.call_args.kwargs["prompt"]
    assert clip.thumbnail_idea in prompt
    assert "contexto de rosto aqui" in prompt


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_prompt_is_in_portuguese(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY)

    prompt = mock_client.images.generate.call_args.kwargs["prompt"]
    assert "Thumbnail" in prompt or "thumbnail" in prompt
    assert "YouTube" in prompt


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._describe_frame_characters")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_generate_uses_frame_description_when_clip_path_provided(
    mock_openai_cls, mock_download, mock_describe, mock_resize, tmp_path
):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()
    mock_describe.return_value = "Homem de terno azul, cabelos grisalhos"

    fake_clip = tmp_path / "clip.mp4"
    fake_clip.write_bytes(b"fake")

    generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY, clip_path=fake_clip)

    mock_describe.assert_called_once()
    prompt = mock_client.images.generate.call_args.kwargs["prompt"]
    assert "Homem de terno azul" in prompt


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_generate_skips_frame_extraction_when_clip_path_missing(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY, clip_path=None)

    mock_client.images.generate.assert_called_once()


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator.httpx.get")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_image_downloaded_and_saved(mock_openai_cls, mock_httpx_get, mock_resize, tmp_path):
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


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator.httpx.get")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_regenerate_calls_generate_thumbnail(mock_openai_cls, mock_httpx_get, mock_resize, tmp_path):
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


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_thumbnails_dir_created_automatically(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    output_dir = tmp_path / "video_stem"
    assert not (output_dir / "thumbnails").exists()

    generate_thumbnail(_make_clip(), "", output_dir, 1, _API_KEY)

    assert (output_dir / "thumbnails").exists()


# ---------------------------------------------------------------------------
# _build_prompt() — Flow Podcast style
# ---------------------------------------------------------------------------

def test_build_prompt_contains_flow_podcast_style():
    clip = _make_clip()
    prompt = _build_prompt(clip, "")
    assert "Flow Podcast" in prompt
    assert "close-up" in prompt
    assert "bold" in prompt or "tipografia" in prompt
    assert "vibrantes" in prompt or "contraste" in prompt
    assert "gradiente" in prompt or "fundo simplificado" in prompt


def test_build_prompt_contains_thumbnail_idea_and_face_context():
    clip = _make_clip()
    prompt = _build_prompt(clip, "Homem de terno preto, cabelo grisalho")
    assert clip.thumbnail_idea in prompt
    assert "Homem de terno preto" in prompt


def test_build_prompt_no_face_context():
    clip = _make_clip()
    prompt = _build_prompt(clip, "")
    assert "características" not in prompt


# ---------------------------------------------------------------------------
# _select_best_face_frame() — intelligent frame selection
# ---------------------------------------------------------------------------

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def _make_mediapipe_detection(score: float, xmin: float, ymin: float, width: float, height: float):
    det = MagicMock()
    det.score = [score]
    bb = MagicMock()
    bb.xmin = xmin
    bb.ymin = ymin
    bb.width = width
    bb.height = height
    det.location_data.relative_bounding_box = bb
    return det


@patch("youcut.thumbnail_generator._extract_frame")
@patch("youcut.thumbnail_generator._extract_frame_at")
@patch("youcut.thumbnail_generator._get_video_duration")
def test_select_best_face_frame_returns_highest_score_frame(
    mock_duration, mock_extract_at, mock_extract_fallback, tmp_path
):
    import sys
    fake_mp = MagicMock()
    detector_instance = MagicMock()

    # Frame at index 0: score 0 (no face)
    # Frame at index 5: score high (good face, centered)
    def make_result(detections):
        r = MagicMock()
        r.detections = detections
        return r

    good_det = _make_mediapipe_detection(score=0.95, xmin=0.35, ymin=0.25, width=0.30, height=0.50)
    bad_result = make_result([])
    good_result = make_result([good_det])

    call_count = [0]

    def process_side_effect(rgb):
        call_count[0] += 1
        # 5th call returns a good detection
        if call_count[0] == 5:
            return good_result
        return bad_result

    detector_instance.process.side_effect = process_side_effect
    fake_mp.solutions.face_detection.FaceDetection.return_value = detector_instance

    fake_frame_bytes = [_FAKE_PNG] * 10
    mock_extract_at.side_effect = fake_frame_bytes
    mock_duration.return_value = 60.0

    with patch.dict(sys.modules, {"mediapipe": fake_mp, "cv2": _make_cv2_mock(), "numpy": _make_numpy_mock(_FAKE_PNG)}):
        result = _select_best_face_frame(tmp_path / "clip.mp4")

    assert result == _FAKE_PNG
    mock_extract_fallback.assert_not_called()


@patch("youcut.thumbnail_generator._extract_frame")
@patch("youcut.thumbnail_generator._extract_frame_at")
@patch("youcut.thumbnail_generator._get_video_duration")
def test_select_best_face_frame_falls_back_when_no_face_detected(
    mock_duration, mock_extract_at, mock_extract_fallback, tmp_path
):
    import sys
    fake_mp = MagicMock()
    detector_instance = MagicMock()
    no_face_result = MagicMock()
    no_face_result.detections = []
    detector_instance.process.return_value = no_face_result
    fake_mp.solutions.face_detection.FaceDetection.return_value = detector_instance

    mock_extract_at.return_value = _FAKE_PNG
    mock_duration.return_value = 30.0
    mock_extract_fallback.return_value = b"fallback_frame"

    with patch.dict(sys.modules, {"mediapipe": fake_mp, "cv2": _make_cv2_mock(), "numpy": _make_numpy_mock(_FAKE_PNG)}):
        result = _select_best_face_frame(tmp_path / "clip.mp4")

    assert result == b"fallback_frame"
    mock_extract_fallback.assert_called_once()


@patch("youcut.thumbnail_generator._extract_frame")
def test_select_best_face_frame_falls_back_when_mediapipe_not_installed(
    mock_extract_fallback, tmp_path
):
    import sys
    mock_extract_fallback.return_value = b"fallback_no_mp"

    modules_without_mediapipe = {k: v for k, v in sys.modules.items()}
    modules_without_mediapipe.pop("mediapipe", None)

    with patch.dict(sys.modules, {"mediapipe": None}):
        try:
            result = _select_best_face_frame(tmp_path / "clip.mp4")
        except Exception:
            # If mediapipe module not in sys.modules at all, patch import differently
            pass

    # Test via ImportError side effect
    with patch("builtins.__import__", side_effect=_import_raising_for_mediapipe):
        result = _select_best_face_frame(tmp_path / "clip.mp4")

    assert result == b"fallback_no_mp"
    mock_extract_fallback.assert_called()


def _import_raising_for_mediapipe(name, *args, **kwargs):
    if name == "mediapipe":
        raise ImportError("No module named 'mediapipe'")
    return __import__(name, *args, **kwargs)


def _make_cv2_mock():
    cv2 = MagicMock()
    cv2.COLOR_BGR2RGB = 4
    cv2.IMREAD_COLOR = 1
    fake_frame = MagicMock()
    fake_frame.shape = (720, 1280, 3)
    cv2.imdecode.return_value = fake_frame
    cv2.cvtColor.return_value = fake_frame
    return cv2


def _make_numpy_mock(png_bytes: bytes):
    np = MagicMock()
    np.frombuffer.return_value = MagicMock()
    np.uint8 = MagicMock()
    return np


# ---------------------------------------------------------------------------
# _resize_to_youtube_format() — 1280x720 output
# ---------------------------------------------------------------------------

def test_resize_to_youtube_format_produces_correct_dimensions(tmp_path):
    from PIL import Image

    # Create a fake 1792x1024 image (DALL-E output size)
    img = Image.new("RGB", (1792, 1024), color=(255, 0, 0))
    img_path = tmp_path / "thumb.png"
    img.save(img_path, format="PNG")

    _resize_to_youtube_format(img_path)

    with Image.open(img_path) as result:
        assert result.size == (1280, 720)


def test_resize_to_youtube_format_preserves_png_format(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (800, 600), color=(0, 255, 0))
    img_path = tmp_path / "thumb.png"
    img.save(img_path, format="PNG")

    _resize_to_youtube_format(img_path)

    with Image.open(img_path) as result:
        assert result.format == "PNG"


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._download_image")
@patch("youcut.thumbnail_generator.openai.OpenAI")
def test_generate_thumbnail_resizes_after_download(mock_openai_cls, mock_download, mock_resize, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.images.generate.return_value = _mock_openai_response()

    generate_thumbnail(_make_clip(), "", tmp_path, 1, _API_KEY)

    mock_resize.assert_called_once()
