import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from youcut.models import ClipRecord, ViralClip
from youcut.thumbnail_generator import (
    _apply_frame_processing,
    _compose_text_overlay,
    _resize_to_youtube_format,
    _select_best_face_frame,
    generate_thumbnail,
)

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def _make_clip() -> ViralClip:
    return ViralClip(
        title="Momento Viral Incrível",
        reason="High energy",
        viral_score=9.0,
        start_time=60.0,
        end_time=660.0,
        description="The host explains the main topic.",
        hashtags=["#youtube"],
        thumbnail_idea="Host explaining excitedly with charts in background",
        thumbnail_text="MOMENTO IMPACTANTE",
        cut_mode="youtube",
    )


def _make_pil_png(width: int = 640, height: int = 480) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def _import_raising_for_mediapipe(name, *args, **kwargs):
    if name == "mediapipe":
        raise ImportError("No module named 'mediapipe'")
    return __import__(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# generate_thumbnail() — assinatura nova (sem OpenAI)
# ---------------------------------------------------------------------------

@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._compose_text_overlay")
@patch("youcut.thumbnail_generator._apply_frame_processing")
@patch("youcut.thumbnail_generator._select_best_face_frame")
def test_generate_thumbnail_returns_correct_path(
    mock_select, mock_process, mock_compose, mock_resize, tmp_path
):
    png_bytes = _make_pil_png()
    mock_select.return_value = png_bytes
    pil_img = Image.new("RGB", (640, 480), color=(50, 100, 150))
    mock_process.return_value = pil_img
    mock_compose.return_value = pil_img

    clip = _make_clip()
    fake_mp4 = tmp_path / "clip.mp4"
    fake_mp4.write_bytes(b"fake")

    result = generate_thumbnail(clip, tmp_path, clip_index=3, clip_path=fake_mp4)

    expected = tmp_path / "thumbnails" / "clip_03.png"
    assert result == expected


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._compose_text_overlay")
@patch("youcut.thumbnail_generator._apply_frame_processing")
@patch("youcut.thumbnail_generator._select_best_face_frame")
def test_generate_thumbnail_creates_thumbnails_dir(
    mock_select, mock_process, mock_compose, mock_resize, tmp_path
):
    png_bytes = _make_pil_png()
    mock_select.return_value = png_bytes
    pil_img = Image.new("RGB", (640, 480))
    mock_process.return_value = pil_img
    mock_compose.return_value = pil_img

    clip = _make_clip()
    output_dir = tmp_path / "video_stem"
    fake_mp4 = output_dir / "clip.mp4"
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"fake")

    assert not (output_dir / "thumbnails").exists()
    generate_thumbnail(clip, output_dir, clip_index=1, clip_path=fake_mp4)
    assert (output_dir / "thumbnails").exists()


@patch("youcut.thumbnail_generator._resize_to_youtube_format")
@patch("youcut.thumbnail_generator._compose_text_overlay")
@patch("youcut.thumbnail_generator._apply_frame_processing")
@patch("youcut.thumbnail_generator._select_best_face_frame")
def test_generate_thumbnail_calls_pipeline_in_order(
    mock_select, mock_process, mock_compose, mock_resize, tmp_path
):
    call_order = []

    png_bytes = _make_pil_png()
    pil_img = Image.new("RGB", (640, 480))

    mock_select.side_effect = lambda *a, **k: (call_order.append("select"), png_bytes)[1]
    mock_process.side_effect = lambda *a, **k: (call_order.append("process"), pil_img)[1]
    mock_compose.side_effect = lambda *a, **k: (call_order.append("compose"), pil_img)[1]
    mock_resize.side_effect = lambda *a, **k: call_order.append("resize")

    clip = _make_clip()
    fake_mp4 = tmp_path / "clip.mp4"
    fake_mp4.write_bytes(b"fake")

    generate_thumbnail(clip, tmp_path, clip_index=0, clip_path=fake_mp4)

    assert call_order == ["select", "process", "compose", "resize"]


def test_generate_thumbnail_raises_when_clip_path_missing(tmp_path):
    clip = _make_clip()
    with pytest.raises((ValueError, Exception)):
        generate_thumbnail(clip, tmp_path, clip_index=0, clip_path=None)


# ---------------------------------------------------------------------------
# _apply_frame_processing() — Pillow fallback (sem MediaPipe/cv2)
# ---------------------------------------------------------------------------

def test_apply_frame_processing_returns_pil_image():
    frame_bytes = _make_pil_png(640, 480)
    result = _apply_frame_processing(frame_bytes)
    assert isinstance(result, Image.Image)


def test_apply_frame_processing_preserves_dimensions():
    frame_bytes = _make_pil_png(800, 600)
    result = _apply_frame_processing(frame_bytes)
    assert result.size == (800, 600)


def test_apply_frame_processing_returns_rgb():
    frame_bytes = _make_pil_png(320, 240)
    result = _apply_frame_processing(frame_bytes)
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# _compose_text_overlay()
# ---------------------------------------------------------------------------

def test_compose_text_overlay_returns_pil_image():
    img = Image.new("RGB", (1280, 720), color=(50, 100, 150))
    result = _compose_text_overlay(img, "Título do Episódio")
    assert isinstance(result, Image.Image)


def test_compose_text_overlay_preserves_dimensions():
    img = Image.new("RGB", (1280, 720))
    result = _compose_text_overlay(img, "Momento Incrível")
    assert result.size == (1280, 720)


def test_compose_text_overlay_empty_title_does_not_raise():
    img = Image.new("RGB", (1280, 720))
    result = _compose_text_overlay(img, "")
    assert result.size == (1280, 720)


def test_compose_text_overlay_long_title_does_not_raise():
    img = Image.new("RGB", (1280, 720))
    long_title = "Uma frase muito longa que tem muitas palavras e não deve quebrar tudo de forma alguma"
    result = _compose_text_overlay(img, long_title)
    assert result.size == (1280, 720)


def test_compose_text_overlay_does_not_modify_original():
    img = Image.new("RGB", (1280, 720), color=(255, 0, 0))
    original_pixel = img.getpixel((0, 0))
    _compose_text_overlay(img, "Título")
    assert img.getpixel((0, 0)) == original_pixel


# ---------------------------------------------------------------------------
# _select_best_face_frame() — frame selection logic
# ---------------------------------------------------------------------------

@patch("youcut.thumbnail_generator._extract_frame")
@patch("youcut.thumbnail_generator._extract_frame_at")
@patch("youcut.thumbnail_generator._get_video_duration")
def test_select_best_face_frame_returns_highest_score_frame(
    mock_duration, mock_extract_at, mock_extract_fallback, tmp_path
):
    import sys
    fake_mp = MagicMock()
    detector_instance = MagicMock()

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
    mock_extract_fallback.return_value = b"fallback_no_mp"

    with patch("builtins.__import__", side_effect=_import_raising_for_mediapipe):
        result = _select_best_face_frame(tmp_path / "clip.mp4")

    assert result == b"fallback_no_mp"
    mock_extract_fallback.assert_called()


# ---------------------------------------------------------------------------
# _resize_to_youtube_format() — 1280x720 output
# ---------------------------------------------------------------------------

def test_resize_to_youtube_format_produces_correct_dimensions(tmp_path):
    img = Image.new("RGB", (1792, 1024), color=(255, 0, 0))
    img_path = tmp_path / "thumb.png"
    img.save(img_path, format="PNG")

    _resize_to_youtube_format(img_path)

    with Image.open(img_path) as result:
        assert result.size == (1280, 720)


def test_resize_to_youtube_format_preserves_png_format(tmp_path):
    img = Image.new("RGB", (800, 600), color=(0, 255, 0))
    img_path = tmp_path / "thumb.png"
    img.save(img_path, format="PNG")

    _resize_to_youtube_format(img_path)

    with Image.open(img_path) as result:
        assert result.format == "PNG"
