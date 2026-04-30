import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from youcut.captioner import build_ass_for_words
from youcut.comic.composer import (
    DURATION_TOLERANCE_SECONDS,
    ComposerError,
    _extend_or_trim,
    _ffprobe_duration,
    compose,
)
from youcut.models import (
    Panel,
    PanelRenderResult,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe não disponível",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clip(path: Path, *, duration: float, color: str = "blue") -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:size=1080x1920:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-t",
        f"{duration:.2f}",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _make_video_with_audio(path: Path, *, duration: float) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:size=320x240:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-t",
        f"{duration:.2f}",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _audio_stream_md5(video_path: Path) -> str:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-f",
        "md5",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    line = result.stdout.strip().splitlines()[0]
    return line.replace("MD5=", "")


def _make_panel(index: int, start: float, end: float) -> Panel:
    return Panel(
        index=index,
        start_time=start,
        end_time=end,
        participants=["narrator"],
        framing="close",
        scene="cena",
        pose_description="pose",
        panel_seconds_target=end - start,
    )


def _make_panel_result(index: int, clip_path: Path, seconds: float) -> PanelRenderResult:
    return PanelRenderResult(
        panel_index=index,
        base_image_path=clip_path.with_suffix(".png"),
        clip_path=clip_path,
        clip_seconds=seconds,
        was_static_fallback=False,
        cost_usd=0.10,
    )


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def transcription():
    words = [
        WordTimestamp(word=f"palavra{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(12)
    ]
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=6.0, text="palavras de teste", words=words)
        ],
        language="pt",
        source_path=Path("v.mp4"),
    )


# ---------------------------------------------------------------------------
# build_ass_for_words refactor — ensure it still works
# ---------------------------------------------------------------------------


def test_build_ass_for_words_includes_words_and_resolution():
    words = [
        WordTimestamp(word="oi", start=0.0, end=0.5),
        WordTimestamp(word="mundo", start=0.6, end=1.0),
    ]
    doc = build_ass_for_words(words, output_size=(720, 1280))
    assert "PlayResX: 720" in doc
    assert "PlayResY: 1280" in doc
    assert "oi" in doc
    assert "mundo" in doc
    assert "Dialogue: 0" in doc


def test_build_ass_for_words_default_resolution_is_1080x1920():
    words = [WordTimestamp(word="x", start=0.0, end=0.1)]
    doc = build_ass_for_words(words)
    assert "PlayResX: 1080" in doc
    assert "PlayResY: 1920" in doc


def test_build_ass_for_words_applies_offset():
    words = [WordTimestamp(word="x", start=10.0, end=10.5)]
    doc = build_ass_for_words(words, offset=10.0)
    assert "0:00:00.00" in doc
    assert "0:00:00.50" in doc


# ---------------------------------------------------------------------------
# _extend_or_trim
# ---------------------------------------------------------------------------


def test_extend_clip_with_hold(tmp_path):
    src = _make_clip(tmp_path / "src.mp4", duration=2.0)
    out = tmp_path / "out.mp4"
    _extend_or_trim(src, target_seconds=3.0, out_path=out, mode="hold")
    duration = _ffprobe_duration(out)
    assert abs(duration - 3.0) <= 0.15


def test_trim_clip_when_longer(tmp_path):
    src = _make_clip(tmp_path / "src.mp4", duration=4.0)
    out = tmp_path / "out.mp4"
    _extend_or_trim(src, target_seconds=2.5, out_path=out, mode="hold")
    duration = _ffprobe_duration(out)
    assert abs(duration - 2.5) <= 0.15


def test_no_op_when_durations_equal(tmp_path):
    src = _make_clip(tmp_path / "src.mp4", duration=2.0)
    out = tmp_path / "out.mp4"
    _extend_or_trim(src, target_seconds=2.0, out_path=out, mode="hold")
    duration = _ffprobe_duration(out)
    assert abs(duration - 2.0) <= 0.1


def test_extend_or_trim_rejects_unknown_mode(tmp_path):
    src = _make_clip(tmp_path / "src.mp4", duration=1.5)
    out = tmp_path / "out.mp4"
    with pytest.raises(ComposerError, match=r"modo de extensão"):
        _extend_or_trim(src, target_seconds=3.0, out_path=out, mode="bogus")


# ---------------------------------------------------------------------------
# compose — integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_compose_produces_final_video_matching_audio_duration(tmp_path, transcription, config):
    video = _make_video_with_audio(tmp_path / "input.mp4", duration=6.0)
    panels = [
        _make_panel(0, 0.0, 3.0),
        _make_panel(1, 3.0, 6.0),
    ]
    clip0 = _make_clip(tmp_path / "clip0.mp4", duration=2.5)
    clip1 = _make_clip(tmp_path / "clip1.mp4", duration=4.0)
    panel_results = [
        _make_panel_result(0, clip0, 2.5),
        _make_panel_result(1, clip1, 4.0),
    ]

    out_dir = tmp_path / "output"
    final = compose(panels, panel_results, transcription, video, out_dir, config)

    assert final.exists()
    final_duration = _ffprobe_duration(final)
    assert abs(final_duration - 6.0) <= DURATION_TOLERANCE_SECONDS + 0.3


@pytest.mark.integration
def test_compose_preserves_audio_stream_hash(tmp_path, transcription, config):
    video = _make_video_with_audio(tmp_path / "input.mp4", duration=6.0)
    panels = [_make_panel(0, 0.0, 6.0)]
    clip = _make_clip(tmp_path / "clip0.mp4", duration=5.0)
    results = [_make_panel_result(0, clip, 5.0)]

    out_dir = tmp_path / "output"
    final = compose(panels, results, transcription, video, out_dir, config)

    src_hash = _audio_stream_md5(video)
    final_hash = _audio_stream_md5(final)
    assert src_hash == final_hash


@pytest.mark.integration
def test_compose_burns_subtitles_visible_in_lower_third(tmp_path, transcription, config):
    video = _make_video_with_audio(tmp_path / "input.mp4", duration=6.0)
    panels = [_make_panel(0, 0.0, 3.0), _make_panel(1, 3.0, 6.0)]
    clip0 = _make_clip(tmp_path / "clip0.mp4", duration=3.0)
    clip1 = _make_clip(tmp_path / "clip1.mp4", duration=3.0, color="green")
    results = [
        _make_panel_result(0, clip0, 3.0),
        _make_panel_result(1, clip1, 3.0),
    ]

    out_dir = tmp_path / "output"
    final = compose(panels, results, transcription, video, out_dir, config)

    frame_path = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "1.0", "-i", str(final), "-vframes", "1", str(frame_path)],
        check=True,
        capture_output=True,
    )
    assert frame_path.exists()

    from PIL import Image

    img = Image.open(frame_path).convert("RGB")
    pixels = list(img.getdata())
    distinct = {pixel for pixel in pixels[::500]}
    assert len(distinct) > 1, (
        "frame final parece uniforme — provavelmente legendas e/ou painéis não foram queimados"
    )


def test_compose_raises_on_empty_panels(tmp_path, transcription, config):
    with pytest.raises(ComposerError, match=r"vazia"):
        compose([], [], transcription, tmp_path / "v.mp4", tmp_path / "out", config)


def test_compose_raises_on_missing_panel_results(tmp_path, transcription, config):
    panels = [_make_panel(0, 0.0, 3.0), _make_panel(1, 3.0, 6.0)]
    clip0 = _make_clip(tmp_path / "clip0.mp4", duration=3.0)
    results = [_make_panel_result(0, clip0, 3.0)]
    with pytest.raises(ComposerError, match=r"Faltam mini-clipes"):
        compose(panels, results, transcription, tmp_path / "v.mp4", tmp_path / "out", config)
