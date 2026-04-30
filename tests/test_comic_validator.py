import shutil
import subprocess
from pathlib import Path

import pytest

from youcut.comic.validator import (
    MAX_DURATION_SECONDS,
    SUPPORTED_EXTENSIONS,
    VideoSpec,
    VideoValidationError,
    validate_video,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe não disponível para testes do validador",
)


def _make_silent_video(path: Path, duration: float, ext: str = "mp4") -> Path:
    out = path.with_suffix(f".{ext}")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:size=320x240:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-t",
        str(duration),
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_validator_accepts_mp4(tmp_path):
    video = _make_silent_video(tmp_path / "ok", duration=2.0, ext="mp4")
    spec = validate_video(video)
    assert isinstance(spec, VideoSpec)
    assert spec.path == video
    assert 1.5 < spec.duration_seconds < 3.0
    assert spec.width == 320
    assert spec.height == 240


def test_validator_accepts_mov(tmp_path):
    video = _make_silent_video(tmp_path / "ok", duration=2.0, ext="mov")
    spec = validate_video(video)
    assert spec.path.suffix == ".mov"


def test_validator_accepts_mkv(tmp_path):
    video = _make_silent_video(tmp_path / "ok", duration=2.0, ext="mkv")
    spec = validate_video(video)
    assert spec.path.suffix == ".mkv"


def test_validator_accepts_webm(tmp_path):
    out = tmp_path / "ok.webm"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:size=320x240:duration=2",
        "-c:v",
        "libvpx",
        "-t",
        "2",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    spec = validate_video(out)
    assert spec.path.suffix == ".webm"


def test_validator_rejects_too_long(tmp_path):
    video = _make_silent_video(tmp_path / "long", duration=121.0, ext="mp4")
    with pytest.raises(VideoValidationError, match=r"Vídeo muito longo|máximo"):
        validate_video(video)


def test_validator_rejects_unknown_extension(tmp_path):
    bogus = tmp_path / "bogus.gif"
    bogus.write_bytes(b"GIF89a")
    with pytest.raises(VideoValidationError, match=r"Formato.*não suportado"):
        validate_video(bogus)


def test_validator_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(VideoValidationError, match=r"não encontrado"):
        validate_video(missing)


def test_validator_rejects_directory(tmp_path):
    folder = tmp_path / "folder.mp4"
    folder.mkdir()
    with pytest.raises(VideoValidationError, match=r"não é um arquivo"):
        validate_video(folder)


def test_supported_extensions_set():
    assert {".mp4", ".mov", ".mkv", ".webm"} <= SUPPORTED_EXTENSIONS


def test_max_duration_constant():
    assert MAX_DURATION_SECONDS == 120.0
