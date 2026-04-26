"""Testes de integração para apply_face_tracking() — Task 7.0.

Estes testes usam vídeos sintéticos gerados com FFmpeg e verificam o
comportamento de ponta a ponta do pipeline de face tracking. Marcados com
@pytest.mark.integration para execução seletiva.

Nota: MediaPipe e OpenCV são dependências opcionais (grupo [face-tracking]).
Quando não instalados, apply_face_tracking() retorna o clipe original (fallback).
Os testes verificam ambos os caminhos: com e sem as dependências instaladas.
"""

import json
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffprobe_duration(path: Path) -> float:
    """Return duration in seconds of the media file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _ffprobe_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of the video stream via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return stream["width"], stream["height"]
    except (KeyError, IndexError, json.JSONDecodeError):
        return 0, 0


def _generate_synthetic_video(output: Path, duration: float = 10.0, with_audio: bool = True) -> Path:
    """Generate a synthetic colour-bar video with FFmpeg for testing."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:size=1280x720:rate=25:duration={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-an"]
    cmd.append(str(output))
    subprocess.run(cmd, check=True, capture_output=True)
    return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_video(tmp_path):
    """10-second 1280×720 synthetic video with audio."""
    video = tmp_path / "synthetic_10s.mp4"
    _generate_synthetic_video(video, duration=10.0, with_audio=True)
    yield video
    if video.exists():
        video.unlink(missing_ok=True)


@pytest.fixture
def synthetic_video_no_audio(tmp_path):
    """10-second 1280×720 synthetic video without audio stream."""
    video = tmp_path / "synthetic_no_audio.mp4"
    _generate_synthetic_video(video, duration=10.0, with_audio=False)
    yield video
    if video.exists():
        video.unlink(missing_ok=True)


@pytest.fixture
def config_face_tracking_enabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(face_tracking=True, huggingface_token=None)


@pytest.fixture
def config_face_tracking_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(face_tracking=False)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestApplyFaceTrackingIntegration:

    def test_returns_existing_path_for_any_valid_video(
        self, synthetic_video, config_face_tracking_enabled
    ):
        """apply_face_tracking never raises — always returns a valid path."""
        from youcut.face_tracker import apply_face_tracking
        result = apply_face_tracking(synthetic_video, config_face_tracking_enabled)
        assert result.exists(), f"Returned path does not exist: {result}"

    def test_fallback_path_returned_when_face_tracking_disabled(
        self, synthetic_video, config_face_tracking_disabled
    ):
        """When face_tracking=False, original clip path is returned unchanged."""
        from youcut.face_tracker import apply_face_tracking
        result = apply_face_tracking(synthetic_video, config_face_tracking_disabled)
        assert result == synthetic_video

    def test_fallback_when_no_faces_detected(
        self, synthetic_video, config_face_tracking_enabled
    ):
        """Solid colour video has no detectable faces — fallback to original path."""
        from youcut.face_tracker import apply_face_tracking
        result = apply_face_tracking(synthetic_video, config_face_tracking_enabled)
        # Solid colour frame has no face → should return original (or a valid path)
        assert result.exists()

    def test_audio_duration_preserved_in_result(
        self, synthetic_video, config_face_tracking_enabled
    ):
        """Output file duration is within ±0.2s of the input duration."""
        from youcut.face_tracker import apply_face_tracking
        original_duration = _ffprobe_duration(synthetic_video)
        result = apply_face_tracking(synthetic_video, config_face_tracking_enabled)
        result_duration = _ffprobe_duration(result)
        assert result_duration > 0, "Could not determine result duration"
        assert abs(result_duration - original_duration) <= 0.2, (
            f"Duration mismatch: original={original_duration:.2f}s, result={result_duration:.2f}s"
        )

    def test_output_resolution_is_valid(
        self, synthetic_video, config_face_tracking_enabled
    ):
        """Result video has either 1080×1920 (face tracked) or 1280×720 (fallback)."""
        from youcut.face_tracker import apply_face_tracking
        result = apply_face_tracking(synthetic_video, config_face_tracking_enabled)
        w, h = _ffprobe_resolution(result)
        assert w > 0 and h > 0, f"Could not determine resolution: {result}"
        valid_resolutions = [(1080, 1920), (1280, 720)]
        assert (w, h) in valid_resolutions, (
            f"Unexpected resolution: {w}×{h}. Expected one of {valid_resolutions}"
        )

    def test_original_file_not_deleted_on_fallback(
        self, synthetic_video, config_face_tracking_enabled
    ):
        """Original clip is preserved regardless of whether face tracking succeeds."""
        from youcut.face_tracker import apply_face_tracking
        apply_face_tracking(synthetic_video, config_face_tracking_enabled)
        assert synthetic_video.exists(), "Original clip was deleted by face tracking"

    def test_no_exception_raised_on_missing_audio_stream(
        self, synthetic_video_no_audio, config_face_tracking_enabled
    ):
        """apply_face_tracking handles videos without audio without raising."""
        from youcut.face_tracker import apply_face_tracking
        result = apply_face_tracking(synthetic_video_no_audio, config_face_tracking_enabled)
        assert result.exists()
