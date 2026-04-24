from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch

import pytest

from youcut.clipper import build_vertical_fill_filter
from youcut.config import PipelineConfig
from youcut.models import ViralClip
from youcut.preview import PreviewArtifact, generate_clip_preview

API_ENV_KWARGS = {"anthropic_api_key": "test-key"}


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(output_dir=tmp_path, **API_ENV_KWARGS)


@pytest.fixture
def clip():
    return ViralClip(
        title="Clipe Teste",
        reason="Gancho forte",
        viral_score=8.0,
        start_time=10.0,
        end_time=40.0,
        description="Descrição",
        hashtags=["#teste"],
        thumbnail_idea="Frame impactante",
    )


class TestPreviewArtifact:
    def test_preview_artifact_fields(self):
        artifact = PreviewArtifact(
            path=Path("clip_01_preview.jpg"),
            source_clip_index=0,
            width=1080,
            height=1920,
        )
        assert artifact.path == Path("clip_01_preview.jpg")
        assert artifact.source_clip_index == 0
        assert artifact.width == 1080
        assert artifact.height == 1920


class TestGenerateClipPreview:
    def test_preview_artifact_naming(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch("youcut.preview.subprocess.run"):
            artifact = generate_clip_preview(video_path, clip, 0, config)
        assert artifact is not None
        assert artifact.path.name == "clip_01_preview.jpg"

    def test_preview_artifact_naming_index_4(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch("youcut.preview.subprocess.run"):
            artifact = generate_clip_preview(video_path, clip, 4, config)
        assert artifact is not None
        assert artifact.path.name == "clip_05_preview.jpg"

    def test_preview_uses_fill_filter(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch("youcut.preview.subprocess.run") as mock_run:
            generate_clip_preview(video_path, clip, 0, config)
        cmd = mock_run.call_args[0][0]
        vf_index = cmd.index("-vf")
        actual_filter = cmd[vf_index + 1]
        assert actual_filter == build_vertical_fill_filter()

    def test_preview_filter_matches_export_filter(self):
        assert build_vertical_fill_filter() == build_vertical_fill_filter(1080, 1920)

    def test_preview_failure_does_not_raise(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch(
            "youcut.preview.subprocess.run",
            side_effect=CalledProcessError(1, "ffmpeg", stderr=b"error"),
        ):
            result = generate_clip_preview(video_path, clip, 0, config)
        assert result is None

    def test_preview_artifact_path_in_clip_dir(self, config, clip, tmp_path):
        video_path = tmp_path / "myvideo.mp4"
        with patch("youcut.preview.subprocess.run"):
            artifact = generate_clip_preview(video_path, clip, 0, config)
        assert artifact is not None
        assert artifact.path.parent == tmp_path / "myvideo"

    def test_preview_artifact_dimensions(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch("youcut.preview.subprocess.run"):
            artifact = generate_clip_preview(video_path, clip, 0, config)
        assert artifact is not None
        assert artifact.width == 1080
        assert artifact.height == 1920

    def test_preview_source_clip_index(self, config, clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        with patch("youcut.preview.subprocess.run"):
            artifact = generate_clip_preview(video_path, clip, 2, config)
        assert artifact is not None
        assert artifact.source_clip_index == 2
