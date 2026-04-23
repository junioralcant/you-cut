import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.clipper import PADDING, check_ffmpeg, cut_clip
from youcut.config import PipelineConfig
from youcut.models import ViralClip


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(
        anthropic_api_key="test-key",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def viral_clip():
    return ViralClip(
        title="Test Clip",
        reason="Great hook",
        viral_score=8.5,
        start_time=10.0,
        end_time=50.0,
        description="A test clip",
        hashtags=["#test"],
        thumbnail_idea="Test thumbnail",
    )


class TestCheckFfmpeg:
    def test_raises_when_ffmpeg_not_found(self):
        with patch("youcut.clipper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="FFmpeg não encontrado"):
                check_ffmpeg()

    def test_passes_when_ffmpeg_found(self):
        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"):
            check_ffmpeg()


class TestCutClip:
    def _run_cut(self, video_path, clip, index, config):
        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("youcut.clipper.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result = cut_clip(video_path, clip, index, config)
                return result, mock_run.call_args[0][0]

    def test_ffmpeg_command_includes_scale_1080x1920(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        assert "scale=1080:1920" in " ".join(cmd)

    def test_ffmpeg_command_includes_pad(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        assert "pad=1080:1920" in " ".join(cmd)

    def test_padding_applied_to_start_time(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == pytest.approx(viral_clip.start_time - PADDING)

    def test_padding_applied_to_end_time(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        t_idx = cmd.index("-t")
        expected_duration = (viral_clip.end_time + PADDING) - (viral_clip.start_time - PADDING)
        assert float(cmd[t_idx + 1]) == pytest.approx(expected_duration)

    def test_output_named_clip_01_for_index_0(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "myvideo.mp4"
        video_path.touch()
        result, _ = self._run_cut(video_path, viral_clip, 0, config)
        assert result.name == "clip_01.mp4"

    def test_output_named_clip_02_for_index_1(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "myvideo.mp4"
        video_path.touch()
        result, _ = self._run_cut(video_path, viral_clip, 1, config)
        assert result.name == "clip_02.mp4"

    def test_output_in_video_stem_subdirectory(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "myvideo.mp4"
        video_path.touch()
        result, _ = self._run_cut(video_path, viral_clip, 0, config)
        assert result.parent.name == "myvideo"
        assert result.parent.parent == config.output_dir

    def test_raises_when_ffmpeg_missing(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        with patch("youcut.clipper.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="FFmpeg não encontrado"):
                cut_clip(video_path, viral_clip, 0, config)

    def test_raises_called_process_error_on_nonzero_returncode(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        error = subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"some ffmpeg error")
        with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("youcut.clipper.subprocess.run", side_effect=error):
                with pytest.raises(subprocess.CalledProcessError):
                    cut_clip(video_path, viral_clip, 0, config)

    def test_start_time_clamped_to_zero_when_less_than_padding(self, config, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        clip = ViralClip(
            title="Test",
            reason="Test",
            viral_score=5.0,
            start_time=0.05,
            end_time=30.0,
            description="Test",
            hashtags=[],
            thumbnail_idea="Test",
        )
        _, cmd = self._run_cut(video_path, clip, 0, config)
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == 0.0

    def test_blur_background_uses_filter_complex_with_boxblur(self, tmp_path, viral_clip):
        config = PipelineConfig(
            anthropic_api_key="test-key",
            output_dir=tmp_path / "output",
            blur_background=True,
        )
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        assert "-filter_complex" in cmd
        assert "boxblur" in " ".join(cmd)

    def test_default_uses_vf_not_filter_complex(self, config, viral_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = self._run_cut(video_path, viral_clip, 0, config)
        assert "-vf" in cmd
        assert "-filter_complex" not in cmd
