from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.clipper import PADDING, cut_clip
from youcut.config import PipelineConfig
from youcut.models import ViralClip


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(
        anthropic_api_key="test-key",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def youtube_clip():
    return ViralClip(
        title="YouTube Clip",
        reason="Great segment",
        viral_score=8.0,
        start_time=300.0,
        end_time=900.0,
        description="A landscape clip",
        hashtags=["#youtube"],
        thumbnail_idea="Landscape thumbnail",
        cut_mode="youtube",
    )


@pytest.fixture
def social_clip():
    return ViralClip(
        title="Social Clip",
        reason="Viral moment",
        viral_score=9.0,
        start_time=10.0,
        end_time=50.0,
        description="A vertical clip",
        hashtags=["#social"],
        thumbnail_idea="Vertical thumbnail",
        cut_mode="social",
    )


def _run_cut(video_path, clip, index, config):
    with patch("youcut.clipper.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("youcut.clipper.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = cut_clip(video_path, clip, index, config)
            return result, mock_run.call_args[0][0]


class TestYoutubeMode:
    def test_uses_c_copy(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        assert "-c" in cmd
        copy_idx = cmd.index("-c")
        assert cmd[copy_idx + 1] == "copy"

    def test_no_vf_filter(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        assert "-vf" not in cmd

    def test_no_scale_filter(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        # check that no filter argument contains "scale" (excludes paths)
        filter_flags = {"-vf", "-filter_complex", "-filter:v"}
        filter_values = [cmd[i + 1] for i, arg in enumerate(cmd) if arg in filter_flags]
        assert not any("scale" in v for v in filter_values)

    def test_no_crop_filter(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        filter_flags = {"-vf", "-filter_complex", "-filter:v"}
        filter_values = [cmd[i + 1] for i, arg in enumerate(cmd) if arg in filter_flags]
        assert not any("crop" in v for v in filter_values)

    def test_no_filter_complex(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        assert "-filter_complex" not in cmd

    def test_ss_matches_start_time_exactly(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == pytest.approx(youtube_clip.start_time)

    def test_duration_matches_end_minus_start(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, youtube_clip, 0, config)
        t_idx = cmd.index("-t")
        expected = youtube_clip.end_time - youtube_clip.start_time
        assert float(cmd[t_idx + 1]) == pytest.approx(expected)

    def test_output_file_named_correctly(self, config, youtube_clip, tmp_path):
        video_path = tmp_path / "myvideo.mp4"
        video_path.touch()
        result, _ = _run_cut(video_path, youtube_clip, 0, config)
        assert result.name == "clip_01.mp4"

    def test_no_padding_applied_near_zero_start(self, config, tmp_path):
        clip = ViralClip(
            title="Early clip",
            reason="Test",
            viral_score=5.0,
            start_time=0.05,
            end_time=60.0,
            description="",
            hashtags=[],
            thumbnail_idea="",
            cut_mode="youtube",
        )
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, clip, 0, config)
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == pytest.approx(0.05)  # no clamping to 0


class TestSocialModeNonRegression:
    def test_social_uses_vf_filter(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        assert "-vf" in cmd

    def test_social_contains_scale_filter(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        assert "scale=1080:1920" in " ".join(cmd)

    def test_social_contains_crop_filter(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        assert "crop=1080:1920" in " ".join(cmd)

    def test_social_does_not_use_c_copy(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        cmd_str = " ".join(cmd)
        assert "-c copy" not in cmd_str

    def test_social_applies_padding_to_start(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        ss_idx = cmd.index("-ss")
        assert float(cmd[ss_idx + 1]) == pytest.approx(social_clip.start_time - PADDING)

    def test_social_applies_padding_to_duration(self, config, social_clip, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        t_idx = cmd.index("-t")
        expected = (social_clip.end_time + PADDING) - (social_clip.start_time - PADDING)
        assert float(cmd[t_idx + 1]) == pytest.approx(expected)

    def test_social_blur_mode_uses_filter_complex(self, tmp_path, social_clip):
        config = PipelineConfig(
            anthropic_api_key="test-key",
            output_dir=tmp_path / "output",
            vertical_fill_mode="blur_background",
        )
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        _, cmd = _run_cut(video_path, social_clip, 0, config)
        assert "-filter_complex" in cmd
        assert "boxblur" in " ".join(cmd)
