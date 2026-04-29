import pytest
from pathlib import Path
from pydantic import ValidationError


def test_missing_api_key_raises_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    from youcut.config import PipelineConfig
    with pytest.raises(ValidationError):
        PipelineConfig()


def test_default_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("CLIP_COUNT", raising=False)
    monkeypatch.delenv("SUBTITLE_STYLE", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)

    from youcut.config import PipelineConfig
    config = PipelineConfig()

    assert config.clip_count == 5
    assert config.subtitle_style == "word"
    assert config.output_dir == Path("output")
    assert config.whisper_model == "medium"
    assert config.dry_run is False
    assert config.upload is False
    assert config.platforms == ["youtube", "instagram", "tiktok"]
    assert config.clips is None


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "my-real-key")
    monkeypatch.setenv("WHISPER_MODEL", "large-v3")
    monkeypatch.setenv("CLIP_COUNT", "10")
    monkeypatch.setenv("SUBTITLE_STYLE", "phrase")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/clips")

    from youcut.config import PipelineConfig
    config = PipelineConfig()

    assert config.whisper_model == "large-v3"
    assert config.clip_count == 10
    assert config.subtitle_style == "phrase"
    assert config.output_dir == Path("/tmp/clips")
    assert config.anthropic_api_key == "my-real-key"


def test_extra_upload_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", "/tmp/client_secret.json")
    monkeypatch.setenv("INSTAGRAM_APP_ID", "abc")
    monkeypatch.setenv("INSTAGRAM_APP_SECRET", "def")
    monkeypatch.setenv("TIKTOK_CLIENT_KEY", "ghi")

    from youcut.config import PipelineConfig
    config = PipelineConfig()

    assert config.anthropic_api_key == "test-key"
    assert config.upload is False


def test_api_key_loaded(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.anthropic_api_key == "sk-ant-test-123"


def test_dry_run_default_false(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.dry_run is False


def test_dry_run_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DRY_RUN", "true")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.dry_run is True


def test_vertical_fill_mode_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.vertical_fill_mode == "fill_crop"


def test_vertical_fill_mode_accepts_blur_background(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(vertical_fill_mode="blur_background")
    assert config.vertical_fill_mode == "blur_background"


def test_title_overlay_default_false(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.title_overlay is False


def test_title_overlay_can_be_set_true(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(title_overlay=True)
    assert config.title_overlay is True


def test_upload_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig()
    assert config.upload is False
    assert config.platforms == ["youtube", "instagram", "tiktok"]
    assert config.clips is None


def test_upload_fields_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(upload=True, platforms=["youtube"], clips=[1, 3])
    assert config.upload is True
    assert config.platforms == ["youtube"]
    assert config.clips == [1, 3]


def test_pipeline_config_new_fields_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(_env_file=None, openai_api_key=None)
    assert config.cut_mode == "social"
    assert config.max_clips is None
    assert config.openai_api_key is None
    assert config.session_timeout_minutes == 7
    assert config.social_layout_mode == "classic"
    assert config.social_layout_title_color_mode == "engagement_default"
    assert config.social_layout_title_bg_color == "#F4C400"


def test_pipeline_config_accepts_social_layout_overrides(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(
        social_layout_mode="speaker_bottom_ai_top",
        social_layout_title_color_mode="orange",
        social_layout_top_image_height=840,
    )
    assert config.social_layout_mode == "speaker_bottom_ai_top"
    assert config.social_layout_title_color_mode == "orange"
    assert config.social_layout_top_image_height == 840


def test_pipeline_config_accepts_youtube_cut_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(cut_mode="youtube", max_clips=5)
    assert config.cut_mode == "youtube"
    assert config.max_clips == 5


def test_pipeline_config_session_timeout_minutes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    config = PipelineConfig(session_timeout_minutes=10)
    assert config.session_timeout_minutes == 10
