import pytest
from pathlib import Path
from pydantic import ValidationError


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
