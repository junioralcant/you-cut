"""Testes das chaves `comic_remotion_*` e do enum `comic_animation_engine` (Task 1.0)."""

import pytest
from pydantic import ValidationError


def test_comic_animation_engine_accepts_remotion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine="remotion")
    assert config.comic_animation_engine == "remotion"


@pytest.mark.parametrize("engine", ["scenes", "panels", "prunaai", "remotion"])
def test_comic_animation_engine_accepts_all_known_engines(monkeypatch, engine):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine=engine)
    assert config.comic_animation_engine == engine


def test_comic_animation_engine_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    with pytest.raises(ValidationError):
        PipelineConfig(comic_animation_engine="xyz")  # type: ignore[arg-type]


def test_comic_animation_engine_default_is_scenes(monkeypatch):
    """Não promover `remotion` a default — `scenes` continua como engine padrão."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig()
    assert config.comic_animation_engine == "scenes"


def test_comic_remotion_defaults(monkeypatch):
    """Os 8 defaults devem bater com a tabela da techspec §Modelos de Dados."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig()
    assert config.comic_remotion_enabled_default is False
    assert config.comic_remotion_fps == 30
    assert config.comic_remotion_node_bin == "node"
    assert config.comic_remotion_concurrency is None
    assert config.comic_remotion_studio_port == 3000
    assert config.comic_remotion_kenburns_default_scale == 1.12
    assert config.comic_remotion_idle_blink_period_sec == 4.5
    assert config.comic_remotion_pyphen_locale_fallback == "pt_BR"


def test_comic_remotion_overrides_via_kwargs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(
        comic_remotion_fps=60,
        comic_remotion_node_bin="/usr/local/bin/node",
        comic_remotion_concurrency=4,
        comic_remotion_studio_port=4000,
        comic_remotion_kenburns_default_scale=1.25,
        comic_remotion_idle_blink_period_sec=3.0,
        comic_remotion_pyphen_locale_fallback="en_US",
    )
    assert config.comic_remotion_fps == 60
    assert config.comic_remotion_node_bin == "/usr/local/bin/node"
    assert config.comic_remotion_concurrency == 4
    assert config.comic_remotion_studio_port == 4000
    assert config.comic_remotion_kenburns_default_scale == 1.25
    assert config.comic_remotion_idle_blink_period_sec == 3.0
    assert config.comic_remotion_pyphen_locale_fallback == "en_US"


def test_comic_remotion_overrides_via_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COMIC_REMOTION_FPS", "24")
    monkeypatch.setenv("COMIC_REMOTION_STUDIO_PORT", "7777")
    from youcut.config import PipelineConfig

    config = PipelineConfig()
    assert config.comic_remotion_fps == 24
    assert config.comic_remotion_studio_port == 7777
