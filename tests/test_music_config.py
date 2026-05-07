"""Testes unitários para campos de música em PipelineConfig."""
import os

import pytest


def test_pipeline_config_music_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)

    from youcut.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.jamendo_client_id is None
    assert cfg.music_volume == 0.25
    assert cfg.music_duck_threshold == 0.015
    assert cfg.music_duck_ratio == 6.0
    assert cfg.music_duck_attack_ms == 200.0
    assert cfg.music_duck_release_ms == 1000.0


def test_pipeline_config_music_from_kwargs():
    from youcut.config import PipelineConfig

    cfg = PipelineConfig(
        anthropic_api_key="test-key",
        jamendo_client_id="jm-abc123",
        music_volume=0.3,
    )
    assert cfg.jamendo_client_id == "jm-abc123"
    assert cfg.music_volume == pytest.approx(0.3)


def test_pipeline_config_no_jamendo_key_does_not_raise(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)

    from youcut.config import PipelineConfig

    # Não deve levantar exceção mesmo sem JAMENDO_CLIENT_ID
    cfg = PipelineConfig()
    assert cfg.jamendo_client_id is None
