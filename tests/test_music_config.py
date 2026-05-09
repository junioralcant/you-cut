"""Testes unitários para campos de música em PipelineConfig."""


def test_pipeline_config_youtube_music_playlist_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("YOUTUBE_MUSIC_PLAYLIST_URL", raising=False)

    from youcut.config import PipelineConfig

    cfg = PipelineConfig()
    assert isinstance(cfg.youtube_music_playlist_url, str)
    assert cfg.youtube_music_playlist_url, "default não pode ser vazio"
    assert "youtube.com/playlist" in cfg.youtube_music_playlist_url


def test_pipeline_config_youtube_music_playlist_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(
        "YOUTUBE_MUSIC_PLAYLIST_URL",
        "https://www.youtube.com/playlist?list=PLcustom",
    )

    from youcut.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.youtube_music_playlist_url == "https://www.youtube.com/playlist?list=PLcustom"


def test_pipeline_config_jamendo_keys_removed(monkeypatch):
    """jamendo_client_id, music_volume, music_duck_* não devem mais existir como campos."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from youcut.config import PipelineConfig

    cfg = PipelineConfig()
    for removed_attr in (
        "jamendo_client_id",
        "pixabay_api_key",
        "music_volume",
        "music_duck_threshold",
        "music_duck_ratio",
        "music_duck_attack_ms",
        "music_duck_release_ms",
    ):
        assert not hasattr(cfg, removed_attr), (
            f"{removed_attr} deveria ter sido removido de PipelineConfig"
        )


def test_pipeline_config_legacy_jamendo_env_does_not_break(monkeypatch):
    """RF-24: ter JAMENDO_CLIENT_ID em .env/env não deve quebrar o boot (extra='ignore')."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("JAMENDO_CLIENT_ID", "legacy-jamendo-key")
    monkeypatch.setenv("MUSIC_VOLUME", "0.5")
    monkeypatch.setenv("PIXABAY_API_KEY", "legacy-pixabay")

    from youcut.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.youtube_music_playlist_url
    assert not hasattr(cfg, "jamendo_client_id")
    assert not hasattr(cfg, "music_volume")
    assert not hasattr(cfg, "pixabay_api_key")
