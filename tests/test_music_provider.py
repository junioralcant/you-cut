"""Testes unitários para PixabayMusicProvider (Jamendo backend)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.config import PipelineConfig
from youcut.models import MusicTrack, ViralClip
from youcut.music_provider import PixabayMusicProvider


def _make_config(**kwargs) -> PipelineConfig:
    defaults = {"anthropic_api_key": "test-key", "jamendo_client_id": "jtest123"}
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_clip(title="", reason="", style="") -> ViralClip:
    return ViralClip(
        title=title,
        reason=reason,
        viral_score=7.0,
        start_time=0.0,
        end_time=60.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="idea",
        social_visual_style=style,
    )


class TestClassifyMood:
    def test_classify_mood_motivacional(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(title="Minha jornada de motivação e superação")
        assert provider.classify_mood(clip) == "motivacional"

    def test_classify_mood_reflexivo(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(reason="Uma reflexão profunda sobre paz e calma")
        assert provider.classify_mood(clip) == "reflexivo"

    def test_classify_mood_energico(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(title="Treino intenso com muita energia e ação")
        assert provider.classify_mood(clip) == "energico"

    def test_classify_mood_emocional(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(reason="Uma história de amor e saudade profunda")
        assert provider.classify_mood(clip) == "emocional"

    def test_classify_mood_feliz(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(title="Celebração de alegria e felicidade")
        assert provider.classify_mood(clip) == "feliz"

    def test_classify_mood_dramatico(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(style="Cena dramática com tensão e conflito intenso")
        assert provider.classify_mood(clip) == "dramatico"

    def test_classify_mood_fallback(self):
        provider = PixabayMusicProvider(_make_config())
        clip = _make_clip(title="Conteúdo genérico sem palavras-chave conhecidas")
        # Sem keywords — retorna default "motivacional"
        result = provider.classify_mood(clip)
        assert result == "motivacional"


class TestFetchTrackCacheHit:
    def test_fetch_track_cache_hit(self, tmp_path, monkeypatch):
        """Arquivo MP3 existente em cache → retorna sem chamar httpx."""
        cache_dir = tmp_path / "motivacional"
        cache_dir.mkdir(parents=True)
        mp3_file = cache_dir / "test-track.mp3"
        mp3_file.write_bytes(b"fake-mp3-data")

        monkeypatch.setattr(
            "youcut.music_provider._MUSIC_CACHE_DIR", tmp_path
        )

        provider = PixabayMusicProvider(_make_config())

        with patch("httpx.Client") as mock_client_cls:
            result = provider.fetch_track("motivacional", 60.0)
            mock_client_cls.assert_not_called()

        assert result is not None
        assert result.mood == "motivacional"
        assert result.local_path == mp3_file


def _make_mock_http_client(api_json: dict, mp3_content: bytes = b"fake-mp3"):
    """Cria mock do httpx.Client para simular resposta Jamendo e download de MP3."""
    mock_client = MagicMock()

    api_resp = MagicMock()
    api_resp.raise_for_status = MagicMock()
    api_resp.json.return_value = api_json

    download_resp = MagicMock()
    download_resp.raise_for_status = MagicMock()
    download_resp.iter_bytes.return_value = [mp3_content]
    download_resp.__enter__ = MagicMock(return_value=download_resp)
    download_resp.__exit__ = MagicMock(return_value=False)

    mock_client.get.return_value = api_resp
    mock_client.stream.return_value = download_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    return mock_client


def _jamendo_results(tracks: list[dict]) -> dict:
    return {"headers": {"status": "success", "code": 0}, "results": tracks}


class TestFetchTrackApiCall:
    def test_fetch_track_api_call(self, tmp_path, monkeypatch):
        """Mock httpx retorna results → baixa e salva MP3; retorna MusicTrack."""
        monkeypatch.setattr("youcut.music_provider._MUSIC_CACHE_DIR", tmp_path)

        api_response = _jamendo_results([
            {
                "id": "1234",
                "name": "Upbeat Morning",
                "duration": 120,
                "artist_name": "Test Artist",
                "audio": "https://mp3d.jamendo.com/?trackid=1234&format=mp32",
                "shareurl": "https://www.jamendo.com/track/1234/upbeat-morning",
            }
        ])

        mock_client = _make_mock_http_client(api_response)
        with patch("youcut.music_provider.httpx.Client", return_value=mock_client):
            provider = PixabayMusicProvider(_make_config())
            result = provider.fetch_track("motivacional", 90.0)

        assert result is not None
        assert result.name == "Upbeat Morning"
        assert result.pixabay_url == "https://www.jamendo.com/track/1234/upbeat-morning"
        assert result.local_path.exists()
        assert result.mood == "motivacional"
        assert result.local_path.parent == tmp_path / "motivacional"

    def test_fetch_track_api_empty_falls_through(self, tmp_path, monkeypatch):
        """Mock httpx retorna results=[] em todas as tentativas → retorna None."""
        monkeypatch.setattr("youcut.music_provider._MUSIC_CACHE_DIR", tmp_path)

        mock_client = _make_mock_http_client(_jamendo_results([]))
        with patch("youcut.music_provider.httpx.Client", return_value=mock_client):
            provider = PixabayMusicProvider(_make_config())
            result = provider.fetch_track("motivacional", 60.0)

        assert result is None

    def test_fetch_track_no_api_key(self, tmp_path, monkeypatch):
        """jamendo_client_id=None → retorna None imediatamente."""
        monkeypatch.setattr("youcut.music_provider._MUSIC_CACHE_DIR", tmp_path)

        provider = PixabayMusicProvider(_make_config(jamendo_client_id=None))

        with patch("youcut.music_provider.httpx.Client") as mock_client_cls:
            result = provider.fetch_track("motivacional", 60.0)
            mock_client_cls.assert_not_called()

        assert result is None
