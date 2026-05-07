"""Testes unitários para integração de música no CLI (run_flow_c)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from youcut.config import PipelineConfig
from youcut.models import MusicTrack, ViralClip


def _make_config(cut_mode="social", **kwargs) -> PipelineConfig:
    defaults = {
        "anthropic_api_key": "test-key",
        "cut_mode": cut_mode,
        "social_layout_mode": "classic",
        "face_tracking": False,
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_clip(title="Título motivacional incrível") -> ViralClip:
    return ViralClip(
        title=title,
        reason="Momento de superação",
        viral_score=8.0,
        start_time=0.0,
        end_time=60.0,
        description="desc",
        hashtags=["#test"],
        thumbnail_idea="idea",
    )


def _make_track(tmp_path: Path) -> MusicTrack:
    mp3 = tmp_path / "track.mp3"
    mp3.write_bytes(b"fake-mp3")
    return MusicTrack(
        name="Upbeat Morning",
        pixabay_url="https://pixabay.com/music/test/",
        local_path=mp3,
        mood="motivacional",
        duration_s=60.0,
    )


def _make_clip_mp4(tmp_path: Path) -> Path:
    p = tmp_path / "clip_01.mp4"
    p.write_bytes(b"fake-video")
    return p


class TestCutsMusicFlagCallsProvider:
    def test_cuts_music_flag_calls_provider(self, tmp_path):
        """--music ativo + modo social → classify_mood e fetch_track são chamados."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        track = _make_track(tmp_path)
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider = MagicMock()
        mock_provider.classify_mood.return_value = "motivacional"
        mock_provider.fetch_track.return_value = track

        mock_mixer = MagicMock()
        mock_mixer.mix.return_value = clip_path

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.PixabayMusicProvider", return_value=mock_provider),
            patch("youcut.cli.MusicMixer", return_value=mock_mixer),
            patch("youcut.cli.review_clips", return_value=[]),
            patch("youcut.cli._resolve_upload_platforms", return_value=[]),
            patch("youcut.cli._show_records_table"),
        ):
            run_flow_c(
                "https://example.com/video",
                config,
                skip_review=True,
                music=True,
                music_mood=None,
            )

        mock_provider.classify_mood.assert_called_once_with(clip)
        mock_provider.fetch_track.assert_called_once_with("motivacional", 60.0)


class TestCutsMusicMoodOverride:
    def test_cuts_music_mood_override(self, tmp_path):
        """--music-mood reflexivo → classify_mood NÃO é chamado; fetch_track recebe 'reflexivo'."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        track = _make_track(tmp_path)
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider = MagicMock()
        mock_provider.classify_mood.return_value = "motivacional"
        mock_provider.fetch_track.return_value = track

        mock_mixer = MagicMock()
        mock_mixer.mix.return_value = clip_path

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.PixabayMusicProvider", return_value=mock_provider),
            patch("youcut.cli.MusicMixer", return_value=mock_mixer),
            patch("youcut.cli.review_clips", return_value=[]),
            patch("youcut.cli._resolve_upload_platforms", return_value=[]),
            patch("youcut.cli._show_records_table"),
        ):
            run_flow_c(
                "https://example.com/video",
                config,
                skip_review=True,
                music=True,
                music_mood="reflexivo",
            )

        mock_provider.classify_mood.assert_not_called()
        mock_provider.fetch_track.assert_called_once()
        call_args = mock_provider.fetch_track.call_args
        assert call_args[0][0] == "reflexivo"


class TestCutsMusicSkippedYoutubeMode:
    def test_cuts_music_skipped_youtube_mode(self, tmp_path):
        """Modo youtube → provider e mixer NÃO são instanciados."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="youtube", output_dir=tmp_path)
        clip = _make_clip()
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider_cls = MagicMock()
        mock_mixer_cls = MagicMock()

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.PixabayMusicProvider", mock_provider_cls),
            patch("youcut.cli.MusicMixer", mock_mixer_cls),
            patch("youcut.cli.review_clips", return_value=[]),
            patch("youcut.cli._resolve_upload_platforms", return_value=[]),
            patch("youcut.cli._show_records_table"),
            patch("youcut.cli._show_youtube_duration_guidance"),
        ):
            run_flow_c(
                "https://example.com/video",
                config,
                skip_review=True,
                music=True,  # flag ativa mas modo youtube ignora
            )

        # No modo youtube, nem provider nem mixer devem ser instanciados
        mock_provider_cls.assert_not_called()
        mock_mixer_cls.assert_not_called()


class TestCutsMusicFallback:
    def test_cuts_music_fallback(self, tmp_path):
        """fetch_track retorna None → pipeline continua sem exceção."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider = MagicMock()
        mock_provider.classify_mood.return_value = "motivacional"
        mock_provider.fetch_track.return_value = None  # fallback

        mock_mixer = MagicMock()

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.PixabayMusicProvider", return_value=mock_provider),
            patch("youcut.cli.MusicMixer", return_value=mock_mixer),
            patch("youcut.cli.review_clips", return_value=[]),
            patch("youcut.cli._resolve_upload_platforms", return_value=[]),
            patch("youcut.cli._show_records_table"),
        ):
            # Não deve levantar exceção
            run_flow_c(
                "https://example.com/video",
                config,
                skip_review=True,
                music=True,
            )

        mock_mixer.mix.assert_not_called()
