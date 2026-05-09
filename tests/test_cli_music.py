"""Testes do registro de `music` na CLI e da integração no `run_flow_c`."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from typer.testing import CliRunner

from youcut.config import PipelineConfig
from youcut.models import MusicTrack, ViralClip


runner = CliRunner()


def _make_config(cut_mode="social", **kwargs) -> PipelineConfig:
    defaults = {
        "anthropic_api_key": "test-key",
        "cut_mode": cut_mode,
        "social_layout_mode": "classic",
        "face_tracking": False,
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_clip(title: str = "Título motivacional incrível") -> ViralClip:
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
    audio = tmp_path / "track.m4a"
    audio.write_bytes(b"fake-audio")
    return MusicTrack(
        video_id="vidT1",
        name="Upbeat Morning",
        source_url="https://www.youtube.com/watch?v=vidT1",
        local_path=audio,
        mood="motivacional",
        duration_s=60.0,
    )


def _make_clip_mp4(tmp_path: Path) -> Path:
    p = tmp_path / "clip_01.mp4"
    p.write_bytes(b"fake-video")
    return p


# ── Smoke: subcomando `music sync` registrado ───────────────────────────────


class TestMusicSubcommandRegistration:
    def test_music_sync_subcommand_registered(self, monkeypatch):
        """Smoke: o subcomando `music sync` aparece no help do CLI."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.cli import app
        result = runner.invoke(app, ["music", "sync", "--help"])
        assert result.exit_code == 0
        assert "Sincroniza" in result.output

    def test_music_sync_invokes_playlist_syncer(self, monkeypatch, tmp_path):
        """Smoke: `youcut music sync` chama PlaylistSyncer.sync."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.cli import app
        from youcut.models import SyncReport

        mock_syncer = MagicMock()
        mock_syncer.sync.return_value = SyncReport(new_tracks=1, cached_tracks=2, failed_tracks=0)

        with (
            patch("youcut.music.cli.PlaylistSyncer", return_value=mock_syncer),
            patch("youcut.music.cli.TrackMoodClassifier"),
            patch("youcut.music.cli.MusicLibrary"),
        ):
            result = runner.invoke(
                app,
                ["music", "sync", "--playlist", "https://www.youtube.com/playlist?list=PLfake"],
            )
        assert result.exit_code == 0
        mock_syncer.sync.assert_called_once_with(
            "https://www.youtube.com/playlist?list=PLfake"
        )


# ── RF-25: flag --music-mood removida ───────────────────────────────────────


class TestMusicMoodFlagRemoved:
    def test_cuts_help_does_not_mention_music_mood(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.cli import app
        result = runner.invoke(app, ["cuts", "--help"])
        assert result.exit_code == 0
        assert "--music-mood" not in result.output


# ── RF-06 + RF-14: run_flow_c não dispara sync e avisa quando acervo vazio ──


class TestRunFlowCMusicIntegration:
    def test_pick_track_called_for_each_clip(self, tmp_path):
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        track = _make_track(tmp_path)
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider = MagicMock()
        mock_provider.pick_track.return_value = track

        mock_mixer = MagicMock()
        mock_mixer.mix.return_value = clip_path

        mock_library = MagicMock()
        mock_library.is_empty.return_value = False

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.MusicLibrary", return_value=mock_library),
            patch("youcut.cli.YouTubeMusicProvider", return_value=mock_provider),
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
            )

        mock_provider.pick_track.assert_called_once_with(clip)
        mock_mixer.mix.assert_called_once_with(clip_path, track)

    def test_run_flow_c_does_not_trigger_playlist_sync(self, tmp_path):
        """RF-06: pipeline de geração nunca dispara `PlaylistSyncer.sync`."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        clip_path = _make_clip_mp4(tmp_path)

        mock_library = MagicMock()
        mock_library.is_empty.return_value = True

        mock_provider = MagicMock()
        mock_provider.pick_track.return_value = None

        mock_mixer = MagicMock()
        mock_syncer_cls = MagicMock()

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.MusicLibrary", return_value=mock_library),
            patch("youcut.cli.YouTubeMusicProvider", return_value=mock_provider),
            patch("youcut.cli.MusicMixer", return_value=mock_mixer),
            patch("youcut.music.sync.PlaylistSyncer", mock_syncer_cls),
            patch("youcut.cli.review_clips", return_value=[]),
            patch("youcut.cli._resolve_upload_platforms", return_value=[]),
            patch("youcut.cli._show_records_table"),
        ):
            run_flow_c(
                "https://example.com/video",
                config,
                skip_review=True,
                music=True,
            )

        # Em nenhum momento o pipeline pode instanciar/chamar PlaylistSyncer
        mock_syncer_cls.assert_not_called()

    def test_empty_library_warns_and_skips_mixing(self, tmp_path, capsys):
        """RF-14: acervo vazio → aviso visível e clipes gerados sem trilha."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="social", output_dir=tmp_path)
        clip = _make_clip()
        clip_path = _make_clip_mp4(tmp_path)

        mock_library = MagicMock()
        mock_library.is_empty.return_value = True

        mock_provider = MagicMock()
        mock_provider.pick_track.return_value = None

        mock_mixer = MagicMock()

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.MusicLibrary", return_value=mock_library),
            patch("youcut.cli.YouTubeMusicProvider", return_value=mock_provider),
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
            )

        captured = capsys.readouterr()
        # Aviso deve aparecer ao menos uma vez (RF-14)
        full_out = captured.out + captured.err
        assert "Acervo de músicas vazio" in full_out
        # Não deve mixar nada
        mock_mixer.mix.assert_not_called()

    def test_music_skipped_in_youtube_mode(self, tmp_path):
        """Modo youtube → MusicLibrary/Provider/Mixer NÃO são instanciados."""
        from youcut.cli import run_flow_c

        config = _make_config(cut_mode="youtube", output_dir=tmp_path)
        clip = _make_clip()
        clip_path = _make_clip_mp4(tmp_path)

        mock_provider_cls = MagicMock()
        mock_mixer_cls = MagicMock()
        mock_library_cls = MagicMock()

        with (
            patch("youcut.cli.download_video", return_value=clip_path),
            patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
            patch("youcut.cli.analyze", return_value=[clip]),
            patch("youcut.cli.cut_clip", return_value=clip_path),
            patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
            patch("youcut.cli._is_editorial_social_layout", return_value=False),
            patch("youcut.cli.MusicLibrary", mock_library_cls),
            patch("youcut.cli.YouTubeMusicProvider", mock_provider_cls),
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
                music=True,  # mas modo é youtube → ignora
            )

        mock_library_cls.assert_not_called()
        mock_provider_cls.assert_not_called()
        mock_mixer_cls.assert_not_called()
