"""Testes unitários e de integração para _run_single_source_pipeline e _run_social_from_clips."""

import threading as _threading
import time as _real_time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from youcut.models import ClipRecord, SessionData, ViralClip


def _make_clip_record(title: str, tmp_path: Path, *, approved: bool = True) -> ClipRecord:
    p = tmp_path / f"{title}.mp4"
    p.touch()
    return ClipRecord(
        title=title,
        start_time=0.0,
        end_time=60.0,
        clip_path=p,
        thumbnail_path=None,
        approved=approved,
    )


def _make_viral_clip(title: str = "Clipe Teste") -> ViralClip:
    return ViralClip(
        title=title,
        reason="Gancho forte",
        viral_score=8.0,
        start_time=0.0,
        end_time=30.0,
        description="Descrição",
        hashtags=["#teste"],
        thumbnail_idea="Frame inicial",
        thumbnail_text="MOMENTO IMPACTANTE",
    )


@pytest.fixture
def social_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")


# ---------------------------------------------------------------------------
# _run_single_source_pipeline
# ---------------------------------------------------------------------------

class TestRunSingleSourcePipeline:
    def test_local_path_passed_to_download_video(self, tmp_path, monkeypatch):
        """download_video deve receber o path local (não URL)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        local_mp4 = tmp_path / "clip_01.mp4"
        local_mp4.touch()

        fake_viral = _make_viral_clip()
        fake_transcription = MagicMock()
        fake_clip_path = tmp_path / "output" / "clip_01" / "clip_0.mp4"
        fake_clip_path.parent.mkdir(parents=True, exist_ok=True)
        fake_clip_path.touch()
        fake_metadata_path = tmp_path / "output" / "clip_01" / "clip_0.json"
        fake_metadata_path.touch()

        with (
            patch("youcut.cli.download_video", return_value=local_mp4) as mock_dl,
            patch("youcut.cli.transcribe", return_value=fake_transcription),
            patch("youcut.cli.analyze", return_value=[fake_viral]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.generate_clip_preview", return_value=None),
            patch("youcut.cli.add_captions"),
            patch("youcut.cli.export_metadata", return_value=fake_metadata_path),
            patch("youcut.cli.review_clips", side_effect=lambda vc, cr, _: cr),
            patch("youcut.cli.upload_clips"),
        ):
            from youcut.cli import _run_single_source_pipeline
            records = _run_single_source_pipeline(
                str(local_mp4),
                config,
                skip_review=True,
                upload=True,
            )

        mock_dl.assert_called_once_with(str(local_mp4), config.output_dir / "downloads")
        assert len(records) == 1

    def test_captions_applied_is_true(self, tmp_path, monkeypatch):
        """ClipRecord retornado deve ter captions_applied=True."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        local_mp4 = tmp_path / "clip_01.mp4"
        local_mp4.touch()

        fake_viral = _make_viral_clip()
        fake_clip_path = tmp_path / "clip_0.mp4"
        fake_clip_path.touch()
        fake_metadata_path = tmp_path / "clip_0.json"
        fake_metadata_path.touch()

        with (
            patch("youcut.cli.download_video", return_value=local_mp4),
            patch("youcut.cli.transcribe", return_value=MagicMock()),
            patch("youcut.cli.analyze", return_value=[fake_viral]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.generate_clip_preview", return_value=None),
            patch("youcut.cli.add_captions"),
            patch("youcut.cli.export_metadata", return_value=fake_metadata_path),
            patch("youcut.cli.upload_clips"),
        ):
            from youcut.cli import _run_single_source_pipeline
            records = _run_single_source_pipeline(
                str(local_mp4),
                config,
                skip_review=True,
                upload=True,
            )

        assert len(records) == 1
        assert records[0].captions_applied is True

    def test_upload_clips_called_when_upload_true(self, tmp_path, monkeypatch):
        """upload_clips deve ser chamado quando upload=True."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        local_mp4 = tmp_path / "clip.mp4"
        local_mp4.touch()
        fake_clip_path = tmp_path / "out.mp4"
        fake_clip_path.touch()
        fake_metadata = tmp_path / "out.json"
        fake_metadata.touch()

        with (
            patch("youcut.cli.download_video", return_value=local_mp4),
            patch("youcut.cli.transcribe", return_value=MagicMock()),
            patch("youcut.cli.analyze", return_value=[_make_viral_clip()]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.generate_clip_preview", return_value=None),
            patch("youcut.cli.add_captions"),
            patch("youcut.cli.export_metadata", return_value=fake_metadata),
            patch("youcut.cli.upload_clips") as mock_upload,
        ):
            from youcut.cli import _run_single_source_pipeline
            _run_single_source_pipeline(
                str(local_mp4),
                config,
                skip_review=True,
                upload=True,
                platforms=["tiktok", "instagram"],
            )

        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args
        assert call_kwargs.kwargs.get("platforms") == ["tiktok", "instagram"]

    def test_upload_clips_not_called_when_upload_false(self, tmp_path, monkeypatch):
        """upload_clips não deve ser chamado quando upload=False."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        local_mp4 = tmp_path / "clip.mp4"
        local_mp4.touch()
        fake_clip_path = tmp_path / "out.mp4"
        fake_clip_path.touch()
        fake_metadata = tmp_path / "out.json"
        fake_metadata.touch()

        with (
            patch("youcut.cli.download_video", return_value=local_mp4),
            patch("youcut.cli.transcribe", return_value=MagicMock()),
            patch("youcut.cli.analyze", return_value=[_make_viral_clip()]),
            patch("youcut.cli.cut_clip", return_value=fake_clip_path),
            patch("youcut.cli.generate_clip_preview", return_value=None),
            patch("youcut.cli.add_captions"),
            patch("youcut.cli.export_metadata", return_value=fake_metadata),
            patch("youcut.cli.upload_clips") as mock_upload,
        ):
            from youcut.cli import _run_single_source_pipeline
            _run_single_source_pipeline(str(local_mp4), config, skip_review=True, upload=False)

        mock_upload.assert_not_called()

    def test_download_error_raises_pipeline_error(self, tmp_path, monkeypatch):
        """Erro no download levanta _PipelineError (chamador decide ação adequada)."""
        import pytest
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        from youcut.downloader import VideoDownloadError
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        with patch("youcut.cli.download_video", side_effect=VideoDownloadError("falha")):
            from youcut.cli import _run_single_source_pipeline, _PipelineError
            with pytest.raises(_PipelineError):
                _run_single_source_pipeline("http://invalid", config, skip_review=True)

    def test_returns_empty_list_on_empty_viral_clips(self, tmp_path, monkeypatch):
        """Retorna [] quando analyze não encontra nenhum clipe."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="social", output_dir=tmp_path / "output")

        local_mp4 = tmp_path / "clip.mp4"
        local_mp4.touch()

        with (
            patch("youcut.cli.download_video", return_value=local_mp4),
            patch("youcut.cli.transcribe", return_value=MagicMock()),
            patch("youcut.cli.analyze", return_value=[]),
            patch("youcut.cli.upload_clips") as mock_upload,
        ):
            from youcut.cli import _run_single_source_pipeline
            records = _run_single_source_pipeline(str(local_mp4), config, skip_review=True, upload=True)

        assert records == []
        mock_upload.assert_not_called()


# ---------------------------------------------------------------------------
# _run_social_from_clips
# ---------------------------------------------------------------------------

class TestRunSocialFromClips:
    def test_calls_pipeline_once_per_clip(self, tmp_path, monkeypatch):
        """_run_single_source_pipeline é chamado uma vez por ClipRecord."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        clip1 = _make_clip_record("clip1", tmp_path)
        clip2 = _make_clip_record("clip2", tmp_path)

        with patch("youcut.cli._run_single_source_pipeline", return_value=[]) as mock_pipeline:
            from youcut.cli import _run_social_from_clips
            _run_social_from_clips([clip1, clip2], config, skip_review=True)

        assert mock_pipeline.call_count == 2
        paths_called = [c.args[0] for c in mock_pipeline.call_args_list]
        assert str(clip1.clip_path) in paths_called
        assert str(clip2.clip_path) in paths_called

    def test_empty_list_returns_empty_without_calling_pipeline(self, tmp_path, monkeypatch):
        """Lista vazia de clipes não aciona o pipeline."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        with patch("youcut.cli._run_single_source_pipeline") as mock_pipeline:
            from youcut.cli import _run_social_from_clips
            result = _run_social_from_clips([], config, skip_review=True)

        assert result == []
        mock_pipeline.assert_not_called()

    def test_skip_review_propagated_to_pipeline(self, tmp_path, monkeypatch):
        """skip_review é repassado para _run_single_source_pipeline."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        clip = _make_clip_record("clip1", tmp_path)

        with patch("youcut.cli._run_single_source_pipeline", return_value=[]) as mock_pipeline:
            from youcut.cli import _run_social_from_clips
            _run_social_from_clips([clip], config, skip_review=True, upload=True, platforms=["tiktok"])

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("skip_review") is True
        assert call_kwargs.kwargs.get("upload") is True
        assert call_kwargs.kwargs.get("platforms") == ["tiktok"]

    def test_upload_propagated_to_pipeline(self, tmp_path, monkeypatch):
        """upload=False é repassado para _run_single_source_pipeline."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        clip = _make_clip_record("clip1", tmp_path)

        with patch("youcut.cli._run_single_source_pipeline", return_value=[]) as mock_pipeline:
            from youcut.cli import _run_social_from_clips
            _run_social_from_clips([clip], config, skip_review=True, upload=False)

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("upload") is False

    def test_accumulates_records_from_all_clips(self, tmp_path, monkeypatch):
        """Registros de todos os clipes são acumulados no resultado."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        clip1 = _make_clip_record("clip1", tmp_path)
        clip2 = _make_clip_record("clip2", tmp_path)

        record_a = _make_clip_record("result_a", tmp_path)
        record_b = _make_clip_record("result_b", tmp_path)

        side_effects = [[record_a], [record_b]]
        with patch("youcut.cli._run_single_source_pipeline", side_effect=side_effects):
            from youcut.cli import _run_social_from_clips
            result = _run_social_from_clips([clip1, clip2], config, skip_review=True)

        assert len(result) == 2
        assert record_a in result
        assert record_b in result


# ---------------------------------------------------------------------------
# Integração: offer_flow_b não chama run_flow_b
# ---------------------------------------------------------------------------

class TestOfferFlowBDoesNotCallRunFlowB:
    def _make_session(self, tmp_path: Path, monkeypatch) -> "SessionData":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from datetime import datetime
        clip = _make_clip_record("clip1", tmp_path)
        return SessionData(
            session_id="test-session",
            source_url="https://youtube.com/watch?v=abc",
            cut_mode="youtube",
            transcription_cache_path=tmp_path / "cache.json",
            clips=[clip],
            created_at=datetime.now(),
            output_dir=tmp_path / "output",
        )

    def test_auto_timeout_calls_social_not_run_flow_b(self, tmp_path, monkeypatch):
        """Após timeout, _run_social_from_clips é chamado em vez de run_flow_b."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output", session_timeout_minutes=0)
        session = self._make_session(tmp_path, monkeypatch)

        # Timer síncrono: dispara o callback antes de selection_thread ter chance de rodar.
        class SyncTimer:
            daemon = True
            def __init__(self, seconds, fn):
                self._fn = fn
            def start(self):
                self._fn()
            def cancel(self):
                pass

        # questionary.confirm bloqueia permanentemente (daemon thread – não causa hang).
        # Garante que user_responded nunca é setado antes da verificação da condição.
        _never = _threading.Event()

        def blocking_confirm(*args, **kwargs):
            _never.wait()  # bloqueia para sempre; thread é daemon
            return MagicMock(ask=MagicMock(return_value=None))

        with (
            patch("youcut.cli._run_social_from_clips", return_value=[]) as mock_social,
            patch("youcut.cli.run_flow_b") as mock_flow_b,
            patch("youcut.cli.questionary.confirm", side_effect=blocking_confirm),
            patch("youcut.cli.time.sleep"),
            patch("threading.Timer", SyncTimer),
        ):
            from youcut.cli import offer_flow_b
            offer_flow_b(session=session, config=config, skip_review=True, upload=False)

        mock_social.assert_called_once()
        mock_flow_b.assert_not_called()

    def test_user_selection_calls_social_not_run_flow_b(self, tmp_path, monkeypatch):
        """Após seleção do usuário, _run_social_from_clips é chamado em vez de run_flow_b."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output", session_timeout_minutes=7)
        session = self._make_session(tmp_path, monkeypatch)

        with (
            patch("youcut.cli._run_social_from_clips", return_value=[]) as mock_social,
            patch("youcut.cli.run_flow_b") as mock_flow_b,
            patch("youcut.cli.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=False))),
        ):
            from youcut.cli import offer_flow_b
            offer_flow_b(session=session, config=config, skip_review=True, upload=False)

        mock_social.assert_called_once()
        mock_flow_b.assert_not_called()


# ---------------------------------------------------------------------------
# Integração: _run_auto_pipeline não chama run_flow_b
# ---------------------------------------------------------------------------

class TestRunAutoPipelineDoesNotCallRunFlowB:
    def _make_session(self, tmp_path: Path, monkeypatch) -> "SessionData":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from datetime import datetime
        clip = _make_clip_record("clip1", tmp_path, approved=True)
        return SessionData(
            session_id="auto-session",
            source_url="https://youtube.com/watch?v=xyz",
            cut_mode="youtube",
            transcription_cache_path=tmp_path / "cache.json",
            clips=[clip],
            created_at=datetime.now(),
            output_dir=tmp_path / "output",
        )

    def test_does_not_call_run_flow_b(self, tmp_path, monkeypatch):
        """_run_auto_pipeline chama _run_social_from_clips em vez de run_flow_b."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")
        session = self._make_session(tmp_path, monkeypatch)

        with (
            patch("youcut.cli.run_flow_a", return_value=session),
            patch("youcut.cli._run_social_from_clips", return_value=[]) as mock_social,
            patch("youcut.cli.run_flow_b") as mock_flow_b,
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=xyz", config)

        mock_social.assert_called_once()
        mock_flow_b.assert_not_called()

    def test_social_receives_approved_clips_only(self, tmp_path, monkeypatch):
        """_run_social_from_clips recebe apenas os clipes aprovados."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(cut_mode="youtube", output_dir=tmp_path / "output")

        approved = _make_clip_record("approved", tmp_path, approved=True)
        rejected = _make_clip_record("rejected", tmp_path, approved=False)
        rejected.approved = False

        from datetime import datetime
        session = SessionData(
            session_id="test",
            source_url="https://youtube.com/watch?v=xyz",
            cut_mode="youtube",
            transcription_cache_path=tmp_path / "cache.json",
            clips=[approved, rejected],
            created_at=datetime.now(),
            output_dir=tmp_path / "output",
        )

        with (
            patch("youcut.cli.run_flow_a", return_value=session),
            patch("youcut.cli._run_social_from_clips", return_value=[]) as mock_social,
            patch("youcut.cli._show_consolidated_summary"),
        ):
            from youcut.cli import _run_auto_pipeline
            _run_auto_pipeline("https://youtube.com/watch?v=xyz", config)

        call_args = mock_social.call_args
        passed_clips = call_args.args[0]
        assert len(passed_clips) == 1
        assert passed_clips[0].title == "approved"
