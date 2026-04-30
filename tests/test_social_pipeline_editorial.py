from pathlib import Path
from unittest.mock import MagicMock, patch

from youcut.models import CaptionBurnResult, ClipRecord, SessionData, TranscriptionResult, TranscriptionSegment, ViralClip, WordTimestamp


def _make_clip() -> ViralClip:
    return ViralClip(
        title="Clip Social",
        reason="Gancho forte",
        viral_score=8.0,
        start_time=0.0,
        end_time=30.0,
        description="Descrição",
        hashtags=["#teste"],
        thumbnail_idea="Frame impactante",
        thumbnail_text="MOMENTO IMPACTANTE",
        social_hook_title="ALERTA TOTAL",
        social_image_prompt="Cena editorial sem texto",
        social_visual_style="claro e vivo",
        cut_mode="social",
    )


def _make_transcription(source_path: Path) -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=30.0,
                text="Conteúdo de teste",
                words=[WordTimestamp(word="Conteúdo", start=0.0, end=0.5)],
            )
        ],
        language="pt",
        source_path=source_path,
    )


def test_build_social_pipeline_config_defaults_to_editorial(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.cli import _build_social_pipeline_config
    from youcut.config import PipelineConfig

    base = PipelineConfig(cut_mode="youtube")
    social = _build_social_pipeline_config(base)
    assert social.cut_mode == "social"
    assert social.social_layout_mode == "speaker_bottom_ai_top"


def test_run_flow_c_composes_and_burns_bottom_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.cli import run_flow_c
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip.mp4"
    raw_clip.write_bytes(b"raw")
    composed_clip = tmp_path / "clip_social.mp4"
    composed_clip.write_bytes(b"composed")
    final_clip = tmp_path / "clip_social_captioned.mp4"
    final_clip.write_bytes(b"captioned")

    config = PipelineConfig(cut_mode="social", social_layout_mode="speaker_bottom_ai_top")
    clip = _make_clip()

    with (
        patch("youcut.cli.download_video", return_value=raw_clip),
        patch("youcut.cli.transcribe", return_value=_make_transcription(raw_clip)),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.frame_for_panel", side_effect=lambda p, **_: p) as mock_frame,
        patch("youcut.cli.compose_social_clip", return_value=composed_clip) as mock_compose,
        patch("youcut.cli.CaptionBurner.burn", return_value=CaptionBurnResult(output_path=final_clip, captions_applied=True)) as mock_burn,
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_c("https://youtube.com/watch?v=test", config, skip_review=True, upload=False)

    mock_frame.assert_called_once()
    assert mock_frame.call_args.args[0] == raw_clip
    mock_compose.assert_called_once_with(raw_clip, clip, config)
    assert mock_burn.call_args.kwargs["layout_mode"] == "bottom_panel"


def test_run_flow_b_composes_and_burns_bottom_panel(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.cli import run_flow_b
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip.mp4"
    raw_clip.write_bytes(b"raw")
    composed_clip = tmp_path / "clip_social.mp4"
    composed_clip.write_bytes(b"composed")
    final_clip = tmp_path / "clip_social_captioned.mp4"
    final_clip.write_bytes(b"captioned")
    transcription_path = tmp_path / "transcription.json"
    transcription_path.write_text(
        '{"result":{"segments":[{"start":0.0,"end":30.0,"text":"Teste","words":[]}],"language":"pt","source_path":"clip.mp4"}}',
        encoding="utf-8",
    )

    session = SessionData(
        session_id="sess-1",
        source_url="https://youtube.com/watch?v=test",
        cut_mode="youtube",
        transcription_cache_path=transcription_path,
        clips=[ClipRecord(title="Long clip", start_time=0.0, end_time=300.0, clip_path=raw_clip, thumbnail_path=None)],
        created_at=__import__("datetime").datetime.now(),
        output_dir=tmp_path / "output",
    )
    config = PipelineConfig(cut_mode="social", social_layout_mode="speaker_bottom_ai_top")
    clip = _make_clip()

    with (
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.frame_for_panel", side_effect=lambda p, **_: p),
        patch("youcut.cli.compose_social_clip", return_value=composed_clip) as mock_compose,
        patch("youcut.cli.CaptionBurner.burn", return_value=CaptionBurnResult(output_path=final_clip, captions_applied=True)) as mock_burn,
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_b(session, session.clips, config, skip_review=True, upload=False)

    mock_compose.assert_called_once()
    assert mock_burn.call_args.kwargs["layout_mode"] == "bottom_panel"


def test_run_single_source_pipeline_uses_editorial_finalizer(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.cli import _run_single_source_pipeline
    from youcut.config import PipelineConfig

    source_clip = tmp_path / "source.mp4"
    source_clip.write_bytes(b"source")
    raw_clip = tmp_path / "clip_01.mp4"
    raw_clip.write_bytes(b"raw")
    final_clip = tmp_path / "clip_01_social_captioned.mp4"
    final_clip.write_bytes(b"final")
    clip = _make_clip()
    transcription = _make_transcription(source_clip)
    config = PipelineConfig(cut_mode="social", social_layout_mode="speaker_bottom_ai_top", clip_count=1)

    with (
        patch("youcut.cli.download_video", return_value=source_clip),
        patch("youcut.cli.transcribe", return_value=transcription),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli._finalize_editorial_social_clip", return_value=(final_clip, True, None)) as mock_finalize,
        patch("youcut.cli.add_captions") as mock_add_captions,
        patch("youcut.cli.generate_clip_preview", return_value=MagicMock(path=tmp_path / "preview.jpg")),
    ):
        records = _run_single_source_pipeline(str(source_clip), config, skip_review=True, upload=False)

    mock_finalize.assert_called_once_with(raw_clip, clip, config)
    mock_add_captions.assert_not_called()
    assert records[0].clip_path == final_clip
