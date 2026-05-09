"""Testes de integração orquestrada do `apply_visual_style` no pipeline social.

Cobertura (Task 4.0):
- ordem das chamadas no caminho social/editorial: composer → apply_visual_style → CaptionBurner;
- ordem no caminho social/classic em `run_flow_c`: cut → apply_visual_style → CaptionBurner → music_mixer;
- isolamento por modo (RF-21): pipeline `youtube` não invoca `apply_visual_style` (gate estático
  no orquestrador para o caminho editorial; gate interno do módulo para o caminho legacy);
- gate por flag `--no-visual-style`: a chamada acontece, mas o módulo é no-op (já testado
  em `tests/test_visual_style.py`); aqui validamos que o orquestrador delega ao módulo.

Mocks limitados a chamadas externas (download/transcribe/analyze/ffmpeg-via-CaptionBurner) +
ao próprio `apply_visual_style` quando precisamos auditar a sequência de chamadas.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.models import (
    CaptionBurnResult,
    TranscriptionResult,
    TranscriptionSegment,
    ViralClip,
    WordTimestamp,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_clip(cut_mode: str = "social") -> ViralClip:
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
        cut_mode=cut_mode,
    )


def _make_transcription(source_path: Path) -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=30.0,
                text="Conteúdo",
                words=[WordTimestamp(word="Conteúdo", start=0.0, end=0.5)],
            )
        ],
        language="pt",
        source_path=source_path,
    )


# ── Fluxo C (modo social, layout editorial) ──────────────────────────────────

def test_run_flow_c_editorial_invokes_apply_visual_style_between_composer_and_caption_burner(
    tmp_path,
):
    """RF-13/14 + RF-16: tratamento visual entra após o composer e antes da legenda."""
    from youcut.cli import run_flow_c
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip_01.mp4"
    raw_clip.write_bytes(b"raw")
    composed_clip = tmp_path / "clip_01_social.mp4"
    composed_clip.write_bytes(b"composed")
    styled_clip = tmp_path / "clip_01_styled.mp4"
    styled_clip.write_bytes(b"styled")
    captioned_clip = tmp_path / "clip_01_styled_captioned.mp4"
    captioned_clip.write_bytes(b"captioned")

    config = PipelineConfig(
        cut_mode="social",
        social_layout_mode="speaker_bottom_ai_top",
        social_visual_style_enabled=True,
    )
    clip = _make_clip()

    call_order: list[str] = []

    def _track_compose(*args, **kwargs):
        call_order.append("compose_social_clip")
        return composed_clip

    def _track_apply(*args, **kwargs):
        call_order.append("apply_visual_style")
        return styled_clip

    def _track_burn(self, path, **kwargs):
        call_order.append("CaptionBurner.burn")
        return CaptionBurnResult(output_path=captioned_clip, captions_applied=True)

    with (
        patch("youcut.cli.download_video", return_value=raw_clip),
        patch("youcut.cli.transcribe", return_value=_make_transcription(raw_clip)),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.frame_for_panel", side_effect=lambda p, **_: p),
        patch("youcut.cli.compose_social_clip", side_effect=_track_compose),
        patch("youcut.cli.apply_visual_style", side_effect=_track_apply),
        patch("youcut.cli.CaptionBurner.burn", autospec=True, side_effect=_track_burn),
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_c("https://youtube.com/watch?v=test", config, skip_review=True, upload=False)

    assert call_order == ["compose_social_clip", "apply_visual_style", "CaptionBurner.burn"], (
        f"ordem incorreta: {call_order}"
    )


# ── Fluxo C (modo social, layout classic) ────────────────────────────────────

def test_run_flow_c_classic_invokes_apply_visual_style_before_caption_burner(tmp_path):
    """Em layout classic, apply_visual_style também é invocado antes do CaptionBurner."""
    from youcut.cli import run_flow_c
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip_01.mp4"
    raw_clip.write_bytes(b"raw")
    styled_clip = tmp_path / "clip_01_styled.mp4"
    styled_clip.write_bytes(b"styled")
    captioned_clip = tmp_path / "clip_01_styled_captioned.mp4"
    captioned_clip.write_bytes(b"captioned")

    config = PipelineConfig(
        cut_mode="social",
        social_layout_mode="classic",
        social_visual_style_enabled=True,
    )
    clip = _make_clip()
    call_order: list[str] = []

    def _track_apply(*args, **kwargs):
        call_order.append("apply_visual_style")
        return styled_clip

    def _track_burn(self, path, **kwargs):
        call_order.append("CaptionBurner.burn")
        return CaptionBurnResult(output_path=captioned_clip, captions_applied=True)

    with (
        patch("youcut.cli.download_video", return_value=raw_clip),
        patch("youcut.cli.transcribe", return_value=_make_transcription(raw_clip)),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.apply_visual_style", side_effect=_track_apply),
        patch("youcut.cli.CaptionBurner.burn", autospec=True, side_effect=_track_burn),
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_c("https://youtube.com/watch?v=test", config, skip_review=True, upload=False)

    assert call_order == ["apply_visual_style", "CaptionBurner.burn"], (
        f"ordem incorreta: {call_order}"
    )


# ── Isolamento por modo (RF-21/22/23) ────────────────────────────────────────

def test_run_flow_a_youtube_does_not_invoke_apply_visual_style(tmp_path):
    """RF-21: pipeline youtube (run_flow_a) NÃO chama apply_visual_style."""
    from youcut.cli import run_flow_a
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip_01.mp4"
    raw_clip.write_bytes(b"raw")
    config = PipelineConfig(cut_mode="youtube")
    clip = _make_clip(cut_mode="youtube")

    with (
        patch("youcut.cli.download_video", return_value=raw_clip),
        patch("youcut.cli.transcribe", return_value=_make_transcription(raw_clip)),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.generate_thumbnail", return_value=None),
        patch("youcut.cli.apply_visual_style") as mock_apply,
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_a(
            "https://youtube.com/watch?v=test",
            config,
            skip_review=True,
            upload=False,
            platforms=[],
        )

    mock_apply.assert_not_called()


# ── Gate por flag --no-visual-style ─────────────────────────────────────────

def test_run_flow_c_classic_still_calls_apply_visual_style_when_flag_disabled(tmp_path):
    """Mesmo com social_visual_style_enabled=False, o orquestrador chama o módulo —
    o gate é interno (no-op silencioso). Garante que o ponto de extensão é único.
    """
    from youcut.cli import run_flow_c
    from youcut.config import PipelineConfig

    raw_clip = tmp_path / "clip_01.mp4"
    raw_clip.write_bytes(b"raw")
    captioned = tmp_path / "clip_01_captioned.mp4"
    captioned.write_bytes(b"x")

    config = PipelineConfig(
        cut_mode="social",
        social_layout_mode="classic",
        social_visual_style_enabled=False,
    )
    clip = _make_clip()

    with (
        patch("youcut.cli.download_video", return_value=raw_clip),
        patch("youcut.cli.transcribe", return_value=_make_transcription(raw_clip)),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=raw_clip),
        patch("youcut.cli.apply_visual_style", side_effect=lambda p, c: p) as mock_apply,
        patch(
            "youcut.cli.CaptionBurner.burn",
            autospec=True,
            return_value=CaptionBurnResult(output_path=captioned, captions_applied=True),
        ),
        patch("youcut.cli._show_records_table"),
    ):
        run_flow_c("https://youtube.com/watch?v=test", config, skip_review=True, upload=False)

    # Orquestrador chama o módulo — o gate fica internalizado lá.
    mock_apply.assert_called()


# ── Validação estrutural: pipeline `comic` e `run` legacy não importam ───────

def test_comic_module_does_not_import_apply_visual_style():
    """RF-22: pipeline comic não invoca apply_visual_style."""
    import importlib

    comic_mod = importlib.import_module("youcut.comic.pipeline")
    src = Path(comic_mod.__file__).read_text(encoding="utf-8")
    assert "apply_visual_style" not in src
