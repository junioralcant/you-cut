"""Testes do mapping CLI `--no-visual-style` → `PipelineConfig.social_visual_style_enabled`.

Cobertura:
- Sem flag → social_visual_style_enabled=True (default, RF-20).
- Com flag → social_visual_style_enabled=False (RF-19).
- Modo `youtube` ainda passa a flag, mas o gate por modo é feito no orquestrador
  (não nesta camada — a config carrega o booleano fielmente).
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from youcut.cli import app

runner = CliRunner()
API_ENV = {"ANTHROPIC_API_KEY": "test-key"}


def _social_run(extra_args: list[str]):
    """Roda `youcut cuts video.mp4 --mode social ...` mockando todo o pipeline.

    Retorna o `PipelineConfig` que foi passado para `run_flow_c`.
    """
    fake_session = MagicMock()
    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("youcut.cli._resolve_cli_yt_dlp_auth_config", return_value=None),
        patch("youcut.cli.normalize_video_url", return_value="video.mp4"),
        patch(
            "youcut.cli.fetch_metadata",
            return_value=MagicMock(title="Video", duration_seconds=120.0),
        ),
        patch(
            "youcut.cli.questionary.confirm",
            return_value=MagicMock(ask=MagicMock(return_value=True)),
        ),
        patch("youcut.cli._can_prompt_interactively", return_value=False),
        patch("youcut.cli.run_flow_c", return_value=fake_session) as mock_flow_c,
    ):
        result = runner.invoke(
            app,
            [
                "cuts",
                "video.mp4",
                "--mode",
                "social",
                "--max-clips",
                "1",
                *extra_args,
            ],
            env=API_ENV,
        )
    return result, mock_flow_c


def test_cuts_without_no_visual_style_keeps_default_true():
    result, mock_flow_c = _social_run([])

    assert result.exit_code == 0
    config = mock_flow_c.call_args.args[1]
    assert config.social_visual_style_enabled is True


def test_cuts_with_no_visual_style_disables_treatment():
    result, mock_flow_c = _social_run(["--no-visual-style"])

    assert result.exit_code == 0
    config = mock_flow_c.call_args.args[1]
    assert config.social_visual_style_enabled is False


def test_cuts_youtube_mode_propagates_default_true():
    """Modo youtube também recebe o booleano (gate por modo é em outra camada)."""
    fake_session = MagicMock()
    with (
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("youcut.cli._resolve_cli_yt_dlp_auth_config", return_value=None),
        patch("youcut.cli.normalize_video_url", return_value="video.mp4"),
        patch(
            "youcut.cli.fetch_metadata",
            return_value=MagicMock(title="Video", duration_seconds=120.0),
        ),
        patch(
            "youcut.cli.questionary.confirm",
            return_value=MagicMock(ask=MagicMock(return_value=True)),
        ),
        patch("youcut.cli._can_prompt_interactively", return_value=False),
        patch("youcut.cli.run_flow_a", return_value=fake_session) as mock_flow_a,
        patch("youcut.cli.offer_flow_b"),
    ):
        result = runner.invoke(
            app,
            ["cuts", "video.mp4", "--mode", "youtube", "--max-clips", "1"],
            env=API_ENV,
        )

    assert result.exit_code == 0
    config = mock_flow_a.call_args.args[1]
    assert config.social_visual_style_enabled is True


def test_build_social_pipeline_config_propagates_flag():
    """O helper de Fluxo B preserva o gate vindo do base_config."""
    from youcut.cli import _build_social_pipeline_config
    from youcut.config import PipelineConfig

    base = PipelineConfig(social_visual_style_enabled=False)
    derived = _build_social_pipeline_config(base)
    assert derived.social_visual_style_enabled is False

    base_on = PipelineConfig(social_visual_style_enabled=True)
    derived_on = _build_social_pipeline_config(base_on)
    assert derived_on.social_visual_style_enabled is True
