import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests._fakes.comic_providers import FakeI2VProvider, FakeImageProvider
from youcut.comic.cli import _parse_panel_indices, comic_app
from youcut.comic.cost_estimator import CostBreakdown
from youcut.comic.pipeline import ComicPipelineError, PipelineCallbacks, run_comic_pipeline
from youcut.models import (
    CastMember,
    Panel,
    PanelRenderResult,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe não disponível",
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setenv("RUNWAY_API_KEY", "runway-test")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("youcut.session_store._SESSIONS_DIR", sessions_dir)
    return {"sessions_dir": sessions_dir, "tmp": tmp_path}


@pytest.fixture
def short_video(tmp_path):
    out = tmp_path / "input.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:size=320x240:duration=4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-t",
        "4",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _build_transcription(duration: float = 4.0) -> TranscriptionResult:
    words = [
        WordTimestamp(word=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(8)
    ]
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=duration, text="texto", words=words)
        ],
        language="pt",
        source_path=Path("v.mp4"),
    )


def _build_cast() -> list[CastMember]:
    return [
        CastMember(
            character_id="narrator",
            kind="person",
            narrative_role="narrator",
            text_card="ficha",
        )
    ]


def _build_panels() -> list[Panel]:
    return [
        Panel(
            index=0,
            start_time=0.0,
            end_time=2.0,
            participants=["narrator"],
            framing="close",
            scene="cena 1",
            pose_description="pose",
            panel_seconds_target=2.0,
        ),
        Panel(
            index=1,
            start_time=2.0,
            end_time=4.0,
            participants=["narrator"],
            framing="close",
            scene="cena 2",
            pose_description="pose",
            panel_seconds_target=2.0,
        ),
    ]


def _patch_pipeline_internals(monkeypatch, *, cast=None, panels=None, transcription=None):
    cast = cast or _build_cast()
    panels = panels or _build_panels()
    transcription = transcription or _build_transcription()
    monkeypatch.setattr("youcut.comic.pipeline.transcribe", lambda v, c: transcription)
    monkeypatch.setattr(
        "youcut.comic.pipeline.diarize",
        lambda v, c: [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=4.0)],
    )
    monkeypatch.setattr(
        "youcut.comic.pipeline.detect_cast", lambda *a, **kw: cast
    )
    monkeypatch.setattr(
        "youcut.comic.pipeline.plan_panels", lambda *a, **kw: panels
    )
    return cast, panels, transcription


# ---------------------------------------------------------------------------
# Help / parsing
# ---------------------------------------------------------------------------


def test_help_lists_main_flags(runner):
    result = runner.invoke(comic_app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--dry-run" in out
    assert "--session" in out
    assert "--regenerate-panel" in out
    assert "--max-panels" in out
    assert "--cost-cap" in out


def test_parse_panel_indices_supports_csv():
    assert _parse_panel_indices("1,3, 5") == [1, 3, 5]
    assert _parse_panel_indices(None) == []
    assert _parse_panel_indices("") == []


def test_parse_panel_indices_rejects_non_int():
    import typer
    with pytest.raises(typer.BadParameter):
        _parse_panel_indices("1,abc")


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_json_and_skips_paid_apis(runner, env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)

    fake_image = FakeImageProvider()
    fake_i2v = MagicMock()

    def _patched_pipeline(video_path, config, **kwargs):
        kwargs["image_provider"] = fake_image
        kwargs["i2v_provider"] = fake_i2v
        return run_comic_pipeline(video_path, config, **kwargs)

    monkeypatch.setattr("youcut.comic.cli.run_comic_pipeline", _patched_pipeline)

    result = runner.invoke(comic_app, [str(short_video), "--dry-run", "-y"])
    assert result.exit_code == 0, result.stdout

    out_dir = env["tmp"] / "out" / short_video.stem
    dry_path = out_dir / "comic" / "dry_run.json"
    assert dry_path.exists()

    payload = json.loads(dry_path.read_text(encoding="utf-8"))
    assert payload["video"]["path"].endswith("input.mp4")
    assert len(payload["cast"]) == 1
    assert len(payload["panels"]) == 2
    assert payload["estimate"]["n_panels"] == 2

    assert len(fake_image.calls) == 0
    fake_i2v.image_to_video.assert_not_called()


# ---------------------------------------------------------------------------
# CLI errors
# ---------------------------------------------------------------------------


def test_missing_video_argument_returns_error(runner, env):
    result = runner.invoke(comic_app, [])
    assert result.exit_code == 2


def test_video_too_long_returns_error(runner, env, tmp_path):
    long_video = tmp_path / "long.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:size=320x240:duration=91",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        "91",
        str(long_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    result = runner.invoke(comic_app, [str(long_video), "--dry-run", "-y"])
    assert result.exit_code != 0
    combined = (result.stderr or "") + (result.stdout or "") + str(result.exception or "")
    assert "muito longo" in combined or "máximo" in combined


def test_cost_cap_exceeded_aborts(runner, env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)

    result = runner.invoke(
        comic_app, [str(short_video), "--cost-cap", "0.001", "-y"]
    )
    combined = (result.stderr or "") + (result.stdout or "") + str(result.exception or "")
    assert result.exit_code == 1
    assert "teto" in combined.lower() or "excede" in combined.lower()


# ---------------------------------------------------------------------------
# Pipeline programmatic
# ---------------------------------------------------------------------------


def test_run_comic_pipeline_dry_run_does_not_call_providers(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)
    fake_image = FakeImageProvider()
    fake_i2v = MagicMock()

    from youcut.config import PipelineConfig
    config = PipelineConfig(
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=10.0,
    )

    session = run_comic_pipeline(
        short_video,
        config,
        dry_run=True,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )

    assert session.output_path is not None
    assert session.output_path.suffix == ".json"
    assert len(fake_image.calls) == 0
    fake_i2v.image_to_video.assert_not_called()
    assert len(session.cast) == 1
    assert len(session.panels) == 2


def test_run_comic_pipeline_full_run_with_fakes(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)
    fake_image = FakeImageProvider()
    fake_i2v = FakeI2VProvider()

    from youcut.config import PipelineConfig
    config = PipelineConfig(
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
        comic_i2v_concurrency=2,
    )

    session = run_comic_pipeline(
        short_video,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )

    assert session.output_path is not None
    assert session.output_path.exists()
    assert session.output_path.name == "motion_comic.mp4"
    assert len(session.panel_results) == 2
    assert len(fake_image.calls) >= 2
    assert len(fake_i2v.calls) == 2


def test_run_comic_pipeline_resume_skips_cast_and_script(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)
    fake_image = FakeImageProvider()
    fake_i2v = FakeI2VProvider()

    from youcut.config import PipelineConfig
    config = PipelineConfig(
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    first = run_comic_pipeline(
        short_video,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )

    detect_called = {"n": 0}
    plan_called = {"n": 0}

    real_detect = lambda *a, **kw: _build_cast()
    real_plan = lambda *a, **kw: _build_panels()

    def _spy_detect(*a, **kw):
        detect_called["n"] += 1
        return real_detect(*a, **kw)

    def _spy_plan(*a, **kw):
        plan_called["n"] += 1
        return real_plan(*a, **kw)

    monkeypatch.setattr("youcut.comic.pipeline.detect_cast", _spy_detect)
    monkeypatch.setattr("youcut.comic.pipeline.plan_panels", _spy_plan)

    second = run_comic_pipeline(
        short_video,
        config,
        session_id=first.session_id,
        image_provider=FakeImageProvider(),
        i2v_provider=FakeI2VProvider(),
    )

    assert second.session_id == first.session_id
    assert detect_called["n"] == 0
    assert plan_called["n"] == 0


def test_run_comic_pipeline_regenerate_panel_only(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)
    fake_image = FakeImageProvider()
    fake_i2v = FakeI2VProvider()

    from youcut.config import PipelineConfig
    config = PipelineConfig(
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    first = run_comic_pipeline(
        short_video,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )
    assert len(first.panel_results) == 2

    image2 = FakeImageProvider()
    i2v2 = FakeI2VProvider()

    second = run_comic_pipeline(
        short_video,
        config,
        session_id=first.session_id,
        regenerate_panels=[1],
        image_provider=image2,
        i2v_provider=i2v2,
    )

    assert len(second.panel_results) == 2
    assert len(i2v2.calls) == 1
    assert i2v2.calls[0]["prompt_image"].name == "panel_01.png"


def test_run_comic_pipeline_user_rejects_cast_aborts(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(output_dir=env["tmp"] / "out")

    callbacks = PipelineCallbacks(confirm_cast=lambda _c: False)

    with pytest.raises(ComicPipelineError, match=r"cast"):
        run_comic_pipeline(
            short_video,
            config,
            callbacks=callbacks,
            image_provider=FakeImageProvider(),
            i2v_provider=FakeI2VProvider(),
        )


def test_run_comic_pipeline_user_rejects_cost_aborts(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(output_dir=env["tmp"] / "out", comic_cost_cap_usd=100.0)

    callbacks = PipelineCallbacks(confirm_cost=lambda _b: False)

    with pytest.raises(ComicPipelineError, match=r"custo"):
        run_comic_pipeline(
            short_video,
            config,
            callbacks=callbacks,
            image_provider=FakeImageProvider(),
            i2v_provider=FakeI2VProvider(),
        )


def test_run_comic_pipeline_unknown_session_raises(env, short_video, monkeypatch):
    _patch_pipeline_internals(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(output_dir=env["tmp"] / "out")

    with pytest.raises(ComicPipelineError):
        run_comic_pipeline(
            short_video,
            config,
            session_id="does-not-exist",
            image_provider=FakeImageProvider(),
            i2v_provider=FakeI2VProvider(),
        )
