"""Testes unitários do orquestrador `run_remotion_pipeline` (Task 9.0)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tests._fakes.comic_providers import FakeImageProvider, FakeRemotionRenderer
from youcut.comic.pipeline import ComicPipelineError, PipelineCallbacks
from youcut.comic.remotion_pipeline import (
    _is_interactive_environment,
    _resolve_speaker_for_panel,
    _build_remotion_scenes,
    _shift_word,
    run_remotion_pipeline,
)
from youcut.comic.cost_estimator import CostCapExceededError
from youcut.models import (
    CastMember,
    Panel,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def remotion_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    return PipelineConfig(
        comic_animation_engine="remotion",
        output_dir=tmp_path,
        openai_api_key="test-openai",
    )


@pytest.fixture
def synthetic_video(tmp_path):
    """Vídeo sintético 6s 1080x1920 azul + sine 440Hz."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg ausente")
    path = tmp_path / "input.mp4"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1080x1920:d=6:r=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        pytest.skip(f"falha ao gerar fixture: {completed.stderr.decode()}")
    return path


@pytest.fixture
def transcription():
    return TranscriptionResult(
        language="pt",
        source_path=Path("/tmp/x.mp4"),
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=6.0,
                text="olá mundo bem vindo",
                words=[
                    WordTimestamp(word="olá", start=0.0, end=0.5),
                    WordTimestamp(word="mundo", start=0.5, end=1.5),
                    WordTimestamp(word="bem", start=2.0, end=2.4),
                    WordTimestamp(word="vindo", start=2.5, end=3.0),
                ],
            )
        ],
    )


@pytest.fixture
def speakers():
    return [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=6.0)]


@pytest.fixture
def cast():
    return [
        CastMember(
            character_id="speaker_a",
            kind="person",
            speaker_id="SPEAKER_00",
            text_card="ficha visual",
        )
    ]


@pytest.fixture
def panels():
    return [
        Panel(
            index=0,
            start_time=0.0,
            end_time=3.0,
            participants=["speaker_a"],
            framing="close",
            scene="cena 0",
            pose_description="pose neutra",
            panel_seconds_target=3.0,
        ),
        Panel(
            index=1,
            start_time=3.0,
            end_time=6.0,
            participants=["speaker_a"],
            framing="medium",
            scene="cena 1",
            pose_description="pose acolhedora",
            panel_seconds_target=3.0,
        ),
    ]


# ── _is_interactive_environment ─────────────────────────────────────────────


def test_is_interactive_environment_returns_false_when_stdin_not_tty(monkeypatch):
    class FakeStream:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stdin", FakeStream())
    monkeypatch.setattr("sys.stdout", FakeStream())
    assert _is_interactive_environment() is False


def test_is_interactive_environment_linux_requires_display(monkeypatch):
    class FakeStream:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", FakeStream())
    monkeypatch.setattr("sys.stdout", FakeStream())
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert _is_interactive_environment() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert _is_interactive_environment() is True


# ── _resolve_speaker_for_panel ──────────────────────────────────────────────


def test_resolve_speaker_for_panel_uses_diarization_when_match(panels, cast, speakers):
    resolved = _resolve_speaker_for_panel(panels[0], cast, speakers)
    assert resolved == "speaker_a"


def test_resolve_speaker_for_panel_falls_back_to_first_participant(panels, cast):
    resolved = _resolve_speaker_for_panel(panels[0], cast, speakers=[])
    assert resolved == "speaker_a"


def test_resolve_speaker_for_panel_handles_empty_participants():
    panel = Panel(
        index=0,
        start_time=0.0,
        end_time=1.0,
        participants=["speaker_x"],  # Pydantic exige non-empty
        framing="close",
        scene="x",
        pose_description="x",
        panel_seconds_target=1.0,
    )
    # Quando participants tem 1 char mas cast vazio, ainda devolve o participant
    assert _resolve_speaker_for_panel(panel, cast=[], speakers=[]) == "speaker_x"


# ── _shift_word ─────────────────────────────────────────────────────────────


def test_shift_word_subtracts_offset():
    w = WordTimestamp(word="oi", start=2.5, end=3.0)
    shifted = _shift_word(w, 2.5)
    assert shifted.start == 0.0
    assert shifted.end == 0.5


def test_shift_word_clamps_negative_to_zero():
    w = WordTimestamp(word="oi", start=0.5, end=1.0)
    shifted = _shift_word(w, 2.0)
    assert shifted.start == 0.0
    assert shifted.end == 0.0


# ── _build_remotion_scenes ──────────────────────────────────────────────────


def test_build_remotion_scenes_produces_one_scene_per_panel(
    panels, cast, speakers, transcription, remotion_config
):
    scenes = _build_remotion_scenes(panels, cast, speakers, transcription, config=remotion_config)
    assert len(scenes) == 2
    assert scenes[0].start_sec == 0.0
    assert scenes[0].end_sec == 3.0
    assert scenes[1].start_sec == 3.0
    assert scenes[1].end_sec == 6.0


def test_build_remotion_scenes_first_uses_cut_others_crossfade(
    panels, cast, speakers, transcription, remotion_config
):
    scenes = _build_remotion_scenes(panels, cast, speakers, transcription, config=remotion_config)
    assert scenes[0].transition_in == "cut"
    assert scenes[1].transition_in == "crossfade"


def test_build_remotion_scenes_lipsync_is_scene_relative(
    panels, cast, speakers, transcription, remotion_config
):
    scenes = _build_remotion_scenes(panels, cast, speakers, transcription, config=remotion_config)
    # Words em panels[0] (0-3s): olá (0-0.5), mundo (0.5-1.5), bem (2.0-2.4), vindo (2.5-3.0)
    # Todos relativos à própria scene (que começa em 0.0 → não há offset).
    first_scene_lipsync = scenes[0].lip_sync
    assert first_scene_lipsync, "scene 0 deve ter lipsync derivado"
    for ev in first_scene_lipsync:
        assert ev.start_sec >= 0.0
        assert ev.end_sec <= 3.05  # cena dura 3s
        assert ev.character_id == "speaker_a"
    # Scene 1 (3-6s) não tem palavras na transcrição → lipsync vazio.
    assert scenes[1].lip_sync == []


def test_build_remotion_scenes_kenburns_uses_config_scale(
    panels, cast, speakers, transcription, remotion_config
):
    scenes = _build_remotion_scenes(panels, cast, speakers, transcription, config=remotion_config)
    for scene in scenes:
        assert scene.ken_burns["scale_to"] == remotion_config.comic_remotion_kenburns_default_scale
        assert scene.ken_burns["scale_from"] == 1.0


# ── run_remotion_pipeline ──────────────────────────────────────────────────


def _patch_pipeline_dependencies(
    monkeypatch,
    *,
    transcription,
    speakers,
    cast,
    panels,
):
    """Patches transcribe/diarize/visual_analyzer/script_planner para fakes."""
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline.transcribe", lambda *a, **k: transcription
    )
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline.diarize", lambda *a, **k: speakers
    )
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline.detect_cast", lambda *a, **k: list(cast)
    )
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline.invent_cast", lambda *a, **k: list(cast)
    )
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline.plan_panels", lambda *a, **k: list(panels)
    )


def test_run_remotion_pipeline_dry_run_stops_after_cost(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels
):
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )
    fake_renderer = FakeRemotionRenderer()

    session = run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=fake_renderer,
        preview=False,
        dry_run=True,
    )
    assert session.output_path is None
    assert session.total_cost_usd == 0.0
    assert fake_renderer.render_calls == []
    assert fake_renderer.studio_calls == []


def test_run_remotion_pipeline_headless_when_no_tty(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels
):
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )
    fake_renderer = FakeRemotionRenderer()

    # Force non-interactive
    monkeypatch.setattr(
        "youcut.comic.remotion_pipeline._is_interactive_environment",
        lambda: False,
    )

    session = run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=fake_renderer,
        preview=True,  # request preview, but environment forces headless
    )
    # Não chamou open_studio, mesmo com preview=True
    assert fake_renderer.studio_calls == []
    # Mas chamou render
    assert len(fake_renderer.render_calls) == 1
    assert session.output_path is not None
    assert session.output_path.exists()


def test_run_remotion_pipeline_preflight_propagates_cap_exception(
    monkeypatch, synthetic_video, transcription, speakers, cast, panels
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    capped_config = PipelineConfig(
        comic_animation_engine="remotion",
        comic_cost_cap_usd=0.001,  # absurdamente baixo → estoura
        output_dir=Path(synthetic_video).parent,
        openai_api_key="test",
    )

    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )

    with pytest.raises(ComicPipelineError):
        run_remotion_pipeline(
            synthetic_video,
            capped_config,
            image_provider=FakeImageProvider(),
            renderer=FakeRemotionRenderer(),
            preview=False,
        )


def test_run_remotion_pipeline_persists_session(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels, tmp_path
):
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )
    sessions_dir = tmp_path / "fake_home_sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(
        "youcut.session_store._SESSIONS_DIR", sessions_dir
    )

    fake_renderer = FakeRemotionRenderer()
    session = run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=fake_renderer,
        preview=False,
    )
    assert session.session_id
    saved = list(sessions_dir.glob("*.json"))
    assert saved, f"esperava sessão persistida em {sessions_dir}"


def test_run_remotion_pipeline_invokes_callbacks_on_stage(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels
):
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )
    stages: list[str] = []

    cb = PipelineCallbacks(
        on_stage=lambda name, payload: stages.append(name),
    )
    run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=FakeRemotionRenderer(),
        preview=False,
        callbacks=cb,
    )
    expected_subset = {"validate", "transcribe", "diarize", "cost_estimate", "render", "compose", "done"}
    assert expected_subset.issubset(set(stages)), (
        f"esperava {expected_subset} ⊆ {set(stages)}"
    )


def test_run_remotion_pipeline_emits_two_versions(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels, tmp_path
):
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )

    fake_renderer = FakeRemotionRenderer()
    session = run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=fake_renderer,
        preview=False,
    )
    output = session.output_path
    assert output is not None
    no_subs = output.parent / "motion_comic_no_subs.mp4"
    assert no_subs.exists(), f"esperava {no_subs}"
    assert output.exists(), f"esperava {output}"


def test_run_remotion_pipeline_total_cost_under_one_dollar(
    monkeypatch, synthetic_video, remotion_config, transcription, speakers, cast, panels
):
    """RF-23: 1 personagem → custo ≤ $1."""
    _patch_pipeline_dependencies(
        monkeypatch,
        transcription=transcription,
        speakers=speakers,
        cast=cast,
        panels=panels,
    )

    session = run_remotion_pipeline(
        synthetic_video,
        remotion_config,
        image_provider=FakeImageProvider(),
        renderer=FakeRemotionRenderer(),
        preview=False,
    )
    assert session.total_cost_usd <= 1.0
