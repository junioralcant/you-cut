"""`--regenerate-panel` reusa cast e painéis intactos, regenerando só o alvo."""

import shutil
import subprocess
from pathlib import Path

import pytest

from tests._fakes.comic_providers import FakeI2VProvider, FakeImageProvider
from youcut.comic.pipeline import run_comic_pipeline
from youcut.models import (
    CastMember,
    Panel,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="FFmpeg/ffprobe não disponível",
    ),
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.setenv("RUNWAY_API_KEY", "runway-test")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("youcut.session_store._SESSIONS_DIR", sessions_dir)
    return {"sessions_dir": sessions_dir, "tmp": tmp_path}


def _build_video(tmp_path: Path) -> Path:
    out = tmp_path / "input.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:size=320x240:duration=5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-t",
        "5",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def _patch_pipeline(monkeypatch):
    cast = [
        CastMember(
            character_id="narrator",
            kind="person",
            narrative_role="narrator",
            text_card="ficha",
        )
    ]
    panels = [
        Panel(
            index=i,
            start_time=i * 1.0,
            end_time=(i + 1) * 1.0,
            participants=["narrator"],
            framing="close",
            scene=f"cena {i+1}",
            pose_description="pose",
            panel_seconds_target=2.0,  # clamp inferior
        )
        for i in range(5)
    ]
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="texto",
                words=[WordTimestamp(word=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(10)],
            )
        ],
        language="pt",
        source_path=Path("v.mp4"),
    )
    monkeypatch.setattr("youcut.comic.pipeline.transcribe", lambda v, c: transcription)
    monkeypatch.setattr(
        "youcut.comic.pipeline.diarize",
        lambda v, c: [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=5.0)],
    )
    monkeypatch.setattr("youcut.comic.pipeline.detect_cast", lambda *a, **kw: cast)
    monkeypatch.setattr("youcut.comic.pipeline.plan_panels", lambda *a, **kw: panels)
    return cast, panels


def test_regenerate_panel_only_replaces_targeted_index(env, tmp_path, monkeypatch):
    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", 
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    first = run_comic_pipeline(
        video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )
    assert len(first.panel_results) == 5

    image2 = FakeImageProvider(size=(1024, 1792))
    i2v2 = FakeI2VProvider()
    second = run_comic_pipeline(
        video,
        config,
        session_id=first.session_id,
        regenerate_panels=[2],
        image_provider=image2,
        i2v_provider=i2v2,
    )

    # Apenas o painel 2 foi regerado.
    assert len(i2v2.calls) == 1
    assert i2v2.calls[0]["prompt_image"].name == "panel_02.png"
    # Saída final ainda existe e tem 5 painéis registrados.
    assert second.output_path is not None
    assert second.output_path.exists()
    assert len(second.panel_results) == 5
    assert {r.panel_index for r in second.panel_results} == {0, 1, 2, 3, 4}


def test_regenerate_multiple_panels_at_once(env, tmp_path, monkeypatch):
    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", output_dir=env["tmp"] / "out", comic_cost_cap_usd=100.0)

    first = run_comic_pipeline(
        video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    image2 = FakeImageProvider(size=(1024, 1792))
    i2v2 = FakeI2VProvider()
    second = run_comic_pipeline(
        video,
        config,
        session_id=first.session_id,
        regenerate_panels=[1, 3],
        image_provider=image2,
        i2v_provider=i2v2,
    )

    assert len(i2v2.calls) == 2
    regenerated = {c["prompt_image"].name for c in i2v2.calls}
    assert regenerated == {"panel_01.png", "panel_03.png"}
    assert len(second.panel_results) == 5


def test_regenerate_skips_cast_build_when_session_has_cast(env, tmp_path, monkeypatch):
    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", output_dir=env["tmp"] / "out", comic_cost_cap_usd=100.0)

    first = run_comic_pipeline(
        video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    image2 = FakeImageProvider(size=(1024, 1792))
    i2v2 = FakeI2VProvider()
    run_comic_pipeline(
        video,
        config,
        session_id=first.session_id,
        regenerate_panels=[0],
        image_provider=image2,
        i2v_provider=i2v2,
    )

    # Cast já materializado: nenhuma chamada extra ao image provider para anchors.
    # Painel 0 ainda gera 1 imagem-base, portanto >=1 chamada.
    assert len(image2.calls) == 1
