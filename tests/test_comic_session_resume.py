"""Session resume — interrupção sintética e retomada via `--session <id>`."""

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests._fakes.comic_providers import FakeI2VProvider, FakeImageProvider
from youcut.comic.pipeline import run_comic_pipeline
from youcut.comic.session import (
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.models import (
    CastMember,
    MotionComicSession,
    Panel,
    PanelRenderResult,
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


def _build_cast() -> list[CastMember]:
    return [
        CastMember(
            character_id="narrator",
            kind="person",
            narrative_role="narrator",
            text_card="ficha",
        )
    ]


def _build_panels(n: int = 5, total: float = 5.0) -> list[Panel]:
    each = total / n
    return [
        Panel(
            index=i,
            start_time=i * each,
            end_time=(i + 1) * each,
            participants=["narrator"],
            framing="close",
            scene=f"cena {i+1}",
            pose_description="pose",
            panel_seconds_target=each,
        )
        for i in range(n)
    ]


def _build_transcription(duration: float = 5.0) -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=duration,
                text="texto",
                words=[WordTimestamp(word=f"w{i}", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(10)],
            )
        ],
        language="pt",
        source_path=Path("v.mp4"),
    )


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


def _patch_pipeline(monkeypatch, *, cast=None, panels=None, transcription=None):
    cast = cast or _build_cast()
    panels = panels or _build_panels()
    transcription = transcription or _build_transcription()
    monkeypatch.setattr("youcut.comic.pipeline.transcribe", lambda v, c: transcription)
    monkeypatch.setattr(
        "youcut.comic.pipeline.diarize",
        lambda v, c: [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=5.0)],
    )
    monkeypatch.setattr("youcut.comic.pipeline.detect_cast", lambda *a, **kw: cast)
    monkeypatch.setattr("youcut.comic.pipeline.plan_panels", lambda *a, **kw: panels)


def test_session_resume_only_regenerates_missing_panels(env, tmp_path, monkeypatch):
    """Simula interrupção após 2 painéis e retoma só com os 3 últimos via --regenerate-panel."""

    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch, panels=_build_panels(n=5))

    from youcut.config import PipelineConfig
    config = PipelineConfig(
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    # Render completo da 1ª passada para criar arquivos físicos no disco.
    fake_image_run1 = FakeImageProvider(size=(1024, 1792))
    fake_i2v_run1 = FakeI2VProvider()
    session_run1 = run_comic_pipeline(
        video,
        config,
        image_provider=fake_image_run1,
        i2v_provider=fake_i2v_run1,
    )
    assert len(session_run1.panel_results) == 5

    # Sessão "interrompida" retém apenas os 2 primeiros painéis.
    interrupted = MotionComicSession(
        session_id=session_run1.session_id,
        video_path=video,
        created_at=datetime.now(timezone.utc),
        cast=session_run1.cast,
        panels=session_run1.panels,
        panel_results=session_run1.panel_results[:2],
        total_cost_usd=sum(r.cost_usd for r in session_run1.panel_results[:2]),
    )
    save_motion_comic_session(interrupted)

    # Retoma regenerando os índices 2, 3, 4.
    fake_image_run2 = FakeImageProvider(size=(1024, 1792))
    fake_i2v_run2 = FakeI2VProvider()
    session_run2 = run_comic_pipeline(
        video,
        config,
        session_id=session_run1.session_id,
        regenerate_panels=[2, 3, 4],
        image_provider=fake_image_run2,
        i2v_provider=fake_i2v_run2,
    )

    assert session_run2.session_id == session_run1.session_id
    assert len(session_run2.panel_results) == 5
    # Apenas os 3 painéis-alvo passaram pelo i2v na 2ª execução.
    assert len(fake_i2v_run2.calls) == 3


def test_unknown_session_raises_pipeline_error(env, tmp_path, monkeypatch):
    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    from youcut.comic.pipeline import ComicPipelineError
    config = PipelineConfig(output_dir=env["tmp"] / "out")

    with pytest.raises(ComicPipelineError, match=r"não encontrada|Sessão"):
        run_comic_pipeline(
            video,
            config,
            session_id="does-not-exist",
            image_provider=FakeImageProvider(),
            i2v_provider=FakeI2VProvider(),
        )


def test_resumed_session_skips_cast_and_script_calls(env, tmp_path, monkeypatch):
    video = _build_video(tmp_path)
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(output_dir=env["tmp"] / "out", comic_cost_cap_usd=100.0)

    first = run_comic_pipeline(
        video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    detect_called = {"n": 0}
    plan_called = {"n": 0}

    def _spy_detect(*a, **kw):
        detect_called["n"] += 1
        return _build_cast()

    def _spy_plan(*a, **kw):
        plan_called["n"] += 1
        return _build_panels()

    monkeypatch.setattr("youcut.comic.pipeline.detect_cast", _spy_detect)
    monkeypatch.setattr("youcut.comic.pipeline.plan_panels", _spy_plan)

    second = run_comic_pipeline(
        video,
        config,
        session_id=first.session_id,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    assert second.session_id == first.session_id
    assert detect_called["n"] == 0
    assert plan_called["n"] == 0
    loaded = load_motion_comic_session(first.session_id)
    assert loaded.session_id == first.session_id
