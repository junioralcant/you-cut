"""E2E do pipeline `youcut comic` — vídeo sintético 5 s + fakes determinísticos.

Marca como `integration` porque depende de FFmpeg/ffprobe reais para
fixtures e composição. Anthropic/OpenAI são bypassados via
`monkeypatch` injetando dados sintéticos.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._fakes.comic_providers import FakeI2VProvider, FakeImageProvider
from youcut.comic.pipeline import run_comic_pipeline
from youcut.comic.run_report import SCHEMA_VERSION, build_run_report
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
def short_video(tmp_path):
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


def _audio_md5(video_path: Path) -> str:
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-f",
        "md5",
        "-",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()[0].replace("MD5=", "")


def _ffprobe_duration(p: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(p),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def _build_transcription(duration: float = 5.0) -> TranscriptionResult:
    words = [
        WordTimestamp(word=f"word{i}", start=i * 0.5, end=i * 0.5 + 0.4)
        for i in range(10)
    ]
    return TranscriptionResult(
        segments=[TranscriptionSegment(start=0.0, end=duration, text="texto", words=words)],
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


def _build_panels(n: int = 2, total: float = 5.0) -> list[Panel]:
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


def _patch_pipeline(monkeypatch, *, transcription=None, cast=None, panels=None):
    transcription = transcription or _build_transcription()
    cast = cast or _build_cast()
    panels = panels or _build_panels()
    monkeypatch.setattr("youcut.comic.pipeline.transcribe", lambda v, c: transcription)
    monkeypatch.setattr(
        "youcut.comic.pipeline.diarize",
        lambda v, c: [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=5.0)],
    )
    monkeypatch.setattr("youcut.comic.pipeline.detect_cast", lambda *a, **kw: cast)
    monkeypatch.setattr("youcut.comic.pipeline.plan_panels", lambda *a, **kw: panels)
    return cast, panels


# ---------------------------------------------------------------------------
# E2E
# ---------------------------------------------------------------------------


def test_e2e_pipeline_produces_final_mp4_with_correct_duration(env, short_video, monkeypatch):
    _patch_pipeline(monkeypatch)

    fake_image = FakeImageProvider(size=(1024, 1792))
    fake_i2v = FakeI2VProvider()

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", 
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
    duration = _ffprobe_duration(session.output_path)
    assert abs(duration - 5.0) <= 0.2


def test_e2e_pipeline_preserves_audio_bit_identical(env, short_video, monkeypatch):
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", 
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    session = run_comic_pipeline(
        short_video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    assert _audio_md5(session.output_path) == _audio_md5(short_video)


def test_e2e_pipeline_writes_run_report_with_schema(env, short_video, monkeypatch):
    _patch_pipeline(monkeypatch)

    from youcut.config import PipelineConfig
    config = PipelineConfig(comic_animation_engine="panels", 
        output_dir=env["tmp"] / "out",
        comic_cost_cap_usd=100.0,
    )

    session = run_comic_pipeline(
        short_video,
        config,
        image_provider=FakeImageProvider(size=(1024, 1792)),
        i2v_provider=FakeI2VProvider(),
    )

    output_dir = env["tmp"] / "out" / short_video.stem
    report_path = output_dir / "comic" / "run_report.json"
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["session_id"] == session.session_id
    assert payload["n_panels"] == 2
    assert payload["n_static_fallbacks"] == 0
    assert payload["total_cost_usd"] == pytest.approx(session.total_cost_usd, abs=1e-4)
    assert payload["total_seconds"] > 0
    assert "panel_clip_p50_seconds" in payload
    assert "panel_clip_p95_seconds" in payload
    assert isinstance(payload["panels"], list)
    assert len(payload["panels"]) == 2
    for panel in payload["panels"]:
        assert {
            "panel_index",
            "clip_seconds",
            "was_static_fallback",
            "image_attempts",
            "i2v_attempts",
            "cost_usd",
        } <= panel.keys()


# ---------------------------------------------------------------------------
# build_run_report unit
# ---------------------------------------------------------------------------


def test_build_run_report_records_static_fallback():
    from youcut.models import MotionComicSession, PanelRenderResult
    from datetime import datetime, timezone

    session = MotionComicSession(
        session_id="abc",
        video_path=Path("v.mp4"),
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        cast=[CastMember(character_id="x")],
        panels=[],
        panel_results=[
            PanelRenderResult(
                panel_index=0,
                base_image_path=Path("a.png"),
                clip_path=Path("a.mp4"),
                clip_seconds=2.0,
                was_static_fallback=False,
                cost_usd=0.05,
            ),
            PanelRenderResult(
                panel_index=1,
                base_image_path=Path("b.png"),
                clip_path=Path("b.mp4"),
                clip_seconds=3.0,
                was_static_fallback=True,
                cost_usd=0.04,
            ),
        ],
        total_cost_usd=0.09,
    )
    report = build_run_report(session)
    assert report["n_panels"] == 2
    assert report["n_static_fallbacks"] == 1
    assert report["total_seconds"] == 5.0
    assert report["total_cost_usd"] == 0.09
