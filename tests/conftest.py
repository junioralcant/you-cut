from pathlib import Path

import pytest

from tests.create_fixtures import create_sample_mp4

_SAMPLE_MP4 = Path(__file__).parent / "fixtures" / "sample.mp4"
_FIXTURE_MIN_SIZE = 10_000


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_mp4():
    """Auto-generate tests/fixtures/sample.mp4 if the checked-in fixture is stale."""
    if _SAMPLE_MP4.exists() and _SAMPLE_MP4.stat().st_size > _FIXTURE_MIN_SIZE:
        if create_sample_mp4(force=False):
            return
    elif create_sample_mp4(force=False):
        return

    pytest.skip("Não foi possível gerar tests/fixtures/sample.mp4 para os testes.")


@pytest.fixture
def pipeline_config(monkeypatch):
    """PipelineConfig com valores de teste."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-fixture")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("CLIP_COUNT", raising=False)
    monkeypatch.delenv("SUBTITLE_STYLE", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)

    from youcut.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def sample_viral_clip():
    from youcut.models import ViralClip
    return ViralClip(
        title="Clipe de Teste",
        reason="Gancho forte no início",
        viral_score=8.5,
        start_time=10.0,
        end_time=45.0,
        description="Descrição de teste para o clipe",
        hashtags=["#teste", "#youcut"],
        thumbnail_idea="Frame com expressão de surpresa no segundo 15",
    )


@pytest.fixture
def sample_transcription_result():
    from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Olá mundo",
                words=[
                    WordTimestamp(word="Olá", start=0.0, end=0.5),
                    WordTimestamp(word="mundo", start=0.6, end=1.0),
                ],
            )
        ],
        language="pt",
        source_path=Path("test_video.mp4"),
    )
