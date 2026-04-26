import logging
import subprocess
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import SpeakerSegment

logger = logging.getLogger(__name__)


def _get_clip_duration(clip_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _single_speaker_fallback(clip_path: Path) -> list[SpeakerSegment]:
    duration = _get_clip_duration(clip_path)
    return [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=max(duration, 0.0))]


def diarize(clip_path: Path, config: PipelineConfig) -> list[SpeakerSegment]:
    """Return speaker segments for the given clip.

    Falls back to a single SPEAKER_00 segment covering the entire clip when
    pyannote is not installed or the HuggingFace token is missing.
    """
    if config.huggingface_token is None:
        logger.info("diarize: huggingface_token ausente — assumindo locutor único")
        return _single_speaker_fallback(clip_path)

    try:
        from pyannote.audio import Pipeline  # type: ignore[import]
    except ImportError:
        logger.info("diarize: pyannote.audio não instalado — assumindo locutor único")
        return _single_speaker_fallback(clip_path)

    logger.info("diarize: iniciando diarização de %s", clip_path.name)
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.huggingface_token,
        )
        diarization = pipeline(str(clip_path))
    except Exception as exc:
        logger.warning("diarize: falha ao executar pyannote (%s) — usando fallback locutor único", exc)
        return _single_speaker_fallback(clip_path)

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(speaker_id=speaker, start=turn.start, end=turn.end))

    logger.info("diarize: %d segmentos de locutor encontrados em %s", len(segments), clip_path.name)

    if not segments:
        return _single_speaker_fallback(clip_path)

    return segments
