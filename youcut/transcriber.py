import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from youcut.config import PipelineConfig
from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp

logger = logging.getLogger(__name__)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(video_path: Path) -> Path:
    return video_path.parent / f"{video_path.stem}_transcript.json"


def _load_cache(cache_file: Path, current_hash: str) -> TranscriptionResult | None:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if data.get("md5") != current_hash:
            logger.info("Cache inválido: hash MD5 diferente, re-transcrevendo.")
            return None
        return TranscriptionResult.model_validate(data["result"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Cache corrompido em %s: %s — ignorando.", cache_file, exc)
        return None


def _save_cache(cache_file: Path, result: TranscriptionResult, file_hash: str) -> None:
    payload = {
        "md5": file_hash,
        "result": json.loads(result.model_dump_json()),
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cache de transcrição salvo em %s.", cache_file)


def _transcribe_faster_whisper(
    video_path: Path, model_name: str, language: str
) -> TranscriptionResult:
    from faster_whisper import WhisperModel  # type: ignore[import]

    logger.info("Transcrevendo com faster-whisper (modelo=%s).", model_name)
    model = WhisperModel(model_name, device="auto")
    segments_iter, info = model.transcribe(
        str(video_path), language=language, word_timestamps=True
    )

    segments: list[TranscriptionSegment] = []
    for seg in segments_iter:
        words: list[WordTimestamp] = []
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(word=w.word, start=w.start, end=w.end))
        segments.append(
            TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
            )
        )

    return TranscriptionResult(
        segments=segments,
        language=info.language,
        source_path=video_path,
    )


def _transcribe_openai_whisper(
    video_path: Path, model_name: str, language: str
) -> TranscriptionResult:
    import whisper  # type: ignore[import]

    logger.info("Transcrevendo com openai-whisper (modelo=%s).", model_name)
    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), language=language, word_timestamps=True)

    segments: list[TranscriptionSegment] = []
    for seg in result.get("segments", []):
        words: list[WordTimestamp] = []
        for w in seg.get("words", []):
            words.append(
                WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
            )
        segments.append(
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=words,
            )
        )

    return TranscriptionResult(
        segments=segments,
        language=result.get("language", language),
        source_path=video_path,
    )


def transcribe(video_path: Path, config: PipelineConfig) -> TranscriptionResult:
    video_path = video_path.resolve()
    file_hash = _md5(video_path)
    cache_file = _cache_path(video_path)

    cached = _load_cache(cache_file, file_hash)
    if cached is not None:
        logger.info("Cache de transcrição reutilizado para %s.", video_path.name)
        return cached

    language = getattr(config, "whisper_language", "pt")

    try:
        result = _transcribe_faster_whisper(video_path, config.whisper_model, language)
    except ImportError:
        logger.warning(
            "faster-whisper não encontrado; usando openai-whisper como fallback."
        )
        try:
            result = _transcribe_openai_whisper(video_path, config.whisper_model, language)
        except ImportError:
            raise RuntimeError(
                "Nenhum backend de transcrição disponível. "
                "Instale faster-whisper (recomendado) ou openai-whisper."
            ) from None

    _save_cache(cache_file, result, file_hash)
    return result
