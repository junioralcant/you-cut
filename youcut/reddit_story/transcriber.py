"""Whisper word-level timestamps pro áudio narrado (sem cache MD5 — cada
sessão tem narração fresca, cache desperdiça espaço).
"""

from __future__ import annotations

from pathlib import Path

from youcut.models import TranscriptionSegment, WordTimestamp


def transcribe_words(audio_path: Path, *, model_name: str) -> tuple[list[WordTimestamp], float]:
    from faster_whisper import WhisperModel  # type: ignore[import]

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments_iter, _info = model.transcribe(
        str(audio_path), language="en", word_timestamps=True
    )

    words: list[WordTimestamp] = []
    end_time = 0.0
    for seg in segments_iter:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(WordTimestamp(word=w.word, start=w.start, end=w.end))
            end_time = max(end_time, w.end)
    return words, end_time


def words_to_segments(words: list[WordTimestamp]) -> list[TranscriptionSegment]:
    """Agrupa words em segments de ~10s pra facilitar inspeção/debug."""
    if not words:
        return []
    segs: list[TranscriptionSegment] = []
    buf: list[WordTimestamp] = []
    seg_start = words[0].start
    for w in words:
        buf.append(w)
        if w.end - seg_start > 10:
            segs.append(
                TranscriptionSegment(
                    start=seg_start,
                    end=buf[-1].end,
                    text="".join(b.word for b in buf),
                    words=list(buf),
                )
            )
            buf = []
            if w is words[-1]:
                break
            seg_start = w.end
    if buf:
        segs.append(
            TranscriptionSegment(
                start=seg_start,
                end=buf[-1].end,
                text="".join(b.word for b in buf),
                words=list(buf),
            )
        )
    return segs
