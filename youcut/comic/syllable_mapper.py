"""Syllable-level lip-sync derivation from WordTimestamps.

Função pura, sem I/O nem chamada de rede. Transforma a saída word-level do
Whisper em uma trilha de `MouthEvent`s cobrindo 100% do intervalo do clipe.
Hifenização via `pyphen` (locale derivado da transcrição). Mapeamento da vogal
dominante de cada sílaba para `MouthShape`. Silêncios entre palavras viram
`CLOSED`. Sílabas curtas (< 80ms) são fundidas com a vizinha mais curta.
"""

from __future__ import annotations

import logging

import pyphen

from youcut.models import MouthEvent, MouthShape, WordTimestamp

logger = logging.getLogger(__name__)


# ── Constantes ─────────────────────────────────────────────────────────────

MIN_SYLLABLE_DUR_SEC = 0.080
SILENCE_GAP_THRESHOLD_SEC = 0.120

_VOWEL_TO_SHAPE: dict[str, MouthShape] = {
    "a": MouthShape.OPEN_WIDE,
    "á": MouthShape.OPEN_WIDE,
    "â": MouthShape.OPEN_WIDE,
    "ã": MouthShape.OPEN_WIDE,
    "à": MouthShape.OPEN_WIDE,
    "e": MouthShape.OPEN_WIDE,
    "é": MouthShape.OPEN_WIDE,
    "ê": MouthShape.OPEN_WIDE,
    "i": MouthShape.OPEN_MID,
    "í": MouthShape.OPEN_MID,
    "y": MouthShape.OPEN_MID,
    "o": MouthShape.OPEN_ROUND,
    "ó": MouthShape.OPEN_ROUND,
    "ô": MouthShape.OPEN_ROUND,
    "õ": MouthShape.OPEN_ROUND,
    "u": MouthShape.OPEN_ROUND,
    "ú": MouthShape.OPEN_ROUND,
    "ü": MouthShape.OPEN_ROUND,
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _dominant_vowel(syllable: str) -> MouthShape:
    """Vogal dominante → MouthShape; sem vogal → CLOSED."""
    for char in syllable.lower():
        if char in _VOWEL_TO_SHAPE:
            return _VOWEL_TO_SHAPE[char]
    return MouthShape.CLOSED


def _hyphenate(word: str, *, locale: str, fallback_locale: str) -> list[str]:
    """Retorna sílabas de `word`. Em locale ausente cai em fallback com warning."""
    try:
        dic = pyphen.Pyphen(lang=locale)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning(
            "pyphen sem dicionário para locale=%s (%s); usando fallback=%s",
            locale, exc, fallback_locale,
        )
        dic = pyphen.Pyphen(lang=fallback_locale)
    inserted = dic.inserted(word)
    return [s for s in inserted.split("-") if s]


def _distribute_word_to_events(
    word: WordTimestamp,
    syllables: list[str],
    character_id: str,
) -> list[MouthEvent]:
    """Distribui [word.start, word.end] entre sílabas, proporcional a #chars."""
    if not syllables:
        return []
    total_chars = sum(len(s) for s in syllables) or 1
    duration = max(0.0, word.end - word.start)

    events: list[MouthEvent] = []
    cursor = word.start
    for idx, syl in enumerate(syllables):
        is_last = idx == len(syllables) - 1
        ev_end = word.end if is_last else cursor + duration * (len(syl) / total_chars)
        events.append(
            MouthEvent(
                character_id=character_id,
                start_sec=cursor,
                end_sec=ev_end,
                shape=_dominant_vowel(syl),
            )
        )
        cursor = ev_end
    return events


def _smooth_short_events(
    events: list[MouthEvent],
    *,
    min_dur: float = MIN_SYLLABLE_DUR_SEC,
) -> list[MouthEvent]:
    """Funde eventos < min_dur com a vizinha mais curta (preserva shape do vizinho)."""
    if len(events) <= 1:
        return events

    while True:
        merged = False
        for i, ev in enumerate(events):
            if ev.end_sec - ev.start_sec >= min_dur:
                continue
            left_dur = (events[i - 1].end_sec - events[i - 1].start_sec) if i > 0 else float("inf")
            right_dur = (
                events[i + 1].end_sec - events[i + 1].start_sec
                if i + 1 < len(events) else float("inf")
            )
            if left_dur == float("inf") and right_dur == float("inf"):
                # único evento — não há como fundir
                break
            if left_dur <= right_dur:
                events[i - 1] = events[i - 1].model_copy(update={"end_sec": ev.end_sec})
                events.pop(i)
            else:
                events[i + 1] = events[i + 1].model_copy(update={"start_sec": ev.start_sec})
                events.pop(i)
            merged = True
            break
        if not merged:
            break
    return events


# ── API pública ────────────────────────────────────────────────────────────


def derive_lipsync_track(
    words: list[WordTimestamp],
    character_id: str,
    *,
    locale: str = "pt_BR",
    fallback_locale: str = "pt_BR",
) -> list[MouthEvent]:
    """Transforma `words` em MouthEvents cobrindo 100% do intervalo.

    Args:
        words: WordTimestamps em ordem cronológica.
        character_id: id do personagem que articula a fala.
        locale: locale pyphen para hifenização (ex.: "pt_BR", "en_US").
        fallback_locale: locale usado quando `locale` não tem dicionário pyphen.

    Returns:
        Lista de MouthEvents contígua e sem sobreposição. Lista vazia se
        `words` for vazia.
    """
    if not words:
        return []

    events: list[MouthEvent] = []
    prev_end: float | None = None

    for word in words:
        clean = word.word.strip()
        if not clean:
            prev_end = word.end if prev_end is None else max(prev_end, word.end)
            continue

        if prev_end is not None and word.start > prev_end:
            gap = word.start - prev_end
            if gap > SILENCE_GAP_THRESHOLD_SEC:
                events.append(
                    MouthEvent(
                        character_id=character_id,
                        start_sec=prev_end,
                        end_sec=word.start,
                        shape=MouthShape.CLOSED,
                    )
                )
            elif events:
                events[-1] = events[-1].model_copy(update={"end_sec": word.start})

        syllables = _hyphenate(clean, locale=locale, fallback_locale=fallback_locale) or [clean]
        events.extend(_distribute_word_to_events(word, syllables, character_id))
        prev_end = word.end

    return _smooth_short_events(events)
