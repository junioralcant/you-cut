"""Detecção de menções de jogadores na transcrição.

Algoritmo: para cada palavra da transcrição (já com timestamps), monta
n-gramas (até o tamanho do maior alias do catálogo) e testa contra o
``alias_index``. Determinístico, O(N · K) onde N = palavras e K =
``max_alias_tokens``.

Não tenta resolver ambiguidade — quando um alias bate em múltiplos
profiles, o detector retorna **todos** como ``PlayerMention`` candidatos.
O :mod:`youcut.players.disambiguator` é quem decide qual fica.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable

from youcut.models import TranscriptionResult, TranscriptionSegment, WordTimestamp
from youcut.players.catalog import PlayerCatalog
from youcut.players.models import PlayerMention

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalize_word(word: str) -> str:
    """Lowercase + remove acentos + remove pontuação adjacente.

    Espelha :func:`youcut.players.catalog._normalize` para garantir que
    transcrição e aliases falem o mesmo dialeto.
    """
    nfkd = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return _PUNCT_RE.sub("", stripped.lower()).strip()


def slice_transcript_for_clip(
    transcript: TranscriptionResult,
    start_time: float,
    end_time: float,
) -> list[WordTimestamp]:
    """Filtra palavras da transcrição cujo timestamp cai dentro do clipe.

    Critério de overlap: a palavra é incluída se ``start < end_time`` e
    ``end > start_time`` (ou seja, intersecção não-vazia). Isso evita
    perder palavras de fronteira.
    """
    if start_time >= end_time:
        return []
    words: list[WordTimestamp] = []
    for segment in transcript.segments:
        # Segment-level early skip
        if segment.end <= start_time or segment.start >= end_time:
            continue
        for word in segment.words:
            if word.end <= start_time or word.start >= end_time:
                continue
            words.append(word)
    return words


def detect_players(
    words: Iterable[WordTimestamp],
    catalog: PlayerCatalog,
) -> list[PlayerMention]:
    """Encontra menções de jogadores em uma sequência de palavras.

    Retorna **todas** as menções candidatas, incluindo as ambíguas
    (mesmo alias casa com múltiplos profiles → uma ``PlayerMention`` por
    profile). Cabe ao caller chamar :func:`disambiguate_mentions` se
    quiser uma menção única por trecho ambíguo.
    """
    word_list = list(words)
    if not word_list:
        return []
    index = catalog.alias_index
    if not index:
        return []

    max_tokens = catalog.max_alias_tokens
    mentions: list[PlayerMention] = []
    normalized = [_normalize_word(w.word) for w in word_list]

    n = len(word_list)
    i = 0
    while i < n:
        if not normalized[i]:
            i += 1
            continue
        # Tenta o maior n-grama possível primeiro (greedy) — assim
        # "Vinicius Junior" casa como bloco em vez de virar duas
        # menções soltas a "Vinicius" e "Junior".
        matched_len = 0
        matched_profiles: list = []
        matched_key = ""
        for length in range(min(max_tokens, n - i), 0, -1):
            tokens = normalized[i : i + length]
            if any(not t for t in tokens):
                continue
            key = " ".join(tokens)
            profiles = index.get(key)
            if profiles:
                matched_len = length
                matched_profiles = profiles
                matched_key = key
                break
        if matched_len == 0:
            i += 1
            continue
        start = word_list[i].start
        end = word_list[i + matched_len - 1].end
        for profile in matched_profiles:
            mentions.append(
                PlayerMention(
                    profile=profile,
                    alias_hit=matched_key,
                    start=start,
                    end=end,
                )
            )
        logger.debug(
            "Menção detectada: alias=%r → %d profile(s) [t=%.2f-%.2f]",
            matched_key,
            len(matched_profiles),
            start,
            end,
        )
        i += matched_len
    return mentions
