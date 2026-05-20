"""Timeline de jogadores para sincronizar foto local com a fala.

Dado o transcript já existente e o catálogo local, constrói uma sequência
contínua de :class:`PlayerSegment` cobrindo todo o intervalo do clipe
``[0, clip_duration]``. Cada segmento aponta para o ``PlayerProfile`` cuja
foto deve aparecer naquele intervalo.

Regras:

- Antes da primeira menção do catálogo, a primeira foto detectada é
  **estendida para trás** até ``t=0``.
- Quando o apresentador cita um nome que NÃO está no catálogo, nenhuma
  troca acontece — a foto anterior permanece.
- Menções consecutivas do mesmo jogador são mescladas num único segmento
  (sem flicker entre citações).
- O ponto de troca entre duas menções de jogadores diferentes é o
  ``start`` da menção seguinte (a foto antiga sai assim que o nome novo
  começa a ser dito).

Diferente de :func:`youcut.players.disambiguator.disambiguate_mentions`,
aqui NÃO deduplicamos por slug — o mesmo jogador citado várias vezes ao
longo do clipe gera um segmento por intervalo entre menções, com merge
apenas quando duas menções consecutivas são do mesmo jogador.

Função pura — testável sem ffmpeg ou chamadas pagas.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from youcut.models import TranscriptionResult, WordTimestamp
from youcut.players.catalog import PlayerCatalog
from youcut.players.detector import detect_players, slice_transcript_for_clip
from youcut.players.disambiguator import _claude_pick_profile, _context_snippet
from youcut.players.models import PlayerMention, PlayerProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerSegment:
    """Intervalo contínuo (clip-relativo, em segundos) com uma foto fixa."""

    start: float
    end: float
    profile: PlayerProfile


def _resolve_overlapping(
    mentions: list[PlayerMention],
    transcript_words: list[WordTimestamp],
    anthropic_client: Any | None,
    claude_model: str,
) -> list[PlayerMention]:
    """Resolve menções ambíguas SEM deduplicar por slug.

    Diferente de :func:`disambiguate_mentions`, preservamos todas as
    ocorrências temporais de um mesmo jogador — é justamente isso que
    permite a timeline ter múltiplos segmentos do mesmo perfil em
    momentos distintos do clipe.
    """
    buckets: dict[tuple[float, float], list[PlayerMention]] = defaultdict(list)
    for m in mentions:
        buckets[(m.start, m.end)].append(m)

    resolved: list[PlayerMention] = []
    for group in buckets.values():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        candidates = [m.profile for m in group]
        if anthropic_client is None:
            logger.info(
                "Timeline: menção ambígua [%.2f-%.2f] descartada — sem Claude. Candidatos: %s",
                group[0].start, group[0].end, [c.slug for c in candidates],
            )
            continue
        snippet = _context_snippet(transcript_words, group[0])
        chosen = _claude_pick_profile(
            anthropic_client, claude_model, snippet, candidates,
        )
        if chosen is None:
            logger.info(
                "Timeline: Claude não decidiu [%.2f-%.2f] — descartada",
                group[0].start, group[0].end,
            )
            continue
        resolved.append(
            PlayerMention(
                profile=chosen,
                alias_hit=group[0].alias_hit,
                start=group[0].start,
                end=group[0].end,
            )
        )
    return sorted(resolved, key=lambda m: m.start)


def build_player_timeline(
    transcription: TranscriptionResult | None,
    clip_start_time: float,
    clip_end_time: float,
    catalog: PlayerCatalog,
    *,
    anthropic_client: Any | None = None,
    claude_model: str = "claude-sonnet-4-6",
) -> list[PlayerSegment]:
    """Constrói a timeline de fotos de jogadores para o clipe.

    Retorna uma lista de :class:`PlayerSegment` cobrindo ``[0, clip_duration]``,
    onde ``clip_duration = clip_end_time - clip_start_time``. Vazia se não
    há transcrição, catálogo vazio, ou nenhuma menção detectada.
    """
    if transcription is None:
        return []
    if clip_end_time <= clip_start_time:
        return []
    if not catalog.profiles:
        return []

    words = slice_transcript_for_clip(transcription, clip_start_time, clip_end_time)
    if not words:
        return []

    raw_mentions = detect_players(words, catalog)
    if not raw_mentions:
        return []

    resolved = _resolve_overlapping(
        raw_mentions, words, anthropic_client, claude_model,
    )
    if not resolved:
        return []

    clip_duration = clip_end_time - clip_start_time

    segments: list[PlayerSegment] = []
    for idx, mention in enumerate(resolved):
        rel_start = max(0.0, min(clip_duration, mention.start - clip_start_time))
        if idx + 1 < len(resolved):
            rel_end = max(0.0, min(clip_duration, resolved[idx + 1].start - clip_start_time))
        else:
            rel_end = clip_duration

        if rel_end <= rel_start:
            continue

        if segments and segments[-1].profile.slug == mention.profile.slug:
            last = segments[-1]
            segments[-1] = PlayerSegment(start=last.start, end=rel_end, profile=last.profile)
            continue

        segments.append(PlayerSegment(start=rel_start, end=rel_end, profile=mention.profile))

    if not segments:
        return []

    first = segments[0]
    if first.start > 0.0:
        segments[0] = PlayerSegment(start=0.0, end=first.end, profile=first.profile)

    last = segments[-1]
    if last.end < clip_duration:
        segments[-1] = PlayerSegment(start=last.start, end=clip_duration, profile=last.profile)

    logger.info(
        "Player timeline: %d segmento(s) — %s",
        len(segments),
        " → ".join(f"{s.profile.slug}@{s.start:.1f}-{s.end:.1f}" for s in segments),
    )
    return segments
