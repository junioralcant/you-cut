"""Desambiguação de menções via Claude.

Quando uma menção curta (ex.: ``"Danilo"``) bate em múltiplos profiles
do catálogo (Danilo do Botafogo vs. Danilo do Flamengo), o detector
gera uma :class:`PlayerMention` por profile. Este módulo usa o Claude
para olhar o contexto do trecho (algumas palavras antes e depois) e
escolher o profile correto — ou descartar a menção se nem o LLM
consegue decidir.

Fora desse cenário ambíguo, o disambiguator é no-op (passa direto).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from youcut.models import WordTimestamp
from youcut.players.models import PlayerMention, PlayerProfile

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW_WORDS = 12  # palavras antes/depois para dar contexto ao LLM


def _group_overlapping(mentions: list[PlayerMention]) -> list[list[PlayerMention]]:
    """Agrupa menções que compartilham o mesmo intervalo de tempo.

    O detector emite uma menção por profile candidato quando há
    ambiguidade — todas têm exatamente o mesmo ``(start, end)``. Esta
    função reúne esses grupos pra processá-los juntos.
    """
    buckets: dict[tuple[float, float], list[PlayerMention]] = defaultdict(list)
    for m in mentions:
        buckets[(m.start, m.end)].append(m)
    return list(buckets.values())


def _context_snippet(
    words: list[WordTimestamp],
    mention: PlayerMention,
    window: int = _CONTEXT_WINDOW_WORDS,
) -> str:
    """Retorna o trecho da transcrição em torno da menção.

    Usado pra dar contexto ao Claude. Janela de ``window`` palavras
    antes e depois da menção; se a transcrição for menor, retorna o
    que tiver.
    """
    if not words:
        return ""
    anchor = None
    for idx, w in enumerate(words):
        if w.start >= mention.start:
            anchor = idx
            break
    if anchor is None:
        anchor = len(words) - 1
    lo = max(0, anchor - window)
    hi = min(len(words), anchor + window + 1)
    return " ".join(w.word for w in words[lo:hi])


def _claude_pick_profile(
    anthropic_client: Any,
    model: str,
    snippet: str,
    candidates: list[PlayerProfile],
) -> PlayerProfile | None:
    """Pergunta ao Claude qual candidate o trecho está mencionando.

    Retorna ``None`` quando o LLM responde algo que não casa com nenhum
    candidate (caller deve descartar a menção, pra evitar usar a foto
    errada).
    """
    options_text = "\n".join(
        f"- slug={c.slug} | nome={c.display_name}" for c in candidates
    )
    prompt = (
        "Você está desambiguando uma menção de jogador de futebol em uma "
        "transcrição de áudio em português. Dado o trecho abaixo, decida "
        "qual dos jogadores listados é o referenciado.\n\n"
        f"Trecho:\n\"{snippet}\"\n\n"
        f"Candidatos:\n{options_text}\n\n"
        "Responda APENAS em JSON com a chave \"slug\" contendo o slug "
        "escolhido, OU o valor null se o trecho não permite decidir com "
        "confiança. Exemplo: {\"slug\": \"danilo_botafogo\"} ou {\"slug\": null}."
    )
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("Claude desambiguação falhou: %s — descartando menção", exc)
        return None

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")
    text = text.strip()
    if not text:
        return None
    # Tenta extrair o JSON mesmo se vier embrulhado em texto extra.
    try:
        first = text.index("{")
        last = text.rindex("}") + 1
        payload = json.loads(text[first:last])
    except (ValueError, json.JSONDecodeError):
        logger.debug("Resposta da desambiguação não é JSON válido: %r", text)
        return None
    chosen_slug = payload.get("slug")
    if not chosen_slug:
        return None
    for candidate in candidates:
        if candidate.slug == chosen_slug:
            return candidate
    logger.debug(
        "Claude escolheu slug %r que não está nos candidatos %s",
        chosen_slug,
        [c.slug for c in candidates],
    )
    return None


def disambiguate_mentions(
    mentions: Iterable[PlayerMention],
    transcript_words: list[WordTimestamp],
    anthropic_client: Any | None,
    claude_model: str,
) -> list[PlayerMention]:
    """Resolve menções ambíguas e retorna lista deduplicada por slug.

    - Menções sem conflito de tempo passam direto.
    - Quando múltiplos profiles compartilham o mesmo ``(start, end)`` e
      ``anthropic_client`` está disponível, pergunta ao Claude.
    - Sem cliente Claude, a menção ambígua é descartada (política
      conservadora — preferimos não exibir uma foto a exibir a errada).

    A saída é deduplicada por ``profile.slug``: se o mesmo jogador é
    mencionado várias vezes no clipe, mantemos só a primeira ocorrência.
    """
    grouped = _group_overlapping(list(mentions))
    resolved: list[PlayerMention] = []
    for group in grouped:
        if len(group) == 1:
            resolved.append(group[0])
            continue
        # Ambíguo: mesmo intervalo, profiles distintos
        candidates = [m.profile for m in group]
        if anthropic_client is None:
            logger.info(
                "Menção ambígua [%.2f-%.2f] descartada — sem cliente Claude. "
                "Candidatos: %s",
                group[0].start,
                group[0].end,
                [c.slug for c in candidates],
            )
            continue
        snippet = _context_snippet(transcript_words, group[0])
        chosen = _claude_pick_profile(
            anthropic_client, claude_model, snippet, candidates
        )
        if chosen is None:
            logger.info(
                "Claude não decidiu menção ambígua [%.2f-%.2f] — descartada",
                group[0].start,
                group[0].end,
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

    # Dedup por slug, mantendo a primeira menção temporal.
    seen: set[str] = set()
    deduped: list[PlayerMention] = []
    for m in sorted(resolved, key=lambda x: x.start):
        if m.profile.slug in seen:
            continue
        seen.add(m.profile.slug)
        deduped.append(m)
    return deduped
