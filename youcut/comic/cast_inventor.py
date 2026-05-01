"""Cast Inventor — inventa reatores fictícios (audiência) a partir do áudio.

Alternativa ao :mod:`visual_analyzer` quando ``config.comic_invent_cast=True``:
em vez de mostrar o falante do vídeo, o pipeline trata a fala como **voz em
off** e inventa de 2 a 4 personagens-reatores (audiência fictícia) que
**escutam** e reagem com expressão facial e linguagem corporal coerentes
com o tom do áudio. Os reatores nunca falam — todos têm
``speaker_id=None``, o que faz o ``panel_renderer`` ativar a branch
"VOZ EM OFF" (boca fechada, só reação).

Os descritores são livres e ficcionais; nenhum frame do vídeo original é
consultado, garantindo que o rosto real do falante jamais apareça em
quadro nem como referência das âncoras.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import anthropic

from youcut.comic.panel_renderer import sanitize_brand_mentions
from youcut.config import PipelineConfig
from youcut.models import CastMember, SpeakerSegment, TranscriptionResult

logger = logging.getLogger(__name__)


REACTOR_MIN: int = 2
REACTOR_MAX: int = 4


_INVENT_TOOL: dict[str, Any] = {
    "name": "invent_reactors",
    "description": (
        "Inventa entre 2 e 4 personagens-reatores (audiência fictícia) que "
        "escutam o áudio. NENHUM dos reatores é o falante; o falante é VOZ "
        "EM OFF e nunca aparece em quadro. Os reatores são variados em "
        "demografia (idade, gênero, estilo) e em postura emocional "
        "(ex.: cético, espantado, indignado, animado), de modo que diferentes "
        "trechos da fala possam ser ilustrados com reações distintas."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "minItems": REACTOR_MIN,
                "maxItems": REACTOR_MAX,
                "items": {
                    "type": "object",
                    "properties": {
                        "narrative_role": {
                            "type": "string",
                            "description": (
                                "Papel narrativo curto do REATOR (ex.: 'jovem "
                                "espantada', 'velho cético', 'adolescente "
                                "entediado'). NUNCA é o falante."
                            ),
                        },
                        "reaction_archetype": {
                            "type": "string",
                            "description": (
                                "Postura emocional dominante deste reator "
                                "(ex.: 'choque', 'deboche', 'indignação', "
                                "'concordância', 'tédio'). Cada reator deve "
                                "ter um arquétipo distinto."
                            ),
                        },
                        "gender_apparent": {"type": "string"},
                        "age_apparent": {"type": "string"},
                        "hair": {"type": "string"},
                        "facial_hair": {"type": "string"},
                        "skin": {"type": "string"},
                        "clothing": {"type": "string"},
                        "accessories": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "narrative_role",
                        "reaction_archetype",
                        "gender_apparent",
                        "age_apparent",
                        "hair",
                        "skin",
                        "clothing",
                    ],
                },
            }
        },
        "required": ["characters"],
    },
}


_SYSTEM_PROMPT = (
    "Você é um diretor de arte que inventa personagens-reatores fictícios "
    "para uma animação editorial em estilo caricatura pastel. O áudio será "
    "tratado como VOZ EM OFF — o falante NUNCA aparece em quadro. Sua "
    "tarefa: inventar entre 2 e 4 reatores (audiência fictícia) que vão "
    "ESCUTAR o áudio e reagir com expressão e gesto. Cada reator deve ter:\n"
    "- demografia distinta (idade, gênero, etnia, estilo de roupa) — "
    "garanta variedade entre eles;\n"
    "- um arquétipo emocional distinto (ex.: chocado, cético, indignado, "
    "animado, entediado, debochado) — para que diferentes trechos da fala "
    "possam ser ilustrados com reatores distintos.\n"
    "PROIBIDO inventar o falante. Os reatores não falam. Vocabulário em "
    "pt-BR; descrições curtas e objetivas; sem estereótipos ofensivos."
)


def _truncate_transcript(transcription: TranscriptionResult, max_chars: int = 4000) -> str:
    text = " ".join(seg.text.strip() for seg in transcription.segments if seg.text.strip())
    if not text:
        return "(sem texto transcrito)"
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "x"


def _build_text_card(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("gender_apparent", "age_apparent", "hair", "facial_hair", "skin", "clothing"):
        val = (raw.get(key) or "").strip()
        if val:
            parts.append(val)
    accessories = [a for a in (raw.get("accessories") or []) if a]
    if accessories:
        parts.append("acessórios: " + ", ".join(accessories))
    archetype = (raw.get("reaction_archetype") or "").strip()
    if archetype:
        parts.append(f"reação: {archetype}")
    role = (raw.get("narrative_role") or "").strip()
    if role:
        parts.append(f"papel: {role}")
    return "; ".join(parts) or (role or "reator fictício")


def _call_claude_text(
    client: anthropic.Anthropic,
    transcription: TranscriptionResult,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    user_text = (
        f"Invente entre {REACTOR_MIN} e {REACTOR_MAX} reatores fictícios "
        f"(audiência) que vão reagir ao áudio abaixo. O falante é VOZ EM OFF "
        f"e nunca aparece em quadro — não invente o falante. Garanta que os "
        f"reatores tenham demografias e arquétipos emocionais distintos para "
        f"cobrir diferentes momentos da fala.\n\n"
        f"TRANSCRIÇÃO:\n{_truncate_transcript(transcription)}"
    )

    try:
        response = client.with_options(timeout=60.0).messages.create(
            model=config.claude_model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_INVENT_TOOL],
            tool_choice={"type": "tool", "name": "invent_reactors"},
            messages=[{"role": "user", "content": user_text}],
        )
    except anthropic.APIError as exc:
        msg = getattr(exc, "message", None) or str(exc)
        raise RuntimeError(
            f"Erro na API do Claude ao inventar reatores a partir do áudio: {msg}"
        ) from exc

    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "invent_reactors"
        ):
            payload = getattr(block, "input", None) or {}
            return list(payload.get("characters") or [])
    return []


def _generic_fallback() -> list[CastMember]:
    return [
        CastMember(
            character_id="reator_chocado",
            kind="person",
            narrative_role="reator chocado",
            speaker_id=None,
            text_card=(
                "reator fictício genérico (Claude não retornou descritores); "
                "expressão dominante: choque/espanto"
            ),
        ),
        CastMember(
            character_id="reator_cetico",
            kind="person",
            narrative_role="reator cético",
            speaker_id=None,
            text_card=(
                "reator fictício genérico (Claude não retornou descritores); "
                "expressão dominante: deboche/ceticismo"
            ),
        ),
    ]


def invent_cast(
    transcription: TranscriptionResult,
    speakers: list[SpeakerSegment],  # noqa: ARG001 — mantido por simetria com detect_cast
    config: PipelineConfig,
    *,
    client: anthropic.Anthropic | None = None,
) -> list[CastMember]:
    """Inventa um cast fictício de 2-4 REATORES (audiência) a partir do áudio.

    Diferente do :func:`detect_cast`, este caminho:

    - **NÃO** olha frames do vídeo original — o rosto real nunca é referência;
    - **NÃO** inventa o falante — ele é tratado como voz em off;
    - inventa 2-4 reatores variados (demografia + arquétipo emocional);
    - todos os reatores ficam com ``speaker_id=None``, o que faz o
      ``panel_renderer`` aplicar a branch "VOZ EM OFF" (boca fechada, só
      reação corporal/facial) — exatamente o comportamento desejado.
    """

    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    raw_characters = _call_claude_text(client, transcription, config)
    if not raw_characters:
        logger.warning(
            "comic.cast_inventor: Claude não retornou reatores; usando fallback genérico (2)."
        )
        return _generic_fallback()

    cast: list[CastMember] = []
    used_ids: set[str] = set()
    for raw in raw_characters[:REACTOR_MAX]:
        role = sanitize_brand_mentions(
            (raw.get("narrative_role") or "reator").strip() or "reator"
        )
        base_slug = _slugify(role)
        slug = base_slug
        n = 1
        while slug in used_ids:
            n += 1
            slug = f"{base_slug}_{n}"
        used_ids.add(slug)

        cast.append(
            CastMember(
                character_id=slug,
                kind="person",
                gender_apparent=(raw.get("gender_apparent") or "").strip(),
                age_apparent=(raw.get("age_apparent") or "").strip(),
                hair=(raw.get("hair") or "").strip(),
                facial_hair=(raw.get("facial_hair") or "").strip(),
                skin=(raw.get("skin") or "").strip(),
                clothing=sanitize_brand_mentions((raw.get("clothing") or "").strip()),
                accessories=[
                    sanitize_brand_mentions(a)
                    for a in (raw.get("accessories") or [])
                    if a
                ],
                narrative_role=role,
                speaker_id=None,
                source_frame_path=None,
                text_card=sanitize_brand_mentions(_build_text_card(raw)),
            )
        )

    if len(cast) < REACTOR_MIN:
        logger.warning(
            "comic.cast_inventor: Claude retornou só %d reatores; completando com fallback.",
            len(cast),
        )
        for fb in _generic_fallback():
            if len(cast) >= REACTOR_MIN:
                break
            if fb.character_id in used_ids:
                continue
            used_ids.add(fb.character_id)
            cast.append(fb)

    logger.info(
        "comic.cast_inventor: %d reatores inventados (todos com speaker_id=None — voz em off).",
        len(cast),
    )
    return cast
