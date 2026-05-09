"""Classificação de mood de faixas musicais via Claude (texto)."""
from __future__ import annotations

import logging
from typing import Iterable

import anthropic

from youcut.config import PipelineConfig

logger = logging.getLogger("youcut.music.classifier")

CANONICAL_MOODS: tuple[str, ...] = (
    "motivacional",
    "reflexivo",
    "energico",
    "emocional",
    "feliz",
    "dramatico",
)

_CLASSIFY_TOOL = {
    "name": "classify_track_mood",
    "description": (
        "Classifica uma faixa musical em exatamente um dos 6 moods canônicos do "
        "projeto YouCut, com base no título, descrição e tags do vídeo do YouTube."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mood": {
                "type": "string",
                "enum": list(CANONICAL_MOODS),
                "description": "Mood canônico que melhor descreve a faixa.",
            }
        },
        "required": ["mood"],
    },
}

_SYSTEM_PROMPT = (
    "Você é um curador musical experiente. Sua tarefa é classificar uma faixa "
    "musical em exatamente um dos 6 moods canônicos: motivacional, reflexivo, "
    "energico, emocional, feliz ou dramatico. Use o título, descrição e tags do "
    "vídeo do YouTube como base. Escolha o mood mais predominante. Não responda "
    "fora do enum."
)


def _format_tags(tags: Iterable[str]) -> str:
    cleaned = [str(t).strip() for t in tags if t and str(t).strip()]
    return ", ".join(cleaned) if cleaned else "(nenhuma)"


class TrackMoodClassifier:
    """Classifica faixas com 1 chamada Claude texto + tools com schema fechado.

    Em caso de falha de API, mood inválido ou ausência de output, retorna `None`
    — o syncer trata isso como mood `"indefinido"` (RF-09).
    """

    def __init__(self, config: PipelineConfig, *, timeout_s: float = 30.0) -> None:
        self._config = config
        self._timeout_s = timeout_s
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def classify(
        self,
        *,
        title: str,
        description: str,
        tags: list[str],
    ) -> str | None:
        user_prompt = (
            f"Título: {title or '(sem título)'}\n"
            f"Descrição: {description or '(sem descrição)'}\n"
            f"Tags: {_format_tags(tags)}\n\n"
            "Classifique esta faixa em um dos 6 moods canônicos."
        )

        try:
            response = self._client.with_options(timeout=self._timeout_s).messages.create(
                model=self._config.claude_model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                tools=[_CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_track_mood"},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            logger.warning(
                "Classificação Claude indisponível para '%s': %s. Faixa marcada como 'indefinido'.",
                title, exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001 — falhar resiliente conforme RF-09
            logger.warning(
                "Erro inesperado ao classificar '%s': %s. Faixa marcada como 'indefinido'.",
                title, exc,
            )
            return None

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "classify_track_mood":
                mood = (block.input or {}).get("mood", "")
                if isinstance(mood, str) and mood in CANONICAL_MOODS:
                    return mood
                logger.warning(
                    "Mood inválido retornado pelo Claude para '%s': %r. Faixa marcada como 'indefinido'.",
                    title, mood,
                )
                return None

        logger.warning(
            "Claude não emitiu tool_use para '%s'. Faixa marcada como 'indefinido'.",
            title,
        )
        return None
