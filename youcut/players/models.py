"""Modelos do catálogo de jogadores."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PlayerProfile(BaseModel):
    """Ficha de um jogador no catálogo local.

    O ``slug`` é o nome do arquivo sem extensão (ex.: ``vinicius_junior``).
    ``aliases`` é a lista normalizada de formas que devem casar com o
    transcript (lowercase, sem acentos).
    """

    slug: str
    image_path: Path
    display_name: str
    aliases: list[str] = Field(default_factory=list)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.slug)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, PlayerProfile):
            return self.slug == other.slug
        return NotImplemented


class PlayerMention(BaseModel):
    """Menção detectada na transcrição de um clipe.

    ``alias_hit`` é a forma normalizada que casou (debug/log). ``start``/``end``
    são os timestamps absolutos da palavra (ou primeira palavra do alias
    composto) dentro do vídeo. Múltiplas menções do mesmo jogador num clipe
    podem aparecer; o consumer faz dedupe por ``profile.slug``.
    """

    profile: PlayerProfile
    alias_hit: str
    start: float
    end: float
