"""Modelos do catálogo de apresentadores."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class PresenterProfile(BaseModel):
    """Ficha de um apresentador no catálogo local.

    ``slug`` = nome do arquivo sem extensão (ex.: ``tiago_leifert``).
    ``display_name`` = forma "humana" usada nos prompts do Claude vision
    (ex.: ``Tiago Leifert``). Diferente de :class:`PlayerProfile`, não
    há aliases — apresentadores são poucos e identificados por rosto,
    não por menção textual.
    """

    slug: str
    image_path: Path
    display_name: str

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.slug)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, PresenterProfile):
            return self.slug == other.slug
        return NotImplemented


class PresenterDetection(BaseModel):
    """Resultado da detecção de apresentadores no vídeo.

    ``profiles`` é a lista deduplicada de apresentadores que o Claude
    vision identificou nos frames amostrados. ``source_method`` indica
    o caminho: ``"vision"`` quando bem-sucedido, ``"manual"`` quando
    veio da flag ``--presenter``, ``"fallback"`` quando o detector
    falhou e retornou os primeiros N do catálogo como suporte mínimo.
    """

    profiles: list[PresenterProfile]
    source_method: str
