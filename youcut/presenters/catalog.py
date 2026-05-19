"""Catálogo local de apresentadores: leitura simples da pasta.

Sem ``aliases.json`` — apresentadores são identificados por rosto,
não por menções textuais (essa é a diferença chave em relação ao
catálogo de :mod:`youcut.players`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from youcut.presenters.models import PresenterProfile

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _slug_to_display_name(slug: str) -> str:
    """``tiago_leifert`` → ``Tiago Leifert``."""
    return " ".join(t.capitalize() for t in slug.split("_"))


class PresenterCatalog:
    """Catálogo carregado em memória. Imutável após inicialização.

    Use :func:`load_catalog`.
    """

    def __init__(self, profiles: list[PresenterProfile]):
        self._profiles = {p.slug: p for p in profiles}

    @property
    def profiles(self) -> list[PresenterProfile]:
        return list(self._profiles.values())

    def get(self, slug: str) -> PresenterProfile | None:
        return self._profiles.get(slug.lower())

    def __len__(self) -> int:
        return len(self._profiles)


def load_catalog(presenters_dir: Path) -> PresenterCatalog:
    """Carrega o catálogo a partir de ``presenters_dir``.

    Sem aliases.json — manter simples. Se a pasta não existir,
    retorna catálogo vazio (feature transparente).
    """
    if not presenters_dir.exists() or not presenters_dir.is_dir():
        logger.debug("presenters_dir não existe: %s", presenters_dir)
        return PresenterCatalog([])

    profiles: list[PresenterProfile] = []
    for entry in sorted(presenters_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        slug = entry.stem.lower()
        profiles.append(
            PresenterProfile(
                slug=slug,
                image_path=entry,
                display_name=_slug_to_display_name(slug),
            )
        )

    if profiles:
        logger.info(
            "Catálogo de apresentadores carregado: %d profile(s) em %s",
            len(profiles),
            presenters_dir,
        )
    return PresenterCatalog(profiles)
