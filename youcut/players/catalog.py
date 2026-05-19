"""Catálogo local de jogadores: leitura da pasta + inferência de aliases.

Cada arquivo de imagem em ``players_dir`` vira um :class:`PlayerProfile`.
O ``slug`` é o nome do arquivo (sem extensão). Aliases são inferidos do
slug e do display name; se existir um ``aliases.json`` na pasta com
overrides explícitos, ele tem prioridade.

Estrutura esperada do ``aliases.json`` (opcional)::

    {
      "vinicius_junior": ["vini", "vini jr", "vinicius"],
      "neymar": ["ney", "ney jr"],
      "danilo_botafogo": ["danilo do botafogo"],
      "danilo_flamengo": ["danilo do flamengo"]
    }
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from youcut.players.models import PlayerProfile

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_ALIASES_FILENAME = "aliases.json"


def _normalize(text: str) -> str:
    """Lowercase + remove acentos + colapsa whitespace.

    Estável para comparação com palavras da transcrição (que também são
    normalizadas via :func:`youcut.players.detector._normalize_word`).
    """
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _slug_to_display_name(slug: str) -> str:
    """``vinicius_junior`` → ``Vinicius Junior``.

    Sufixos descritivos como ``_botafogo`` / ``_flamengo`` ficam entre
    parênteses pra desambiguação visual: ``Danilo (Botafogo)``.
    """
    tokens = slug.split("_")
    if len(tokens) >= 2 and tokens[-1].lower() in _CLUB_HINTS:
        base = " ".join(t.capitalize() for t in tokens[:-1])
        club = tokens[-1].capitalize()
        return f"{base} ({club})"
    return " ".join(t.capitalize() for t in tokens)


_CLUB_HINTS = {
    "flamengo", "botafogo", "fluminense", "vasco", "palmeiras", "corinthians",
    "santos", "saopaulo", "gremio", "internacional", "atletico", "cruzeiro",
}


def _infer_aliases_from_slug(slug: str) -> list[str]:
    """Aliases default a partir do slug.

    Para ``vinicius_junior`` retorna ``["vinicius junior", "vinicius"]``.
    Para slugs com sufixo de clube (``danilo_botafogo``), o sufixo é
    removido — só sobra ``danilo`` (a desambiguação fica por conta do
    ``disambiguator``).
    """
    tokens = slug.split("_")
    if len(tokens) >= 2 and tokens[-1].lower() in _CLUB_HINTS:
        tokens = tokens[:-1]
    base = " ".join(tokens)
    aliases = {base}
    if len(tokens) >= 2:
        aliases.add(tokens[0])  # primeiro nome isolado
    return sorted(aliases, key=lambda x: (-len(x), x))


class PlayerCatalog:
    """Catálogo carregado em memória. Imutável após inicialização.

    Use :func:`load_catalog` em vez de instanciar diretamente.
    """

    def __init__(self, profiles: list[PlayerProfile]):
        self._profiles = {p.slug: p for p in profiles}
        index: dict[str, list[PlayerProfile]] = {}
        for profile in profiles:
            for alias in profile.aliases:
                key = _normalize(alias)
                if not key:
                    continue
                index.setdefault(key, []).append(profile)
        self._alias_index = index
        self._max_alias_tokens = max(
            (len(key.split()) for key in index),
            default=1,
        )

    @property
    def profiles(self) -> list[PlayerProfile]:
        return list(self._profiles.values())

    @property
    def alias_index(self) -> dict[str, list[PlayerProfile]]:
        """Map de alias normalizado → lista de profiles que casam.

        Múltiplos profiles podem ter o mesmo alias (ex.: dois "danilo");
        o detector trata como ambíguo e passa para o disambiguator.
        """
        return self._alias_index

    @property
    def max_alias_tokens(self) -> int:
        """Maior número de tokens de qualquer alias.

        Usado pelo detector para limitar a janela de lookahead ao varrer
        a transcrição (não precisa testar n-gramas maiores que isso).
        """
        return self._max_alias_tokens

    def get(self, slug: str) -> PlayerProfile | None:
        return self._profiles.get(slug)


def load_catalog(players_dir: Path) -> PlayerCatalog:
    """Carrega o catálogo a partir de ``players_dir``.

    Se a pasta não existir ou estiver vazia, retorna um catálogo vazio
    (caso normal pra projetos que não usam a feature). Logs em nível
    ``INFO`` quando descobre profiles, ``DEBUG`` em cada inferência.
    """
    if not players_dir.exists() or not players_dir.is_dir():
        logger.debug("players_dir não existe: %s", players_dir)
        return PlayerCatalog([])

    overrides: dict[str, list[str]] = {}
    aliases_path = players_dir / _ALIASES_FILENAME
    if aliases_path.exists():
        try:
            raw = json.loads(aliases_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                overrides = {
                    str(k): [str(a) for a in v]
                    for k, v in raw.items()
                    if isinstance(v, list)
                }
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Falha ao ler %s: %s — ignorando overrides", aliases_path, exc)

    profiles: list[PlayerProfile] = []
    for entry in sorted(players_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        slug = entry.stem.lower()
        display_name = _slug_to_display_name(slug)
        aliases = overrides.get(slug) or _infer_aliases_from_slug(slug)
        # Sempre inclui o display_name normalizado como alias adicional
        # (sem acentos) para garantir match mesmo quando o usuário só
        # configurou apelidos no overrides.
        aliases = list({*aliases, _normalize(display_name.split(" (")[0])})
        profiles.append(
            PlayerProfile(
                slug=slug,
                image_path=entry,
                display_name=display_name,
                aliases=aliases,
            )
        )

    if profiles:
        logger.info(
            "Catálogo de jogadores carregado: %d profile(s) em %s",
            len(profiles),
            players_dir,
        )
    return PlayerCatalog(profiles)
