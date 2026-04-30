"""Persistência de `MotionComicSession` em ``~/.youcut/sessions/``.

Reusa o diretório criado por :mod:`youcut.session_store`. Sessões legadas
(`SessionData`) e sessões de motion comic (`MotionComicSession`) coexistem
no mesmo diretório; o discriminador é a presença do campo ``cast`` no JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from youcut.models import MotionComicSession
from youcut.session_store import _ensure_sessions_dir

logger = logging.getLogger(__name__)


def save_motion_comic_session(session: MotionComicSession) -> Path:
    """Persiste a sessão em ``~/.youcut/sessions/<id>.json`` e retorna o path."""

    path = _ensure_sessions_dir() / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(), encoding="utf-8")
    logger.info("comic.session: salvou %s", path.name)
    return path


def load_motion_comic_session(session_id: str) -> MotionComicSession:
    """Carrega uma sessão de motion comic por ``session_id``."""

    path = _ensure_sessions_dir() / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Sessão de motion comic não encontrada: {session_id}")
    return MotionComicSession.model_validate_json(path.read_text(encoding="utf-8"))


def _is_motion_comic_payload(raw: str) -> bool:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    return "cast" in data and "panels" in data


def list_motion_comic_sessions() -> list[MotionComicSession]:
    """Lista todas as sessões de motion comic, ignorando sessões legadas."""

    sessions_path = _ensure_sessions_dir()
    sessions: list[MotionComicSession] = []
    for f in sorted(sessions_path.glob("*.json")):
        try:
            raw = f.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("comic.session: falha ao ler %s (%s) — ignorado", f.name, exc)
            continue
        if not _is_motion_comic_payload(raw):
            continue
        try:
            sessions.append(MotionComicSession.model_validate_json(raw))
        except ValidationError as exc:
            logger.warning(
                "comic.session: arquivo %s parece motion comic mas é inválido (%s)",
                f.name,
                exc,
            )
    return sorted(sessions, key=lambda s: s.created_at, reverse=True)
