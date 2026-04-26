import logging
from pathlib import Path

from youcut.models import SessionData

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path.home() / ".youcut" / "sessions"


def _ensure_sessions_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def save_session(session: SessionData) -> Path:
    path = _ensure_sessions_dir() / f"{session.session_id}.json"
    path.write_text(session.model_dump_json(), encoding="utf-8")
    return path


def load_session(session_id: str) -> SessionData:
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return SessionData.model_validate_json(path.read_text(encoding="utf-8"))


def list_sessions() -> list[SessionData]:
    sessions_path = _ensure_sessions_dir()
    sessions = []
    for f in sessions_path.glob("*.json"):
        try:
            sessions.append(SessionData.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Corrupt session file ignored: %s", f.name)
    return sorted(sessions, key=lambda s: s.created_at, reverse=True)
