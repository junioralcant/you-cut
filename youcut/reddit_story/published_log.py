"""Registro permanente de vídeos já gerados/publicados pelo pipeline
``youcut reddit-story``. Vive em ``~/.youcut/published_videos.json``.

Função principal: **dedup por `reddit_thread_id`** — antes de processar uma
nova URL, o pipeline checa se aquele thread já virou vídeo. Se sim, aborta
(evita queimar ~$0.63 reprocessando) a menos que `--force` seja passado.

Estados possíveis (`status`):
- ``generated`` — vídeo + thumb + metadata existem em disco mas NÃO foi upload
- ``uploaded`` — publicado no YouTube (URL guardada em ``youtube_url``)

Atomic write via tempfile + rename pra evitar log corrompido em crash.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


LOG_PATH = Path.home() / ".youcut" / "published_videos.json"


Status = Literal["generated", "uploaded"]


@dataclass
class PublishedEntry:
    """1 vídeo registrado. Mantenha campos opcionais por último pra evolução."""

    session_id: str
    reddit_thread_id: str
    reddit_url: str
    subreddit: str
    title: str
    video_path: str
    channel: str = "ThreadCourt"
    status: Status = "generated"
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    uploaded_at: str | None = None
    youtube_url: str | None = None


class PublishedLog:
    """Read/write wrapper sobre ``published_videos.json``."""

    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "videos": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Backup do corrompido + começa do zero (não bloqueia o pipeline)
            backup = self.path.with_suffix(".corrupt.bak")
            self.path.rename(backup)
            return {"schema_version": 1, "videos": []}
        data.setdefault("schema_version", 1)
        data.setdefault("videos", [])
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def is_published(self, reddit_thread_id: str) -> PublishedEntry | None:
        """Retorna a entry se já existe vídeo pra esse thread, senão None.

        Match é exato pelo ID do thread (3-7 chars depois de /comments/).
        """
        for raw in self._data["videos"]:
            if raw.get("reddit_thread_id") == reddit_thread_id:
                # Filtra campos extras pra ser tolerante a evolução de schema
                fields = {
                    k: v for k, v in raw.items()
                    if k in PublishedEntry.__dataclass_fields__
                }
                return PublishedEntry(**fields)
        return None

    def register(self, entry: PublishedEntry) -> None:
        """Adiciona uma nova entry e persiste imediatamente."""
        self._data["videos"].append(asdict(entry))
        self.save()

    def mark_uploaded(
        self,
        session_id: str,
        *,
        youtube_url: str | None = None,
    ) -> PublishedEntry | None:
        """Atualiza status=uploaded + uploaded_at + youtube_url. Retorna entry
        atualizada ou None se session_id não existe."""
        for raw in self._data["videos"]:
            if raw.get("session_id") == session_id:
                raw["status"] = "uploaded"
                raw["uploaded_at"] = datetime.utcnow().isoformat()
                if youtube_url:
                    raw["youtube_url"] = youtube_url
                self.save()
                fields = {
                    k: v for k, v in raw.items()
                    if k in PublishedEntry.__dataclass_fields__
                }
                return PublishedEntry(**fields)
        return None

    def list_entries(self) -> list[PublishedEntry]:
        out: list[PublishedEntry] = []
        for raw in self._data["videos"]:
            fields = {
                k: v for k, v in raw.items()
                if k in PublishedEntry.__dataclass_fields__
            }
            out.append(PublishedEntry(**fields))
        return out
