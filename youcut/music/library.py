"""Persistência do acervo local de faixas sincronizadas a partir da playlist YouTube."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from youcut.models import MusicTrack

logger = logging.getLogger("youcut.music.library")

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path.home() / ".youcut" / "music"
TRACKS_SUBDIR = "tracks"
INDEX_FILENAME = "index.json"


class MusicLibrary:
    """Lê e escreve o índice JSON de faixas locais (`~/.youcut/music/index.json`).

    Estrutura persistida segue `schema_version=1` (ver techspec): mapa de
    `video_id` → metadata (`name`, `source_url`, `file`, `mood`, `duration_s`,
    `synced_at`). A escrita é atômica (arquivo temporário + `os.replace`) e
    `add()` é idempotente: chamar com a mesma `video_id` 2× não duplica.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_ROOT
        self.index_path = self.root / INDEX_FILENAME
        self.tracks_dir = self.root / TRACKS_SUBDIR
        self._tracks: dict[str, dict] = {}
        self._playlist_url: str = ""
        self._loaded = False

    @property
    def playlist_url(self) -> str:
        return self._playlist_url

    def set_playlist_url(self, url: str) -> None:
        if not self._loaded:
            self.load()
        self._playlist_url = url

    def load(self) -> None:
        self._loaded = True
        if not self.index_path.exists():
            self._tracks = {}
            self._playlist_url = ""
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Falha ao ler '%s': %s. Tratando acervo como vazio.",
                self.index_path, exc,
            )
            self._tracks = {}
            self._playlist_url = ""
            return

        if not isinstance(data, dict):
            self._tracks = {}
            self._playlist_url = ""
            return

        self._playlist_url = str(data.get("playlist_url") or "")
        raw_tracks = data.get("tracks") or {}
        self._tracks = dict(raw_tracks) if isinstance(raw_tracks, dict) else {}

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "playlist_url": self._playlist_url,
            "tracks": self._tracks,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.root,
            delete=False,
            prefix=f".{INDEX_FILENAME}.",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.index_path)

    def has(self, video_id: str) -> bool:
        if not self._loaded:
            self.load()
        return video_id in self._tracks

    def add(self, entry: MusicTrack) -> None:
        if not self._loaded:
            self.load()
        try:
            relative_file = entry.local_path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            relative_file = str(entry.local_path)
        self._tracks[entry.video_id] = {
            "name": entry.name,
            "source_url": entry.source_url,
            "file": relative_file,
            "mood": entry.mood,
            "duration_s": float(entry.duration_s),
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def candidates_for(self, mood: str) -> list[MusicTrack]:
        if not self._loaded:
            self.load()
        return [
            self._entry_to_track(vid, rec)
            for vid, rec in sorted(self._tracks.items())
            if rec.get("mood") == mood
        ]

    def all_tracks(self) -> list[MusicTrack]:
        if not self._loaded:
            self.load()
        return [self._entry_to_track(vid, rec) for vid, rec in sorted(self._tracks.items())]

    def is_empty(self) -> bool:
        if not self.index_path.exists():
            return True
        if not self._loaded:
            self.load()
        return len(self._tracks) == 0

    def _entry_to_track(self, video_id: str, record: dict) -> MusicTrack:
        file_value = record.get("file", "")
        if file_value:
            file_path = Path(file_value)
            local_path = file_path if file_path.is_absolute() else self.root / file_path
        else:
            local_path = self.root
        return MusicTrack(
            video_id=video_id,
            name=str(record.get("name", "")),
            source_url=str(record.get("source_url", "")),
            local_path=local_path,
            mood=str(record.get("mood", "indefinido")),
            duration_s=float(record.get("duration_s", 0.0)),
        )
