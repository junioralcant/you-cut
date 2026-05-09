"""Sincronização sob comando: enumera playlist YouTube, baixa novas faixas e classifica mood."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from youcut.models import MusicTrack, SyncReport
from youcut.music.classifier import TrackMoodClassifier
from youcut.music.library import MusicLibrary
from youcut.yt_dlp_auth import (
    YtDlpAuthConfig,
    apply_yt_dlp_auth,
    resolve_yt_dlp_auth_config,
)

logger = logging.getLogger("youcut.music.sync")

_INDEFINIDO_MOOD = "indefinido"


class PlaylistSyncer:
    """Sincroniza uma playlist do YouTube com o acervo local de músicas.

    Idempotente por `video_id` (RF-04): faixas já presentes no índice nunca são
    re-baixadas nem re-classificadas. Falhas individuais (download ou classificação)
    incrementam `failed_tracks` e a sync segue (RF-05).
    """

    def __init__(
        self,
        lib: MusicLibrary,
        classifier: TrackMoodClassifier,
        *,
        audio_format: str = "m4a",
        auth_config: YtDlpAuthConfig | None = None,
    ) -> None:
        self._lib = lib
        self._classifier = classifier
        self._audio_format = audio_format
        if auth_config is None:
            try:
                auth_config = resolve_yt_dlp_auth_config()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Não foi possível resolver auth do yt-dlp: %s", exc)
                auth_config = None
        self._auth_config = auth_config

    def sync(
        self,
        playlist_url: str,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> SyncReport:
        logger.info("🎵 Sync iniciado: %s", playlist_url)
        self._lib.load()
        self._lib.set_playlist_url(playlist_url)
        self._lib.tracks_dir.mkdir(parents=True, exist_ok=True)

        report = SyncReport()

        try:
            entries = self._enumerate_playlist(playlist_url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao enumerar playlist '%s': %s", playlist_url, exc)
            return report

        for entry in entries:
            video_id = self._extract_video_id(entry)
            if not video_id:
                report.failed_tracks += 1
                report.failed_details.append(("(sem id)", "entrada sem video_id"))
                continue

            if self._lib.has(video_id):
                report.cached_tracks += 1
                if on_progress:
                    on_progress(f"cache: {video_id}")
                continue

            try:
                track = self._process_new_entry(video_id, entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Falha ao processar faixa nova '%s': %s. Contabilizando como falha.",
                    video_id, exc,
                )
                report.failed_tracks += 1
                report.failed_details.append((video_id, str(exc)))
                continue

            if track is None:
                # Falha de download já contou em _process_new_entry → mas como ele
                # propaga via exception, aqui é apenas guarda defensiva.
                report.failed_tracks += 1
                report.failed_details.append((video_id, "falha desconhecida"))
                continue

            self._lib.add(track)
            report.new_tracks += 1
            logger.info("🎵 Faixa nova: %s → mood=%s", track.name, track.mood)
            if on_progress:
                on_progress(f"novo: {track.name}")

        try:
            self._lib.save()
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao persistir índice local: %s", exc)

        logger.info(
            "🎵 Sync concluído: novas=%d cache=%d falhas=%d",
            report.new_tracks, report.cached_tracks, report.failed_tracks,
        )
        return report

    def _enumerate_playlist(self, playlist_url: str) -> list[dict]:
        ydl_opts = apply_yt_dlp_auth(
            {
                "extract_flat": "in_playlist",
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
            },
            self._auth_config,
        )
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if not info:
            return []
        if isinstance(info, dict) and "entries" in info:
            return [e for e in (info.get("entries") or []) if e]
        if isinstance(info, dict):
            return [info]
        return []

    def _extract_video_id(self, entry: dict) -> str:
        if not isinstance(entry, dict):
            return ""
        return str(entry.get("id") or entry.get("video_id") or "").strip()

    def _process_new_entry(self, video_id: str, entry: dict) -> Optional[MusicTrack]:
        local_path = self._download_audio(video_id)
        meta = self._fetch_metadata(video_id, entry)

        title = str(meta.get("title") or entry.get("title") or video_id)
        description = str(meta.get("description") or "")
        tags = list(meta.get("tags") or [])
        duration_s = float(meta.get("duration") or 0.0)

        mood = self._classifier.classify(
            title=title,
            description=description,
            tags=tags,
        )
        mood_value = mood if mood else _INDEFINIDO_MOOD

        return MusicTrack(
            video_id=video_id,
            name=title,
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            local_path=local_path,
            mood=mood_value,
            duration_s=duration_s,
        )

    def _download_audio(self, video_id: str) -> Path:
        out_template = str(self._lib.tracks_dir / f"{video_id}.%(ext)s")
        ydl_opts = apply_yt_dlp_auth(
            {
                "format": "bestaudio/best",
                "outtmpl": out_template,
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": self._audio_format,
                    }
                ],
            },
            self._auth_config,
        )
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        expected = self._lib.tracks_dir / f"{video_id}.{self._audio_format}"
        if expected.exists():
            return expected

        # fallback: o postprocessor pode escolher uma extensão alternativa
        for candidate in self._lib.tracks_dir.glob(f"{video_id}.*"):
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Download finalizou mas arquivo de áudio não encontrado para '{video_id}'."
        )

    def _fetch_metadata(self, video_id: str, fallback_entry: dict) -> dict:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = apply_yt_dlp_auth(
            {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
            },
            self._auth_config,
        )
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falha ao buscar metadata de '%s': %s. Usando fallback.", video_id, exc)
            info = {}

        result: dict = {
            "title": info.get("title") or fallback_entry.get("title"),
            "description": info.get("description") or fallback_entry.get("description"),
            "tags": info.get("tags") or fallback_entry.get("tags") or [],
            "duration": info.get("duration") or fallback_entry.get("duration"),
        }
        return result
