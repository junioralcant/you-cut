"""Provedor de músicas CC via Jamendo para trilha sonora automática."""
from __future__ import annotations

import json
import logging
import random
import subprocess
import uuid
from pathlib import Path

import httpx

from youcut.config import PipelineConfig
from youcut.models import MusicTrack, ViralClip

logger = logging.getLogger(__name__)

MOOD_MAP: dict[str, str] = {
    "motivacional": "motivational",
    "reflexivo": "relaxing",
    "energico": "upbeat",
    "emocional": "emotional",
    "feliz": "happy",
    "dramatico": "dramatic",
}

_MOOD_KEYWORDS: dict[str, list[str]] = {
    "motivacional": [
        "motivação", "motivacional", "inspire", "inspirador", "conquista",
        "superação", "determinação", "força", "poder", "crescimento",
    ],
    "reflexivo": [
        "reflexão", "reflexivo", "pensamento", "calma", "paz", "tranquilo",
        "suave", "suavemente", "meditação", "consciência",
    ],
    "energico": [
        "energia", "energico", "enérgico", "dinâmico", "acelerado", "rápido",
        "intenso", "ação", "agitado", "esporte", "treino", "workout",
    ],
    "emocional": [
        "emoção", "emocional", "sentimento", "coração", "amor", "saudade",
        "melancolia", "nostálgico", "profundo",
    ],
    "feliz": [
        "feliz", "felicidade", "alegria", "diversão", "animado", "positivo",
        "comemorando", "celebração", "sorriso",
    ],
    "dramatico": [
        "drama", "dramático", "tenso", "tensão", "suspense", "conflito",
        "impacto", "impactante", "chocante",
    ],
}

# Tags Jamendo por mood (tentadas em ordem até encontrar resultado)
_MOOD_TAGS: dict[str, list[str]] = {
    "motivacional": ["motivational", "uplifting", "inspiring", "positive"],
    "reflexivo": ["relaxing", "calm", "ambient", "peaceful"],
    "energico": ["energetic", "upbeat", "electronic", "dance"],
    "emocional": ["emotional", "sentimental", "melancholic"],
    "feliz": ["happy", "joyful", "fun", "cheerful"],
    "dramatico": ["dramatic", "epic", "cinematic", "intense"],
}

_MUSIC_CACHE_DIR = Path.home() / ".youcut" / "music"
_JAMENDO_API_URL = "https://api.jamendo.com/v3.0/tracks/"

DEFAULT_MOOD = "motivacional"


def _probe_duration(path: Path) -> float:
    """Retorna duração em segundos via ffprobe; 0.0 se falhar."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


class PixabayMusicProvider:
    """Classifica mood e busca/mantém cache de faixas CC via Jamendo API."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def classify_mood(self, clip: ViralClip) -> str:
        """Deriva mood a partir de title + reason + social_visual_style.

        Retorna chave pt-BR de MOOD_MAP (ex: "motivacional").
        """
        text = " ".join([
            clip.title or "",
            clip.reason or "",
            clip.social_visual_style or "",
        ]).lower()

        scores: dict[str, int] = {mood: 0 for mood in MOOD_MAP}
        for mood, keywords in _MOOD_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[mood] += 1

        best = max(scores, key=lambda m: scores[m])
        if scores[best] == 0:
            logger.debug("Nenhum keyword de mood encontrado; usando default '%s'", DEFAULT_MOOD)
            return DEFAULT_MOOD

        logger.info("🎵 Mood detectado: %s", best)
        return best

    def fetch_track(self, mood: str, duration_s: float) -> MusicTrack | None:
        """Retorna faixa do cache local ou baixa da Jamendo.

        Retorna None em caso de fallback sem lançar exceção.
        """
        if not self._config.jamendo_client_id:
            logger.warning("⚠️  JAMENDO_CLIENT_ID não configurado. Seguindo sem trilha.")
            return None

        cached = self._get_from_cache(mood)
        if cached is not None:
            logger.debug("Cache hit: %s", cached.local_path)
            return cached

        return self._download_from_api(mood, duration_s)

    def _get_from_cache(self, mood: str) -> MusicTrack | None:
        cache_dir = _MUSIC_CACHE_DIR / mood
        if not cache_dir.exists():
            return None

        mp3_files = list(cache_dir.glob("*.mp3"))
        if not mp3_files:
            return None

        chosen = random.choice(mp3_files)
        sidecar = chosen.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                return MusicTrack(
                    name=meta.get("name", chosen.stem),
                    pixabay_url=meta.get("pixabay_url", ""),
                    local_path=chosen,
                    mood=mood,
                    duration_s=float(meta.get("duration_s", 0.0)),
                )
            except Exception:
                pass

        return MusicTrack(
            name=chosen.stem,
            pixabay_url="",
            local_path=chosen,
            mood=mood,
            duration_s=_probe_duration(chosen),
        )

    def _download_from_api(self, mood: str, duration_s: float) -> MusicTrack | None:
        tags_list = _MOOD_TAGS.get(mood, ["motivational"])

        for tag in tags_list:
            result = self._try_fetch(tag, mood, duration_s)
            if result is not None:
                return result

        # Último recurso: sem filtro de tag
        return self._try_fetch(None, mood, duration_s)

    def _try_fetch(self, tag: str | None, mood: str, duration_s: float) -> MusicTrack | None:
        params: dict = {
            "client_id": self._config.jamendo_client_id,
            "format": "json",
            "limit": 10,
            "audioformat": "mp32",
            "order": "popularity_total",
            "boost": "popularity_total",
        }
        if tag:
            params["fuzzytags"] = tag

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(_JAMENDO_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("⚠️  Falha ao consultar Jamendo API: %s. Seguindo sem trilha.", exc)
            return None

        results = data.get("results", [])
        if not results:
            if tag:
                logger.debug("Nenhuma faixa Jamendo para tag '%s'. Tentando próxima.", tag)
            else:
                logger.warning("⚠️  Nenhuma música encontrada para o mood '%s'. Clipe gerado sem trilha.", mood)
            return None

        track_data = self._select_track(results, duration_s)
        if track_data is None:
            return None

        mp3_url: str = track_data.get("audio", "") or track_data.get("audiodownload", "")
        if not mp3_url:
            logger.warning("⚠️  Faixa sem URL de MP3. Seguindo sem trilha.")
            return None

        local_path = self._download_mp3(mp3_url, mood)
        if local_path is None:
            return None

        track_duration = float(track_data.get("duration", 0))
        track = MusicTrack(
            name=track_data.get("name", "unknown"),
            pixabay_url=track_data.get("shareurl", ""),
            local_path=local_path,
            mood=mood,
            duration_s=track_duration,
        )

        sidecar = local_path.with_suffix(".json")
        try:
            sidecar.write_text(
                json.dumps({
                    "name": track.name,
                    "pixabay_url": track.pixabay_url,
                    "duration_s": track_duration,
                }),
                encoding="utf-8",
            )
        except Exception:
            pass

        logger.info("🎵 Trilha: \"%s\" — Jamendo", track.name)
        return track

    def _select_track(self, results: list[dict], duration_s: float) -> dict | None:
        min_dur = duration_s * 0.5
        max_dur = duration_s * 3.0

        for hit in results:
            hit_dur = float(hit.get("duration", 0))
            if min_dur <= hit_dur <= max_dur:
                return hit

        return results[0] if results else None

    def _download_mp3(self, url: str, mood: str) -> Path | None:
        cache_dir = _MUSIC_CACHE_DIR / mood
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / f"{uuid.uuid4()}.mp3"

        try:
            with httpx.Client(timeout=30.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)
        except Exception as exc:
            logger.warning("⚠️  Falha ao baixar MP3: %s. Seguindo sem trilha.", exc)
            if dest.exists():
                dest.unlink()
            return None

        return dest
