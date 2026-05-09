"""Provedor determinístico de trilha sonora a partir do acervo local."""
from __future__ import annotations

import hashlib
import logging

from youcut.models import MusicTrack, ViralClip
from youcut.music.library import MusicLibrary

logger = logging.getLogger("youcut.music.provider")

DEFAULT_MOOD = "motivacional"

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


class YouTubeMusicProvider:
    """Escolhe a faixa do acervo local para um clipe social.

    `classify_mood` preserva 100% da heurística por keywords pt-BR usada pelo
    backend anterior (Pixabay/Jamendo). `pick_track` é determinístico via
    SHA-256 de (title + reason + social_visual_style), garantindo que re-render
    do mesmo clipe produza a mesma trilha enquanto o acervo não muda
    (RF-12/15).
    """

    def __init__(self, library: MusicLibrary) -> None:
        self._library = library

    def classify_mood(self, clip: ViralClip) -> str:
        text = " ".join([
            clip.title or "",
            clip.reason or "",
            clip.social_visual_style or "",
        ]).lower()

        scores: dict[str, int] = {mood: 0 for mood in _MOOD_KEYWORDS}
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

    def pick_track(self, clip: ViralClip) -> MusicTrack | None:
        """Escolhe uma faixa do acervo de forma determinística para `clip`.

        - Se o acervo está vazio, retorna `None` (RF-14).
        - Se há candidatas com `mood == clip mood`, escolhe entre elas (RF-11).
        - Se nenhuma faixa bate com o mood do clipe, faz fallback global em
          todas as faixas do acervo (RF-13).
        - Determinismo: SHA-256 de `title + "\\n" + reason + "\\n" +
          social_visual_style` % len(candidates), candidates ordenadas por
          `video_id` (RF-12/15).
        """
        if self._library.is_empty():
            return None

        mood = self.classify_mood(clip)
        candidates = self._library.candidates_for(mood)
        if not candidates:
            logger.info(
                "🎵 Sem faixas para mood='%s'. Aplicando fallback global (RF-13).",
                mood,
            )
            candidates = self._library.all_tracks()

        if not candidates:
            return None

        candidates = sorted(candidates, key=lambda t: t.video_id)
        seed = "\n".join([
            clip.title or "",
            clip.reason or "",
            clip.social_visual_style or "",
        ])
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        index = int.from_bytes(digest, "big") % len(candidates)
        chosen = candidates[index]
        logger.info("🎵 Trilha do clipe: %s (mood=%s)", chosen.name, chosen.mood)
        return chosen
