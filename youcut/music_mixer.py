"""Mixagem de trilha sonora em clipes sociais via ffmpeg com duck automático."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import MusicTrack

logger = logging.getLogger(__name__)


class MusicMixer:
    """Aplica uma faixa musical a um clipe MP4 com duck automático na fala."""

    def mix(self, clip_path: Path, track: MusicTrack, config: PipelineConfig) -> Path:
        """Aplica trilha ao clipe via ffmpeg. Retorna path do arquivo final.

        Se ffmpeg falhar, loga o erro e retorna clip_path original sem música.
        """
        dur = self._get_clip_duration(clip_path)
        filter_graph = self._build_filter_graph(dur, config)

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            dir=clip_path.parent,
            delete=False,
        ) as tmp_f:
            tmp_path = Path(tmp_f.name)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-i", str(track.local_path),
            "-filter_complex", filter_graph,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            str(tmp_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Falha ao mixar trilha sonora: %s\nstderr: %s",
                exc,
                exc.stderr.decode(errors="replace") if exc.stderr else "",
            )
            tmp_path.unlink(missing_ok=True)
            return clip_path

        # Substituir arquivo original pela versão com trilha (atomico no mesmo fs)
        try:
            tmp_path.replace(clip_path)
        except OSError as e:
            logger.error("Falha ao substituir clipe com versão mixada: %s", e)
            tmp_path.unlink(missing_ok=True)
            return clip_path
        return clip_path

    def _build_filter_graph(self, clip_dur: float, config: PipelineConfig) -> str:
        fade_start = max(0.0, clip_dur - 2.0)
        # Música em volume fixo sobre o áudio original sem alterar a voz
        return (
            f"[1:a]aloop=loop=-1:size=2000000000,"
            f"atrim=0:{clip_dur},"
            f"afade=t=out:st={fade_start}:d=2,"
            f"volume={config.music_volume}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:normalize=0[aout]"
        )

    def _get_clip_duration(self, clip_path: Path) -> float:
        """Retorna duração do clipe em segundos via ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception as exc:
            logger.warning(
                "Não foi possível obter duração do clipe '%s': %s. Usando 60s como fallback.",
                clip_path.name, exc,
            )
            return 60.0
