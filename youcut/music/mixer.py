"""Mixagem de trilha sonora em clipes sociais com padrão fixo de produto."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from youcut.models import MusicTrack

logger = logging.getLogger("youcut.music.mixer")


class MusicMixer:
    """Aplica trilha sonora a um clipe social com padrão fixo (RF-16 a RF-21).

    Regra única (não configurável por execução):
    - Skip dos primeiros 10s da música (RF-16);
    - Fade-in 1.0s e fade-out 1.2s (RF-17/18);
    - Voz original em volume integral (RF-19) — sidechain ducking abaixa a
      música automaticamente quando há voz, então o limitador raramente atua
      sobre o sinal vocal;
    - Música a 12% nos silêncios da voz (RF-20);
    - Limitador final em 0.97 como rede de segurança (RF-21).
    """

    def mix(self, clip_path: Path, track: MusicTrack) -> Path:
        """Aplica trilha ao clipe via ffmpeg. Retorna path do arquivo final.

        Em caso de falha do ffmpeg, loga ERROR e retorna `clip_path` original
        (mesmo comportamento do mixer anterior, sem trilha aplicada).
        """
        clip_dur = self._get_clip_duration(clip_path)
        filter_graph = self._build_filter_graph(clip_dur)

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
        except FileNotFoundError as exc:
            logger.error("ffmpeg não encontrado no PATH: %s", exc)
            tmp_path.unlink(missing_ok=True)
            return clip_path

        try:
            tmp_path.replace(clip_path)
        except OSError as exc:
            logger.error("Falha ao substituir clipe com versão mixada: %s", exc)
            tmp_path.unlink(missing_ok=True)
            return clip_path
        return clip_path

    def _build_filter_graph(self, clip_dur: float) -> str:
        """Filter graph fixo conforme techspec §Filter Graph (RF-16 a RF-21).

        A voz é duplicada (asplit): uma cópia entra no mix final em volume
        integral SEM passar por nenhum processador, a outra atua como gatilho
        de ducking sobre a música via sidechaincompress. O alimiter é aplicado
        apenas no ramo da música (rede de segurança contra picos da trilha) —
        nunca toca o sinal vocal, garantindo voz original 100% preservada.
        """
        fade_out_start = max(0.0, clip_dur - 1.2)
        return (
            "[0:a]asplit=2[voice][voice_sc];"
            "[1:a]"
            "atrim=start=10,asetpts=PTS-STARTPTS,"
            "apad,"
            f"atrim=0:{clip_dur},"
            "afade=t=in:st=0:d=1.0,"
            f"afade=t=out:st={fade_out_start}:d=1.2,"
            "volume=0.12"
            "[m];"
            "[m][voice_sc]"
            "sidechaincompress=threshold=0.05:ratio=8:attack=5:release=250,"
            "alimiter=limit=0.97"
            "[m_safe];"
            "[voice][m_safe]amix=inputs=2:duration=first:normalize=0"
            "[aout]"
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Não foi possível obter duração do clipe '%s': %s. Usando 60s como fallback.",
                clip_path.name, exc,
            )
            return 60.0
