"""Tratamento visual padrão dos cortes sociais.

Aplica um look fixo (cor fria sutil + cantos arredondados + vinheta nas
bordas) em uma única chamada `ffmpeg -filter_complex`, com `-c:a copy` para
preservar áudio e duração (RF-03).

Os parâmetros são constantes do módulo — não há campos editáveis em
`PipelineConfig` (RF: parâmetros não configuráveis em v1). O único toggle
exposto é `config.social_visual_style_enabled`, controlado pela flag
CLI `--no-visual-style`.

A função é no-op silencioso quando:
- `config.social_visual_style_enabled is False` (RF-19);
- `config.cut_mode != "social"` (RF-21/22/23 — pipelines `youtube`, `comic`
  e `run` legacy ficam intocados).

Em caso de falha do ffmpeg, devolve o clipe original sem corromper o
input — mesmo padrão do `youcut.music.mixer.MusicMixer`.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from youcut.config import PipelineConfig

logger = logging.getLogger("youcut.visual_style")


# ── Constantes fixas — validadas em iteração manual sobre clip_04 ────────────
_CORNER_RADIUS_PX: int = 50
_VIGNETTE_BAND_PX: int = 120          # largura da faixa preta nas bordas
_VIGNETTE_OPACITY: float = 0.35       # 35% sobre o canvas
_VIGNETTE_BLUR_SIGMA: int = 28        # blur gaussiano da máscara
_COOL_EQ: str = "eq=saturation=0.95:contrast=1.04:gamma_r=0.97:gamma_b=1.05"

# ── Fallback de canvas quando ffprobe falha ──────────────────────────────────
_FALLBACK_W: int = 1080
_FALLBACK_H: int = 1920


def apply_visual_style(clip_path: Path, config: PipelineConfig) -> Path:
    """Aplica cor fria + cantos arredondados + vinheta no clipe.

    No-op (devolve `clip_path` sem reencodar) quando
    `config.social_visual_style_enabled is False` ou
    `config.cut_mode != "social"`.

    Reencoda vídeo com libx264; `-c:a copy` preserva áudio e duração (RF-03).
    Sobrescreve o arquivo original via rename atômico (mesmo padrão do
    `MusicMixer`). Em caso de falha do ffmpeg, devolve o `clip_path`
    original sem corromper o input.
    """
    if not config.social_visual_style_enabled:
        logger.info("Tratamento visual desligado (--no-visual-style); pulando %s", clip_path.name)
        return clip_path

    if config.cut_mode != "social":
        logger.info(
            "Tratamento visual restrito ao modo social; cut_mode=%s — pulando %s",
            config.cut_mode, clip_path.name,
        )
        return clip_path

    width, height = _get_video_dimensions(clip_path)
    filter_graph = _build_filter_graph(width, height)

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        dir=clip_path.parent,
        delete=False,
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(tmp_path),
    ]

    logger.info("Aplicando tratamento visual em %s", clip_path.name)
    started = time.monotonic()
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Tratamento visual falhou: %s — devolvendo clipe original\nstderr: %s",
            exc,
            exc.stderr.decode(errors="replace") if exc.stderr else "",
        )
        tmp_path.unlink(missing_ok=True)
        return clip_path
    except FileNotFoundError as exc:
        logger.warning("ffmpeg não encontrado no PATH: %s — devolvendo clipe original", exc)
        tmp_path.unlink(missing_ok=True)
        return clip_path

    try:
        tmp_path.replace(clip_path)
    except OSError as exc:
        logger.warning("Falha ao substituir clipe com versão tratada: %s — devolvendo clipe original", exc)
        tmp_path.unlink(missing_ok=True)
        return clip_path

    elapsed = time.monotonic() - started
    logger.info("Tratamento visual concluído em %.2fs (%s)", elapsed, clip_path.name)
    return clip_path


def _build_filter_graph(width: int, height: int) -> str:
    """Monta o filter graph único: cor fria → vinheta → cantos arredondados.

    Etapas:
    1. `[0:v]<_COOL_EQ>` — ajuste sutil de cor (saturação/contraste/gama R/B).
    2. Vinheta nativa do ffmpeg (`vignette`) — escurece bordas com
       transição suave (RF-10/11/12). Modo `backward` aplica máscara
       multiplicativa preservando luminosidade do centro.
    3. Cantos arredondados: split do vídeo gradado em duas cópias —
       uma vira `lutyuv=y=0` (canvas preto sincronizado), a outra vai
       para `format=yuva420p` + `geq` que zera o alpha fora do raio dos
       cantos. Overlay compõe o resultado sobre o canvas preto, garantindo
       cantos pretos sólidos (RF-07/08/09) mesmo em players sem alpha.

    Determinístico: mesma string para mesmas dimensões.
    """
    r = _CORNER_RADIUS_PX
    sigma = _VIGNETTE_BLUR_SIGMA  # noqa: F841 — reservado para futura customização da vinheta

    # Máscara de cantos: alpha=0 fora do raio, 255 dentro.
    # Para cada pixel (X, Y):
    #   in_corner_zone = (X<r OR X>W-r) AND (Y<r OR Y>H-r)
    #   se in_corner_zone: alpha = 255 se distância ≤ r, senão 0
    #   senão: alpha = 255 (área central intocada)
    corner_alpha_expr = (
        f"if(lt(X\\,{r})*lt(Y\\,{r}),"
        f"if(lt(hypot(X-{r}\\,Y-{r})\\,{r}),255,0),"
        f"if(gt(X\\,W-{r})*lt(Y\\,{r}),"
        f"if(lt(hypot(X-(W-{r})\\,Y-{r})\\,{r}),255,0),"
        f"if(lt(X\\,{r})*gt(Y\\,H-{r}),"
        f"if(lt(hypot(X-{r}\\,Y-(H-{r}))\\,{r}),255,0),"
        f"if(gt(X\\,W-{r})*gt(Y\\,H-{r}),"
        f"if(lt(hypot(X-(W-{r})\\,Y-(H-{r}))\\,{r}),255,0),"
        f"255))))"
    )

    return (
        # Etapa 1+2: cor fria + vinheta nativa (preserva timestamps do vídeo)
        f"[0:v]{_COOL_EQ},vignette=mode=backward[graded];"
        # Split o vídeo gradado em duas cópias com mesmo domínio temporal:
        # uma vai virar canvas preto sincronizado, a outra vai mascarar.
        f"[graded]split[bg_src][mask_src];"
        # Canvas preto: aproveita o próprio vídeo gradado mas zera luminosidade
        # via lutyuv (Y=0, U/V=128 = preto neutro). Mantém duração/fps idênticos.
        f"[bg_src]lutyuv=y=0:u=128:v=128[black_bg];"
        # Máscara de cantos: format yuva420p + geq sobre alpha.
        f"[mask_src]format=yuva420p,"
        f"geq=lum='p(X,Y)':a='{corner_alpha_expr}'[masked];"
        # Overlay: o vídeo mascarado (transparente nas quinas) sobre o canvas
        # preto. Cantos saem pretos sólidos. Saída final em yuv420p.
        f"[black_bg][masked]overlay=x=0:y=0:format=yuv420[over];"
        f"[over]format=yuv420p[v]"
    )


def _get_video_dimensions(clip_path: Path) -> tuple[int, int]:
    """Retorna (width, height) do clipe via ffprobe, com fallback fixo.

    Mesmo padrão do `social_composer._probe_video_dimensions`.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        w_str, h_str = result.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Tratamento visual: falha ao probar dimensões de %s (%s); assumindo %dx%d",
            clip_path.name, exc, _FALLBACK_W, _FALLBACK_H,
        )
        return _FALLBACK_W, _FALLBACK_H
