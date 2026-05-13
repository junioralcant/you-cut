"""Presets de filtro de cor aplicáveis ao ``-vf`` do FFmpeg.

Cada preset combina ``curves`` + ``eq`` para criar uma assinatura visual
discreta (não distorce rosto/legenda) que diferencia o clipe de outros
re-uploads do mesmo trecho — uma das técnicas que o ``adoc.md`` cita como
"blindagem contra conteúdo reutilizado" no YouTube.
"""
from __future__ import annotations

from typing import Final

PRESET_NONE: Final = "none"

_PRESETS: Final[dict[str, str]] = {
    PRESET_NONE: "",
    # quente, alaranjado, levemente saturado
    "warm": "curves=preset=lighter,eq=saturation=1.10:contrast=1.05:gamma_r=1.04:gamma_b=0.96",
    # frio, azulado, com leve corte de saturação
    "cool": "curves=preset=darker,eq=saturation=0.95:contrast=1.05:gamma_r=0.96:gamma_b=1.06",
    # filme antigo: cross-process + saturação reduzida
    "vintage": "curves=preset=cross_process,eq=saturation=0.85:contrast=1.10:gamma=0.95",
    # punchy: contraste e saturação reforçados, sem viés de cor
    "punchy": "eq=saturation=1.20:contrast=1.12:brightness=0.02",
    # motivacao_lilac: sombras puxadas pra azul-violeta, curva S, vinheta sutil.
    # Casa com a luz lilás do set típico dos Reels motivacionais pt-BR
    # (ver tasks/prd-preset-motivacao/analise-video-referencia.md §7).
    "motivacao_lilac": (
        "eq=contrast=1.10:saturation=0.92:brightness=-0.02,"
        "curves=master='0/0 0.25/0.16 0.75/0.84 1/1',"
        "colorbalance=rs=-0.05:gs=-0.02:bs=0.06:rh=0.03:bh=-0.02,"
        "vignette=PI/5:eval=init"
    ),
}

VALID_PRESETS: Final[tuple[str, ...]] = tuple(_PRESETS)


def get_filter_chain(preset: str) -> str:
    """Devolve o fragmento de ``vf`` para o preset (string vazia para ``none``).

    Lança ``ValueError`` para presets desconhecidos para evitar passar uma string
    arbitrária do usuário direto pro FFmpeg.
    """
    if preset not in _PRESETS:
        raise ValueError(
            f"Preset de cor desconhecido: {preset!r}. Disponíveis: {sorted(_PRESETS)}"
        )
    return _PRESETS[preset]
