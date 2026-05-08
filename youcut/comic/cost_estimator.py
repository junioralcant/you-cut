"""Estimador de custo do pipeline `youcut comic` (RF-33).

Calcula o custo total de IA antes da execução dos provedores pagos
(``gpt-image-1`` para imagens, Runway Gen-4 Turbo para image-to-video) e
aplica o teto duro configurável (``comic_cost_cap_usd``).

Tabela de preços é parametrizável; defaults reproduzem o pricing público
dos provedores na data da feature, podendo ser sobrescritos por
``PriceTable`` para testes determinísticos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from youcut.config import PipelineConfig
from youcut.models import CastMember, Panel

logger = logging.getLogger(__name__)


_DEFAULT_ANCHOR_USD = 0.04
_DEFAULT_BASE_IMAGE_USD = 0.04
_DEFAULT_I2V_PER_SECOND_USD = 0.05
_DEFAULT_MOUTH_SHEET_USD = 0.04


@dataclass(frozen=True)
class PriceTable:
    """Preços unitários em USD usados pelo estimador.

    Defaults aproximados (gpt-image-1 ~ $0.04/imagem 1024², Runway Gen-4 Turbo
    $0.05/s de saída). Pode ser sobrescrito em testes para garantir
    determinismo independente de oscilações de preço.
    """

    anchor_image_usd: float = _DEFAULT_ANCHOR_USD
    base_image_usd: float = _DEFAULT_BASE_IMAGE_USD
    i2v_per_second_usd: float = _DEFAULT_I2V_PER_SECOND_USD
    mouth_sheet_usd: float = _DEFAULT_MOUTH_SHEET_USD


@dataclass(frozen=True)
class CostBreakdown:
    """Detalhamento do custo estimado para auditoria/UX.

    Engines `scenes`/`prunaai`/`panels` populam `n_panels`/`base_image_cost_usd`/
    `i2v_cost_usd`. Engine `remotion` popula `mouth_sheet_cost_usd` e mantém
    os demais campos em 0; o render é local (`render_local_usd=0.0`).
    """

    n_cast: int
    n_panels: int
    panel_seconds_total: float
    anchor_cost_usd: float
    base_image_cost_usd: float
    i2v_cost_usd: float
    total_usd: float
    mouth_sheet_cost_usd: float = 0.0
    render_local_usd: float = 0.0
    engine: str = "scenes"


class CostCapExceededError(Exception):
    """Custo estimado excede o teto duro configurado."""


def estimate_cost(
    cast: Iterable[CastMember],
    panels: Iterable[Panel],
    config: PipelineConfig,
    *,
    prices: PriceTable | None = None,
) -> CostBreakdown:
    """Calcula o custo estimado de IA do pipeline.

    Engines i2v (`scenes`/`prunaai`/`panels`):
      ``n_pessoas × anchor_image_usd
       + n_painéis × base_image_usd
       + Σ panel_seconds × i2v_per_second_usd``.

    Engine `remotion` (render local):
      ``n_pessoas × (anchor_image_usd + mouth_sheet_usd)``
      (sem painéis, sem i2v).
    """

    table = prices or PriceTable()
    cast_list = list(cast)
    panels_list = list(panels)

    n_cast = len(cast_list)
    engine = config.comic_animation_engine

    if engine == "remotion":
        anchor_cost = round(n_cast * table.anchor_image_usd, 4)
        mouth_sheet_cost = round(n_cast * table.mouth_sheet_usd, 4)
        total = round(anchor_cost + mouth_sheet_cost, 4)
        breakdown = CostBreakdown(
            n_cast=n_cast,
            n_panels=0,
            panel_seconds_total=0.0,
            anchor_cost_usd=anchor_cost,
            base_image_cost_usd=0.0,
            i2v_cost_usd=0.0,
            mouth_sheet_cost_usd=mouth_sheet_cost,
            render_local_usd=0.0,
            total_usd=total,
            engine=engine,
        )
        logger.info(
            "comic.cost_estimator[remotion]: cast=%d total=$%.2f "
            "(anchor=$%.2f mouth_sheets=$%.2f render_local=$0.00)",
            n_cast,
            total,
            anchor_cost,
            mouth_sheet_cost,
        )
        return breakdown

    n_panels = len(panels_list)
    panel_seconds_total = sum(max(0.0, p.panel_seconds_target) for p in panels_list)

    anchor_cost = round(n_cast * table.anchor_image_usd, 4)
    base_image_cost = round(n_panels * table.base_image_usd, 4)
    i2v_cost = round(panel_seconds_total * table.i2v_per_second_usd, 4)
    total = round(anchor_cost + base_image_cost + i2v_cost, 4)

    breakdown = CostBreakdown(
        n_cast=n_cast,
        n_panels=n_panels,
        panel_seconds_total=panel_seconds_total,
        anchor_cost_usd=anchor_cost,
        base_image_cost_usd=base_image_cost,
        i2v_cost_usd=i2v_cost,
        total_usd=total,
        engine=engine,
    )
    logger.info(
        "comic.cost_estimator[%s]: cast=%d painéis=%d total=$%.2f "
        "(anchor=$%.2f base=$%.2f i2v=$%.2f)",
        engine,
        n_cast,
        n_panels,
        total,
        anchor_cost,
        base_image_cost,
        i2v_cost,
    )
    return breakdown


def format_breakdown(breakdown: CostBreakdown) -> str:
    """Retorna string pt-BR descrevendo o `breakdown` para exibição no CLI."""
    if breakdown.engine == "remotion":
        return (
            f"Engine remotion · cast: {breakdown.n_cast} personagens · "
            f"total: US$ {breakdown.total_usd:.2f} "
            f"(âncoras US$ {breakdown.anchor_cost_usd:.2f} · "
            f"mouth sheets US$ {breakdown.mouth_sheet_cost_usd:.2f} · "
            f"Render local: US$ 0.00)."
        )
    return (
        f"Engine {breakdown.engine} · {breakdown.n_panels} painéis · "
        f"total: US$ {breakdown.total_usd:.2f} "
        f"(âncoras US$ {breakdown.anchor_cost_usd:.2f} · "
        f"imagens US$ {breakdown.base_image_cost_usd:.2f} · "
        f"i2v US$ {breakdown.i2v_cost_usd:.2f})."
    )


def enforce_cap(estimated_usd: float, cap_usd: float) -> None:
    """Levanta ``CostCapExceededError`` quando o custo excede o teto.

    Mensagem é em pt-BR e contém o valor estimado e o cap configurado para
    facilitar a UX e o diagnóstico.
    """

    if cap_usd < 0:
        raise ValueError(f"cap_usd deve ser ≥ 0, recebido {cap_usd}")
    if estimated_usd > cap_usd:
        raise CostCapExceededError(
            f"Custo estimado de US$ {estimated_usd:.2f} excede o teto duro de "
            f"US$ {cap_usd:.2f}. Reduza --max-panels ou ajuste --cost-cap."
        )


def preflight(
    cast: Iterable[CastMember],
    panels: Iterable[Panel],
    config: PipelineConfig,
    *,
    prices: PriceTable | None = None,
) -> CostBreakdown:
    """Calcula custo e aplica o teto duro de ``config.comic_cost_cap_usd``.

    Atalho usado pelo CLI antes de iniciar a renderização paga (RF-33).
    """

    breakdown = estimate_cost(cast, panels, config, prices=prices)
    enforce_cap(breakdown.total_usd, config.comic_cost_cap_usd)
    return breakdown
