"""Testes do branch `remotion` do cost_estimator (Task 4.0)."""

from __future__ import annotations

import pytest

from youcut.comic.cost_estimator import (
    CostBreakdown,
    CostCapExceededError,
    PriceTable,
    estimate_cost,
    format_breakdown,
    preflight,
)
from youcut.models import CastMember


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cast(n: int) -> list[CastMember]:
    return [CastMember(character_id=f"speaker_{i}") for i in range(n)]


@pytest.fixture
def remotion_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    return PipelineConfig(comic_animation_engine="remotion")


@pytest.fixture
def fixed_prices() -> PriceTable:
    return PriceTable(
        anchor_image_usd=0.04,
        base_image_usd=0.04,
        i2v_per_second_usd=0.05,
        mouth_sheet_usd=0.04,
    )


# ── Engine remotion ──────────────────────────────────────────────────────────


def test_remotion_zero_cast(remotion_config, fixed_prices):
    breakdown = estimate_cost([], [], remotion_config, prices=fixed_prices)
    assert breakdown.engine == "remotion"
    assert breakdown.n_cast == 0
    assert breakdown.total_usd == 0.0
    assert breakdown.render_local_usd == 0.0


def test_remotion_one_cast_member(remotion_config, fixed_prices):
    breakdown = estimate_cost(_cast(1), [], remotion_config, prices=fixed_prices)
    assert breakdown.anchor_cost_usd == pytest.approx(0.04)
    assert breakdown.mouth_sheet_cost_usd == pytest.approx(0.04)
    assert breakdown.total_usd == pytest.approx(0.08)


def test_remotion_two_cast_members(remotion_config, fixed_prices):
    breakdown = estimate_cost(_cast(2), [], remotion_config, prices=fixed_prices)
    assert breakdown.anchor_cost_usd == pytest.approx(0.08)
    assert breakdown.mouth_sheet_cost_usd == pytest.approx(0.08)
    assert breakdown.total_usd == pytest.approx(0.16)


def test_remotion_four_cast_members_under_one_dollar(remotion_config, fixed_prices):
    """RF-23: ≤ $1 com até 4 personagens."""
    breakdown = estimate_cost(_cast(4), [], remotion_config, prices=fixed_prices)
    assert breakdown.total_usd <= 1.0
    assert breakdown.total_usd == pytest.approx(0.32)


def test_remotion_ignores_panels_argument(remotion_config, fixed_prices):
    """No engine remotion, painéis enviados (mesmo que erroneamente) não contam."""
    from youcut.models import Panel

    panels = [
        Panel(
            index=0,
            start_time=0.0,
            end_time=3.0,
            participants=["x"],
            framing="close",
            scene="cena",
            pose_description="pose",
            panel_seconds_target=3.0,
        )
    ]
    breakdown = estimate_cost(
        _cast(2), panels, remotion_config, prices=fixed_prices
    )
    assert breakdown.n_panels == 0
    assert breakdown.base_image_cost_usd == 0.0
    assert breakdown.i2v_cost_usd == 0.0


def test_remotion_render_local_is_zero(remotion_config, fixed_prices):
    breakdown = estimate_cost(_cast(3), [], remotion_config, prices=fixed_prices)
    assert breakdown.render_local_usd == 0.0


# ── Cap enforcement ─────────────────────────────────────────────────────────


def test_remotion_preflight_respects_cap(monkeypatch, fixed_prices):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine="remotion", comic_cost_cap_usd=0.05)
    with pytest.raises(CostCapExceededError):
        preflight(_cast(4), [], config, prices=fixed_prices)


def test_remotion_preflight_passes_under_cap(remotion_config, fixed_prices):
    breakdown = preflight(_cast(4), [], remotion_config, prices=fixed_prices)
    assert breakdown.total_usd <= remotion_config.comic_cost_cap_usd


# ── Não-regressão dos engines existentes ────────────────────────────────────


@pytest.mark.parametrize("engine", ["scenes", "prunaai", "panels"])
def test_existing_engines_unchanged(engine, monkeypatch, fixed_prices):
    """Engines i2v continuam usando n_panels × base_image + Σ seg × i2v."""
    from youcut.models import Panel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig

    config = PipelineConfig(comic_animation_engine=engine)
    cast = _cast(2)
    panels = [
        Panel(
            index=i,
            start_time=i * 3.0,
            end_time=(i + 1) * 3.0,
            participants=["x"],
            framing="close",
            scene="s",
            pose_description="p",
            panel_seconds_target=3.0,
        )
        for i in range(3)
    ]
    breakdown = estimate_cost(cast, panels, config, prices=fixed_prices)
    assert breakdown.engine == engine
    assert breakdown.n_panels == 3
    # 2 anchors × 0.04 + 3 base × 0.04 + 9s × 0.05 = 0.08 + 0.12 + 0.45 = 0.65
    assert breakdown.total_usd == pytest.approx(0.65)
    assert breakdown.mouth_sheet_cost_usd == 0.0


# ── format_breakdown ────────────────────────────────────────────────────────


def test_format_breakdown_remotion(remotion_config, fixed_prices):
    breakdown = estimate_cost(_cast(2), [], remotion_config, prices=fixed_prices)
    text = format_breakdown(breakdown)
    assert "remotion" in text
    assert "personagens" in text
    assert "Render local" in text
    assert "0.00" in text  # render local zero
    assert "US$" in text


def test_format_breakdown_scenes(monkeypatch, fixed_prices):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    from youcut.models import Panel

    config = PipelineConfig(comic_animation_engine="scenes")
    panels = [
        Panel(
            index=0,
            start_time=0,
            end_time=3,
            participants=["x"],
            framing="close",
            scene="s",
            pose_description="p",
            panel_seconds_target=3.0,
        )
    ]
    breakdown = estimate_cost(_cast(1), panels, config, prices=fixed_prices)
    text = format_breakdown(breakdown)
    assert "scenes" in text
    assert "painéis" in text
    assert "i2v" in text


# ── PriceTable ──────────────────────────────────────────────────────────────


def test_price_table_default_mouth_sheet():
    table = PriceTable()
    assert table.mouth_sheet_usd == 0.04


def test_cost_breakdown_engine_default_is_scenes():
    """CostBreakdown construído sem `engine` mantém default scenes (compat)."""
    bd = CostBreakdown(
        n_cast=0,
        n_panels=0,
        panel_seconds_total=0.0,
        anchor_cost_usd=0.0,
        base_image_cost_usd=0.0,
        i2v_cost_usd=0.0,
        total_usd=0.0,
    )
    assert bd.engine == "scenes"
    assert bd.mouth_sheet_cost_usd == 0.0
    assert bd.render_local_usd == 0.0
