import pytest

from youcut.comic.cost_estimator import (
    CostBreakdown,
    CostCapExceededError,
    PriceTable,
    enforce_cap,
    estimate_cost,
    preflight,
)
from youcut.models import CastMember, Panel


def _make_panels(count: int, seconds: float = 3.0) -> list[Panel]:
    return [
        Panel(
            index=i,
            start_time=i * seconds,
            end_time=(i + 1) * seconds,
            participants=["p1"],
            framing="close",
            scene="cena",
            pose_description="pose",
            panel_seconds_target=seconds,
        )
        for i in range(count)
    ]


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def fixed_prices():
    return PriceTable(
        anchor_image_usd=0.04,
        base_image_usd=0.04,
        i2v_per_second_usd=0.05,
    )


def test_estimate_with_three_panels(config, fixed_prices):
    cast = [CastMember(character_id="p1"), CastMember(character_id="p2")]
    panels = _make_panels(3, seconds=3.0)
    breakdown = estimate_cost(cast, panels, config, prices=fixed_prices)

    assert isinstance(breakdown, CostBreakdown)
    assert breakdown.n_cast == 2
    assert breakdown.n_panels == 3
    assert breakdown.panel_seconds_total == pytest.approx(9.0)
    assert breakdown.anchor_cost_usd == pytest.approx(0.08)
    assert breakdown.base_image_cost_usd == pytest.approx(0.12)
    assert breakdown.i2v_cost_usd == pytest.approx(0.45)
    assert breakdown.total_usd == pytest.approx(0.65)


def test_estimate_is_deterministic(config, fixed_prices):
    cast = [CastMember(character_id="p1")]
    panels = _make_panels(5, seconds=4.0)
    a = estimate_cost(cast, panels, config, prices=fixed_prices)
    b = estimate_cost(cast, panels, config, prices=fixed_prices)
    assert a == b


def test_estimate_zero_panels_and_cast(config, fixed_prices):
    breakdown = estimate_cost([], [], config, prices=fixed_prices)
    assert breakdown.total_usd == 0.0


def test_estimate_uses_default_prices_when_none_provided(config):
    cast = [CastMember(character_id="p1")]
    panels = _make_panels(2, seconds=3.0)
    breakdown = estimate_cost(cast, panels, config)
    assert breakdown.total_usd > 0


def test_enforce_cap_passes_when_under(config):
    enforce_cap(estimated_usd=1.50, cap_usd=10.0)


def test_enforce_cap_passes_when_exactly_at_cap(config):
    enforce_cap(estimated_usd=10.0, cap_usd=10.0)


def test_enforce_cap_raises_when_over():
    with pytest.raises(CostCapExceededError) as exc_info:
        enforce_cap(estimated_usd=12.34, cap_usd=10.0)
    msg = str(exc_info.value)
    assert "12.34" in msg
    assert "10.00" in msg
    assert "teto" in msg.lower() or "cap" in msg.lower()


def test_enforce_cap_rejects_negative_cap():
    with pytest.raises(ValueError, match=r"cap_usd"):
        enforce_cap(estimated_usd=1.0, cap_usd=-1.0)


def test_preflight_raises_when_total_exceeds_cap(config, fixed_prices, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    capped_config = PipelineConfig(comic_cost_cap_usd=0.1)

    cast = [CastMember(character_id="p1")]
    panels = _make_panels(5, seconds=5.0)
    with pytest.raises(CostCapExceededError):
        preflight(cast, panels, capped_config, prices=fixed_prices)


def test_preflight_returns_breakdown_when_under_cap(config, fixed_prices):
    cast = [CastMember(character_id="p1")]
    panels = _make_panels(2, seconds=3.0)
    breakdown = preflight(cast, panels, config, prices=fixed_prices)
    assert breakdown.n_panels == 2
    assert breakdown.total_usd <= config.comic_cost_cap_usd


def test_price_table_defaults_match_techspec():
    table = PriceTable()
    assert table.anchor_image_usd == 0.04
    assert table.base_image_usd == 0.04
    assert table.i2v_per_second_usd == 0.05
