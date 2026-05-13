import pytest

from youcut.color_filter import VALID_PRESETS, get_filter_chain


def test_none_preset_returns_empty_chain():
    assert get_filter_chain("none") == ""


@pytest.mark.parametrize("preset", [p for p in VALID_PRESETS if p != "none"])
def test_named_presets_return_non_empty_ffmpeg_chain(preset):
    chain = get_filter_chain(preset)
    assert chain
    # cada preset usa pelo menos eq= ou curves=
    assert "eq=" in chain or "curves=" in chain


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_filter_chain("teal-and-orange")


def test_valid_presets_includes_documented_set():
    assert set(VALID_PRESETS) == {"none", "warm", "cool", "vintage", "punchy", "motivacao_lilac"}
