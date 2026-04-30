import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._fakes.comic_providers import FakeI2VProvider, FakeImageProvider
from youcut.comic.panel_renderer import (
    MAX_REFERENCES_PER_PANEL,
    _build_i2v_prompt,
    _build_image_base_prompt,
    _select_references,
    _split_speaking_vs_silent,
    render_all,
    render_panel,
)
from youcut.comic.providers.i2v import I2VGenerationError
from youcut.comic.providers.images import ImageGenerationError
from youcut.models import CastMember, Panel, SpeakerSegment


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def _make_anchor(tmp_path: Path, character_id: str) -> Path:
    p = tmp_path / f"{character_id}.png"
    Image.new("RGB", (1024, 1024), (180, 200, 220)).save(p)
    return p


def _make_cast(tmp_path: Path, count: int = 2) -> list[CastMember]:
    out: list[CastMember] = []
    for i in range(count):
        cid = f"person_{i+1}"
        out.append(
            CastMember(
                character_id=cid,
                kind="person",
                gender_apparent="feminino" if i % 2 == 0 else "masculino",
                hair="cabelo curto",
                clothing="camiseta",
                narrative_role=f"papel_{i+1}",
                anchor_image_path=_make_anchor(tmp_path, cid),
                text_card=f"ficha de {cid}",
            )
        )
    return out


def _make_panel(participants: list[str], index: int = 0, seconds: float = 3.0) -> Panel:
    return Panel(
        index=index,
        start_time=index * seconds,
        end_time=(index + 1) * seconds,
        participants=participants,
        framing="close",
        scene="cena urbana",
        pose_description="rindo, olhar surpreso",
        panel_seconds_target=seconds,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_image_base_prompt_includes_style_and_no_brand_rule(tmp_path):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1", "person_2"], index=0)
    prompt = _build_image_base_prompt(panel, cast)
    assert "9:16" in prompt
    assert "caricatura editorial" in prompt
    assert "marcas/logos/handles" in prompt
    assert "person_1" in prompt
    assert "person_2" in prompt
    assert "cena urbana" in prompt
    assert "rindo" in prompt


def test_image_base_prompt_uses_framing_label(tmp_path):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"]).model_copy(update={"framing": "two_shot"})
    prompt = _build_image_base_prompt(panel, cast)
    assert "two-shot" in prompt


def test_i2v_prompt_uses_expressive_motion():
    panel = _make_panel(["person_1"])
    prompt = _build_i2v_prompt(panel)
    assert "Movimento expressivo" in prompt
    assert "SINCRONIZADO com a fala" in prompt
    assert "PICO EMOCIONAL" in prompt
    assert "frame final" in prompt
    assert "sem mudanças de cenário" in prompt


# ---------------------------------------------------------------------------
# Lipsync split (quem fala vs quem está calado no painel)
# ---------------------------------------------------------------------------


def _cast_with_speakers(tmp_path: Path) -> list[CastMember]:
    out = [
        CastMember(
            character_id="apresentador",
            kind="person",
            anchor_image_path=_make_anchor(tmp_path, "apresentador"),
            speaker_id="SPEAKER_00",
        ),
        CastMember(
            character_id="cozinheira",
            kind="person",
            anchor_image_path=_make_anchor(tmp_path, "cozinheira"),
            speaker_id=None,  # não tem speaker mapeado (ela é do vídeo reagido, não fala aqui)
        ),
        CastMember(
            character_id="tacho",
            kind="object",
            anchor_image_path=_make_anchor(tmp_path, "tacho"),
        ),
    ]
    return out


def test_split_active_speaker_only_apresentador_fala_durante_painel(tmp_path):
    cast = _cast_with_speakers(tmp_path)
    panel = Panel(
        index=0,
        start_time=0.0,
        end_time=3.5,
        participants=["apresentador", "cozinheira", "tacho"],
        framing="medium",
        scene="cozinha",
        pose_description="reator falando enquanto cozinheira despeja tripas",
        panel_seconds_target=3.5,
    )
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=3.5)]
    speaking, silent = _split_speaking_vs_silent(panel, cast, speakers)
    assert [m.character_id for m in speaking] == ["apresentador"]
    assert [m.character_id for m in silent] == ["cozinheira"]  # objeto excluído


def test_split_voiceover_when_no_visible_character_speaks(tmp_path):
    cast = _cast_with_speakers(tmp_path)
    panel = Panel(
        index=1,
        start_time=3.5,
        end_time=6.5,
        participants=["cozinheira", "tacho"],  # apresentador não está em cena
        framing="wide",
        scene="cozinha",
        pose_description="cozinheira despejando tripas",
        panel_seconds_target=3.0,
    )
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=3.5, end=6.5)]
    speaking, silent = _split_speaking_vs_silent(panel, cast, speakers)
    assert speaking == []  # apresentador fala mas não está visível -> voz em off
    assert [m.character_id for m in silent] == ["cozinheira"]


def test_i2v_prompt_includes_lipsync_voiceover_block(tmp_path):
    cast = _cast_with_speakers(tmp_path)
    panel = Panel(
        index=1,
        start_time=3.5,
        end_time=6.5,
        participants=["cozinheira"],
        framing="wide",
        scene="cozinha",
        pose_description="despejando tripas",
        panel_seconds_target=3.0,
    )
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=3.5, end=6.5)]
    prompt = _build_i2v_prompt(panel, cast=cast, speakers=speakers)
    assert "VOZ EM OFF" in prompt
    assert "boca FECHADA" in prompt
    assert "cozinheira" in prompt


def test_i2v_prompt_includes_lipsync_with_silent_partners(tmp_path):
    cast = _cast_with_speakers(tmp_path)
    panel = Panel(
        index=0,
        start_time=0.0,
        end_time=3.5,
        participants=["apresentador", "cozinheira"],
        framing="two_shot",
        scene="cozinha",
        pose_description="reator falando ao lado da cozinheira",
        panel_seconds_target=3.5,
    )
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=3.5)]
    prompt = _build_i2v_prompt(panel, cast=cast, speakers=speakers)
    assert "APENAS apresentador" in prompt
    assert "cozinheira NÃO fala" in prompt or "cozinheira" in prompt and "NÃO fala" in prompt


def test_i2v_prompt_no_lipsync_block_when_speakers_unavailable(tmp_path):
    cast = _cast_with_speakers(tmp_path)
    panel = Panel(
        index=0,
        start_time=0.0,
        end_time=3.0,
        participants=["apresentador", "cozinheira"],
        framing="medium",
        scene="cozinha",
        pose_description="rindo",
        panel_seconds_target=3.0,
    )
    prompt = _build_i2v_prompt(panel, cast=cast, speakers=None)
    # Sem speakers info, fallback conservador → todos vão pra "speaking",
    # nenhum bloco LIP-SYNC com restrição é emitido
    assert "VOZ EM OFF" not in prompt
    assert "NÃO fala" not in prompt


def test_select_references_truncates_to_three(tmp_path):
    cast = _make_cast(tmp_path, count=5)
    panel = _make_panel([m.character_id for m in cast])
    refs = _select_references(panel, cast)
    assert len(refs) == MAX_REFERENCES_PER_PANEL
    assert refs == [m.anchor_image_path for m in cast[:MAX_REFERENCES_PER_PANEL]]


def test_select_references_skips_missing_anchors(tmp_path):
    cast = _make_cast(tmp_path, count=2)
    cast[0] = cast[0].model_copy(update={"anchor_image_path": tmp_path / "missing.png"})
    panel = _make_panel(["person_1", "person_2"])
    refs = _select_references(panel, cast)
    assert len(refs) == 1
    assert refs[0].name == "person_2.png"


# ---------------------------------------------------------------------------
# render_panel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_render_panel_happy_path(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1", "person_2"], index=0, seconds=3.0)

    fake_image = FakeImageProvider(size=(1024, 1792))
    fake_i2v = FakeI2VProvider()

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )

    assert result.panel_index == 0
    assert result.base_image_path.exists()
    assert result.base_image_path.read_bytes().startswith(b"\x89PNG")
    assert result.clip_path.exists()
    assert result.was_static_fallback is False
    assert result.image_attempts == 1
    assert result.i2v_attempts == 1
    assert result.clip_seconds == 3.0
    assert result.cost_usd > 0
    assert len(fake_image.calls) == 1
    assert len(fake_i2v.calls) == 1
    refs_used = fake_image.calls[0]["reference_images"]
    assert len(refs_used) == 2


def test_render_panel_image_failure_propagates(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], index=0)

    bad_image = MagicMock()
    bad_image.generate.side_effect = ImageGenerationError("api down")

    fake_i2v = FakeI2VProvider()

    with pytest.raises(ImageGenerationError, match=r"api down"):
        render_panel(
            panel,
            cast,
            tmp_path / "output",
            config,
            image_provider=bad_image,
            i2v_provider=fake_i2v,
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_render_panel_falls_back_when_i2v_fails(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], index=2, seconds=3.0)

    fake_image = FakeImageProvider(size=(1024, 1792))
    bad_i2v = MagicMock()
    bad_i2v.image_to_video.side_effect = I2VGenerationError("runway down")

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=bad_i2v,
    )

    assert result.was_static_fallback is True
    assert result.clip_path.exists()
    assert result.clip_path.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_render_panel_falls_back_when_i2v_returns_empty_bytes(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], index=0, seconds=2.0)

    fake_image = FakeImageProvider(size=(1024, 1792))
    bad_i2v = MagicMock()
    bad_i2v.image_to_video.return_value = b""

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=bad_i2v,
    )

    assert result.was_static_fallback is True
    assert result.clip_path.exists()


def test_render_panel_clamps_duration_to_min(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], seconds=2.0).model_copy(
        update={"panel_seconds_target": 0.1}
    )

    fake_image = FakeImageProvider()
    fake_i2v = MagicMock()
    fake_i2v.image_to_video.return_value = b"\x00\x00mp4"

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )
    assert result.clip_seconds == config.comic_panel_min_seconds


def test_render_panel_clamps_duration_to_max(tmp_path, config):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], seconds=2.0).model_copy(
        update={"panel_seconds_target": 9.0}
    )

    fake_image = FakeImageProvider()
    fake_i2v = MagicMock()
    fake_i2v.image_to_video.return_value = b"\x00\x00mp4"

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )
    assert result.clip_seconds == config.comic_panel_max_seconds


def test_render_panel_cost_excludes_i2v_when_fallback(tmp_path, config, monkeypatch):
    cast = _make_cast(tmp_path)
    panel = _make_panel(["person_1"], index=0, seconds=4.0)

    fake_image = FakeImageProvider()
    bad_i2v = MagicMock()
    bad_i2v.image_to_video.side_effect = I2VGenerationError("nope")

    monkeypatch.setattr(
        "youcut.comic.panel_renderer._render_static_fallback",
        lambda image_path, clip_path, duration: clip_path.write_bytes(b"fakemp4"),
    )

    out = tmp_path / "output"
    result = render_panel(
        panel,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=bad_i2v,
    )
    assert result.was_static_fallback is True
    assert result.cost_usd == pytest.approx(0.04, abs=0.01)


# ---------------------------------------------------------------------------
# render_all
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_render_all_with_six_panels_in_parallel(tmp_path, config, monkeypatch):
    cast = _make_cast(tmp_path)
    panels = [_make_panel(["person_1"], index=i, seconds=2.0) for i in range(6)]

    PER_PANEL_DELAY = 0.15

    class SlowImageProvider:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, prompt, *, reference_images=None, size="1024x1024", input_fidelity="high"):
            time.sleep(PER_PANEL_DELAY)
            self.calls.append({"prompt": prompt})
            from PIL import Image as _Image
            import io
            img = _Image.new("RGB", (1024, 1792), (200, 200, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    class SlowI2VProvider:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def image_to_video(self, prompt_image, prompt_text, reference_images, duration_seconds=3.0, ratio="720:1280"):
            time.sleep(PER_PANEL_DELAY)
            self.calls.append({"duration": duration_seconds})
            return b"\x00\x00mp4-bytes"

    image_provider = SlowImageProvider()
    i2v_provider = SlowI2VProvider()

    config = config.model_copy(update={"comic_i2v_concurrency": 3})
    out = tmp_path / "output"

    start = time.monotonic()
    results = render_all(
        panels,
        cast,
        out,
        config,
        image_provider=image_provider,
        i2v_provider=i2v_provider,
    )
    elapsed = time.monotonic() - start

    assert len(results) == 6
    assert [r.panel_index for r in results] == [0, 1, 2, 3, 4, 5]
    assert all(r.clip_path.exists() for r in results)
    sequential = 6 * 2 * PER_PANEL_DELAY
    assert elapsed < sequential, f"paralelismo não acelerou (elapsed={elapsed:.2f}s, seq={sequential:.2f}s)"


def test_render_all_returns_empty_for_empty_panels(tmp_path, config):
    out = tmp_path / "output"
    fake_image = FakeImageProvider()
    fake_i2v = MagicMock()
    results = render_all([], [], out, config, image_provider=fake_image, i2v_provider=fake_i2v)
    assert results == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg ausente")
def test_render_all_preserves_index_order(tmp_path, config):
    cast = _make_cast(tmp_path)
    panels = [_make_panel(["person_1"], index=i, seconds=2.0) for i in range(4)]
    fake_image = FakeImageProvider()
    fake_i2v = FakeI2VProvider()
    out = tmp_path / "output"
    results = render_all(
        panels,
        cast,
        out,
        config,
        image_provider=fake_image,
        i2v_provider=fake_i2v,
    )
    assert [r.panel_index for r in results] == [0, 1, 2, 3]
