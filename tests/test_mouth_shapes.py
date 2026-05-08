"""Testes do mouth shape sheet builder (Task 3.0)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from youcut.comic.mouth_shapes import (
    CELL_SIZE,
    MIN_CELL_VARIANCE,
    SHEET_HEIGHT,
    SHEET_LAYOUT,
    SHEET_WIDTH,
    _validate_sheet_image,
    build_mouth_sheet,
)
from youcut.comic.providers.images import ImageGenerationError
from youcut.models import CastMember, MouthShape


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_quadrant_png(width: int, height: int) -> bytes:
    """Sheet sintética com 4 quadrantes em cores distintas (passa validação)."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    half_w, half_h = width // 2, height // 2
    quadrant_colors = [
        ((0, 0, half_w, half_h), (220, 200, 180)),
        ((half_w, 0, width, half_h), (180, 220, 200)),
        ((0, half_h, half_w, height), (200, 180, 220)),
        ((half_w, half_h, width, height), (200, 220, 180)),
    ]
    for box, color in quadrant_colors:
        draw.rectangle(box, fill=color)
        # adiciona um traço para garantir variância > 1.0
        draw.line(
            [(box[0] + 10, box[1] + 10), (box[2] - 10, box[3] - 10)],
            fill=(20, 20, 30),
            width=4,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_blank_png(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_anchor_png(tmp_path: Path, name: str = "anchor.png") -> Path:
    """Anchor PNG válido para uso como reference_image."""
    img = Image.new("RGB", (1024, 1024), color=(180, 200, 220))
    path = tmp_path / name
    img.save(path, format="PNG")
    return path


class _DeterministicProvider:
    """Provider de teste com payload configurável por chamada."""

    def __init__(self, payloads: list[bytes]) -> None:
        self._payloads = payloads
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[Path] | None = None,
        size: str = "1024x1024",
        input_fidelity: str = "high",
    ) -> bytes:
        idx = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "reference_images": list(reference_images or []),
                "size": size,
                "input_fidelity": input_fidelity,
            }
        )
        if idx >= len(self._payloads):
            raise ImageGenerationError("payloads esgotados")
        payload = self._payloads[idx]
        if isinstance(payload, Exception):
            raise payload
        return payload


# ── Validador ────────────────────────────────────────────────────────────────


def test_validate_sheet_image_accepts_valid_dimensions():
    img = Image.open(io.BytesIO(_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)))
    ok, reason = _validate_sheet_image(img)
    assert ok, reason


def test_validate_sheet_image_accepts_within_tolerance():
    # ±2% de 1024 ≈ 20px
    img = Image.open(io.BytesIO(_make_quadrant_png(1010, 1024)))
    ok, _ = _validate_sheet_image(img)
    assert ok


def test_validate_sheet_image_rejects_wrong_dimensions():
    img = Image.open(io.BytesIO(_make_quadrant_png(512, 512)))
    ok, reason = _validate_sheet_image(img)
    assert not ok
    assert "largura" in reason or "altura" in reason


def test_validate_sheet_image_rejects_blank_quadrants():
    img = Image.open(io.BytesIO(_make_blank_png(SHEET_WIDTH, SHEET_HEIGHT)))
    ok, reason = _validate_sheet_image(img)
    assert not ok
    assert "quadrante" in reason.lower() or "stddev" in reason.lower()


# ── Happy path ───────────────────────────────────────────────────────────────


def test_build_mouth_sheet_happy_path(tmp_path):
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)]
    )
    member = CastMember(character_id="speaker_a", text_card="ficha visual")

    sheet = build_mouth_sheet(
        member, anchor, tmp_path, image_provider=provider
    )

    assert isinstance(sheet.character_id, str)
    assert sheet.sheet_path.exists()
    assert sheet.sheet_path.name == "speaker_a_mouth_sheet.png"
    # 4 cells presentes no dicionário
    assert set(sheet.cells.keys()) == set(MouthShape)
    # 4 cells no disco com nome correto
    cast_dir = tmp_path / "comic" / "cast"
    for shape in MouthShape:
        cell_path = cast_dir / f"mouth_speaker_a_{shape.value}.png"
        assert cell_path.exists(), f"esperava {cell_path}"
        with Image.open(cell_path) as cell_img:
            assert cell_img.size == (CELL_SIZE, CELL_SIZE)
    # apenas 1 chamada paga
    assert len(provider.calls) == 1
    # anchor passada como referência
    assert provider.calls[0]["reference_images"] == [anchor]


def test_build_mouth_sheet_passes_input_fidelity_high(tmp_path):
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)]
    )
    member = CastMember(character_id="x")
    build_mouth_sheet(member, anchor, tmp_path, image_provider=provider)
    assert provider.calls[0]["input_fidelity"] == "high"


# ── Cache idempotente ────────────────────────────────────────────────────────


def test_build_mouth_sheet_cache_hit_skips_provider(tmp_path):
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)]
    )
    member = CastMember(character_id="cached_one")

    # primeira execução
    first = build_mouth_sheet(member, anchor, tmp_path, image_provider=provider)
    assert len(provider.calls) == 1

    # segunda execução com provider que falharia se chamado
    failing = _DeterministicProvider([])
    second = build_mouth_sheet(member, anchor, tmp_path, image_provider=failing)
    assert len(failing.calls) == 0
    assert second.sheet_path == first.sheet_path
    assert second.cells == first.cells


def test_cache_hit_requires_all_4_cells_present(tmp_path):
    """Sheet sem todos os 4 cells deve regenerar."""
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)]
    )
    member = CastMember(character_id="partial")

    build_mouth_sheet(member, anchor, tmp_path, image_provider=provider)
    # remove um dos cells
    cell = tmp_path / "comic" / "cast" / "mouth_partial_open_wide.png"
    cell.unlink()

    provider2 = _DeterministicProvider(
        [_make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)]
    )
    build_mouth_sheet(member, anchor, tmp_path, image_provider=provider2)
    assert len(provider2.calls) == 1, "deveria regenerar quando cells faltam"


# ── Retry ────────────────────────────────────────────────────────────────────


def test_retry_after_invalid_dimensions(tmp_path):
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [
            _make_quadrant_png(512, 512),  # tentativa 1: dimensão errada
            _make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT),  # retry: ok
        ]
    )
    member = CastMember(character_id="retry_one")

    sheet = build_mouth_sheet(
        member, anchor, tmp_path, image_provider=provider
    )

    assert sheet.sheet_path.exists()
    assert len(provider.calls) == 2
    # retry foi com prompt reforçado
    assert "ATENÇÃO" in provider.calls[1]["prompt"]


def test_retry_after_blank_quadrants(tmp_path):
    anchor = _make_anchor_png(tmp_path)
    provider = _DeterministicProvider(
        [
            _make_blank_png(SHEET_WIDTH, SHEET_HEIGHT),  # blank → falha variância
            _make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT),
        ]
    )
    member = CastMember(character_id="x")
    sheet = build_mouth_sheet(member, anchor, tmp_path, image_provider=provider)
    assert sheet.sheet_path.exists()
    assert len(provider.calls) == 2


# ── Fallback ─────────────────────────────────────────────────────────────────


def test_fallback_after_two_failures(tmp_path):
    """Após 2 retries falharem, faz 4 chamadas separadas (1 por shape)."""
    anchor = _make_anchor_png(tmp_path)
    blank = _make_blank_png(SHEET_WIDTH, SHEET_HEIGHT)
    full_cell = _make_quadrant_png(SHEET_WIDTH, SHEET_HEIGHT)
    provider = _DeterministicProvider(
        [
            blank,  # tentativa 1 (sheet) falha
            blank,  # tentativa 2 (sheet reforçado) falha
            full_cell,  # fallback shape closed
            full_cell,  # fallback shape open_mid
            full_cell,  # fallback shape open_wide
            full_cell,  # fallback shape open_round
        ]
    )
    member = CastMember(character_id="fallback_x")
    sheet = build_mouth_sheet(
        member, anchor, tmp_path, image_provider=provider
    )

    assert sheet.sheet_path.exists()
    assert len(provider.calls) == 6  # 2 sheet + 4 fallback cells
    # Fallback solicita tamanho aceito por gpt-image-1 (não o CELL_SIZE)
    assert provider.calls[2]["size"] == f"{SHEET_WIDTH}x{SHEET_HEIGHT}"
    # Cada cell salva tem dimensão correta após Pillow downscale
    cast_dir = tmp_path / "comic" / "cast"
    for shape in MouthShape:
        with Image.open(cast_dir / f"mouth_fallback_x_{shape.value}.png") as img:
            assert img.size == (CELL_SIZE, CELL_SIZE)


def test_fallback_propagates_provider_errors(tmp_path):
    """Se o fallback também falhar (provider levanta), propaga o erro."""
    anchor = _make_anchor_png(tmp_path)
    blank = _make_blank_png(SHEET_WIDTH, SHEET_HEIGHT)
    provider = _DeterministicProvider(
        [
            blank,
            blank,
            ImageGenerationError("OpenAI down"),  # primeira chamada de fallback explode
        ]
    )
    member = CastMember(character_id="x")
    with pytest.raises(ImageGenerationError):
        build_mouth_sheet(member, anchor, tmp_path, image_provider=provider)


# ── Erros de input ───────────────────────────────────────────────────────────


def test_missing_anchor_raises(tmp_path):
    member = CastMember(character_id="no_anchor")
    provider = _DeterministicProvider([])
    with pytest.raises(ImageGenerationError):
        build_mouth_sheet(
            member,
            tmp_path / "doesnt_exist.png",
            tmp_path,
            image_provider=provider,
        )


# ── Layout / contrato ────────────────────────────────────────────────────────


def test_sheet_layout_covers_all_4_shapes():
    assert set(SHEET_LAYOUT.keys()) == set(MouthShape)
    for shape, (x1, y1, x2, y2) in SHEET_LAYOUT.items():
        assert x2 - x1 == CELL_SIZE
        assert y2 - y1 == CELL_SIZE


def test_min_cell_variance_threshold():
    assert MIN_CELL_VARIANCE > 0.0
