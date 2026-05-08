"""Mouth shape sheet builder — gera 4 mouth shapes por personagem.

Para cada `CastMember` do cast, faz **1 chamada paga** ao `gpt-image-1`
gerando uma sheet 1024×1024 contendo as 4 mouth shapes em grid 2×2:

    +--------+--------+
    | CLOSED | OPEN_MID
    +--------+--------+
    | OPEN_WIDE | OPEN_ROUND
    +--------+--------+

A anchor do personagem é passada como `reference_images` para preservar
identidade (mesma cabeça, escala e iluminação entre as 4 células). A
imagem original é validada (dimensões + variância por quadrante) e
recortada via Pillow em 4 PNGs 512×512 nomeados
``mouth_<character_id>_<shape>.png`` em ``output/<video>/comic/cast/``.

Estratégia da grid 2×2 vs lado-a-lado: o gpt-image-1 só aceita
`1024×1024` / `1024×1536` / `1536×1024`. A grid 2×2 entrega cells nativas
512×512 sem warp Pillow. A techspec descreve "lado-a-lado 2048×512" como
intenção; aqui a entrega funcional é equivalente — 4 cells 512×512
distintas por personagem — sem extrapolar limites do provider.

Idempotência: se a sheet e os 4 PNGs já existem em disco, a função pula
a chamada paga e devolve o `MouthSheet` reconstruído.

Em caso de falha de validação após 2 tentativas (prompt original +
retry com prompt corretivo), cai para o fallback de 4 chamadas
separadas — uma por shape, custo até ~$0.20 por personagem.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageStat

from youcut.comic.providers.images import (
    DEFAULT_SIZE,
    ImageGenerationError,
    ImageProvider,
)
from youcut.models import CastMember, MouthShape, MouthSheet

logger = logging.getLogger(__name__)


SHEET_SIZE: str = DEFAULT_SIZE  # "1024x1024"
SHEET_WIDTH: int = 1024
SHEET_HEIGHT: int = 1024
CELL_SIZE: int = 512
DIMENSION_TOLERANCE: float = 0.02  # ±2%
MIN_CELL_VARIANCE: float = 1.0  # stddev em escala 0-255

# Layout 2×2: ordem "abertura crescente" → top-left ao bottom-right
SHEET_LAYOUT: dict[MouthShape, tuple[int, int, int, int]] = {
    MouthShape.CLOSED: (0, 0, CELL_SIZE, CELL_SIZE),
    MouthShape.OPEN_MID: (CELL_SIZE, 0, 2 * CELL_SIZE, CELL_SIZE),
    MouthShape.OPEN_WIDE: (0, CELL_SIZE, CELL_SIZE, 2 * CELL_SIZE),
    MouthShape.OPEN_ROUND: (CELL_SIZE, CELL_SIZE, 2 * CELL_SIZE, 2 * CELL_SIZE),
}


# ── Prompts ────────────────────────────────────────────────────────────────


def _build_sheet_prompt(member: CastMember, *, reinforced: bool = False) -> str:
    text_card = member.text_card or member.character_id
    base = (
        "Gere uma folha de referência (model sheet) com EXATAMENTE 4 retratos "
        "do MESMO personagem em grid 2×2 (canto-superior-esquerdo, "
        "canto-superior-direito, canto-inferior-esquerdo, canto-inferior-direito). "
        "Em todas as 4 células: mesma cabeça, mesma escala, mesma iluminação, "
        "mesma roupa, mesma cor de fundo. Apenas a BOCA muda entre as células. "
        "Ordem das bocas: "
        "(1) superior-esquerdo: BOCA FECHADA, lábios relaxados; "
        "(2) superior-direito: BOCA SEMI-ABERTA, dentes parcialmente visíveis; "
        "(3) inferior-esquerdo: BOCA BEM ABERTA, retangular, dentes superiores "
        "e inferiores visíveis (vogal A/E); "
        "(4) inferior-direito: BOCA ABERTA E REDONDA, lábios projetados (vogal O/U). "
        f"Personagem: {text_card}. "
        "Estilo: caricatura editorial moderna idêntico à imagem de referência "
        "(traço, paleta, cabelo, rosto, acessórios). Cada célula deve ser um "
        "quadrado limpo, sem bordas, sem texto, sem rótulos. Fundo neutro "
        "uniforme entre as 4 células."
    )
    if reinforced:
        base += (
            " ATENÇÃO: a tentativa anterior falhou na validação. CRÍTICO entregar "
            "exatamente 4 células distintas em layout 2×2 perfeitamente alinhado, "
            "sem margens nem letras nem numeração. As 4 bocas devem ser claramente "
            "diferentes umas das outras."
        )
    return base


def _build_single_shape_prompt(member: CastMember, shape: MouthShape) -> str:
    text_card = member.text_card or member.character_id
    shape_descriptions = {
        MouthShape.CLOSED: "BOCA FECHADA, lábios relaxados, expressão neutra",
        MouthShape.OPEN_MID: "BOCA SEMI-ABERTA, dentes parcialmente visíveis",
        MouthShape.OPEN_WIDE: (
            "BOCA BEM ABERTA, retangular, dentes superiores e inferiores "
            "visíveis (vogal A/E)"
        ),
        MouthShape.OPEN_ROUND: (
            "BOCA ABERTA E REDONDA, lábios projetados (vogal O/U)"
        ),
    }
    return (
        f"Retrato único do personagem em pose neutra frontal, plano americano. "
        f"{shape_descriptions[shape]}. "
        f"Personagem: {text_card}. "
        "Estilo: caricatura editorial moderna idêntico à imagem de referência. "
        "Sem texto embutido, sem rótulos, sem grade, sem múltiplas células — "
        "apenas UM retrato centralizado em fundo neutro."
    )


# ── Filesystem ─────────────────────────────────────────────────────────────


def _ensure_cast_dir(output_dir: Path) -> Path:
    cast_dir = Path(output_dir) / "comic" / "cast"
    cast_dir.mkdir(parents=True, exist_ok=True)
    return cast_dir


def _sheet_path_for(cast_dir: Path, character_id: str) -> Path:
    return cast_dir / f"{character_id}_mouth_sheet.png"


def _cell_path_for(cast_dir: Path, character_id: str, shape: MouthShape) -> Path:
    return cast_dir / f"mouth_{character_id}_{shape.value}.png"


def _is_existing_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_existing_sheet(
    cast_dir: Path, character_id: str
) -> MouthSheet | None:
    """Reconstitui um `MouthSheet` se o arquivo da sheet e os 4 cells existem."""
    sheet_path = _sheet_path_for(cast_dir, character_id)
    if not _is_existing_file(sheet_path):
        return None
    cells_paths = {
        shape: _cell_path_for(cast_dir, character_id, shape)
        for shape in MouthShape
    }
    if not all(_is_existing_file(p) for p in cells_paths.values()):
        return None
    return MouthSheet(
        character_id=character_id,
        sheet_path=sheet_path,
        cells=dict(SHEET_LAYOUT),
    )


# ── Validação Pillow ───────────────────────────────────────────────────────


def _within_tolerance(actual: int, expected: int, tolerance: float) -> bool:
    delta = abs(actual - expected)
    return delta <= max(1, int(round(expected * tolerance)))


def _validate_sheet_image(image: Image.Image) -> tuple[bool, str]:
    """Valida dims (±2%) e que cada quadrante tem variância > limiar."""
    if not _within_tolerance(image.width, SHEET_WIDTH, DIMENSION_TOLERANCE):
        return False, f"largura inválida {image.width} (esperado ~{SHEET_WIDTH})"
    if not _within_tolerance(image.height, SHEET_HEIGHT, DIMENSION_TOLERANCE):
        return False, f"altura inválida {image.height} (esperado ~{SHEET_HEIGHT})"

    # converte para grayscale para variância robusta
    gray = image.convert("L") if image.mode != "L" else image
    half_w = image.width // 2
    half_h = image.height // 2
    quadrants = [
        gray.crop((0, 0, half_w, half_h)),
        gray.crop((half_w, 0, image.width, half_h)),
        gray.crop((0, half_h, half_w, image.height)),
        gray.crop((half_w, half_h, image.width, image.height)),
    ]
    for idx, quad in enumerate(quadrants):
        stddev = _stddev(quad)
        if stddev < MIN_CELL_VARIANCE:
            return False, (
                f"quadrante {idx} sem conteúdo (stddev={stddev:.2f} < "
                f"{MIN_CELL_VARIANCE})"
            )
    return True, ""


def _stddev(image: Image.Image) -> float:
    """Calcula stddev em escala 0–255 (input em modo `L`)."""
    stats = ImageStat.Stat(image)
    return stats.stddev[0] if stats.stddev else 0.0


# ── Recorte e persistência ─────────────────────────────────────────────────


def _save_sheet_and_cells(
    sheet_image: Image.Image,
    member: CastMember,
    cast_dir: Path,
) -> MouthSheet:
    sheet_path = _sheet_path_for(cast_dir, member.character_id)
    # Normaliza dimensões para 1024×1024 quando vier ligeiramente fora.
    if (sheet_image.width, sheet_image.height) != (SHEET_WIDTH, SHEET_HEIGHT):
        sheet_image = sheet_image.resize(
            (SHEET_WIDTH, SHEET_HEIGHT), Image.LANCZOS
        )
    sheet_image.save(sheet_path, format="PNG")

    cells: dict[MouthShape, tuple[int, int, int, int]] = {}
    for shape, box in SHEET_LAYOUT.items():
        cell_path = _cell_path_for(cast_dir, member.character_id, shape)
        cell_image = sheet_image.crop(box)
        if cell_image.size != (CELL_SIZE, CELL_SIZE):
            cell_image = cell_image.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        cell_image.save(cell_path, format="PNG")
        cells[shape] = box

    return MouthSheet(
        character_id=member.character_id,
        sheet_path=sheet_path,
        cells=cells,
    )


def _save_individual_cells(
    cell_images: dict[MouthShape, Image.Image],
    member: CastMember,
    cast_dir: Path,
) -> MouthSheet:
    """Compõe 1024×1024 a partir de 4 cells separadas e persiste tudo."""
    sheet = Image.new("RGB", (SHEET_WIDTH, SHEET_HEIGHT), color=(255, 255, 255))
    for shape, box in SHEET_LAYOUT.items():
        cell = cell_images[shape]
        if cell.size != (CELL_SIZE, CELL_SIZE):
            cell = cell.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        sheet.paste(cell, (box[0], box[1]))
    return _save_sheet_and_cells(sheet, member, cast_dir)


# ── Fluxo principal ────────────────────────────────────────────────────────


def _generate_single_cell(
    member: CastMember,
    anchor_path: Path,
    shape: MouthShape,
    *,
    image_provider: ImageProvider,
) -> Image.Image:
    """Gera 1 cell isolada. Solicita tamanho aceito por gpt-image-1 e reduz
    via Pillow para `CELL_SIZE`x`CELL_SIZE` em `_save_individual_cells`."""
    prompt = _build_single_shape_prompt(member, shape)
    png_bytes = image_provider.generate(
        prompt,
        reference_images=[anchor_path],
        size=SHEET_SIZE,
        input_fidelity="high",
    )
    if not png_bytes:
        raise ImageGenerationError(
            f"Provider retornou bytes vazios para mouth shape {shape.value} "
            f"de {member.character_id}"
        )
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _try_generate_sheet(
    member: CastMember,
    anchor_path: Path,
    *,
    image_provider: ImageProvider,
    reinforced: bool,
) -> tuple[Image.Image | None, str]:
    prompt = _build_sheet_prompt(member, reinforced=reinforced)
    try:
        png_bytes = image_provider.generate(
            prompt,
            reference_images=[anchor_path],
            size=SHEET_SIZE,
            input_fidelity="high",
        )
    except ImageGenerationError as exc:
        return None, f"provider error: {exc}"
    except Exception as exc:  # provider pode levantar tipos inesperados
        return None, f"provider raised {type(exc).__name__}: {exc}"
    if not png_bytes:
        return None, "provider returned empty bytes"
    try:
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as exc:
        return None, f"falha ao decodificar PNG: {exc}"
    ok, reason = _validate_sheet_image(image)
    if not ok:
        return None, f"validação falhou: {reason}"
    return image, ""


def build_mouth_sheet(
    member: CastMember,
    anchor_path: Path,
    output_dir: Path,
    *,
    image_provider: ImageProvider,
) -> MouthSheet:
    """Gera (ou recupera do cache) a `MouthSheet` de `member`.

    Args:
        member: personagem do cast (precisa ter `character_id`, `text_card` opc.).
        anchor_path: caminho da ficha-âncora gerada por `cast_builder`.
        output_dir: diretório raiz do output (`output/<video>/`).
        image_provider: provider compatível com `ImageProvider`.

    Returns:
        `MouthSheet` apontando para a sheet 1024×1024 e com `cells` no
        layout 2×2 padrão.

    Raises:
        ImageGenerationError: se a sheet falhar e o fallback de 4 chamadas
            separadas também falhar.
    """
    cast_dir = _ensure_cast_dir(output_dir)
    if not _is_existing_file(Path(anchor_path)):
        raise ImageGenerationError(
            f"Anchor inexistente para {member.character_id}: {anchor_path}"
        )

    cached = _load_existing_sheet(cast_dir, member.character_id)
    if cached is not None:
        logger.info(
            "comic.mouth_shapes: cache hit para %s — pulando chamada paga",
            member.character_id,
        )
        return cached

    # Tentativa 1 + retry com prompt reforçado
    last_reason = ""
    for attempt, reinforced in enumerate([False, True]):
        logger.info(
            "comic.mouth_shapes: gerando sheet para %s (tentativa %d, reforçado=%s)",
            member.character_id,
            attempt + 1,
            reinforced,
        )
        image, reason = _try_generate_sheet(
            member,
            Path(anchor_path),
            image_provider=image_provider,
            reinforced=reinforced,
        )
        if image is not None:
            return _save_sheet_and_cells(image, member, cast_dir)
        last_reason = reason
        logger.warning(
            "comic.mouth_shapes: tentativa %d falhou para %s (%s)",
            attempt + 1,
            member.character_id,
            reason,
        )

    # Fallback: 4 chamadas separadas (1 por shape)
    logger.warning(
        "comic.mouth_shapes: caindo em fallback (4 chamadas separadas) para %s "
        "após 2 tentativas falharem (%s)",
        member.character_id,
        last_reason,
    )
    cell_images: dict[MouthShape, Image.Image] = {}
    for shape in MouthShape:
        cell_images[shape] = _generate_single_cell(
            member, Path(anchor_path), shape, image_provider=image_provider
        )
    return _save_individual_cells(cell_images, member, cast_dir)
