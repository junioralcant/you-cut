"""Cast Builder — gera fichas textuais e imagens-âncora por personagem (RF-07..RF-11).

A imagem-âncora é gerada **uma única vez** por personagem (1024×1024, pose
neutra, fundo neutro, estilo minimalista pastel) e reusada como referência
em todas as chamadas subsequentes do `panel_renderer`. Reexecuções da task
não regeneram arquivos já presentes em ``output/<video>/comic/cast/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from youcut.comic.providers.images import (
    DEFAULT_SIZE,
    ImageGenerationError,
    ImageProvider,
)
from youcut.config import PipelineConfig
from youcut.models import CastMember

logger = logging.getLogger(__name__)


_STYLE_PROMPT = (
    "Estilo visual: caricatura editorial moderna, traço preto fino e expressivo, "
    "proporções levemente exageradas (cabeça ~20% maior que realista), olhos "
    "grandes e expressivos. Identidade FIEL às características descritas — "
    "gênero, idade, formato de rosto, cabelo, pelos faciais, tom de pele, "
    "roupa e acessórios devem ser claramente reconhecíveis. Paleta pastel "
    "dessaturada, fundo aquarela digital neutra. Sem fotorrealismo, sem texto "
    "embutido, sem marcas/logotipos. Pose neutra, frontal, plano americano."
)


def _ensure_cast_dir(output_dir: Path) -> Path:
    cast_dir = Path(output_dir) / "comic" / "cast"
    cast_dir.mkdir(parents=True, exist_ok=True)
    return cast_dir


def _build_text_card(member: CastMember) -> str:
    """Consolida descritores visíveis em uma ficha textual usada nos prompts."""

    parts: list[str] = []
    if member.gender_apparent:
        parts.append(member.gender_apparent)
    if member.age_apparent:
        parts.append(member.age_apparent)
    if member.hair:
        parts.append(f"cabelo: {member.hair}")
    if member.facial_hair:
        parts.append(f"pelos faciais: {member.facial_hair}")
    if member.skin:
        parts.append(f"pele: {member.skin}")
    if member.clothing:
        parts.append(f"roupa: {member.clothing}")
    if member.accessories:
        parts.append("acessórios: " + ", ".join(member.accessories))
    if member.narrative_role:
        parts.append(f"papel narrativo: {member.narrative_role}")
    if not parts:
        return member.narrative_role or member.kind
    return "; ".join(parts)


def _build_anchor_prompt(member: CastMember, *, has_source_frame: bool = False) -> str:
    """Prompt de imagem-âncora segmentado por ``kind`` (RF-10).

    Quando ``has_source_frame`` é True, o prompt assume que o frame real do
    vídeo será passado como ``reference_image`` ao ``gpt-image-1`` (via
    ``images.edit``) e instrui o modelo a TRANSFORMAR a pessoa observada em
    caricatura, mantendo as feições reais; descritores textuais são apenas
    complementos de desambiguação.
    """

    descriptors: list[str] = []
    if member.gender_apparent:
        descriptors.append(f"gênero aparente {member.gender_apparent}")
    if member.age_apparent:
        descriptors.append(f"idade aparente {member.age_apparent}")
    if member.hair:
        descriptors.append(f"cabelo {member.hair}")
    if member.facial_hair:
        descriptors.append(f"pelos faciais {member.facial_hair}")
    if member.skin:
        descriptors.append(f"pele {member.skin}")
    if member.clothing:
        descriptors.append(f"vestindo {member.clothing}")
    if member.accessories:
        descriptors.append("com acessórios: " + ", ".join(member.accessories))

    descriptor_block = "; ".join(descriptors) if descriptors else ""

    if member.kind == "person":
        if has_source_frame:
            subject = (
                "Transforme a pessoa visível na imagem de referência em uma caricatura "
                "editorial: pose neutra, plano americano, olhar frontal, fundo aquarela "
                "neutro. PRESERVE com fidelidade as feições reais observadas — formato e "
                "proporções do rosto, cor e comprimento exato do cabelo, tipo/densidade/"
                "cor da barba, tom de pele, marcas distintivas, formato de óculos e "
                "demais acessórios visíveis. Pode estilizar (traço caricatural, olhos "
                "expressivos, proporções levemente exageradas), mas a pessoa deve ser "
                "imediatamente reconhecível por quem assiste o vídeo original. "
                f"Descritores complementares: {descriptor_block or 'usar somente o que é visível'}."
            )
        else:
            subject = (
                f"Personagem único em pose neutra, plano americano, olhar frontal. "
                f"Características visíveis: {descriptor_block or 'não especificadas'}."
            )
    elif member.kind == "animal":
        subject = (
            f"Animal ilustrado em pose neutra, frontal, fundo neutro. "
            f"Detalhes: {member.narrative_role or descriptor_block or 'animal genérico'}."
        )
    else:  # object
        subject = (
            f"Objeto narrativo em pose neutra, fundo neutro, sem pessoas. "
            f"Detalhes: {member.narrative_role or descriptor_block or 'objeto genérico'}."
        )

    return (
        f"{subject} {_STYLE_PROMPT} Apenas um personagem central, sem multidão, "
        f"sem fundo detalhado, foco em identidade visual reusável como ficha-âncora."
    )


def _is_existing_anchor(path: Path | None) -> bool:
    return path is not None and Path(path).exists() and Path(path).stat().st_size > 0


def build_cast(
    cast: list[CastMember],
    output_dir: Path,
    config: PipelineConfig,
    *,
    image_provider: ImageProvider,
) -> list[CastMember]:
    """Popula ``anchor_image_path`` e refina ``text_card`` para cada `CastMember`.

    - Idempotente: se a imagem-âncora já existe e o `anchor_image_path` aponta
      para arquivo válido, reaproveita.
    - Lança :class:`ImageGenerationError` se a geração de qualquer ficha-âncora
      falhar após os retries do provider.
    """

    cast_dir = _ensure_cast_dir(output_dir)
    out: list[CastMember] = []

    for member in cast:
        text_card = _build_text_card(member) if not member.text_card.strip() else member.text_card
        anchor_path = Path(cast_dir) / f"{member.character_id}.png"

        existing = member.anchor_image_path if _is_existing_anchor(member.anchor_image_path) else None
        if existing is None and _is_existing_anchor(anchor_path):
            existing = anchor_path

        if existing is not None:
            logger.info(
                "comic.cast_builder: reusando ficha-âncora existente para %s (%s)",
                member.character_id,
                existing,
            )
            out.append(member.model_copy(update={"anchor_image_path": Path(existing), "text_card": text_card}))
            continue

        source_frame: Path | None = None
        if (
            member.kind == "person"
            and member.source_frame_path is not None
            and Path(member.source_frame_path).exists()
            and Path(member.source_frame_path).stat().st_size > 0
        ):
            source_frame = Path(member.source_frame_path)

        prompt = _build_anchor_prompt(member, has_source_frame=source_frame is not None)
        logger.info(
            "comic.cast_builder: gerando ficha-âncora para %s (%s%s)",
            member.character_id,
            member.kind,
            ", com frame de referência" if source_frame else "",
        )
        try:
            png_bytes = image_provider.generate(
                prompt,
                reference_images=[source_frame] if source_frame else None,
                size=DEFAULT_SIZE,
                input_fidelity="high",
            )
        except ImageGenerationError:
            raise
        except Exception as exc:  # provedor pode levantar exceções não-tipadas
            raise ImageGenerationError(
                f"Falha ao gerar ficha-âncora de {member.character_id}: {exc}"
            ) from exc

        if not png_bytes:
            raise ImageGenerationError(
                f"Provider retornou bytes vazios para a ficha-âncora de {member.character_id}."
            )

        anchor_path.write_bytes(png_bytes)
        out.append(
            member.model_copy(
                update={
                    "anchor_image_path": anchor_path,
                    "text_card": text_card,
                }
            )
        )

    return out
