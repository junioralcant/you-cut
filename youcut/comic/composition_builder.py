"""Composition Builder — gera a master composition image pra modo prunaai.

A master composition é UMA imagem 1024×1536 (≈9:16) mostrando os personagens
do cast já posicionados no cenário canônico. Essa imagem vira a 1ª referência
quando o ``PrunaaiAnimationProvider`` cria a animação completa, garantindo:

- posições fixas dos personagens (quem fica onde)
- cenário coerente do início ao fim do vídeo
- estilo visual cartoon consistente

Quando ``config.comic_scene_seed`` é ``None``, faz uma chamada extra ao
Claude pra inferir um cenário coerente a partir da transcrição + cast.
A geração é **idempotente**: se a master já existe no path esperado e
``regenerate=False``, reaproveita.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anthropic

from youcut.comic.panel_renderer import sanitize_brand_mentions
from youcut.comic.providers.images import ImageGenerationError, ImageProvider
from youcut.config import PipelineConfig
from youcut.models import CastMember, TranscriptionResult

logger = logging.getLogger(__name__)


MASTER_FILENAME: str = "_composition_master.png"
MASTER_SIZE: str = "1024x1536"


_SCENE_INFERENCE_TOOL: dict[str, Any] = {
    "name": "infer_scene",
    "description": (
        "Infere um cenário coerente para um motion comic baseado na transcrição "
        "do áudio + cast disponível. O cenário deve ser visualizável, fixo "
        "(mesmo lugar do início ao fim) e adequado ao tom da fala."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scene": {
                "type": "string",
                "description": (
                    "Descrição visual do cenário em pt-BR, 1-3 frases. "
                    "Inclua: ambiente principal, paleta dominante, elementos "
                    "decorativos. Ex.: \"sala de estar minimalista com sofá "
                    "bege, parede pastel azul-claro, planta no canto, luz "
                    "natural suave\"."
                ),
            },
        },
        "required": ["scene"],
    },
}


_SCENE_SYSTEM_PROMPT = (
    "Você é um diretor de arte que escolhe cenários para motion comics. "
    "Dado o áudio + personagens, sugira UM cenário visualmente rico, fixo "
    "(o mesmo do começo ao fim do vídeo) e coerente com o tom da fala. "
    "Em pt-BR, 1-3 frases curtas."
)


def _master_path(output_dir: Path) -> Path:
    return Path(output_dir) / "comic" / "cast" / MASTER_FILENAME


def _format_cast_for_prompt(cast: list[CastMember]) -> str:
    if not cast:
        return "(cast vazio)"
    lines: list[str] = []
    for c in cast:
        snippet = c.text_card or c.narrative_role or c.character_id
        lines.append(f"- `{c.character_id}`: {sanitize_brand_mentions(snippet)}")
    return "\n".join(lines)


def _format_transcription_excerpt(
    transcription: TranscriptionResult, max_chars: int = 800
) -> str:
    text = " ".join(seg.text.strip() for seg in transcription.segments)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text or "(sem fala)"


def infer_scene_seed(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    config: PipelineConfig,
    *,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Pede ao Claude um cenário coerente quando o usuário não forneceu ``--scene``."""

    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    user_text = (
        f"Transcrição:\n{_format_transcription_excerpt(transcription)}\n\n"
        f"Cast:\n{_format_cast_for_prompt(cast)}\n\n"
        "Sugira UM cenário visualmente rico, fixo, adequado a esse áudio + "
        "personagens."
    )

    response = client.with_options(timeout=60.0).messages.create(
        model=config.claude_model,
        max_tokens=512,
        system=_SCENE_SYSTEM_PROMPT,
        tools=[_SCENE_INFERENCE_TOOL],
        tool_choice={"type": "tool", "name": "infer_scene"},
        messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    )
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "infer_scene"
        ):
            payload = getattr(block, "input", None) or {}
            scene = str(payload.get("scene", "")).strip()
            if scene:
                return scene
    raise ImageGenerationError(
        "Claude não retornou um cenário inferido — passe --scene manualmente."
    )


_MASTER_STYLE_PROMPT = (
    "Estilo OBRIGATÓRIO: caricatura cartoon flat 2D pastel, contorno preto único "
    "espesso (~3px), cores chapadas com sombras simples, OLHOS ENORMES E "
    "REDONDOS com pupila preta marcada (super-deformed big-eyes), cabeças "
    "grandes com corpos pequenos. Paleta pastel suave. PROIBIDO incluir "
    "QUALQUER texto, letras, palavras, balões de fala, legendas, números, "
    "símbolos tipográficos ou estampas com palavras dentro da imagem. Sem "
    "multidão, sem marcas/logos visíveis."
)


def _build_master_prompt(cast: list[CastMember], scene_seed: str) -> str:
    cast_descriptions: list[str] = []
    for member in cast:
        snippet = sanitize_brand_mentions(
            member.text_card or member.narrative_role or member.character_id
        )
        cast_descriptions.append(f"- {member.character_id}: {snippet}")
    cast_block = "\n".join(cast_descriptions) or "(cast vazio)"

    safe_scene = sanitize_brand_mentions(scene_seed)

    return (
        f"Painel ilustrado em proporção 9:16 — composição MASTER de cena. "
        f"Cenário: {safe_scene}. "
        f"Personagens em quadro (CRÍTICO — respeite TODAS as características "
        f"visíveis e mantenha posições espaciais fixas):\n{cast_block}\n"
        f"Vista 3/4 com todos os personagens enquadrados juntos, em pose "
        f"natural relaxada (não falando ainda — esta é a pose-base que vai "
        f"servir como ancoragem para a animação). "
        f"{_MASTER_STYLE_PROMPT}"
    )


def build_master_composition(
    cast: list[CastMember],
    transcription: TranscriptionResult,
    config: PipelineConfig,
    output_dir: Path,
    *,
    image_provider: ImageProvider,
    regenerate: bool = False,
    anthropic_client: anthropic.Anthropic | None = None,
) -> tuple[Path, str]:
    """Constrói (ou reaproveita) a master composition image.

    Retorna ``(master_path, scene_seed_usado)``. Idempotente: se o arquivo
    em ``output_dir/comic/cast/_composition_master.png`` já existe e
    ``regenerate=False``, reaproveita.
    """

    if not cast:
        raise ImageGenerationError(
            "Cast vazio: master composition exige pelo menos 1 personagem."
        )

    out_path = _master_path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scene_seed = config.comic_scene_seed
    if not scene_seed:
        logger.info("comic.composition_builder: inferindo scene via Claude (sem --scene)")
        scene_seed = infer_scene_seed(
            transcription, cast, config, client=anthropic_client
        )
        logger.info("comic.composition_builder: scene inferido = %r", scene_seed[:120])

    if out_path.exists() and not regenerate:
        logger.info("comic.composition_builder: master já existe em %s — reuso", out_path)
        return out_path, scene_seed

    references: list[Path] = []
    for member in cast:
        if member.anchor_image_path is None:
            continue
        anchor = Path(member.anchor_image_path)
        if anchor.exists():
            references.append(anchor)

    prompt = _build_master_prompt(cast, scene_seed)
    logger.info(
        "comic.composition_builder: gerando master (cast=%d, refs=%d)",
        len(cast),
        len(references),
    )
    png_bytes = image_provider.generate(
        prompt,
        reference_images=references or None,
        size=MASTER_SIZE,
        input_fidelity="high",
    )
    out_path.write_bytes(png_bytes)
    logger.info("comic.composition_builder: master salvo em %s", out_path)
    return out_path, scene_seed
