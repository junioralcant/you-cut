"""Metadata Generator — gera título, descrição e hashtags por plataforma.

Final stage opcional do pipeline `youcut comic`. Usa Claude com a transcrição
+ contexto do cast/cenário pra produzir 3 conjuntos editoriais (TikTok,
Instagram Reels, YouTube Shorts) com convenções específicas:

- **TikTok**: títulos curtos com gancho viral, 12-20 hashtags incluindo #fyp,
  emojis abundantes.
- **Instagram Reels**: descrição mais leve, 5-10 hashtags, tom amigável.
- **YouTube Shorts**: título SEO-friendly, descrição mais longa (~200 chars),
  inclui #shorts obrigatoriamente.

Saída em ``output/<video>/comic/metadata.json`` (estruturado) +
``metadata.txt`` (legível para copy-paste).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from youcut.config import PipelineConfig
from youcut.models import (
    CastMember,
    ComicMetadata,
    PlatformMetadata,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class MetadataGenerationError(Exception):
    """Falha ao gerar metadados (Claude indisponível ou JSON inválido)."""


_METADATA_TOOL: dict[str, Any] = {
    "name": "generate_platform_metadata",
    "description": (
        "Gera título + descrição + hashtags para 3 plataformas de vídeo "
        "vertical curto: TikTok, Instagram Reels e YouTube Shorts. Cada "
        "plataforma tem convenções editoriais distintas que devem ser "
        "respeitadas."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Resumo curto (1-2 frases) do conteúdo do vídeo, em pt-BR. "
                    "Usado como contexto interno e exibido no terminal."
                ),
            },
            "tiktok": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Título/gancho viral, ≤80 caracteres, em pt-BR, "
                            "com emoji forte e hook que gere curiosidade ou "
                            "comédia. Ex.: \"Quando o amigo NÃO presta atenção 😂\""
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Descrição/legenda completa, 100-300 caracteres, "
                            "tom coloquial pt-BR, incluir 1-2 perguntas que "
                            "engajem comentários (ex.: \"Marca o amigo que…\"), "
                            "emojis abundantes mas equilibrados."
                        ),
                    },
                    "hashtags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 12,
                        "maxItems": 20,
                        "description": (
                            "12-20 hashtags em lowercase sem espaços, incluindo "
                            "obrigatoriamente #fyp e #foryou; mix de hashtags "
                            "amplas (#humor, #brasileiro) com nicho relacionadas "
                            "ao conteúdo. Sem o caractere #."
                        ),
                    },
                },
                "required": ["title", "description", "hashtags"],
            },
            "instagram_reels": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Título/primeira linha, ≤60 caracteres, em pt-BR, "
                            "tom mais amigável que TikTok, 1 emoji."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Caption Instagram: 80-200 caracteres, tom natural, "
                            "1 chamada à ação suave, emojis moderados (2-4)."
                        ),
                    },
                    "hashtags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 5,
                        "maxItems": 10,
                        "description": (
                            "5-10 hashtags em lowercase sem #, foco em "
                            "descoberta no Reels — incluir #reels e mix de "
                            "amplas/nicho."
                        ),
                    },
                },
                "required": ["title", "description", "hashtags"],
            },
            "youtube_shorts": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "Título SEO-friendly em pt-BR, 40-80 caracteres, "
                            "frase clara descrevendo o conteúdo (não só hook). "
                            "Sem ALL CAPS. Pode ter 1 emoji no início OU final."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Descrição YouTube: 150-400 caracteres, contexto "
                            "claro do clipe, 1-2 calls-to-action (inscreva-se, "
                            "comente). Termina com bloco de hashtags entre #."
                        ),
                    },
                    "hashtags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 5,
                        "maxItems": 12,
                        "description": (
                            "5-12 hashtags em lowercase sem #. OBRIGATÓRIO "
                            "incluir 'shorts' como primeira hashtag. Outras: "
                            "termos de busca em pt-BR relacionados ao conteúdo."
                        ),
                    },
                },
                "required": ["title", "description", "hashtags"],
            },
        },
        "required": ["summary", "tiktok", "instagram_reels", "youtube_shorts"],
    },
}


_SYSTEM_PROMPT = (
    "Você é um especialista em copy de redes sociais brasileiras (pt-BR), "
    "focado em vídeos verticais curtos. Seu trabalho é criar título, "
    "descrição e hashtags otimizados para cada plataforma a partir do "
    "conteúdo de uma animação cartoon. Respeite as convenções específicas "
    "de cada plataforma: TikTok é mais agressivo/viral, Instagram Reels é "
    "mais leve/amigável, YouTube Shorts é mais SEO-friendly e descritivo. "
    "Use linguagem coloquial brasileira, emojis estratégicos, e hooks que "
    "gerem engajamento (curiosidade, identificação, humor). NUNCA use "
    "linguagem ofensiva, política polarizante ou marcas registradas."
)


def _format_transcription(transcription: TranscriptionResult) -> str:
    if not transcription.segments:
        return "(sem fala)"
    return "\n".join(seg.text.strip() for seg in transcription.segments)


def _format_cast(cast: list[CastMember]) -> str:
    if not cast:
        return "(sem cast)"
    lines = []
    for c in cast:
        snippet = c.text_card or c.narrative_role or c.character_id
        lines.append(f"- {c.character_id}: {snippet[:160]}")
    return "\n".join(lines)


def _build_user_prompt(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    scene_seed: str | None,
) -> str:
    parts = [
        "Gere os metadados editoriais para a animação descrita abaixo.\n",
        f"TRANSCRIÇÃO DO ÁUDIO:\n{_format_transcription(transcription)}\n",
        f"\nCAST DA ANIMAÇÃO:\n{_format_cast(cast)}\n",
    ]
    if scene_seed:
        parts.append(f"\nCENÁRIO FIXO:\n{scene_seed}\n")
    parts.append(
        "\nProduza title + description + hashtags para TikTok, Instagram "
        "Reels e YouTube Shorts respeitando as convenções de cada plataforma."
    )
    return "".join(parts)


def _parse_platform(raw: dict[str, Any], platform: str) -> PlatformMetadata:
    title = str(raw.get("title", "")).strip()
    description = str(raw.get("description", "")).strip()
    hashtags_raw = raw.get("hashtags") or []
    hashtags = [
        str(h).strip().lstrip("#").lower().replace(" ", "")
        for h in hashtags_raw
        if str(h).strip()
    ]
    if not title:
        raise MetadataGenerationError(f"{platform}: título vazio")
    if not description:
        raise MetadataGenerationError(f"{platform}: descrição vazia")
    return PlatformMetadata(
        platform=platform,
        title=title,
        description=description,
        hashtags=hashtags,
    )


def generate_metadata(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    config: PipelineConfig,
    *,
    scene_seed: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> ComicMetadata:
    """Gera metadados multi-plataforma para o vídeo final.

    Faz 1 chamada ao Claude com a transcrição + cast + scene_seed. Retorna
    :class:`ComicMetadata` com 3 :class:`PlatformMetadata` (tiktok,
    instagram_reels, youtube_shorts). Custo estimado ~$0.01/run.
    """

    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    user_text = _build_user_prompt(transcription, cast, scene_seed)

    try:
        response = client.with_options(timeout=60.0).messages.create(
            model=config.claude_model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_METADATA_TOOL],
            tool_choice={"type": "tool", "name": "generate_platform_metadata"},
            messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        )
    except anthropic.APIError as exc:
        msg = getattr(exc, "message", None) or str(exc)
        raise MetadataGenerationError(f"Erro na API do Claude: {msg}") from exc

    payload: dict[str, Any] | None = None
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "generate_platform_metadata"
        ):
            payload = getattr(block, "input", None)
            break
    if not payload:
        raise MetadataGenerationError(
            "Resposta do Claude não contém chamada à ferramenta esperada."
        )

    return ComicMetadata(
        summary=str(payload.get("summary", "")).strip(),
        tiktok=_parse_platform(payload.get("tiktok") or {}, "tiktok"),
        instagram_reels=_parse_platform(
            payload.get("instagram_reels") or {}, "instagram_reels"
        ),
        youtube_shorts=_parse_platform(
            payload.get("youtube_shorts") or {}, "youtube_shorts"
        ),
    )


def write_metadata_files(
    metadata: ComicMetadata, output_dir: Path
) -> tuple[Path, Path]:
    """Escreve ``metadata.json`` (estruturado) e ``metadata.txt`` (copy-paste)."""

    comic_dir = Path(output_dir) / "comic"
    comic_dir.mkdir(parents=True, exist_ok=True)
    json_path = comic_dir / "metadata.json"
    txt_path = comic_dir / "metadata.txt"

    json_path.write_text(
        metadata.model_dump_json(indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    if metadata.summary:
        lines.append(f"# Resumo\n{metadata.summary}\n")
    for platform in (metadata.tiktok, metadata.instagram_reels, metadata.youtube_shorts):
        label = platform.platform.replace("_", " ").title()
        lines.append(f"\n## {label}\n")
        lines.append(f"**Título:** {platform.title}\n")
        lines.append(f"**Descrição:**\n{platform.description}\n")
        tags = " ".join(f"#{h}" for h in platform.hashtags)
        lines.append(f"**Hashtags:**\n{tags}\n")
    txt_path.write_text("".join(lines), encoding="utf-8")

    return json_path, txt_path
