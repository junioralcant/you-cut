import logging
import re

import anthropic

from youcut.config import PipelineConfig
from youcut.models import CutMode, TranscriptionResult, ViralClip

CHUNK_DURATION = 30 * 60  # 30 minutes in seconds

SOCIAL_MIN_DURATION = 15
SOCIAL_MAX_DURATION = 180

YOUTUBE_MIN_DURATION = 900
YOUTUBE_MAX_DURATION = 1500
YOUTUBE_FALLBACK_MIN_DURATION = 300

# Backward-compat aliases — social is the legacy default
MIN_CLIP_DURATION = SOCIAL_MIN_DURATION
MAX_CLIP_DURATION = SOCIAL_MAX_DURATION

YOUTUBE_TITLE_MIN_WORDS = 5
YOUTUBE_TITLE_MAX_WORDS = 9
YOUTUBE_TITLE_IDEAL_MAX_CHARS = 30
THUMBNAIL_TEXT_MAX_WORDS = 6
SOCIAL_HOOK_MAX_WORDS = 6
_SOCIAL_VISUAL_STYLE_DEFAULT = "editorial claro e vivo, alto contraste, sem texto embutido"

logger = logging.getLogger(__name__)


def _get_duration_limits(cut_mode: CutMode) -> tuple[int, int]:
    if cut_mode == "youtube":
        return YOUTUBE_MIN_DURATION, YOUTUBE_MAX_DURATION
    return SOCIAL_MIN_DURATION, SOCIAL_MAX_DURATION


def _is_youtube_ideal_duration(duration: float) -> bool:
    return YOUTUBE_MIN_DURATION <= duration <= YOUTUBE_MAX_DURATION


def _is_valid_duration(cut_mode: CutMode, duration: float) -> bool:
    if cut_mode == "youtube":
        return YOUTUBE_FALLBACK_MIN_DURATION <= duration <= YOUTUBE_MAX_DURATION
    return SOCIAL_MIN_DURATION <= duration <= SOCIAL_MAX_DURATION


def _clip_sort_key(clip: ViralClip) -> tuple[int, float]:
    duration = clip.end_time - clip.start_time
    if clip.cut_mode == "youtube":
        priority = 0 if _is_youtube_ideal_duration(duration) else 1
        return priority, -clip.viral_score
    return 0, -clip.viral_score


def _build_system_prompt(cut_mode: CutMode, min_dur: int, max_dur: int) -> str:
    if cut_mode == "youtube":
        audience = "YouTube (vídeos longos em paisagem 16:9)"
        style = "informativos, aprofundados e com começo, meio e fim bem definidos"
        duration_rule = (
            f"idealmente entre {min_dur} e {max_dur} segundos; "
            f"se não houver trechos fortes nessa faixa, você pode retornar clipes menores, "
            f"desde que tenham pelo menos {YOUTUBE_FALLBACK_MIN_DURATION} segundos e ainda "
            "façam sentido como vídeo completo para YouTube"
        )
        title_rule = (
            f"- O título de cada clipe deve ter idealmente entre {YOUTUBE_TITLE_MIN_WORDS} e "
            f"{YOUTUBE_TITLE_MAX_WORDS} palavras\n"
            f"- Prefira títulos com até {YOUTUBE_TITLE_IDEAL_MAX_CHARS} caracteres, mas pode "
            "ultrapassar quando isso deixar o título mais claro e natural\n"
            "- Evite clickbait genérico e priorize clareza editorial"
        )
    else:
        audience = "redes sociais (Shorts, Reels, TikTok)"
        style = "virais e de alto impacto"
        duration_rule = f"entre {min_dur} e {max_dur} segundos"
        title_rule = "- O título deve ser curto, chamativo e adequado para redes sociais"
    thumbnail_text_rule = (
        "- Gere também o campo thumbnail_text para cada clipe\n"
        f"- O thumbnail_text deve ter no máximo {THUMBNAIL_TEXT_MAX_WORDS} palavras\n"
        "- O thumbnail_text deve ser independente do título e claramente diferente dele\n"
        "- Não reaproveite a mesma estrutura, mesma abertura ou as mesmas palavras principais do título\n"
        "- O thumbnail_text deve ser curto, editorial, impactante e baseado no tema central do trecho\n"
        "- Evite frases genéricas, clickbait vazio, aspas, emojis e pontuação desnecessária\n"
        "- Pense no thumbnail_text como texto embutido na thumbnail, não como título do vídeo"
    )
    social_rule = ""
    if cut_mode == "social":
        social_rule = (
            "- Gere também social_hook_title: um hook curto para a tarja fixa do clipe, com no máximo "
            f"{SOCIAL_HOOK_MAX_WORDS} palavras\n"
            "- Gere também social_image_prompt: um prompt visual editorial para a imagem do topo, coerente com o tema do trecho\n"
            "- Gere também social_visual_style: direção visual curta para reforçar look claro, vivo e de alto contraste\n"
            "- O social_hook_title deve ser independente do thumbnail_text\n"
            "- O social_image_prompt deve evitar texto embutido na imagem\n"
        )

    return f"""\
Você é um especialista em criação de conteúdo {style} para {audience}.

Analise a transcrição fornecida e identifique os melhores trechos com potencial \
de engajamento, seguindo os critérios abaixo:

1. Gancho forte no início — o trecho começa com algo que prende atenção imediatamente
2. Conteúdo impactante, curiosidade ou opinião forte — algo que provoca reação emocional
3. Dica prática, humor ou emoção — conteúdo que agrega valor real ou entretém
4. Momentos de alta intensidade — picos de energia, surpresa ou emoção elevada
5. Trecho que faça sentido assistido isoladamente — sem precisar de contexto externo

REGRAS OBRIGATÓRIAS:
- Cada clipe deve ter {duration_rule} de duração (end_time - start_time)
- Priorize os trechos com maior potencial de engajamento
- O viral_score deve ser um número de 0 a 10
- Retorne apenas os timestamps precisos encontrados na transcrição
- Os clipes NÃO devem se sobrepor (sem repetição de conteúdo entre clipes)
{title_rule}
{thumbnail_text_rule}
{social_rule}
"""


# Social-mode default — kept as module-level export for backward compatibility
_SYSTEM_PROMPT = _build_system_prompt("social", SOCIAL_MIN_DURATION, SOCIAL_MAX_DURATION)

_VIRAL_TOOL = {
    "name": "identify_viral_clips",
    "description": (
        "Identifica os melhores trechos virais da transcrição e retorna uma lista de clipes "
        "com metadados completos para publicação nas redes sociais."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": (
                                "Título do clipe. Em modo youtube: título editorial claro, idealmente "
                                f"com {YOUTUBE_TITLE_MIN_WORDS} a {YOUTUBE_TITLE_MAX_WORDS} palavras "
                                f"e preferencialmente até {YOUTUBE_TITLE_IDEAL_MAX_CHARS} caracteres, "
                                "podendo ultrapassar esse teto quando necessário. "
                                "Em modo social: título curto e chamativo."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "Motivo da escolha (critério de viralidade identificado)",
                        },
                        "viral_score": {
                            "type": "number",
                            "description": "Nota de potencial viral de 0 a 10",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Timestamp de início em segundos",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Timestamp de fim em segundos",
                        },
                        "description": {
                            "type": "string",
                            "description": "Descrição pronta para publicação nas redes sociais",
                        },
                        "hashtags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Hashtags relevantes para o clipe",
                        },
                        "thumbnail_idea": {
                            "type": "string",
                            "description": "Descrição textual do frame mais impactante para thumbnail",
                        },
                        "thumbnail_text": {
                            "type": "string",
                            "description": (
                                "Texto curto para embutir na thumbnail, seguindo o PRD: "
                                f"máximo de {THUMBNAIL_TEXT_MAX_WORDS} palavras, independente do título e diferente dele, "
                                "baseado no tema central do clipe, editorial e impactante, sem clickbait genérico. "
                                "Ex: 'CRISE NA DIREITA', 'MBL EM CHOQUE', 'PRESSÃO NO STF'"
                            ),
                        },
                        "social_hook_title": {
                            "type": "string",
                            "description": (
                                "Hook curto para a tarja fixa do clipe social. Máximo de "
                                f"{SOCIAL_HOOK_MAX_WORDS} palavras, direto e independente do thumbnail_text."
                            ),
                        },
                        "social_image_prompt": {
                            "type": "string",
                            "description": (
                                "Prompt editorial para a imagem superior do clipe social, coerente com o tema do trecho "
                                "e sem texto embutido."
                            ),
                        },
                        "social_visual_style": {
                            "type": "string",
                            "description": (
                                "Direção visual curta do clipe social. Preferir look claro, vivo, editorial e de alto contraste."
                            ),
                        },
                    },
                    "required": [
                        "title",
                        "reason",
                        "viral_score",
                        "start_time",
                        "end_time",
                        "description",
                        "hashtags",
                        "thumbnail_idea",
                        "thumbnail_text",
                    ],
                },
                "description": "Lista de clipes virais identificados na transcrição",
            }
        },
        "required": ["clips"],
    },
}

_USER_PROMPT_PREFIX = "Identifique os clipes virais na seguinte transcrição:\n\n"


def _format_transcription(segments) -> str:
    lines = [f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}" for seg in segments]
    return "\n".join(lines)


def _remove_overlapping(clips: list[ViralClip]) -> list[ViralClip]:
    """Remove clips that overlap with a higher-scored clip. Input must be sorted by score desc."""
    kept: list[ViralClip] = []
    for clip in clips:
        if not any(
            clip.start_time < k.end_time and clip.end_time > k.start_time for k in kept
        ):
            kept.append(clip)
    return kept


def _build_user_prompt(segments, max_clips: int | None) -> str:
    transcription_text = _format_transcription(segments)
    prefix = _USER_PROMPT_PREFIX
    if max_clips is not None:
        prefix = f"Retorne no máximo {max_clips} clipes.\n\n" + prefix
    return prefix + transcription_text


def _normalize_clip_title(title: str) -> str:
    return " ".join(title.split())


def _normalize_thumbnail_token(token: str) -> str:
    token = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", token, flags=re.UNICODE)
    return token


def _derive_thumbnail_text(title: str, reason: str = "", thumbnail_idea: str = "") -> str:
    source = reason or thumbnail_idea or ""
    stopwords = {
        "A", "AS", "O", "OS", "DE", "DA", "DO", "DAS", "DOS", "E", "EM", "NA", "NO",
        "NAS", "NOS", "PARA", "POR", "COM", "SEM", "UM", "UMA", "SOBRE", "QUE",
    }
    tokens = [_normalize_thumbnail_token(token).upper() for token in source.split()]
    filtered = [token for token in tokens if token and token not in stopwords]
    if not filtered:
        filtered = ["MOMENTO", "EM", "DESTAQUE"]
    if not filtered:
        return "MOMENTO EM DESTAQUE"
    return " ".join(filtered[:THUMBNAIL_TEXT_MAX_WORDS])


def _thumbnail_text_too_similar_to_title(raw_text: str, normalized_title: str) -> bool:
    raw_tokens = [token for token in raw_text.split() if token]
    title_tokens = [token for token in normalized_title.split() if token]
    if not raw_tokens or not title_tokens:
        return False

    raw_set = set(raw_tokens)
    title_set = set(title_tokens)
    overlap_ratio = len(raw_set & title_set) / max(1, min(len(raw_set), len(title_set)))
    if raw_text == normalized_title:
        return True
    if raw_text.startswith(title_tokens[0]) and overlap_ratio >= 0.5:
        return True
    return overlap_ratio >= 0.8


def _normalize_thumbnail_text(text: str, title: str, reason: str = "", thumbnail_idea: str = "") -> str:
    tokens = [_normalize_thumbnail_token(token).upper() for token in text.split()]
    tokens = [token for token in tokens if token]
    raw_normalized = " ".join(tokens).strip()
    normalized = " ".join(tokens[:THUMBNAIL_TEXT_MAX_WORDS]).strip()

    title_tokens = [_normalize_thumbnail_token(token).upper() for token in title.split()]
    normalized_title = " ".join(token for token in title_tokens if token)

    if not normalized or _thumbnail_text_too_similar_to_title(raw_normalized, normalized_title):
        return _derive_thumbnail_text(title, reason, thumbnail_idea)
    return normalized


def _count_words(text: str) -> int:
    return len([word for word in text.split(" ") if word])


def _derive_social_hook_title(title: str, reason: str = "") -> str:
    source = title or reason or "MOMENTO EM DESTAQUE"
    tokens = [_normalize_thumbnail_token(token).upper() for token in source.split()]
    filtered = [token for token in tokens if token]
    return " ".join(filtered[:SOCIAL_HOOK_MAX_WORDS]).strip() or "MOMENTO EM DESTAQUE"


def _normalize_social_hook_title(text: str, title: str, reason: str = "") -> str:
    raw = _derive_social_hook_title(text, reason) if text.strip() else _derive_social_hook_title(title, reason)
    tokens = [_normalize_thumbnail_token(token).upper() for token in raw.split()]
    filtered = [token for token in tokens if token]
    return " ".join(filtered[:SOCIAL_HOOK_MAX_WORDS]).strip() or "MOMENTO EM DESTAQUE"


def _normalize_social_image_prompt(
    prompt: str,
    *,
    title: str,
    reason: str,
    thumbnail_idea: str,
    description: str,
) -> str:
    normalized = " ".join(prompt.split()).strip()
    if normalized:
        return normalized
    base_parts = [title.strip(), reason.strip(), thumbnail_idea.strip(), description.strip()]
    base = ". ".join(part for part in base_parts if part)
    return (
        f"{base}. Ilustração editorial forte, limpa, clara e viva, alto contraste, sem texto, "
        "poucos elementos, leitura imediata em tela pequena."
    ).strip()


def _normalize_social_visual_style(style: str) -> str:
    normalized = " ".join(style.split()).strip()
    return normalized or _SOCIAL_VISUAL_STYLE_DEFAULT


def _log_title_guidance(clip: ViralClip) -> None:
    if clip.cut_mode != "youtube":
        return

    word_count = _count_words(clip.title)
    char_count = len(clip.title)

    if (
        YOUTUBE_TITLE_MIN_WORDS <= word_count <= YOUTUBE_TITLE_MAX_WORDS
        and char_count <= YOUTUBE_TITLE_IDEAL_MAX_CHARS
    ):
        return

    logger.info(
        (
            "Título fora da faixa ideal para YouTube longo: '%s' "
            "(%d palavras, %d caracteres; alvo %d-%d palavras, ideal até %d caracteres)"
        ),
        clip.title,
        word_count,
        char_count,
        YOUTUBE_TITLE_MIN_WORDS,
        YOUTUBE_TITLE_MAX_WORDS,
        YOUTUBE_TITLE_IDEAL_MAX_CHARS,
    )


def _analyze_chunk(
    client: anthropic.Anthropic,
    segments: list,
    config: PipelineConfig,
    cut_mode: CutMode,
    max_clips: int | None,
) -> list[ViralClip]:
    min_dur, max_dur = _get_duration_limits(cut_mode)
    system_prompt = _build_system_prompt(cut_mode, min_dur, max_dur)
    user_text = _build_user_prompt(segments, max_clips)

    try:
        response = client.with_options(timeout=120.0).messages.create(
            model=config.claude_model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_VIRAL_TOOL],
            tool_choice={"type": "tool", "name": "identify_viral_clips"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        msg = e.message if hasattr(e, "message") else str(e)
        raise RuntimeError(
            f"Erro na API do Claude ao analisar transcrição: {msg}"
        ) from e

    clips: list[ViralClip] = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "identify_viral_clips":
            for raw in block.input.get("clips", []):
                try:
                    raw.pop("cut_mode", None)
                    raw["title"] = _normalize_clip_title(raw.get("title", ""))
                    raw["thumbnail_text"] = _normalize_thumbnail_text(
                        raw.get("thumbnail_text", ""),
                        raw["title"],
                        raw.get("reason", ""),
                        raw.get("thumbnail_idea", ""),
                    )
                    raw["social_hook_title"] = _normalize_social_hook_title(
                        raw.get("social_hook_title", ""),
                        raw["title"],
                        raw.get("reason", ""),
                    )
                    raw["social_image_prompt"] = _normalize_social_image_prompt(
                        raw.get("social_image_prompt", ""),
                        title=raw["title"],
                        reason=raw.get("reason", ""),
                        thumbnail_idea=raw.get("thumbnail_idea", ""),
                        description=raw.get("description", ""),
                    )
                    raw["social_visual_style"] = _normalize_social_visual_style(
                        raw.get("social_visual_style", "")
                    )
                    clip = ViralClip(**raw, cut_mode=cut_mode)
                    duration = clip.end_time - clip.start_time
                    if _is_valid_duration(cut_mode, duration):
                        _log_title_guidance(clip)
                        clips.append(clip)
                except Exception:
                    continue
    return clips


def analyze(transcription: TranscriptionResult, config: PipelineConfig) -> list[ViralClip]:
    """Analyze transcription and return ViralClip list sorted by viral_score descending."""
    if not transcription.segments:
        return []

    cut_mode: CutMode = config.cut_mode
    max_clips: int | None = config.max_clips

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    segments = transcription.segments
    total_duration = segments[-1].end

    if total_duration <= CHUNK_DURATION:
        clips = _analyze_chunk(client, segments, config, cut_mode, max_clips)
    else:
        clips = []
        chunk_start = 0.0
        while chunk_start < total_duration:
            chunk_end = chunk_start + CHUNK_DURATION
            chunk_segs = [s for s in segments if chunk_start <= s.start < chunk_end]
            if chunk_segs:
                clips.extend(_analyze_chunk(client, chunk_segs, config, cut_mode, max_clips))
            chunk_start = chunk_end

    clips.sort(key=_clip_sort_key)
    clips = _remove_overlapping(clips)

    if max_clips is not None:
        clips = clips[:max_clips]

    return clips
