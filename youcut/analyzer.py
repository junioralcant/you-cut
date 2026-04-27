import anthropic

from youcut.config import PipelineConfig
from youcut.models import CutMode, TranscriptionResult, ViralClip

CHUNK_DURATION = 30 * 60  # 30 minutes in seconds

SOCIAL_MIN_DURATION = 15
SOCIAL_MAX_DURATION = 180

YOUTUBE_MIN_DURATION = 300
YOUTUBE_MAX_DURATION = 1200

# Backward-compat aliases — social is the legacy default
MIN_CLIP_DURATION = SOCIAL_MIN_DURATION
MAX_CLIP_DURATION = SOCIAL_MAX_DURATION


def _get_duration_limits(cut_mode: CutMode) -> tuple[int, int]:
    if cut_mode == "youtube":
        return YOUTUBE_MIN_DURATION, YOUTUBE_MAX_DURATION
    return SOCIAL_MIN_DURATION, SOCIAL_MAX_DURATION


def _build_system_prompt(cut_mode: CutMode, min_dur: int, max_dur: int) -> str:
    if cut_mode == "youtube":
        audience = "YouTube (vídeos longos em paisagem 16:9)"
        style = "informativos, aprofundados e com começo, meio e fim bem definidos"
    else:
        audience = "redes sociais (Shorts, Reels, TikTok)"
        style = "virais e de alto impacto"

    duration_rule = f"entre {min_dur} e {max_dur} segundos"

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
                            "description": "Título chamativo estilo redes sociais",
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
                                "Frase curta e impactante para exibir na thumbnail — "
                                "MÁXIMO 6 palavras, estilo chamativo/impactante, "
                                "DIFERENTE do título, baseada no tema central do clipe. "
                                "Ex: 'IA VAI DOMINAR O MUNDO', 'SEGREDO QUE NINGUÉM CONTA'"
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
                    clip = ViralClip(**raw, cut_mode=cut_mode)
                    duration = clip.end_time - clip.start_time
                    if min_dur <= duration <= max_dur:
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

    clips.sort(key=lambda c: c.viral_score, reverse=True)
    clips = _remove_overlapping(clips)

    if max_clips is not None:
        clips = clips[:max_clips]

    return clips
