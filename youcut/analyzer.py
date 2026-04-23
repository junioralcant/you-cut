import anthropic

from youcut.config import PipelineConfig
from youcut.models import TranscriptionResult, ViralClip

CHUNK_DURATION = 30 * 60  # 30 minutes in seconds
MIN_CLIP_DURATION = 15
MAX_CLIP_DURATION = 60

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
                    ],
                },
                "description": "Lista de clipes virais identificados na transcrição",
            }
        },
        "required": ["clips"],
    },
}

_SYSTEM_PROMPT = """\
Você é um especialista em criação de conteúdo viral para redes sociais (Shorts, Reels, TikTok).

Analise a transcrição fornecida e identifique os melhores trechos com potencial viral, \
seguindo os critérios abaixo:

1. Gancho forte no início — o trecho começa com algo que prende atenção imediatamente
2. Conteúdo impactante, curiosidade ou opinião forte — algo que provoca reação emocional
3. Dica prática, humor ou emoção — conteúdo que agrega valor real ou entretém
4. Momentos de alta intensidade — picos de energia, surpresa ou emoção elevada
5. Trecho que faça sentido assistido isoladamente — sem precisar de contexto externo

REGRAS OBRIGATÓRIAS:
- Cada clipe deve ter entre 15 e 60 segundos de duração (end_time - start_time)
- Priorize os trechos com maior potencial de engajamento
- O viral_score deve ser um número de 0 a 10
- Retorne apenas os timestamps precisos encontrados na transcrição
"""

_USER_PROMPT_PREFIX = "Identifique os clipes virais na seguinte transcrição:\n\n"


def _format_transcription(segments) -> str:
    lines = [f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}" for seg in segments]
    return "\n".join(lines)


def _analyze_chunk(
    client: anthropic.Anthropic,
    segments: list,
    config: PipelineConfig,
) -> list[ViralClip]:
    transcription_text = _format_transcription(segments)

    try:
        response = client.with_options(timeout=120.0).messages.create(
            model=config.claude_model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
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
                            "text": _USER_PROMPT_PREFIX + transcription_text,
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
                    clip = ViralClip(**raw)
                    duration = clip.end_time - clip.start_time
                    if MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
                        clips.append(clip)
                except Exception:
                    continue
    return clips


def analyze(transcription: TranscriptionResult, config: PipelineConfig) -> list[ViralClip]:
    """Analyze transcription and return ViralClip list sorted by viral_score descending."""
    if not transcription.segments:
        return []

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    segments = transcription.segments
    total_duration = segments[-1].end

    if total_duration <= CHUNK_DURATION:
        clips = _analyze_chunk(client, segments, config)
    else:
        clips = []
        chunk_start = 0.0
        while chunk_start < total_duration:
            chunk_end = chunk_start + CHUNK_DURATION
            chunk_segs = [s for s in segments if chunk_start <= s.start < chunk_end]
            if chunk_segs:
                clips.extend(_analyze_chunk(client, chunk_segs, config))
            chunk_start = chunk_end

    clips.sort(key=lambda c: c.viral_score, reverse=True)
    return clips
