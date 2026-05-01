"""Script Planner — divide a transcrição em painéis via Claude texto (RF-12..RF-17).

Invariantes do output:
- cadência: 1 painel a cada 5 s (mín) e 1 a cada 1,5 s (máx);
- duração de painel: entre ``comic_panel_min_seconds`` e ``comic_panel_max_seconds``;
- soma de durações ≈ duração do áudio (±0,2 s);
- painéis não se sobrepõem;
- ``participants`` não vazio (pelo menos um id presente no cast).

Quando o Claude retorna JSON que viola alguma invariante, o módulo tenta uma
segunda chamada com mensagem corretiva citando a violação. Se a 2ª tentativa
ainda falhar, levanta :class:`ScriptPlanError`.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from pydantic import ValidationError

from youcut.config import PipelineConfig
from youcut.models import (
    CastMember,
    Panel,
    SpeakerSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


CADENCE_MIN_SECONDS_PER_PANEL: float = 1.5
CADENCE_MAX_SECONDS_PER_PANEL: float = 5.0
DURATION_TOLERANCE_SECONDS: float = 0.2

# Para snap de transição em pausa natural da fala
SPEECH_GAP_MIN_SECONDS: float = 0.10  # silêncio mínimo pra contar como gap (micro-pausas)
SPEECH_SNAP_WINDOW_SECONDS: float = 0.6  # janela ±0.6s pra procurar gap próximo


class ScriptPlanError(Exception):
    """Falha no planejamento do roteiro (Claude indisponível ou inválido)."""


_PLAN_TOOL: dict[str, Any] = {
    "name": "plan_panels",
    "description": (
        "Divide a transcrição em painéis ilustrados, cada um com timestamps, "
        "participantes (ids do cast fornecido), enquadramento, cenário inferido "
        "e descrição de pose/expressão dominante."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "participants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ids do cast presentes no painel.",
                        },
                        "framing": {
                            "type": "string",
                            "enum": ["close", "medium", "wide", "two_shot"],
                        },
                        "scene": {"type": "string"},
                        "pose_description": {"type": "string"},
                        "narrative_mode": {
                            "type": "boolean",
                            "description": (
                                "true quando o painel visualiza a cena que está "
                                "sendo NARRADA pelo falante (ex.: criança em "
                                "montanha-russa) em vez de mostrar o falante. "
                                "Quando true, `participants` deve ser [] e "
                                "`narrative_elements` deve listar os elementos "
                                "fictícios da cena."
                            ),
                        },
                        "narrative_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Lista de personagens/objetos fictícios a inventar "
                                "no painel narrativo. Apenas quando narrative_mode=true. "
                                "Ex.: [\"criança chorona em carrinho\", \"montanha-russa "
                                "amarela com olhos arregalados\", \"pai sorrindo no chão\"]."
                            ),
                        },
                    },
                    "required": [
                        "start_time",
                        "end_time",
                        "participants",
                        "framing",
                        "scene",
                        "pose_description",
                    ],
                },
            }
        },
        "required": ["panels"],
    },
}


def _format_cast(cast: list[CastMember]) -> str:
    lines: list[str] = []
    for c in cast:
        snippet = c.text_card or c.narrative_role or c.character_id
        lines.append(f"- id `{c.character_id}` ({c.kind}): {snippet}")
    return "\n".join(lines) or "(cast vazio)"


def _format_transcription(transcription: TranscriptionResult) -> str:
    lines: list[str] = []
    for seg in transcription.segments:
        lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
    return "\n".join(lines)


def _format_speakers(speakers: list[SpeakerSegment]) -> str:
    if not speakers:
        return "(sem diarização)"
    return "; ".join(f"{s.speaker_id}: {s.start:.2f}-{s.end:.2f}" for s in speakers)


def _audio_duration(transcription: TranscriptionResult) -> float:
    if not transcription.segments:
        return 0.0
    return float(transcription.segments[-1].end)


def _is_voiceover_mode(cast: list[CastMember]) -> bool:
    """True quando NENHUM membro do cast está mapeado a um speaker.

    Indica que o falante deve ser tratado como voz em off (nunca em quadro)
    e os personagens disponíveis são reatores/audiência.
    """

    return bool(cast) and all((c.speaker_id is None) for c in cast)


_VOICEOVER_BLOCK = (
    "\n\nMODO VOZ EM OFF (ATIVADO — todo o cast é de REATORES, ninguém fala):\n"
    "- O falante do áudio é VOZ EM OFF e NUNCA aparece em quadro. PROIBIDO "
    "renderizar, descrever ou referenciar o falante visualmente em qualquer "
    "painel.\n"
    "- Painéis com `narrative_mode=false` mostram UM ou DOIS reatores do cast "
    "(audiência) reagindo ao que está sendo dito naquele instante. Use close-up "
    "ou plano médio. O `pose_description` deve descrever a REAÇÃO do reator "
    "(expressão facial, sobrancelhas, boca, gesto) coerente com o tom do "
    "trecho de fala daquele painel — NUNCA descreva o falante.\n"
    "- ALTERNE entre os reatores ao longo do vídeo: cada reator tem um "
    "arquétipo emocional diferente; escolha o reator cuja postura combina "
    "melhor com cada trecho. Não use sempre o mesmo reator.\n"
    "- Aumente o uso de `narrative_mode=true` para ~60-80% dos painéis, "
    "intercalando com painéis de reator. O objetivo é ilustrar VISUALMENTE "
    "o que a voz em off descreve, com reatores reagindo entre uma cena e "
    "outra.\n"
    "- Os reatores são MUDOS: o `pose_description` NÃO deve descrever lábios "
    "articulando palavras nem fala. Apenas reação (espanto, deboche, riso, "
    "indignação, tédio, concordância)."
)


def _build_system_prompt(
    audio_duration: float,
    min_p: float,
    max_p: float,
    *,
    voiceover_mode: bool = False,
) -> str:
    base = (
        "Você é um roteirista visual de motion comics em pt-BR.\n"
        "Sua tarefa: dividir a transcrição em painéis ilustrados.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        f"- Cada painel deve durar entre {min_p:.1f} e {max_p:.1f} segundos.\n"
        f"- Cadência: pelo menos 1 painel a cada 5 s e no máximo 1 painel a cada 1,5 s.\n"
        f"- A soma das durações dos painéis deve ser ≈ {audio_duration:.2f} s (±0,2 s).\n"
        "- Painéis NÃO devem se sobrepor; o `start_time` de cada painel deve coincidir "
        "com o `end_time` do anterior.\n"
        "- O 1º painel deve começar em 0.0 e o último deve terminar na duração do áudio.\n"
        "- Para diálogo entre ≥2 pessoas, alternar close-ups com `framing=\"close\"` "
        "acompanhando quem fala em cada momento.\n"
        "- Para selfie/monólogo (1 pessoa só), manter o mesmo participante em todos os "
        "painéis variando `pose_description` e `framing`.\n"
        "- Para narração em off, distribuir múltiplos cenários distintos derivados do "
        "trecho narrado.\n"
        "- O `scene` deve ser inferido a partir da transcrição. Se não houver pista "
        "explícita, usar cenário neutro coerente (ex.: \"interior neutro\", "
        "\"rua urbana neutra\").\n"
        "- `participants` deve conter pelo menos um id do cast fornecido. Use os ids "
        "exatos como aparecem no cast. EXCEÇÃO: painéis em modo narrativo (ver abaixo) "
        "devem ter `participants=[]`.\n"
        "- O `pose_description` deve refletir o TOM EMOCIONAL do trecho falado: "
        "infira do conteúdo, da pontuação (exclamações, perguntas) e dos "
        "intensificadores qual é o estado emocional dominante e descreva pose, "
        "expressão facial e gestual em vocabulário visual concreto. Exemplos: "
        "\"olhos arregalados, boca aberta em espanto, mão na têmpora\" para "
        "surpresa; \"sobrancelhas franzidas, lábios apertados, dedo apontado\" "
        "para indignação; \"sorriso aberto, ombros relaxados, mão gesticulando\" "
        "para empolgação. Evite descritores abstratos como \"emocionado\" sem "
        "indicar a expressão correspondente.\n\n"
        "PAINÉIS EM MODO NARRATIVO (`narrative_mode=true`):\n"
        "- Quando a fala DESCREVE uma cena visualizável — uma história, um lugar, "
        "uma situação com personagens fictícios, objetos animados, animais, etc. "
        "(ex.: \"criança gritando na montanha-russa\", \"barata voando no quarto\", "
        "\"montanha gigante com cara de raiva\") — você DEVE marcar esse painel como "
        "`narrative_mode=true` e VISUALIZAR A CENA NARRADA, em vez de mostrar o "
        "falante encenando.\n"
        "- META: quando a transcrição for ricamente descritiva, use ~40-60% dos "
        "painéis em modo narrativo, alternados com painéis do falante (não-narrativos). "
        "Comece e termine com painéis do falante para ancorar; intercale narrativos "
        "no miolo onde a história é contada.\n"
        "- Em painéis narrativos:\n"
        "    * `participants` = [] (vazio).\n"
        "    * `narrative_elements` = lista de strings descrevendo CADA personagem/"
        "objeto fictício a aparecer, com expressão e ação. ANTROPOMORFIZE objetos "
        "(montanhas-russas, brinquedos, lagartas) com olhos, boca e expressões "
        "emotivas. Ex.: [\"criança chorona com lágrimas voando, boca aberta em "
        "berro\", \"montanha-russa amarela antropomorfizada com olhos esbugalhados "
        "e dentes à mostra\", \"pai sorridente acenando do chão com cara de tédio\"].\n"
        "    * `scene` = ambiente fictício curto (ex.: \"parque de diversões pastel "
        "com nuvens\", \"quarto de criança com poster\").\n"
        "    * `pose_description` = ação/expressão DOMINANTE da cena fictícia "
        "(ex.: \"criança esticada para trás pela velocidade enquanto montanha-russa "
        "ri maleficamente\").\n"
        "    * `framing` = escolha o que melhor enquadra a cena (geralmente `wide` "
        "ou `medium`).\n"
        "- Não force narrativo: se o trecho é só \"então eu falei…\", \"sabe né?\", "
        "comentário direto, mantenha o falante (`narrative_mode=false`)."
    )
    if voiceover_mode:
        base += _VOICEOVER_BLOCK
    return base


def _build_user_prompt(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    speakers: list[SpeakerSegment],
    correction_hint: str | None = None,
) -> str:
    audio_dur = _audio_duration(transcription)
    head = (
        f"Duração total do áudio: {audio_dur:.2f}s.\n"
        f"Cast disponível:\n{_format_cast(cast)}\n\n"
        f"Diarização:\n{_format_speakers(speakers)}\n\n"
        f"Transcrição (timestamps em segundos):\n{_format_transcription(transcription)}"
    )
    if correction_hint:
        head += (
            "\n\nA tentativa anterior violou as regras. Corrija o que segue: "
            f"{correction_hint}"
        )
    return head


# ---------------------------------------------------------------------------
# Speech-aware boundary snapping
# ---------------------------------------------------------------------------


def _extract_speech_gaps(
    transcription: TranscriptionResult, *, min_gap_seconds: float = SPEECH_GAP_MIN_SECONDS
) -> list[tuple[float, float]]:
    """Extrai intervalos de silêncio entre palavras com word-level timestamps.

    Retorna lista de ``(gap_start, gap_end)`` ordenada cronologicamente. Cai
    para boundaries de segments quando word-level está ausente.
    """

    times: list[tuple[float, float]] = []  # (word_end, next_word_start)
    last_end: float | None = None

    for seg in transcription.segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for w in seg_words:
                if last_end is not None and w.start > last_end:
                    gap = w.start - last_end
                    if gap >= min_gap_seconds:
                        times.append((last_end, w.start))
                last_end = max(last_end or 0.0, float(w.end))
        else:
            if last_end is not None and seg.start > last_end:
                gap = seg.start - last_end
                if gap >= min_gap_seconds:
                    times.append((last_end, seg.start))
            last_end = max(last_end or 0.0, float(seg.end))

    return times


def _snap_to_nearest_gap(
    target: float,
    gaps: list[tuple[float, float]],
    *,
    window: float = SPEECH_SNAP_WINDOW_SECONDS,
) -> float | None:
    """Devolve o ponto-alvo dentro de uma pausa, ou ``None`` se não houver
    gap dentro de ``±window`` de ``target``.

    Estratégia de snap:
    - Quando ``target`` cai dentro de um gap, retorna o **midpoint** do gap
      (corte no centro da pausa = mais natural perceptualmente).
    - Caso contrário, retorna o **midpoint do gap mais próximo** dentro da janela.

    Empate em distância → prefere o gap MAIOR (pausa mais longa = corte mais
    invisível pro ouvinte).
    """

    best: tuple[float, float, float] | None = None  # (distance, -gap_size, midpoint)
    for gap_start, gap_end in gaps:
        midpoint = (gap_start + gap_end) / 2.0
        gap_size = gap_end - gap_start

        if gap_start <= target <= gap_end:
            dist = 0.0
        elif gap_end < target:
            dist = target - gap_end
        else:
            dist = gap_start - target

        if dist > window:
            continue

        # Tuple ordering: menor distância primeiro; se empatar, maior gap_size
        # (negative pra ordenar reverso) ganha.
        candidate = (dist, -gap_size, midpoint)
        if best is None or candidate < best:
            best = candidate

    return None if best is None else best[2]


def _snap_panel_boundaries(
    panels: list[Panel],
    transcription: TranscriptionResult,
    *,
    min_panel: float,
    max_panel: float,
    audio_duration: float,
) -> list[Panel]:
    """Ajusta transições internas dos painéis para coincidirem com pausas
    naturais da fala. Não move o ``start`` do 1º painel nem o ``end`` do último.

    Cada snap só é aplicado se as durações resultantes continuarem dentro de
    ``[min_panel, max_panel]``. Caso contrário, mantém o valor original.
    """

    if len(panels) < 2:
        return panels

    gaps = _extract_speech_gaps(transcription)
    if not gaps:
        return panels

    new_times: list[tuple[float, float]] = [(p.start_time, p.end_time) for p in panels]

    for i in range(len(panels) - 1):
        current_end = new_times[i][1]
        snapped = _snap_to_nearest_gap(current_end, gaps)
        if snapped is None or abs(snapped - current_end) < 1e-3:
            continue

        new_start_i = new_times[i][0]
        new_end_iplus1 = new_times[i + 1][1]
        dur_i = snapped - new_start_i
        dur_iplus1 = new_end_iplus1 - snapped
        if not (min_panel <= dur_i <= max_panel + 1e-6):
            continue
        if not (min_panel <= dur_iplus1 <= max_panel + 1e-6):
            continue

        new_times[i] = (new_times[i][0], snapped)
        new_times[i + 1] = (snapped, new_times[i + 1][1])
        logger.info(
            "comic.script_planner: snap painel %d→%d para pausa em %.3fs (era %.3fs)",
            panels[i].index,
            panels[i + 1].index,
            snapped,
            current_end,
        )

    snapped_panels: list[Panel] = []
    for p, (s, e) in zip(panels, new_times):
        snapped_panels.append(
            p.model_copy(update={"start_time": s, "end_time": e, "panel_seconds_target": e - s})
        )
    return snapped_panels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_panels(
    raw_panels: list[dict[str, Any]],
    *,
    audio_duration: float,
    cast_ids: set[str],
    min_p: float,
    max_p: float,
) -> tuple[list[Panel], str | None]:
    """Valida invariantes; retorna ``(panels, error_hint)`` — hint é ``None`` se OK."""

    if not raw_panels:
        return [], "lista de painéis vazia"

    panels: list[Panel] = []
    for index, raw in enumerate(raw_panels):
        try:
            start = float(raw.get("start_time", 0.0))
            end = float(raw.get("end_time", 0.0))
        except (TypeError, ValueError):
            return [], f"painel {index}: timestamps inválidos"

        duration = end - start
        if duration < min_p or duration > max_p + 1e-6:
            return [], (
                f"painel {index}: duração {duration:.2f}s fora do intervalo "
                f"[{min_p:.1f}, {max_p:.1f}]s"
            )

        narrative_mode = bool(raw.get("narrative_mode") or False)
        narrative_elements = [
            str(e).strip()
            for e in (raw.get("narrative_elements") or [])
            if str(e).strip()
        ]

        participants = list(raw.get("participants") or [])
        if narrative_mode:
            if participants:
                return [], (
                    f"painel {index}: narrative_mode=true exige participants=[] "
                    f"(recebido {participants})"
                )
            if not narrative_elements:
                return [], (
                    f"painel {index}: narrative_mode=true exige "
                    f"narrative_elements não-vazio"
                )
        else:
            if not participants:
                return [], f"painel {index}: participants vazio"
            unknown = [p for p in participants if p not in cast_ids]
            if unknown:
                return [], (
                    f"painel {index}: ids desconhecidos {unknown}; use somente "
                    f"ids do cast: {sorted(cast_ids)}"
                )

        try:
            panel = Panel(
                index=index,
                start_time=start,
                end_time=end,
                participants=participants,
                framing=raw.get("framing", "medium"),
                scene=str(raw.get("scene", "") or "cenário neutro"),
                pose_description=str(raw.get("pose_description", "") or "neutro"),
                panel_seconds_target=duration,
                narrative_mode=narrative_mode,
                narrative_elements=narrative_elements,
            )
        except ValidationError as exc:
            return [], f"painel {index}: pydantic falhou: {exc.errors()[0]['msg']}"
        panels.append(panel)

    panels.sort(key=lambda p: p.start_time)

    for prev, cur in zip(panels, panels[1:]):
        if cur.start_time < prev.end_time - 1e-6:
            return [], f"painel {cur.index} sobrepõe painel {prev.index}"

    if panels[0].start_time > 1e-6:
        return [], f"primeiro painel não começa em 0 (começa em {panels[0].start_time:.2f}s)"

    last_end = panels[-1].end_time
    if abs(last_end - audio_duration) > DURATION_TOLERANCE_SECONDS:
        return [], (
            f"soma das durações ({last_end:.2f}s) difere do áudio ({audio_duration:.2f}s) "
            f"em mais de {DURATION_TOLERANCE_SECONDS:.1f}s"
        )

    expected_min = max(1, int(audio_duration // CADENCE_MAX_SECONDS_PER_PANEL))
    expected_max = max(1, int(audio_duration / CADENCE_MIN_SECONDS_PER_PANEL))
    if not (expected_min <= len(panels) <= expected_max + 1):
        return [], (
            f"cadência inválida: {len(panels)} painéis em {audio_duration:.1f}s "
            f"(esperado entre {expected_min} e {expected_max})"
        )

    return panels, None


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------


def _call_claude(
    client: anthropic.Anthropic,
    *,
    transcription: TranscriptionResult,
    cast: list[CastMember],
    speakers: list[SpeakerSegment],
    config: PipelineConfig,
    correction_hint: str | None = None,
) -> list[dict[str, Any]]:
    audio_dur = _audio_duration(transcription)
    system_prompt = _build_system_prompt(
        audio_dur,
        config.comic_panel_min_seconds,
        config.comic_panel_max_seconds,
        voiceover_mode=_is_voiceover_mode(cast),
    )
    user_text = _build_user_prompt(transcription, cast, speakers, correction_hint)

    try:
        response = client.with_options(timeout=120.0).messages.create(
            model=config.claude_model,
            max_tokens=4096,
            system=system_prompt,
            tools=[_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "plan_panels"},
            messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        )
    except anthropic.APIError as exc:
        msg = getattr(exc, "message", None) or str(exc)
        raise ScriptPlanError(f"Erro na API do Claude ao planejar painéis: {msg}") from exc

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "plan_panels":
            payload = getattr(block, "input", None) or {}
            return list(payload.get("panels") or [])
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_panels(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    speakers: list[SpeakerSegment],
    config: PipelineConfig,
    *,
    client: anthropic.Anthropic | None = None,
) -> list[Panel]:
    """Planeja a lista de painéis para o motion comic.

    Realiza até duas tentativas: a 2ª inclui no prompt a mensagem corretiva
    da invariante violada na 1ª. Após esgotar tentativas, levanta
    :class:`ScriptPlanError`.
    """

    if not transcription.segments:
        raise ScriptPlanError("Transcrição vazia: impossível planejar painéis.")
    if not cast:
        raise ScriptPlanError("Cast vazio: pelo menos um personagem é necessário.")

    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    audio_dur = _audio_duration(transcription)
    cast_ids = {c.character_id for c in cast}

    last_hint: str | None = None
    for attempt in range(2):
        raw_panels = _call_claude(
            client,
            transcription=transcription,
            cast=cast,
            speakers=speakers,
            config=config,
            correction_hint=last_hint,
        )
        panels, hint = _validate_panels(
            raw_panels,
            audio_duration=audio_dur,
            cast_ids=cast_ids,
            min_p=config.comic_panel_min_seconds,
            max_p=config.comic_panel_max_seconds,
        )
        if hint is None:
            logger.info(
                "comic.script_planner: %d painéis válidos para %.2fs de áudio",
                len(panels),
                audio_dur,
            )
            panels = _snap_panel_boundaries(
                panels,
                transcription,
                min_panel=config.comic_panel_min_seconds,
                max_panel=config.comic_panel_max_seconds,
                audio_duration=audio_dur,
            )
            return panels
        logger.warning(
            "comic.script_planner: tentativa %d violou invariante (%s)",
            attempt + 1,
            hint,
        )
        last_hint = hint

    raise ScriptPlanError(
        f"Falha ao planejar painéis após 2 tentativas. Último motivo: {last_hint}"
    )
