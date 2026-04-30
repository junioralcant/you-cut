"""Panel Renderer — imagem-base 9:16 + mini-clipe i2v com fallback estático (RF-18..RF-25)."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Iterable

from youcut.comic.providers.i2v import (
    DEFAULT_RATIO,
    I2VGenerationError,
    ImageToVideoProvider,
)
from youcut.comic.providers.images import (
    ImageGenerationError,
    ImageProvider,
    SOCIAL_PORTRAIT_SIZE,
)
from youcut.config import PipelineConfig
from youcut.models import (
    CastMember,
    Panel,
    PanelRenderResult,
    SpeakerSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


PANEL_IMAGE_SIZE: str = SOCIAL_PORTRAIT_SIZE  # 1024x1536 — portrait 2:3 (Runway reescala p/ 9:16)
PANEL_OUTPUT_WIDTH: int = 1080
PANEL_OUTPUT_HEIGHT: int = 1920
MAX_REFERENCES_PER_PANEL: int = 3

PRICE_PER_IMAGE_USD: float = 0.04
PRICE_PER_I2V_SECOND_USD: float = 0.05


_STYLE_PROMPT = (
    "Estilo visual fixo: caricatura editorial moderna, traço preto fino e "
    "expressivo, proporções levemente exageradas, olhos grandes e expressivos. "
    "MANTENHA fidelidade ESTRITA às fichas-âncora referenciadas — feições, "
    "cabelo, barba, pele, roupa e acessórios devem ser claramente reconhecíveis "
    "e consistentes em todos os painéis. Paleta pastel dessaturada, fundo "
    "aquarela digital. Sem fotorrealismo, sem texto embutido, sem "
    "marcas/logotipos/handles de terceiros."
)

_I2V_MOTION_PROMPT = (
    "Movimento expressivo SINCRONIZADO com a fala fornecida: a animação de "
    "lábios, expressão facial e linguagem corporal devem refletir EXATAMENTE "
    "o tom, ritmo e intensidade do que o personagem diz no áudio durante "
    "este painel. Use as palavras como guia de timing e emoção (exclamações "
    "→ surpresa/indignação visível; perguntas → expressão interrogativa; "
    "risadas → ombros e cabeça acompanhando; pausas → expressão sustentada). "
    "ESTRUTURA EMOCIONAL EM ARCO: comece com expressão moderada e CONSTRUA "
    "PROGRESSIVAMENTE até atingir o PICO EMOCIONAL no último 1.5s do clipe — "
    "o frame final deve ser o ponto de máxima expressividade (boca o mais "
    "aberta possível em fala/grito, olhos arregalados, sobrancelhas em "
    "extrema posição, gestos amplos), funcionando como o gancho visual "
    "da transição para o próximo painel. Manter identidade fiel às "
    "referências; sem mudanças de cenário ou de figurino."
)


# ---------------------------------------------------------------------------
# Prompt builders & helpers
# ---------------------------------------------------------------------------


def _ensure_panels_dir(output_dir: Path) -> Path:
    panels_dir = Path(output_dir) / "comic" / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    return panels_dir


def _select_references(panel: Panel, cast: list[CastMember]) -> list[Path]:
    """Seleciona até ``MAX_REFERENCES_PER_PANEL`` fichas-âncora dos participantes.

    Quando o painel tem mais participantes do que o limite, prioriza:
      1. quem aparece primeiro em ``panel.participants`` (ordem do roteiro);
      2. somente personagens com ficha-âncora válida no disco.
    """

    cast_by_id = {m.character_id: m for m in cast}
    refs: list[Path] = []
    for char_id in panel.participants:
        member = cast_by_id.get(char_id)
        if member is None or member.anchor_image_path is None:
            continue
        path = Path(member.anchor_image_path)
        if not path.exists():
            continue
        refs.append(path)
        if len(refs) >= MAX_REFERENCES_PER_PANEL:
            break
    return refs


def _build_image_base_prompt(panel: Panel, cast: list[CastMember]) -> str:
    cast_by_id = {m.character_id: m for m in cast}
    descriptions: list[str] = []
    for char_id in panel.participants:
        member = cast_by_id.get(char_id)
        if member is None:
            continue
        descriptions.append(f"`{member.character_id}` ({member.text_card or member.narrative_role})")

    cast_block = "; ".join(descriptions) or "personagens conforme cenário"
    framing_pt = {
        "close": "close-up",
        "medium": "plano médio",
        "wide": "plano aberto",
        "two_shot": "two-shot (dois personagens enquadrados juntos)",
    }.get(panel.framing, "plano médio")

    return (
        f"Painel ilustrado em proporção 9:16. Personagens em cena: {cast_block}. "
        f"Cenário: {panel.scene}. Enquadramento: {framing_pt}. "
        f"Pose/expressão dominante: {panel.pose_description}. "
        f"{_STYLE_PROMPT} Apenas os personagens listados; sem multidão, sem "
        "marcas/logos/handles de terceiros, sem texto embutido."
    )


def _identify_active_speakers_for_panel(
    panel: Panel, speakers: list[SpeakerSegment] | None
) -> set[str]:
    """Conjunto de ``speaker_id``s com fala ativa em qualquer momento do painel."""

    if not speakers:
        return set()
    active: set[str] = set()
    start, end = float(panel.start_time), float(panel.end_time)
    for s in speakers:
        if s.end <= start or s.start >= end:
            continue
        active.add(s.speaker_id)
    return active


def _split_speaking_vs_silent(
    panel: Panel,
    cast: list[CastMember],
    speakers: list[SpeakerSegment] | None,
) -> tuple[list[CastMember], list[CastMember]]:
    """Particiona personagens-pessoa do painel em (falando_no_painel, calado_no_painel).

    Cast members com ``kind != "person"`` são ignorados (objetos/animais não
    falam). Quando ``speakers`` está ausente, todos vão pra ``falando``
    (fallback conservador — sem info de diarização, assume que falam).
    """

    cast_by_id = {c.character_id: c for c in cast}
    persons = [
        cast_by_id[cid]
        for cid in panel.participants
        if cid in cast_by_id and cast_by_id[cid].kind == "person"
    ]
    if not speakers:
        return persons, []

    active = _identify_active_speakers_for_panel(panel, speakers)
    speaking: list[CastMember] = []
    silent: list[CastMember] = []
    for member in persons:
        if member.speaker_id and member.speaker_id in active:
            speaking.append(member)
        else:
            silent.append(member)
    return speaking, silent


def _extract_panel_dialogue(
    panel: Panel, transcription: TranscriptionResult | None
) -> str:
    """Devolve a fala (verbatim) que ocorre dentro de ``[panel.start_time, panel.end_time]``.

    Prefere palavras com timestamp (``WordTimestamp``); cai para segments quando
    word-level está ausente. Retorna string vazia quando o painel é silencioso.
    """

    if transcription is None:
        return ""

    start, end = float(panel.start_time), float(panel.end_time)

    words: list[str] = []
    for seg in transcription.segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for w in seg_words:
                if w.end <= start or w.start >= end:
                    continue
                token = (w.word or "").strip()
                if token:
                    words.append(token)
        else:
            if seg.end <= start or seg.start >= end:
                continue
            text = (seg.text or "").strip()
            if text:
                words.append(text)

    return " ".join(words).strip()


def _build_lipsync_block(
    speaking: list[CastMember],
    silent: list[CastMember],
    has_dialogue: bool,
    has_active_speaker: bool,
) -> str:
    """Constrói a diretiva de lip-sync por personagem para o prompt do i2v.

    Casos:
    - 1+ personagens com fala ativa no painel → especifica quem deve animar
      lábios e quem deve ficar calado.
    - Apenas personagens calados (mas há áudio/speaker ativo) → voz em OFF;
      ninguém move os lábios, só reagem corporalmente.
    - Sem fala no painel (silencioso) → sem lip-sync.
    """

    def _names(members: list[CastMember]) -> str:
        return ", ".join(m.character_id for m in members)

    if not has_dialogue and not has_active_speaker:
        return ""

    if speaking and not silent:
        return (
            f" LIP-SYNC: apenas {_names(speaking)} fala neste painel — "
            f"animar articulação labial sincronizada com a fala."
        )
    if speaking and silent:
        return (
            f" LIP-SYNC: APENAS {_names(speaking)} fala neste painel "
            f"(animar articulação labial sincronizada). "
            f"{_names(silent)} NÃO fala — manter a boca FECHADA OU "
            f"relaxada, sem qualquer movimento de articulação labial; "
            f"animar apenas expressão facial e gestos corporais coerentes."
        )
    if not speaking and silent:
        return (
            f" VOZ EM OFF: a fala do painel vem de fora de cena ou de "
            f"narrador. {_names(silent)} NÃO fala — TODOS os personagens "
            f"visíveis devem manter a boca FECHADA ou relaxada, SEM "
            f"qualquer movimento labial. Animar apenas reação facial "
            f"(olhos, sobrancelhas) e gestos corporais coerentes com o "
            f"áudio, mas SEM articulação labial."
        )
    return ""


def _build_i2v_prompt(
    panel: Panel,
    transcription: TranscriptionResult | None = None,
    cast: list[CastMember] | None = None,
    speakers: list[SpeakerSegment] | None = None,
) -> str:
    dialogue = _extract_panel_dialogue(panel, transcription)
    dialogue_block = (
        f' Fala neste painel (verbatim, use como guia de timing/emoção): "{dialogue}".'
        if dialogue
        else " (Painel silencioso — manter movimento ambiente sutil.)"
    )

    lipsync_block = ""
    if cast:
        speaking, silent = _split_speaking_vs_silent(panel, cast, speakers)
        has_active_speaker = bool(_identify_active_speakers_for_panel(panel, speakers))
        lipsync_block = _build_lipsync_block(
            speaking,
            silent,
            has_dialogue=bool(dialogue),
            has_active_speaker=has_active_speaker,
        )

    return (
        f"Cena: {panel.scene}. Pose dominante: {panel.pose_description}."
        f"{dialogue_block}{lipsync_block} "
        f"{_I2V_MOTION_PROMPT}"
    )


def _panel_paths(panels_dir: Path, panel_index: int) -> tuple[Path, Path]:
    image_path = panels_dir / f"panel_{panel_index:02d}.png"
    clip_path = panels_dir / f"panel_{panel_index:02d}.mp4"
    return image_path, clip_path


def _clamp_panel_seconds(panel: Panel, config: PipelineConfig) -> float:
    duration = panel.panel_seconds_target
    duration = max(config.comic_panel_min_seconds, min(config.comic_panel_max_seconds, duration))
    return float(duration)


# ---------------------------------------------------------------------------
# Image base
# ---------------------------------------------------------------------------


def _render_image_base(
    panel: Panel,
    cast: list[CastMember],
    image_provider: ImageProvider,
    image_path: Path,
) -> int:
    prompt = _build_image_base_prompt(panel, cast)
    references = _select_references(panel, cast)

    try:
        png_bytes = image_provider.generate(
            prompt,
            reference_images=references or None,
            size=PANEL_IMAGE_SIZE,
            input_fidelity="high",
        )
    except ImageGenerationError:
        raise
    except Exception as exc:
        raise ImageGenerationError(
            f"Falha inesperada ao gerar imagem-base do painel {panel.index}: {exc}"
        ) from exc

    if not png_bytes:
        raise ImageGenerationError(
            f"Provider retornou bytes vazios para imagem-base do painel {panel.index}."
        )

    image_path.write_bytes(png_bytes)
    return 1


# ---------------------------------------------------------------------------
# I2V + fallback
# ---------------------------------------------------------------------------


def _render_static_fallback(image_path: Path, clip_path: Path, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(image_path),
        "-vf",
        f"scale={PANEL_OUTPUT_WIDTH}:{PANEL_OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={PANEL_OUTPUT_WIDTH}:{PANEL_OUTPUT_HEIGHT}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        str(clip_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise I2VGenerationError(
            "ffmpeg não encontrado no PATH; impossível gerar fallback estático."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore")
        raise I2VGenerationError(
            f"Falha do ffmpeg ao gerar fallback estático do painel: {stderr.strip() or 'erro desconhecido'}"
        ) from exc


def _render_i2v_with_fallback(
    panel: Panel,
    cast: list[CastMember],
    i2v_provider: ImageToVideoProvider,
    image_path: Path,
    clip_path: Path,
    *,
    duration: float,
    transcription: TranscriptionResult | None = None,
    speakers: list[SpeakerSegment] | None = None,
) -> tuple[bool, int]:
    """Gera mini-clipe via i2v ou cai para fallback estático.

    Retorna ``(was_static_fallback, attempts_used)``.
    """

    prompt_text = _build_i2v_prompt(panel, transcription, cast=cast, speakers=speakers)
    references = _select_references(panel, cast)

    try:
        video_bytes = i2v_provider.image_to_video(
            prompt_image=image_path,
            prompt_text=prompt_text,
            reference_images=references,
            duration_seconds=duration,
            ratio=DEFAULT_RATIO,
        )
    except I2VGenerationError as exc:
        logger.warning(
            "comic.panel_renderer: i2v falhou para painel %d (%s); usando fallback estático.",
            panel.index,
            exc,
        )
        _render_static_fallback(image_path, clip_path, duration)
        return True, 1
    except Exception as exc:
        logger.warning(
            "comic.panel_renderer: erro inesperado no i2v do painel %d (%s); fallback estático.",
            panel.index,
            exc,
        )
        _render_static_fallback(image_path, clip_path, duration)
        return True, 1

    if not video_bytes:
        logger.warning(
            "comic.panel_renderer: i2v retornou bytes vazios para painel %d; fallback estático.",
            panel.index,
        )
        _render_static_fallback(image_path, clip_path, duration)
        return True, 1

    clip_path.write_bytes(video_bytes)
    return False, 1


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def _compute_cost(image_attempts: int, was_static_fallback: bool, duration: float) -> float:
    cost = image_attempts * PRICE_PER_IMAGE_USD
    if not was_static_fallback:
        cost += duration * PRICE_PER_I2V_SECOND_USD
    return round(cost, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_panel(
    panel: Panel,
    cast: list[CastMember],
    output_dir: Path,
    config: PipelineConfig,
    *,
    image_provider: ImageProvider,
    i2v_provider: ImageToVideoProvider,
    transcription: TranscriptionResult | None = None,
    speakers: list[SpeakerSegment] | None = None,
) -> PanelRenderResult:
    panels_dir = _ensure_panels_dir(output_dir)
    image_path, clip_path = _panel_paths(panels_dir, panel.index)
    duration = _clamp_panel_seconds(panel, config)

    image_attempts = _render_image_base(panel, cast, image_provider, image_path)
    was_static, i2v_attempts = _render_i2v_with_fallback(
        panel,
        cast,
        i2v_provider,
        image_path,
        clip_path,
        duration=duration,
        transcription=transcription,
        speakers=speakers,
    )

    cost = _compute_cost(image_attempts, was_static, duration)
    return PanelRenderResult(
        panel_index=panel.index,
        base_image_path=image_path,
        clip_path=clip_path,
        clip_seconds=duration,
        was_static_fallback=was_static,
        image_attempts=image_attempts,
        i2v_attempts=i2v_attempts,
        cost_usd=cost,
    )


async def _render_panel_async(
    panel: Panel,
    cast: list[CastMember],
    output_dir: Path,
    config: PipelineConfig,
    image_provider: ImageProvider,
    i2v_provider: ImageToVideoProvider,
    semaphore: asyncio.Semaphore,
    transcription: TranscriptionResult | None = None,
    speakers: list[SpeakerSegment] | None = None,
) -> PanelRenderResult:
    async with semaphore:
        return await asyncio.to_thread(
            render_panel,
            panel,
            cast,
            output_dir,
            config,
            image_provider=image_provider,
            i2v_provider=i2v_provider,
            transcription=transcription,
            speakers=speakers,
        )


def render_all(
    panels: Iterable[Panel],
    cast: list[CastMember],
    output_dir: Path,
    config: PipelineConfig,
    *,
    image_provider: ImageProvider,
    i2v_provider: ImageToVideoProvider,
    transcription: TranscriptionResult | None = None,
    speakers: list[SpeakerSegment] | None = None,
) -> list[PanelRenderResult]:
    """Renderiza todos os painéis em paralelo limitado por ``comic_i2v_concurrency``.

    Quando ``transcription`` é fornecida, o trecho de fala dentro de cada
    ``[panel.start_time, panel.end_time]`` é injetado verbatim no prompt do i2v
    para sincronizar a animação corporal/facial com o áudio do vídeo original.

    Quando ``speakers`` é fornecido, o prompt instrui lip-sync APENAS para os
    personagens cujo ``speaker_id`` está ativo no time range do painel — os
    demais ficam com a boca fechada (evita boca animada em personagem que
    não está falando, ex: voz em off do reator sobre cena de cozinha).
    """

    panels_list = list(panels)
    if not panels_list:
        return []

    concurrency = max(1, int(config.comic_i2v_concurrency))
    semaphore = asyncio.Semaphore(concurrency)

    async def _runner() -> list[PanelRenderResult]:
        tasks = [
            _render_panel_async(
                p, cast, output_dir, config, image_provider, i2v_provider, semaphore,
                transcription=transcription,
                speakers=speakers,
            )
            for p in panels_list
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        results = asyncio.run(_runner())
    else:  # pragma: no cover - chamado em ambientes async
        future = asyncio.ensure_future(_runner(), loop=loop)
        results = loop.run_until_complete(future)

    results.sort(key=lambda r: r.panel_index)
    return results
