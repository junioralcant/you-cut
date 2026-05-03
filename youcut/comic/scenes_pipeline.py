"""Engine ``scenes`` — pipeline multi-cena com word-level lip-sync.

Diferenças do ``prunaai`` (single-master) e ``panels`` (per-beat):

1. Quebra a transcrição em N **cenas narrativas** (Claude scene planner).
2. Gera 1 master por cena (todos referenciam um **anchor visual canônico**
   pra consistência de estilo, paleta e design dos personagens).
3. **Word-level visual attribution**: 1 frame por palavra (escala 240px) é
   enviado ao Claude vision, que identifica qual ator (Eva ou Cobra) está
   com a boca aberta em cada palavra.
4. **Smoothing conservador** das atribuições — flipa apenas chunks curtos
   cujo texto é repetição clara do vizinho (não interjeições legítimas).
5. **Smart-cuts** estendem chunks < 1.05s pra atender o mínimo do Prunaai.
6. **Gap absorption** — gaps > 0.5s entre chunks (provável "harrá", risadas)
   são absorvidos pelo chunk anterior pra animar em vez de freeze.
7. **Crossfade** suave entre chunks consecutivos (~0.25s).
8. Emite versões com e sem legenda, ambas com mesma proporção (scale+crop)
   e watermark configurável (@username) na safe zone.

Output: ``output/<video>/motion_comic_scenes.mp4`` (com legendas) e
``motion_comic_scenes_no_subs.mp4`` (opcional).
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from youcut.captioner import build_ass_for_words
from youcut.comic.cast_builder import build_cast
from youcut.comic.cast_inventor import invent_cast
from youcut.comic.composer import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    _filter_words_in_range,
    _mux_audio,
)
from youcut.comic.providers.images import ImageGenerationError, ImageProvider, OpenAIImageProvider
from youcut.comic.providers.prunaai import PrunaaiAnimationProvider
from youcut.comic.session import (
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.comic.validator import validate_video
from youcut.comic.visual_analyzer import detect_cast
from youcut.config import PipelineConfig
from youcut.diarizer import diarize
from youcut.models import (
    CastMember,
    MotionComicSession,
    SpeakerSegment,
    TranscriptionResult,
)
from youcut.transcriber import transcribe

logger = logging.getLogger(__name__)

WATERMARK_FONT: Path = Path(__file__).resolve().parents[1] / "assets" / "Roboto-Regular.ttf"


class ScenesPipelineError(Exception):
    """Erro fatal do pipeline ``scenes``."""


# ---------------------------------------------------------------------------
# Scene planner (Claude)
# ---------------------------------------------------------------------------

_SCENE_PLANNER_TOOL: dict[str, Any] = {
    "name": "plan_scenes",
    "description": (
        "Divide a transcrição em N cenas narrativas em sequência (cada cena "
        "8-25s), cobrindo todo o áudio."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_id": {"type": "string"},
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "narrative_action": {"type": "string"},
                        "active_speaker": {
                            "type": "string",
                            "enum": ["both", "none", "primary", "secondary"],
                        },
                        "characters_in_frame": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "expression_primary": {"type": "string"},
                        "expression_secondary": {"type": "string"},
                    },
                    "required": [
                        "scene_id", "start_time", "end_time",
                        "narrative_action", "active_speaker",
                        "characters_in_frame",
                    ],
                },
            }
        },
        "required": ["scenes"],
    },
}


def _build_scene_system_prompt(n_scenes: int, cast: list[CastMember]) -> str:
    cast_lines = "\n".join(f"- {m.character_id}: {(m.text_card or m.narrative_role or '')[:200]}" for m in cast)
    return (
        f"Você é um diretor de animação. Divida a transcrição em exatamente {n_scenes} cenas com arco narrativo.\n\n"
        f"CAST DISPONÍVEL:\n{cast_lines}\n\n"
        "Use o character_id (ex.: 'eva', 'cobra') em characters_in_frame e em primary/secondary "
        "speakers. Cada cena cobre 8-25s. Cubra todo o áudio. Use timestamps reais da transcrição."
    )


def plan_scenes(
    transcription: TranscriptionResult,
    cast: list[CastMember],
    audio_duration: float,
    config: PipelineConfig,
    *,
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    """Pede ao Claude pra dividir a transcrição em ``config.comic_scenes_count`` cenas."""
    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    n = config.comic_scenes_count
    transcript_text = "\n".join(
        f"[{seg.start:.1f}-{seg.end:.1f}s] {seg.text.strip()}"
        for seg in transcription.segments
    )
    user = (
        f"Transcrição com timestamps:\n{transcript_text}\n\n"
        f"Duração total: {audio_duration:.2f}s.\n\n"
        f"Divida em {n} cenas com arco narrativo coerente."
    )
    response = client.with_options(timeout=120.0).messages.create(
        model=config.claude_model,
        max_tokens=2048,
        system=_build_scene_system_prompt(n, cast),
        tools=[_SCENE_PLANNER_TOOL],
        tool_choice={"type": "tool", "name": "plan_scenes"},
        messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "plan_scenes":
            scenes = block.input["scenes"]
            if len(scenes) != n:
                logger.warning("plan_scenes: esperava %d cenas, recebi %d", n, len(scenes))
            return scenes
    raise ScenesPipelineError("Claude não retornou plano de cenas")


# ---------------------------------------------------------------------------
# Word-level visual speaker attribution (Claude vision)
# ---------------------------------------------------------------------------

_WORD_TOOL: dict[str, Any] = {
    "name": "attribute_words_visual",
    "description": "Atribui cada palavra a um character_id baseado em qual ator articula a boca no frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "attributions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word_idx": {"type": "integer"},
                        "character": {"type": "string"},
                    },
                    "required": ["word_idx", "character"],
                },
            }
        },
        "required": ["attributions"],
    },
}


def _build_word_system_prompt(cast: list[CastMember]) -> str:
    cast_lines = "\n".join(
        f"- {m.character_id}: {(m.text_card or m.narrative_role or '')[:240]}"
        for m in cast
    )
    return (
        "Você é um analista visual de fala palavra-por-palavra. Vai receber palavras "
        "de uma transcrição e 1 frame por palavra (extraído no midpoint temporal).\n\n"
        f"CAST:\n{cast_lines}\n\n"
        "Para CADA palavra, observe o frame correspondente e identifique qual personagem "
        "tem a boca CLARAMENTE ABERTA (lip movement). Atribua a palavra ao character_id "
        "correspondente. Se nenhum mostra movimento claro, use o tom da fala como tiebreaker. "
        "VOCÊ DEVE atribuir TODAS as palavras."
    )


@dataclass
class WordEntry:
    global_idx: int
    seg_idx: int
    word_idx: int
    start: float
    end: float
    text: str
    frame_path: Path | None = None


def _flatten_words(transcription: TranscriptionResult) -> list[WordEntry]:
    out: list[WordEntry] = []
    gidx = 0
    for s_idx, seg in enumerate(transcription.segments):
        for w_idx, word in enumerate(getattr(seg, "words", None) or []):
            out.append(WordEntry(
                global_idx=gidx, seg_idx=s_idx, word_idx=w_idx,
                start=word.start, end=word.end, text=(word.word or "").strip(),
            ))
            gidx += 1
    return out


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_frame(video_path: Path, ts: float, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    cmd = [
        "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
        "-vframes", "1", "-vf", "scale=240:-2", "-q:v", "5",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def attribute_words_visual(
    video_path: Path,
    words: list[WordEntry],
    scenes: list[dict],
    cast: list[CastMember],
    work_dir: Path,
    config: PipelineConfig,
    *,
    inter_call_pause_s: float = 60.0,
) -> dict[int, str]:
    """Atribuição visual word-level. Batched por cena (rate-limit Claude 30K ITPM)."""
    frames_dir = work_dir / "word_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for w in words:
        ts = (w.start + w.end) / 2
        fp = frames_dir / f"w{w.global_idx:04d}.jpg"
        _extract_frame(video_path, ts, fp)
        w.frame_path = fp
    logger.info("scenes: %d frames extraídos pra word-level attribution", len(words))

    def _scene_of_ts(ts: float) -> str:
        for s in scenes:
            if s["start_time"] <= ts < s["end_time"]:
                return s["scene_id"]
        return scenes[-1]["scene_id"]

    by_scene: dict[str, list[WordEntry]] = {s["scene_id"]: [] for s in scenes}
    for w in words:
        by_scene[_scene_of_ts((w.start + w.end) / 2)].append(w)

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    valid_ids = {m.character_id for m in cast}
    out: dict[int, str] = {}
    for idx, scene in enumerate(scenes):
        sid = scene["scene_id"]
        scene_words = by_scene.get(sid, [])
        if not scene_words:
            continue
        word_list = "\n".join(
            f"[w{w.global_idx:04d}] {w.start:.2f}-{w.end:.2f}s: \"{w.text}\""
            for w in scene_words
        )
        ctx = (
            f"CENA: {sid} ({scene['start_time']:.1f}-{scene['end_time']:.1f}s)\n"
            f"AÇÃO: {scene['narrative_action']}\n"
            f"Personagens em cena: {scene.get('characters_in_frame', [])}\n\n"
            f"Palavras desta cena (em ordem):\n{word_list}\n\n"
            "Frames (1 por palavra):"
        )
        content: list[dict] = [{"type": "text", "text": ctx}]
        for w in scene_words:
            content.append({"type": "text", "text": f"\n[w{w.global_idx:04d}] {w.text}"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode_image_b64(w.frame_path)},
            })
        content.append({
            "type": "text",
            "text": (
                f"\n\nAtribua TODAS as {len(scene_words)} palavras a um character_id "
                f"(valores aceitos: {sorted(valid_ids)})."
            ),
        })

        logger.info("scenes: claude vision cena %s (%d palavras)…", sid, len(scene_words))
        response = client.with_options(timeout=300.0).messages.create(
            model=config.claude_model,
            max_tokens=8192,
            system=_build_word_system_prompt(cast),
            tools=[_WORD_TOOL],
            tool_choice={"type": "tool", "name": "attribute_words_visual"},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "attribute_words_visual":
                for a in block.input["attributions"]:
                    char = a["character"]
                    if char in valid_ids:
                        out[a["word_idx"]] = char

        if idx < len(scenes) - 1:
            logger.info("scenes: aguardando %.0fs antes da próxima cena (rate-limit Claude)…", inter_call_pause_s)
            time.sleep(inter_call_pause_s)
    return out


# ---------------------------------------------------------------------------
# Smoothing conservador (substring-based)
# ---------------------------------------------------------------------------

def smooth_attributions(
    words: list[WordEntry],
    attributions: dict[int, str],
    *,
    min_chunk_dur: float = 1.5,
) -> dict[int, str]:
    """Flipa chunks curtos cercados por mesmo speaker SE texto é repetição (substring)."""
    if not attributions:
        return attributions
    chunks: list[tuple[str, int, int, float, float, str]] = []
    cur_speaker = None
    cur_w_start = 0
    for i, w in enumerate(words):
        sp = attributions.get(w.global_idx)
        if sp != cur_speaker:
            if cur_speaker is not None:
                txt = " ".join(words[k].text for k in range(cur_w_start, i)).strip()
                chunks.append((cur_speaker, cur_w_start, i, words[cur_w_start].start, words[i - 1].end, txt))
            cur_speaker = sp
            cur_w_start = i
    if cur_speaker is not None:
        txt = " ".join(words[k].text for k in range(cur_w_start, len(words))).strip()
        chunks.append((cur_speaker, cur_w_start, len(words), words[cur_w_start].start, words[-1].end, txt))

    def _norm(s: str) -> str:
        return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()

    n_smoothed = n_skipped = 0
    for i, (sp, w_s, w_e, t_s, t_e, txt) in enumerate(chunks):
        if (t_e - t_s) >= min_chunk_dur:
            continue
        prev_sp = chunks[i - 1][0] if i > 0 else None
        next_sp = chunks[i + 1][0] if i + 1 < len(chunks) else None
        if prev_sp != next_sp or prev_sp is None or prev_sp == sp:
            continue
        c_norm = _norm(txt)
        if not c_norm:
            continue
        if c_norm in _norm(chunks[i - 1][5]) or c_norm in _norm(chunks[i + 1][5]):
            for w_idx in range(w_s, w_e):
                attributions[words[w_idx].global_idx] = prev_sp
            logger.info("scenes smooth [%s→%s]: '%s' — repetição contida no vizinho", sp, prev_sp, txt[:40])
            n_smoothed += 1
        else:
            n_skipped += 1
    if n_smoothed or n_skipped:
        logger.info("scenes smoothing: %d corrigidos, %d preservados (interjeições)", n_smoothed, n_skipped)
    return attributions


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    scene_id: str
    character: str
    start: float
    end: float
    text: str
    master_path: Path
    raw_path: Path = field(default_factory=Path)


def _scene_of_ts(scenes: list[dict], ts: float) -> str:
    for s in scenes:
        if s["start_time"] <= ts < s["end_time"]:
            return s["scene_id"]
    return scenes[-1]["scene_id"]


def build_chunks_from_words(
    words: list[WordEntry],
    attributions: dict[int, str],
    scenes: list[dict],
    masters: dict[str, Path],
    work_dir: Path,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur_scene = cur_char = None
    cur_words: list[WordEntry] = []

    def _flush():
        nonlocal cur_words, cur_char, cur_scene
        if not cur_words or cur_char is None or cur_scene is None:
            cur_words = []
            return
        text = " ".join(w.text for w in cur_words if w.text).strip()
        cid = f"scenes_{cur_scene}_{cur_char}_{cur_words[0].start:.2f}".replace(".", "p")
        chunks.append(Chunk(
            chunk_id=cid, scene_id=cur_scene, character=cur_char,
            start=cur_words[0].start, end=cur_words[-1].end, text=text,
            master_path=masters[cur_scene],
            raw_path=work_dir / f"chunk_{cid}.mp4",
        ))
        cur_words = []

    for w in words:
        char = attributions.get(w.global_idx)
        if char is None:
            if cur_char is not None:
                char = cur_char
            else:
                continue
        sid = _scene_of_ts(scenes, (w.start + w.end) / 2)
        if char != cur_char or sid != cur_scene:
            _flush()
            cur_char = char
            cur_scene = sid
        cur_words.append(w)
    _flush()
    return chunks


def absorb_long_gaps(chunks: list[Chunk], gap_threshold: float = 0.5) -> list[Chunk]:
    """Estende chunks que precedem gaps > threshold pra absorver laughs/expressões."""
    chunks_sorted = sorted(chunks, key=lambda c: c.start)
    n = 0
    for i in range(len(chunks_sorted) - 1):
        gap = chunks_sorted[i + 1].start - chunks_sorted[i].end
        if gap > gap_threshold:
            chunks_sorted[i].end = chunks_sorted[i + 1].start
            n += 1
    if n:
        logger.info("scenes: %d gaps > %.2fs absorvidos pelos chunks anteriores", n, gap_threshold)
    return chunks_sorted


def expand_short_chunks(chunks: list[Chunk], audio_duration: float, *, min_dur: float = 1.05) -> list[Chunk]:
    chunks_sorted = sorted(chunks, key=lambda c: c.start)
    for i, c in enumerate(chunks_sorted):
        if (c.end - c.start) >= min_dur:
            continue
        needed = min_dur - (c.end - c.start)
        next_start = chunks_sorted[i + 1].start if i + 1 < len(chunks_sorted) else audio_duration
        gap_after = max(0.0, next_start - c.end)
        take = min(gap_after, needed)
        if take > 0:
            c.end += take
            needed -= take
        if needed > 0:
            prev_end = chunks_sorted[i - 1].end if i > 0 else 0.0
            gap_before = max(0.0, c.start - prev_end)
            take = min(gap_before, needed)
            if take > 0:
                c.start -= take
                needed -= take
    return chunks_sorted


def reconcile_cache(chunks: list[Chunk], work_dir: Path) -> None:
    """Apaga raw/audio chunks cacheados cujo conteúdo divergiu (após smoothing/absorb)."""
    for c in chunks:
        audio_path = work_dir / f"audio_chunk_{c.chunk_id}.mp3"
        if not audio_path.exists():
            continue
        try:
            audio_dur = _ffprobe_duration(audio_path)
        except Exception:
            continue
        expected_dur = c.end - c.start
        if abs(audio_dur - expected_dur) > 0.3:
            logger.info(
                "scenes: cache STALE [%s] audio %.2fs vs esperado %.2fs — apagando",
                c.chunk_id, audio_dur, expected_dur,
            )
            audio_path.unlink(missing_ok=True)
            if c.raw_path.exists():
                c.raw_path.unlink(missing_ok=True)
            ext_path = work_dir / f"ext_{c.chunk_id}.mp4"
            ext_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Visual anchor + scene masters (consistência visual)
# ---------------------------------------------------------------------------

ANCHOR_FILENAME = "_visual_anchor.png"


def _build_anchor_prompt(cast: list[CastMember], scene_seed: str | None) -> str:
    cast_lines = []
    for m in cast:
        snippet = (m.text_card or m.narrative_role or m.character_id)[:240]
        cast_lines.append(f"- {m.character_id}: {snippet}")
    cast_block = "\n".join(cast_lines) or "(cast vazio)"
    scene = scene_seed or "ambiente coerente com a história"
    return (
        "Painel 9:16 — IMAGEM DE REFERÊNCIA VISUAL CANÔNICA pra motion comic.\n\n"
        f"PERSONAGENS CANÔNICOS:\n{cast_block}\n\n"
        f"CENÁRIO: {scene}\n\n"
        "ESTILO: cartoon flat 2D clean (linhas pretas finas digitais regulares, NÃO desenhado à mão), "
        "cores chapadas com sombras simples, OLHOS GRANDES PRETOS REDONDOS, expressões expressivas. "
        "Luz natural neutra de dia (SEM golden hour, SEM filtro warm/dourado-laranja).\n\n"
        "*** REGRA ABSOLUTA — ZERO TEXTO NA IMAGEM ***\n"
        "PROIBIDO desenhar QUALQUER texto, letras, palavras, balões, números, símbolos, marcas. "
        "A imagem deve ser PURAMENTE PICTÓRICA."
    )


def _build_scene_master_prompt(scene: dict, cast: list[CastMember]) -> str:
    chars = scene.get("characters_in_frame", [])
    cast_by_id = {m.character_id: m for m in cast}
    char_lines = []
    for cid in chars:
        m = cast_by_id.get(cid)
        if m:
            char_lines.append(f"- {cid}: {(m.text_card or '')[:160]}")
    chars_block = "\n".join(char_lines) or f"(personagens: {chars})"
    return (
        f"Painel 9:16 — composição da cena {scene['scene_id']}.\n\n"
        "FRAMING: WIDE SHOT (vista aberta), personagens ocupam a metade central inferior, "
        "deixando bastante cenário visível ao redor.\n\n"
        f"AÇÃO DA CENA: {scene['narrative_action']}\n\n"
        f"PERSONAGENS EM QUADRO:\n{chars_block}\n\n"
        "REGRAS DE CONSISTÊNCIA VISUAL (CRÍTICAS):\n"
        "- Use EXATAMENTE a paleta e estilo da 1ª imagem de referência (visual anchor).\n"
        "- Personagens IDÊNTICOS aos da imagem-referência (design, cores, proporções).\n"
        "- Cenário coerente com a paleta da referência.\n\n"
        "ESTILO: cartoon flat 2D clean, linhas pretas finas, OLHOS GRANDES, cores chapadas. "
        "Luz natural neutra (SEM tom dourado-laranja).\n\n"
        "*** REGRA ABSOLUTA — ZERO TEXTO ***\n"
        "PROIBIDO texto, letras, balões de fala, números, marcas. Imagem puramente pictórica."
    )


def generate_visual_anchor(
    cast: list[CastMember],
    scene_seed: str | None,
    work_dir: Path,
    image_provider: ImageProvider,
    config: PipelineConfig,
) -> Path:
    out_path = work_dir / ANCHOR_FILENAME
    if out_path.exists():
        logger.info("scenes: anchor reusado de %s", out_path)
        return out_path
    refs: list[Path] = []
    if config.comic_scenes_style_ref_image and Path(config.comic_scenes_style_ref_image).exists():
        refs.append(Path(config.comic_scenes_style_ref_image))
    else:
        for m in cast:
            if m.anchor_image_path and Path(m.anchor_image_path).exists():
                refs.append(Path(m.anchor_image_path))
    prompt = _build_anchor_prompt(cast, scene_seed)
    logger.info("scenes: gerando visual anchor (refs=%d)…", len(refs))
    png = image_provider.generate(
        prompt, reference_images=refs or None,
        size="1024x1536", input_fidelity="high",
    )
    out_path.write_bytes(png)
    return out_path


def generate_scene_masters(
    scenes: list[dict],
    cast: list[CastMember],
    anchor_path: Path,
    work_dir: Path,
    image_provider: ImageProvider,
    config: PipelineConfig,
) -> dict[str, Path]:
    refs_base = [anchor_path]
    if config.comic_scenes_style_ref_image and Path(config.comic_scenes_style_ref_image).exists():
        refs_base.append(Path(config.comic_scenes_style_ref_image))
    masters: dict[str, Path] = {}
    for scene in scenes:
        sid = scene["scene_id"]
        out_path = work_dir / f"master_{sid}.png"
        masters[sid] = out_path
        if out_path.exists():
            logger.info("scenes: master %s reusado", sid)
            continue
        prompt = _build_scene_master_prompt(scene, cast)
        logger.info("scenes: gerando master %s (refs=%d)…", sid, len(refs_base))
        png = image_provider.generate(
            prompt, reference_images=refs_base,
            size="1024x1536", input_fidelity="high",
        )
        out_path.write_bytes(png)
    return masters


# ---------------------------------------------------------------------------
# Render chunks (Prunaai)
# ---------------------------------------------------------------------------

def _build_chunk_video_prompt(chunk: Chunk, scene: dict, cast: list[CastMember]) -> str:
    """Prompt do video_prompt do Prunaai. NÃO inclui chunk.text (evita legendas embutidas)."""
    cast_by_id = {m.character_id: m for m in cast}
    speaker_id = chunk.character
    speaker_desc = ""
    silent_blocks = []
    for cid in scene.get("characters_in_frame", []):
        m = cast_by_id.get(cid)
        if not m:
            continue
        snippet = (m.text_card or m.narrative_role or cid)[:120]
        if cid == speaker_id:
            speaker_desc = f"{cid}: {snippet}"
        else:
            silent_blocks.append(f"{cid}: {snippet}")
    silence_directive = ""
    if silent_blocks:
        silence_directive = (
            "\nThe following characters MUST keep mouth FULLY CLOSED, ZERO mouth/lip movement, "
            "ZERO lip-sync — they are listening in silence (only subtle facial reactions allowed):\n"
            + "\n".join(f"  - {s}" for s in silent_blocks)
        )
    speak_block = (
        f"\n*** APPLY LIP-SYNC ONLY TO: {speaker_id} ***\n"
        f"Active speaker: {speaker_desc}\n"
        "Articulate ONLY this character's mouth in sync with the audio."
        f"{silence_directive}"
    )
    return (
        "Cartoon flat 2D pastel animation preserving EXACTLY the composition, "
        "soft pastel palette and character design from the master image.\n"
        f"SCENE ACTION: {scene['narrative_action']}\n"
        f"{speak_block}\n"
        "Style: thick black outline, BIG eyes, soft pastel cartoon style.\n\n"
        "*** CRITICAL — NO TEXT/SUBTITLES/CAPTIONS IN THE VIDEO ***\n"
        "DO NOT render, burn-in, overlay, or generate ANY text, subtitles, captions, "
        "speech bubbles, words, letters, numbers or watermarks in the video. "
        "The dialogue exists ONLY in the audio track."
    )


def _ffprobe_duration(video_path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ])
    return float(out.decode().strip())


def _extract_audio_segment(source_video: Path, start: float, end: float, out_path: Path, min_dur: float = 1.05) -> None:
    duration = end - start
    if duration < min_dur:
        cmd = [
            "ffmpeg", "-y", "-i", str(source_video),
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-vn", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-af", "apad", "-t", f"{min_dur:.3f}",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(source_video),
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-vn", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            str(out_path),
        ]
    subprocess.run(cmd, check=True, capture_output=True)


def _render_chunk(
    chunk: Chunk, scene: dict, cast: list[CastMember],
    source_video: Path, work_dir: Path,
    animation_provider: PrunaaiAnimationProvider,
    min_chunk_dur: float = 1.05,
) -> Path:
    raw = chunk.raw_path
    if raw.exists() and raw.stat().st_size > 0:
        return raw
    audio_path = work_dir / f"audio_chunk_{chunk.chunk_id}.mp3"
    if not audio_path.exists():
        _extract_audio_segment(source_video, chunk.start, chunk.end, audio_path, min_chunk_dur)
    prompt = _build_chunk_video_prompt(chunk, scene, cast)
    logger.info("scenes [%s] %s [%.2f-%.2fs]…", chunk.chunk_id, chunk.character, chunk.start, chunk.end)
    video_bytes = animation_provider.animate(chunk.master_path, audio_path, video_prompt=prompt)
    raw.write_bytes(video_bytes)
    return raw


# ---------------------------------------------------------------------------
# Concat com crossfades (preserva duração via tpad)
# ---------------------------------------------------------------------------

def _extend_video_by(video_path: Path, extra_dur: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={extra_dur:.3f}",
        "-af", f"apad=pad_dur={extra_dur:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _extend_chunk_to_gap(chunk_video: Path, freeze_duration: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(chunk_video),
        "-vf", f"tpad=stop_mode=clone:stop_duration={freeze_duration:.3f}",
        "-af", f"apad=pad_dur={freeze_duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_videos_with_crossfades(
    chunks: list[Chunk], out_path: Path, work_dir: Path,
    total_duration: float, *, crossfade_dur: float = 0.25,
) -> None:
    chunks_sorted = sorted(chunks, key=lambda c: c.start)
    base = []
    for i, c in enumerate(chunks_sorted):
        next_start = chunks_sorted[i + 1].start if i + 1 < len(chunks_sorted) else total_duration
        gap = max(0.0, next_start - c.end)
        if gap > 0.05:
            ext = work_dir / f"ext_{c.chunk_id}.mp4"
            if not ext.exists():
                _extend_chunk_to_gap(c.raw_path, gap, ext)
            base.append(ext)
        else:
            base.append(c.raw_path)

    xfade_dir = work_dir / "_xfade"
    xfade_dir.mkdir(exist_ok=True, parents=True)
    padded = []
    for i, p in enumerate(base):
        if i < len(base) - 1:
            pad = xfade_dir / f"pad_{i:02d}.mp4"
            if not pad.exists():
                _extend_video_by(p, crossfade_dur, pad)
            padded.append(pad)
        else:
            padded.append(p)

    current = padded[0]
    for i, nxt in enumerate(padded[1:], 1):
        step = xfade_dir / f"step_{i:02d}.mp4"
        if not step.exists():
            dur = _ffprobe_duration(current)
            offset = max(0.0, dur - crossfade_dur)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(current), "-i", str(nxt),
                "-filter_complex",
                f"[0:v][1:v]xfade=transition=fade:duration={crossfade_dur}:offset={offset}[v];"
                f"[0:a][1:a]acrossfade=d={crossfade_dur}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                str(step),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        current = step

    import shutil
    shutil.copy(current, out_path)


# ---------------------------------------------------------------------------
# Final encode com scale+crop + watermark
# ---------------------------------------------------------------------------

def _final_encode(
    input_path: Path, output_path: Path,
    width: int, height: int,
    *, watermark_text: str | None,
    watermark_opacity: float,
    watermark_y_from_bottom: int,
    extra_filter: str = "",
    font_size: int = 44,
) -> None:
    """Scale+crop pra width×height + watermark opcional. Inclui extra_filter (ass) se passado."""
    parts = [f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
             f"crop={width}:{height}"]
    if extra_filter:
        parts.append(extra_filter)
    if watermark_text:
        parts.append(
            f"drawtext=text='{watermark_text}':"
            f"fontfile={WATERMARK_FONT}:"
            f"fontsize={font_size}:"
            f"fontcolor=white@{watermark_opacity:.2f}:"
            f"x=(w-text_w)/2:"
            f"y=h-{watermark_y_from_bottom}:"
            f"shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )
    vf = ",".join(parts)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _video_output_dir(video_path: Path, output_root: Path) -> Path:
    return output_root / video_path.stem


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def run_scenes_pipeline(
    video_path: Path,
    config: PipelineConfig,
    *,
    session_id: str | None = None,
    callbacks=None,
    image_provider: ImageProvider | None = None,
    animation_provider: PrunaaiAnimationProvider | None = None,
) -> MotionComicSession:
    """Executa o pipeline scenes (multi-cena com word-level lip-sync)."""
    from youcut.comic.pipeline import PipelineCallbacks  # lazy import

    callbacks = callbacks or PipelineCallbacks()
    callbacks.on_stage("validate", {"video_path": str(video_path)})
    video_spec = validate_video(video_path)

    output_dir = _video_output_dir(video_spec.path, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "comic" / "_scenes"
    work_dir.mkdir(parents=True, exist_ok=True)

    existing: MotionComicSession | None = None
    if session_id:
        try:
            existing = load_motion_comic_session(session_id)
        except FileNotFoundError as exc:
            raise ScenesPipelineError(str(exc)) from exc

    # 1) Transcript
    callbacks.on_stage("transcribe", {})
    transcription = transcribe(video_spec.path, config)
    audio_duration = transcription.segments[-1].end if transcription.segments else 0.0
    logger.info("scenes: transcrição %d segmentos, %.2fs", len(transcription.segments), audio_duration)

    # 2) Diarização (apenas pra cast detection — não usado pro lip-sync)
    callbacks.on_stage("diarize", {})
    speakers: list[SpeakerSegment] = diarize(video_spec.path, config)

    # 3) Cast
    cast_already_built = False
    if existing and existing.cast:
        cast = existing.cast
        cast_already_built = all(
            m.anchor_image_path is not None
            and Path(m.anchor_image_path).exists()
            and Path(m.anchor_image_path).stat().st_size > 0
            for m in cast
        )
        callbacks.on_stage("cast_reused", {"n": len(cast)})
    else:
        if config.comic_invent_cast:
            callbacks.on_stage("cast_invent", {})
            cast = invent_cast(transcription, speakers, config)
        else:
            callbacks.on_stage("visual_analyzer", {})
            cast = detect_cast(video_spec.path, transcription, speakers, config, output_dir=output_dir)
        if not callbacks.confirm_cast(cast):
            raise ScenesPipelineError("Pipeline abortado pelo usuário durante revisão do cast.")

    image_provider = image_provider or OpenAIImageProvider(
        api_key=config.openai_api_key, max_retries=config.comic_image_retries,
    )
    if not cast_already_built:
        callbacks.on_stage("cast_anchors", {"n": len(cast)})
        cast = build_cast(cast, output_dir, config, image_provider=image_provider)

    # 4) Plano de cenas (cache)
    scenes_path = work_dir / "scenes.json"
    if scenes_path.exists():
        scenes = json.loads(scenes_path.read_text())
        callbacks.on_stage("scenes_reused", {"n": len(scenes)})
    else:
        callbacks.on_stage("scenes_plan", {})
        scenes = plan_scenes(transcription, cast, audio_duration, config)
        scenes_path.write_text(json.dumps(scenes, indent=2, ensure_ascii=False))

    # 5) Visual anchor + scene masters
    callbacks.on_stage("scenes_anchor", {})
    anchor_path = generate_visual_anchor(cast, config.comic_scene_seed, work_dir, image_provider, config)
    callbacks.on_stage("scenes_masters", {"n": len(scenes)})
    masters = generate_scene_masters(scenes, cast, anchor_path, work_dir, image_provider, config)

    # 6) Word-level visual attribution (cache)
    attr_path = work_dir / "speaker_attribution_word.json"
    if attr_path.exists():
        attributions = {int(k): v for k, v in json.loads(attr_path.read_text()).items()}
        callbacks.on_stage("scenes_attribution_reused", {"n": len(attributions)})
    else:
        callbacks.on_stage("scenes_attribution", {})
        words = _flatten_words(transcription)
        attributions = attribute_words_visual(
            video_spec.path, words, scenes, cast, work_dir, config,
            inter_call_pause_s=60.0,  # rate-limit Claude
        )
        attr_path.write_text(json.dumps({str(k): v for k, v in attributions.items()}, indent=2, ensure_ascii=False))

    if config.comic_scenes_smooth_attribution:
        words_for_smooth = _flatten_words(transcription)
        attributions = smooth_attributions(words_for_smooth, attributions)

    # 7) Build chunks
    words = _flatten_words(transcription)
    chunks = build_chunks_from_words(words, attributions, scenes, masters, work_dir)
    chunks = absorb_long_gaps(chunks, gap_threshold=config.comic_scenes_gap_absorb_threshold)
    chunks = expand_short_chunks(chunks, audio_duration, min_dur=config.comic_scenes_min_chunk_dur)
    reconcile_cache(chunks, work_dir)

    # 8) Render chunks (sequencial, rate-limit)
    callbacks.on_stage("scenes_render", {"n": len(chunks)})
    animation_provider = animation_provider or PrunaaiAnimationProvider(
        api_token=config.replicate_api_token,
        max_retries=5, backoff_base=2.0, backoff_cap=20.0,
    )
    scene_by_id = {s["scene_id"]: s for s in scenes}
    pause = config.comic_scenes_inter_call_pause_s
    for i, chunk in enumerate(chunks):
        scene = scene_by_id[chunk.scene_id]
        cache_hit = chunk.raw_path.exists() and chunk.raw_path.stat().st_size > 0
        try:
            _render_chunk(chunk, scene, cast, video_spec.path, work_dir, animation_provider,
                          min_chunk_dur=config.comic_scenes_min_chunk_dur)
        except Exception:
            logger.exception("scenes: falha no chunk %s", chunk.chunk_id)
            raise
        if not cache_hit and i < len(chunks) - 1 and pause > 0:
            time.sleep(pause)

    # 9) Concat com crossfades
    callbacks.on_stage("scenes_compose", {})
    concat_path = work_dir / "_concat.mp4"
    concat_videos_with_crossfades(
        chunks, concat_path, work_dir, audio_duration,
        crossfade_dur=config.comic_scenes_crossfade_dur,
    )

    # 10) Mux áudio original
    muxed_path = work_dir / "_muxed.mp4"
    _mux_audio(concat_path, video_spec.path, muxed_path)

    # 11) Build .ass de legendas
    cap_words = _filter_words_in_range(transcription, 0.0, audio_duration + 1.0)
    width = config.comic_output_width
    height = config.comic_output_height
    ass_doc = build_ass_for_words(cap_words, output_size=(width, height), offset=0.0)
    ass_path = work_dir / "captions.ass"
    ass_path.write_text(ass_doc, encoding="utf-8")
    ass_escaped = str(ass_path.absolute()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    ass_filter = f"ass='{ass_escaped}'"

    # 12) Finals
    final_with = output_dir / "motion_comic_scenes.mp4"
    _final_encode(
        muxed_path, final_with, width, height,
        watermark_text=config.comic_scenes_watermark_text,
        watermark_opacity=config.comic_scenes_watermark_opacity,
        watermark_y_from_bottom=config.comic_scenes_watermark_y_from_bottom,
        extra_filter=ass_filter,
    )
    callbacks.on_stage("scenes_done_with_subs", {"path": str(final_with)})

    final_no_subs = None
    if config.comic_scenes_emit_no_subs_version:
        final_no_subs = output_dir / "motion_comic_scenes_no_subs.mp4"
        _final_encode(
            muxed_path, final_no_subs, width, height,
            watermark_text=config.comic_scenes_watermark_text,
            watermark_opacity=config.comic_scenes_watermark_opacity,
            watermark_y_from_bottom=config.comic_scenes_watermark_y_from_bottom,
        )
        callbacks.on_stage("scenes_done_no_subs", {"path": str(final_no_subs)})

    # 13) Sessão
    n_anchors = sum(1 for m in cast if m.anchor_image_path and Path(m.anchor_image_path).exists())
    estimated_cost = round(0.04 * (n_anchors + 1 + len(scenes)) + 0.034 * len(chunks) + 0.02, 4)
    session = MotionComicSession(
        session_id=session_id or _new_session_id(),
        video_path=video_spec.path,
        created_at=datetime.now(timezone.utc),
        cast=cast,
        panels=[],
        panel_results=[],
        total_cost_usd=estimated_cost,
        output_path=final_with,
    )
    save_motion_comic_session(session)
    callbacks.on_stage("done", {"path": str(final_with), "cost": estimated_cost})
    return session
