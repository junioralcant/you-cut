"""Visual Analyzer — detecção de cast com MediaPipe + Claude vision (RF-04..RF-06, RF-10)."""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

import anthropic

from youcut.config import PipelineConfig
from youcut.models import (
    CastKind,
    CastMember,
    SpeakerSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


MAX_REFERENCE_FRAMES: int = 8
DEFAULT_MIN_FACE_CONFIDENCE: float = 0.5


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

_CAST_TOOL: dict[str, Any] = {
    "name": "extract_cast",
    "description": (
        "Extrai a lista de pessoas, animais ou objetos com presença narrativa "
        "visíveis nos frames de referência. Cada entrada deve descrever o "
        "personagem com fidelidade visível, sem suposições sobre identidade real."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["person", "animal", "object"],
                            "description": "Tipo do personagem.",
                        },
                        "gender_apparent": {"type": "string"},
                        "age_apparent": {"type": "string"},
                        "hair": {"type": "string"},
                        "facial_hair": {"type": "string"},
                        "skin": {"type": "string"},
                        "clothing": {"type": "string"},
                        "accessories": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "narrative_role": {
                            "type": "string",
                            "description": "Papel narrativo curto (ex.: protagonista, narrador, cachorro).",
                        },
                        "spatial_position": {
                            "type": "string",
                            "enum": ["left", "center", "right", "unknown"],
                            "description": "Posição horizontal dominante na cena.",
                        },
                        "reference_frame_index": {
                            "type": "integer",
                            "description": (
                                "Índice (0-based) do frame que melhor representa visualmente "
                                "este personagem — escolha o frame em que o rosto/forma do "
                                "personagem aparece mais claro, mais frontal e maior. "
                                "Os frames estão indexados na ordem em que aparecem abaixo. "
                                "Omita o campo se nenhum frame mostra o personagem com clareza."
                            ),
                        },
                    },
                    "required": ["kind", "narrative_role"],
                },
            }
        },
        "required": ["characters"],
    },
}


_SYSTEM_PROMPT = (
    "Você é um analista visual cuidadoso. Receberá frames amostrados de um "
    "vídeo curto e deverá identificar todos os personagens visíveis com "
    "presença narrativa: pessoas, animais ou objetos centrais (ex.: moto, "
    "instrumento). Para cada personagem retorne apenas características "
    "visíveis no frame, sem inferir identidade real. Use vocabulário em "
    "pt-BR, descrições curtas e objetivas."
)

_USER_PROMPT_TEMPLATE = (
    "Analise os {n_frames} frames a seguir (indexados de 0 a {last_index}, "
    "na ordem exibida) e identifique os personagens (pessoas, animais ou "
    "objetos com presença narrativa) presentes em cena. Para cada um "
    "retorne os campos do schema. Para personagens do tipo `object` ou "
    "`animal`, descreva também sua função narrativa.\n\n"
    "IMPORTANTE: para cada pessoa, preencha `reference_frame_index` com o "
    "índice do frame em que o rosto aparece mais claro e frontal — esse "
    "frame será usado como referência visual direta para gerar a caricatura "
    "do personagem.\n\n"
    "Resumo da fala (para contexto):\n{transcript_excerpt}"
)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def _sample_reference_frames(
    video_path: Path,
    *,
    max_frames: int = MAX_REFERENCE_FRAMES,
    min_confidence: float = DEFAULT_MIN_FACE_CONFIDENCE,
) -> list[dict[str, Any]]:
    """Amostra até ``max_frames`` frames priorizando aqueles com rostos.

    Retorna lista de dicts: ``{"timestamp": float, "png_bytes": bytes,
    "faces": list[(x, y, w, h)]}``. Quando MediaPipe e OpenCV falham,
    devolve amostragem temporal uniforme com ``faces=[]``.
    """

    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        logger.info("comic.visual_analyzer: OpenCV ausente (%s)", exc)
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("comic.visual_analyzer: não foi possível abrir %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0 or fps <= 0:
        cap.release()
        return []

    sample_count = max(1, min(max_frames, total))
    positions = [int(total * (i + 0.5) / sample_count) for i in range(sample_count)]

    from youcut.face_tracker import _detect_faces_with_mediapipe, _detect_faces_with_opencv

    samples: list[dict[str, Any]] = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        faces = _detect_faces_with_mediapipe(frame, min_confidence)
        if not faces:
            faces = _detect_faces_with_opencv(frame)
        ok2, png = cv2.imencode(".png", frame)
        if not ok2:
            continue
        samples.append(
            {
                "timestamp": pos / fps,
                "png_bytes": png.tobytes(),
                "faces": [tuple(int(v) for v in (b.x, b.y, b.w, b.h)) for b in faces],
            }
        )

    cap.release()

    samples.sort(key=lambda s: (-len(s["faces"]), s["timestamp"]))
    return samples[:max_frames]


# ---------------------------------------------------------------------------
# Claude vision call
# ---------------------------------------------------------------------------


def _build_image_blocks(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for s in samples:
        b64 = base64.b64encode(s["png_bytes"]).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }
        )
    return blocks


def _truncate_transcript(transcription: TranscriptionResult, max_chars: int = 1500) -> str:
    text = " ".join(seg.text.strip() for seg in transcription.segments if seg.text.strip())
    if len(text) <= max_chars:
        return text or "(sem texto transcrito)"
    return text[:max_chars] + "…"


def _persist_sampled_frames(
    samples: list[dict[str, Any]], output_dir: Path
) -> list[Path]:
    """Grava os frames amostrados em ``<output_dir>/comic/frames/`` e retorna os paths
    na mesma ordem em que serão exibidos para o Claude (o índice na lista corresponde
    ao ``reference_frame_index`` que o modelo deve devolver).
    """

    frames_dir = Path(output_dir) / "comic" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, sample in enumerate(samples):
        path = frames_dir / f"frame_{idx:02d}.png"
        path.write_bytes(sample["png_bytes"])
        paths.append(path)
    return paths


def _call_claude_vision(
    client: anthropic.Anthropic,
    samples: list[dict[str, Any]],
    transcription: TranscriptionResult,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    if not samples:
        return []

    transcript_excerpt = _truncate_transcript(transcription)
    user_text = _USER_PROMPT_TEMPLATE.format(
        n_frames=len(samples),
        last_index=len(samples) - 1,
        transcript_excerpt=transcript_excerpt,
    )
    content: list[dict[str, Any]] = list(_build_image_blocks(samples))
    content.append({"type": "text", "text": user_text})

    try:
        response = client.with_options(timeout=60.0).messages.create(
            model=config.claude_model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[_CAST_TOOL],
            tool_choice={"type": "tool", "name": "extract_cast"},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIError as exc:
        msg = getattr(exc, "message", None) or str(exc)
        raise RuntimeError(
            f"Erro na API do Claude ao analisar cast visual: {msg}"
        ) from exc

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "extract_cast":
            payload = getattr(block, "input", None) or {}
            return list(payload.get("characters") or [])
    return []


# ---------------------------------------------------------------------------
# Speaker mapping
# ---------------------------------------------------------------------------


def _unique_speaker_ids(speakers: list[SpeakerSegment]) -> list[str]:
    seen: dict[str, None] = {}
    for seg in speakers:
        seen.setdefault(seg.speaker_id, None)
    return list(seen.keys())


def _map_speakers_to_persons(
    persons: list[dict[str, Any]],
    speakers: list[SpeakerSegment],
) -> dict[int, str]:
    speaker_ids = _unique_speaker_ids(speakers)
    if not persons or not speaker_ids:
        return {}
    if len(persons) > 2 or len(speaker_ids) > 2:
        return {}

    def _spatial_rank(person: dict[str, Any]) -> int:
        pos = (person.get("spatial_position") or "unknown").lower()
        return {"left": 0, "center": 1, "right": 2}.get(pos, 1)

    ordered_indices = sorted(range(len(persons)), key=lambda i: (_spatial_rank(persons[i]), i))
    mapping: dict[int, str] = {}
    for slot, idx in enumerate(ordered_indices):
        if slot < len(speaker_ids):
            mapping[idx] = speaker_ids[slot]
    return mapping


# ---------------------------------------------------------------------------
# Cast assembly
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or "x"


def _build_text_card(member: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("gender_apparent", "age_apparent", "hair", "facial_hair", "skin", "clothing"):
        val = (member.get(key) or "").strip()
        if val:
            parts.append(val)
    accessories = [a for a in (member.get("accessories") or []) if a]
    if accessories:
        parts.append("acessórios: " + ", ".join(accessories))
    role = (member.get("narrative_role") or "").strip()
    if role:
        parts.append(f"papel: {role}")
    return "; ".join(parts) or (role or "personagem genérico")


def _assemble_cast(
    raw_characters: list[dict[str, Any]],
    speakers: list[SpeakerSegment],
    frame_paths: list[Path] | None = None,
) -> list[CastMember]:
    persons_indexes = [i for i, c in enumerate(raw_characters) if (c.get("kind") or "person") == "person"]
    persons = [raw_characters[i] for i in persons_indexes]
    person_speaker_map = _map_speakers_to_persons(persons, speakers)

    frame_paths = frame_paths or []

    cast: list[CastMember] = []
    used_ids: set[str] = set()
    for idx, raw in enumerate(raw_characters):
        kind: CastKind = raw.get("kind") or "person"  # type: ignore[assignment]
        role = (raw.get("narrative_role") or kind).strip() or kind
        base_slug = _slugify(role)
        slug = base_slug
        n = 1
        while slug in used_ids:
            n += 1
            slug = f"{base_slug}_{n}"
        used_ids.add(slug)

        speaker_id: str | None = None
        if kind == "person" and idx in persons_indexes:
            local_idx = persons_indexes.index(idx)
            speaker_id = person_speaker_map.get(local_idx)

        source_frame_path: Path | None = None
        ref_idx = raw.get("reference_frame_index")
        if isinstance(ref_idx, int) and 0 <= ref_idx < len(frame_paths):
            source_frame_path = frame_paths[ref_idx]

        cast.append(
            CastMember(
                character_id=slug,
                kind=kind,
                gender_apparent=(raw.get("gender_apparent") or "").strip(),
                age_apparent=(raw.get("age_apparent") or "").strip(),
                hair=(raw.get("hair") or "").strip(),
                facial_hair=(raw.get("facial_hair") or "").strip(),
                skin=(raw.get("skin") or "").strip(),
                clothing=(raw.get("clothing") or "").strip(),
                accessories=[a for a in (raw.get("accessories") or []) if a],
                narrative_role=role,
                speaker_id=speaker_id,
                source_frame_path=source_frame_path,
                text_card=_build_text_card(raw),
            )
        )
    return cast


def _generic_fallback_cast(transcription: TranscriptionResult) -> list[CastMember]:
    snippet = _truncate_transcript(transcription, max_chars=200)
    text_card = (
        "personagem genérico inferido do áudio (nenhum rosto detectado nos frames). "
        f"Resumo: {snippet}"
    )
    return [
        CastMember(
            character_id="narrator",
            kind="person",
            narrative_role="narrador",
            text_card=text_card,
        )
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_cast(
    video_path: Path,
    transcription: TranscriptionResult,
    speakers: list[SpeakerSegment],
    config: PipelineConfig,
    *,
    output_dir: Path | None = None,
    client: anthropic.Anthropic | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> list[CastMember]:
    """Detecta o cast do vídeo (RF-04, RF-05, RF-06, RF-10).

    Quando ``output_dir`` é informado, os frames amostrados são persistidos em
    ``<output_dir>/comic/frames/`` e o índice escolhido pelo Claude
    (``reference_frame_index``) é mapeado para ``CastMember.source_frame_path``,
    permitindo que o ``cast_builder`` use o frame real como referência visual
    do personagem ao gerar a ficha-âncora.

    ``samples`` e ``client`` podem ser injetados para testes.
    """

    if samples is None:
        samples = _sample_reference_frames(
            video_path,
            max_frames=MAX_REFERENCE_FRAMES,
            min_confidence=config.face_detection_confidence,
        )

    if not samples or all(not s.get("faces") for s in samples):
        logger.warning(
            "comic.visual_analyzer: sem rostos detectados; usando cast genérico (1 personagem)."
        )
        return _generic_fallback_cast(transcription)

    frame_paths: list[Path] = []
    if output_dir is not None:
        frame_paths = _persist_sampled_frames(samples, output_dir)

    if client is None:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    raw_characters = _call_claude_vision(client, samples, transcription, config)
    if not raw_characters:
        logger.warning(
            "comic.visual_analyzer: Claude vision não retornou personagens; usando fallback."
        )
        return _generic_fallback_cast(transcription)

    cast = _assemble_cast(raw_characters, speakers, frame_paths=frame_paths)
    logger.info(
        "comic.visual_analyzer: %d personagens detectados (%d com source_frame_path)",
        len(cast),
        sum(1 for m in cast if m.source_frame_path is not None),
    )
    return cast
