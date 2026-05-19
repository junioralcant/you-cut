"""Detecção de apresentadores via Claude vision.

Amostra N frames esparsos do vídeo source, envia para o Claude vision
junto com as imagens-âncora do catálogo, e pede pra identificar qual
(quais) dos candidatos aparece nos frames.

Falha gracioso: se o Anthropic API estiver indisponível (créditos
esgotados, rede caída, sem cliente), retorna uma detecção vazia em
vez de quebrar o pipeline.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from youcut.presenters.catalog import PresenterCatalog
from youcut.presenters.models import PresenterDetection, PresenterProfile

logger = logging.getLogger(__name__)

_DEFAULT_FRAME_COUNT = 5
_DEFAULT_TIMESTAMPS_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _video_duration_seconds(video_path: Path) -> float:
    """Retorna duração via ``ffprobe``. ``0.0`` se falhar."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError) as exc:
        logger.warning("ffprobe falhou em %s: %s", video_path, exc)
        return 0.0


def _extract_frame_at(video_path: Path, timestamp: float, output_path: Path) -> bool:
    """Extrai 1 frame no timestamp dado via ``ffmpeg``. Retorna True se OK."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{timestamp:.2f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "scale=512:-1",  # menor pra reduzir tokens vision
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except subprocess.SubprocessError as exc:
        logger.debug("ffmpeg frame extract falhou @ %.2fs: %s", timestamp, exc)
        return False


def _sample_video_frames(video_path: Path, count: int) -> list[bytes]:
    """Amostra ``count`` frames esparsos do vídeo. Bytes PNG."""
    duration = _video_duration_seconds(video_path)
    if duration <= 0:
        return []
    fractions = _DEFAULT_TIMESTAMPS_FRACTIONS[:count]
    frames: list[bytes] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, fraction in enumerate(fractions):
            ts = duration * fraction
            frame_path = tmpdir_path / f"frame_{idx:02d}.png"
            if _extract_frame_at(video_path, ts, frame_path):
                frames.append(frame_path.read_bytes())
    return frames


def _image_block(image_bytes: bytes, mime: str = "image/png") -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        },
    }


def _build_vision_content(
    catalog_profiles: list[PresenterProfile],
    video_frames: list[bytes],
) -> list[dict[str, Any]]:
    """Constrói o content block intercalando catálogo + frames do vídeo."""
    blocks: list[dict[str, Any]] = []
    blocks.append({
        "type": "text",
        "text": (
            "Você vai receber uma lista de imagens-âncora de apresentadores "
            "(cada uma com um label) seguida de frames extraídos de um vídeo. "
            "Sua tarefa: identificar quais dos apresentadores das âncoras "
            "aparecem nos frames do vídeo.\n\n"
            "ÂNCORAS:"
        ),
    })
    for profile in catalog_profiles:
        try:
            img_bytes = profile.image_path.read_bytes()
        except OSError as exc:
            logger.warning("Falha ao ler %s: %s", profile.image_path, exc)
            continue
        mime = "image/webp" if profile.image_path.suffix.lower() == ".webp" else "image/png"
        blocks.append({"type": "text", "text": f"  Label: {profile.slug} (nome: {profile.display_name})"})
        blocks.append(_image_block(img_bytes, mime=mime))
    blocks.append({
        "type": "text",
        "text": (
            "\nAGORA OS FRAMES DO VÍDEO:"
        ),
    })
    for idx, frame_bytes in enumerate(video_frames):
        blocks.append({"type": "text", "text": f"  Frame {idx}:"})
        blocks.append(_image_block(frame_bytes))
    blocks.append({
        "type": "text",
        "text": (
            "\nResponda APENAS em JSON, sem texto adicional, no formato:\n"
            "{\"detected_slugs\": [\"slug1\", \"slug2\", ...]}\n"
            "Inclua um slug se o rosto da âncora correspondente aparecer em "
            "pelo menos UM frame do vídeo. Se nenhum aparecer, retorne "
            "{\"detected_slugs\": []}. Use apenas slugs exatamente como "
            "listados nas âncoras."
        ),
    })
    return blocks


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Extrai o primeiro objeto JSON da resposta do Claude."""
    try:
        first = text.index("{")
        last = text.rindex("}") + 1
        return json.loads(text[first:last])
    except (ValueError, json.JSONDecodeError):
        return {}


def detect_presenters(
    video_path: Path,
    catalog: PresenterCatalog,
    anthropic_client: Any | None,
    claude_model: str,
    frame_count: int = _DEFAULT_FRAME_COUNT,
) -> PresenterDetection:
    """Identifica apresentadores nos frames do vídeo via Claude vision.

    Retorna :class:`PresenterDetection` com:
    - ``profiles`` vazia se o catálogo está vazio, Claude indisponível,
      vídeo sem frames extraíveis, ou se o LLM disse "ninguém apareceu".
    - ``source_method == "vision"`` quando o Claude foi consultado com
      sucesso (mesmo retornando lista vazia).
    """
    if not catalog.profiles:
        return PresenterDetection(profiles=[], source_method="vision")
    if anthropic_client is None:
        logger.info("Detecção de apresentadores pulada — sem cliente Anthropic")
        return PresenterDetection(profiles=[], source_method="vision")

    video_frames = _sample_video_frames(video_path, frame_count)
    if not video_frames:
        logger.warning("Detecção de apresentadores: nenhum frame amostrado de %s", video_path)
        return PresenterDetection(profiles=[], source_method="vision")

    content = _build_vision_content(catalog.profiles, video_frames)
    try:
        response = anthropic_client.messages.create(
            model=claude_model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        logger.warning("Claude vision (presenters) falhou: %s — seguindo sem apresentador", exc)
        return PresenterDetection(profiles=[], source_method="vision")

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")
    payload = _extract_json_payload(text.strip())
    slugs_raw = payload.get("detected_slugs") or []
    if not isinstance(slugs_raw, list):
        return PresenterDetection(profiles=[], source_method="vision")

    detected: list[PresenterProfile] = []
    seen: set[str] = set()
    for slug in slugs_raw:
        if not isinstance(slug, str):
            continue
        key = slug.strip().lower()
        if not key or key in seen:
            continue
        profile = catalog.get(key)
        if profile is None:
            logger.debug("Slug retornado pelo Claude não está no catálogo: %r", slug)
            continue
        seen.add(key)
        detected.append(profile)

    logger.info(
        "Detecção de apresentadores: %d encontrado(s) — %s",
        len(detected),
        [p.slug for p in detected],
    )
    return PresenterDetection(profiles=detected, source_method="vision")
