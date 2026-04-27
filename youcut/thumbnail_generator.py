import base64
import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from youcut.models import ClipRecord, ThumbnailFrameResult, ViralClip

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont
    from youcut.config import PipelineConfig

logger = logging.getLogger(__name__)

_THUMBNAIL_W = 1280
_THUMBNAIL_H = 720
_FRAME_MAX_W = 1000
_FRAME_MAX_H = 563
_MAX_TEXT_AREA_RATIO = 0.07
_MIN_HERO_FONT_SIZE = 44
_MIN_SECONDARY_FONT_SIZE = 24
_ASSETS_DIR = Path(__file__).parent / "assets"
_FONT_PATH = _ASSETS_DIR / "Roboto-Regular.ttf"
_BOLD_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Black.ttf")


def generate_thumbnail(
    clip: ViralClip,
    output_dir: Path,
    clip_index: int,
    clip_path: Path | None = None,
    config: "PipelineConfig | None" = None,
) -> Path:
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    output_path = thumbnails_dir / f"clip_{clip_index:02d}.png"

    source = clip_path if clip_path and clip_path.exists() else None
    if source is None:
        raise ValueError(f"clip_path não fornecido ou não existe: {clip_path}")

    result = _build_thumbnail_result(clip, source, output_path, config)
    logger.info(
        "Thumbnail gerada: path=%s timestamp=%.2f selection_method=%s generation_method=%s",
        result.output_path,
        result.frame_timestamp,
        result.selection_method,
        result.generation_method,
    )
    logger.info("ThumbnailFrameResult: %s", json.dumps({
        "frame_timestamp": result.frame_timestamp,
        "frame_score": result.frame_score,
        "segmentation_applied": result.segmentation_applied,
        "output_path": str(result.output_path),
        "selection_method": result.selection_method,
        "generation_method": result.generation_method,
    }))
    return result.output_path


def regenerate_thumbnail(
    clip: ViralClip,
    clip_record: ClipRecord,
    config: "PipelineConfig | str | None" = None,
) -> Path:
    if clip_record.thumbnail_path is not None:
        output_path = clip_record.thumbnail_path
    else:
        output_path = clip_record.clip_path.parent / "thumbnails" / f"{clip_record.clip_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = _build_thumbnail_result(clip, clip_record.clip_path, output_path, config)
    logger.info(
        "Thumbnail regenerada: path=%s timestamp=%.2f selection_method=%s generation_method=%s",
        result.output_path,
        result.frame_timestamp,
        result.selection_method,
        result.generation_method,
    )
    return result.output_path


def _build_thumbnail_result(
    clip: ViralClip,
    source: Path,
    output_path: Path,
    config: "PipelineConfig | str | None",
) -> ThumbnailFrameResult:
    frames = _extract_frames_candidates(source)
    anthropic_client, openai_client = _build_ai_clients(config)

    if anthropic_client is not None:
        frame_timestamp, frame_bytes, selection_method, frame_score = _select_best_frame_via_ai_result(
            frames,
            clip.title,
            anthropic_client,
        )
    else:
        frame_timestamp, frame_bytes, frame_score = _select_best_local_candidate(frames)
        selection_method = "local"
        logger.warning("Seleção por IA indisponível; usando método local")

    if openai_client is not None:
        thumbnail_bytes, generation_method = _generate_thumbnail_via_ai_result(
            frame_bytes,
            clip.title,
            openai_client,
        )
    else:
        thumbnail_bytes, generation_method = _generate_thumbnail_via_ai_result(
            frame_bytes,
            clip.title,
            None,
        )

    image = _load_image_from_bytes(thumbnail_bytes)
    composed = _compose_text_overlay(image, clip.thumbnail_text)
    composed.save(output_path, format="PNG", optimize=True)
    _resize_to_youtube_format(output_path)

    if generation_method == "ai":
        logger.info("Thumbnail gerada por IA: method=ai output_path=%s", output_path)
    else:
        logger.warning("Thumbnail gerada por fallback local: method=local output_path=%s", output_path)

    return ThumbnailFrameResult(
        frame_timestamp=frame_timestamp,
        frame_score=frame_score,
        segmentation_applied=False,
        output_path=output_path,
        selection_method=selection_method,
        generation_method=generation_method,
    )


def _build_ai_clients(
    config: "PipelineConfig | str | None",
) -> tuple[Any | None, Any | None]:
    anthropic_client = None
    openai_client = None

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    if isinstance(config, str):
        openai_api_key = config
    elif config is not None:
        anthropic_api_key = getattr(config, "anthropic_api_key", None)
        openai_api_key = getattr(config, "openai_api_key", None)

    if anthropic_api_key:
        try:
            from anthropic import Anthropic  # type: ignore[import]
            anthropic_client = Anthropic(api_key=anthropic_api_key)
        except Exception as exc:
            logger.warning("Falha ao inicializar cliente Anthropic: %s", exc)

    if openai_api_key:
        try:
            from openai import OpenAI  # type: ignore[import]
            openai_client = OpenAI(api_key=openai_api_key)
        except Exception as exc:
            logger.warning("Falha ao inicializar cliente OpenAI: %s", exc)

    return anthropic_client, openai_client


def _extract_frame(clip_path: Path) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-sseof", "-1",
                "-i", str(clip_path),
                "-ss", "00:00:01",
                "-vframes", "1",
                "-q:v", "2",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(clip_path), "-vframes", "1", "-q:v", "2", str(tmp_path)],
                capture_output=True,
                timeout=30,
                check=True,
            )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _get_video_duration(clip_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(clip_path),
        ],
        capture_output=True,
        timeout=10,
        check=True,
        text=True,
    )
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise RuntimeError(f"Não foi possível determinar a duração do vídeo: {clip_path}")
    return duration


def _extract_frame_at(clip_path: Path, timestamp: float) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        capture_output=True,
        timeout=15,
        check=True,
    )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg retornou frame vazio para timestamp {timestamp}")
    return result.stdout


def _extract_frames_candidates(clip_path: str | Path, n_frames: int = 10) -> list[tuple[float, bytes]]:
    clip_path = Path(clip_path)
    frame_count = max(5, min(15, n_frames))
    duration = _get_video_duration(clip_path)

    start = duration * 0.05
    end = duration * 0.95
    if end <= start:
        start = 0.0
        end = duration

    if frame_count == 1:
        timestamps = [start]
    else:
        step = (end - start) / (frame_count - 1)
        timestamps = [start + (step * index) for index in range(frame_count)]

    candidates: list[tuple[float, bytes]] = []
    for timestamp in timestamps:
        try:
            frame_bytes = _extract_frame_at(clip_path, timestamp)
            resized = _resize_frame_bytes(frame_bytes, max_width=_FRAME_MAX_W, max_height=_FRAME_MAX_H)
            candidates.append((timestamp, resized))
        except Exception as exc:
            logger.warning("Falha ao extrair frame candidato em %.2fs: %s", timestamp, exc)

    if not candidates:
        raise RuntimeError(f"Nenhum frame candidato pôde ser extraído de {clip_path}")

    return candidates


def _score_frame(frame_bytes: bytes, detector: Any) -> float:
    try:
        import cv2  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        return 0.0

    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return 0.0

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if not result.detections:
        return 0.0

    cx_frame = w / 2.0
    cy_frame = h / 2.0
    max_dist = ((cx_frame) ** 2 + (cy_frame) ** 2) ** 0.5
    best = 0.0

    for det in result.detections:
        confidence = det.score[0] if det.score else 0.0
        bb = det.location_data.relative_bounding_box
        face_w = bb.width * w
        face_h = bb.height * h
        area = face_w * face_h
        face_cx = (bb.xmin + bb.width / 2.0) * w
        face_cy = (bb.ymin + bb.height / 2.0) * h
        dist = ((face_cx - cx_frame) ** 2 + (face_cy - cy_frame) ** 2) ** 0.5
        centralization = 1.0 - dist / max_dist if max_dist > 0 else 0.0
        best = max(best, area * confidence * centralization)

    return best


def _select_best_local_candidate(
    frames: list[tuple[float, bytes]],
    min_confidence: float = 0.5,
) -> tuple[float, bytes, float]:
    try:
        import mediapipe as mp  # type: ignore[import]
    except ImportError:
        logger.warning("MediaPipe não disponível — usando primeiro frame candidato")
        timestamp, frame_bytes = frames[0]
        return timestamp, frame_bytes, 0.0

    best_candidate = frames[0]
    best_score = -1.0
    detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)
    try:
        for timestamp, frame_bytes in frames:
            try:
                score = _score_frame(frame_bytes, detector)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_candidate = (timestamp, frame_bytes)
    finally:
        detector.close()

    timestamp, frame_bytes = best_candidate
    if best_score <= 0:
        logger.warning("Nenhum rosto detectado nos frames candidatos — usando primeiro frame disponível")
        return timestamp, frame_bytes, 0.0

    logger.info("Frame local selecionado: method=local timestamp=%.2f score=%.3f", timestamp, best_score)
    return timestamp, frame_bytes, best_score


def _select_best_face_frame(
    clip_path: Path,
    n_samples: int = 10,
    min_confidence: float = 0.5,
) -> bytes:
    try:
        frames = _extract_frames_candidates(clip_path, n_samples)
    except Exception as exc:
        logger.warning("Falha ao extrair frames candidatos: %s — usando fallback para _extract_frame()", exc)
        return _extract_frame(clip_path)

    _, frame_bytes, _ = _select_best_local_candidate(frames, min_confidence=min_confidence)
    return frame_bytes


def _select_best_frame_via_ai(
    frames: list[tuple[float, bytes]],
    title: str,
    anthropic_client: Any,
    timeout: float = 30.0,
) -> tuple[float, bytes]:
    timestamp, frame_bytes, _, _ = _select_best_frame_via_ai_result(frames, title, anthropic_client, timeout)
    return timestamp, frame_bytes


def _select_best_frame_via_ai_result(
    frames: list[tuple[float, bytes]],
    title: str,
    anthropic_client: Any,
    timeout: float = 30.0,
) -> tuple[float, bytes, str, float]:
    if anthropic_client is None:
        timestamp, frame_bytes, frame_score = _select_best_local_candidate(frames)
        logger.warning("Cliente Anthropic ausente; usando fallback local")
        return timestamp, frame_bytes, "local", frame_score

    prompt = (
        "Selecione o frame com maior potencial de CTR para uma thumbnail de YouTube. "
        "Avalie brilho percebido, clareza visual, paleta vibrante (ciano, verde, amarelo, laranja), "
        "expressividade facial e composição narrativa. Penalize frame escuro, vermelho dominante, ruído visual "
        "e composição confusa. Responda somente JSON válido no formato "
        '{"selected_index": <int>, "reason": "<string>"}'
        f'. Título do clipe: "{title}".'
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, (_, frame_bytes) in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {index}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(frame_bytes).decode("utf-8"),
                },
            }
        )

    try:
        client = _client_with_timeout(anthropic_client, timeout)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        payload = _extract_json_payload(_anthropic_response_text(response))
        selected_index = int(payload["selected_index"])
        reason = str(payload.get("reason", "")).strip()
        if not 0 <= selected_index < len(frames):
            raise ValueError(f"selected_index fora do range: {selected_index}")
        timestamp, frame_bytes = frames[selected_index]
        logger.info(
            "Frame selecionado por IA: method=ai timestamp=%.2f reason=%s",
            timestamp,
            reason or "sem motivo informado",
        )
        return timestamp, frame_bytes, "ai", 1.0
    except Exception as exc:
        logger.warning("Falha na seleção por IA; usando fallback local: %s", exc)
        timestamp, frame_bytes, frame_score = _select_best_local_candidate(frames)
        return timestamp, frame_bytes, "local", frame_score


def _generate_thumbnail_via_ai(
    frame_bytes: bytes,
    title: str,
    openai_client: Any,
    timeout: float = 30.0,
) -> bytes:
    thumbnail_bytes, _ = _generate_thumbnail_via_ai_result(frame_bytes, title, openai_client, timeout)
    return thumbnail_bytes


def _generate_thumbnail_via_ai_result(
    frame_bytes: bytes,
    title: str,
    openai_client: Any,
    timeout: float = 30.0,
) -> tuple[bytes, str]:
    if openai_client is None:
        logger.warning("Cliente OpenAI ausente; usando fallback local para geração de thumbnail")
        return _render_local_thumbnail_bytes(frame_bytes), "local"

    prompt = (
        "Create a premium YouTube thumbnail derived from the reference frame. "
        "Preserve the exact identity of the main subject from the reference frame. "
        "Do not change the person's face, age, skin tone, hair, body type, or identity. "
        "Do not replace the person with another character or invent a different subject. "
        "Preserve gaze direction, gesture, clothing cues, and overall scene truthfully. "
        "Increase contrast, subject separation, facial readability, and narrative tension without inventing false events. "
        "Keep all text and critical subjects in the vertical center safe-zone for 16:9 cropping. "
        "The top 12% of the frame should remain mostly empty background space. "
        "Favor cinematic clarity, bold subject framing, clean background separation, and dramatic but realistic lighting. "
        f"Video title context: {title}."
    )

    try:
        client = _client_with_timeout(openai_client, timeout)
        response = client.images.edit(
            model="gpt-image-1",
            image=("frame.png", frame_bytes, "image/png"),
            prompt=prompt,
            size="1536x1024",
            quality="low",
        )
        data = getattr(response, "data", None) or []
        if not data:
            raise ValueError("Resposta de images.edit sem imagens")
        image_item = data[0]
        b64_json = getattr(image_item, "b64_json", None)
        if b64_json is None and isinstance(image_item, dict):
            b64_json = image_item.get("b64_json")
        if not b64_json:
            raise ValueError("Resposta de images.edit sem b64_json")
        image_bytes = base64.b64decode(b64_json)
        resized_bytes = _resize_image_bytes(image_bytes, target_size=(_THUMBNAIL_W, _THUMBNAIL_H))
        logger.info("Thumbnail gerada por IA: method=ai output_path=%s", "<in-memory>")
        return resized_bytes, "ai"
    except Exception as exc:
        logger.warning("Falha na geração por IA; usando fallback local: %s", exc)
        return _render_local_thumbnail_bytes(frame_bytes), "local"


def _load_font(size: int) -> "ImageFont.FreeTypeFont":  # type: ignore[name-defined]
    from PIL import ImageFont

    for path in (_BOLD_FONT_PATH, _FONT_PATH):
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text_centered(
    draw: "ImageDraw.ImageDraw",  # type: ignore[name-defined]
    text: str,
    font: "ImageFont.FreeTypeFont",  # type: ignore[name-defined]
    y: int,
    canvas_w: int,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_w - text_w) // 2
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return text_h


def _split_title_hierarchy(title: str) -> tuple[list[str], str]:
    import textwrap

    words = title.upper().split()
    if len(words) <= 2:
        return [], " ".join(words)

    hero = words[-1]
    secondary_words = words[:-1]
    secondary_lines = textwrap.wrap(" ".join(secondary_words), width=16)
    return secondary_lines, hero


def _compose_text_overlay(image: "Image.Image", title: str) -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageDraw

    img = image.copy().convert("RGBA")
    w, h = img.size

    if not title or not title.strip():
        logger.debug("thumbnail_text vazio; sem sobreposição aplicada")
        return img.convert("RGB")

    secondary_lines, hero = _split_title_hierarchy(title)
    draw = ImageDraw.Draw(img)
    layout = _fit_text_layout(draw, w, h, secondary_lines, hero)

    vignette = Image.new("RGBA", (w, h), (0, 0, 0, 145))
    img = Image.alpha_composite(img, vignette)
    draw = ImageDraw.Draw(img)

    current_y = layout["y_start"]
    for line in layout["secondary_lines"]:
        _draw_text_centered(
            draw,
            line,
            layout["secondary_font"],
            y=current_y,
            canvas_w=w,
            fill=(255, 255, 255, 255),
            stroke_width=layout["secondary_stroke"],
        )
        current_y += layout["secondary_line_h"]

    current_y += layout["gap"]
    _draw_text_centered(
        draw,
        layout["hero"],
        layout["hero_font"],
        y=current_y,
        canvas_w=w,
        fill=(255, 215, 0, 255),
        stroke_width=layout["hero_stroke"],
    )

    return img.convert("RGB")


def _fit_text_layout(
    draw: "ImageDraw.ImageDraw",  # type: ignore[name-defined]
    width: int,
    height: int,
    secondary_lines: list[str],
    hero: str,
) -> dict[str, Any]:
    original_sizes = (int(height * 0.28), int(height * 0.10))
    max_area = width * height * _MAX_TEXT_AREA_RATIO
    hero_text = hero
    secondary_text = list(secondary_lines)

    for attempt in range(10):
        hero_size = max(_MIN_HERO_FONT_SIZE, original_sizes[0] - (attempt * 12))
        secondary_size = max(_MIN_SECONDARY_FONT_SIZE, original_sizes[1] - (attempt * 6))

        while True:
            layout = _measure_text_layout(draw, width, height, secondary_text, hero_text, hero_size, secondary_size)
            if layout["block_area"] <= max_area:
                if (hero_size, secondary_size) != original_sizes:
                    logger.warning(
                        "Texto da thumbnail excedeu 7%%; reduzindo fonte: original_size=%s adjusted_size=%s",
                        original_sizes,
                        (hero_size, secondary_size),
                    )
                return layout
            if hero_size <= _MIN_HERO_FONT_SIZE and secondary_size <= _MIN_SECONDARY_FONT_SIZE:
                break
            if hero_size > _MIN_HERO_FONT_SIZE:
                hero_size = max(_MIN_HERO_FONT_SIZE, hero_size - 6)
            if secondary_size > _MIN_SECONDARY_FONT_SIZE:
                secondary_size = max(_MIN_SECONDARY_FONT_SIZE, secondary_size - 4)

        truncated = _truncate_title(" ".join(secondary_text + [hero_text]))
        if truncated == " ".join(secondary_text + [hero_text]):
            break
        secondary_text, hero_text = _split_title_hierarchy(truncated)

    fallback = _measure_text_layout(
        draw,
        width,
        height,
        secondary_text,
        hero_text,
        _MIN_HERO_FONT_SIZE,
        _MIN_SECONDARY_FONT_SIZE,
    )
    logger.warning(
        "Texto da thumbnail exigiu truncamento para respeitar 7%%: original_size=%s adjusted_size=%s",
        original_sizes,
        (_MIN_HERO_FONT_SIZE, _MIN_SECONDARY_FONT_SIZE),
    )
    return fallback


def _measure_text_layout(
    draw: "ImageDraw.ImageDraw",  # type: ignore[name-defined]
    width: int,
    height: int,
    secondary_lines: list[str],
    hero: str,
    hero_size: int,
    secondary_size: int,
) -> dict[str, Any]:
    hero_font = _load_font(hero_size)
    secondary_font = _load_font(secondary_size)
    hero_stroke = max(4, hero_size // 12)
    secondary_stroke = max(2, secondary_size // 14)
    gap = int(height * 0.018)

    hero_bbox = draw.textbbox((0, 0), hero, font=hero_font, stroke_width=hero_stroke)
    hero_w = hero_bbox[2] - hero_bbox[0]
    hero_h = hero_bbox[3] - hero_bbox[1]
    secondary_dims = [
        draw.textbbox((0, 0), line, font=secondary_font, stroke_width=secondary_stroke)
        for line in secondary_lines
    ]
    secondary_widths = [bbox[2] - bbox[0] for bbox in secondary_dims] or [0]
    secondary_heights = [bbox[3] - bbox[1] for bbox in secondary_dims] or [0]
    secondary_line_h = (max(secondary_heights) + int(secondary_size * 0.25)) if secondary_lines else 0
    secondary_total_h = secondary_line_h * len(secondary_lines)

    block_w = max(max(secondary_widths), hero_w)
    block_h = secondary_total_h + (gap if secondary_lines else 0) + hero_h
    y_start = max(0, (height - block_h) // 2)

    return {
        "secondary_lines": secondary_lines,
        "hero": hero,
        "secondary_font": secondary_font,
        "hero_font": hero_font,
        "secondary_stroke": secondary_stroke,
        "hero_stroke": hero_stroke,
        "secondary_line_h": secondary_line_h,
        "gap": gap if secondary_lines else 0,
        "y_start": y_start,
        "block_area": block_w * block_h,
    }


def _truncate_title(title: str, max_words: int = 6) -> str:
    words = title.upper().split()
    if len(words) <= max_words:
        return title.upper()
    truncated = words[: max_words - 1] + [f"{words[max_words - 1]}…"]
    return " ".join(truncated)


def _apply_frame_processing(frame_bytes: bytes) -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageEnhance, ImageFilter

    nparr = None
    try:
        import cv2  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        nparr_arr = np.frombuffer(frame_bytes, np.uint8)
        nparr = cv2.imdecode(nparr_arr, cv2.IMREAD_COLOR)
    except ImportError:
        nparr = None

    if nparr is None:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Sharpness(img).enhance(1.8)
        logger.info("Segmentação indisponível — processamento sem separação de fundo")
        return img

    try:
        import cv2  # type: ignore[import]
        import mediapipe as mp  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        rgb = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
            result = seg.process(rgb)

        mask = result.segmentation_mask
        mask_3ch = np.stack([mask] * 3, axis=-1)

        fg_pil = Image.fromarray(rgb)
        bg_pil = fg_pil.copy().filter(ImageFilter.GaussianBlur(radius=16))

        bg_arr = np.array(bg_pil, dtype=np.float32)
        fg_arr = np.array(fg_pil, dtype=np.float32)
        composite = (fg_arr * mask_3ch + bg_arr * (1.0 - mask_3ch)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(composite)

        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Sharpness(img).enhance(1.6)
        logger.info("Segmentação de fundo aplicada para thumbnail %s", id(img))
        return img
    except Exception:
        logger.info("Segmentação indisponível — processamento sem separação de fundo")
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Sharpness(img).enhance(1.8)
        return img


def _resize_to_youtube_format(image_path: Path) -> None:
    from PIL import Image

    with Image.open(image_path) as img:
        resized = img.resize((_THUMBNAIL_W, _THUMBNAIL_H), Image.LANCZOS)
        resized.save(image_path, format="PNG", optimize=True)
    logger.info("Thumbnail redimensionada para %dx%d: %s", _THUMBNAIL_W, _THUMBNAIL_H, image_path)


def _resize_frame_bytes(frame_bytes: bytes, max_width: int, max_height: int) -> bytes:
    return _resize_image_bytes(frame_bytes, max_size=(max_width, max_height))


def _resize_image_bytes(
    image_bytes: bytes,
    max_size: tuple[int, int] | None = None,
    target_size: tuple[int, int] | None = None,
) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        converted = img.convert("RGB")
        if target_size is not None:
            converted = converted.resize(target_size, Image.LANCZOS)
        elif max_size is not None:
            converted.thumbnail(max_size, Image.LANCZOS)

        buf = io.BytesIO()
        converted.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _load_image_from_bytes(image_bytes: bytes) -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _render_local_thumbnail_bytes(frame_bytes: bytes) -> bytes:
    processed = _apply_frame_processing(frame_bytes)
    buf = io.BytesIO()
    processed.resize((_THUMBNAIL_W, _THUMBNAIL_H)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Resposta sem JSON válido")
        return json.loads(text[start : end + 1])


def _anthropic_response_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _client_with_timeout(client: Any, timeout: float) -> Any:
    if hasattr(client, "with_options"):
        return client.with_options(timeout=timeout)
    return client
