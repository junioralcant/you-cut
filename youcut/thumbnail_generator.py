import json
import logging
import subprocess
import tempfile
from pathlib import Path

from youcut.models import ClipRecord, ThumbnailFrameResult, ViralClip

logger = logging.getLogger(__name__)

_THUMBNAIL_W = 1280
_THUMBNAIL_H = 720
_ASSETS_DIR = Path(__file__).parent / "assets"
_FONT_PATH = _ASSETS_DIR / "Roboto-Regular.ttf"
_TEXT_FONT_SIZE = 72
_MAX_LINES = 2


def generate_thumbnail(
    clip: ViralClip,
    output_dir: Path,
    clip_index: int,
    clip_path: Path | None = None,
) -> Path:
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    output_path = thumbnails_dir / f"clip_{clip_index:02d}.png"

    source = clip_path if clip_path and clip_path.exists() else None
    if source is None:
        raise ValueError(f"clip_path não fornecido ou não existe: {clip_path}")

    frame_bytes = _select_best_face_frame(source)
    processed = _apply_frame_processing(frame_bytes)
    composed = _compose_text_overlay(processed, clip.thumbnail_text)
    composed.save(output_path, format="PNG", optimize=True)
    _resize_to_youtube_format(output_path)

    result = ThumbnailFrameResult(
        frame_timestamp=0.0,
        frame_score=0.0,
        segmentation_applied=False,
        output_path=output_path,
    )
    logger.info(
        "Thumbnail gerada com frame real: path=%s timestamp=%.2f score=%.3f",
        output_path,
        result.frame_timestamp,
        result.frame_score,
    )
    logger.info("ThumbnailFrameResult: %s", json.dumps({
        "frame_timestamp": result.frame_timestamp,
        "frame_score": result.frame_score,
        "segmentation_applied": result.segmentation_applied,
        "output_path": str(result.output_path),
    }))
    return output_path


def regenerate_thumbnail(
    clip: ViralClip,
    clip_record: ClipRecord,
) -> Path:
    if clip_record.thumbnail_path is not None:
        output_path = clip_record.thumbnail_path
    else:
        output_path = clip_record.clip_path.parent / "thumbnails" / f"{clip_record.clip_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_bytes = _select_best_face_frame(clip_record.clip_path)
    processed = _apply_frame_processing(frame_bytes)
    composed = _compose_text_overlay(processed, clip.thumbnail_text)
    composed.save(output_path, format="PNG", optimize=True)
    _resize_to_youtube_format(output_path)

    result = ThumbnailFrameResult(
        frame_timestamp=0.0,
        frame_score=0.0,
        segmentation_applied=False,
        output_path=output_path,
    )
    logger.info(
        "Thumbnail regenerada com frame real: path=%s timestamp=%.2f score=%.3f",
        output_path,
        result.frame_timestamp,
        result.frame_score,
    )
    return output_path


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
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(clip_path)],
        capture_output=True,
        timeout=10,
        check=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            duration = float(stream.get("duration", 0))
            if duration > 0:
                return duration
    raise RuntimeError(f"Não foi possível determinar a duração do vídeo: {clip_path}")


def _extract_frame_at(clip_path: Path, timestamp: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(clip_path),
                "-vframes", "1",
                "-q:v", "2",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=15,
            check=True,
        )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def _score_frame(frame_bytes: bytes, detector) -> float:
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


def _select_best_face_frame(
    clip_path: Path,
    n_samples: int = 10,
    min_confidence: float = 0.5,
) -> bytes:
    try:
        import mediapipe as mp  # type: ignore[import]
    except ImportError:
        logger.warning("MediaPipe não disponível — usando fallback para _extract_frame()")
        return _extract_frame(clip_path)

    try:
        duration = _get_video_duration(clip_path)
    except Exception as exc:
        logger.warning("Falha ao obter duração do vídeo: %s — usando fallback", exc)
        return _extract_frame(clip_path)

    start = duration * 0.05
    end = duration * 0.95
    if end <= start:
        start = 0.0
        end = duration

    step = (end - start) / max(n_samples - 1, 1)
    timestamps = [start + i * step for i in range(n_samples)]

    best_bytes: bytes | None = None
    best_score = -1.0

    detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)
    try:
        for ts in timestamps:
            try:
                frame_bytes = _extract_frame_at(clip_path, ts)
            except Exception:
                continue
            score = _score_frame(frame_bytes, detector)
            if score > best_score:
                best_score = score
                best_bytes = frame_bytes
    finally:
        detector.close()

    if best_bytes is None or best_score <= 0:
        logger.warning(
            "Nenhum rosto detectado em %d frames amostrados — usando fallback para _extract_frame()",
            n_samples,
        )
        return _extract_frame(clip_path)

    logger.info("Melhor frame selecionado com score %.3f de %d amostras", best_score, n_samples)
    return best_bytes


_BOLD_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Black.ttf")


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
    fill: tuple,
    stroke_width: int,
    stroke_fill: tuple = (0, 0, 0, 255),
) -> int:
    """Draws centered text with stroke. Returns the rendered height."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_w - text_w) // 2

    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill)
    return text_h


def _split_title_hierarchy(title: str) -> tuple[list[str], str]:
    """Split title into secondary lines + hero (last impactful word/phrase)."""
    import textwrap
    words = title.upper().split()
    if len(words) <= 2:
        return [], " ".join(words)

    # last 1 word is the hero; everything else wraps into secondary lines
    hero = words[-1]
    secondary_words = words[:-1]
    secondary_lines = textwrap.wrap(" ".join(secondary_words), width=16)
    return secondary_lines, hero


def _compose_text_overlay(image: "Image.Image", title: str) -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageDraw

    img = image.copy().convert("RGBA")
    w, h = img.size

    if not title or not title.strip():
        return img.convert("RGB")

    # --- dark vignette over whole image ---
    vignette = Image.new("RGBA", (w, h), (0, 0, 0, 145))
    img = Image.alpha_composite(img, vignette)
    draw = ImageDraw.Draw(img)

    secondary_lines, hero = _split_title_hierarchy(title)

    # --- size tuning ---
    hero_size = int(h * 0.28)       # ~200px on 720h
    secondary_size = int(h * 0.10)  # ~72px on 720h

    hero_font = _load_font(hero_size)
    sec_font = _load_font(secondary_size)

    # shrink hero font until it fits 88% of width
    max_w = int(w * 0.88)
    while hero_size > 60:
        hero_font = _load_font(hero_size)
        bbox = draw.textbbox((0, 0), hero, font=hero_font)
        if bbox[2] - bbox[0] <= max_w:
            break
        hero_size -= 6

    # shrink secondary until all lines fit 80% of width
    max_sec_w = int(w * 0.80)
    while secondary_size > 30:
        sec_font = _load_font(secondary_size)
        if all(draw.textbbox((0, 0), l, font=sec_font)[2] <= max_sec_w for l in secondary_lines):
            break
        secondary_size -= 4

    hero_bbox = draw.textbbox((0, 0), hero, font=hero_font)
    hero_h = hero_bbox[3] - hero_bbox[1]

    sec_line_h = (draw.textbbox((0, 0), "A", font=sec_font)[3] + int(secondary_size * 0.25)) if secondary_lines else 0
    sec_total_h = sec_line_h * len(secondary_lines)

    gap = int(h * 0.018)
    block_h = sec_total_h + gap + hero_h
    y_start = (h - block_h) // 2

    hero_stroke = max(6, hero_size // 12)
    sec_stroke = max(3, secondary_size // 14)

    # draw secondary lines
    for i, line in enumerate(secondary_lines):
        _draw_text_centered(
            draw, line, sec_font,
            y=y_start + i * sec_line_h,
            canvas_w=w,
            fill=(255, 255, 255, 255),
            stroke_width=sec_stroke,
        )

    # draw hero word
    hero_y = y_start + sec_total_h + gap
    _draw_text_centered(
        draw, hero, hero_font,
        y=hero_y,
        canvas_w=w,
        fill=(255, 215, 0, 255),
        stroke_width=hero_stroke,
    )

    return img.convert("RGB")


def _apply_frame_processing(frame_bytes: bytes) -> "Image.Image":  # type: ignore[name-defined]
    import io

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
        import mediapipe as mp  # type: ignore[import]
        import numpy as np  # type: ignore[import]
        import cv2  # type: ignore[import]

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
