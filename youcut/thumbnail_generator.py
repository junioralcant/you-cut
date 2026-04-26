import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import openai

from youcut.models import ClipRecord, ViralClip

logger = logging.getLogger(__name__)

_DALLE_MODEL = "dall-e-3"
_DALLE_SIZE = "1792x1024"  # closest DALL-E 3 size to 16:9; resized to YouTube spec after download
_DALLE_QUALITY = "standard"
_VISION_MODEL = "gpt-4o"
_THUMBNAIL_W = 1280
_THUMBNAIL_H = 720


def generate_thumbnail(
    clip: ViralClip,
    face_context: str,
    output_dir: Path,
    clip_index: int,
    api_key: str,
    clip_path: Optional[Path] = None,
) -> Path:
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    output_path = thumbnails_dir / f"clip_{clip_index:02d}.png"

    resolved_face_context = face_context
    if clip_path and clip_path.exists():
        try:
            resolved_face_context = _describe_frame_characters(clip_path, clip, api_key)
        except Exception as e:
            logger.warning("Falha ao descrever personagens do frame: %s", e)

    return _generate_and_save(clip, resolved_face_context, output_path, api_key)


def regenerate_thumbnail(clip: ViralClip, clip_record: ClipRecord, api_key: str) -> Path:
    if clip_record.thumbnail_path is not None:
        output_path = clip_record.thumbnail_path
    else:
        output_path = clip_record.clip_path.parent / "thumbnails" / f"{clip_record.clip_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_context = ""
    if clip_record.clip_path.exists():
        try:
            face_context = _describe_frame_characters(clip_record.clip_path, clip, api_key)
        except Exception as e:
            logger.warning("Falha ao descrever personagens do frame: %s", e)

    return _generate_and_save(clip, face_context, output_path, api_key)


def _extract_frame(clip_path: Path) -> bytes:
    """Extrai um frame do meio do clipe usando ffmpeg e retorna os bytes PNG."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-sseof", "-1",  # 1 segundo antes do fim como fallback
                "-i", str(clip_path),
                "-ss", "00:00:01",  # 1 segundo do início
                "-vframes", "1",
                "-q:v", "2",
                str(tmp_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            # fallback: pega o primeiro frame
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


def _describe_frame_characters(clip_path: Path, clip: ViralClip, api_key: str) -> str:
    """Usa GPT-4o Vision para descrever as pessoas visíveis no frame do clipe."""
    frame_bytes = _select_best_face_frame(clip_path)
    b64 = base64.b64encode(frame_bytes).decode("utf-8")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analise esta imagem e descreva de forma objetiva as pessoas visíveis: "
                            "aparência física (cor de pele, cabelo, roupas, expressão facial, idade aproximada). "
                            "Seja conciso e específico. Se não houver pessoas, descreva os elementos visuais principais. "
                            "Responda em português do Brasil."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    description = response.choices[0].message.content or ""
    logger.info("Descrição de personagens extraída do frame: %s", description[:100])
    return description


def _build_prompt(clip: ViralClip, face_context: str) -> str:
    prompt = (
        f"Thumbnail para YouTube em estilo chamativo e profissional, "
        f"inspirado em canais de podcast de alto desempenho como Flow Podcast. "
        f"Tema: {clip.thumbnail_idea}. "
        "Estilo: rosto do personagem principal em close-up ocupando pelo menos 50% da imagem, "
        "texto chamativo em tipografia bold e de alta legibilidade sobreposto à imagem, "
        "paleta de cores vibrantes e de alto contraste (vermelho, azul, amarelo ou laranja intensos), "
        "fundo simplificado com cor sólida ou gradiente para eliminar ruído visual, "
        "iluminação dramática no rosto, expressão intensa e engajadora. "
        "Formato paisagem 16:9, realista."
    )
    if face_context:
        prompt += (
            f" Inclua representação visual fiel dos participantes com as seguintes características: {face_context}. "
            "Mantenha a aparência dos personagens consistente com a descrição."
        )
    return prompt


def _generate_and_save(
    clip: ViralClip,
    face_context: str,
    output_path: Path,
    api_key: str,
) -> Path:
    prompt = _build_prompt(clip, face_context)
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.images.generate(
            model=_DALLE_MODEL,
            prompt=prompt,
            size=_DALLE_SIZE,
            quality=_DALLE_QUALITY,
            n=1,
        )
    except openai.RateLimitError as e:
        raise RuntimeError("OpenAI rate limit reached. Try again later.") from e
    except openai.AuthenticationError as e:
        raise RuntimeError("Invalid OpenAI API key. Check your openai_api_key configuration.") from e
    except openai.OpenAIError as e:
        raise RuntimeError(f"OpenAI API error generating thumbnail: {e}") from e

    image_url = response.data[0].url
    if not image_url:
        raise RuntimeError("DALL-E 3 returned no image URL. Check response_format or API status.")
    _download_image(image_url, output_path)
    _resize_to_youtube_format(output_path)
    logger.info("Thumbnail saved: %s", output_path)
    return output_path


def _download_image(url: str, dest: Path) -> None:
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    dest.write_bytes(response.content)


def _resize_to_youtube_format(image_path: Path) -> None:
    from PIL import Image  # type: ignore[import]

    with Image.open(image_path) as img:
        resized = img.resize((_THUMBNAIL_W, _THUMBNAIL_H), Image.LANCZOS)
        resized.save(image_path, format="PNG", optimize=True)
    logger.info("Thumbnail redimensionada para %dx%d: %s", _THUMBNAIL_W, _THUMBNAIL_H, image_path)
