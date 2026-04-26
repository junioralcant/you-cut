import base64
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
_DALLE_SIZE = "1792x1024"
_DALLE_QUALITY = "standard"
_VISION_MODEL = "gpt-4o"


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


def _describe_frame_characters(clip_path: Path, clip: ViralClip, api_key: str) -> str:
    """Usa GPT-4o Vision para descrever as pessoas visíveis no frame do clipe."""
    frame_bytes = _extract_frame(clip_path)
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
        f"Thumbnail para YouTube em estilo chamativo e profissional. "
        f"Tema: {clip.thumbnail_idea}. "
        "Estilo: alto contraste, cores vibrantes, thumbnail profissional de YouTube, realista. "
        "Formato paisagem 16:9."
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
    logger.info("Thumbnail saved: %s", output_path)
    return output_path


def _download_image(url: str, dest: Path) -> None:
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    dest.write_bytes(response.content)
