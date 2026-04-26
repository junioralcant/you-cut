import logging
from pathlib import Path

import httpx
import openai

from youcut.models import ClipRecord, ViralClip

logger = logging.getLogger(__name__)

_DALLE_MODEL = "dall-e-3"
_DALLE_SIZE = "1792x1024"
_DALLE_QUALITY = "standard"


def generate_thumbnail(
    clip: ViralClip,
    face_context: str,
    output_dir: Path,
    clip_index: int,
    api_key: str,
) -> Path:
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    output_path = thumbnails_dir / f"clip_{clip_index:02d}.png"
    return _generate_and_save(clip, face_context, output_path, api_key)


def regenerate_thumbnail(clip: ViralClip, clip_record: ClipRecord, api_key: str) -> Path:
    if clip_record.thumbnail_path is not None:
        output_path = clip_record.thumbnail_path
    else:
        output_path = clip_record.clip_path.parent / "thumbnails" / f"{clip_record.clip_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _generate_and_save(clip, "", output_path, api_key)


def _build_prompt(clip: ViralClip, face_context: str) -> str:
    prompt = (
        f"YouTube thumbnail for a video clip. Theme: {clip.thumbnail_idea}. "
        "Style: eye-catching, high contrast, professional YouTube thumbnail. "
        "16:9 landscape format, photorealistic."
    )
    if face_context:
        prompt += f" Include visual representation of the participants: {face_context}."
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
