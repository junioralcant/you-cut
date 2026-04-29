import logging
import json
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont

from youcut.config import PipelineConfig
from youcut.models import ViralClip
from youcut.thumbnail_generator import (
    _build_ai_clients,
    _resolve_openai_api_key,
    _run_thumbnail_skill_script,
    generate_social_top_image,
)

logger = logging.getLogger(__name__)

_CANVAS_W = 1080
_CANVAS_H = 1920
_ASSETS_DIR = Path(__file__).parent / "assets"
_FONT_PATH = _ASSETS_DIR / "Roboto-Regular.ttf"
_DEFAULT_BG = "#F4C400"
_DEFAULT_TEXT = "#111111"
_ORANGE_BG = "#FF8A00"
_MAX_LABEL_LINES = 2

# Where the speaker's face center should land vertically inside the bottom panel,
# expressed as a fraction of the panel height. The detector reports the face
# bounding box (chin → forehead) without hair, so we aim slightly below the
# panel midline to keep hair / top of the head inside the frame.
_BOTTOM_PANEL_FACE_TARGET = 0.55


def generate_social_label(
    clip: ViralClip,
    config: PipelineConfig,
) -> tuple[str, str | None]:
    fallback_text = clip.social_hook_title or clip.title or "MOMENTO EM DESTAQUE"
    fallback_color_mode = None

    if not config.anthropic_api_key:
        return fallback_text, fallback_color_mode

    prompt = (
        "Você cria labels editoriais para vídeos curtos de política e atualidades. "
        "Responda somente JSON válido no formato "
        '{"label_text":"...","color_mode":"yellow|orange"}.\n'
        "Regras:\n"
        "- label_text com 2 a 5 palavras\n"
        "- usar caixa alta\n"
        "- sem pontuação final\n"
        "- sem quebrar palavras ou sílabas\n"
        "- direto, agressivo, editorial e claro\n"
        "- evitar nomes próprios longos quando uma formulação mais forte funcionar melhor\n"
        "- a paleta editorial permitida e idealizada no PRD e a seguinte: amarelo vivo como default e laranja vivo como alternativa\n"
        "- preferir yellow como default; usar orange so quando combinar melhor com urgencia, choque ou alerta\n"
        "- pensar no texto como label fixa de tarja central, nao como manchete longa\n"
        f"Título do corte: {clip.title}\n"
        f"Motivo: {clip.reason}\n"
        f"Descrição: {clip.description}\n"
        f"Prompt visual: {clip.social_image_prompt}\n"
    )

    try:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        response = client.with_options(timeout=30.0).messages.create(
            model=config.claude_model,
            max_tokens=120,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ).strip()
        payload = json.loads(raw_text[raw_text.find("{"): raw_text.rfind("}") + 1])
        label_text = str(payload.get("label_text", fallback_text)).strip().upper()
        color_mode = str(payload.get("color_mode", "")).strip().lower() or fallback_color_mode
        return label_text or fallback_text, color_mode
    except Exception as exc:
        logger.warning("Social composer: falha ao gerar label via IA; usando fallback: %s", exc)
        return fallback_text, fallback_color_mode


def resolve_title_band_colors(config: PipelineConfig) -> tuple[str, str]:
    mode = config.social_layout_title_color_mode
    if mode in {"engagement_default", "yellow"}:
        return _DEFAULT_BG, _DEFAULT_TEXT
    if mode == "orange":
        return _ORANGE_BG, _DEFAULT_TEXT
    return config.social_layout_title_bg_color, config.social_layout_title_text_color


def compose_social_clip(
    clip_path: Path,
    clip: ViralClip,
    config: PipelineConfig,
) -> Path:
    output_dir = clip_path.parent
    label_text, suggested_color_mode = generate_social_label(clip, config)
    top_h = config.social_layout_top_image_height
    band_h = config.social_layout_title_band_height if config.social_layout_title_enabled else 0
    header_h = top_h + band_h
    header_image_path = _render_social_header_image(
        clip=clip,
        clip_path=clip_path,
        output_dir=output_dir,
        title=label_text,
        config=config,
        width=_CANVAS_W,
        height=header_h,
        top_height=top_h,
        band_height=band_h,
        suggested_color_mode=suggested_color_mode,
    )
    output_path = clip_path.with_stem(clip_path.stem + "_social")

    bottom_h = _CANVAS_H - top_h - band_h
    if bottom_h <= 0:
        raise ValueError("Alturas do layout social inválidas; bottom panel ficou sem espaço")

    src_w, src_h = _probe_video_dimensions(clip_path)
    face_y_norm = _detect_face_y_norm(clip_path)
    bottom_crop = _build_bottom_crop_filter(
        src_w=src_w,
        src_h=src_h,
        target_w=_CANVAS_W,
        target_h=bottom_h,
        face_y_norm=face_y_norm,
    )

    overlay_steps = ["[base][header]overlay=0:0[tmp1]", f"[tmp1][bottom]overlay=0:{header_h}[v]"]

    filter_parts = [
        f"[0:v]scale={_CANVAS_W}:{header_h}:force_original_aspect_ratio=increase,crop={_CANVAS_W}:{header_h}[header]",
        f"[1:v]{bottom_crop}[bottom]",
        f"color=c=black:size={_CANVAS_W}x{_CANVAS_H}[base]",
    ]
    filter_parts.extend(overlay_steps)
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-i", str(header_image_path),
        "-i", str(clip_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-y",
        str(output_path),
    ]

    logger.info("Social composer: composing layout mode=%s", config.social_layout_mode)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        logger.error("Social composer falhou (código %d): %s", exc.returncode, stderr)
        raise
    finally:
        header_image_path.unlink(missing_ok=True)

    logger.info("Social composer: final clip generated at %s", output_path)
    return output_path


def _render_social_header_image(
    *,
    clip: ViralClip,
    clip_path: Path,
    output_dir: Path,
    title: str,
    config: PipelineConfig,
    width: int,
    height: int,
    top_height: int,
    band_height: int,
    suggested_color_mode: str | None,
) -> Path:
    top_image_path = generate_social_top_image(clip, output_dir, clip_path, config)
    fallback_path = _render_social_header_image_local(
        top_image_path=top_image_path,
        title=title,
        config=config,
        width=width,
        height=height,
        top_height=top_height,
        band_height=band_height,
        suggested_color_mode=suggested_color_mode,
    )
    ai_path = _render_social_header_image_via_ai(
        top_image_path=top_image_path,
        title=title,
        config=config,
        width=width,
        height=height,
        top_height=top_height,
        band_height=band_height,
        suggested_color_mode=suggested_color_mode,
        fallback_path=fallback_path,
    )
    top_image_path.unlink(missing_ok=True)
    if ai_path is not None:
        fallback_path.unlink(missing_ok=True)
        return ai_path
    return fallback_path


def _render_social_header_image_via_ai(
    *,
    top_image_path: Path,
    title: str,
    config: PipelineConfig,
    width: int,
    height: int,
    top_height: int,
    band_height: int,
    suggested_color_mode: str | None,
    fallback_path: Path,
) -> Path | None:
    _, openai_client = _build_ai_clients(config)
    openai_api_key = _resolve_openai_api_key(config, openai_client)
    if openai_client is None or not openai_api_key:
        return None

    effective_config = config.model_copy(
        update={
            "social_layout_title_color_mode": suggested_color_mode
            if suggested_color_mode in {"yellow", "orange"}
            else config.social_layout_title_color_mode
        }
    )
    bg_hex, text_hex = resolve_title_band_colors(effective_config)
    prompt = _build_social_header_generation_prompt(
        title=title,
        bg_hex=bg_hex,
        text_hex=text_hex,
        width=width,
        height=height,
        top_height=top_height,
        band_height=band_height,
    )

    try:
        image_bytes = _run_thumbnail_skill_script(
            prompt=prompt,
            reference_frames=[top_image_path.read_bytes(), fallback_path.read_bytes()],
            openai_api_key=openai_api_key,
            timeout=60.0,
        )
        return _write_resized_png(image_bytes, width=width, height=height)
    except Exception as exc:
        logger.warning("Social composer: falha ao gerar header via IA; usando fallback local: %s", exc)
        return None


def _render_social_header_image_local(
    *,
    top_image_path: Path,
    title: str,
    config: PipelineConfig,
    width: int,
    height: int,
    top_height: int,
    band_height: int,
    suggested_color_mode: str | None,
) -> Path:
    with Image.open(top_image_path) as top_image:
        top_panel = top_image.convert("RGB").resize((width, top_height), Image.Resampling.LANCZOS)

    header_image = Image.new("RGB", (width, height), color=(0, 0, 0))
    header_image.paste(top_panel, (0, 0))
    if band_height > 0:
        band_path = _render_title_band_image_local(
            title,
            config.model_copy(
                update={
                    "social_layout_title_color_mode": suggested_color_mode
                    if suggested_color_mode in {"yellow", "orange"}
                    else config.social_layout_title_color_mode
                }
            ),
            width=width,
            height=band_height,
        )
        with Image.open(band_path) as band_image:
            header_image.paste(band_image.convert("RGB"), (0, top_height))
        band_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        output_path = Path(tmp.name)
    header_image.save(output_path, format="PNG", optimize=True)
    return output_path


def _render_title_band_image(
    title: str,
    config: PipelineConfig,
    *,
    width: int,
    height: int,
    suggested_color_mode: str | None = None,
) -> Path:
    effective_config = config.model_copy(
        update={
            "social_layout_title_color_mode": suggested_color_mode
            if suggested_color_mode in {"yellow", "orange"}
            else config.social_layout_title_color_mode
        }
    )
    fallback_path = _render_title_band_image_local(
        title,
        effective_config,
        width=width,
        height=height,
    )
    ai_path = _render_title_band_image_via_ai(
        title,
        effective_config,
        fallback_path,
        width=width,
        height=height,
    )
    if ai_path is not None:
        fallback_path.unlink(missing_ok=True)
        return ai_path
    return fallback_path


def _render_title_band_image_via_ai(
    title: str,
    config: PipelineConfig,
    reference_path: Path,
    *,
    width: int,
    height: int,
) -> Path | None:
    _, openai_client = _build_ai_clients(config)
    openai_api_key = _resolve_openai_api_key(config, openai_client)
    if openai_client is None or not openai_api_key:
        return None

    bg_hex, text_hex = resolve_title_band_colors(config)
    prompt = _build_title_band_generation_prompt(
        title=title,
        bg_hex=bg_hex,
        text_hex=text_hex,
        width=width,
        height=height,
    )

    try:
        image_bytes = _run_thumbnail_skill_script(
            prompt=prompt,
            reference_frames=[reference_path.read_bytes()],
            openai_api_key=openai_api_key,
            timeout=60.0,
        )
        return _write_resized_png(image_bytes, width=width, height=height)
    except Exception as exc:
        logger.warning("Social composer: falha ao gerar tarja via IA; usando fallback local: %s", exc)
        return None


def _render_title_band_image_local(
    title: str,
    config: PipelineConfig,
    *,
    width: int,
    height: int,
) -> Path:
    effective_config = config.model_copy(
        update={"social_layout_title_color_mode": config.social_layout_title_color_mode}
    )
    bg_hex, text_hex = resolve_title_band_colors(effective_config)
    bg_rgb = _hex_to_rgb(bg_hex, fallback=_DEFAULT_BG)
    text_rgb = _hex_to_rgb(text_hex, fallback=_DEFAULT_TEXT)
    image = _create_band_background(width, height, bg_rgb)
    draw = ImageDraw.Draw(image)

    font_size = 84
    line_gap = 6
    lines: list[str] = []
    font = _load_font(font_size)
    while font_size >= 42:
        font = _load_font(font_size)
        lines = _wrap_title(draw, title, font, width - 120, max_lines=_MAX_LABEL_LINES)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
        block_height = sum(line_heights) + (line_gap * max(0, len(lines) - 1))
        max_line_width = max(
            (draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]) for line in lines
        )
        if block_height <= height - 34 and max_line_width <= width - 120:
            break
        font_size -= 6

    stripe_color = _darken_rgb(bg_rgb, 0.24)
    inner_border = _lighten_rgb(bg_rgb, 0.12)
    draw.rectangle([(0, 0), (width, 10)], fill=stripe_color)
    draw.rectangle([(0, height - 10), (width, height)], fill=stripe_color)
    draw.rounded_rectangle(
        [(22, 18), (width - 22, height - 18)],
        radius=26,
        outline=inner_border,
        width=4,
    )
    _draw_band_accents(draw, width, height, text_rgb, bg_rgb)
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    block_height = sum(line_heights) + (line_gap * max(0, len(lines) - 1))
    current_y = max(14, (height - block_height) // 2)
    for line, bbox, line_height in zip(lines, line_boxes, line_heights):
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        _draw_bold_text(draw, (x, current_y - bbox[1]), line, font, fill=text_rgb)
        current_y += line_height + line_gap

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        output_path = Path(tmp.name)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def _build_title_band_generation_prompt(
    *,
    title: str,
    bg_hex: str,
    text_hex: str,
    width: int,
    height: int,
) -> str:
    return (
        "Create a premium editorial title band for a short-form political social video. "
        f"The final image must be exactly {width}x{height}. "
        "Use the provided reference image only as a layout and proportion guide, then improve the design quality substantially. "
        "Keep the composition clean, expensive, legitimate, and highly clickable, with a polished Brazilian editorial social-video look. "
        "The band must feel bold and beautiful, not generic or childish. "
        f"Embed exactly this text and no other text: \"{title}\". "
        "Do not paraphrase, translate, add punctuation, add logos, or invent extra words. "
        "The title must remain fully readable on small mobile screens. "
        "Respect a centered composition with strong hierarchy and generous safe margins. "
        "You may use tasteful graphic accents, framing devices, depth, lighting, texture, or layered shapes if they improve the design. "
        "Do not create a fake screenshot, newspaper, or background scene. This must remain a title band asset. "
        "Avoid red as the dominant color. "
        f"Base background color direction: {bg_hex}. "
        f"Base text color direction: {text_hex}. "
        "Prefer bright yellow or vivid orange editorial energy, with dark high-contrast text. "
        "Return only the designed band image."
    )


def _build_social_header_generation_prompt(
    *,
    title: str,
    bg_hex: str,
    text_hex: str,
    width: int,
    height: int,
    top_height: int,
    band_height: int,
) -> str:
    min_title_zone = max(140, int(height * 0.18))
    max_title_zone = max(min_title_zone + 20, int(height * 0.35))
    return (
        "Create a premium editorial header for a short-form political social video. "
        f"The final image must be exactly {width}x{height}. "
        "Use the first reference image as the thematic image base and keep its overall subject, mood, and composition believable. "
        "Use the second reference only as a fallback layout guide, not as a style limit. "
        f"Embed exactly this title and no other text: \"{title}\". "
        "Do not paraphrase, translate, add punctuation, add logos, or invent extra words. "
        "The title treatment and its background must feel harmonious with the rest of the image, not like a separate flat strip pasted on top. "
        "You are free to choose the internal title container size and proportion based on what looks best for the composition. "
        f"The title zone should visually occupy something between about {min_title_zone} and {max_title_zone} pixels of the header height, "
        "typically near the lower portion of the header, but it does not need to be a rigid rectangle. "
        f"The upper visual region should remain dominant and should roughly preserve the intended editorial image area near {top_height} pixels. "
        f"The previous workflow used a title band around {band_height} pixels, but you may reinterpret that proportion more elegantly. "
        "The final result must look polished, legitimate, expensive, and native to Brazilian social political editorial content. "
        "Avoid a squashed, flattened, or cheap banner look. "
        f"Color direction should stay within this palette: background energy around {bg_hex}, text direction around {text_hex}. "
        "Prefer warm yellow/orange editorial energy with dark, high-contrast readable text. "
        "Return only the completed header image."
    )


def _write_resized_png(image_bytes: bytes, *, width: int, height: int) -> Path:
    with Image.open(BytesIO(image_bytes)) as image:
        resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            output_path = Path(tmp.name)
        resized.save(output_path, format="PNG", optimize=True)
    return output_path


def _wrap_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    *,
    max_lines: int,
) -> list[str]:
    normalized = " ".join(title.upper().split())
    if not normalized:
        return ["MOMENTO EM DESTAQUE"]

    words = normalized.split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word]).strip()
        width = draw.textbbox((0, 0), candidate, font=font)[2] - draw.textbbox((0, 0), candidate, font=font)[0]
        if current and width > max_width:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines - 1:
                break
        else:
            current.append(word)

    remaining_words = words[len(" ".join(lines + [" ".join(current)]).split()):]
    tail = " ".join(current + remaining_words).strip()
    if tail:
        while tail:
            tail_width = draw.textbbox((0, 0), tail, font=font)[2] - draw.textbbox((0, 0), tail, font=font)[0]
            if tail_width <= max_width:
                break
            tail = " ".join(tail.split()[:-1]).strip()
        if tail and tail != " ".join(current + remaining_words).strip():
            tail = f"{tail}…"
        if tail:
            lines.append(tail)

    return lines[:max_lines] or [normalized]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _draw_bold_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int],
) -> None:
    x, y = position
    shadow = _lighten_rgb(fill, 0.72)
    for dx, dy in ((3, 3), (2, 2), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=fill)


def _darken_rgb(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, 1.0 - amount))
    return tuple(max(0, min(255, int(channel * factor))) for channel in rgb)


def _lighten_rgb(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, amount))
    return tuple(max(0, min(255, int(channel + ((255 - channel) * factor)))) for channel in rgb)


def _create_band_background(width: int, height: int, bg_rgb: tuple[int, int, int]) -> Image.Image:
    top_rgb = _lighten_rgb(bg_rgb, 0.18)
    mid_rgb = _lighten_rgb(bg_rgb, 0.08)
    bottom_rgb = _darken_rgb(bg_rgb, 0.08)
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        progress = y / max(1, height - 1)
        if progress < 0.45:
            local = progress / 0.45
            color = _mix_rgb(top_rgb, mid_rgb, local)
        else:
            local = (progress - 0.45) / 0.55
            color = _mix_rgb(mid_rgb, bottom_rgb, local)
        for x in range(width):
            pixels[x, y] = color

    return image


def _draw_band_accents(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    text_rgb: tuple[int, int, int],
    bg_rgb: tuple[int, int, int],
) -> None:
    accent_dark = _darken_rgb(text_rgb, 0.08)
    accent_light = _lighten_rgb(bg_rgb, 0.42)
    left = 72
    right = width - 72
    center_y = height // 2

    draw.polygon(
        [(left, center_y), (left + 28, center_y - 24), (left + 28, center_y + 24)],
        fill=accent_dark,
    )
    draw.polygon(
        [(left + 40, center_y), (left + 68, center_y - 24), (left + 68, center_y + 24)],
        fill=accent_light,
    )
    draw.polygon(
        [(right, center_y), (right - 28, center_y - 24), (right - 28, center_y + 24)],
        fill=accent_dark,
    )
    draw.polygon(
        [(right - 40, center_y), (right - 68, center_y - 24), (right - 68, center_y + 24)],
        fill=accent_light,
    )


def _mix_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, amount))
    return tuple(
        max(0, min(255, int(round(start[idx] + ((end[idx] - start[idx]) * factor)))))
        for idx in range(3)
    )


def _hex_to_rgb(value: str, *, fallback: str) -> tuple[int, int, int]:
    normalized = value.strip().lstrip("#")
    if len(normalized) != 6:
        normalized = fallback.lstrip("#")
    return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))


def _probe_video_dimensions(clip_path: Path) -> tuple[int, int]:
    """Return (width, height) of *clip_path*. Falls back to canvas size on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        w_str, h_str = result.stdout.strip().split("x")
        return int(w_str), int(h_str)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as exc:
        logger.warning("Social composer: falha ao probar dimensões de %s (%s); assumindo %dx%d",
                       clip_path.name, exc, _CANVAS_W, _CANVAS_H)
        return _CANVAS_W, _CANVAS_H


def _detect_face_y_norm(clip_path: Path) -> float | None:
    """Wrap face_tracker.detect_dominant_face_y_norm so a missing/raising
    implementation never blocks the social composer."""
    try:
        from youcut.face_tracker import detect_dominant_face_y_norm
        return detect_dominant_face_y_norm(clip_path)
    except Exception as exc:
        logger.warning("Social composer: face anchor falhou (%s); usando enquadramento central", exc)
        return None


def _build_bottom_crop_filter(
    *,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    face_y_norm: float | None,
) -> str:
    """Build the ``scale,crop`` filter chain that places the bottom-panel slice
    around the speaker's face when *face_y_norm* is known.

    Mirrors what ``force_original_aspect_ratio=increase`` would do (scale-to-cover),
    then offsets the crop so the face ends up at ``_BOTTOM_PANEL_FACE_TARGET`` of
    the panel height — clamped so the crop never leaves the scaled frame.
    """
    if src_w <= 0 or src_h <= 0:
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"

    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio >= target_ratio:
        scaled_h = target_h
        scaled_w = int(round(target_h * src_ratio))
        if face_y_norm is None:
            x_offset = max(0, (scaled_w - target_w) // 2)
        else:
            x_offset = max(0, (scaled_w - target_w) // 2)
        return (
            f"scale={scaled_w}:{scaled_h},"
            f"crop={target_w}:{target_h}:{x_offset}:0"
        )

    scaled_w = target_w
    scaled_h = int(round(target_w / src_ratio))
    max_offset = max(0, scaled_h - target_h)
    if face_y_norm is None:
        y_offset = max_offset // 2
    else:
        face_y_scaled = face_y_norm * scaled_h
        desired = face_y_scaled - target_h * _BOTTOM_PANEL_FACE_TARGET
        y_offset = int(round(max(0.0, min(float(max_offset), desired))))
    return (
        f"scale={scaled_w}:{scaled_h},"
        f"crop={target_w}:{target_h}:0:{y_offset}"
    )
