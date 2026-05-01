import logging
import json
import math
import random
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont

from youcut.config import PipelineConfig
from youcut.models import ViralClip
from youcut.social_band_styles import BandStyle, PRESETS, select_band_style
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
_DEFAULT_STYLE = PRESETS[0]

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
    style = select_band_style(clip_path.stem)
    logger.info("Social composer: band style preset=%s for clip=%s", style.name, clip_path.stem)
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
        style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
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
        style=style,
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
        style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
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
        style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
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
            style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
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
        style=style,
    )
    ai_path = _render_title_band_image_via_ai(
        title,
        effective_config,
        fallback_path,
        width=width,
        height=height,
        style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
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
        style=style,
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
    style: BandStyle = _DEFAULT_STYLE,
) -> Path:
    effective_config = config.model_copy(
        update={"social_layout_title_color_mode": config.social_layout_title_color_mode}
    )
    bg_hex, text_hex = resolve_title_band_colors(effective_config)
    bg_rgb = _hex_to_rgb(bg_hex, fallback=_DEFAULT_BG)
    text_rgb = _hex_to_rgb(text_hex, fallback=_DEFAULT_TEXT)
    image = _create_band_background(width, height, bg_rgb, style.bg_treatment)
    draw = ImageDraw.Draw(image)

    text_zone = _resolve_text_zone(width, height, style)
    avail_w = max(160, text_zone[2] - text_zone[0])
    avail_h = max(40, text_zone[3] - text_zone[1])

    font_size = style.font_size_start
    line_gap = max(4, style.font_size_start // 14)
    lines: list[str] = []
    font = _load_font_from_style(style, font_size)
    while font_size >= style.font_size_min:
        font = _load_font_from_style(style, font_size)
        lines = _wrap_title(draw, title, font, avail_w, max_lines=_MAX_LABEL_LINES)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
        block_height = sum(line_heights) + (line_gap * max(0, len(lines) - 1))
        max_line_width = max(
            (draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0])
            + style.letter_spacing_px * max(0, len(line) - 1)
            for line in lines
        )
        if block_height <= avail_h and max_line_width <= avail_w:
            break
        font_size -= 6

    _draw_decorations_back(image, draw, width, height, bg_rgb, text_rgb, style)

    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    block_height = sum(line_heights) + (line_gap * max(0, len(lines) - 1))
    zone_x0, zone_y0, zone_x1, zone_y1 = text_zone
    zone_w = zone_x1 - zone_x0
    current_y = zone_y0 + max(0, (zone_y1 - zone_y0 - block_height) // 2)
    text_lines_geometry: list[tuple[str, int, int, int, int]] = []
    for line, bbox, line_height in zip(lines, line_boxes, line_heights):
        line_pixel_width = (
            bbox[2] - bbox[0] + style.letter_spacing_px * max(0, len(line) - 1)
        )
        x = zone_x0 + max(0, (zone_w - line_pixel_width) // 2)
        baseline_y = current_y - bbox[1]
        _draw_styled_text(
            draw,
            (x, baseline_y),
            line,
            font,
            fill=text_rgb,
            shadow_style=style.text_shadow,
            letter_spacing=style.letter_spacing_px,
        )
        text_lines_geometry.append((line, x, baseline_y, line_pixel_width, line_height))
        current_y += line_height + line_gap

    _draw_decorations_front(
        image, draw, width, height, bg_rgb, text_rgb, style, text_lines_geometry,
    )

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
    style: BandStyle = _DEFAULT_STYLE,
) -> str:
    return (
        "Create an editorial title band for a short-form Brazilian social video. "
        f"The final image must be exactly {width}x{height}. "
        "Use the provided reference image only as a layout and proportion guide, then improve the design quality substantially. "
        "Keep the composition expensive, legitimate, and highly clickable. "
        "The band must feel bold and beautiful, not generic or childish. "
        f"### REQUIRED VISUAL TREATMENT (preset: {style.name}) ### {style.ai_directive} "
        "Apply this preset thoroughly so this band looks visually distinct from generic editorial bands. "
        f"Embed exactly this text and no other text: \"{title}\". "
        "Do not paraphrase, translate, add punctuation, add logos, or invent extra words. "
        # TEXT SAFETY — never crop or truncate the title.
        "CRITICAL TEXT RULE: the title must be rendered fully, with every single letter, accent, and word completely visible inside the band. "
        "The title MUST NOT be cropped, clipped, cut off, truncated, or run past the edges of the image — not at the top, bottom, left, or right. "
        "Keep the title strictly inside a generous safe-zone with at least 8% horizontal padding and 12% vertical padding from every edge. "
        "If the title is long, reduce the font size or wrap it onto two lines so the entire text fits — never let any letter be partially or fully outside the visible area. "
        "Do not add ellipsis, do not split words across lines arbitrarily; the text given is short enough to fit fully when sized correctly. "
        "The title must remain fully readable on small mobile screens. "
        "You may use tasteful graphic accents, framing devices, depth, lighting, texture, or layered shapes only if consistent with the preset above. "
        "Do not create a fake screenshot, newspaper, or background scene. This must remain a title band asset. "
        "Avoid red as the dominant color. "
        f"Base background color direction: {bg_hex}. "
        f"Base text color direction: {text_hex}. "
        "Stay inside a yellow / orange editorial palette with dark high-contrast readable text. "
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
    style: BandStyle = _DEFAULT_STYLE,
) -> str:
    min_title_zone = max(140, int(height * 0.18))
    max_title_zone = max(min_title_zone + 20, int(height * 0.35))
    return (
        "Create a premium editorial header for a short-form political social video. "
        f"The final image must be exactly {width}x{height}. "
        "Use the first reference image as the thematic image base and keep its overall subject, mood, and composition believable. "
        "Use the second reference only as a fallback layout guide, not as a style limit. "
        # NO PEOPLE — context-only imagery to avoid fabricated/generic faces.
        "STRICT NO-PEOPLE RULE: do not show any human figure, face, body, hand, silhouette, or person of any kind in the header — full body, partial body, background extras, crowd, or stylized human shapes are all forbidden. "
        "Do not invent or imagine any politician, celebrity, generic stock model, or extra figure. The bottom panel of the final video already shows the real speaker, so this header must stay people-free to avoid confusing the viewer. "
        "Instead, build the visual narrative entirely from contextual elements: environments, places, landscapes, architecture, symbolic objects, props, textures, maps, abstract editorial graphics, atmospheric lighting, or thematic still-life compositions that match the topic of the clip. "
        "The image should evoke the subject through context and atmosphere, not through any human likeness. "
        "Use the reference images only to capture the mood, lighting, environment, and theme — not to depict any person from those frames. "
        f"Embed exactly this title and no other text: \"{title}\". "
        "Do not paraphrase, translate, add punctuation, add logos, or invent extra words. "
        # TEXT SAFETY — never crop or truncate the title.
        "CRITICAL TEXT RULE: the title must be rendered fully, with every single letter, accent, and word completely visible inside the image. "
        "The title MUST NOT be cropped, clipped, cut off, truncated, or run past the edges of the image — not at the top, bottom, left, or right. "
        "Keep the title strictly inside a generous safe-zone with at least 6% padding from every edge of the header. "
        "If the title is long, reduce the font size, wrap it onto two lines, or shrink the title container so the entire text fits — never let any letter be partially or fully outside the visible area. "
        "Do not split words across two lines arbitrarily and do not add ellipsis; the text given is short enough to fit fully when sized correctly. "
        "The title treatment and its background must feel harmonious with the rest of the image, not like a separate flat strip pasted on top. "
        "You are free to choose the internal title container size and proportion based on what looks best for the composition, as long as the entire title fits inside the safe-zone. "
        f"The title zone should visually occupy something between about {min_title_zone} and {max_title_zone} pixels of the header height, "
        "typically near the lower portion of the header, but it does not need to be a rigid rectangle. "
        f"The upper visual region should remain dominant and should roughly preserve the intended editorial image area near {top_height} pixels. "
        f"The previous workflow used a title band around {band_height} pixels, but you may reinterpret that proportion more elegantly. "
        f"### REQUIRED VISUAL TREATMENT FOR THE TITLE ZONE (preset: {style.name}) ### {style.ai_directive} "
        "Apply this preset thoroughly to the title treatment of this header so it looks visually distinct from generic editorial bands. "
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


def _load_font_from_style(style: BandStyle, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(style.font_path), size=size)
    except (OSError, IOError):
        return _load_font(size)


def _draw_bold_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int],
) -> None:
    """Legacy soft-shadow renderer kept for the editorial_clean preset."""
    x, y = position
    shadow = _lighten_rgb(fill, 0.72)
    for dx, dy in ((3, 3), (2, 2), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=fill)


def _draw_styled_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int],
    shadow_style: str,
    letter_spacing: int = 0,
) -> None:
    x, y = position
    if letter_spacing == 0:
        if shadow_style == "soft":
            _draw_bold_text(draw, (x, y), text, font, fill=fill)
            return
        if shadow_style == "hard_offset":
            offset = max(4, font.size // 14)
            shadow_color = _darken_rgb(fill, 0.55)
            draw.text((x + offset, y + offset), text, font=font, fill=shadow_color)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
                draw.text((x + dx, y + dy), text, font=font, fill=fill)
            return
        # no_shadow
        for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            draw.text((x + dx, y + dy), text, font=font, fill=fill)
        return

    cursor_x = x
    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_w = bbox[2] - bbox[0]
        if shadow_style == "hard_offset":
            offset = max(4, font.size // 14)
            shadow_color = _darken_rgb(fill, 0.55)
            draw.text((cursor_x + offset, y + offset), char, font=font, fill=shadow_color)
        elif shadow_style == "soft":
            shadow_color = _lighten_rgb(fill, 0.72)
            for dx, dy in ((3, 3), (2, 2), (1, 1)):
                draw.text((cursor_x + dx, y + dy), char, font=font, fill=shadow_color)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            draw.text((cursor_x + dx, y + dy), char, font=font, fill=fill)
        cursor_x += char_w + letter_spacing


def _resolve_text_zone(width: int, height: int, style: BandStyle) -> tuple[int, int, int, int]:
    if style.accent_style == "side_block":
        return (int(width * 0.20) + 24, 14, width - 60, height - 14)
    if style.accent_style == "stripe_top_bottom":
        # Reserve room for the thick top/bottom stripes painted by the decoration.
        stripe_h = max(10, height // 12) + 6
        return (40, stripe_h, width - 40, height - stripe_h)
    if style.border_style in ("rounded_rect", "double_inset"):
        return (60, 26, width - 60, height - 26)
    if style.border_style == "polygon_burst":
        return (90, 38, width - 90, height - 38)
    if style.border_style == "torn_paper":
        return (50, 22, width - 50, height - 22)
    return (40, 14, width - 40, height - 14)


def _darken_rgb(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, 1.0 - amount))
    return tuple(max(0, min(255, int(channel * factor))) for channel in rgb)


def _lighten_rgb(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, amount))
    return tuple(max(0, min(255, int(channel + ((255 - channel) * factor)))) for channel in rgb)


def _create_band_background(
    width: int,
    height: int,
    bg_rgb: tuple[int, int, int],
    treatment: str = "gradient_smooth",
) -> Image.Image:
    if treatment == "solid_flat":
        return Image.new("RGB", (width, height), color=bg_rgb)

    if treatment == "paper_grain":
        image = Image.new("RGB", (width, height), color=_lighten_rgb(bg_rgb, 0.06))
        rng = random.Random(0xBADCAFE)
        pixels = image.load()
        for _ in range((width * height) // 18):
            x = rng.randrange(width)
            y = rng.randrange(height)
            jitter = rng.randint(-14, 14)
            base = pixels[x, y]
            pixels[x, y] = tuple(max(0, min(255, base[i] + jitter)) for i in range(3))
        return image

    if treatment == "split_blocks":
        top_color = bg_rgb
        # Yellow bg -> orange bottom; orange bg -> yellow bottom; otherwise darken.
        if bg_rgb[1] > 180:
            bottom_color = _hex_to_rgb(_ORANGE_BG, fallback=_ORANGE_BG)
        elif bg_rgb[0] > 220 and bg_rgb[1] < 180:
            bottom_color = _hex_to_rgb(_DEFAULT_BG, fallback=_DEFAULT_BG)
        else:
            bottom_color = _darken_rgb(bg_rgb, 0.28)
        image = Image.new("RGB", (width, height), color=top_color)
        ImageDraw.Draw(image).rectangle(
            [(0, height // 2), (width, height)], fill=bottom_color
        )
        return image

    if treatment == "burst_rays":
        # Darker base + subtler rays so the bright starburst polygon (rendered later
        # by the decoration layer) reads as a clean text surface in front of them.
        image = Image.new("RGB", (width, height), color=_darken_rgb(bg_rgb, 0.14))
        draw = ImageDraw.Draw(image)
        cx, cy = width // 2, height // 2
        ray_color = _darken_rgb(bg_rgb, 0.30)
        radius = int(math.hypot(width, height))
        for i in range(24):
            angle = (math.tau * i) / 24
            x2 = cx + int(math.cos(angle) * radius)
            y2 = cy + int(math.sin(angle) * radius)
            draw.line([(cx, cy), (x2, y2)], fill=ray_color, width=14 if i % 2 == 0 else 4)
        return image

    # gradient_smooth (default)
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


def _draw_decorations_back(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    bg_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int],
    style: BandStyle,
) -> None:
    """Decorations rendered BEFORE the text (background-level)."""
    if style.border_style == "rounded_rect":
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
    elif style.border_style == "sharp_rect":
        if style.accent_style == "stripe_top_bottom":
            stripe_color = _darken_rgb(text_rgb, 0.0)
            stripe_h = max(8, height // 12)
            draw.rectangle([(0, 0), (width, stripe_h)], fill=stripe_color)
            draw.rectangle([(0, height - stripe_h), (width, height)], fill=stripe_color)
    elif style.border_style == "double_inset":
        outer = _darken_rgb(bg_rgb, 0.40)
        inner = _darken_rgb(bg_rgb, 0.30)
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=outer, width=3)
        draw.rectangle([(34, 34), (width - 34, height - 34)], outline=inner, width=2)
    elif style.border_style == "torn_paper":
        sticker_color = _lighten_rgb(bg_rgb, 0.06)
        shadow_color = _darken_rgb(bg_rgb, 0.45)
        polygon = _torn_polygon(width, height, inset_x=24, inset_y=14, jitter=10, seed=style.name)
        offset = [(x + 6, y + 8) for (x, y) in polygon]
        draw.polygon(offset, fill=shadow_color)
        draw.polygon(polygon, fill=sticker_color)
    elif style.border_style == "polygon_burst":
        burst_color = _lighten_rgb(bg_rgb, 0.10)
        outline_color = _darken_rgb(text_rgb, 0.0)
        polygon = _starburst_polygon(width, height, points=10, jitter=0.08, seed=style.name)
        draw.polygon(polygon, fill=burst_color, outline=outline_color)

    if style.accent_style == "side_block":
        block_w = int(width * 0.18)
        block_color = _darken_rgb(text_rgb, 0.0)
        draw.rectangle([(0, 0), (block_w, height)], fill=block_color)
        dot_r = max(14, height // 8)
        dot_cx = block_w // 2
        dot_cy = height // 2
        draw.ellipse(
            [(dot_cx - dot_r, dot_cy - dot_r), (dot_cx + dot_r, dot_cy + dot_r)],
            fill=_lighten_rgb(bg_rgb, 0.18),
        )

    if style.accent_style == "chevrons":
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

    if style.accent_style == "corner_squares":
        sq_size = max(20, height // 8)
        accent_dark = _darken_rgb(text_rgb, 0.0)
        accent_light = _darken_rgb(bg_rgb, 0.32)
        draw.rectangle([(18, 18), (18 + sq_size, 18 + sq_size)], fill=accent_dark)
        draw.rectangle(
            [(width - 18 - sq_size, height - 18 - sq_size), (width - 18, height - 18)],
            fill=accent_light,
        )
        dot_r = sq_size // 3
        cx = width - 18 - dot_r * 2
        cy = 18 + dot_r * 2
        draw.ellipse([(cx - dot_r, cy - dot_r), (cx + dot_r, cy + dot_r)], fill=accent_dark)


def _draw_decorations_front(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    bg_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int],
    style: BandStyle,
    text_lines: list[tuple[str, int, int, int, int]],
) -> None:
    """Decorations rendered AFTER the text (foreground-level)."""
    if style.accent_style == "marker_underline" and text_lines:
        line, x, baseline_y, line_w, line_h = text_lines[-1]
        underline_color = _darken_rgb(text_rgb, 0.0)
        underline_y = baseline_y + line_h + max(6, line_h // 8)
        thickness = max(8, line_h // 9)
        rng = random.Random(hash(("under", style.name)) & 0xFFFFFFFF)
        prev_x = x - 6
        prev_y = underline_y + rng.randint(-3, 3)
        for step in range(1, 14):
            nx = x + int((line_w + 12) * step / 13)
            ny = underline_y + rng.randint(-5, 5)
            draw.line([(prev_x, prev_y), (nx, ny)], fill=underline_color, width=thickness)
            prev_x, prev_y = nx, ny


def _torn_polygon(
    width: int,
    height: int,
    *,
    inset_x: int,
    inset_y: int,
    jitter: int,
    seed: str,
) -> list[tuple[int, int]]:
    rng = random.Random(hash(("torn", seed)) & 0xFFFFFFFF)
    points: list[tuple[int, int]] = []
    steps = 24
    # top edge
    for i in range(steps + 1):
        x = inset_x + int((width - inset_x * 2) * i / steps)
        y = inset_y + rng.randint(-jitter, jitter)
        points.append((x, y))
    # right edge
    for i in range(1, steps + 1):
        x = width - inset_x + rng.randint(-jitter, jitter)
        y = inset_y + int((height - inset_y * 2) * i / steps)
        points.append((x, y))
    # bottom edge
    for i in range(1, steps + 1):
        x = width - inset_x - int((width - inset_x * 2) * i / steps)
        y = height - inset_y + rng.randint(-jitter, jitter)
        points.append((x, y))
    # left edge
    for i in range(1, steps):
        x = inset_x + rng.randint(-jitter, jitter)
        y = height - inset_y - int((height - inset_y * 2) * i / steps)
        points.append((x, y))
    return points


def _starburst_polygon(
    width: int,
    height: int,
    *,
    points: int,
    jitter: float,
    seed: str,
) -> list[tuple[int, int]]:
    rng = random.Random(hash(("burst", seed)) & 0xFFFFFFFF)
    cx = width / 2
    cy = height / 2
    rx_outer = (width / 2) - 12
    ry_outer = (height / 2) - 8
    # Shallow notches keep a clean, readable surface under the title;
    # the spikes just nudge past the rectangle for the comic-burst silhouette.
    rx_inner = rx_outer * 0.88
    ry_inner = ry_outer * 0.88
    coords: list[tuple[int, int]] = []
    total = points * 2
    for i in range(total):
        angle = (math.tau * i) / total - math.pi / 2
        is_outer = (i % 2 == 0)
        rx = rx_outer if is_outer else rx_inner
        ry = ry_outer if is_outer else ry_inner
        wob = 1.0 + rng.uniform(-jitter, jitter)
        x = cx + math.cos(angle) * rx * wob
        y = cy + math.sin(angle) * ry * wob
        coords.append((int(x), int(y)))
    return coords


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
