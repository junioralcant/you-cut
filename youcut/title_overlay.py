import json
import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from youcut.config import PipelineConfig
from youcut.models import ViralClip

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"
_FONT_PATH = _ASSETS_DIR / "Roboto-Regular.ttf"

_FONT_SIZE = 56
_CARD_PADDING = 40
_CARD_RADIUS = 24
_MAX_WIDTH_RATIO = 0.85


def get_video_dimensions(clip_path: Path) -> tuple[int, int]:
    """Return (width, height) via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(clip_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_dominant_color(clip_path: Path) -> tuple[int, int, int]:
    """Extract dominant RGB color from frame 0 of the video via FFmpeg + Pillow."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-ss", "0",
            "-i", str(clip_path),
            "-vframes", "1",
            "-y",
            str(frame_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        img = Image.open(frame_path).convert("RGB")
        img = img.resize((50, 50))
        quantized = img.quantize(colors=5)
        palette = quantized.getpalette()
        dominant = (palette[0], palette[1], palette[2])
        logger.info("Cor dominante extraída: RGB%s de %s", dominant, clip_path)
        return dominant
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        logger.error("FFmpeg falhou ao extrair frame (código %d): %s", e.returncode, stderr)
        raise
    finally:
        frame_path.unlink(missing_ok=True)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Compute relative luminance per W3C WCAG formula."""
    def channel(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def compute_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return white or black with contrast >= 4.5:1 against bg_rgb (WCAG 2.1 AA)."""
    bg_lum = _relative_luminance(bg_rgb)
    white = (255, 255, 255)
    black = (0, 0, 0)
    white_contrast = (1.0 + 0.05) / (bg_lum + 0.05)
    black_contrast = (bg_lum + 0.05) / (0.0 + 0.05)
    return white if white_contrast >= black_contrast else black


def render_card_image(
    title: str,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    video_width: int,
    font_path: Path,
) -> Path:
    """Generate a temporary PNG of the card with rounded corners and wrapped text."""
    try:
        font = ImageFont.truetype(str(font_path), size=_FONT_SIZE)
    except (OSError, IOError):
        logger.warning("Fonte bundled não encontrada em %s; usando fonte padrão.", font_path)
        font = ImageFont.load_default()

    max_text_width = int(video_width * _MAX_WIDTH_RATIO) - 2 * _CARD_PADDING

    # Estimate chars per line using a dummy draw
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    avg_char_width = dummy_draw.textlength("W", font=font)
    chars_per_line = max(1, int(max_text_width / avg_char_width))
    lines = textwrap.wrap(title, width=chars_per_line)
    if not lines:
        lines = [title]

    line_height = _FONT_SIZE + 8
    text_block_width = max(
        int(dummy_draw.textlength(line, font=font)) for line in lines
    )
    text_block_height = len(lines) * line_height

    card_w = text_block_width + 2 * _CARD_PADDING
    card_h = text_block_height + 2 * _CARD_PADDING

    img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (card_w - 1, card_h - 1)],
        radius=_CARD_RADIUS,
        fill=(*bg_color, 220),
    )

    y = _CARD_PADDING
    for line in lines:
        line_w = int(draw.textlength(line, font=font))
        x = (card_w - line_w) // 2
        draw.text((x, y), line, font=font, fill=(*text_color, 255))
        y += line_height

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        card_path = Path(tmp.name)

    img.save(card_path, format="PNG")
    logger.info("Card PNG gerado: %s", card_path)
    return card_path


def add_title_overlay(
    clip_path: Path,
    clip: ViralClip,
    config: PipelineConfig,
) -> Path:
    """Burn title card into first 5s of clip. Returns clip_path unchanged if skipped."""
    if not config.title_overlay:
        return clip_path

    if not clip.title.strip():
        logger.warning("Título vazio para clipe %s; overlay ignorado.", clip_path)
        return clip_path

    logger.info("Iniciando title overlay para %s", clip_path)

    video_width, video_height = get_video_dimensions(clip_path)
    bg_color = extract_dominant_color(clip_path)
    text_color = compute_text_color(bg_color)

    card_path = render_card_image(
        title=clip.title,
        bg_color=bg_color,
        text_color=text_color,
        video_width=video_width,
        font_path=_FONT_PATH,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=clip_path.parent
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        card_w_expr = "overlay_w"
        card_h_expr = "overlay_h"
        x_expr = f"(main_w-{card_w_expr})/2"
        y_expr = f"(main_h-{card_h_expr})/2"

        filter_complex = (
            f"[0:v][1:v]overlay={x_expr}:{y_expr}:enable='lte(t,5)'"
        )

        cmd = [
            "ffmpeg",
            "-i", str(clip_path),
            "-i", str(card_path),
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            logger.error("FFmpeg falhou ao queimar overlay (código %d): %s", e.returncode, stderr)
            raise

        tmp_path.replace(clip_path)
        logger.info("Title overlay concluído para %s", clip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        card_path.unlink(missing_ok=True)

    return clip_path
