import logging
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import TranscriptionResult, ViralClip, WordTimestamp

logger = logging.getLogger(__name__)

_WORD_STYLE = (
    "Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,4,1,5,10,10,800,1"
)
_PHRASE_STYLE = (
    "Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1"
)

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@lru_cache(maxsize=1)
def _ffmpeg_supports_ass() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    return any(" ass " in line for line in result.stdout.splitlines())


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format H:MM:SS.cs (centiseconds)."""
    total_cs = int(round(max(0.0, seconds) * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    """Escape special characters for ASS format."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _filter_words(
    transcription: TranscriptionResult, clip_start: float, clip_end: float
) -> list[WordTimestamp]:
    """Return words whose start falls within [clip_start, clip_end)."""
    result = []
    for segment in transcription.segments:
        for word in segment.words:
            if word.start >= clip_start and word.start < clip_end:
                result.append(word)
    return result


def _generate_word_events(words: list[WordTimestamp], offset: float) -> str:
    """Build ASS Dialogue lines for word-by-word style."""
    lines = []
    for word in words:
        start = _format_ass_time(word.start - offset)
        end = _format_ass_time(word.end - offset)
        text = _escape_ass(word.word.strip())
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return "\n".join(lines)


def _generate_phrase_events(
    transcription: TranscriptionResult,
    clip_start: float,
    clip_end: float,
    offset: float,
) -> str:
    """Build ASS Dialogue lines for phrase/segment style."""
    lines = []
    for segment in transcription.segments:
        if segment.end <= clip_start or segment.start >= clip_end:
            continue
        start = _format_ass_time(max(segment.start, clip_start) - offset)
        end = _format_ass_time(min(segment.end, clip_end) - offset)
        text = _escape_ass(segment.text.strip())
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return "\n".join(lines)


def add_captions(
    clip_path: Path,
    transcription: TranscriptionResult,
    clip: ViralClip,
    config: PipelineConfig,
) -> Path:
    """Burn subtitles into clip_path and return the same path."""
    offset = clip.start_time
    style_line = _WORD_STYLE if config.subtitle_style == "word" else _PHRASE_STYLE
    header = _ASS_HEADER.format(style=style_line)

    if config.subtitle_style == "word":
        words = _filter_words(transcription, clip.start_time, clip.end_time)
        events = _generate_word_events(words, offset)
    else:
        events = _generate_phrase_events(
            transcription, clip.start_time, clip.end_time, offset
        )

    ass_content = header + events + "\n"

    ass_file = clip_path.with_suffix(".ass")
    ass_file.write_text(ass_content, encoding="utf-8")

    if not _ffmpeg_supports_ass():
        logger.warning(
            "FFmpeg sem suporte ao filtro 'ass'; pulando burn-in e mantendo o clipe sem legendas embutidas."
        )
        ass_file.unlink(missing_ok=True)
        return clip_path

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=clip_path.parent
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # `ass` expects filename=... for absolute paths; escape characters parsed by FFmpeg.
        escaped_ass = (
            str(ass_file)
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", r"\'")
            .replace(" ", "\\ ")
        )
        cmd = [
            "ffmpeg",
            "-i", str(clip_path),
            "-vf", f"ass=filename={escaped_ass}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            str(tmp_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            logger.error("FFmpeg falhou ao queimar legendas (código %d): %s", e.returncode, stderr)
            raise
        tmp_path.replace(clip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        ass_file.unlink(missing_ok=True)

    return clip_path
