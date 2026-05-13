import logging
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from youcut.config import PipelineConfig
from youcut.models import TranscriptionResult, ViralClip, WordTimestamp

logger = logging.getLogger(__name__)

ASSETS_FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
SERIF_FONT_FILE = ASSETS_FONTS_DIR / "EBGaramond-Regular.ttf"
SERIF_FONT_NAME = "EB Garamond"
# Fonte do preset "motivacao". Crimson Text BoldItalic foi escolhida via
# comparação visual frame-a-frame com o vídeo de referência (tasks/prd-preset-
# motivacao/analise-video-referencia.md §4). Letras compactas + peso bold +
# serifas sutis bateram melhor que Lora/Playfair/Merriweather/Source Serif.
MOTIVACAO_ITALIC_FONT_FILE = ASSETS_FONTS_DIR / "CrimsonText-BoldItalic.ttf"
MOTIVACAO_ITALIC_FONT_NAME = "Crimson Text"

_WORD_STYLE = (
    "Style: Default,Arial,110,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,5,1,8,10,10,820,1"
)
_PHRASE_STYLE = (
    "Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,3,1,2,10,10,60,1"
)
# MarginV define a distância do texto até a borda inferior do canvas
# (Alignment=2 = bottom-center). Para canvas 1920: MarginV=864 ≈ Y=55% (solo),
# MarginV=480 ≈ Y=75% (sobre o speaker no layout speaker_bottom_ai_top).
_PHRASE_SERIF_MARGIN_V_DEFAULT = 864
_PHRASE_SERIF_MARGIN_V_SPEAKER_BOTTOM = 480

# Posições absolutas (PlayRes 1080×1920) usadas pelo preset motivacao.
# Y=964 = centro vertical alinhado com a legenda do vídeo de referência.
# Offset=80 dimensiona pra fontsize=130 (Default) + 65 (Handle) — handle fica
# colado abaixo da legenda principal sem sobreposição.
_MOTIVACAO_WORD_X = 540
_MOTIVACAO_WORD_Y = 964
_MOTIVACAO_HANDLE_Y_OFFSET = 80  # gap centro-a-centro entre palavra e handle


def _phrase_serif_centered_style(margin_v: int) -> str:
    """Monta a linha Style do preset serif com MarginV adaptativo.

    Alignment=2 (bottom-center, MarginV é distância da borda inferior — comportamento
    previsível). Shadow=3 + BackColour ~63% preto dá um drop shadow sutil que
    mantém legibilidade em fundos claros sem virar contorno duro tipo
    "TikTok karaokê".
    """
    return (
        f"Style: Default,{SERIF_FONT_NAME},96,"
        "&H00FFFFFF,&H000000FF,&H00000000,&H40000000,"
        "0,0,0,0,100,100,0,0,1,0,4,2,40,40,"
        f"{margin_v},1"
    )


# Janela máxima de palavras agrupadas num chunk de legenda serif.
_PHRASE_SERIF_MAX_WORDS = 4
# Gap (segundos) entre palavras que força quebra de chunk.
_PHRASE_SERIF_GAP_BREAK_S = 0.35
# Duração mínima visível de um chunk (segundos) — chunks muito curtos
# são estendidos pra evitar piscar.
_PHRASE_SERIF_MIN_DURATION_S = 0.8
# Duração máxima visível de um chunk (segundos).
_PHRASE_SERIF_MAX_DURATION_S = 3.0

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _word_serif_italic_styles() -> str:
    """Estilos ASS do preset motivacao: legenda (Default) + handle (Handle).

    Fontsize=130 para a legenda; handle fontsize=70 totalmente opaco.
    Outline preto leve (2px Default, 1px Handle) pra destacar texto sobre
    fundos variados sem virar "TikTok karaokê". Shadow 3 + BackColour 50%
    preto preserva o "halo" sutil do referência.
    """
    default = (
        f"Style: Default,{MOTIVACAO_ITALIC_FONT_NAME},130,"
        "&H00FFFFFA,&H000000FF,&H00000000,&H80000000,"
        "0,-1,0,0,100,100,0,0,1,2,3,5,40,40,0,1"
    )
    handle = (
        f"Style: Handle,{MOTIVACAO_ITALIC_FONT_NAME},70,"
        "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,-1,0,0,100,100,0,0,1,1,3,5,40,40,0,1"
    )
    return default + "\n" + handle


_MOTIVACAO_MAX_WORDS_PER_CHUNK = 3
_MOTIVACAO_GAP_BREAK_S = 0.35
_MOTIVACAO_MAX_DURATION_S = 3.0
_MOTIVACAO_MIN_DURATION_S = 0.6


def _chunk_words_for_motivacao(
    words: list[WordTimestamp],
) -> list[list[WordTimestamp]]:
    """Agrupa palavras em chunks de 1–3 com base em gaps de fala e duração.

    Quebra um chunk quando: (1) atinge ``_MOTIVACAO_MAX_WORDS_PER_CHUNK``,
    (2) o gap para a próxima palavra excede ``_MOTIVACAO_GAP_BREAK_S``,
    (3) o chunk já dura mais que ``_MOTIVACAO_MAX_DURATION_S``.
    """
    chunks: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    for i, w in enumerate(words):
        current.append(w)
        chunk_dur = current[-1].end - current[0].start
        next_gap = (
            words[i + 1].start - w.end if i + 1 < len(words) else float("inf")
        )
        should_break = (
            len(current) >= _MOTIVACAO_MAX_WORDS_PER_CHUNK
            or next_gap >= _MOTIVACAO_GAP_BREAK_S
            or chunk_dur >= _MOTIVACAO_MAX_DURATION_S
        )
        if should_break:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _generate_word_serif_italic_events(
    words: list[WordTimestamp], offset: float, handle: str | None
) -> str:
    """Eventos por chunk (1-3 palavras) centralizados + handle opcional.

    Cada chunk produz 1 Dialogue 'Default' posicionado em
    (_MOTIVACAO_WORD_X, _MOTIVACAO_WORD_Y) via \\pos. Quando ``handle`` é
    não-vazio, emite também 1 Dialogue 'Handle' sincronizado,
    ``_MOTIVACAO_HANDLE_Y_OFFSET`` px abaixo.
    """
    chunks = _chunk_words_for_motivacao(words)
    lines: list[str] = []
    handle_clean = (handle or "").strip().lstrip("@")
    # Handle persistente: um único Dialogue que dura do início ao fim do
    # clipe (usamos 9:59:59.99 como sentinela "fim infinito"). Assim o
    # @handle não pisca junto com as palavras — fica fixo embaixo o tempo todo.
    if handle_clean:
        handle_y = _MOTIVACAO_WORD_Y + _MOTIVACAO_HANDLE_Y_OFFSET
        pos_handle = f"{{\\pos({_MOTIVACAO_WORD_X},{handle_y})}}"
        handle_text = _escape_ass(f"@{handle_clean}")
        lines.append(
            f"Dialogue: 0,0:00:00.00,9:59:59.99,Handle,,0,0,0,,{pos_handle}{handle_text}"
        )
    for i, chunk in enumerate(chunks):
        start_t = chunk[0].start - offset
        end_t = chunk[-1].end - offset
        # Estender chunks curtos pra MIN_DURATION sem invadir o próximo
        # (evita sobreposição visual onde dois chunks renderizam juntos).
        if end_t - start_t < _MOTIVACAO_MIN_DURATION_S:
            desired_end = start_t + _MOTIVACAO_MIN_DURATION_S
            if i + 1 < len(chunks):
                next_start = chunks[i + 1][0].start - offset
                desired_end = min(desired_end, next_start)
            end_t = max(end_t, desired_end)
        start = _format_ass_time(start_t)
        end = _format_ass_time(end_t)
        text = _escape_ass(" ".join(w.word.strip() for w in chunk).strip())
        if not text:
            continue
        pos_word = f"{{\\pos({_MOTIVACAO_WORD_X},{_MOTIVACAO_WORD_Y})}}"
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_word}{text}"
        )
    return "\n".join(lines)


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


def _chunk_words_for_serif(
    words: list[WordTimestamp],
) -> list[list[WordTimestamp]]:
    """Agrupa palavras em chunks de 1–4 com base em gaps de fala.

    Quebra um chunk quando: (1) atinge `_PHRASE_SERIF_MAX_WORDS`, (2) o gap
    para a próxima palavra excede `_PHRASE_SERIF_GAP_BREAK_S`, ou (3) o chunk
    já dura mais que `_PHRASE_SERIF_MAX_DURATION_S`.
    """
    chunks: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    for i, w in enumerate(words):
        current.append(w)
        chunk_dur = current[-1].end - current[0].start
        next_gap = (
            words[i + 1].start - w.end if i + 1 < len(words) else float("inf")
        )
        should_break = (
            len(current) >= _PHRASE_SERIF_MAX_WORDS
            or next_gap >= _PHRASE_SERIF_GAP_BREAK_S
            or chunk_dur >= _PHRASE_SERIF_MAX_DURATION_S
        )
        if should_break:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _generate_phrase_serif_events(
    words: list[WordTimestamp], offset: float
) -> str:
    """Emite Dialogue ASS para o preset serif centralizado."""
    chunks = _chunk_words_for_serif(words)
    lines: list[str] = []
    for chunk in chunks:
        start_t = chunk[0].start - offset
        end_t = chunk[-1].end - offset
        duration = end_t - start_t
        if duration < _PHRASE_SERIF_MIN_DURATION_S:
            end_t = start_t + _PHRASE_SERIF_MIN_DURATION_S
        start = _format_ass_time(start_t)
        end = _format_ass_time(end_t)
        text = _escape_ass(" ".join(w.word.strip() for w in chunk).strip())
        if not text:
            continue
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


def build_ass_for_words(
    words: list[WordTimestamp],
    *,
    output_size: tuple[int, int] = (1080, 1920),
    offset: float = 0.0,
) -> str:
    """Constrói um documento ASS palavra-a-palavra reusável.

    ``words`` deve estar com timestamps absolutos; ``offset`` os normaliza
    para o início do segmento alvo. ``output_size`` ajusta ``PlayResX``/Y.
    """
    res_x, res_y = output_size
    header = _ASS_HEADER.format(res_x=res_x, res_y=res_y, style=_WORD_STYLE)
    events = _generate_word_events(words, offset)
    return header + events + "\n"


def add_captions(
    clip_path: Path,
    transcription: TranscriptionResult,
    clip: ViralClip,
    config: PipelineConfig,
) -> Path:
    """Burn subtitles into clip_path and return the same path."""
    offset = clip.start_time
    style = config.subtitle_style
    if style == "word":
        style_line = _WORD_STYLE
    elif style == "phrase_serif_centered":
        margin_v = (
            _PHRASE_SERIF_MARGIN_V_SPEAKER_BOTTOM
            if config.social_layout_mode == "speaker_bottom_ai_top"
            else _PHRASE_SERIF_MARGIN_V_DEFAULT
        )
        style_line = _phrase_serif_centered_style(margin_v)
    elif style == "word_serif_italic":
        style_line = _word_serif_italic_styles()
    else:
        style_line = _PHRASE_STYLE
    header = _ASS_HEADER.format(res_x=1080, res_y=1920, style=style_line)

    if style == "word":
        words = _filter_words(transcription, clip.start_time, clip.end_time)
        events = _generate_word_events(words, offset)
    elif style == "phrase_serif_centered":
        words = _filter_words(transcription, clip.start_time, clip.end_time)
        events = _generate_phrase_serif_events(words, offset)
    elif style == "word_serif_italic":
        words = _filter_words(transcription, clip.start_time, clip.end_time)
        events = _generate_word_serif_italic_events(
            words, offset, config.motivacao_handle
        )
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

    # Copy ASS to a temp file with a safe name so FFmpeg filter parsing isn't
    # confused by commas, quotes, or other special chars in the original path.
    with tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, dir=tempfile.gettempdir()
    ) as safe_ass_tmp:
        safe_ass_path = Path(safe_ass_tmp.name)

    try:
        safe_ass_path.write_bytes(ass_file.read_bytes())
        ass_filter = f"ass={safe_ass_path}"
        if ASSETS_FONTS_DIR.exists():
            ass_filter += f":fontsdir={ASSETS_FONTS_DIR}"
        cmd = [
            "ffmpeg",
            "-i", str(clip_path),
            "-vf", ass_filter,
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
        safe_ass_path.unlink(missing_ok=True)

    return clip_path
