import logging
import subprocess
import tempfile
from pathlib import Path

from youcut.models import CaptionBurnResult

logger = logging.getLogger(__name__)

# Force-style params for SRT burn-in: Arial Bold 48, white fill, black outline 2px,
# bottom-center alignment, MarginV ~15% of typical 1920px height = 288px.
_SRT_FORCE_STYLE = (
    "FontName=Arial,Bold=1,FontSize=48,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "Outline=2,Alignment=2,MarginV=288"
)


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class CaptionBurner:
    def burn(self, video_path: Path, style: str = "word") -> CaptionBurnResult:
        try:
            words = self._transcribe_words(video_path)
        except Exception as exc:
            warning = f"Transcrição falhou: {exc}"
            logger.warning("CaptionBurner: transcrição falhou para %s: %s — sem legenda.", video_path.name, exc)
            return CaptionBurnResult(output_path=video_path, captions_applied=False, warning=warning)

        try:
            srt_path = self._write_word_srt(words, video_path)
        except Exception as exc:
            warning = f"Geração de SRT falhou: {exc}"
            logger.warning("CaptionBurner: falha ao gerar SRT para %s: %s — sem legenda.", video_path.name, exc)
            return CaptionBurnResult(output_path=video_path, captions_applied=False, warning=warning)

        try:
            output_path = self._ffmpeg_burn(video_path, srt_path)
            return CaptionBurnResult(output_path=output_path, captions_applied=True)
        except Exception as exc:
            warning = f"FFmpeg falhou: {exc}"
            logger.warning("CaptionBurner: FFmpeg falhou para %s: %s — sem legenda.", video_path.name, exc)
            return CaptionBurnResult(output_path=video_path, captions_applied=False, warning=warning)
        finally:
            srt_path.unlink(missing_ok=True)

    def _transcribe_words(self, video_path: Path) -> list[dict]:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            model = WhisperModel("base", device="auto")
            segments_iter, _ = model.transcribe(str(video_path), word_timestamps=True)
            words = []
            for seg in segments_iter:
                if seg.words:
                    for w in seg.words:
                        words.append({"word": w.word, "start": w.start, "end": w.end})
            return words
        except ImportError:
            pass

        import whisper  # type: ignore[import]

        model = whisper.load_model("base")
        result = model.transcribe(str(video_path), word_timestamps=True)
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        return words

    def _write_word_srt(self, words: list[dict], video_path: Path) -> Path:
        srt_path = video_path.with_suffix(".srt")
        lines = []
        for i, w in enumerate(words, start=1):
            start = _format_srt_time(w["start"])
            end = _format_srt_time(w["end"])
            text = w["word"].strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        return srt_path

    def _ffmpeg_burn(self, video_path: Path, srt_path: Path) -> Path:
        out_path = video_path.with_stem(video_path.stem + "_captioned")

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, dir=tempfile.gettempdir()) as tmp:
            safe_srt = Path(tmp.name)
        safe_srt.write_bytes(srt_path.read_bytes())

        try:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"subtitles={safe_srt}:force_style='{_SRT_FORCE_STYLE}'",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-y",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            safe_srt.unlink(missing_ok=True)

        return out_path
