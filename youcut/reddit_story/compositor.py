"""Compõe vídeo final 1920×1080 — Ken Burns por cena + concat + ASS subs + mux."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from youcut.models import WordTimestamp


class CompositorError(Exception):
    """Falha em qualquer chamada do ffmpeg."""


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompositorError(
            f"ffmpeg falhou ({' '.join(cmd[:4])}...):\n{proc.stderr[-2000:]}"
        )


def build_scene_clip(
    image_path: Path,
    duration: float,
    out_path: Path,
    *,
    idx: int,
    out_w: int = 1920,
    out_h: int = 1080,
) -> None:
    """Ken Burns: scale up + animated crop com pan sinusoidal + leve zoom.
    Direção alterna por idx pra variedade visual.

    Flux Schnell 16:9 entrega ~1408×768; scale up pra 2304×1296 (1.2× canvas)
    dá headroom pra pan/zoom sem revelar borda.
    """
    direction = 1 if idx % 2 == 0 else -1
    scaled_w, scaled_h = int(out_w * 1.2), int(out_h * 1.2)
    vf = (
        f"scale={scaled_w}:{scaled_h}:flags=lanczos,"
        f"crop={out_w}:{out_h}:"
        f"'(iw-{out_w})/2 + {direction}*120*sin(PI*t/{duration:.3f})':"
        f"'(ih-{out_h})/2 + {direction}*70*cos(PI*t/{duration:.3f})',"
        f"format=yuv420p"
    )
    _run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-t", f"{duration:.3f}",
            "-i", str(image_path),
            "-vf", vf,
            "-r", "30",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
    )


def _ass_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def build_ass_16x9(
    words: list[WordTimestamp],
    *,
    res_x: int = 1920,
    res_y: int = 1080,
    font: str = "Arial",
    fontsize: int = 72,
    margin_v: int = 80,
) -> str:
    """ASS word-by-word bottom-third pra 16:9 long-form. Stroke 4 + shadow 2
    pra legibilidade em qualquer fundo."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,"
        f"-1,0,0,0,100,100,0,0,1,4,2,2,40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    events: list[str] = []
    for w in words:
        text = w.word.strip().replace("\n", " ")
        if not text:
            continue
        events.append(
            f"Dialogue: 0,{_ass_time(w.start)},{_ass_time(w.end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def compose_long_form(
    scene_clips: list[Path],
    narration: Path,
    ass_path: Path,
    out_path: Path,
    work_dir: Path,
) -> None:
    """Concat scenes silenciosas → mux narração → burn ASS. Output match
    a duração da narração (-shortest)."""
    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in scene_clips) + "\n"
    )
    silent_video = work_dir / "video_silent.mp4"
    _run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(silent_video),
        ]
    )
    # Copy ASS pra path seguro (filter parser do ffmpeg é sensível a
    # caracteres especiais no nome do arquivo).
    with NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        safe_ass = Path(tmp.name)
    safe_ass.write_bytes(ass_path.read_bytes())
    try:
        _run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_video),
                "-i", str(narration),
                "-vf", f"ass={safe_ass}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(out_path),
            ]
        )
    finally:
        safe_ass.unlink(missing_ok=True)
