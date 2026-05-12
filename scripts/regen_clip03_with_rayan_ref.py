"""Teste: regerar thumbnail + recompor MP4 social do clip_03 (Rayan) usando
uma foto externa do jogador como reference_image adicional para gpt-image-1.5.

Preserva versões atuais em `_v1/` antes de sobrescrever.

Run:
    .venv/bin/python scripts/regen_clip03_with_rayan_ref.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "SAIU A LISTA! OS 55 CONVOCADOS PRA COPA"
CLIP_BASENAME = "clip_03"
CLIP_TXT = OUTPUT_DIR / f"{CLIP_BASENAME}.txt"
CLIP_SOURCE_VIDEO = OUTPUT_DIR / f"{CLIP_BASENAME}.mp4"  # original 9:16 sem header
CLIP_SOCIAL_CAPTIONED = OUTPUT_DIR / f"{CLIP_BASENAME}_framed_social_captioned.mp4"
THUMBNAIL_PATH = OUTPUT_DIR / "thumbnails" / f"{CLIP_BASENAME}.png"
RAYAN_WEBP = REPO_ROOT / "rayan_comemora_vasco-e1768997216344.webp"

HEADER_H = 740
TOP_H = 600
BAND_H = 140
CANVAS_W = 1080
CANVAS_H = 1920
THUMB_W = 1280
THUMB_H = 720

SKILL_SCRIPT = REPO_ROOT / ".claude" / "skills" / "thumbnail-generator" / "scripts" / "generate_thumbnail.py"


def _load_env() -> str:
    """Carrega OPENAI_API_KEY do .env do projeto."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Erro: OPENAI_API_KEY ausente.")
    return api_key


def _backup(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "_v1"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    if target.exists():
        print(f"[backup] {target} já existe; mantendo.")
        return
    shutil.copy2(path, target)
    print(f"[backup] {path.name} → {target}")


def _webp_to_png_tmp(webp_path: Path) -> Path:
    with Image.open(webp_path) as img:
        rgb = img.convert("RGB")
        # Limit to ~1024 longest side to keep API payload small.
        rgb.thumbnail((1024, 1024), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        rgb.save(tmp.name, format="PNG", optimize=True)
        return Path(tmp.name)


def _parse_clip_txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    current_key: str | None = None
    buf: list[str] = []
    SECTIONS = {
        "TÍTULO": "title",
        "DESCRIÇÃO": "description",
        "HASHTAGS": "hashtags",
        "SUGESTÃO DE THUMBNAIL": "thumbnail_idea",
        "MOTIVO DA SELEÇÃO": "reason",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() in SECTIONS:
            if current_key is not None:
                out[current_key] = "\n".join(buf).strip()
            current_key = SECTIONS[line.strip()]
            buf = []
        elif current_key is not None:
            buf.append(line)
    if current_key is not None:
        out[current_key] = "\n".join(buf).strip()
    return out


def _build_thumbnail_prompt(meta: dict[str, str]) -> str:
    return (
        "Premium YouTube thumbnail for a Brazilian football/humor short clip. "
        f"Title: {meta.get('title','')}. "
        f"Thumbnail idea: {meta.get('thumbnail_idea','')}. "
        "Use the FIRST reference image (Rayan, the Bournemouth player) as the main subject — keep his face and identity unmistakable. "
        "The other reference frames are context (the streamer's clip); use them only for mood/style consistency. "
        "Composition: cinematic close-up of Rayan with a humorous/conflicted expression (hand near face if natural). "
        "Vibrant saturated palette dominated by cyan/green/yellow/orange editorial energy. Avoid red as dominant. "
        "Strong contrast, expressive face, clear subject/background separation. "
        "Do NOT embed text in the image. No logos, no watermarks, no captions. "
        "Final output should look like a high-CTR YouTube thumbnail."
    )


def _build_header_prompt(meta: dict[str, str], label: str) -> str:
    return (
        "Create a premium editorial header for a Brazilian short-form social video about football/humor. "
        f"The final image must be exactly {CANVAS_W}x{HEADER_H}. "
        "Use the FIRST reference image (Rayan, the Bournemouth player) as the main subject for the upper visual region. "
        f"Title: {meta.get('title','')}. "
        f"Reason: {meta.get('reason','')}. "
        f"Embed exactly this title text and no other text: \"{label}\". "
        "Do not paraphrase, translate, or invent extra words. "
        "Title rendered fully, all letters visible, generous safe-zone (≥6% padding). Wrap onto two lines if needed; do not truncate. "
        f"Title zone occupies roughly the bottom {BAND_H}px of the header on a yellow/orange editorial band; the upper {TOP_H}px holds Rayan's portrait/composition. "
        "Vibrant saturated palette (cyan/green/yellow/orange editorial energy). Avoid red as dominant. "
        "Polished, expensive, native-Brazilian editorial style. Return only the completed header image."
    )


def _run_skill_script(
    prompt: str,
    references: list[Path],
    *,
    size: str,
    api_key: str,
) -> bytes:
    """Invoca o skill script externo `generate_thumbnail.py` e retorna os bytes do PNG."""
    if not SKILL_SCRIPT.exists():
        sys.exit(f"Skill script ausente: {SKILL_SCRIPT}")

    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key
        cmd = [
            sys.executable,
            str(SKILL_SCRIPT),
            prompt,
            "out.png",
            *[str(r) for r in references],
            "--model", "gpt-image-1.5",
            "--size", size,
            "--quality", "low",
        ]
        print(f"[skill] size={size} refs={len(references)} → invocando…")
        result = subprocess.run(
            cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            sys.exit(f"Skill script falhou (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        out_png = Path(tmp) / "thumbnails" / "out.png"
        if not out_png.exists():
            sys.exit(f"Skill script não gerou {out_png}.\nSTDOUT:\n{result.stdout}")
        return out_png.read_bytes()


def _extract_frame(video: Path, ts: float = 5.0) -> bytes:
    """Extrai 1 frame em PNG via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out = Path(tmp.name)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", str(video),
        "-frames:v", "1", "-f", "image2", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    data = out.read_bytes()
    out.unlink(missing_ok=True)
    return data


def _save_bytes_resized(image_bytes: bytes, target: tuple[int, int], dest: Path) -> None:
    with Image.open(BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB").resize(target, Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest, format="PNG", optimize=True)
    print(f"[saved] {target[0]}×{target[1]} → {dest}")


def _replace_header_in_video(
    source_video: Path,
    new_header_png: Path,
    dest_video: Path,
) -> None:
    """Substitui apenas os primeiros HEADER_H pixels do vídeo pelo novo header.

    Preserva o bottom (com legendas burnt-in) intacto.
    """
    filter_complex = (
        f"[0:v]crop={CANVAS_W}:{CANVAS_H - HEADER_H}:0:{HEADER_H}[bottom];"
        f"[1:v]scale={CANVAS_W}:{HEADER_H}[hdr];"
        f"[hdr][bottom]vstack[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),
        "-i", str(new_header_png),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        str(dest_video),
    ]
    print(f"[ffmpeg] recompondo {dest_video.name} (preservando legendas do bottom)…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg falhou:\n{result.stderr[-2000:]}")
    print(f"[done] {dest_video}")


def main() -> None:
    api_key = _load_env()

    if not CLIP_TXT.exists():
        sys.exit(f"clip_03.txt ausente: {CLIP_TXT}")
    if not CLIP_SOCIAL_CAPTIONED.exists():
        sys.exit(f"vídeo social ausente: {CLIP_SOCIAL_CAPTIONED}")
    if not RAYAN_WEBP.exists():
        sys.exit(f"foto do Rayan ausente: {RAYAN_WEBP}")

    print(f"[setup] OUTPUT_DIR={OUTPUT_DIR}")
    meta = _parse_clip_txt(CLIP_TXT)
    print(f"[meta] title={meta.get('title','')!r}")

    rayan_png = _webp_to_png_tmp(RAYAN_WEBP)
    print(f"[ref] Rayan → {rayan_png}")

    frame_bytes = _extract_frame(CLIP_SOCIAL_CAPTIONED, ts=5.0)
    frame_tmp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
    frame_tmp.write_bytes(frame_bytes)
    print(f"[ref] frame do clipe → {frame_tmp}")

    label = "RAYAN VAI PRA COPA?"

    # --- 1) THUMBNAIL (1536x1024 → resize 1280x720) -----------------------
    _backup(THUMBNAIL_PATH)
    thumb_prompt = _build_thumbnail_prompt(meta)
    thumb_bytes = _run_skill_script(
        thumb_prompt,
        references=[rayan_png, frame_tmp],
        size="1536x1024",
        api_key=api_key,
    )
    _save_bytes_resized(thumb_bytes, (THUMB_W, THUMB_H), THUMBNAIL_PATH)

    # --- 2) HEADER do MP4 social (1024x1024 → resize 1080x740) ------------
    _backup(CLIP_SOCIAL_CAPTIONED)
    header_prompt = _build_header_prompt(meta, label)
    header_bytes = _run_skill_script(
        header_prompt,
        references=[rayan_png, frame_tmp],
        size="1024x1024",
        api_key=api_key,
    )
    header_png = OUTPUT_DIR / "social_images" / f"{CLIP_BASENAME}_header.png"
    _save_bytes_resized(header_bytes, (CANVAS_W, HEADER_H), header_png)

    tmp_out = CLIP_SOCIAL_CAPTIONED.with_suffix(".regen.tmp.mp4")
    _replace_header_in_video(
        source_video=CLIP_SOCIAL_CAPTIONED,
        new_header_png=header_png,
        dest_video=tmp_out,
    )
    shutil.move(str(tmp_out), str(CLIP_SOCIAL_CAPTIONED))

    # cleanup
    rayan_png.unlink(missing_ok=True)
    frame_tmp.unlink(missing_ok=True)

    print("\n=== OK ===")
    print(f"Thumbnail nova: {THUMBNAIL_PATH}")
    print(f"MP4 social novo: {CLIP_SOCIAL_CAPTIONED}")
    print(f"Header PNG salvo em: {header_png}")
    print(f"Versões anteriores preservadas em {THUMBNAIL_PATH.parent / '_v1'} e {CLIP_SOCIAL_CAPTIONED.parent / '_v1'}")


if __name__ == "__main__":
    main()
