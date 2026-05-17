"""Resume an interrupted HFY POC run from a given session directory.

Picks up after scenes.json has been generated/edited. Skips images already
on disk, regenerates missing ones, then runs the compose step.

Usage:
    .venv/bin/python poc/hfy_resume.py poc/hfy/20260517_114012
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from poc.hfy_poc import (
    build_scene_clip,
    compose_final,
    generate_image,
    load_env,
    log,
)
from youcut.captioner import build_ass_for_words
from youcut.models import WordTimestamp


def main(session_dir: Path) -> None:
    load_env()
    oai = OpenAI()

    scenes = json.loads((session_dir / "scenes.json").read_text())
    words_raw = json.loads((session_dir / "transcript.json").read_text())
    words = [WordTimestamp(**w) for w in words_raw]
    total_duration = max(w.end for w in words)
    narration = session_dir / "narration.mp3"
    images_dir = session_dir / "images"
    scenes_dir = session_dir / "scene_clips"
    images_dir.mkdir(exist_ok=True)
    scenes_dir.mkdir(exist_ok=True)

    log(f"Resuming session: {session_dir}")
    log(f"  total narration duration: {total_duration:.2f}s")

    # Images
    image_paths: list[Path] = []
    for i, scene in enumerate(scenes, 1):
        img = images_dir / f"scene_{i:02d}.png"
        if img.exists() and img.stat().st_size > 0:
            log(f"  scene {i}: already exists, skipping ({img.stat().st_size/1024:.0f} KB)")
        else:
            log(f"  scene {i}: gpt-image-1...")
            ti = time.time()
            try:
                generate_image(oai, scene["prompt"], img)
                log(f"    done in {time.time()-ti:.1f}s — {img.stat().st_size/1024:.0f} KB")
            except Exception as exc:
                log(f"    FAILED: {exc}")
                raise
        image_paths.append(img)

    # Scene clips (Ken Burns)
    scene_duration = total_duration / 6
    log(f"Rendering 6 Ken Burns clips ({scene_duration:.2f}s each)...")
    t0 = time.time()
    scene_clip_paths: list[Path] = []
    for i, img in enumerate(image_paths):
        clip = scenes_dir / f"scene_{i+1:02d}.mp4"
        build_scene_clip(img, scene_duration, clip, i)
        scene_clip_paths.append(clip)
    log(f"  done in {time.time()-t0:.1f}s")

    # ASS captions
    ass_path = session_dir / "captions.ass"
    ass_path.write_text(
        build_ass_for_words(words, output_size=(1080, 1920), offset=0.0),
        encoding="utf-8",
    )

    # Final mux
    final = session_dir / "final.mp4"
    log("Composing final video...")
    t0 = time.time()
    compose_final(scene_clip_paths, narration, ass_path, final, session_dir)
    log(f"  done in {time.time()-t0:.1f}s")
    log(f"DONE → {final}  ({final.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: hfy_resume.py <session_dir>")
    main(Path(sys.argv[1]).resolve())
