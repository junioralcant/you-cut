"""POC v2: Cosmic horror — lonely cosmonaut receives transmissions from his future self.

Tests Flux Schnell (Replicate) as image backend for cost comparison vs gpt-image-1.
Reuses hfy_poc.py helpers for everything except image generation.

Run:
    .venv/bin/python poc/cosmic_poc.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import anthropic
import httpx
import replicate
from openai import OpenAI

from poc.hfy_poc import (
    CLAUDE_MODEL,
    build_scene_clip,
    compose_final,
    generate_tts,
    load_env,
    log,
    transcribe_words,
)
from youcut.captioner import build_ass_for_words


SCRIPT_PROMPT = """You are writing a SHORT first-person cosmic-horror narration for a 60-second YouTube Short.

Premise: A lone cosmonaut aboard a derelict deep-space station starts receiving radio transmissions. The voice on the other end is HIS OWN — and it's warning him about something that hasn't happened yet.

Constraints:
- TARGET: 160-170 words of pure narration. This must read aloud in ~55-58 seconds (the video locks at 60s with a brief end-card hold).
- First-person, present tense, low-grave delivery.
- Tight hook in the first sentence. Build dread, not jump-scares.
- Structure: setup -> first transmission -> realization (the voice is HIS) -> chilling warning -> twist (the warning has already happened OR he is the one making the transmissions OR he is the one being warned about).
- Tone: 2001 Space Odyssey + The Last Voyage of the Demeter + r/nosleep.
- The last line must be a haunting kicker that recontextualizes everything.
- Output ONLY the narration text, no preamble, no headers, no quotes."""


SCENES_PROMPT = """You are a cosmic-horror visual designer. Given a 60-second narration, produce 6 image prompts (one per scene beat) for the Flux Schnell image model.

Hard rules:
- Output VALID JSON only, schema below, no markdown fences.
- Each prompt must be self-contained and describe a STILL CINEMATIC frame for a 9:16 vertical Short.
- Visual style across all 6: SATURATED COLD palette - deep cobalt blue, sickly fluorescent green, dim crimson warning lights, pitch black. VIBRANT but cold, NOT pastel, NOT washed out, NOT sepia. Cinematic 35mm, volumetric god rays, dust particles floating in zero-G, lens flares, dramatic chiaroscuro. Tarkovsky meets Alien meets Event Horizon.
- Each prompt must include: the cosmonaut (face hidden by reflective helmet visor, scratched white-and-orange Soviet-style suit with cyrillic patches), setting (cramped derelict station interior, frost on portholes, dead instruments, glowing red telegraph speaker), and a recurring visual symbol of a black radio waveform.
- NO TEXT, NO LOGOS, NO WORDS in the image.
- Vary the camera per beat: wide isolation establishing -> medium speaker closeup -> POV helmet HUD -> mirror/reflection shock -> wide cosmic vista with station tiny -> final silhouette walking into the dark.
- Each prompt ≤ 60 words.

Narration:
<<SCRIPT>>

Return JSON: {"scenes": [{"beat":"<one-line beat description>","prompt":"<image prompt>"}, ... 6 items]}"""


def generate_script(client: anthropic.Anthropic) -> str:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": SCRIPT_PROMPT}],
    )
    return msg.content[0].text.strip()


def split_into_scenes(client: anthropic.Anthropic, script: str) -> list[dict]:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": SCENES_PROMPT.replace("<<SCRIPT>>", script)}
        ],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("`").strip()
    data = json.loads(raw)
    scenes = data["scenes"]
    assert len(scenes) == 6, f"Expected 6 scenes, got {len(scenes)}"
    return scenes


def generate_image_flux(prompt: str, out_path: Path) -> None:
    """Generate a 9:16 image via Replicate Flux Schnell (~$0.003/image)."""
    full_prompt = (
        prompt
        + " Vertical 9:16 framing. Saturated cold palette - vibrant cobalt blue, "
        "fluorescent green, crimson red, pitch black. NOT pastel, NOT washed out. "
        "Cinematic 35mm, ultra-detailed. Pure visual, absolutely no text, no "
        "letters, no captions, no logos."
    )
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": full_prompt,
            "aspect_ratio": "9:16",
            "output_format": "png",
            "num_outputs": 1,
            "num_inference_steps": 4,
            "go_fast": True,
            "megapixels": "1",
        },
    )
    # Replicate SDK 1.x returns list[FileOutput]; FileOutput has .read() + .url
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=60).content)


def main() -> None:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing")
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN missing")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = REPO_ROOT / "poc" / "cosmic" / ts
    images_dir = work_dir / "images"
    scenes_dir = work_dir / "scene_clips"
    images_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    log(f"Work dir: {work_dir}")
    log("Backend: Flux Schnell via Replicate (~$0.003/image)")

    anth = anthropic.Anthropic()
    oai = OpenAI()

    # 1. Script
    log("[1/6] Generating cosmic horror script with Claude...")
    t0 = time.time()
    script = generate_script(anth)
    (work_dir / "script.txt").write_text(script)
    log(f"  done in {time.time()-t0:.1f}s — {len(script.split())} words")
    log(f"  hook: {script.splitlines()[0][:90]}...")

    # 2. TTS (reuse from hfy_poc — same onyx voice)
    log("[2/6] Generating narration (Replicate Kokoro am_onyx)...")
    t0 = time.time()
    narration_path = work_dir / "narration.mp3"
    generate_tts(oai, script, narration_path)
    log(f"  done in {time.time()-t0:.1f}s — {narration_path.stat().st_size/1024:.0f} KB")

    # 3. Whisper
    log("[3/6] Transcribing narration...")
    t0 = time.time()
    words, total_duration = transcribe_words(narration_path)
    (work_dir / "transcript.json").write_text(
        json.dumps([w.model_dump() for w in words], indent=2)
    )
    log(
        f"  done in {time.time()-t0:.1f}s — {len(words)} words, "
        f"duration {total_duration:.2f}s"
    )

    # 4. Scenes
    log("[4/6] Splitting into 6 scene prompts with Claude...")
    t0 = time.time()
    scenes = split_into_scenes(anth, script)
    (work_dir / "scenes.json").write_text(json.dumps(scenes, indent=2))
    log(f"  done in {time.time()-t0:.1f}s")
    for i, s in enumerate(scenes, 1):
        log(f"  scene {i}: {s['beat'][:80]}")

    # 5. Images via Flux Schnell
    log("[5/6] Generating 6 images via Flux Schnell (Replicate)...")
    t0 = time.time()
    image_paths: list[Path] = []
    for i, scene in enumerate(scenes, 1):
        img_path = images_dir / f"scene_{i:02d}.png"
        log(f"  [{i}/6] flux-schnell...")
        ti = time.time()
        generate_image_flux(scene["prompt"], img_path)
        log(
            f"    done in {time.time()-ti:.1f}s — "
            f"{img_path.stat().st_size/1024:.0f} KB"
        )
        image_paths.append(img_path)
    log(f"  all 6 images in {time.time()-t0:.1f}s")

    # 6. Compose (reuse from hfy_poc) — locked at exactly 60s
    TARGET_DURATION = 60.0
    log("[6/6] Composing final video...")
    t0 = time.time()
    scene_duration = TARGET_DURATION / 6  # 10s each, video lands at exact 60s
    if total_duration > TARGET_DURATION:
        log(
            f"  WARNING: narration {total_duration:.1f}s > target "
            f"{TARGET_DURATION}s — final mux will trim audio"
        )
    else:
        log(
            f"  narration {total_duration:.1f}s + "
            f"{TARGET_DURATION - total_duration:.1f}s silent end-card = "
            f"{TARGET_DURATION:.1f}s"
        )
    log(f"  scene duration: {scene_duration:.2f}s each ({TARGET_DURATION:.0f}s total)")

    scene_clip_paths: list[Path] = []
    for i, img in enumerate(image_paths):
        clip = scenes_dir / f"scene_{i+1:02d}.mp4"
        build_scene_clip(img, scene_duration, clip, i)
        scene_clip_paths.append(clip)
    log(f"  6 scene clips rendered in {time.time()-t0:.1f}s")

    ass_path = work_dir / "captions.ass"
    ass_path.write_text(
        build_ass_for_words(words, output_size=(1080, 1920), offset=0.0),
        encoding="utf-8",
    )

    final = work_dir / "final.mp4"
    compose_final(
        scene_clip_paths,
        narration_path,
        ass_path,
        final,
        work_dir,
        target_duration=TARGET_DURATION,
    )
    log(f"  final compose: {time.time()-t0:.1f}s")

    log(f"DONE → {final}")
    log(f"  size: {final.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
