"""POC: 1-min HFY combat heroism story, 100% AI-generated.

Pipeline:
  Claude (script) -> OpenAI TTS (narration) -> faster-whisper (word ts) ->
  Claude (scene prompts) -> gpt-image-1 (6 images) ->
  ffmpeg (Ken Burns + ASS subs via youcut.captioner) -> final.mp4 (1080x1920)

Run from repo root:
  .venv/bin/python poc/hfy_poc.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import anthropic
from openai import OpenAI

from youcut.captioner import build_ass_for_words
from youcut.models import WordTimestamp


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------- 1. Script ----------

SCRIPT_PROMPT = """You are writing a SHORT first-person HFY (Humanity Fuck Yeah) narration for a 60-second YouTube Short. Combat heroism flavor.

Constraints:
- ~145-160 words of pure narration (no scene directions, no dialogue tags).
- Tight hook in the first sentence. "Sergeant Cole" is the human. Aliens vastly outnumber.
- Build: setup -> threat -> twist (Cole's improvised tactic) -> victory -> punchline.
- Tone: low, grave, Halo-meets-Helldivers. Short sentences. No purple prose.
- The last line must be a quotable kicker.
- Output ONLY the narration text, no preamble, no headers, no quotes."""


CLAUDE_MODEL = "claude-sonnet-4-6"


def generate_script(client: anthropic.Anthropic) -> str:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": SCRIPT_PROMPT}],
    )
    return msg.content[0].text.strip()


# ---------- 2. TTS ----------


def generate_tts(client: OpenAI, text: str, out_path: Path) -> None:
    """Replicate Kokoro-82M, voice am_onyx (deep male, mimics OpenAI onyx).

    ``client`` is kept in the signature for backward compat but unused.
    Costs ~$0.0003/sec audio vs $0.030/1k chars on OpenAI TTS-1-HD.
    """
    import replicate

    output = replicate.run(
        "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
        input={
            "text": text,
            "voice": "am_onyx",
            "speed": 1.0,
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        import httpx

        out_path.write_bytes(httpx.get(str(item), timeout=120).content)


# ---------- 3. Whisper word timestamps ----------


def transcribe_words(audio_path: Path) -> tuple[list[WordTimestamp], float]:
    from faster_whisper import WhisperModel

    model = WhisperModel("small.en", device="auto", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path), language="en", word_timestamps=True
    )
    words: list[WordTimestamp] = []
    end_time = 0.0
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(WordTimestamp(word=w.word, start=w.start, end=w.end))
            end_time = max(end_time, w.end)
    return words, end_time


# ---------- 4. Scene splitting ----------

SCENES_PROMPT = """You are a sci-fi storyboard artist. Given a 60-second HFY combat narration, produce 6 image prompts (one per scene beat) for the gpt-image-1 model.

Hard rules:
- Output VALID JSON only, schema below, no markdown fences.
- Each prompt must be self-contained and describe a STILL CINEMATIC frame for a 9:16 vertical Short.
- Visual style across all 6: vibrant saturated colors (NOT pastel), Halo/Helldivers/Starship Troopers vibe, volumetric light, lens flares, dust particles, cinematic 35mm, dramatic rim lighting. Palette favors orange and cyan with deep shadow black.
- Each prompt must include: hero (Sergeant Cole - human soldier in scratched exo-armor, helmet visor cracked, no face visible), enemy if any (insectoid xenos with chitinous plates and bioluminescent eyes), setting (ruined moon base, lunar surface, Earth on horizon).
- NO TEXT, NO LOGOS, NO WORDS in the image.
- Vary the camera: wide establishing -> medium tactical -> closeup of armor -> POV firing -> wide carnage -> hero silhouette victory.
- Each prompt ≤ 60 words.

Narration:
<<SCRIPT>>

Return JSON: {"scenes": [{"beat":"<one-line beat description>","prompt":"<image prompt>"}, ... 6 items]}"""


def split_into_scenes(client: anthropic.Anthropic, script: str) -> list[dict]:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": SCENES_PROMPT.replace("<<SCRIPT>>", script)}
        ],
    )
    raw = msg.content[0].text.strip()
    # Strip accidental code fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("`").strip()
    data = json.loads(raw)
    scenes = data["scenes"]
    assert len(scenes) == 6, f"Expected 6 scenes, got {len(scenes)}"
    return scenes


# ---------- 5. Image generation ----------


def generate_image(client: OpenAI, prompt: str, out_path: Path) -> None:
    full_prompt = (
        prompt
        + " Cinematic 35mm, ultra-detailed, vibrant saturated palette, dramatic rim "
        "lighting, volumetric god rays, dust particles. Vertical 9:16 framing. "
        "Pure visual, absolutely no text, no logos, no letters, no captions."
    )
    result = client.images.generate(
        model="gpt-image-1",
        prompt=full_prompt,
        size="1024x1536",
        n=1,
        quality="medium",
    )
    img_b64 = result.data[0].b64_json
    out_path.write_bytes(base64.b64decode(img_b64))


# ---------- 6. Compose ----------


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"ffmpeg failed: {' '.join(cmd[:4])}...")
        print(proc.stderr[-2000:])
        raise RuntimeError("ffmpeg failed")


def build_scene_clip(
    image_path: Path,
    duration: float,
    out_path: Path,
    idx: int,
) -> None:
    """Render a 1080x1920 Ken Burns clip from a single 1024x1536 image.

    Strategy: upscale to 1296x1944 (some headroom), then crop 1080x1920 with
    a slow sinusoidal pan + light zoom. Alternates pan direction per scene.
    """
    # Image is 1024x1536, AR 2:3 ≈ 0.667. We need 1080x1920 (9:16 = 0.5625).
    # Scale to height 2160 → width = 1440. Crop 1080x1920 from that with motion.
    direction = 1 if idx % 2 == 0 else -1
    # Light zoom: 1.0 → 1.06 over duration
    zoom_expr = f"1+0.06*t/{duration:.3f}"
    # Horizontal pan ±60px, vertical pan ±80px
    x_expr = f"(iw-iw/({zoom_expr})*1080/1080)/2 + {direction}*30*sin(2*PI*t/{2*duration:.3f})"
    # Simpler: scale + animated crop offset
    vf = (
        f"scale=1440:2160:flags=lanczos,"
        f"crop=1080:1920:"
        f"'(iw-1080)/2 + {direction}*60*sin(PI*t/{duration:.3f})':"
        f"'(ih-1920)/2 + {direction}*40*cos(PI*t/{duration:.3f})',"
        f"format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(image_path),
            "-vf",
            vf,
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )


def compose_final(
    scene_clips: list[Path],
    narration: Path,
    ass_path: Path,
    out_path: Path,
    work_dir: Path,
    target_duration: float | None = None,
) -> None:
    """When ``target_duration`` is set, pad audio with silence and clip output
    to that exact length; otherwise behave like before (``-shortest`` semantics).
    """
    # 1. Concat scene clips (silent)
    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in scene_clips) + "\n"
    )
    silent_video = work_dir / "video_silent.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(silent_video),
        ]
    )
    # 2. Burn ASS subs + mux narration. Copy ASS to a temp file (safer path
    # for the filter parser, same pattern as youcut.captioner.add_captions).
    with NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        safe_ass = Path(tmp.name)
    safe_ass.write_bytes(ass_path.read_bytes())
    fonts_dir = REPO_ROOT / "youcut" / "assets" / "fonts"
    ass_filter = f"ass={safe_ass}"
    if fonts_dir.exists():
        ass_filter += f":fontsdir={fonts_dir}"
    audio_filter_args = (
        ["-af", f"apad=whole_dur={target_duration:.3f}"]
        if target_duration is not None
        else []
    )
    duration_args = (
        ["-t", f"{target_duration:.3f}"]
        if target_duration is not None
        else ["-shortest"]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(narration),
            "-vf",
            ass_filter,
            *audio_filter_args,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            *duration_args,
            str(out_path),
        ]
    )
    safe_ass.unlink(missing_ok=True)


# ---------- Main ----------


def main() -> None:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = REPO_ROOT / "poc" / "hfy" / ts
    images_dir = work_dir / "images"
    scenes_dir = work_dir / "scene_clips"
    images_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    log(f"Work dir: {work_dir}")

    anth = anthropic.Anthropic()
    oai = OpenAI()

    # 1. Script
    log("[1/6] Generating script with Claude...")
    t0 = time.time()
    script = generate_script(anth)
    (work_dir / "script.txt").write_text(script)
    log(f"  done in {time.time()-t0:.1f}s — {len(script.split())} words")
    log(f"  hook: {script.splitlines()[0][:90]}...")

    # 2. TTS
    log("[2/6] Generating narration (OpenAI TTS onyx)...")
    t0 = time.time()
    narration_path = work_dir / "narration.mp3"
    generate_tts(oai, script, narration_path)
    log(f"  done in {time.time()-t0:.1f}s — {narration_path.stat().st_size/1024:.0f} KB")

    # 3. Whisper
    log("[3/6] Transcribing narration with faster-whisper small.en...")
    t0 = time.time()
    words, total_duration = transcribe_words(narration_path)
    (work_dir / "transcript.json").write_text(
        json.dumps([w.model_dump() for w in words], indent=2)
    )
    log(
        f"  done in {time.time()-t0:.1f}s — {len(words)} words, "
        f"total duration {total_duration:.2f}s"
    )

    # 4. Scenes
    log("[4/6] Splitting into 6 scene prompts with Claude...")
    t0 = time.time()
    scenes = split_into_scenes(anth, script)
    (work_dir / "scenes.json").write_text(json.dumps(scenes, indent=2))
    log(f"  done in {time.time()-t0:.1f}s")
    for i, s in enumerate(scenes, 1):
        log(f"  scene {i}: {s['beat'][:80]}")

    # 5. Images (sequential for clarity; could parallelize)
    log("[5/6] Generating 6 images with gpt-image-1...")
    t0 = time.time()
    image_paths: list[Path] = []
    for i, scene in enumerate(scenes, 1):
        img_path = images_dir / f"scene_{i:02d}.png"
        log(f"  [{i}/6] gpt-image-1...")
        ti = time.time()
        generate_image(oai, scene["prompt"], img_path)
        log(f"    done in {time.time()-ti:.1f}s — {img_path.stat().st_size/1024:.0f} KB")
        image_paths.append(img_path)
    log(f"  all 6 images in {time.time()-t0:.1f}s")

    # 6. Compose: Ken Burns per scene + mux + ASS subs
    log("[6/6] Composing final video...")
    t0 = time.time()
    scene_duration = total_duration / 6
    log(f"  scene duration: {scene_duration:.2f}s each ({total_duration:.2f}s total)")

    scene_clip_paths: list[Path] = []
    for i, img in enumerate(image_paths):
        clip = scenes_dir / f"scene_{i+1:02d}.mp4"
        build_scene_clip(img, scene_duration, clip, i)
        scene_clip_paths.append(clip)
    log(f"  6 scene clips rendered in {time.time()-t0:.1f}s")

    # ASS
    ass_path = work_dir / "captions.ass"
    ass_content = build_ass_for_words(words, output_size=(1080, 1920), offset=0.0)
    ass_path.write_text(ass_content, encoding="utf-8")
    log(f"  ASS written: {ass_path.name}")

    # Final mux
    final = work_dir / "final.mp4"
    compose_final(scene_clip_paths, narration_path, ass_path, final, work_dir)
    log(f"  final compose: {time.time()-t0:.1f}s")

    log(f"DONE → {final}")
    size_mb = final.stat().st_size / 1024 / 1024
    log(f"  size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
