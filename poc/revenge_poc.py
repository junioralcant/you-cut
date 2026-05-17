"""POC long-form: Reddit revenge story (r/MaliciousCompliance).

Stack: Claude (formatador + scenes) + Replicate Kokoro am_adam @ 1.05x +
Replicate Flux Schnell 16:9 + ffmpeg Ken Burns + ASS subs.

Run:
    .venv/bin/python poc/revenge_poc.py
"""

from __future__ import annotations

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
import httpx
import replicate

from poc.hfy_poc import (
    CLAUDE_MODEL,
    load_env,
    log,
    run,
    transcribe_words,
)
from youcut.models import WordTimestamp


SOURCE_RAW = REPO_ROOT / "poc" / "revenge" / "_source" / "hoa_thread_q426qi.txt"


FORMAT_PROMPT = """You are formatting a viral Reddit r/MaliciousCompliance story into a 22-25 minute narration script for a YouTube long-form video targeting an English-speaking audience.

Input is the raw Reddit post (markdown). Your job:

1. **Open with a tight 25-second HOOK** — punchy first sentence, tease the most satisfying moment, then say "Today's story comes from r/MaliciousCompliance, and it's one of the highest-voted of all time. Here's what happened." Then dive in.

2. **Clean the body**:
   - Remove all markdown (**bold**, [links], asterisks)
   - Remove TL;DR sections (the whole story replaces them)
   - Remove "Edit:" and "Update:" header lines but KEEP the content
   - Expand abbreviations on first use (HOA = "Homeowners Association", OP can stay if you keep it natural)
   - Convert internet-isms ("ngl", "tbh", "lmao") to plain English
   - Replace URLs/image links with "and I'll spare you the visuals" or similar
   - Keep first-person voice and the original tone (don't sanitize the bite)

3. **Pace it for narration**:
   - Break long paragraphs into shorter narrated sentences
   - Add brief connectors ("Now here's where it gets good", "But the HOA wasn't done", "What they didn't realize was...") at section boundaries to keep momentum
   - Lean into the satisfying moments — don't rush past the payoff

4. **Close with a 15-second OUTRO**:
   - One sentence reflection on the karma/lesson
   - Then: "What would YOU have done in OP's position? Drop a comment, and if you enjoyed this story, hit that subscribe button for more from the Reddit vault. Thanks for listening."

5. **Target length**: 4500-5200 words (will narrate in ~22-25 min at speed 1.05).

Output ONLY the final narration script. No preamble, no headers, no labels, no markdown. Pure prose meant to be read aloud.

Raw Reddit post:
<<RAW>>"""


SCENES_PROMPT = """You are a visual director for a Reddit MaliciousCompliance long-form YouTube video. Given the narration script below, produce 8 image prompts (one per scene beat) for the Flux Schnell model.

Hard rules:
- Output VALID JSON only, schema below, no markdown fences.
- Each prompt describes a STILL CINEMATIC frame for a 16:9 HORIZONTAL YouTube video.
- Visual style across all 8: cinematic dramatic reenactment, golden-hour or moody indoor lighting, SATURATED VIBRANT palette (not pastel, not washed out), shallow depth of field, 35mm film aesthetic. Think: Better Call Saul + courtroom drama + Succession.
- Each prompt depicts a moment from the story but does NOT show readable text, faces of named individuals (use angles that hide faces or shoot from behind), or copyrighted material.
- Scenes should cover the story arc: hook moment (1), setup (2), inciting incident (3), rising action (4-5), turning point (6), payoff/revenge moment (7), aftermath (8).
- Settings should match the story (suburban houses, HOA meeting rooms, property, documents, courtrooms etc.)
- NO TEXT, NO LOGOS, NO WORDS in the image.
- Each prompt ≤ 70 words.

Narration script:
<<SCRIPT>>

Return JSON: {"scenes": [{"beat":"<one-line beat>","prompt":"<image prompt>"}, ... 8 items]}"""


def claude_format_script(client: anthropic.Anthropic, raw: str) -> str:
    log("  Claude formatando script (input: %d palavras)..." % len(raw.split()))
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[
            {"role": "user", "content": FORMAT_PROMPT.replace("<<RAW>>", raw)}
        ],
    )
    return msg.content[0].text.strip()


def claude_split_scenes(client: anthropic.Anthropic, script: str) -> list[dict]:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
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
    assert len(scenes) == 8, f"Expected 8 scenes, got {len(scenes)}"
    return scenes


def generate_tts_long(text: str, out_path: Path) -> None:
    """Kokoro am_adam @ 1.05x. Kokoro auto-splits long input."""
    output = replicate.run(
        "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13",
        input={
            "text": text,
            "voice": "am_adam",
            "speed": 1.05,
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=300).content)


def generate_image_flux_16x9(prompt: str, out_path: Path) -> None:
    full_prompt = (
        prompt
        + " Horizontal 16:9 framing. Cinematic 35mm, vibrant saturated colors, "
        "dramatic lighting, ultra-detailed. Pure visual, absolutely no text, no "
        "letters, no captions, no logos."
    )
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": full_prompt,
            "aspect_ratio": "16:9",
            "output_format": "png",
            "num_outputs": 1,
            "num_inference_steps": 4,
            "go_fast": True,
            "megapixels": "1",
        },
    )
    item = output[0] if isinstance(output, list) else output
    if hasattr(item, "read"):
        out_path.write_bytes(item.read())
    else:
        out_path.write_bytes(httpx.get(str(item), timeout=120).content)


def build_scene_clip_16x9(
    image_path: Path,
    duration: float,
    out_path: Path,
    idx: int,
) -> None:
    """1920x1080 Ken Burns. Flux outputs ~1408x768; we scale to 2304x1296 (1.2x
    canvas) and crop 1920x1080 with sinusoidal pan + light zoom. Direction
    alternates by scene index for visual variety."""
    direction = 1 if idx % 2 == 0 else -1
    vf = (
        f"scale=2304:1296:flags=lanczos,"
        f"crop=1920:1080:"
        f"'(iw-1920)/2 + {direction}*120*sin(PI*t/{duration:.3f})':"
        f"'(ih-1080)/2 + {direction}*70*cos(PI*t/{duration:.3f})',"
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
            "20",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )


def build_ass_16x9(words: list[WordTimestamp]) -> str:
    """Word-by-word, bottom-third, 1920x1080. Bigger font + outline for legibility."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,72,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,"
        "-1,0,0,0,100,100,0,0,1,4,2,2,40,40,80,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )

    def ass_time(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h}:{m:02d}:{sec:05.2f}"

    events = []
    for w in words:
        text = w.word.strip().replace("\n", " ")
        if not text:
            continue
        events.append(
            f"Dialogue: 0,{ass_time(w.start)},{ass_time(w.end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def compose_long_form(
    scene_clips: list[Path],
    narration: Path,
    ass_path: Path,
    out_path: Path,
    work_dir: Path,
) -> None:
    concat_list = work_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in scene_clips) + "\n"
    )
    silent_video = work_dir / "video_silent.mp4"
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(silent_video),
        ]
    )
    with NamedTemporaryFile(suffix=".ass", delete=False) as tmp:
        safe_ass = Path(tmp.name)
    safe_ass.write_bytes(ass_path.read_bytes())
    run(
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
    safe_ass.unlink(missing_ok=True)


def main() -> None:
    load_env()
    for key in ("ANTHROPIC_API_KEY", "REPLICATE_API_TOKEN"):
        if not os.environ.get(key):
            sys.exit(f"{key} missing")

    if not SOURCE_RAW.exists():
        sys.exit(f"Source raw text missing: {SOURCE_RAW}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = REPO_ROOT / "poc" / "revenge" / ts
    images_dir = work_dir / "images"
    scenes_dir = work_dir / "scene_clips"
    images_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    log(f"Work dir: {work_dir}")
    log("Stack: Claude + Kokoro am_adam @ 1.05x + Flux Schnell 16:9")

    anth = anthropic.Anthropic()

    # 1. Format script with Claude
    log("[1/6] Claude formatando roteiro long-form...")
    t0 = time.time()
    raw = SOURCE_RAW.read_text()
    script = claude_format_script(anth, raw)
    (work_dir / "script.txt").write_text(script)
    log(f"  done in {time.time()-t0:.1f}s — {len(script.split())} palavras")
    log(f"  hook: {script.splitlines()[0][:90]}...")

    # 2. TTS Kokoro am_adam (long form, auto-split)
    log("[2/6] Kokoro am_adam @ 1.05x narrando ~{:.0f} min...".format(
        len(script.split()) / 165
    ))
    t0 = time.time()
    narration_path = work_dir / "narration.wav"
    generate_tts_long(script, narration_path)
    log(f"  done in {time.time()-t0:.1f}s — {narration_path.stat().st_size/1024/1024:.1f} MB")

    # 3. Whisper word timestamps (long form)
    log("[3/6] Whisper transcrevendo narração longa...")
    t0 = time.time()
    words, total_duration = transcribe_words(narration_path)
    (work_dir / "transcript.json").write_text(
        json.dumps([w.model_dump() for w in words], indent=2)
    )
    log(
        f"  done in {time.time()-t0:.1f}s — {len(words)} words, "
        f"duration {total_duration:.1f}s ({total_duration/60:.1f} min)"
    )

    # 4. Split into 8 visual beats
    log("[4/6] Claude dividindo em 8 beats visuais...")
    t0 = time.time()
    scenes = claude_split_scenes(anth, script)
    (work_dir / "scenes.json").write_text(json.dumps(scenes, indent=2))
    log(f"  done in {time.time()-t0:.1f}s")
    for i, s in enumerate(scenes, 1):
        log(f"  scene {i}: {s['beat'][:80]}")

    # 5. 8 Flux Schnell images 16:9
    log("[5/6] Gerando 8 imagens 16:9 via Flux Schnell...")
    t0 = time.time()
    image_paths: list[Path] = []
    for i, scene in enumerate(scenes, 1):
        img_path = images_dir / f"scene_{i:02d}.png"
        log(f"  [{i}/8] flux-schnell...")
        ti = time.time()
        generate_image_flux_16x9(scene["prompt"], img_path)
        log(f"    done in {time.time()-ti:.1f}s — {img_path.stat().st_size/1024:.0f} KB")
        image_paths.append(img_path)
    log(f"  all 8 images in {time.time()-t0:.1f}s")

    # 6. Compose: scene clips + ass + mux
    log("[6/6] Renderizando vídeo final 1920x1080...")
    t0 = time.time()
    scene_duration = total_duration / 8
    log(f"  scene duration: {scene_duration:.1f}s each (total {total_duration/60:.1f} min)")

    scene_clip_paths: list[Path] = []
    for i, img in enumerate(image_paths):
        clip = scenes_dir / f"scene_{i+1:02d}.mp4"
        build_scene_clip_16x9(img, scene_duration, clip, i)
        scene_clip_paths.append(clip)
    log(f"  8 scene clips renderizados em {time.time()-t0:.1f}s")

    ass_path = work_dir / "captions.ass"
    ass_path.write_text(build_ass_16x9(words), encoding="utf-8")
    log(f"  ASS escrito: {ass_path.name} ({len(words)} eventos)")

    final = work_dir / "final.mp4"
    compose_long_form(scene_clip_paths, narration_path, ass_path, final, work_dir)
    log(f"  total compose: {time.time()-t0:.1f}s")

    log(f"DONE → {final}")
    log(f"  size: {final.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
