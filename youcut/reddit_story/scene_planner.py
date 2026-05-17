"""Claude divide o script em N visual beats com prompts pra Flux Schnell."""

from __future__ import annotations

import json

import anthropic

from youcut.reddit_story.models import ScenePlan


_SCENES_PROMPT = """You are a visual director for a Reddit long-form YouTube video. Given the narration script below, produce <<N>> image prompts (one per scene beat) for the Flux Schnell image model.

Hard rules:
- Output VALID JSON only, schema below, no markdown fences.
- Each prompt describes a STILL CINEMATIC frame for a 16:9 HORIZONTAL YouTube video.
- Visual style across all <<N>>: cinematic dramatic reenactment, golden-hour or moody indoor lighting, SATURATED VIBRANT palette (not pastel, not washed out), shallow depth of field, 35mm film aesthetic. Think: Better Call Saul + courtroom drama + Succession.
- Each prompt depicts a moment from the story but does NOT show readable text, faces of named individuals (use angles that hide faces or shoot from behind), or copyrighted material.
- Scenes should cover the story arc end-to-end (hook → setup → inciting → rising × N-4 → turning → payoff → aftermath). Distribute beats evenly across the narrative.
- Settings should match the story (homes, offices, courtrooms, documents, etc.)
- NO TEXT, NO LOGOS, NO WORDS in the image.
- Each prompt ≤ 70 words.

Narration script:
<<SCRIPT>>

Return JSON: {"scenes": [{"beat":"<one-line beat>","prompt":"<image prompt>"}, ... <<N>> items]}"""


def plan_scenes(
    client: anthropic.Anthropic,
    script: str,
    *,
    model: str,
    count: int,
) -> list[ScenePlan]:
    prompt = (
        _SCENES_PROMPT.replace("<<SCRIPT>>", script).replace("<<N>>", str(count))
    )
    msg = client.messages.create(
        model=model,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip("`").strip()
    data = json.loads(raw)
    scenes = [ScenePlan(**s) for s in data["scenes"]]
    if len(scenes) != count:
        raise ValueError(
            f"Esperado {count} cenas do Claude, recebido {len(scenes)}."
        )
    return scenes
