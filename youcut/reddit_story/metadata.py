"""Gera title/alt-titles/description/hashtags/chapters/4-thumb-briefs prontos
pra upload no YouTube. Claude faz uma única chamada estruturada que devolve
o pack completo de metadata + briefs visuais pras 4 thumbs.

Saída: ``metadata.json`` em ``work_dir`` + ``metadata.txt`` human-readable.
Compliance: inclui disclaimer de dramatização + atribuição do post original
(condição mínima pra fair use + community guidelines pra monetização).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

from youcut.reddit_story.models import RedditStorySession, ScenePlan


@dataclass
class ThumbVariant:
    """1 briefing visual pra uma variante de thumbnail."""

    name: str  # slug curto, ex.: "documents", "property", "courtroom", "aftermath"
    scene_brief: str  # descrição visual pro Flux Schnell
    headline_line1: str  # UPPERCASE 2-3 palavras
    headline_line2: str  # UPPERCASE 2-3 palavras
    accent_banner: str  # UPPERCASE 1-2 palavras pro banner vermelho top-right


@dataclass
class MetadataPack:
    """Tudo gerado pelo Claude numa chamada só."""

    main_title: str
    alt_titles: list[str]
    logline: str
    thumb_variants: list[ThumbVariant]
    chapters: list[tuple[str, str]] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)


_PACK_PROMPT = """You are a YouTube content strategist for a long-form Reddit storytelling channel. You'll receive:
- A Reddit post header (subreddit, title, upvotes)
- The dramatized narration script

Produce a complete metadata pack as VALID JSON (no markdown fences, no preamble).

Schema:
{
  "main_title": "<= 70 chars, viral-clickbait following pattern '<Setup> — <Hook/Promise> | r/<subreddit>'. Must accurately reflect the story (no misleading). Use ALL CAPS sparingly for emphasis (1-2 key words max).",
  "alt_titles": [
    "3 alternative titles with different hook angles (curiosity gap, payoff tease, character-driven, etc.). Same length/style rules."
  ],
  "logline": "2-3 sentence description hook for the YouTube description, no spoilers, ends with a curiosity question or tease.",
  "thumb_variants": [
    {
      "name": "<short snake_case slug of the visual angle, e.g. 'documents', 'property', 'courtroom', 'aftermath'>",
      "scene_brief": "<1-2 sentence visual scene description for a Flux Schnell image gen, focused on a SINGLE evocative moment from the story. Must include strong negative space on the LEFT THIRD for text overlay. Cinematic dramatic reenactment style. NO TEXT in the image itself.>",
      "headline_line1": "<UPPERCASE 2-3 words, the SETUP of the hook>",
      "headline_line2": "<UPPERCASE 2-3 words, the TWIST or PROMISE — complementary to line1>",
      "accent_banner": "<UPPERCASE 1-2 words for top-right red banner, e.g. 'BIG MISTAKE', 'WAIT FOR IT', 'WORTH IT', 'KARMA HIT'>"
    }
    // 4 total — explore DIFFERENT visual angles of the same story
  ]
}

Hard rules:
- All text in English US, family-friendly enough for YouTube monetization (no profanity, no slurs).
- Titles and headlines must NOT spoil the payoff — tease the setup + curiosity gap.
- The 4 thumb_variants should depict DIFFERENT moments/angles (don't repeat the same visual).
- Each headline_line1 + headline_line2 should read as a complete punchy statement when stacked.
- Do NOT include "r/<sub>" in the headline_lines — that goes in the subreddit_tag at bottom (handled later).

Reddit thread:
Subreddit: r/<<SUBREDDIT>>
Title: <<TITLE>>
Upvotes: <<UPS>>

Narration script:
<<SCRIPT>>"""


def _chapter_from_beat(beat: str) -> str:
    if ":" in beat:
        _, body = beat.split(":", 1)
        beat = body.strip()
    if "—" in beat:
        beat = beat.split("—", 1)[0]
    beat = beat.strip().rstrip(".")
    if len(beat) > 60:
        beat = beat[:57].rstrip() + "…"
    return beat.title()


def build_chapters(
    scenes: list[ScenePlan], total_duration_s: float
) -> list[tuple[str, str]]:
    if not scenes:
        return []
    per = total_duration_s / len(scenes)
    out: list[tuple[str, str]] = []
    for i, sc in enumerate(scenes):
        start = i * per
        m = int(start // 60)
        sec = int(start % 60)
        out.append((f"{m:02d}:{sec:02d}", _chapter_from_beat(sc.beat)))
    return out


_DESCRIPTION_TEMPLATE = """{logline}

This story comes from r/{subreddit} and has racked up over {ups:,} upvotes for a reason.

⏱️ CHAPTERS
{chapters_block}

If you enjoyed this story, hit subscribe for more legendary internet stories. Drop a comment with what YOU would have done in OP's position!

📌 SOURCE
Original post by u/{author}:
{permalink}

⚠️ DISCLAIMER
Story is dramatized for entertainment. Names, locations, and identifying details have been altered for storytelling purposes and to protect the privacy of those involved. Shared under fair use for commentary and entertainment.

🏷️ TAGS
{hashtags}
"""


_DEFAULT_TAGS_BY_SUB = {
    "MaliciousCompliance": [
        "RedditStories", "MaliciousCompliance", "RedditRevenge", "ProRevenge",
        "StoryTime", "LongFormStories", "RedditReadings",
    ],
    "ProRevenge": [
        "RedditStories", "ProRevenge", "RedditRevenge", "RevengeStories",
        "StoryTime", "LongFormStories", "RedditReadings",
    ],
    "pettyrevenge": [
        "RedditStories", "PettyRevenge", "RedditRevenge", "RevengeStories",
        "StoryTime", "LongFormStories",
    ],
    "AmItheAsshole": [
        "RedditStories", "AITA", "AmITheAsshole", "RedditDrama",
        "StoryTime", "LongFormStories", "RedditReadings",
    ],
    "EntitledParents": [
        "RedditStories", "EntitledParents", "EntitledPeople", "RedditDrama",
        "StoryTime", "LongFormStories", "RedditReadings",
    ],
    "JustNoMIL": [
        "RedditStories", "JustNoMIL", "MotherInLaw", "RedditDrama",
        "StoryTime", "LongFormStories",
    ],
}


def generate_metadata_pack(
    client: anthropic.Anthropic,
    *,
    session: RedditStorySession,
    model: str,
) -> MetadataPack:
    """Claude monta title + alts + logline + 4 thumb briefs numa chamada só.

    Chapters/description/tags são montados localmente depois com base nos beats
    (que já existem em ``session.scenes``) e no logline retornado.
    """
    if session.script is None:
        raise ValueError("session.script ausente — chame format_script antes.")

    prompt = (
        _PACK_PROMPT.replace("<<SUBREDDIT>>", session.source.subreddit)
        .replace("<<TITLE>>", session.source.title)
        .replace("<<UPS>>", str(session.source.ups))
        .replace("<<SCRIPT>>", session.script)
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

    thumb_variants = [ThumbVariant(**v) for v in data["thumb_variants"]]
    if len(thumb_variants) != 4:
        raise ValueError(
            f"Esperado 4 thumb_variants do Claude, recebido {len(thumb_variants)}."
        )

    chapters = build_chapters(
        session.scenes, session.narration_duration_s or 0
    )
    tags = list(
        _DEFAULT_TAGS_BY_SUB.get(
            session.source.subreddit, ["RedditStories", "StoryTime"]
        )
    )
    hashtags = " ".join(f"#{t}" for t in tags)
    chapters_block = "\n".join(f"{ts} {label}" for ts, label in chapters)
    description = _DESCRIPTION_TEMPLATE.format(
        logline=data["logline"],
        subreddit=session.source.subreddit,
        ups=session.source.ups,
        chapters_block=chapters_block,
        author=session.source.author,
        permalink=session.source.permalink,
        hashtags=hashtags,
    )

    return MetadataPack(
        main_title=data["main_title"],
        alt_titles=data["alt_titles"],
        logline=data["logline"],
        thumb_variants=thumb_variants,
        chapters=chapters,
        description=description,
        tags=tags,
    )


def save_metadata(meta: MetadataPack, work_dir: Path) -> Path:
    out_json = work_dir / "metadata.json"
    out_json.write_text(
        json.dumps(
            {
                "main_title": meta.main_title,
                "alt_titles": meta.alt_titles,
                "logline": meta.logline,
                "description": meta.description,
                "tags": meta.tags,
                "chapters": [
                    {"timestamp": ts, "label": l} for ts, l in meta.chapters
                ],
                "thumb_variants": [asdict(v) for v in meta.thumb_variants],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_txt = work_dir / "metadata.txt"
    alt_block = "\n".join(f"  - {t}" for t in meta.alt_titles)
    thumb_block = "\n".join(
        f"  · {v.name}: {v.headline_line1} / {v.headline_line2} · banner: {v.accent_banner}"
        for v in meta.thumb_variants
    )
    out_txt.write_text(
        f"TITLE:\n{meta.main_title}\n\n"
        f"ALT TITLES:\n{alt_block}\n\n"
        f"DESCRIPTION:\n{meta.description}\n"
        f"THUMB VARIANTS:\n{thumb_block}\n",
        encoding="utf-8",
    )
    return out_json
