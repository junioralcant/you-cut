from pathlib import Path

from .base import ClipMetadata

_PLATFORM_LIMITS = {
    "youtube": {"title": 100, "description": 5000},
    "instagram": {"caption": 2200},
    "tiktok": {"caption": 2200},
}

_STOP_SECTIONS = frozenset({
    "SUGESTÃO DE THUMBNAIL",
    "NOTA DE VIRALIDADE:",
    "MOTIVO DA SELEÇÃO",
})


def _is_stop_section(line: str) -> bool:
    return line in _STOP_SECTIONS or line.startswith("NOTA DE VIRALIDADE")


def parse_clip_metadata(txt_path: Path) -> ClipMetadata:
    text = txt_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_key: str | None = None
    lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("TÍTULO", "DESCRIÇÃO", "HASHTAGS"):
            if current_key is not None:
                sections[current_key] = "\n".join(lines).strip()
            current_key = stripped
            lines = []
        elif current_key is not None:
            if _is_stop_section(stripped):
                sections[current_key] = "\n".join(lines).strip()
                current_key = None
                lines = []
            else:
                lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(lines).strip()

    title = sections.get("TÍTULO", "")
    description = sections.get("DESCRIÇÃO", "")
    hashtags_raw = sections.get("HASHTAGS", "")
    hashtags = [tag.strip() for tag in hashtags_raw.split() if tag.startswith("#")] if hashtags_raw else []

    caption = _build_caption(description, hashtags)

    return ClipMetadata(
        title=title,
        description=description,
        hashtags=hashtags,
        caption=caption,
    )


def _build_caption(description: str, hashtags: list[str]) -> str:
    hashtag_block = " ".join(hashtags)
    if hashtag_block:
        return f"{description}\n\n{hashtag_block}" if description else hashtag_block
    return description


def apply_platform_limits(metadata: ClipMetadata, platform: str) -> ClipMetadata:
    limits = _PLATFORM_LIMITS.get(platform, {})

    title = metadata.title
    description = metadata.description
    hashtags = metadata.hashtags

    if platform == "youtube":
        title_limit = limits["title"]
        desc_limit = limits["description"]
        if len(title) > title_limit:
            title = title[:title_limit]

        hashtag_block = " ".join(hashtags)
        hashtag_len = len(hashtag_block) + (2 if hashtag_block else 0)
        available = max(0, desc_limit - hashtag_len)
        if len(description) > available:
            description = description[:available]

        caption = _build_caption(description, hashtags)
        if len(caption) > desc_limit:
            caption = caption[:desc_limit]

    elif platform == "instagram":
        cap_limit = limits["caption"]
        hashtag_block = " ".join(hashtags)
        hashtag_len = len(hashtag_block) + (2 if hashtag_block else 0)
        available = max(0, cap_limit - hashtag_len)
        truncated_description = description[:available] if len(description) > available else description
        caption = _build_caption(truncated_description, hashtags)
        if len(caption) > cap_limit:
            caption = caption[:cap_limit]
        description = truncated_description

    elif platform == "tiktok":
        cap_limit = limits["caption"]
        # TikTok caption = title + description + hashtags (per tech spec)
        title_prefix = title + "\n\n" if title else ""
        hashtag_block = " ".join(hashtags)
        hashtag_len = len(hashtag_block) + (2 if hashtag_block else 0)
        title_len = len(title_prefix)
        available = max(0, cap_limit - title_len - hashtag_len)
        truncated_description = description[:available] if len(description) > available else description
        body = title_prefix + truncated_description
        caption = _build_caption(body, hashtags)
        if len(caption) > cap_limit:
            caption = caption[:cap_limit]
        description = truncated_description

    else:
        caption = _build_caption(description, hashtags)

    return ClipMetadata(
        title=title,
        description=description,
        hashtags=hashtags,
        caption=caption,
    )
