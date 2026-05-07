from __future__ import annotations

from pathlib import Path

from youcut.models import MusicTrack, ViralClip


def export_metadata(
    clip: ViralClip,
    index: int,
    output_dir: Path,
    music_track: MusicTrack | None = None,
) -> Path:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"clip_{index + 1:02d}.txt"
    output_path = output_dir / filename

    hashtags = " ".join(
        tag if tag.startswith("#") else f"#{tag}" for tag in clip.hashtags
    )

    content = (
        f"TÍTULO\n"
        f"{clip.title}\n"
        f"\n"
        f"DESCRIÇÃO\n"
        f"{clip.description}\n"
        f"\n"
        f"HASHTAGS\n"
        f"{hashtags}\n"
        f"\n"
        f"SUGESTÃO DE THUMBNAIL\n"
        f"{clip.thumbnail_idea}\n"
        f"\n"
        f"NOTA DE VIRALIDADE: {clip.viral_score:g}/10\n"
        f"\n"
        f"MOTIVO DA SELEÇÃO\n"
        f"{clip.reason}\n"
    )

    if music_track is not None:
        content += (
            f"\n"
            f"TRILHA SONORA\n"
            f"Nome: {music_track.name}\n"
            f"Fonte: {music_track.pixabay_url}\n"
        )

    output_path.write_text(content, encoding="utf-8")
    return output_path
