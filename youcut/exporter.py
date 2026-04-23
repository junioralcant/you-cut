from pathlib import Path

from youcut.models import ViralClip


def export_metadata(clip: ViralClip, index: int, output_dir: Path) -> Path:
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

    output_path.write_text(content, encoding="utf-8")
    return output_path
