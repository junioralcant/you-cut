from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .base import UploadResult
from .instagram import InstagramUploader
from .metadata import parse_clip_metadata
from .report import generate_report
from .tiktok import TikTokUploader
from .youtube import YouTubeUploader

logger = logging.getLogger(__name__)

ClipEntry = tuple[Path, Path]
IndexedClipEntry = tuple[int, Path, Path]


def _build_uploaders(token_dir: Path) -> dict[str, object]:
    client_secrets_file = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE")
    youtube_uploader = (
        YouTubeUploader(token_dir, client_secrets_file=Path(client_secrets_file))
        if client_secrets_file
        else YouTubeUploader(token_dir)
    )
    return {
        "youtube": youtube_uploader,
        "instagram": InstagramUploader(token_dir),
        "tiktok": TikTokUploader(token_dir),
    }


def _authenticate_platforms(
    platforms: list[str],
    uploader_map: dict[str, object],
) -> dict[str, str]:
    failures: dict[str, str] = {}
    for platform in platforms:
        uploader = uploader_map[platform]
        try:
            uploader.authenticate()
        except Exception as exc:
            logger.error("Authentication failed for %s: %s", platform, exc)
            failures[platform] = str(exc)
    return failures


def resolve_clip_selection(
    clips: list[ClipEntry],
    clips_filter: list[int] | None,
) -> tuple[list[IndexedClipEntry], list[IndexedClipEntry]]:
    indexed_clips: list[IndexedClipEntry] = [
        (index, video_path, txt_path)
        for index, (video_path, txt_path) in enumerate(clips, start=1)
    ]
    if clips_filter is None:
        return indexed_clips, []

    normalized_filter = sorted(set(clips_filter))
    invalid = [clip_index for clip_index in normalized_filter if clip_index < 1 or clip_index > len(clips)]
    if invalid:
        invalid_str = ", ".join(str(clip_index) for clip_index in invalid)
        raise ValueError(
            f"Invalid clip indices in --clips: {invalid_str}. "
            f"Valid range is 1-{len(clips)}."
        )

    selected_indices = set(normalized_filter)
    selected = [clip for clip in indexed_clips if clip[0] in selected_indices]
    skipped = [clip for clip in indexed_clips if clip[0] not in selected_indices]
    return selected, skipped


def display_clip_list(
    selected: list[IndexedClipEntry],
    skipped: list[IndexedClipEntry],
    console: Console | None = None,
) -> Table:
    console = console or Console()
    selected_indices = {clip_index for clip_index, _, _ in selected}

    table = Table(title="Clip Upload Selection")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Status", justify="center")

    for clip_index, _, txt_path in sorted(selected + skipped, key=lambda clip: clip[0]):
        metadata = parse_clip_metadata(txt_path)
        status = "[upload]" if clip_index in selected_indices else "[ignorar]"
        title = metadata.title or txt_path.stem
        table.add_row(str(clip_index), title, Text(status))

    console.print(table)
    return table


def upload_clips(
    clips: list[ClipEntry],
    platforms: list[str],
    token_dir: Path,
    clips_filter: list[int] | None = None,
) -> list[UploadResult]:
    selected, skipped = resolve_clip_selection(clips, clips_filter)
    display_clip_list(selected, skipped)

    uploader_map = _build_uploaders(token_dir)
    unsupported_platforms = sorted(set(platforms) - set(uploader_map))
    if unsupported_platforms:
        raise ValueError(f"Unsupported platforms: {', '.join(unsupported_platforms)}")

    results: list[UploadResult] = []
    auth_failures = _authenticate_platforms(platforms, uploader_map)

    for clip_index, video_path, txt_path in selected:
        metadata = parse_clip_metadata(txt_path)
        for platform in platforms:
            if platform in auth_failures:
                results.append(
                    UploadResult(
                        platform=platform,
                        clip_index=clip_index,
                        status="failed",
                        error=auth_failures[platform],
                    )
                )
                continue

            uploader = uploader_map[platform]
            try:
                result = uploader.upload(video_path, metadata, clip_index=clip_index)
            except Exception as exc:
                logger.exception(
                    "Upload failed unexpectedly for clip %s on %s",
                    clip_index,
                    platform,
                )
                result = UploadResult(
                    platform=platform,
                    clip_index=clip_index,
                    status="failed",
                    error=str(exc),
                )
            results.append(result)

    for clip_index, _, _ in skipped:
        for platform in platforms:
            results.append(
                UploadResult(
                    platform=platform,
                    clip_index=clip_index,
                    status="skipped",
                )
            )

    output_dir = clips[0][0].parent if clips else Path(".")
    generate_report(results, output_dir=output_dir)
    return results
