from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from youcut.uploader import display_clip_list, resolve_clip_selection, upload_clips


@pytest.fixture()
def clip_files(tmp_path: Path) -> list[tuple[Path, Path]]:
    clips: list[tuple[Path, Path]] = []
    for index in range(1, 4):
        video_path = tmp_path / f"clip_{index:02d}.mp4"
        txt_path = tmp_path / f"clip_{index:02d}.txt"
        video_path.write_bytes(b"video")
        txt_path.write_text(
            f"TÍTULO\nClip {index}\n\nDESCRIÇÃO\nDescricao {index}\n\nHASHTAGS\n#tag{index}\n",
            encoding="utf-8",
        )
        clips.append((video_path, txt_path))
    return clips


class TestResolveClipSelection:
    def test_none_selects_all(self, clip_files):
        selected, skipped = resolve_clip_selection(clip_files, None)

        assert [clip_index for clip_index, _, _ in selected] == [1, 2, 3]
        assert skipped == []

    def test_valid_subset_marks_remaining_as_skipped(self, clip_files):
        selected, skipped = resolve_clip_selection(clip_files, [1, 3])

        assert [clip_index for clip_index, _, _ in selected] == [1, 3]
        assert [clip_index for clip_index, _, _ in skipped] == [2]

    def test_invalid_index_raises_descriptive_error(self, clip_files):
        with pytest.raises(ValueError, match=r"Invalid clip indices.*4.*Valid range is 1-3"):
            resolve_clip_selection(clip_files, [4])

    def test_duplicates_are_deduplicated_without_error(self, clip_files):
        selected, skipped = resolve_clip_selection(clip_files, [3, 1, 3, 1])

        assert [clip_index for clip_index, _, _ in selected] == [1, 3]
        assert [clip_index for clip_index, _, _ in skipped] == [2]

    def test_out_of_order_indices_are_sorted(self, clip_files):
        selected, skipped = resolve_clip_selection(clip_files, [3, 1])

        assert [clip_index for clip_index, _, _ in selected] == [1, 3]
        assert [clip_index for clip_index, _, _ in skipped] == [2]


class TestDisplayClipList:
    def test_prints_index_title_and_status(self, clip_files):
        selected, skipped = resolve_clip_selection(clip_files, [2])
        console = Console(record=True, width=120)

        display_clip_list(selected, skipped, console=console)

        rendered = console.export_text()
        assert "Clip Upload Selection" in rendered
        assert "1" in rendered and "[ignorar]" in rendered
        assert "2" in rendered and "[upload]" in rendered
        assert "Clip 2" in rendered


class TestUploadClipsIntegration:
    def test_continues_after_platform_failure_and_records_skipped(self, clip_files, monkeypatch):
        upload_calls: list[tuple[str, int]] = []

        class FakeUploader:
            def __init__(self, token_dir: Path, platform: str) -> None:
                self.platform = platform
                self.token_dir = token_dir

            def authenticate(self) -> None:
                return None

            def upload(self, video_path: Path, metadata, clip_index: int = 0):
                upload_calls.append((self.platform, clip_index))
                if self.platform == "instagram" and clip_index == 1:
                    raise RuntimeError("instagram exploded")
                from youcut.uploader.base import UploadResult

                return UploadResult(
                    platform=self.platform,
                    clip_index=clip_index,
                    status="success",
                    url=f"https://example.com/{self.platform}/{clip_index}",
                )

        monkeypatch.setattr(
            "youcut.uploader.YouTubeUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "youtube"),
        )
        monkeypatch.setattr(
            "youcut.uploader.InstagramUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "instagram"),
        )
        monkeypatch.setattr(
            "youcut.uploader.TikTokUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "tiktok"),
        )

        results = upload_clips(
            clips=clip_files,
            platforms=["youtube", "instagram"],
            token_dir=clip_files[0][0].parent / "credentials",
            clips_filter=[1, 3],
        )

        assert upload_calls == [("youtube", 1), ("instagram", 1), ("youtube", 3), ("instagram", 3)]
        assert [(result.platform, result.clip_index, result.status) for result in results] == [
            ("youtube", 1, "success"),
            ("instagram", 1, "failed"),
            ("youtube", 3, "success"),
            ("instagram", 3, "success"),
            ("youtube", 2, "skipped"),
            ("instagram", 2, "skipped"),
        ]
        failed = next(result for result in results if result.platform == "instagram" and result.clip_index == 1)
        assert failed.error == "instagram exploded"

        report_path = clip_files[0][0].parent / "upload_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(payload["results"]) == 6

    def test_auth_failure_is_reused_for_all_selected_clips(self, clip_files, monkeypatch):
        auth_calls: list[str] = []
        upload_calls: list[tuple[str, int]] = []

        class FakeUploader:
            def __init__(self, token_dir: Path, platform: str) -> None:
                self.platform = platform
                self.token_dir = token_dir

            def authenticate(self) -> None:
                auth_calls.append(self.platform)
                if self.platform == "youtube":
                    raise RuntimeError("missing youtube auth")

            def upload(self, video_path: Path, metadata, clip_index: int = 0):
                upload_calls.append((self.platform, clip_index))
                from youcut.uploader.base import UploadResult

                return UploadResult(
                    platform=self.platform,
                    clip_index=clip_index,
                    status="success",
                    url=f"https://example.com/{self.platform}/{clip_index}",
                )

        monkeypatch.setattr(
            "youcut.uploader.YouTubeUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "youtube"),
        )
        monkeypatch.setattr(
            "youcut.uploader.InstagramUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "instagram"),
        )
        monkeypatch.setattr(
            "youcut.uploader.TikTokUploader",
            lambda token_dir, **kw: FakeUploader(token_dir, "tiktok"),
        )

        results = upload_clips(
            clips=clip_files,
            platforms=["youtube", "instagram"],
            token_dir=clip_files[0][0].parent / "credentials",
            clips_filter=[1, 3],
        )

        assert auth_calls == ["youtube", "instagram"]
        assert upload_calls == [("instagram", 1), ("instagram", 3)]
        assert [(result.platform, result.clip_index, result.status) for result in results] == [
            ("youtube", 1, "failed"),
            ("instagram", 1, "success"),
            ("youtube", 3, "failed"),
            ("instagram", 3, "success"),
            ("youtube", 2, "skipped"),
            ("instagram", 2, "skipped"),
        ]
        youtube_failures = [result for result in results if result.platform == "youtube" and result.status == "failed"]
        assert [result.error for result in youtube_failures] == [
            "missing youtube auth",
            "missing youtube auth",
        ]

    def test_youtube_respects_client_secrets_env_during_upload(self, clip_files, monkeypatch):
        received_client_secrets: list[Path | None] = []

        class FakeYouTubeUploader:
            def __init__(self, token_dir: Path, client_secrets_file: Path | None = None) -> None:
                self.token_dir = token_dir
                self.client_secrets_file = client_secrets_file
                received_client_secrets.append(client_secrets_file)

            def authenticate(self) -> None:
                return None

            def upload(self, video_path: Path, metadata, clip_index: int = 0):
                from youcut.uploader.base import UploadResult

                return UploadResult(
                    platform="youtube",
                    clip_index=clip_index,
                    status="success",
                    url=f"https://example.com/youtube/{clip_index}",
                )

        class FakeUploader:
            def __init__(self, token_dir: Path) -> None:
                self.token_dir = token_dir

            def authenticate(self) -> None:
                return None

            def upload(self, video_path: Path, metadata, clip_index: int = 0):
                from youcut.uploader.base import UploadResult

                return UploadResult(
                    platform="instagram",
                    clip_index=clip_index,
                    status="success",
                    url=f"https://example.com/instagram/{clip_index}",
                )

        secrets_file = clip_files[0][0].parent / "client_secrets.json"
        secrets_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(secrets_file))
        monkeypatch.setattr("youcut.uploader.YouTubeUploader", FakeYouTubeUploader)
        monkeypatch.setattr("youcut.uploader.InstagramUploader", FakeUploader)
        monkeypatch.setattr("youcut.uploader.TikTokUploader", FakeUploader)

        upload_clips(
            clips=clip_files,
            platforms=["youtube"],
            token_dir=clip_files[0][0].parent / "credentials",
            clips_filter=[1],
        )

        assert received_client_secrets == [secrets_file]

    def test_invalid_filter_fails_before_any_upload(self, clip_files, monkeypatch):
        invoked = False

        class FakeUploader:
            def __init__(self, token_dir: Path) -> None:
                self.token_dir = token_dir

            def authenticate(self) -> None:
                return None

            def upload(self, video_path: Path, metadata, clip_index: int = 0):
                nonlocal invoked
                invoked = True
                raise AssertionError("upload should not be called")

        monkeypatch.setattr("youcut.uploader.YouTubeUploader", FakeUploader)
        monkeypatch.setattr("youcut.uploader.InstagramUploader", FakeUploader)
        monkeypatch.setattr("youcut.uploader.TikTokUploader", FakeUploader)

        with pytest.raises(ValueError, match="Invalid clip indices"):
            upload_clips(
                clips=clip_files,
                platforms=["youtube"],
                token_dir=clip_files[0][0].parent / "credentials",
                clips_filter=[99],
            )

        assert invoked is False
