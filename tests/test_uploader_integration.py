from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from youcut.uploader import upload_clips
from youcut.uploader.base import UploadResult


def _write_clip(tmp_path: Path, index: int) -> tuple[Path, Path]:
    video_path = tmp_path / f"clip_{index:02d}.mp4"
    txt_path = tmp_path / f"clip_{index:02d}.txt"
    video_path.write_bytes(b"")
    txt_path.write_text(
        (
            "TÍTULO\n"
            f"Clip {index}\n\n"
            "DESCRIÇÃO\n"
            f"Descricao {index}\n\n"
            "HASHTAGS\n"
            f"#tag{index}\n"
        ),
        encoding="utf-8",
    )
    return video_path, txt_path


class _RespxYouTubeUploader:
    def __init__(self, token_dir: Path, **kwargs) -> None:
        self.token_dir = token_dir

    def authenticate(self) -> None:
        pass

    def upload(self, video_path: Path, metadata, clip_index: int = 0) -> UploadResult:
        response = httpx.post(
            "https://youtube.test/upload",
            json={"clip_index": clip_index, "title": metadata.title},
        )
        response.raise_for_status()
        payload = response.json()
        return UploadResult(
            platform="youtube",
            clip_index=clip_index,
            status="success",
            url=payload["url"],
        )


class _RespxInstagramUploader:
    def __init__(self, token_dir: Path, **kwargs) -> None:
        self.token_dir = token_dir

    def authenticate(self) -> None:
        pass

    def upload(self, video_path: Path, metadata, clip_index: int = 0) -> UploadResult:
        response = httpx.post(
            "https://instagram.test/upload",
            json={"clip_index": clip_index, "caption": metadata.caption},
        )
        response.raise_for_status()
        payload = response.json()
        return UploadResult(
            platform="instagram",
            clip_index=clip_index,
            status="success",
            url=payload["url"],
        )


class TestUploadClipsIntegration:
    @respx.mock
    def test_full_flow_with_two_clips_and_two_platforms(self, tmp_path: Path, monkeypatch) -> None:
        clips = [_write_clip(tmp_path, 1), _write_clip(tmp_path, 2)]

        monkeypatch.setattr("youcut.uploader.YouTubeUploader", _RespxYouTubeUploader)
        monkeypatch.setattr("youcut.uploader.InstagramUploader", _RespxInstagramUploader)

        youtube_route = respx.post("https://youtube.test/upload").mock(
            side_effect=[
                httpx.Response(200, json={"url": "https://youtu.be/clip-1"}),
                httpx.Response(200, json={"url": "https://youtu.be/clip-2"}),
            ]
        )
        instagram_route = respx.post("https://instagram.test/upload").mock(
            side_effect=[
                httpx.Response(200, json={"url": "https://instagram.com/reel/clip-1"}),
                httpx.Response(200, json={"url": "https://instagram.com/reel/clip-2"}),
            ]
        )

        results = upload_clips(
            clips=clips,
            platforms=["youtube", "instagram"],
            token_dir=tmp_path / "credentials",
        )

        assert [(result.platform, result.clip_index, result.status) for result in results] == [
            ("youtube", 1, "success"),
            ("instagram", 1, "success"),
            ("youtube", 2, "success"),
            ("instagram", 2, "success"),
        ]
        assert youtube_route.call_count == 2
        assert instagram_route.call_count == 2

        report_path = tmp_path / "upload_report.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(payload["results"]) == 4

    @respx.mock
    def test_platform_failure_does_not_interrupt_other_uploads(self, tmp_path: Path, monkeypatch) -> None:
        clips = [_write_clip(tmp_path, 1), _write_clip(tmp_path, 2)]

        monkeypatch.setattr("youcut.uploader.YouTubeUploader", _RespxYouTubeUploader)
        monkeypatch.setattr("youcut.uploader.InstagramUploader", _RespxInstagramUploader)

        youtube_route = respx.post("https://youtube.test/upload").mock(
            side_effect=[
                httpx.Response(200, json={"url": "https://youtu.be/clip-1"}),
                httpx.Response(200, json={"url": "https://youtu.be/clip-2"}),
            ]
        )
        instagram_route = respx.post("https://instagram.test/upload").mock(
            side_effect=[
                httpx.Response(500, json={"error": "boom"}),
                httpx.Response(200, json={"url": "https://instagram.com/reel/clip-2"}),
            ]
        )

        results = upload_clips(
            clips=clips,
            platforms=["youtube", "instagram"],
            token_dir=tmp_path / "credentials",
        )

        assert [(result.platform, result.clip_index, result.status) for result in results] == [
            ("youtube", 1, "success"),
            ("instagram", 1, "failed"),
            ("youtube", 2, "success"),
            ("instagram", 2, "success"),
        ]
        failed = next(result for result in results if result.platform == "instagram" and result.clip_index == 1)
        assert "500" in (failed.error or "")
        assert youtube_route.call_count == 2
        assert instagram_route.call_count == 2

    @respx.mock
    def test_skipped_clips_are_reported_without_api_calls(self, tmp_path: Path, monkeypatch) -> None:
        clips = [_write_clip(tmp_path, 1), _write_clip(tmp_path, 2)]

        monkeypatch.setattr("youcut.uploader.YouTubeUploader", _RespxYouTubeUploader)
        monkeypatch.setattr("youcut.uploader.InstagramUploader", _RespxInstagramUploader)

        youtube_route = respx.post("https://youtube.test/upload").mock(
            return_value=httpx.Response(200, json={"url": "https://youtu.be/clip-1"})
        )
        instagram_route = respx.post("https://instagram.test/upload").mock(
            return_value=httpx.Response(200, json={"url": "https://instagram.com/reel/clip-1"})
        )

        results = upload_clips(
            clips=clips,
            platforms=["youtube", "instagram"],
            token_dir=tmp_path / "credentials",
            clips_filter=[1],
        )

        assert [(result.platform, result.clip_index, result.status) for result in results] == [
            ("youtube", 1, "success"),
            ("instagram", 1, "success"),
            ("youtube", 2, "skipped"),
            ("instagram", 2, "skipped"),
        ]
        assert youtube_route.call_count == 1
        assert instagram_route.call_count == 1
