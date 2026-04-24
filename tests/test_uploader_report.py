from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from youcut.uploader.base import UploadResult
from youcut.uploader.report import generate_report


def _results() -> list[UploadResult]:
    return [
        UploadResult(platform="youtube", clip_index=1, status="success", url="https://youtu.be/abc"),
        UploadResult(platform="instagram", clip_index=1, status="failed", error="bad request"),
        UploadResult(platform="tiktok", clip_index=2, status="skipped"),
    ]


class TestGenerateReport:
    def test_generates_table_with_mixed_statuses(self, tmp_path: Path):
        console = Console(record=True, width=120)

        report = generate_report(_results(), output_dir=tmp_path, console=console)

        rendered = console.export_text()
        assert "Upload Report" in rendered
        assert "youtube" in rendered
        assert "instagram" in rendered
        assert "tiktok" in rendered
        assert "✓" in rendered
        assert "✗" in rendered
        assert "–" in rendered
        assert report.output_dir == tmp_path

    def test_serializes_valid_json(self, tmp_path: Path):
        generate_report(_results(), output_dir=tmp_path, console=Console(record=True))

        report_path = tmp_path / "upload_report.json"
        payload = json.loads(report_path.read_text(encoding="utf-8"))

        assert isinstance(payload["results"], list)
        assert payload["results"][0]["status"] == "success"
        assert payload["results"][1]["status"] == "failed"
        assert payload["results"][2]["status"] == "skipped"

    def test_skipped_clips_appear_in_table(self, tmp_path: Path):
        console = Console(record=True, width=120)

        generate_report(_results(), output_dir=tmp_path, console=console)

        rendered = console.export_text()
        assert "tiktok" in rendered
        assert "2" in rendered
        assert "–" in rendered

    def test_saves_report_to_expected_output_directory(self, tmp_path: Path):
        output_dir = tmp_path / "output" / "video-title"

        generate_report(_results(), output_dir=output_dir, console=Console(record=True))

        assert (output_dir / "upload_report.json").exists()
