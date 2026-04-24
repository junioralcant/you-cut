from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .base import UploadReport, UploadResult

_STATUS_SYMBOLS = {
    "success": "✓",
    "failed": "✗",
    "skipped": "–",
}


def _build_report_table(results: list[UploadResult]) -> Table:
    table = Table(title="Upload Report")
    table.add_column("Clip", justify="right", style="cyan", no_wrap=True)
    table.add_column("Platform", style="white")
    table.add_column("Status", justify="center")
    table.add_column("URL", overflow="fold")

    for result in results:
        table.add_row(
            str(result.clip_index),
            result.platform,
            _STATUS_SYMBOLS.get(result.status, result.status),
            result.url or "",
        )

    return table


def generate_report(
    results: list[UploadResult],
    output_dir: Path,
    console: Console | None = None,
) -> UploadReport:
    console = console or Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = _build_report_table(results)
    console.print(table)

    report = UploadReport(
        results=results,
        timestamp=datetime.now(UTC),
        output_dir=output_dir,
    )
    report_path = output_dir / "upload_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
