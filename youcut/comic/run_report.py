"""`run_report.json` — métricas auditáveis por execução do `youcut comic`.

Schema versionado para que consumidores possam evoluir sem quebrar
compatibilidade. Versão atual: ``schema_version=1``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from youcut.models import MotionComicSession, PanelRenderResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])
    idx = (len(sorted_values) - 1) * (q / 100)
    lo, hi = int(idx), min(len(sorted_values) - 1, int(idx) + 1)
    frac = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def build_run_report(session: MotionComicSession) -> dict[str, Any]:
    """Constrói o payload do ``run_report.json`` a partir de um `MotionComicSession`."""

    results: list[PanelRenderResult] = list(session.panel_results)
    n_panels = len(results)
    total_seconds = round(sum(r.clip_seconds for r in results), 4)
    n_static_fallbacks = sum(1 for r in results if r.was_static_fallback)
    durations = [r.clip_seconds for r in results]

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session.session_id,
        "video_path": str(session.video_path),
        "output_path": str(session.output_path) if session.output_path else None,
        "total_cost_usd": round(session.total_cost_usd, 4),
        "n_panels": n_panels,
        "n_static_fallbacks": n_static_fallbacks,
        "total_seconds": total_seconds,
        "panel_clip_p50_seconds": _percentile(durations, 50),
        "panel_clip_p95_seconds": _percentile(durations, 95),
        "panels": [
            {
                "panel_index": r.panel_index,
                "clip_seconds": r.clip_seconds,
                "was_static_fallback": r.was_static_fallback,
                "image_attempts": r.image_attempts,
                "i2v_attempts": r.i2v_attempts,
                "cost_usd": r.cost_usd,
            }
            for r in sorted(results, key=lambda x: x.panel_index)
        ],
    }


def write_run_report(session: MotionComicSession, output_dir: Path) -> Path:
    """Grava ``output_dir/comic/run_report.json`` e retorna o path."""

    report_dir = Path(output_dir) / "comic"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = build_run_report(session)
    report_path = report_dir / "run_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "comic.run_report: %s (cost=$%.2f, panels=%d, fallbacks=%d)",
        report_path,
        payload["total_cost_usd"],
        payload["n_panels"],
        payload["n_static_fallbacks"],
    )
    return report_path
