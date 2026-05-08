#!/usr/bin/env python3
"""Validação batch do engine `youcut comic --engine remotion`.

Roda `run_remotion_pipeline` para cada vídeo passado em `--video` (pode
repetir) e coleta:
- duração do vídeo de input;
- wall-clock do pipeline (excluindo download/transcrição cacheada);
- razão `wall_clock / video_duration` (alvo ≤ 5× — RF performance);
- custo total reportado pelo cost_estimator (alvo ≤ $1 — RF-23/24);
- caminho dos MP4s de saída (com e sem legendas).

Emite um relatório em Markdown na saída (stdout ou arquivo via `--out`)
com a tabela por vídeo + médias + alertas. NÃO faz avaliação humana de
lip-sync — esta deve ser feita por ≥ 2 revisores externos sobre os MP4s
gerados.

Uso típico:

    python scripts/validate_remotion_engine.py \
        --video tests/fixtures/validation/clip_01.mp4 \
        --video tests/fixtures/validation/clip_02.mp4 \
        --out tasks/prd-comic-engine-remotion/validation_report.md

Pré-requisitos: `node ≥ 20` no PATH, `OPENAI_API_KEY` definida, `ffmpeg`.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_remotion")


def _ffprobe_duration(video: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe falhou: {completed.stderr}")
    return float(completed.stdout.strip())


def _format_seconds(s: float) -> str:
    minutes = int(s // 60)
    seconds = s - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def _run_one(video: Path) -> dict:
    """Roda o pipeline e devolve métricas; falhas reportadas em `error`."""
    from youcut.comic.remotion_pipeline import run_remotion_pipeline
    from youcut.config import PipelineConfig

    record: dict = {
        "video": str(video),
        "duration_sec": None,
        "wall_clock_sec": None,
        "ratio": None,
        "cost_usd": None,
        "output_path": None,
        "no_subs_path": None,
        "error": None,
    }

    try:
        record["duration_sec"] = _ffprobe_duration(video)
        config = PipelineConfig(comic_animation_engine="remotion")
        t0 = time.perf_counter()
        session = run_remotion_pipeline(video, config, preview=False)
        elapsed = time.perf_counter() - t0
        record["wall_clock_sec"] = elapsed
        if record["duration_sec"]:
            record["ratio"] = elapsed / record["duration_sec"]
        record["cost_usd"] = session.total_cost_usd
        if session.output_path:
            record["output_path"] = str(session.output_path)
            no_subs = Path(session.output_path).parent / "motion_comic_no_subs.mp4"
            if no_subs.exists():
                record["no_subs_path"] = str(no_subs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("falha ao processar %s", video)
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def _emit_report(records: list[dict], out: TextIO) -> None:
    print("# Relatório de Validação — Engine Remotion", file=out)
    print(file=out)
    print(
        "Gerado por `scripts/validate_remotion_engine.py`. Avaliação humana "
        "de lip-sync (RF-09) deve ser feita separadamente sobre os MP4s "
        "listados abaixo.",
        file=out,
    )
    print(file=out)
    print("## Tabela de Métricas", file=out)
    print(file=out)
    print(
        "| # | Vídeo | Duração | Render (wall) | Razão | Custo (US$) | Output |",
        file=out,
    )
    print(
        "|---|-------|---------|---------------|-------|-------------|--------|",
        file=out,
    )
    for idx, rec in enumerate(records, start=1):
        if rec.get("error"):
            print(
                f"| {idx} | `{Path(rec['video']).name}` | — | — | — | — | "
                f"❌ {rec['error']} |",
                file=out,
            )
            continue
        duration = rec["duration_sec"] or 0.0
        wall = rec["wall_clock_sec"] or 0.0
        ratio = rec["ratio"] or 0.0
        cost = rec["cost_usd"] or 0.0
        out_path = rec.get("output_path") or "—"
        print(
            f"| {idx} | `{Path(rec['video']).name}` | {_format_seconds(duration)} | "
            f"{_format_seconds(wall)} | {ratio:.2f}× | {cost:.2f} | "
            f"`{Path(out_path).name if out_path != '—' else '—'}` |",
            file=out,
        )

    print(file=out)
    print("## Resumo", file=out)
    print(file=out)
    successes = [r for r in records if not r.get("error")]
    if not successes:
        print("Nenhuma execução bem-sucedida.", file=out)
        return
    ratios = [r["ratio"] for r in successes if r["ratio"] is not None]
    costs = [r["cost_usd"] for r in successes if r["cost_usd"] is not None]
    print(f"- Total processados: **{len(records)}**", file=out)
    print(f"- Sucessos: **{len(successes)}**", file=out)
    print(f"- Falhas: **{len(records) - len(successes)}**", file=out)
    if ratios:
        print(
            f"- Razão render/duração — média: **{statistics.mean(ratios):.2f}×**, "
            f"mediana: **{statistics.median(ratios):.2f}×**, "
            f"max: **{max(ratios):.2f}×** "
            f"(alvo ≤ 5×)",
            file=out,
        )
    if costs:
        print(
            f"- Custo — média: **US$ {statistics.mean(costs):.2f}**, "
            f"max: **US$ {max(costs):.2f}** "
            f"(alvo ≤ US$ 1)",
            file=out,
        )

    # Alerts
    print(file=out)
    print("## Alertas", file=out)
    print(file=out)
    alerts: list[str] = []
    for rec in successes:
        name = Path(rec["video"]).name
        if rec["ratio"] is not None and rec["ratio"] > 5.0:
            alerts.append(f"- `{name}`: razão {rec['ratio']:.2f}× excede 5× (alvo de performance)")
        if rec["cost_usd"] is not None and rec["cost_usd"] > 1.0:
            alerts.append(f"- `{name}`: custo US$ {rec['cost_usd']:.2f} excede US$ 1 (RF-23/24)")
    if not alerts:
        print("Nenhum alerta.", file=out)
    else:
        for a in alerts:
            print(a, file=out)

    print(file=out)
    print("## Avaliação Humana de Lip-Sync (RF-09)", file=out)
    print(file=out)
    print(
        "Pendente. Distribua os MP4s acima a ≥ 2 revisores e registre o "
        "veredito binário (correto/incorreto) por clipe. Meta: ≥ 90% aprovação.",
        file=out,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Caminho de vídeo (use múltiplas vezes para batch).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Caminho do relatório Markdown (default: stdout).",
    )
    parser.add_argument(
        "--json-dump",
        type=Path,
        default=None,
        help="Caminho opcional de saída em JSON com os registros brutos.",
    )
    args = parser.parse_args()

    records: list[dict] = []
    for path_str in args.video:
        video = Path(path_str)
        if not video.exists():
            logger.error("vídeo não encontrado: %s", video)
            records.append({"video": str(video), "error": "arquivo inexistente"})
            continue
        logger.info("processando %s", video)
        records.append(_run_one(video))

    if args.json_dump:
        args.json_dump.parent.mkdir(parents=True, exist_ok=True)
        args.json_dump.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            _emit_report(records, f)
        logger.info("relatório salvo em %s", args.out)
    else:
        _emit_report(records, sys.stdout)

    return 0 if all(not r.get("error") for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
