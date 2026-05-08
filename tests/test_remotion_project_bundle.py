"""Integration test do projeto Remotion vendored (Task 5.0).

Executa `node render.mjs` em uma fixture sintética e valida o MP4 gerado
com `ffprobe`. Pula em ambientes sem `node` ou `ffmpeg`/`ffprobe` no PATH
(CI sem Node ou builds Linux estritos).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REMOTION_PROJECT = Path(__file__).resolve().parent.parent / "youcut" / "comic" / "remotion_project"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "remotion"

pytestmark = pytest.mark.integration


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def _node_modules_present() -> bool:
    return (REMOTION_PROJECT / "node_modules").exists()


@pytest.fixture(scope="module")
def smoke_audio() -> Path:
    audio = FIXTURES_DIR / "minimal_audio.aac"
    if not audio.exists():
        pytest.skip(f"Áudio fixture ausente: {audio}")
    return audio


@pytest.fixture(scope="module")
def smoke_props(tmp_path_factory, smoke_audio) -> Path:
    """Cria props.json apontando o audio_path para um caminho absoluto."""
    tmp = tmp_path_factory.mktemp("remotion_smoke")
    props_path = tmp / "smoke_props.json"
    props_path.write_text(
        json.dumps(
            {
                "audio_path": str(smoke_audio),
                "duration_sec": 2.0,
                "fps": 30,
                "width": 1080,
                "height": 1920,
                "characters": {},
                "scenes": [],
                "background_color": "#101820",
            }
        )
    )
    return props_path


def test_render_mjs_produces_valid_mp4(smoke_props, tmp_path):
    """Smoke render: node render.mjs gera MP4 1080×1920 30fps com áudio."""
    if not _has("node"):
        pytest.skip("node não disponível no PATH")
    if not _has("ffprobe"):
        pytest.skip("ffprobe não disponível no PATH")
    if not _node_modules_present():
        pytest.skip(
            "node_modules ausente — rode `npm install` em "
            "youcut/comic/remotion_project/ antes do teste"
        )

    output = tmp_path / "smoke.mp4"
    completed = subprocess.run(
        [
            "node",
            str(REMOTION_PROJECT / "render.mjs"),
            "--props",
            str(smoke_props),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REMOTION_PROJECT),
    )

    assert completed.returncode == 0, (
        f"render.mjs falhou (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists()
    assert output.stat().st_size > 0

    # ffprobe — valida dimensões, fps, codec
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    out = probe.stdout
    assert "width=1080" in out
    assert "height=1920" in out
    assert "codec_name=h264" in out
    assert "r_frame_rate=30/1" in out


def test_render_emits_progress_json_lines(smoke_props, tmp_path):
    """stdout deve emitir ≥ 1 linha JSON com `progress`."""
    if not _has("node") or not _node_modules_present():
        pytest.skip("node ausente ou node_modules não instalado")
    output = tmp_path / "progress.mp4"
    completed = subprocess.run(
        [
            "node",
            str(REMOTION_PROJECT / "render.mjs"),
            "--props",
            str(smoke_props),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REMOTION_PROJECT),
    )
    assert completed.returncode == 0, completed.stderr

    progress_lines = []
    for line in completed.stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "progress" in record:
            progress_lines.append(record["progress"])

    assert progress_lines, (
        f"Nenhuma linha de progresso emitida em stdout:\n{completed.stdout}"
    )
    assert progress_lines[-1] == 1 or progress_lines[-1] >= 0.99


def test_render_fails_with_missing_args(tmp_path):
    """render.mjs deve falhar com mensagem clara quando faltam args."""
    if not _has("node") or not _node_modules_present():
        pytest.skip("node ausente ou node_modules não instalado")
    completed = subprocess.run(
        ["node", str(REMOTION_PROJECT / "render.mjs")],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(REMOTION_PROJECT),
    )
    assert completed.returncode != 0
    assert "props" in completed.stdout or "props" in completed.stderr


def test_render_fails_with_missing_props_file(tmp_path):
    if not _has("node") or not _node_modules_present():
        pytest.skip("node ausente ou node_modules não instalado")
    completed = subprocess.run(
        [
            "node",
            str(REMOTION_PROJECT / "render.mjs"),
            "--props",
            str(tmp_path / "nonexistent.json"),
            "--out",
            str(tmp_path / "out.mp4"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(REMOTION_PROJECT),
    )
    assert completed.returncode != 0
    assert "failed to read props" in completed.stdout
