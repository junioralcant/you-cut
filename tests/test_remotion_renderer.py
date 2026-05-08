"""Testes do RemotionRenderer (Task 7.0)."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.comic.providers.remotion_renderer import (
    NodeNotFoundError,
    RemotionDepsInstallError,
    RemotionRenderError,
    RemotionRenderer,
    _parse_node_major,
)
from youcut.models import RemotionInputProps


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTION_PROJECT = REPO_ROOT / "youcut" / "comic" / "remotion_project"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "remotion"


# ── helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def fake_project_dir(tmp_path: Path) -> Path:
    """Diretório minimalista com apenas o `render.mjs` esperado pelo construtor."""
    proj = tmp_path / "remotion_project"
    proj.mkdir()
    (proj / "render.mjs").write_text("// fake")
    return proj


def _props(audio: str = "/tmp/a.aac") -> RemotionInputProps:
    return RemotionInputProps(audio_path=audio, duration_sec=2.0)


class _FakePopen:
    """Substituto de `subprocess.Popen` para teste."""

    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdout = io.StringIO("".join(stdout_lines or []))
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.killed = False
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True


# ── _parse_node_major ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("v20.10.0\n", 20),
        ("v18.17.1\n", 18),
        ("v22.0.0", 22),
        ("invalid", None),
        ("", None),
    ],
)
def test_parse_node_major(stdout, expected):
    assert _parse_node_major(stdout) == expected


# ── _ensure_node ─────────────────────────────────────────────────────────


def test_ensure_node_raises_when_node_absent(fake_project_dir, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    monkeypatch.setattr(shutil, "which", lambda b: None)
    with pytest.raises(NodeNotFoundError, match="PATH"):
        renderer._ensure_node()


def test_ensure_node_raises_when_version_old(fake_project_dir, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    monkeypatch.setattr(shutil, "which", lambda b: "/fake/node")
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="v18.17.1\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_result)
    with pytest.raises(NodeNotFoundError, match="≥ 20"):
        renderer._ensure_node()


def test_ensure_node_passes_when_version_ok(fake_project_dir, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    monkeypatch.setattr(shutil, "which", lambda b: "/fake/node")
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="v20.10.0\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_result)
    renderer._ensure_node()  # não levanta


def test_constructor_rejects_dir_without_render_mjs(tmp_path):
    with pytest.raises(FileNotFoundError, match="render.mjs ausente"):
        RemotionRenderer(tmp_path)


# ── _ensure_deps ─────────────────────────────────────────────────────────


def test_ensure_deps_skips_when_node_modules_present(fake_project_dir, monkeypatch):
    (fake_project_dir / "node_modules").mkdir()
    renderer = RemotionRenderer(fake_project_dir)

    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a) or None)
    renderer._ensure_deps()
    assert called == []


def test_ensure_deps_runs_npm_install_when_missing(fake_project_dir, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    monkeypatch.setattr(shutil, "which", lambda b: "/fake/npm")
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr=""
    )
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(subprocess, "run", fake_run)
    renderer._ensure_deps()
    assert captured["args"][0][:2] == ["npm", "install"]


def test_ensure_deps_raises_on_install_failure(fake_project_dir, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    monkeypatch.setattr(shutil, "which", lambda b: "/fake/npm")
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="boom", stderr="npm error"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_result)
    with pytest.raises(RemotionDepsInstallError, match="npm install"):
        renderer._ensure_deps()


# ── render() — mocked ────────────────────────────────────────────────────


def test_render_invokes_subprocess_with_correct_args(fake_project_dir, tmp_path, monkeypatch):
    """Verifica que `node render.mjs --props ... --out ...` é chamado."""
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True  # bypass install
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    out_path = tmp_path / "out.mp4"
    fake = _FakePopen(stdout_lines=['{"progress": 1}\n'])

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        # cria o arquivo para passar a checagem final
        out_path.write_bytes(b"fake mp4")
        return fake

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = renderer.render(_props(), out_path)
    assert result == out_path.resolve()
    cmd = captured["cmd"]
    assert cmd[0] == "node"
    assert cmd[1] == "render.mjs"
    assert "--props" in cmd
    assert "--out" in cmd
    props_idx = cmd.index("--props")
    out_idx = cmd.index("--out")
    assert Path(cmd[props_idx + 1]).exists()  # arquivo props.json escrito
    assert cmd[out_idx + 1] == str(out_path.resolve())


def test_render_calls_on_progress_for_each_progress_line(fake_project_dir, tmp_path, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    fake = _FakePopen(
        stdout_lines=[
            '{"stage": "bundle"}\n',
            '{"progress": 0.25}\n',
            '{"progress": 0.5}\n',
            '{"progress": 1.0}\n',
            '{"stage": "done"}\n',
        ]
    )
    out_path = tmp_path / "x.mp4"

    def fake_popen(*a, **k):
        out_path.write_bytes(b"x")
        return fake

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    progresses: list[float] = []
    renderer.render(_props(), out_path, on_progress=progresses.append)
    assert progresses == [0.25, 0.5, 1.0]


def test_render_raises_on_nonzero_returncode(fake_project_dir, tmp_path, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    fake = _FakePopen(
        stdout_lines=['{"error": "oops bundling failed"}\n'],
        stderr="webpack stack trace",
        returncode=1,
    )
    out_path = tmp_path / "x.mp4"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)

    with pytest.raises(RemotionRenderError) as exc_info:
        renderer.render(_props(), out_path)
    err = exc_info.value
    assert "oops bundling failed" in str(err)
    assert err.stderr == "webpack stack trace"


def test_render_raises_when_output_missing_after_zero_returncode(
    fake_project_dir, tmp_path, monkeypatch
):
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    fake = _FakePopen(stdout_lines=['{"progress": 1}\n'], returncode=0)
    out_path = tmp_path / "missing.mp4"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)

    with pytest.raises(RemotionRenderError, match="ausente"):
        renderer.render(_props(), out_path)


def test_render_writes_props_json_in_output_dir(fake_project_dir, tmp_path, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    fake = _FakePopen(stdout_lines=['{"progress": 1}\n'])
    out_path = tmp_path / "out.mp4"

    written_props_path = []

    def fake_popen(cmd, **kwargs):
        idx = cmd.index("--props")
        written_props_path.append(Path(cmd[idx + 1]))
        out_path.write_bytes(b"x")
        return fake

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    props = _props()
    renderer.render(props, out_path)
    pp = written_props_path[0]
    assert pp.exists()
    payload = json.loads(pp.read_text())
    assert payload["audio_path"] == "/tmp/a.aac"
    assert payload["duration_sec"] == 2.0


def test_render_ignores_malformed_stdout_lines(fake_project_dir, tmp_path, monkeypatch):
    renderer = RemotionRenderer(fake_project_dir)
    renderer._deps_checked = True
    monkeypatch.setattr(renderer, "_ensure_node", lambda: None)

    fake = _FakePopen(
        stdout_lines=[
            "non-json log\n",
            '{"progress": 0.3}\n',
            "more garbage\n",
            '{"progress": 1}\n',
        ]
    )
    out_path = tmp_path / "x.mp4"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (out_path.write_bytes(b"x"), fake)[1])

    progresses: list[float] = []
    renderer.render(_props(), out_path, on_progress=progresses.append)
    assert progresses == [0.3, 1.0]


# ── Integração — render real ─────────────────────────────────────────────


@pytest.mark.integration
def test_render_real_with_minimal_fixture(tmp_path):
    """Render real ponta-a-ponta usando o projeto vendored (exige Node)."""
    if shutil.which("node") is None:
        pytest.skip("node ausente no PATH")
    if not (REMOTION_PROJECT / "node_modules").exists():
        pytest.skip("node_modules ausente — rode `npm install` no projeto Remotion")
    audio = FIXTURES_DIR / "minimal_audio.aac"
    if not audio.exists():
        pytest.skip("fixture audio ausente")

    renderer = RemotionRenderer(REMOTION_PROJECT)
    output = tmp_path / "real.mp4"
    progresses: list[float] = []
    result = renderer.render(
        RemotionInputProps(audio_path=str(audio), duration_sec=2.0),
        output,
        on_progress=progresses.append,
    )
    assert result.exists()
    assert result.stat().st_size > 0
    assert progresses, "esperava ≥ 1 chamada de on_progress"
    assert progresses[-1] >= 0.99
