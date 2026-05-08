"""Wrapper Python ↔ Node para o engine `youcut comic --engine remotion`.

`RemotionRenderer` invoca `node render.mjs` por subprocess, faz parse das
linhas JSON emitidas em stdout (progresso, estágio, erros) e levanta
exceções tipadas em pt-BR. Também expõe `open_studio()` para o modo de
preview interativo.

Health-check: confirma `node --version ≥ 20` antes de qualquer ação.
Auto-install: roda `npm install` quando `node_modules/` está ausente.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from youcut.models import RemotionInputProps

logger = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────────


class NodeNotFoundError(RuntimeError):
    """Node.js ≥ 20 não encontrado no PATH."""


class RemotionDepsInstallError(RuntimeError):
    """Falha ao instalar dependências npm do projeto Remotion vendored."""


class RemotionRenderError(RuntimeError):
    """O subprocess `node render.mjs` retornou código de erro."""

    def __init__(self, message: str, *, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


# ── Helpers ────────────────────────────────────────────────────────────────


_NODE_MIN_MAJOR = 20
_NODE_VERSION_RE = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def _parse_node_major(output: str) -> int | None:
    match = _NODE_VERSION_RE.search(output.strip())
    if not match:
        return None
    return int(match.group(1))


# ── Renderer ───────────────────────────────────────────────────────────────


class RemotionRenderer:
    """Wrapper de subprocess do projeto Remotion vendored."""

    def __init__(
        self,
        project_dir: Path | str,
        *,
        node_bin: str = "node",
        npm_bin: str = "npm",
        npx_bin: str = "npx",
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        if not (self.project_dir / "render.mjs").exists():
            raise FileNotFoundError(
                f"render.mjs ausente em {self.project_dir} — projeto Remotion não está vendored."
            )
        self.node_bin = node_bin
        self.npm_bin = npm_bin
        self.npx_bin = npx_bin
        self._deps_checked = False

    # ── health-checks ──────────────────────────────────────────────────────

    def _ensure_node(self) -> None:
        if shutil.which(self.node_bin) is None:
            raise NodeNotFoundError(
                "Node.js ≥ 20 não encontrado no PATH. Instale via "
                "`brew install node` (macOS) ou siga https://nodejs.org/ "
                "antes de usar `--engine remotion`."
            )
        try:
            result = subprocess.run(
                [self.node_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NodeNotFoundError(
                f"Falha ao executar `{self.node_bin} --version`: {exc}"
            ) from exc
        if result.returncode != 0:
            raise NodeNotFoundError(
                f"`{self.node_bin} --version` retornou rc={result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        major = _parse_node_major(result.stdout)
        if major is None:
            raise NodeNotFoundError(
                f"Não foi possível determinar a versão do Node a partir de: {result.stdout!r}"
            )
        if major < _NODE_MIN_MAJOR:
            raise NodeNotFoundError(
                f"Node.js {major} detectado; o engine remotion exige ≥ "
                f"{_NODE_MIN_MAJOR}. Atualize via "
                "`brew upgrade node` ou siga https://nodejs.org/."
            )
        logger.debug("comic.remotion_renderer: Node major=%s detectado", major)

    def _ensure_deps(self) -> None:
        if self._deps_checked:
            return
        node_modules = self.project_dir / "node_modules"
        if node_modules.exists():
            self._deps_checked = True
            return
        if shutil.which(self.npm_bin) is None:
            raise RemotionDepsInstallError(
                f"`{self.npm_bin}` não encontrado no PATH; impossível instalar deps Remotion."
            )
        logger.warning(
            "comic.remotion_renderer: node_modules ausente em %s — rodando `npm install` "
            "(pode levar alguns minutos na primeira execução)",
            self.project_dir,
        )
        result = subprocess.run(
            [self.npm_bin, "install", "--no-fund", "--no-audit"],
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RemotionDepsInstallError(
                "Falha em `npm install` para o projeto Remotion vendored.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        self._deps_checked = True

    # ── render ─────────────────────────────────────────────────────────────

    def render(
        self,
        props: RemotionInputProps,
        output_path: Path | str,
        *,
        on_progress: Callable[[float], None] | None = None,
        composition: str = "ComicVideo",
        timeout: float = 1800.0,
    ) -> Path:
        """Renderiza `props` para `output_path` via subprocess Node."""
        self._ensure_node()
        self._ensure_deps()

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        props_path = out.with_suffix(out.suffix + ".props.json")
        props_path.write_text(props.model_dump_json())

        cmd = [
            self.node_bin,
            "render.mjs",
            "--props",
            str(props_path),
            "--out",
            str(out),
            "--composition",
            composition,
        ]
        logger.info(
            "comic.remotion_renderer: invocando `%s` (cwd=%s)",
            " ".join(cmd),
            self.project_dir,
        )

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines: list[str] = []
        last_error: str | None = None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if "progress" in record and on_progress is not None:
                    try:
                        on_progress(float(record["progress"]))
                    except (TypeError, ValueError):
                        pass
                if "error" in record:
                    last_error = str(record["error"])
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RemotionRenderError(
                f"Render Remotion excedeu timeout de {timeout}s",
                stdout="".join(stdout_lines),
                stderr=(proc.stderr.read() if proc.stderr else ""),
            )

        stderr = proc.stderr.read() if proc.stderr else ""
        stdout = "".join(stdout_lines)
        if proc.returncode != 0:
            msg = (
                last_error
                or f"Subprocess `node render.mjs` retornou código {proc.returncode}"
            )
            raise RemotionRenderError(msg, stdout=stdout, stderr=stderr)
        if not out.exists() or out.stat().st_size == 0:
            raise RemotionRenderError(
                f"Render concluído mas arquivo de saída ausente/vazio: {out}",
                stdout=stdout,
                stderr=stderr,
            )
        return out

    # ── studio ─────────────────────────────────────────────────────────────

    def open_studio(
        self,
        props: RemotionInputProps,
        *,
        port: int = 3000,
        prompt: str = "Pressione ENTER para renderizar (ou Ctrl+C para abortar)…",
    ) -> None:
        """Abre o Remotion Studio em background e bloqueia até ENTER."""
        self._ensure_node()
        self._ensure_deps()

        props_path = self.project_dir / ".studio_props.json"
        props_path.write_text(props.model_dump_json())

        cmd = [
            self.npx_bin,
            "remotion",
            "studio",
            "--port",
            str(port),
            "--props",
            str(props_path),
        ]
        logger.info("comic.remotion_renderer: abrindo Studio (`%s`)", " ".join(cmd))
        studio_proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
        )
        try:
            input(prompt)
        except (EOFError, KeyboardInterrupt):
            logger.info("comic.remotion_renderer: Studio interrompido pelo usuário")
        finally:
            studio_proc.terminate()
            try:
                studio_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                studio_proc.kill()
            try:
                props_path.unlink(missing_ok=True)
            except OSError:
                pass
