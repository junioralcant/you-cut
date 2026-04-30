"""Image-to-video provider para o `youcut comic`.

Define o Protocol :class:`ImageToVideoProvider` e a implementação concreta
:class:`RunwayProvider` baseada no SDK ``runwayml`` com modelo
``gen4_turbo`` e ratio ``720:1280``.

Reaplica retry com backoff exponencial respeitando ``comic_i2v_retries``.
"""

from __future__ import annotations

import base64
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_RATIO: str = "720:1280"
DEFAULT_DURATION_SECONDS: float = 5.0
TERMINAL_TASK_STATES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
POLL_INTERVAL_SECONDS: float = 2.0
# Default generoso (1 hora) — em geral o Runway responde em <1min, mas dias com
# fila congestionada já levaram >4min. O usuário pode interromper com Ctrl+C.
# Ajustável via `PipelineConfig.comic_i2v_max_poll_seconds`.
MAX_POLL_TIME_SECONDS: float = 3600.0
POLL_PROGRESS_LOG_EVERY_SECONDS: float = 60.0


class I2VGenerationError(Exception):
    """Erro terminal de image-to-video após esgotar as tentativas."""


@runtime_checkable
class ImageToVideoProvider(Protocol):
    """Provider de image-to-video.

    ``prompt_image`` é a imagem-base que serve como start-frame.
    ``reference_images`` reforça identidade (até 3 referências).
    Retorna ``bytes`` MP4.
    """

    def image_to_video(
        self,
        prompt_image: Path,
        prompt_text: str,
        reference_images: list[Path],
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        ratio: str = DEFAULT_RATIO,
    ) -> bytes: ...


def _exp_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 8.0) -> float:
    return min(cap, base * (2**attempt))


_RUNWAY_DATA_URL_LIMIT_BYTES: int = 5_242_880  # cf. /v1/image_to_video — promptImage data URI


def _file_to_data_url(path: Path) -> str:
    """Converte um PNG/JPG local em data URL (`data:image/...;base64,...`).

    Quando o data URL excederia o limite de 5 MB do Runway, recomprime a
    imagem como JPEG (q=88) em memória antes de codificar — preserva a
    qualidade visual suficiente para o i2v sem extrapolar o limite duro
    da API.
    """

    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix or 'jpeg'}"

    encoded = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    if len(data_url) <= _RUNWAY_DATA_URL_LIMIT_BYTES:
        return data_url

    try:
        from io import BytesIO

        from PIL import Image  # type: ignore[import]
    except ImportError:
        return data_url

    with Image.open(BytesIO(raw)) as img:
        rgb = img.convert("RGB")
        for quality in (88, 80, 70, 60):
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            jpeg_bytes = buf.getvalue()
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            jpeg_url = f"data:image/jpeg;base64,{jpeg_b64}"
            if len(jpeg_url) <= _RUNWAY_DATA_URL_LIMIT_BYTES:
                logger.info(
                    "comic.i2v: PNG %s (%.1f MB) reencoded como JPEG q=%d (%.1f MB) para caber no limite Runway",
                    path.name,
                    len(raw) / 1_048_576,
                    quality,
                    len(jpeg_bytes) / 1_048_576,
                )
                return jpeg_url

    return data_url


class RunwayProvider:
    """`ImageToVideoProvider` para Runway Gen-4 Turbo via SDK ``runwayml``.

    Uso:
        ``RunwayProvider(api_key=..., max_retries=2)``

    Sem `api_key`, levanta erro em pt-BR. O `client` pode ser injetado para
    testes (qualquer objeto compatível com
    ``client.image_to_video.create(...)`` e ``client.tasks.retrieve(...)``).
    """

    DEFAULT_MODEL: str = "gen4_turbo"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        backoff_cap: float = 8.0,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        max_poll_time: float = MAX_POLL_TIME_SECONDS,
    ) -> None:
        if client is None and not (api_key and api_key.strip()):
            raise I2VGenerationError(
                "RUNWAY_API_KEY é obrigatório para o `youcut comic`. "
                "Defina a variável de ambiente ou configure runway_api_key."
            )
        self._api_key = api_key
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._poll_interval = poll_interval
        self._max_poll_time = max_poll_time

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from runwayml import RunwayML  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depende do extra `comic`
            raise I2VGenerationError(
                "SDK `runwayml` não instalado. Instale o extra `comic`: "
                "`pip install -e .[comic]`."
            ) from exc
        self._client = RunwayML(api_key=self._api_key)
        return self._client

    def image_to_video(
        self,
        prompt_image: Path,
        prompt_text: str,
        reference_images: list[Path],
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        ratio: str = DEFAULT_RATIO,
    ) -> bytes:
        prompt_image_path = Path(prompt_image)
        if not prompt_image_path.exists():
            raise I2VGenerationError(
                f"prompt_image não encontrada: {prompt_image_path}"
            )
        for ref in reference_images:
            if not Path(ref).exists():
                raise I2VGenerationError(f"reference_image não encontrada: {ref}")

        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._submit_and_wait(
                    client,
                    prompt_image=prompt_image_path,
                    prompt_text=prompt_text,
                    reference_images=[Path(r) for r in reference_images],
                    duration_seconds=duration_seconds,
                    ratio=ratio,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                delay = _exp_backoff_seconds(attempt, self._backoff_base, self._backoff_cap)
                logger.warning(
                    "comic.i2v: falha na tentativa %d (%s); aguardando %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise I2VGenerationError(
            f"Falha ao gerar mini-clipe após {self._max_retries + 1} tentativas: {last_exc}"
        ) from last_exc

    def _submit_and_wait(
        self,
        client: Any,
        *,
        prompt_image: Path,
        prompt_text: str,
        reference_images: list[Path],
        duration_seconds: float,
        ratio: str,
    ) -> bytes:
        prompt_image_url = _file_to_data_url(prompt_image)
        # SDK runwayml>=4 (Gen-4 Turbo) removeu o parâmetro `reference_images`.
        # A consistência de personagens vem da imagem-base (gpt-image-1 já edita
        # com âncoras via input_fidelity=high); aqui o Runway só anima o frame.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "prompt_image": prompt_image_url,
            "prompt_text": prompt_text,
            "duration": int(round(max(1.0, duration_seconds))),
            "ratio": ratio,
        }

        task = client.image_to_video.create(**kwargs)
        task_id = getattr(task, "id", None) or task["id"]

        result = self._poll_until_done(client, task_id)
        video_url = self._extract_video_url(result)
        return self._download_bytes(video_url)

    def _poll_until_done(self, client: Any, task_id: str) -> Any:
        start = time.monotonic()
        deadline = start + self._max_poll_time
        last_progress_log = start
        while True:
            task = client.tasks.retrieve(task_id)
            status = getattr(task, "status", None) or (
                task.get("status") if isinstance(task, dict) else None
            )
            if status in TERMINAL_TASK_STATES:
                if status != "SUCCEEDED":
                    failure = (
                        getattr(task, "failure_reason", None)
                        or (task.get("failure_reason") if isinstance(task, dict) else None)
                        or status
                    )
                    raise I2VGenerationError(f"Runway retornou status {status}: {failure}")
                return task
            now = time.monotonic()
            if now > deadline:
                raise I2VGenerationError(
                    f"Timeout aguardando Runway (>{self._max_poll_time:.0f}s) para task {task_id}"
                )
            if now - last_progress_log >= POLL_PROGRESS_LOG_EVERY_SECONDS:
                progress = getattr(task, "progress", None) or (
                    task.get("progress") if isinstance(task, dict) else None
                )
                logger.info(
                    "comic.i2v: aguardando task %s — %.0fs decorridos · status=%s · progress=%s",
                    task_id,
                    now - start,
                    status,
                    progress,
                )
                last_progress_log = now
            time.sleep(self._poll_interval)

    @staticmethod
    def _extract_video_url(task: Any) -> str:
        output = getattr(task, "output", None)
        if output is None and isinstance(task, dict):
            output = task.get("output")
        if not output:
            raise I2VGenerationError(f"Resposta sem `output` da Runway: {task!r}")
        if isinstance(output, list):
            first = output[0]
        else:
            first = output
        if isinstance(first, str):
            return first
        url = getattr(first, "url", None)
        if url is None and isinstance(first, dict):
            url = first.get("url")
        if not url:
            raise I2VGenerationError(f"Não foi possível extrair URL do vídeo: {task!r}")
        return str(url)

    @staticmethod
    def _download_bytes(url: str) -> bytes:
        if url.startswith("data:"):
            header, _, b64 = url.partition(",")
            if "base64" not in header:
                raise I2VGenerationError(f"data URL sem base64: {url[:60]}…")
            return base64.b64decode(b64)
        if url.startswith(("file://", "/")):
            path = Path(url.removeprefix("file://"))
            return path.read_bytes()
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - URL controlada
            return resp.read()
