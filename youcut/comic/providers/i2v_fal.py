"""Image-to-video provider via fal.ai (default: Kling 2.5 Turbo Pro).

fal.ai é um agregador que dá acesso a múltiplos modelos de i2v em uma única
API: Kling, Luma Ray, Hailuo (MiniMax), Pika, Wan, etc. Vantagens vs Runway:

- Pay-per-use sem cap diário rígido (Tier 1 Runway = 50/dia gen4_turbo).
- Concorrência maior (>1 task simultânea no plano padrão).
- Acesso a vários modelos via 1 chave; troca via ``comic_i2v_fal_model``.

Modelo default: ``fal-ai/kling-video/v2.5-turbo/pro/image-to-video`` —
boa relação qualidade × custo (~$0.10/s gerado). Outros bons:
``fal-ai/kling-video/v1.6/pro/image-to-video`` (~$0.05/s),
``fal-ai/luma-dream-machine/ray-2/image-to-video`` (~$0.07/s),
``fal-ai/minimax/hailuo-02/standard/image-to-video`` (~$0.04/s).
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from youcut.comic.providers.i2v import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_RATIO,
    I2VGenerationError,
    _file_to_data_url,
)

logger = logging.getLogger(__name__)

DEFAULT_FAL_MODEL: str = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"

# fal.ai usa labels textuais ("9:16") em vez de "WxH" (Runway). Mapeia os
# ratios mais comuns; se não estiver no dict, repassa o ratio bruto.
_RATIO_MAP: dict[str, str] = {
    "720:1280": "9:16",
    "1280:720": "16:9",
    "960:960": "1:1",
    "1104:832": "4:3",
    "832:1104": "3:4",
    "1584:672": "21:9",
}


def _exp_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 8.0) -> float:
    return min(cap, base * (2**attempt))


class FalImageToVideoProvider:
    """`ImageToVideoProvider` baseado em fal.ai (multi-modelo).

    Uso:
        ``FalImageToVideoProvider(api_key=..., model="fal-ai/kling-video/...")``

    O ``client`` pode ser injetado para testes (objeto compatível com
    ``client.subscribe(model, arguments=...)`` retornando ``{"video": {"url": str}}``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        backoff_cap: float = 8.0,
    ) -> None:
        if client is None and not (api_key and api_key.strip()):
            raise I2VGenerationError(
                "FAL_KEY é obrigatório para o provider fal.ai. "
                "Defina a variável de ambiente FAL_KEY ou configure fal_api_key."
            )
        if api_key:
            os.environ["FAL_KEY"] = api_key
        self._client = client
        self._model = model or DEFAULT_FAL_MODEL
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import fal_client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise I2VGenerationError(
                "SDK `fal-client` não instalado. Instale o extra `comic`: "
                "`pip install -e .[comic]`."
            ) from exc
        self._client = fal_client
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

        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._submit_and_wait(
                    client,
                    prompt_image=prompt_image_path,
                    prompt_text=prompt_text,
                    duration_seconds=duration_seconds,
                    ratio=ratio,
                )
            except I2VGenerationError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                delay = _exp_backoff_seconds(attempt, self._backoff_base, self._backoff_cap)
                logger.warning(
                    "comic.i2v.fal: falha na tentativa %d (%s); aguardando %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise I2VGenerationError(
            f"Falha ao gerar mini-clipe via fal.ai após {self._max_retries + 1} tentativas: {last_exc}"
        ) from last_exc

    def _submit_and_wait(
        self,
        client: Any,
        *,
        prompt_image: Path,
        prompt_text: str,
        duration_seconds: float,
        ratio: str,
    ) -> bytes:
        image_url = _file_to_data_url(prompt_image)
        aspect_ratio = _RATIO_MAP.get(ratio, ratio)

        # Kling aceita "5" ou "10" (string). Outros modelos aceitam int.
        # Para máxima compatibilidade, mandamos como string.
        duration = str(int(round(max(1.0, duration_seconds))))

        arguments: dict[str, Any] = {
            "image_url": image_url,
            "prompt": prompt_text,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        result = client.subscribe(
            self._model,
            arguments=arguments,
            with_logs=False,
        )

        video_url = self._extract_video_url(result)
        return self._download_bytes(video_url)

    @staticmethod
    def _extract_video_url(result: Any) -> str:
        if isinstance(result, dict):
            video = result.get("video")
        else:
            video = getattr(result, "video", None)
        if isinstance(video, dict):
            url = video.get("url")
        else:
            url = getattr(video, "url", None)
        if not url:
            raise I2VGenerationError(
                f"Resposta inesperada do fal.ai (sem video.url): {result!r}"
            )
        return str(url)

    @staticmethod
    def _download_bytes(url: str, timeout: float = 60.0) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            raise I2VGenerationError(
                f"Falha ao baixar o mini-clipe gerado pelo fal.ai: {exc}"
            ) from exc
