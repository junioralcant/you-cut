"""Animation provider via Prunaai (image + audio → video animado completo).

Diferente do `ImageToVideoProvider` (Runway/Replicate Hailuo) que precisa
ser chamado N vezes (uma por painel), o **Prunaai gera o vídeo final
inteiro de uma vez** a partir de:

- ``image``: PNG/JPG do master composition (cenário + personagens posicionados)
- ``audio``: arquivo MP3/WAV com a fala completa (≤30s recomendado)
- ``video_prompt``: descrição rica do estilo, posições e ações
- ``voice_prompt``: tom emocional do diálogo

A animação inclui automaticamente: lip-sync, camera-punches, gesticulação
dos personagens, leve scroll de fundo. Custo aproximado: $0.0005/s de output
(uma run de 26s sai em ~$0.01).

Modelo no Replicate: ``prunaai/p-video-avatar``.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_RESOLUTION: str = "720p"
MAX_RETRIES_DEFAULT: int = 2


class PrunaaiAnimationError(Exception):
    """Erro terminal ao gerar animação via Prunaai (após retries esgotados)."""


def _exp_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 8.0) -> float:
    return min(cap, base * (2**attempt))


class PrunaaiAnimationProvider:
    """Provider de animação completa baseada em Prunaai/p-video-avatar.

    Uso:
        ``PrunaaiAnimationProvider(api_token=..., max_retries=2)``

    O ``client`` pode ser injetado para testes (objeto compatível com
    ``client.run(model, input=...)`` retornando URL string ou objeto FileOutput
    com método ``read()``).
    """

    MODEL: str = "prunaai/p-video-avatar"

    def __init__(
        self,
        *,
        api_token: str | None = None,
        client: Any | None = None,
        max_retries: int = MAX_RETRIES_DEFAULT,
        backoff_base: float = 1.0,
        backoff_cap: float = 8.0,
    ) -> None:
        if client is None and not (api_token and api_token.strip()):
            raise PrunaaiAnimationError(
                "REPLICATE_API_TOKEN é obrigatório para o provider Prunaai. "
                "Defina a variável de ambiente REPLICATE_API_TOKEN ou configure "
                "replicate_api_token."
            )
        self._api_token = api_token
        self._client = client
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import replicate  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PrunaaiAnimationError(
                "SDK `replicate` não instalado. Instale o extra `comic`: "
                "`pip install -e .[comic]`."
            ) from exc
        self._client = replicate.Client(api_token=self._api_token)
        return self._client

    def animate(
        self,
        image_path: Path,
        audio_path: Path,
        *,
        video_prompt: str,
        voice_prompt: str | None = None,
        resolution: str = DEFAULT_RESOLUTION,
    ) -> bytes:
        """Gera o vídeo final.

        Levanta :class:`PrunaaiAnimationError` se as ``max_retries+1``
        tentativas falharem.
        """

        image_path = Path(image_path)
        audio_path = Path(audio_path)
        if not image_path.exists():
            raise PrunaaiAnimationError(f"image não encontrada: {image_path}")
        if not audio_path.exists():
            raise PrunaaiAnimationError(f"audio não encontrado: {audio_path}")

        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._run_and_download(
                    client,
                    image_path=image_path,
                    audio_path=audio_path,
                    video_prompt=video_prompt,
                    voice_prompt=voice_prompt,
                    resolution=resolution,
                )
            except PrunaaiAnimationError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                delay = _exp_backoff_seconds(
                    attempt, self._backoff_base, self._backoff_cap
                )
                logger.warning(
                    "comic.prunaai: falha na tentativa %d (%s); aguardando %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise PrunaaiAnimationError(
            f"Falha ao gerar animação via Prunaai após "
            f"{self._max_retries + 1} tentativas: {last_exc}"
        ) from last_exc

    def _run_and_download(
        self,
        client: Any,
        *,
        image_path: Path,
        audio_path: Path,
        video_prompt: str,
        voice_prompt: str | None,
        resolution: str,
    ) -> bytes:
        payload: dict[str, Any] = {
            "video_prompt": video_prompt,
            "resolution": resolution,
        }
        if voice_prompt:
            payload["voice_prompt"] = voice_prompt

        with open(image_path, "rb") as image_file, open(audio_path, "rb") as audio_file:
            payload["image"] = image_file
            payload["audio"] = audio_file
            output = client.run(self.MODEL, input=payload)

        return self._extract_video_bytes(output)

    @staticmethod
    def _extract_video_bytes(output: Any) -> bytes:
        # Output pode ser FileOutput com .read()/.url, URL string ou lista.
        url = PrunaaiAnimationProvider._extract_url(output)
        if url:
            return PrunaaiAnimationProvider._download_bytes(url)

        if hasattr(output, "read") and callable(output.read):
            data = output.read()
            if isinstance(data, bytes) and data:
                return data

        raise PrunaaiAnimationError(
            f"Resposta inesperada do Prunaai (URL ausente): {output!r}"
        )

    @staticmethod
    def _extract_url(output: Any) -> str | None:
        candidate: Any = None
        if isinstance(output, str):
            candidate = output
        elif isinstance(output, list) and output:
            first = output[0]
            candidate = getattr(first, "url", None) or first
        elif hasattr(output, "url"):
            candidate = output.url
        if candidate is None:
            return None
        url = str(candidate).strip()
        if not url.startswith(("http://", "https://")):
            return None
        return url

    @staticmethod
    def _download_bytes(url: str, timeout: float = 180.0) -> bytes:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "youcut/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            raise PrunaaiAnimationError(
                f"Falha ao baixar o vídeo gerado pelo Prunaai: {exc}"
            ) from exc
