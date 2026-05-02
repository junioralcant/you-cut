"""Image-to-video provider via Replicate (default: Kling v2.1 Master).

Replicate é uma plataforma agregadora US-based que hospeda modelos Kling
da Kuaishou (e muitos outros). Vantagens sobre conexão direta com Kling:

- Auth simples via Bearer token (sem JWT/HMAC).
- Onboarding internacional (cartão BR funciona normalmente).
- Pay-per-use sem daily cap rígido (vs Runway Tier 1 = 50/dia).
- SDK Python ``replicate`` maduro e estável.

Modelo default: ``kwaivgi/kling-v2.1-master`` — melhor qualidade Kling
disponível atualmente. Outros úteis:
``kwaivgi/kling-v2.1-pro`` (mais barato),
``kwaivgi/kling-v1.6-pro`` (geração anterior, ainda excelente).

Custo aproximado: $0.06–0.12/s gerado dependendo do modelo.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from youcut.comic.providers.i2v import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_RATIO,
    I2VGenerationError,
)

logger = logging.getLogger(__name__)

DEFAULT_REPLICATE_MODEL: str = "kwaivgi/kling-v1.6-pro"

# Replicate/Kling usa labels textuais ("9:16") em vez de "WxH" do Runway.
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


class ReplicateImageToVideoProvider:
    """`ImageToVideoProvider` baseado em Replicate (multi-modelo Kling).

    Uso:
        ``ReplicateImageToVideoProvider(api_token=..., model="kwaivgi/kling-v2.1-master")``

    O ``client`` pode ser injetado para testes (objeto compatível com
    ``client.run(model, input=...)`` retornando URL string ou objeto FileOutput
    com método ``read()``).
    """

    def __init__(
        self,
        *,
        api_token: str | None = None,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        backoff_cap: float = 8.0,
    ) -> None:
        if client is None and not (api_token and api_token.strip()):
            raise I2VGenerationError(
                "REPLICATE_API_TOKEN é obrigatório para o provider Replicate. "
                "Defina a variável de ambiente REPLICATE_API_TOKEN ou configure replicate_api_token."
            )
        self._api_token = api_token
        self._client = client
        self._model = model or DEFAULT_REPLICATE_MODEL
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import replicate  # type: ignore[import-not-found]
        except ImportError as exc:
            raise I2VGenerationError(
                "SDK `replicate` não instalado. Instale o extra `comic`: "
                "`pip install -e .[comic]`."
            ) from exc
        self._client = replicate.Client(api_token=self._api_token)
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
                return self._run_and_download(
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
                    "comic.i2v.replicate: falha na tentativa %d (%s); aguardando %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise I2VGenerationError(
            f"Falha ao gerar mini-clipe via Replicate após {self._max_retries + 1} tentativas: {last_exc}"
        ) from last_exc

    def _run_and_download(
        self,
        client: Any,
        *,
        prompt_image: Path,
        prompt_text: str,
        duration_seconds: float,
        ratio: str,
    ) -> bytes:
        aspect_ratio = _RATIO_MAP.get(ratio, ratio)

        with open(prompt_image, "rb") as image_file:
            payload = self._build_payload(
                model=self._model,
                image_file=image_file,
                prompt_text=prompt_text,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
            )
            output = client.run(self._model, input=payload)

        return self._extract_video_bytes(output)

    @staticmethod
    def _build_payload(
        *,
        model: str,
        image_file: Any,
        prompt_text: str,
        duration_seconds: float,
        aspect_ratio: str,
    ) -> dict[str, Any]:
        """Mapeia parâmetros para o schema de cada família de modelo.

        Os modelos no Replicate têm schemas distintos:
        - Kling (kwaivgi/*): ``start_image``, ``aspect_ratio``, duration ∈ {5, 10}
        - Hailuo / MiniMax (minimax/*): ``first_frame_image``, ``resolution``,
          duration ∈ {6, 10}, ``prompt_optimizer``
        - Outros: padrão Kling (com truncagem 1000-char no prompt)
        """

        # Kling tem limite de 1000 chars no prompt; truncar com ellipsis se exceder.
        clipped_prompt = prompt_text if len(prompt_text) <= 990 else prompt_text[:987] + "..."

        if model.startswith("minimax/"):
            # Hailuo aceita duration 6 ou 10. Painéis curtos (2-5s) → 6s.
            duration = 10 if duration_seconds > 8.0 else 6
            return {
                "first_frame_image": image_file,
                "prompt": clipped_prompt,
                "duration": duration,
                "resolution": "768p",
                "prompt_optimizer": True,
            }

        # Default: schema Kling (kwaivgi/*).
        # Kling aceita APENAS duration ∈ {5, 10}.
        duration = 10 if duration_seconds > 7.5 else 5
        return {
            "start_image": image_file,
            "prompt": clipped_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

    def _extract_video_bytes(self, output: Any) -> bytes:
        # Replicate.run retorna formatos diferentes por modelo:
        # - URL string direta
        # - lista [url1, url2, ...]
        # - FileOutput (objeto com .read() e .url)
        # Em SDK >=1.x, ``.read()`` pode devolver bytes vazios se o stream já
        # foi consumido internamente; por isso preferimos ``.url`` quando
        # disponível e só caímos em ``.read()`` como último recurso.
        url = self._extract_url(output)
        if url:
            return self._download_bytes(url)

        if hasattr(output, "read") and callable(output.read):
            data = output.read()
            if isinstance(data, bytes) and data:
                return data

        raise I2VGenerationError(
            f"Resposta inesperada do Replicate (URL ausente): {output!r}"
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
    def _download_bytes(url: str, timeout: float = 60.0) -> bytes:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "youcut/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            raise I2VGenerationError(
                f"Falha ao baixar o mini-clipe gerado pelo Replicate: {exc}"
            ) from exc
