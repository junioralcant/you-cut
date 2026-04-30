"""Image provider para o `youcut comic`.

Define o Protocol :class:`ImageProvider` e a implementação concreta
:class:`OpenAIImageProvider` baseada no SDK ``openai>=1`` com modelo
``gpt-image-1`` e ``input_fidelity="high"`` quando há referências.

Reaplica retry com backoff exponencial respeitando ``comic_image_retries``.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_SIZE: str = "1024x1024"
SOCIAL_PORTRAIT_SIZE: str = "1024x1536"


class ImageGenerationError(Exception):
    """Erro terminal de geração de imagem após esgotar as tentativas."""


@runtime_checkable
class ImageProvider(Protocol):
    """Provider de geração de imagem (texto-para-imagem com referências).

    Retorna ``bytes`` PNG. ``reference_images`` é opcional; quando presente,
    o provider deve respeitar fidelidade alta às pessoas/objetos referenciados.
    """

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[Path] | None = None,
        size: str = DEFAULT_SIZE,
        input_fidelity: str = "high",
    ) -> bytes: ...


def _exp_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 8.0) -> float:
    return min(cap, base * (2**attempt))


class OpenAIImageProvider:
    """`ImageProvider` para gpt-image-1 via SDK ``openai``.

    Uso:
        ``OpenAIImageProvider(api_key=..., max_retries=2)``

    Sem `api_key`, levanta erro em pt-BR. O `client` pode ser injetado para
    testes (qualquer objeto compatível com ``client.images.generate(...)`` e
    ``client.images.edit(...)``).
    """

    DEFAULT_MODEL: str = "gpt-image-1"

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
            raise ImageGenerationError(
                "OPENAI_API_KEY é obrigatório para o `youcut comic`. "
                "Defina a variável de ambiente ou use --openai-api-key."
            )
        self._api_key = api_key
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI  # type: ignore[import-not-found]

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[Path] | None = None,
        size: str = DEFAULT_SIZE,
        input_fidelity: str = "high",
    ) -> bytes:
        if not prompt or not prompt.strip():
            raise ImageGenerationError("prompt vazio não é permitido para geração de imagem.")

        client = self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                if reference_images:
                    return self._call_edit(
                        client,
                        prompt=prompt,
                        reference_images=reference_images,
                        size=size,
                        input_fidelity=input_fidelity,
                    )
                return self._call_generate(
                    client,
                    prompt=prompt,
                    size=size,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                delay = _exp_backoff_seconds(attempt, self._backoff_base, self._backoff_cap)
                logger.warning(
                    "comic.images: falha na tentativa %d (%s); aguardando %.1fs",
                    attempt + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise ImageGenerationError(
            f"Falha ao gerar imagem após {self._max_retries + 1} tentativas: {last_exc}"
        ) from last_exc

    def _call_generate(self, client: Any, *, prompt: str, size: str) -> bytes:
        response = client.images.generate(
            model=self._model,
            prompt=prompt,
            size=size,
            n=1,
        )
        return self._extract_png_bytes(response)

    def _call_edit(
        self,
        client: Any,
        *,
        prompt: str,
        reference_images: list[Path],
        size: str,
        input_fidelity: str,
    ) -> bytes:
        opened: list[Any] = []
        try:
            for ref in reference_images:
                ref_path = Path(ref)
                if not ref_path.exists():
                    raise ImageGenerationError(
                        f"Imagem de referência não encontrada: {ref_path}"
                    )
                opened.append(ref_path.open("rb"))
            response = client.images.edit(
                model=self._model,
                image=opened,
                prompt=prompt,
                size=size,
                input_fidelity=input_fidelity,
                n=1,
            )
            return self._extract_png_bytes(response)
        finally:
            for handle in opened:
                try:
                    handle.close()
                except Exception:
                    pass

    @staticmethod
    def _extract_png_bytes(response: Any) -> bytes:
        try:
            payload = response.data[0]
            b64 = getattr(payload, "b64_json", None) or payload["b64_json"]
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ImageGenerationError(
                f"Resposta inesperada do provider de imagem: {response!r}"
            ) from exc
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError("b64_json inválido na resposta do provider.") from exc
