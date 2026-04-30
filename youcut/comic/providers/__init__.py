"""Camada de providers de IA do `youcut comic`.

Define os Protocols e expõe as implementações reais (OpenAI/Runway) e os
fakes determinísticos (no pacote de testes) usados pelo restante do
pipeline.
"""

from youcut.comic.providers.i2v import (
    I2VGenerationError,
    ImageToVideoProvider,
    RunwayProvider,
)
from youcut.comic.providers.images import (
    ImageGenerationError,
    ImageProvider,
    OpenAIImageProvider,
)

__all__ = [
    "ImageProvider",
    "OpenAIImageProvider",
    "ImageGenerationError",
    "ImageToVideoProvider",
    "RunwayProvider",
    "I2VGenerationError",
]
