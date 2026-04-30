"""Pipeline de motion comic do YouCut.

Expõe :func:`run_comic_pipeline` como única superfície pública usada pelo
subcomando ``youcut comic`` e por integrações externas.
"""

from youcut.comic.pipeline import (  # noqa: F401
    ComicPipelineError,
    PipelineCallbacks,
    run_comic_pipeline,
)

__all__ = ["run_comic_pipeline", "PipelineCallbacks", "ComicPipelineError"]
