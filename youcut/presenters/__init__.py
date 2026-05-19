"""Catálogo local de apresentadores + detecção via Claude vision.

Análogo ao módulo :mod:`youcut.players` mas pra hosts/apresentadores
do canal. A diferença chave: detecção é por **rosto nos frames**
(Claude vision) em vez de menção na transcrição. O apresentador
costuma não nomear a si próprio, então transcript-scan não funciona.

Estratégia padrão: identifica os apresentadores **uma vez por vídeo
source** (cache em memória), e injeta as fotos como reference frame
em **todas** as thumbnails desse vídeo.
"""

from youcut.presenters.catalog import PresenterCatalog, load_catalog
from youcut.presenters.detector import detect_presenters
from youcut.presenters.models import PresenterDetection, PresenterProfile

__all__ = [
    "PresenterCatalog",
    "PresenterDetection",
    "PresenterProfile",
    "detect_presenters",
    "load_catalog",
]
