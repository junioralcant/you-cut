"""Composição local do painel "imagem IA" usando a foto do jogador.

Quando ``PipelineConfig.player_panel_use_local=True`` e o detector encontra
ao menos um jogador no transcript do clipe, em vez de gerar a imagem via
DALL·E/seedream a partir das fotos como reference frames, geramos
diretamente uma imagem PNG do tamanho do painel social com a foto do
jogador em **cover full-bleed** (preenche o painel inteiro, recortando as
bordas do retrato se preciso pra preservar o aspect ratio).

A PNG resultante é entregue ao ``_render_social_header_image_local`` no
lugar do ``top_image_path`` que viria da IA — o fluxo de banda+composição
do social_composer não muda.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def _cover_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale-crop preservando aspect ratio pra cobrir ``target_w × target_h``.

    Análogo CSS ``background-size: cover`` + ``object-fit: cover`` — escala
    pela menor dimensão pra garantir cobertura, recorta o excedente no
    centro. Sem letterbox/pillarbox.
    """
    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    new_w = max(target_w, int(iw * scale))
    new_h = max(target_h, int(ih * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x_off = max(0, (new_w - target_w) // 2)
    y_off = max(0, (new_h - target_h) // 2)
    return resized.crop((x_off, y_off, x_off + target_w, y_off + target_h))


def compose_player_panel(
    *,
    photo_bytes: bytes,
    player_display_name: str,
    width: int,
    height: int,
) -> Path:
    """Gera PNG temp do painel com a foto do jogador em cover full-bleed.

    O parâmetro ``player_display_name`` é mantido por compatibilidade de API
    mas atualmente é ignorado — o nome do jogador foi removido do painel a
    pedido do usuário (visual de transmissão esportiva, sem tarja de nome).
    O contexto do áudio do apresentador identifica o jogador.

    Layout:
        ┌──────────────────────────────┐
        │ ████ FOTO DO JOGADOR ███████ │
        │ ████ (cover full-bleed) ███ │
        │ ████ recorta bordas se ████ │
        │ ████ precisa preservar ████ │
        │ ████ aspect ratio █████████ │
        └──────────────────────────────┘

    Returns:
        Path da PNG temporária criada.
    """
    del player_display_name  # mantido na assinatura por compat; não usado

    try:
        photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Falha ao decodificar foto do jogador: %s — usando placeholder cinza", exc)
        photo = Image.new("RGB", (width, height), color=(40, 40, 40))

    canvas = _cover_crop(photo, width, height)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out = Path(tmp.name)
    canvas.save(out, format="PNG", optimize=True)
    return out
