"""Catálogo local de imagens de jogadores + detecção de menções na transcrição.

Quando o nome de um jogador presente em ``players_dir`` aparece no áudio de um
clipe, a imagem dele é injetada como reference frame extra na geração da
thumbnail (YouTube 16:9) ou da imagem social (9:16). Determinístico e sem
custo de API — opcionalmente usa Claude para desambiguar quando uma menção
curta (ex.: "Danilo") bate em múltiplos jogadores do catálogo.
"""

from youcut.players.catalog import PlayerCatalog, load_catalog
from youcut.players.detector import detect_players, slice_transcript_for_clip
from youcut.players.disambiguator import disambiguate_mentions
from youcut.players.models import PlayerMention, PlayerProfile
from youcut.players.timeline import PlayerSegment, build_player_timeline

__all__ = [
    "PlayerCatalog",
    "PlayerMention",
    "PlayerProfile",
    "PlayerSegment",
    "build_player_timeline",
    "detect_players",
    "disambiguate_mentions",
    "load_catalog",
    "slice_transcript_for_clip",
]
