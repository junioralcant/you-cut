"""Testes unitários para YouTubeMusicProvider em youcut/music/provider.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from youcut.models import MusicTrack, ViralClip
from youcut.music.library import MusicLibrary
from youcut.music.provider import YouTubeMusicProvider


def _make_clip(title: str = "", reason: str = "", style: str = "") -> ViralClip:
    return ViralClip(
        title=title,
        reason=reason,
        viral_score=7.0,
        start_time=0.0,
        end_time=60.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="idea",
        social_visual_style=style,
    )


def _seed_library(root: Path, tracks: list[dict]) -> MusicLibrary:
    lib = MusicLibrary(root=root)
    for spec in tracks:
        lib.add(
            MusicTrack(
                video_id=spec["video_id"],
                name=spec.get("name", spec["video_id"]),
                source_url=f"https://www.youtube.com/watch?v={spec['video_id']}",
                local_path=root / "tracks" / f"{spec['video_id']}.m4a",
                mood=spec.get("mood", "feliz"),
                duration_s=float(spec.get("duration_s", 90.0)),
            )
        )
    lib.save()
    return lib


# ── classify_mood (preserva heurística existente) ───────────────────────────


@pytest.fixture
def empty_lib(tmp_path):
    return MusicLibrary(root=tmp_path / "music")


class TestClassifyMood:
    def test_motivacional(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(title="Minha jornada de motivação e superação")
        assert provider.classify_mood(clip) == "motivacional"

    def test_reflexivo(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(reason="Uma reflexão profunda sobre paz e calma")
        assert provider.classify_mood(clip) == "reflexivo"

    def test_energico(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(title="Treino intenso com muita energia e ação")
        assert provider.classify_mood(clip) == "energico"

    def test_emocional(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(reason="Uma história de amor e saudade profunda")
        assert provider.classify_mood(clip) == "emocional"

    def test_feliz(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(title="Celebração de alegria e felicidade")
        assert provider.classify_mood(clip) == "feliz"

    def test_dramatico(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(style="Cena dramática com tensão e conflito intenso")
        assert provider.classify_mood(clip) == "dramatico"

    def test_default_fallback(self, empty_lib):
        provider = YouTubeMusicProvider(empty_lib)
        clip = _make_clip(title="Conteúdo genérico sem palavras-chave conhecidas")
        assert provider.classify_mood(clip) == "motivacional"


# ── pick_track ──────────────────────────────────────────────────────────────


class TestPickTrack:
    def test_returns_none_when_library_empty(self, tmp_path):
        """RF-14: acervo vazio → retorna None."""
        lib = MusicLibrary(root=tmp_path / "music")
        provider = YouTubeMusicProvider(lib)
        clip = _make_clip(title="Qualquer coisa")
        assert provider.pick_track(clip) is None

    def test_picks_from_matching_mood(self, tmp_path):
        """RF-11: faixa do acervo deve ter mood igual ao mood do clipe."""
        lib = _seed_library(
            tmp_path / "music",
            [
                {"video_id": "vidF1", "mood": "feliz"},
                {"video_id": "vidF2", "mood": "feliz"},
                {"video_id": "vidD1", "mood": "dramatico"},
            ],
        )
        provider = YouTubeMusicProvider(lib)
        clip = _make_clip(title="Celebração de alegria e felicidade")
        chosen = provider.pick_track(clip)
        assert chosen is not None
        assert chosen.mood == "feliz"
        assert chosen.video_id in {"vidF1", "vidF2"}

    def test_fallback_global_when_no_matching_mood(self, tmp_path):
        """RF-13: nenhuma faixa com o mood do clipe → escolhe entre todas."""
        lib = _seed_library(
            tmp_path / "music",
            [
                {"video_id": "vidF1", "mood": "feliz"},
                {"video_id": "vidF2", "mood": "feliz"},
                {"video_id": "vidF3", "mood": "feliz"},
            ],
        )
        provider = YouTubeMusicProvider(lib)
        clip = _make_clip(title="Cena dramática com tensão e conflito intenso")
        chosen = provider.pick_track(clip)
        assert chosen is not None
        # Acervo só tem 'feliz' → fallback global retorna alguma 'feliz'.
        assert chosen.video_id in {"vidF1", "vidF2", "vidF3"}

    def test_deterministic_same_clip(self, tmp_path):
        """RF-12: mesmo clipe → mesma faixa em chamadas sucessivas."""
        lib = _seed_library(
            tmp_path / "music",
            [
                {"video_id": f"vid{i}", "mood": "feliz"}
                for i in range(8)
            ],
        )
        clip = _make_clip(
            title="Celebração de alegria",
            reason="alegria",
            style="claro",
        )
        provider = YouTubeMusicProvider(lib)
        a = provider.pick_track(clip)
        b = provider.pick_track(clip)
        c = provider.pick_track(clip)
        assert a is not None and b is not None and c is not None
        assert a.video_id == b.video_id == c.video_id

    def test_deterministic_across_provider_instances(self, tmp_path):
        """RF-12/15: dois providers diferentes (mesmo acervo) → mesma escolha."""
        root = tmp_path / "music"
        _seed_library(
            root,
            [
                {"video_id": f"v{i:02d}", "mood": "feliz"}
                for i in range(10)
            ],
        )
        clip = _make_clip(title="alegria pura", reason="diversão", style="vibrante")

        lib1 = MusicLibrary(root=root)
        lib2 = MusicLibrary(root=root)
        a = YouTubeMusicProvider(lib1).pick_track(clip)
        b = YouTubeMusicProvider(lib2).pick_track(clip)
        assert a is not None and b is not None
        assert a.video_id == b.video_id

    def test_different_clips_pick_different_tracks(self, tmp_path):
        """Clipes diferentes em acervo grande devem (em geral) cair em buckets distintos."""
        lib = _seed_library(
            tmp_path / "music",
            [{"video_id": f"vid{i:02d}", "mood": "feliz"} for i in range(20)],
        )
        provider = YouTubeMusicProvider(lib)
        a = provider.pick_track(_make_clip(title="alegria pura A", style="vibrante"))
        b = provider.pick_track(_make_clip(title="alegria pura B", style="vibrante"))
        c = provider.pick_track(_make_clip(title="alegria pura C", style="vibrante"))
        # SHA-256 muito improvavelmente colide para 3 strings distintas em buckets de 20
        assert {a.video_id, b.video_id, c.video_id} != {a.video_id}

    def test_fallback_also_deterministic(self, tmp_path):
        """RF-15: fallback global também deve ser determinístico."""
        lib = _seed_library(
            tmp_path / "music",
            [{"video_id": f"v{i:02d}", "mood": "feliz"} for i in range(6)],
        )
        clip = _make_clip(title="tensão e drama profundo")
        provider = YouTubeMusicProvider(lib)
        a = provider.pick_track(clip)
        b = provider.pick_track(clip)
        assert a is not None and b is not None
        assert a.video_id == b.video_id
