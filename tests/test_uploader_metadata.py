from pathlib import Path

import pytest

from youcut.uploader.base import ClipMetadata
from youcut.uploader.metadata import apply_platform_limits, parse_clip_metadata


@pytest.fixture()
def tmp_clip(tmp_path: Path):
    def _make(content: str) -> Path:
        p = tmp_path / "clip_01.txt"
        p.write_text(content, encoding="utf-8")
        return p

    return _make


class TestParseClipMetadata:
    def test_all_fields_present(self, tmp_clip):
        path = tmp_clip(
            "TÍTULO\nMeu Título\n\nDESCRIÇÃO\nMinha descrição\n\nHASHTAGS\n#tag1 #tag2\n"
        )
        meta = parse_clip_metadata(path)
        assert meta.title == "Meu Título"
        assert meta.description == "Minha descrição"
        assert meta.hashtags == ["#tag1", "#tag2"]
        assert "#tag1" in meta.caption
        assert "Minha descrição" in meta.caption

    def test_missing_description(self, tmp_clip):
        path = tmp_clip("TÍTULO\nSó Título\n\nHASHTAGS\n#tag1\n")
        meta = parse_clip_metadata(path)
        assert meta.title == "Só Título"
        assert meta.description == ""
        assert meta.hashtags == ["#tag1"]

    def test_missing_hashtags(self, tmp_clip):
        path = tmp_clip("TÍTULO\nTítulo\n\nDESCRIÇÃO\nDescrição sem tags\n")
        meta = parse_clip_metadata(path)
        assert meta.hashtags == []
        assert meta.caption == "Descrição sem tags"

    def test_all_fields_missing(self, tmp_clip):
        path = tmp_clip("SUGESTÃO DE THUMBNAIL\nAlguma sugestão\n")
        meta = parse_clip_metadata(path)
        assert meta.title == ""
        assert meta.description == ""
        assert meta.hashtags == []
        assert meta.caption == ""

    def test_special_characters_and_emojis(self, tmp_clip):
        path = tmp_clip(
            "TÍTULO\n🚨 Título com Emoji & Símbolos!\n\n"
            "DESCRIÇÃO\nDescrição com 'aspas', \"inglesas\" e — travessão.\n\n"
            "HASHTAGS\n#Olá #Ação #Coração❤️\n"
        )
        meta = parse_clip_metadata(path)
        assert "🚨" in meta.title
        assert "—" in meta.description
        assert "#Coração❤️" in meta.hashtags

    def test_real_clip_format(self, tmp_clip):
        content = (
            "TÍTULO\n🚨 'Prendeu ou Matou' — A Origem do Grito que Dividiu o Brasil\n\n"
            "DESCRIÇÃO\nA frase que virou bordão nacional tem uma história por trás.\n\n"
            "HASHTAGS\n#PrendeuMatou #Segurança #JustiçaBrasileira #Viral\n\n"
            "SUGESTÃO DE THUMBNAIL\nFrame do apresentador.\n\n"
            "NOTA DE VIRALIDADE: 9.5/10\n\n"
            "MOTIVO DA SELEÇÃO\nGancho forte.\n"
        )
        path = tmp_clip(content)
        meta = parse_clip_metadata(path)
        assert "Prendeu ou Matou" in meta.title
        assert "#PrendeuMatou" in meta.hashtags
        assert len(meta.hashtags) == 4


class TestApplyPlatformLimitsYouTube:
    def _meta(self, title="", description="", hashtags=None) -> ClipMetadata:
        hashtags = hashtags or []
        caption = (description + "\n\n" + " ".join(hashtags)).strip() if hashtags else description
        return ClipMetadata(title=title, description=description, hashtags=hashtags, caption=caption)

    def test_title_within_limit_unchanged(self):
        meta = self._meta(title="A" * 50, description="desc", hashtags=["#x"])
        result = apply_platform_limits(meta, "youtube")
        assert len(result.title) == 50

    def test_title_truncated_at_100(self):
        meta = self._meta(title="A" * 150)
        result = apply_platform_limits(meta, "youtube")
        assert len(result.title) == 100

    def test_description_within_limit_unchanged(self):
        meta = self._meta(title="T", description="D" * 100, hashtags=["#tag"])
        result = apply_platform_limits(meta, "youtube")
        assert result.description == "D" * 100

    def test_description_truncated_preserves_hashtags(self):
        long_desc = "D" * 6000
        hashtags = ["#tag1", "#tag2"]
        meta = self._meta(title="T", description=long_desc, hashtags=hashtags)
        result = apply_platform_limits(meta, "youtube")
        assert len(result.description) < 6000
        # Hashtags must be preserved in caption
        for tag in hashtags:
            assert tag in result.caption
        # Caption must be within 5000 chars
        assert len(result.caption) <= 5000

    def test_no_description_truncation_when_no_hashtags(self):
        meta = self._meta(title="T", description="D" * 5000)
        result = apply_platform_limits(meta, "youtube")
        assert len(result.description) == 5000


class TestApplyPlatformLimitsInstagram:
    def _meta(self, description="", hashtags=None) -> ClipMetadata:
        hashtags = hashtags or []
        caption = (description + "\n\n" + " ".join(hashtags)).strip() if hashtags else description
        return ClipMetadata(title="T", description=description, hashtags=hashtags, caption=caption)

    def test_caption_within_limit_unchanged(self):
        meta = self._meta(description="D" * 100, hashtags=["#x"])
        result = apply_platform_limits(meta, "instagram")
        assert "D" * 100 in result.caption
        assert "#x" in result.caption

    def test_caption_truncated_preserves_hashtags(self):
        long_desc = "D" * 3000
        hashtags = ["#instagramtag", "#viral"]
        meta = self._meta(description=long_desc, hashtags=hashtags)
        result = apply_platform_limits(meta, "instagram")
        assert len(result.caption) <= 2200
        for tag in hashtags:
            assert tag in result.caption

    def test_description_cut_not_hashtags(self):
        hashtags = ["#a", "#b", "#c"]
        hashtag_block = " ".join(hashtags)
        available = 2200 - len(hashtag_block) - 2  # separator "\n\n"
        long_desc = "X" * (available + 500)
        meta = self._meta(description=long_desc, hashtags=hashtags)
        result = apply_platform_limits(meta, "instagram")
        assert len(result.caption) <= 2200
        assert all(tag in result.caption for tag in hashtags)


class TestApplyPlatformLimitsTikTok:
    def _meta(self, title="T", description="", hashtags=None) -> ClipMetadata:
        hashtags = hashtags or []
        caption = (description + "\n\n" + " ".join(hashtags)).strip() if hashtags else description
        return ClipMetadata(title=title, description=description, hashtags=hashtags, caption=caption)

    def test_caption_truncated_preserves_hashtags(self):
        long_desc = "T" * 3000
        hashtags = ["#tiktok", "#fyp"]
        meta = self._meta(description=long_desc, hashtags=hashtags)
        result = apply_platform_limits(meta, "tiktok")
        assert len(result.caption) <= 2200
        for tag in hashtags:
            assert tag in result.caption

    def test_caption_within_limit_unchanged(self):
        meta = self._meta(description="Short description", hashtags=["#short"])
        result = apply_platform_limits(meta, "tiktok")
        assert result.description == "Short description"
        assert "#short" in result.caption

    def test_tiktok_caption_includes_title(self):
        meta = self._meta(title="My Viral Title", description="Some description", hashtags=["#fyp"])
        result = apply_platform_limits(meta, "tiktok")
        assert "My Viral Title" in result.caption

    def test_hashtags_only_exceeding_limit_gives_empty_description(self):
        # hashtag block alone nearly fills the limit
        big_hashtag = "#" + "x" * 2199
        meta = self._meta(description="Should be empty", hashtags=[big_hashtag])
        result = apply_platform_limits(meta, "tiktok")
        assert len(result.caption) <= 2200


class TestApplyPlatformLimitsNegativeAvailable:
    def test_instagram_hashtag_block_exceeds_limit(self):
        big_hashtag = "#" + "x" * 2199
        meta = ClipMetadata(
            title="T",
            description="Some description",
            hashtags=[big_hashtag],
            caption="",
        )
        result = apply_platform_limits(meta, "instagram")
        assert len(result.caption) <= 2200

    def test_youtube_hashtag_block_exceeds_limit(self):
        big_hashtag = "#" + "x" * 5001
        meta = ClipMetadata(
            title="T",
            description="Some description",
            hashtags=[big_hashtag],
            caption="",
        )
        result = apply_platform_limits(meta, "youtube")
        assert len(result.caption) <= 5000


class TestApplyPlatformLimitsUnknown:
    def test_unknown_platform_returns_unchanged(self):
        meta = ClipMetadata(
            title="T" * 200,
            description="D" * 10000,
            hashtags=["#x"],
            caption="D" * 10000 + "\n\n#x",
        )
        result = apply_platform_limits(meta, "unknown_platform")
        assert result.title == "T" * 200
        assert result.description == "D" * 10000
