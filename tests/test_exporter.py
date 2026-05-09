from pathlib import Path

import pytest

from youcut.exporter import export_metadata
from youcut.models import MusicTrack, ViralClip


@pytest.fixture
def sample_clip() -> ViralClip:
    return ViralClip(
        title="Você sabia que pode ganhar tempo fazendo isso?",
        reason="Gancho forte nos primeiros 3 segundos com pergunta retórica + dica prática e acionável.",
        viral_score=8.5,
        start_time=30.0,
        end_time=75.0,
        description="Neste clipe, revelo a técnica que mudou minha produtividade. Assista até o final!",
        hashtags=["produtividade", "dica", "shorts", "viral"],
        thumbnail_idea="Frame em 00:32 onde o apresentador aponta para a câmera com expressão de surpresa.",
        thumbnail_text="MOMENTO IMPACTANTE",
    )


def test_file_created_with_correct_name(sample_clip, tmp_path):
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    assert result == tmp_path / "clip_01.txt"
    assert result.exists()


def test_sequential_naming(sample_clip, tmp_path):
    path1 = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    path2 = export_metadata(sample_clip, index=1, output_dir=tmp_path)
    assert path1.name == "clip_01.txt"
    assert path2.name == "clip_02.txt"


def test_all_fields_present(sample_clip, tmp_path):
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")

    assert sample_clip.title in content
    assert sample_clip.description in content
    assert sample_clip.thumbnail_idea in content
    assert sample_clip.reason in content
    assert f"{sample_clip.viral_score}/10" in content


def test_hashtags_formatted_with_prefix(sample_clip, tmp_path):
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")

    for tag in sample_clip.hashtags:
        assert f"#{tag}" in content


def test_hashtags_already_prefixed(tmp_path):
    clip = ViralClip(
        title="Teste",
        reason="Motivo",
        viral_score=7.0,
        start_time=0.0,
        end_time=30.0,
        description="Desc",
        hashtags=["#python", "#dev"],
        thumbnail_idea="Frame inicial",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    result = export_metadata(clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")

    assert "##python" not in content
    assert "#python" in content
    assert "#dev" in content


def test_utf8_encoding(tmp_path):
    clip = ViralClip(
        title="Título com acentuação e ç",
        reason="Seleção por emoção",
        viral_score=9.0,
        start_time=0.0,
        end_time=45.0,
        description="Descrição completa com caracteres especiais: ã, ê, ó, ú.",
        hashtags=["português", "conteúdo"],
        thumbnail_idea="Apresentador com expressão de emoção",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    result = export_metadata(clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")

    assert "Título com acentuação e ç" in content
    assert "Seleção por emoção" in content
    assert "ã, ê, ó, ú" in content
    assert "#português" in content


def test_returns_path(sample_clip, tmp_path):
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    assert isinstance(result, Path)


def test_creates_output_dir_if_missing(sample_clip, tmp_path):
    nested = tmp_path / "output" / "myvideo"
    assert not nested.exists()
    export_metadata(sample_clip, index=0, output_dir=nested)
    assert nested.exists()
    assert (nested / "clip_01.txt").exists()


def test_negative_index_raises(sample_clip, tmp_path):
    with pytest.raises(ValueError, match="index must be >= 0"):
        export_metadata(sample_clip, index=-1, output_dir=tmp_path)


def test_integer_viral_score_no_decimal(tmp_path):
    clip = ViralClip(
        title="Test",
        reason="Reason",
        viral_score=7.0,
        start_time=0.0,
        end_time=30.0,
        description="Desc",
        hashtags=["test"],
        thumbnail_idea="Frame",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    result = export_metadata(clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")
    assert "7/10" in content
    assert "7.0/10" not in content


def test_viral_score_format(tmp_path):
    clip = ViralClip(
        title="Test",
        reason="Reason",
        viral_score=7.3,
        start_time=0.0,
        end_time=30.0,
        description="Desc",
        hashtags=["test"],
        thumbnail_idea="Frame",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    result = export_metadata(clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")
    assert "7.3/10" in content


def test_export_metadata_with_music_track(sample_clip, tmp_path):
    """RF-22: clip_NN.txt com MusicTrack deve conter `Fonte: YouTube (<source_url>)`."""
    track = MusicTrack(
        video_id="vidU1",
        name="Upbeat Morning",
        source_url="https://www.youtube.com/watch?v=vidU1",
        local_path=tmp_path / "vidU1.m4a",
        mood="motivacional",
        duration_s=120.0,
    )
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path, music_track=track)
    content = result.read_text(encoding="utf-8")

    assert "TRILHA SONORA" in content
    assert "Upbeat Morning" in content
    assert "Fonte: YouTube" in content
    assert "https://www.youtube.com/watch?v=vidU1" in content


def test_export_metadata_without_music_and_not_requested(sample_clip, tmp_path):
    """Sem trilha e sem `music_requested=True` → seção TRILHA SONORA fica fora."""
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path, music_track=None)
    content = result.read_text(encoding="utf-8")

    assert "TRILHA SONORA" not in content
    assert sample_clip.title in content
    assert sample_clip.description in content


def test_export_metadata_music_requested_but_no_track(sample_clip, tmp_path):
    """RF-23: música pedida (--music) mas sem trilha → registra ausência explicitamente."""
    result = export_metadata(
        sample_clip,
        index=0,
        output_dir=tmp_path,
        music_track=None,
        music_requested=True,
    )
    content = result.read_text(encoding="utf-8")

    assert "TRILHA SONORA" in content
    assert "Trilha: nenhuma" in content
    assert "youcut music sync" in content


def test_export_metadata_existing_tests_still_pass(sample_clip, tmp_path):
    """Compatibilidade: chamada sem novos parâmetros continua válida."""
    result = export_metadata(sample_clip, index=0, output_dir=tmp_path)
    content = result.read_text(encoding="utf-8")

    assert sample_clip.title in content
    assert "TRILHA SONORA" not in content
