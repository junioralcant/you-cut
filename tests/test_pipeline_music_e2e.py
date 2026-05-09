"""Teste de integração end-to-end: run_flow_c com FFmpeg real e mocks de IO/IA.

Verifica que um clipe gerado por `run_flow_c` (mode=social, --music) recebe
trilha real do acervo (mock yt-dlp) e que o `clip_01.txt` exportado cita
`Fonte: YouTube`.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(autouse=True)
def _skip_if_no_ffmpeg():
    if not _ffmpeg_available():
        pytest.skip("FFmpeg/ffprobe não estão no PATH")


def _make_synthetic_clip(dest: Path, duration: float = 8.0) -> Path:
    """Gera um clipe MP4 sintético com vídeo + áudio de fala (sine 440Hz)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=25",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(dest),
        ],
        check=True, capture_output=True,
    )
    return dest


def _make_synthetic_audio(dest: Path, duration: float = 30.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
            "-c:a", "aac",
            str(dest),
        ],
        check=True, capture_output=True,
    )
    return dest


def test_run_flow_c_with_music_writes_youtube_source_in_clip_txt(tmp_path):
    from youcut.config import PipelineConfig
    from youcut.models import MusicTrack, ViralClip
    from youcut.music.library import MusicLibrary

    # Acervo real com 1 faixa real (m4a sintético)
    music_root = tmp_path / "music"
    track_audio = _make_synthetic_audio(music_root / "tracks" / "vidE1.m4a")
    library = MusicLibrary(root=music_root)
    library.add(
        MusicTrack(
            video_id="vidE1",
            name="E2E Track",
            source_url="https://www.youtube.com/watch?v=vidE1",
            local_path=track_audio,
            mood="motivacional",
            duration_s=30.0,
        )
    )
    library.save()

    # Clipe sintético (vídeo + áudio)
    output_dir = tmp_path / "out"
    clip_path = _make_synthetic_clip(output_dir / "vid_e2e" / "clip_01.mp4")

    clip = ViralClip(
        title="Motivação para superar desafios",
        reason="Inspirador e direto",
        viral_score=8.0,
        start_time=0.0,
        end_time=8.0,
        description="Descrição",
        hashtags=["#motivacao"],
        thumbnail_idea="frame inicial",
        social_visual_style="claro",
    )

    config = PipelineConfig(
        anthropic_api_key="test-key",
        cut_mode="social",
        social_layout_mode="classic",
        face_tracking=False,
        output_dir=output_dir,
    )

    with (
        patch("youcut.cli.download_video", return_value=clip_path),
        patch("youcut.cli.transcribe", return_value=MagicMock(segments=[])),
        patch("youcut.cli.analyze", return_value=[clip]),
        patch("youcut.cli.cut_clip", return_value=clip_path),
        patch("youcut.cli._extract_cut_result", return_value=(clip_path, True, None)),
        patch("youcut.cli._is_editorial_social_layout", return_value=False),
        patch("youcut.cli.MusicLibrary", return_value=library),
        patch("youcut.cli.review_clips", return_value=[]),
        patch("youcut.cli._resolve_upload_platforms", return_value=[]),
        patch("youcut.cli._show_records_table"),
    ):
        from youcut.cli import run_flow_c
        run_flow_c(
            "https://example.com/video",
            config,
            skip_review=True,
            music=True,
        )

    # run_flow_c grava em config.output_dir / video_path.stem / clip_NN.txt
    txt_path = output_dir / clip_path.stem / "clip_01.txt"
    assert txt_path.exists(), f"clip_NN.txt não foi gerado em {txt_path}"
    content = txt_path.read_text(encoding="utf-8")
    assert "TRILHA SONORA" in content
    assert "Fonte: YouTube" in content
    assert "https://www.youtube.com/watch?v=vidE1" in content

    # MP4 final tem áudio (combinado pelo MusicMixer real, sem mock)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip_path),
        ],
        capture_output=True, text=True,
    )
    assert "audio" in probe.stdout
