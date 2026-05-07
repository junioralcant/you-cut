"""Testes para MusicMixer — mixagem de trilha sonora via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from youcut.config import PipelineConfig
from youcut.models import MusicTrack
from youcut.music_mixer import MusicMixer


def _make_config(**kwargs) -> PipelineConfig:
    defaults = {
        "anthropic_api_key": "test-key",
        "music_volume": 0.25,
        "music_duck_threshold": 0.015,
        "music_duck_ratio": 6.0,
        "music_duck_attack_ms": 200.0,
        "music_duck_release_ms": 1000.0,
    }
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _make_track(tmp_path: Path) -> MusicTrack:
    mp3 = tmp_path / "test.mp3"
    mp3.write_bytes(b"fake-mp3")
    return MusicTrack(
        name="Test Track",
        pixabay_url="https://pixabay.com/music/test/",
        local_path=mp3,
        mood="motivacional",
        duration_s=60.0,
    )


def _make_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip_01.mp4"
    clip.write_bytes(b"fake-video")
    return clip


class TestMixCommand:
    def _capture_ffmpeg_cmd(self, tmp_path: Path) -> list[str]:
        """Executa mix() com subprocess mockado e retorna o comando capturado."""
        clip = _make_clip(tmp_path)
        track = _make_track(tmp_path)
        config = _make_config()
        mixer = MusicMixer()

        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            # Criar arquivo temp esperado pelo mixer para simular saída ffmpeg
            tmp_files = [
                f for f in tmp_path.iterdir()
                if f.suffix == ".mp4" and f != clip
            ]
            if tmp_files:
                tmp_files[0].write_bytes(b"mixed-video")
            return MagicMock(returncode=0)

        with patch("youcut.music_mixer.subprocess.run", side_effect=fake_run):
            # Também mock o ffprobe (primeira chamada para duração)
            with patch.object(mixer, "_get_clip_duration", return_value=10.0):
                mixer.mix(clip, track, config)

        return captured_cmd

    def test_mix_command_contains_sidechaincompress(self, tmp_path):
        cmd = self._capture_ffmpeg_cmd(tmp_path)
        cmd_str = " ".join(cmd)
        assert "sidechaincompress" in cmd_str

    def test_mix_command_contains_copy_video(self, tmp_path):
        cmd = self._capture_ffmpeg_cmd(tmp_path)
        assert "-c:v" in cmd
        v_idx = cmd.index("-c:v")
        assert cmd[v_idx + 1] == "copy"

    def test_mix_command_contains_aloop(self, tmp_path):
        cmd = self._capture_ffmpeg_cmd(tmp_path)
        cmd_str = " ".join(cmd)
        assert "aloop" in cmd_str

    def test_mix_command_contains_afade(self, tmp_path):
        cmd = self._capture_ffmpeg_cmd(tmp_path)
        cmd_str = " ".join(cmd)
        assert "afade" in cmd_str

    def test_mix_command_contains_amix(self, tmp_path):
        cmd = self._capture_ffmpeg_cmd(tmp_path)
        cmd_str = " ".join(cmd)
        assert "amix" in cmd_str


class TestMixFallback:
    def test_mix_fallback_on_ffmpeg_error(self, tmp_path):
        """subprocess levanta CalledProcessError → retorna clip_path original."""
        clip = _make_clip(tmp_path)
        track = _make_track(tmp_path)
        config = _make_config()
        mixer = MusicMixer()

        original_bytes = clip.read_bytes()

        with patch("youcut.music_mixer.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["ffmpeg"], stderr=b"erro ffmpeg"
            )
            with patch.object(mixer, "_get_clip_duration", return_value=10.0):
                result = mixer.mix(clip, track, config)

        assert result == clip
        # Arquivo original preservado
        assert clip.read_bytes() == original_bytes


@pytest.mark.integration
def test_mix_integration(tmp_path):
    """Gera clipe sintético de 10s + MP3 de 5s e verifica que output existe com áudio."""
    clip = tmp_path / "clip_01.mp4"
    mp3 = tmp_path / "track.mp3"

    # Gerar vídeo sintético de 10s com áudio de fala (lavfi)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-c:v", "libx264", "-c:a", "aac",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )

    # Gerar MP3 sintético de 5s
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=5",
            str(mp3),
        ],
        check=True,
        capture_output=True,
    )

    track = MusicTrack(
        name="Test Sine",
        pixabay_url="https://pixabay.com/music/test/",
        local_path=mp3,
        mood="motivacional",
        duration_s=5.0,
    )

    config = PipelineConfig(anthropic_api_key="test-key")
    mixer = MusicMixer()
    result = mixer.mix(clip, track, config)

    assert result.exists()
    assert result.stat().st_size > 0

    # Verificar que output tem áudio
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert "audio" in probe.stdout

    # Verificar que a duração do output é igual ao clipe original (~10s)
    probe_dur = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(result),
        ],
        capture_output=True,
        text=True,
    )
    assert abs(float(probe_dur.stdout.strip()) - 10.0) < 0.5
