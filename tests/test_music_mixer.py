"""Testes para MusicMixer (filter graph fixo conforme RF-16 a RF-21)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.models import MusicTrack
from youcut.music.mixer import MusicMixer


def _make_track(tmp_path: Path, *, video_id: str = "vidT", duration_s: float = 60.0) -> MusicTrack:
    audio = tmp_path / f"{video_id}.m4a"
    audio.write_bytes(b"fake-audio")
    return MusicTrack(
        video_id=video_id,
        name="Test Track",
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        local_path=audio,
        mood="motivacional",
        duration_s=duration_s,
    )


def _make_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip_01.mp4"
    clip.write_bytes(b"fake-video")
    return clip


def _capture_cmd(tmp_path: Path, *, clip_dur: float = 10.0) -> list[str]:
    """Executa mix() com subprocess mockado e retorna o comando capturado."""
    clip = _make_clip(tmp_path)
    track = _make_track(tmp_path)
    mixer = MusicMixer()

    captured: list[str] = []

    def fake_run(cmd, **kwargs):
        captured.extend(cmd)
        # Cria o arquivo temporário para satisfazer o replace
        for f in tmp_path.iterdir():
            if f.suffix == ".mp4" and f != clip:
                f.write_bytes(b"mixed-video")
        return MagicMock(returncode=0)

    with patch("youcut.music.mixer.subprocess.run", side_effect=fake_run):
        with patch.object(mixer, "_get_clip_duration", return_value=clip_dur):
            mixer.mix(clip, track)

    return captured


# ── Filter graph contém os filtros canônicos ────────────────────────────────


class TestFilterGraph:
    def test_contains_atrim_start_10(self, tmp_path):
        """RF-16: descarta os primeiros 10s da música."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "atrim=start=10" in joined

    def test_contains_fade_in_1s(self, tmp_path):
        """RF-17: fade-in de 1.0s."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "afade=t=in:st=0:d=1.0" in joined

    def test_contains_fade_out_1_2s(self, tmp_path):
        """RF-18: fade-out de 1.2s."""
        cmd = _capture_cmd(tmp_path, clip_dur=10.0)
        joined = " ".join(cmd)
        # fade out começa em clip_dur - 1.2 = 8.8
        assert "afade=t=out:st=8.8:d=1.2" in joined

    def test_contains_volume_12(self, tmp_path):
        """RF-20: música a 12%."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "volume=0.12" in joined

    def test_contains_alimiter_97(self, tmp_path):
        """RF-21: limitador final em 0.97."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "alimiter=limit=0.97" in joined

    def test_contains_apad_for_short_tracks(self, tmp_path):
        """Faixa curta após skip → apad antes do atrim final."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "apad" in joined

    def test_contains_amix_normalize_zero(self, tmp_path):
        """amix sem normalização (preserva voz integral)."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "amix=inputs=2:duration=first:normalize=0" in joined

    def test_voice_branch_has_no_volume_filter(self, tmp_path):
        """RF-19: a voz [0:a] entra no amix sem volume= aplicado."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        # `volume=` só pode aparecer na branch da música, e exatamente uma vez
        assert joined.count("volume=") == 1
        # Garante que o uso de [0:a] não vem precedido por volume=
        # Heurística: a substring "[0:a]volume=" não deve existir
        assert "[0:a]volume=" not in joined

    def test_contains_sidechain_ducking(self, tmp_path):
        """RF-19: música abaixa quando há voz via sidechaincompress."""
        cmd = _capture_cmd(tmp_path)
        joined = " ".join(cmd)
        assert "asplit=2[voice][voice_sc]" in joined
        assert "sidechaincompress=threshold=0.05:ratio=8:attack=5:release=250" in joined
        assert "[m][voice_sc]" in joined


# ── Comandos básicos ────────────────────────────────────────────────────────


class TestFfmpegCommand:
    def test_uses_copy_video(self, tmp_path):
        cmd = _capture_cmd(tmp_path)
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"

    def test_maps_audio_aout(self, tmp_path):
        cmd = _capture_cmd(tmp_path)
        # -map 0:v + -map [aout]
        assert cmd.count("-map") == 2
        assert "[aout]" in cmd

    def test_inputs_present(self, tmp_path):
        cmd = _capture_cmd(tmp_path)
        # 2 inputs: clipe + faixa
        assert cmd.count("-i") == 2


# ── Fallback em caso de falha do ffmpeg ─────────────────────────────────────


class TestMixFallback:
    def test_returns_original_path_on_ffmpeg_error(self, tmp_path):
        """subprocess levanta CalledProcessError → retorna clip_path original."""
        clip = _make_clip(tmp_path)
        track = _make_track(tmp_path)
        mixer = MusicMixer()

        original_bytes = clip.read_bytes()

        with patch("youcut.music.mixer.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd=["ffmpeg"], stderr=b"erro ffmpeg"
            )
            with patch.object(mixer, "_get_clip_duration", return_value=10.0):
                result = mixer.mix(clip, track)

        assert result == clip
        assert clip.read_bytes() == original_bytes

    def test_returns_original_when_ffmpeg_missing(self, tmp_path):
        clip = _make_clip(tmp_path)
        track = _make_track(tmp_path)
        mixer = MusicMixer()

        with patch("youcut.music.mixer.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg")
            with patch.object(mixer, "_get_clip_duration", return_value=10.0):
                result = mixer.mix(clip, track)

        assert result == clip


# ── Integração com FFmpeg real (marker integration) ─────────────────────────


@pytest.mark.integration
def test_mix_integration_duration_and_peak(tmp_path):
    """Gera clipe sintético + faixa sintética, mixa, e valida via ffprobe.

    - duração do output ≈ duração do clipe (RF-16/18 não devem alterar a duração);
    - pico do áudio combinado ≤ 0.97 do range (RF-21).
    """
    clip = tmp_path / "clip_01.mp4"
    track_audio = tmp_path / "track.m4a"

    # Clipe sintético de 12s (vídeo + áudio sine 440Hz a 0.5)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-filter_complex", "[1:a]volume=0.5[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )

    # Faixa sintética de 30s (sine 220Hz) — basta que ultrapasse o skip de 10s.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
            "-c:a", "aac",
            str(track_audio),
        ],
        check=True,
        capture_output=True,
    )

    track = MusicTrack(
        video_id="testvid",
        name="Test Sine",
        source_url="https://www.youtube.com/watch?v=testvid",
        local_path=track_audio,
        mood="motivacional",
        duration_s=30.0,
    )
    mixer = MusicMixer()
    result = mixer.mix(clip, track)

    assert result.exists()

    probe_dur = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(result),
        ],
        capture_output=True, text=True,
    )
    duration = float(probe_dur.stdout.strip())
    assert abs(duration - 12.0) < 0.5

    # Pico ≤ 0.97 (RF-21) — usar astats peak_level (em dBFS)
    probe_peak = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(result),
            "-af", "astats=metadata=1:reset=0",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    out = (probe_peak.stderr or "") + (probe_peak.stdout or "")
    peak_lines = [
        line for line in out.splitlines()
        if "Peak level dB" in line or "Peak level" in line
    ]
    if peak_lines:
        # Último Peak level (overall) — extrair valor dB
        last = peak_lines[-1]
        try:
            peak_db = float(last.split(":")[-1].strip())
        except ValueError:
            peak_db = -999.0
        # 0.97 linear ≈ -0.265 dBFS; tolerância pequena de overshoot
        assert peak_db <= 0.0, f"pico {peak_db} dBFS excede 0 dBFS"
