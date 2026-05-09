"""Testes unitários do `youcut/visual_style.py`.

Cobertura:
- gate por `social_visual_style_enabled=False` → no-op silencioso (RF-19);
- gate por `cut_mode != "social"` → no-op (RF-21/22/23);
- comando ffmpeg determinístico (snapshot) quando ligado;
- `CalledProcessError` → devolve path original sem propagar exceção;
- `FileNotFoundError` → devolve path original com warning;
- `_get_video_dimensions` fallback quando ffprobe falha;
- `_build_filter_graph` é determinístico para entradas iguais.

Mocks limitados a `subprocess.run` — nenhum mock interno do módulo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youcut.visual_style import (
    _COOL_EQ,
    _CORNER_RADIUS_PX,
    _VIGNETTE_BAND_PX,
    _build_filter_graph,
    _get_video_dimensions,
    apply_visual_style,
)


@pytest.fixture
def clip_file(tmp_path: Path) -> Path:
    """Cria um arquivo dummy no tmp_path para servir de input."""
    p = tmp_path / "clip_01.mp4"
    p.write_bytes(b"\x00" * 64)
    return p


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_config(**overrides):
    from youcut.config import PipelineConfig
    base = {
        "social_visual_style_enabled": True,
        "cut_mode": "social",
    }
    base.update(overrides)
    return PipelineConfig(**base)


# ── Gates ────────────────────────────────────────────────────────────────────

def test_apply_visual_style_noop_when_flag_disabled(clip_file):
    """RF-19: flag desligada → não invoca subprocess, devolve path original."""
    config = _make_config(social_visual_style_enabled=False)
    with patch("youcut.visual_style.subprocess.run") as mock_run:
        result = apply_visual_style(clip_file, config)
    assert result == clip_file
    mock_run.assert_not_called()


@pytest.mark.parametrize("mode", ["youtube"])
def test_apply_visual_style_noop_when_cut_mode_not_social(clip_file, mode):
    """RF-21/22/23: outros modos não recebem o tratamento."""
    config = _make_config(cut_mode=mode)
    with patch("youcut.visual_style.subprocess.run") as mock_run:
        result = apply_visual_style(clip_file, config)
    assert result == clip_file
    mock_run.assert_not_called()


# ── Caminho feliz ────────────────────────────────────────────────────────────

def test_apply_visual_style_invokes_ffmpeg_with_expected_args(clip_file):
    """Caminho feliz: subprocess.run é chamado uma vez para ffmpeg + uma para ffprobe."""
    config = _make_config()

    # ffprobe responde com dimensões 1080x1920; ffmpeg retorna sucesso.
    probe_result = MagicMock(stdout="1080x1920\n", returncode=0)
    ffmpeg_result = MagicMock(returncode=0)

    with (
        patch("youcut.visual_style.subprocess.run", side_effect=[probe_result, ffmpeg_result]) as mock_run,
        patch("youcut.visual_style.tempfile.NamedTemporaryFile") as mock_tmp,
        patch.object(Path, "replace") as mock_replace,
    ):
        tmp_path = clip_file.parent / "tmp_styled.mp4"
        tmp_path.write_bytes(b"\x00")
        mock_tmp.return_value.__enter__.return_value.name = str(tmp_path)
        result = apply_visual_style(clip_file, config)

    assert result == clip_file
    # 1ª chamada: ffprobe, 2ª: ffmpeg
    assert mock_run.call_count == 2
    ffmpeg_cmd = mock_run.call_args_list[1][0][0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert "-filter_complex" in ffmpeg_cmd
    # áudio preservado conforme RF-03
    assert "-c:a" in ffmpeg_cmd
    audio_codec_idx = ffmpeg_cmd.index("-c:a")
    assert ffmpeg_cmd[audio_codec_idx + 1] == "copy"
    # vídeo recodificado em libx264 + yuv420p
    assert "libx264" in ffmpeg_cmd
    assert "yuv420p" in ffmpeg_cmd
    # rename atômico do tmp sobre o original
    mock_replace.assert_called_once_with(clip_file)


def test_apply_visual_style_filter_graph_is_deterministic(clip_file):
    """Filter graph idêntico para chamadas iguais — habilita snapshot."""
    g1 = _build_filter_graph(1080, 1920)
    g2 = _build_filter_graph(1080, 1920)
    assert g1 == g2
    # Sanity checks de constantes/filtros usados
    assert _COOL_EQ in g1
    assert str(_CORNER_RADIUS_PX) in g1
    assert "vignette" in g1


def test_apply_visual_style_filter_graph_uses_runtime_dimensions():
    """A máscara de cantos usa `W`/`H` do ffmpeg em runtime — o graph é
    independente das dimensões passadas, o que evita branches duplicados
    e simplifica o snapshot.
    """
    g_portrait = _build_filter_graph(1080, 1920)
    g_square = _build_filter_graph(1080, 1080)
    # Mesma string — ffmpeg resolve W/H em runtime.
    assert g_portrait == g_square
    # Conferir que a expressão usa W/H (não valores literais por dimensão).
    assert "W-" in g_portrait and "H-" in g_portrait


# ── Falhas do ffmpeg ─────────────────────────────────────────────────────────

def test_apply_visual_style_called_process_error_returns_original(clip_file):
    """ffmpeg falha → devolve path original, sem propagar, sem corromper input."""
    config = _make_config()

    probe_result = MagicMock(stdout="1080x1920\n", returncode=0)
    ffmpeg_err = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"boom")

    original_bytes = clip_file.read_bytes()

    with (
        patch("youcut.visual_style.subprocess.run", side_effect=[probe_result, ffmpeg_err]),
        patch("youcut.visual_style.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_path = clip_file.parent / "tmp_styled_fail.mp4"
        tmp_path.write_bytes(b"\x00")
        mock_tmp.return_value.__enter__.return_value.name = str(tmp_path)
        result = apply_visual_style(clip_file, config)

    assert result == clip_file
    assert clip_file.read_bytes() == original_bytes  # input intacto
    assert not tmp_path.exists()  # tmp limpo


def test_apply_visual_style_ffmpeg_not_in_path_returns_original(clip_file):
    """FileNotFoundError → devolve path original com warning logado."""
    config = _make_config()

    probe_result = MagicMock(stdout="1080x1920\n", returncode=0)

    with (
        patch(
            "youcut.visual_style.subprocess.run",
            side_effect=[probe_result, FileNotFoundError("ffmpeg")],
        ),
        patch("youcut.visual_style.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_path = clip_file.parent / "tmp_styled_nf.mp4"
        tmp_path.write_bytes(b"\x00")
        mock_tmp.return_value.__enter__.return_value.name = str(tmp_path)
        result = apply_visual_style(clip_file, config)

    assert result == clip_file
    assert not tmp_path.exists()


# ── ffprobe fallback ─────────────────────────────────────────────────────────

def test_get_video_dimensions_returns_probed(clip_file):
    probe_result = MagicMock(stdout="1080x1920\n", returncode=0)
    with patch("youcut.visual_style.subprocess.run", return_value=probe_result):
        w, h = _get_video_dimensions(clip_file)
    assert (w, h) == (1080, 1920)


@pytest.mark.parametrize(
    "side_effect",
    [
        subprocess.CalledProcessError(1, ["ffprobe"], stderr=b"boom"),
        FileNotFoundError("ffprobe"),
    ],
)
def test_get_video_dimensions_falls_back_on_failure(clip_file, side_effect):
    with patch("youcut.visual_style.subprocess.run", side_effect=side_effect):
        w, h = _get_video_dimensions(clip_file)
    assert (w, h) == (1080, 1920)


def test_get_video_dimensions_falls_back_on_malformed_output(clip_file):
    bad_result = MagicMock(stdout="not-a-resolution\n", returncode=0)
    with patch("youcut.visual_style.subprocess.run", return_value=bad_result):
        w, h = _get_video_dimensions(clip_file)
    assert (w, h) == (1080, 1920)


# ── Integração com ffmpeg real (marker integration) ─────────────────────────
#
# Testes deste bloco geram vídeo sintético via `ffmpeg -f lavfi`, executam
# `apply_visual_style` de verdade e validam pixels e propriedades do output.
# Todos pulam graciosamente quando o `ffmpeg` não estiver disponível.


def _ffmpeg_available() -> bool:
    import shutil as _shutil
    return _shutil.which("ffmpeg") is not None and _shutil.which("ffprobe") is not None


def _make_synthetic_clip(out_path: Path, *, color: str = "red", w: int = 320, h: int = 568, duration: int = 2) -> None:
    """Gera vídeo sintético colorido + áudio sine via ffmpeg lavfi.

    Resolução baixa (320x568, mantendo proporção 9:16) para acelerar — ainda
    valida o filter graph completo (cor + cantos + vinheta).
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:d={duration}:r=24",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def _probe_duration(clip: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _probe_audio_streams(clip: Path) -> str:
    """Retorna `codec_name,sample_rate,channels` do primeiro stream de áudio."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "csv=p=0",
            str(clip),
        ],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _extract_pixel_rgb(clip: Path, x: int, y: int, *, frame_time: float = 0.5) -> tuple[int, int, int]:
    """Extrai um único pixel `(R, G, B)` de um frame em `frame_time`."""
    from PIL import Image
    import io
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", str(frame_time),
            "-i", str(clip),
            "-vframes", "1",
            "-f", "image2", "-c:v", "png",
            "-",
        ],
        check=True, capture_output=True,
    )
    img = Image.open(io.BytesIO(out.stdout)).convert("RGB")
    return img.getpixel((x, y))


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_preserves_duration_and_audio(tmp_path):
    """RF-03: tratamento não altera duração nem áudio do clipe."""
    clip = tmp_path / "synth.mp4"
    _make_synthetic_clip(clip)
    dur_before = _probe_duration(clip)
    audio_before = _probe_audio_streams(clip)

    config = _make_config()
    result = apply_visual_style(clip, config)

    assert result.exists()
    dur_after = _probe_duration(result)
    audio_after = _probe_audio_streams(result)

    assert abs(dur_after - dur_before) < 0.05, (
        f"duração mudou: {dur_before:.3f}s → {dur_after:.3f}s"
    )
    assert audio_before == audio_after, (
        f"áudio re-encodado: antes={audio_before!r} depois={audio_after!r}"
    )


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_corner_pixel_is_black(tmp_path):
    """RF-08: cantos arredondados — pixel (0,0) é preto sólido."""
    clip = tmp_path / "synth.mp4"
    _make_synthetic_clip(clip, color="red")  # vídeo cru é vermelho — cantos NÃO devem ficar vermelhos

    config = _make_config()
    result = apply_visual_style(clip, config)
    assert result.exists()

    # Os 4 cantos do canvas devem ser pretos sólidos após o tratamento.
    w, h = 320, 568
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        r, g, b = _extract_pixel_rgb(result, x, y)
        assert (r, g, b) <= (15, 15, 15), (
            f"canto ({x},{y}) não é preto sólido: rgb=({r},{g},{b})"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_center_pixel_preserved(tmp_path):
    """RF-12: centro do canvas mantém luminosidade próxima ao original."""
    clip = tmp_path / "synth.mp4"
    _make_synthetic_clip(clip, color="red")
    w, h = 320, 568

    r0, g0, b0 = _extract_pixel_rgb(clip, w // 2, h // 2)

    config = _make_config()
    result = apply_visual_style(clip, config)
    r1, g1, b1 = _extract_pixel_rgb(result, w // 2, h // 2)

    # O filtro frio reduz saturação e desloca gama R/B; uma tolerância larga
    # garante que o centro NÃO fica preto/escurecido pela vinheta.
    lum_before = 0.299 * r0 + 0.587 * g0 + 0.114 * b0
    lum_after = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
    assert lum_after > lum_before * 0.5, (
        f"centro escurecido demais: lum_before={lum_before:.1f} → lum_after={lum_after:.1f}"
    )


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_disabled_keeps_raw_clip(tmp_path):
    """RF-19: com flag desligada, clipe sai cru — cantos não ficam pretos."""
    clip = tmp_path / "synth.mp4"
    _make_synthetic_clip(clip, color="red")

    original_size = clip.stat().st_size
    config = _make_config(social_visual_style_enabled=False)
    result = apply_visual_style(clip, config)

    assert result == clip
    assert clip.stat().st_size == original_size  # arquivo intocado
    # canto continua vermelho (não houve máscara de preto)
    r, g, b = _extract_pixel_rgb(result, 0, 0)
    assert r > 100, f"canto (0,0) deveria estar vermelho cru, ficou rgb=({r},{g},{b})"


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_skipped_for_youtube_mode(tmp_path):
    """RF-21: modo youtube não recebe tratamento (gate interno do módulo)."""
    clip = tmp_path / "synth.mp4"
    _make_synthetic_clip(clip, color="red")

    config = _make_config(cut_mode="youtube")
    result = apply_visual_style(clip, config)

    assert result == clip
    r, g, b = _extract_pixel_rgb(result, 0, 0)
    assert r > 100, f"clipe youtube deveria sair cru, canto rgb=({r},{g},{b})"


@pytest.mark.integration
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg não disponível")
def test_apply_visual_style_handles_ffmpeg_failure_gracefully(tmp_path):
    """Falha real do ffmpeg (input inválido) → devolve path original sem corromper."""
    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"not-a-real-video")
    original_bytes = clip.read_bytes()

    config = _make_config()
    result = apply_visual_style(clip, config)

    assert result == clip
    assert clip.read_bytes() == original_bytes
    # nenhum tmp_*.mp4 sobrando no diretório
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("tmp") and p.suffix == ".mp4"]
    assert leftovers == []
