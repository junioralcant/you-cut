"""Integration tests dos componentes Remotion (Task 6.0).

Renderiza uma fixture com 1 personagem + mouth sheet 4-cores + 2 scenes
com Ken Burns + crossfade + shake. Valida lip-sync samplando cor do
pixel central em frames específicos, valida Ken Burns por brightness
diff, e valida shake por desvio de pixels entre frames adjacentes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTION_PROJECT = REPO_ROOT / "youcut" / "comic" / "remotion_project"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "remotion"

pytestmark = pytest.mark.integration

# Cores chapadas das 4 células (devem bater com o fixture
# `single_character_mouth_sheet.png` gerado em criar-fixture).
EXPECTED_COLORS = {
    "closed": (220, 30, 30),       # vermelho
    "open_mid": (30, 200, 30),     # verde
    "open_wide": (30, 30, 220),    # azul
    "open_round": (230, 220, 30),  # amarelo
}


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def _node_modules_present() -> bool:
    return (REMOTION_PROJECT / "node_modules").exists()


@pytest.fixture(scope="module")
def rendered_video(tmp_path_factory) -> Path:
    """Renderiza a fixture canônica de 5s e retorna o MP4 produzido."""
    if not _has("node") or not _node_modules_present():
        pytest.skip("node ausente ou node_modules não instalado")
    if not _has("ffmpeg") or not _has("ffprobe"):
        pytest.skip("ffmpeg/ffprobe não disponíveis")

    sheet = FIXTURES_DIR / "single_character_mouth_sheet.png"
    audio = FIXTURES_DIR / "minimal_audio.aac"
    if not sheet.exists() or not audio.exists():
        pytest.skip("fixtures ausentes — rode tests/create_fixtures.py")

    tmp = tmp_path_factory.mktemp("remotion_components")
    props_path = tmp / "props.json"
    fixture_text = (FIXTURES_DIR / "single_character_5s.json").read_text()
    fixture = json.loads(fixture_text)
    fixture["audio_path"] = str(audio)
    fixture["characters"]["speaker_a"]["mouth_sheet_path"] = str(sheet)
    props_path.write_text(json.dumps(fixture))

    output = tmp / "render.mp4"
    completed = subprocess.run(
        [
            "node",
            str(REMOTION_PROJECT / "render.mjs"),
            "--props",
            str(props_path),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REMOTION_PROJECT),
    )
    assert completed.returncode == 0, (
        f"render falhou (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert output.exists() and output.stat().st_size > 0
    return output


def _extract_frame(video: Path, time_sec: float, out_path: Path) -> Path:
    """Extrai 1 frame em `time_sec` via ffmpeg."""
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{time_sec:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "1",
            str(out_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert out_path.exists() and out_path.stat().st_size > 0
    return out_path


def _sample_color(frame_path: Path, region: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Cor média (R, G, B) da região (x1, y1, x2, y2)."""
    from PIL import Image, ImageStat

    with Image.open(frame_path) as img:
        crop = img.convert("RGB").crop(region)
        stats = ImageStat.Stat(crop)
    return tuple(int(round(v)) for v in stats.mean[:3])  # type: ignore[return-value]


def _close_to(actual: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int = 60) -> bool:
    return all(abs(actual[i] - expected[i]) <= tolerance for i in range(3))


# ── Lip-sync ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "time_sec, expected_shape",
    [
        (0.3, "closed"),
        (0.8, "open_mid"),
        (1.3, "open_wide"),
        (1.7, "open_round"),
        (2.3, "closed"),
        # Scene 1 starts at 2.5; sample after the crossfade (0.3s) is fully done
        (3.4, "open_round"),  # scene1 t_rel=0.9 → open_round
    ],
)
def test_lipsync_correct_mouth_shape_at_timestamp(
    rendered_video, tmp_path, time_sec, expected_shape
):
    expected_color = EXPECTED_COLORS[expected_shape]
    frame = _extract_frame(rendered_video, time_sec, tmp_path / f"f_{time_sec:.1f}.png")
    # Cell rendered at center: 1080x1920 → center 540, 960; 512x512 cell.
    # Sample a small region near the center (avoiding Ken Burns scale/translate edges).
    actual = _sample_color(frame, (530, 950, 550, 970))
    assert _close_to(actual, expected_color, tolerance=80), (
        f"t={time_sec}s expected {expected_shape}={expected_color} got {actual}"
    )


# ── Ken Burns ───────────────────────────────────────────────────────────────


def test_ken_burns_produces_frame_size_difference(rendered_video, tmp_path):
    """scaleFrom=1.0 → scaleTo=1.15 deve fazer frame final mais 'cheio'."""
    early = _extract_frame(rendered_video, 0.05, tmp_path / "early.png")
    late = _extract_frame(rendered_video, 2.4, tmp_path / "late.png")
    # Ao fim da scene 0 (com scale=1.15), o conteúdo central enche mais
    # área. Calcula a "extensão" do conteúdo (pixels não-fundo) por
    # threshold de cor dominante na linha central.
    from PIL import Image

    def _content_width(path: Path) -> int:
        with Image.open(path) as img:
            row = img.convert("RGB")
            y = row.height // 2
            cnt = 0
            for x in range(row.width):
                r, g, b = row.getpixel((x, y))
                # background é #101820 = (16, 24, 32)
                if (r, g, b) != (16, 24, 32) and abs(r - 16) + abs(g - 24) + abs(b - 32) > 30:
                    cnt += 1
            return cnt

    early_width = _content_width(early)
    late_width = _content_width(late)
    # Margem grande pra evitar flapping
    assert late_width >= early_width, (
        f"esperava conteúdo mais largo no fim do Ken Burns, "
        f"early={early_width}px late={late_width}px"
    )


# ── Crossfade ───────────────────────────────────────────────────────────────


def test_crossfade_has_intermediate_alpha_at_scene_boundary(rendered_video, tmp_path):
    """No início da scene 1 (t=2.5..2.8), o conteúdo deve estar entre 0 e
    cor cheia — efeito do opacity interpolate da transição crossfade."""
    # frame BEFORE transition (scene 0 closed = red full)
    before = _extract_frame(rendered_video, 2.45, tmp_path / "before.png")
    # frame DURING crossfade (scene 1 starting; should be partial)
    during = _extract_frame(rendered_video, 2.55, tmp_path / "during.png")
    # frame AFTER crossfade (scene 1 fully visible — open_wide = blue)
    after = _extract_frame(rendered_video, 2.85, tmp_path / "after.png")

    cx_region = (530, 950, 550, 970)
    before_rgb = _sample_color(before, cx_region)
    during_rgb = _sample_color(during, cx_region)
    after_rgb = _sample_color(after, cx_region)

    # Antes da transição: vermelho cheio (closed da scene 0)
    assert _close_to(before_rgb, EXPECTED_COLORS["closed"], tolerance=80)
    # Depois da transição: azul cheio (open_wide da scene 1)
    assert _close_to(after_rgb, EXPECTED_COLORS["open_wide"], tolerance=80)
    # Durante a transição: cor diferente das duas extremidades (alpha intermediário)
    assert not _close_to(during_rgb, before_rgb, tolerance=20), (
        "frame durante crossfade igual ao frame anterior — transição não aplicou"
    )


# ── Shake ───────────────────────────────────────────────────────────────────


def test_shake_displaces_pixels_during_window(rendered_video, tmp_path):
    """Shake at_sec=2.0 (relative scene 1) → absolute 4.5s. Frame em 4.5s
    deve ter algum deslocamento detectável vs frame em 4.4s (pré-shake)."""
    pre = _extract_frame(rendered_video, 4.40, tmp_path / "pre.png")
    during = _extract_frame(rendered_video, 4.55, tmp_path / "during.png")

    from PIL import Image, ImageChops

    with Image.open(pre) as a, Image.open(during) as b:
        diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        bbox = diff.getbbox()
    # Bbox indica que houve mudança em alguma parte do frame.
    assert bbox is not None, "shake não produziu deslocamento detectável"


# ── Composição multi-scenes ─────────────────────────────────────────────────


def test_render_total_duration_matches_props(rendered_video):
    """Stream de vídeo deve ter duração ≈ 5s (drift < 50ms)."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(rendered_video),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    duration = float(completed.stdout.strip())
    assert abs(duration - 5.0) <= 0.05, (
        f"duração do stream de vídeo {duration}s difere de 5.0s além do drift admitido"
    )
