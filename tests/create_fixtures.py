#!/usr/bin/env python3
"""Generate test fixtures. Run this script before executing integration tests."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURES_DIR / "sample.mp4"
SAMPLE_META = FIXTURES_DIR / "sample.meta.json"

FIXTURE_VERSION = 3
FIXTURE_DURATION_SECONDS = 30
SYSTEM_SPEECH_SOURCE = Path(
    "/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Siri_TextToSpeech/"
    "purpose_auto/2afc6d1be31d583ebf6d02cb69ebaef3e16d4361.asset/AssetData/pt-BR_nando.caf"
)


def _has_command(command: str) -> bool:
    return shutil.which(command) is not None


def _fixture_is_current() -> bool:
    if not SAMPLE_MP4.exists() or SAMPLE_MP4.stat().st_size <= 10_000:
        return False
    if not SAMPLE_META.exists():
        return False

    try:
        metadata = json.loads(SAMPLE_META.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return metadata.get("version") == FIXTURE_VERSION


def _validate_speech_source() -> None:
    if not SYSTEM_SPEECH_SOURCE.exists():
        raise RuntimeError(
            f"Áudio de voz PT-BR não encontrado em {SYSTEM_SPEECH_SOURCE}."
        )


def _mux_video_and_audio() -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "20",
        "-i",
        str(SYSTEM_SPEECH_SOURCE),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={FIXTURE_DURATION_SECONDS}",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:size=1280x720:duration={FIXTURE_DURATION_SECONDS}",
        "-filter_complex",
        (
            "[0:a]atrim=duration=30,asetpts=N/SR/TB,volume=1.8[speech];"
            "[1:a]volume=0.02[tone];"
            "[speech][tone]amix=inputs=2:duration=first[audio]"
        ),
        "-map",
        "2:v",
        "-map",
        "[audio]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        str(FIXTURE_DURATION_SECONDS),
        str(SAMPLE_MP4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _write_metadata() -> None:
    SAMPLE_META.write_text(
        json.dumps(
            {
                "version": FIXTURE_VERSION,
                "duration_seconds": FIXTURE_DURATION_SECONDS,
                "speech_source": str(SYSTEM_SPEECH_SOURCE),
                "video": "ffmpeg color blue 1280x720",
                "tone": "ffmpeg sine 440Hz at low volume",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_sample_mp4(force: bool = False) -> bool:
    if not force and _fixture_is_current():
        print(f"Fixture já existe: {SAMPLE_MP4} ({SAMPLE_MP4.stat().st_size} bytes)")
        return True

    if not _has_command("ffmpeg"):
        print("ERRO: FFmpeg não encontrado no PATH.", file=sys.stderr)
        print("Instale o FFmpeg: https://ffmpeg.org/download.html", file=sys.stderr)
        return False

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _validate_speech_source()
        _mux_video_and_audio()
        _write_metadata()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERRO ao gerar fixture:\n{exc}", file=sys.stderr)
        return False

    print(f"Fixture gerado: {SAMPLE_MP4} ({SAMPLE_MP4.stat().st_size} bytes)")
    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    ok = create_sample_mp4(force=force)
    sys.exit(0 if ok else 1)
