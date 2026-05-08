"""Testes dos modelos Pydantic do engine `remotion` (Task 1.0)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import (
    MouthEvent,
    MouthShape,
    MouthSheet,
    RemotionInputProps,
    RemotionScene,
)


# ── Enum MouthShape ──────────────────────────────────────────────────────────


def test_mouth_shape_enum_values():
    assert MouthShape.CLOSED.value == "closed"
    assert MouthShape.OPEN_MID.value == "open_mid"
    assert MouthShape.OPEN_WIDE.value == "open_wide"
    assert MouthShape.OPEN_ROUND.value == "open_round"


def test_mouth_shape_enum_is_str():
    """MouthShape herda de str para serialização limpa em JSON."""
    assert isinstance(MouthShape.CLOSED, str)
    assert MouthShape.CLOSED == "closed"


# ── MouthSheet ───────────────────────────────────────────────────────────────


def test_mouth_sheet_minimal_required_shapes():
    sheet = MouthSheet(
        character_id="speaker_a",
        sheet_path=Path("output/test/mouths/speaker_a.png"),
        cells={
            MouthShape.CLOSED: (0, 0, 512, 512),
            MouthShape.OPEN_MID: (512, 0, 1024, 512),
            MouthShape.OPEN_WIDE: (1024, 0, 1536, 512),
        },
    )
    assert sheet.character_id == "speaker_a"
    assert sheet.cells[MouthShape.CLOSED] == (0, 0, 512, 512)


def test_mouth_sheet_with_optional_open_round():
    sheet = MouthSheet(
        character_id="speaker_b",
        sheet_path=Path("output/test/mouths/speaker_b.png"),
        cells={
            MouthShape.CLOSED: (0, 0, 512, 512),
            MouthShape.OPEN_MID: (512, 0, 1024, 512),
            MouthShape.OPEN_WIDE: (1024, 0, 1536, 512),
            MouthShape.OPEN_ROUND: (1536, 0, 2048, 512),
        },
    )
    assert MouthShape.OPEN_ROUND in sheet.cells


def test_mouth_sheet_rejects_missing_required_shape():
    with pytest.raises(ValidationError) as exc_info:
        MouthSheet(
            character_id="x",
            sheet_path=Path("x.png"),
            cells={
                MouthShape.CLOSED: (0, 0, 100, 100),
                MouthShape.OPEN_MID: (100, 0, 200, 100),
            },
        )
    assert "open_wide" in str(exc_info.value)


def test_mouth_sheet_rejects_invalid_box():
    with pytest.raises(ValidationError):
        MouthSheet(
            character_id="x",
            sheet_path=Path("x.png"),
            cells={
                MouthShape.CLOSED: (10, 10, 5, 5),  # x2<x1
                MouthShape.OPEN_MID: (100, 0, 200, 100),
                MouthShape.OPEN_WIDE: (200, 0, 300, 100),
            },
        )


# ── MouthEvent ───────────────────────────────────────────────────────────────


def test_mouth_event_basic():
    ev = MouthEvent(
        character_id="speaker_a",
        start_sec=1.5,
        end_sec=1.8,
        shape=MouthShape.OPEN_WIDE,
    )
    assert ev.shape == MouthShape.OPEN_WIDE
    assert ev.start_sec == 1.5


def test_mouth_event_accepts_zero_duration_silence():
    """Silêncios pontuais são representados como eventos com duração ≥0."""
    ev = MouthEvent(
        character_id="speaker_a",
        start_sec=2.0,
        end_sec=2.0,
        shape=MouthShape.CLOSED,
    )
    assert ev.shape == MouthShape.CLOSED


def test_mouth_event_rejects_end_before_start():
    with pytest.raises(ValidationError):
        MouthEvent(
            character_id="x",
            start_sec=2.0,
            end_sec=1.0,
            shape=MouthShape.CLOSED,
        )


def test_mouth_event_accepts_string_shape_coercion():
    ev = MouthEvent(
        character_id="x",
        start_sec=0.0,
        end_sec=0.5,
        shape="closed",  # type: ignore[arg-type]
    )
    assert ev.shape == MouthShape.CLOSED


# ── RemotionScene ────────────────────────────────────────────────────────────


def test_remotion_scene_minimal():
    scene = RemotionScene(
        index=0,
        start_sec=0.0,
        end_sec=3.0,
        character_ids=["speaker_a"],
    )
    assert scene.speaker_id is None
    assert scene.transition_in == "crossfade"
    assert scene.shakes == []
    assert scene.lip_sync == []
    assert scene.ken_burns == {}


def test_remotion_scene_full_fields():
    scene = RemotionScene(
        index=2,
        start_sec=10.0,
        end_sec=15.0,
        character_ids=["speaker_a", "speaker_b"],
        speaker_id="speaker_a",
        ken_burns={"scale_from": 1.0, "scale_to": 1.12, "from": (0, 0), "to": (10, 10)},
        transition_in="cut",
        shakes=[{"at_sec": 11.0, "intensity": 0.7}],
        lip_sync=[
            MouthEvent(
                character_id="speaker_a",
                start_sec=10.0,
                end_sec=10.3,
                shape=MouthShape.OPEN_WIDE,
            )
        ],
    )
    assert scene.transition_in == "cut"
    assert len(scene.lip_sync) == 1


def test_remotion_scene_rejects_invalid_transition():
    with pytest.raises(ValidationError):
        RemotionScene(
            index=0,
            start_sec=0.0,
            end_sec=2.0,
            character_ids=["a"],
            transition_in="fade_to_black",  # type: ignore[arg-type]
        )


def test_remotion_scene_rejects_end_not_after_start():
    with pytest.raises(ValidationError):
        RemotionScene(
            index=0,
            start_sec=2.0,
            end_sec=2.0,
            character_ids=["a"],
        )


# ── RemotionInputProps ───────────────────────────────────────────────────────


def test_remotion_input_props_defaults():
    props = RemotionInputProps(
        audio_path="/tmp/audio.wav",
        duration_sec=12.5,
    )
    assert props.fps == 30
    assert props.width == 1080
    assert props.height == 1920
    assert props.background_color == "#000000"
    assert props.characters == {}
    assert props.scenes == []


def test_remotion_input_props_full():
    scene = RemotionScene(
        index=0,
        start_sec=0.0,
        end_sec=2.0,
        character_ids=["speaker_a"],
    )
    props = RemotionInputProps(
        audio_path="/tmp/audio.wav",
        duration_sec=2.0,
        fps=24,
        width=720,
        height=1280,
        characters={
            "speaker_a": {
                "anchor_path": "/tmp/a.png",
                "mouth_sheet_path": "/tmp/mouth_a.png",
            }
        },
        scenes=[scene],
        background_color="#101010",
    )
    assert props.fps == 24
    assert "speaker_a" in props.characters
    assert len(props.scenes) == 1


def test_remotion_input_props_rejects_zero_duration():
    with pytest.raises(ValidationError):
        RemotionInputProps(audio_path="/tmp/a.wav", duration_sec=0.0)


def test_remotion_input_props_rejects_zero_fps():
    with pytest.raises(ValidationError):
        RemotionInputProps(audio_path="/tmp/a.wav", duration_sec=2.0, fps=0)


def test_remotion_input_props_round_trip_json():
    props = RemotionInputProps(
        audio_path="/tmp/a.wav",
        duration_sec=5.0,
        scenes=[
            RemotionScene(
                index=0,
                start_sec=0.0,
                end_sec=5.0,
                character_ids=["x"],
                lip_sync=[
                    MouthEvent(
                        character_id="x",
                        start_sec=0.0,
                        end_sec=0.4,
                        shape=MouthShape.OPEN_WIDE,
                    )
                ],
            )
        ],
    )
    payload = props.model_dump_json()
    parsed = RemotionInputProps.model_validate_json(payload)
    assert parsed.audio_path == "/tmp/a.wav"
    assert parsed.scenes[0].lip_sync[0].shape == MouthShape.OPEN_WIDE
