from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import (
    CastMember,
    MotionComicSession,
    Panel,
    PanelRenderResult,
)


def test_cast_member_minimal_creation():
    cast = CastMember(character_id="person_1")
    assert cast.character_id == "person_1"
    assert cast.kind == "person"
    assert cast.accessories == []
    assert cast.speaker_id is None
    assert cast.anchor_image_path is None
    assert cast.text_card == ""


def test_cast_member_full_fields():
    cast = CastMember(
        character_id="speaker_a",
        kind="person",
        gender_apparent="masculino",
        age_apparent="adulto",
        hair="cabelo curto preto",
        facial_hair="barba média",
        skin="pele clara",
        clothing="camiseta verde",
        accessories=["óculos", "boné"],
        narrative_role="protagonista",
        speaker_id="SPEAKER_00",
        anchor_image_path=Path("output/test/cast/speaker_a.png"),
        text_card="Homem adulto, cabelo curto, barba média, óculos, camiseta verde.",
    )
    assert cast.gender_apparent == "masculino"
    assert cast.accessories == ["óculos", "boné"]
    assert cast.speaker_id == "SPEAKER_00"
    assert cast.anchor_image_path == Path("output/test/cast/speaker_a.png")


def test_cast_member_animal_kind():
    cast = CastMember(character_id="dog_1", kind="animal", narrative_role="cachorro do narrador")
    assert cast.kind == "animal"


def test_cast_member_object_kind():
    cast = CastMember(character_id="moto_1", kind="object", narrative_role="moto vermelha")
    assert cast.kind == "object"


def test_cast_member_invalid_kind_rejected():
    with pytest.raises(ValidationError):
        CastMember(character_id="x", kind="alien")  # type: ignore[arg-type]


def test_panel_minimal_creation():
    panel = Panel(
        index=0,
        start_time=0.0,
        end_time=3.0,
        participants=["person_1"],
        framing="close",
        scene="cozinha de casa",
        pose_description="rindo, olhar surpreso",
        panel_seconds_target=3.0,
    )
    assert panel.index == 0
    assert panel.framing == "close"
    assert panel.participants == ["person_1"]


def test_panel_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Panel(
            index=1,
            start_time=5.0,
            end_time=4.0,
            participants=["a"],
            framing="medium",
            scene="rua",
            pose_description="andando",
            panel_seconds_target=3.0,
        )


def test_panel_rejects_zero_duration():
    with pytest.raises(ValidationError):
        Panel(
            index=2,
            start_time=2.0,
            end_time=2.0,
            participants=["a"],
            framing="medium",
            scene="rua",
            pose_description="andando",
            panel_seconds_target=3.0,
        )


def test_panel_rejects_invalid_framing():
    with pytest.raises(ValidationError):
        Panel(
            index=0,
            start_time=0.0,
            end_time=3.0,
            participants=["a"],
            framing="ultra_wide",  # type: ignore[arg-type]
            scene="rua",
            pose_description="andando",
            panel_seconds_target=3.0,
        )


def test_panel_rejects_negative_seconds_target():
    with pytest.raises(ValidationError):
        Panel(
            index=0,
            start_time=0.0,
            end_time=3.0,
            participants=["a"],
            framing="close",
            scene="rua",
            pose_description="andando",
            panel_seconds_target=0.0,
        )


def test_panel_render_result_defaults():
    res = PanelRenderResult(
        panel_index=0,
        base_image_path=Path("base.png"),
        clip_path=Path("clip.mp4"),
        clip_seconds=3.0,
    )
    assert res.was_static_fallback is False
    assert res.image_attempts == 1
    assert res.i2v_attempts == 0
    assert res.cost_usd == 0.0


def test_panel_render_result_static_fallback():
    res = PanelRenderResult(
        panel_index=2,
        base_image_path=Path("base.png"),
        clip_path=Path("clip.mp4"),
        clip_seconds=4.0,
        was_static_fallback=True,
        image_attempts=1,
        i2v_attempts=2,
        cost_usd=0.04,
    )
    assert res.was_static_fallback is True
    assert res.i2v_attempts == 2
    assert res.cost_usd == 0.04


def test_motion_comic_session_minimal():
    session = MotionComicSession(
        session_id="abc123",
        video_path=Path("input.mp4"),
        created_at=datetime(2025, 1, 1, 10, 0, 0),
    )
    assert session.session_id == "abc123"
    assert session.cast == []
    assert session.panels == []
    assert session.panel_results == []
    assert session.total_cost_usd == 0.0
    assert session.output_path is None


def test_motion_comic_session_full():
    cast = [CastMember(character_id="p1")]
    panels = [
        Panel(
            index=0,
            start_time=0.0,
            end_time=2.0,
            participants=["p1"],
            framing="close",
            scene="rua",
            pose_description="caminhando",
            panel_seconds_target=2.0,
        )
    ]
    results = [
        PanelRenderResult(
            panel_index=0,
            base_image_path=Path("base.png"),
            clip_path=Path("clip.mp4"),
            clip_seconds=2.0,
            cost_usd=0.10,
        )
    ]
    session = MotionComicSession(
        session_id="abc",
        video_path=Path("v.mp4"),
        created_at=datetime(2025, 1, 1),
        transcription_cache_path=Path("v_transcript.json"),
        cast=cast,
        panels=panels,
        panel_results=results,
        total_cost_usd=0.10,
        output_path=Path("output/v/motion_comic.mp4"),
    )
    assert len(session.cast) == 1
    assert len(session.panels) == 1
    assert len(session.panel_results) == 1
    assert session.total_cost_usd == 0.10
    assert session.output_path == Path("output/v/motion_comic.mp4")


def test_motion_comic_session_round_trip_json():
    session = MotionComicSession(
        session_id="rt",
        video_path=Path("v.mp4"),
        created_at=datetime(2025, 1, 1, 12, 0, 0),
    )
    payload = session.model_dump_json()
    parsed = MotionComicSession.model_validate_json(payload)
    assert parsed.session_id == "rt"
    assert parsed.video_path == Path("v.mp4")
