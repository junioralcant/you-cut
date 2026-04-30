import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from youcut.comic.script_planner import (
    ScriptPlanError,
    _validate_panels,
    plan_panels,
)
from youcut.models import (
    CastMember,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


_FIXTURES_DIR = Path(__file__).parent / "_fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def _transcription(duration: float, language: str = "pt") -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=duration,
                text="Falando sobre uma manhã de terça-feira, no mercado, no café e na praça.",
                words=[WordTimestamp(word="Falando", start=0.0, end=0.5)],
            )
        ],
        language=language,
        source_path=Path("v.mp4"),
    )


def _cast(*ids: str) -> list[CastMember]:
    return [
        CastMember(character_id=i, kind="person", narrative_role=i, text_card=f"ficha de {i}")
        for i in ids
    ]


def _tool_response(panels_payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="plan_panels", input=panels_payload)
    return SimpleNamespace(content=[block])


def _client_returning(*payloads: dict) -> MagicMock:
    client = MagicMock()
    client.with_options.return_value.messages.create.side_effect = [
        _tool_response(p) for p in payloads
    ]
    return client


# ---------------------------------------------------------------------------
# _validate_panels
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed_monologue():
    raw = _load_fixture("comic_planner_monologue.json")["panels"]
    panels, hint = _validate_panels(
        raw,
        audio_duration=30.0,
        cast_ids={"narrator"},
        min_p=2.0,
        max_p=5.0,
    )
    assert hint is None
    assert len(panels) == 8
    assert panels[0].start_time == 0.0
    assert panels[-1].end_time == pytest.approx(30.0, abs=0.2)


def test_validate_rejects_panel_too_long():
    raw = [
        {
            "start_time": 0.0,
            "end_time": 9.0,
            "participants": ["a"],
            "framing": "close",
            "scene": "rua",
            "pose_description": "p",
        }
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=9.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "fora do intervalo" in (hint or "")


def test_validate_rejects_overlap():
    raw = [
        {"start_time": 0.0, "end_time": 4.0, "participants": ["a"], "framing": "close", "scene": "x", "pose_description": "p"},
        {"start_time": 3.0, "end_time": 7.0, "participants": ["a"], "framing": "close", "scene": "x", "pose_description": "p"},
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=7.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "sobrepõe" in (hint or "")


def test_validate_rejects_sum_outside_tolerance():
    raw = [
        {"start_time": 0.0, "end_time": 4.0, "participants": ["a"], "framing": "close", "scene": "x", "pose_description": "p"},
        {"start_time": 4.0, "end_time": 8.0, "participants": ["a"], "framing": "close", "scene": "x", "pose_description": "p"},
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=10.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "difere do áudio" in (hint or "")


def test_validate_rejects_empty_participants():
    raw = [
        {"start_time": 0.0, "end_time": 3.0, "participants": [], "framing": "close", "scene": "x", "pose_description": "p"}
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=3.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "participants vazio" in (hint or "")


def test_validate_rejects_unknown_participant_id():
    raw = [
        {"start_time": 0.0, "end_time": 3.0, "participants": ["unknown"], "framing": "close", "scene": "x", "pose_description": "p"}
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=3.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "desconhecidos" in (hint or "")


def test_validate_rejects_first_panel_not_at_zero():
    raw = [
        {"start_time": 1.0, "end_time": 4.0, "participants": ["a"], "framing": "close", "scene": "x", "pose_description": "p"}
    ]
    panels, hint = _validate_panels(
        raw,
        audio_duration=3.0,
        cast_ids={"a"},
        min_p=2.0,
        max_p=5.0,
    )
    assert panels == []
    assert "não começa em 0" in (hint or "")


def test_validate_accepts_dialogue_two_shots():
    raw = _load_fixture("comic_planner_dialogue.json")["panels"]
    panels, hint = _validate_panels(
        raw,
        audio_duration=20.0,
        cast_ids={"entrevistador", "convidada"},
        min_p=2.0,
        max_p=5.0,
    )
    assert hint is None
    framings = [p.framing for p in panels]
    assert "two_shot" in framings
    assert framings.count("close") >= 2


def test_validate_accepts_offnarration_with_multiple_scenes():
    raw = _load_fixture("comic_planner_offnarration.json")["panels"]
    panels, hint = _validate_panels(
        raw,
        audio_duration=18.0,
        cast_ids={"narrator"},
        min_p=2.0,
        max_p=5.0,
    )
    assert hint is None
    scenes = {p.scene for p in panels}
    assert len(scenes) >= 3


# ---------------------------------------------------------------------------
# plan_panels — public API
# ---------------------------------------------------------------------------


def test_plan_panels_monologue(config):
    payload = _load_fixture("comic_planner_monologue.json")
    client = _client_returning(payload)
    panels = plan_panels(
        _transcription(30.0),
        _cast("narrator"),
        speakers=[SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=30.0)],
        config=config,
        client=client,
    )
    assert len(panels) >= 6
    assert all(p.framing in ("close", "medium", "wide", "two_shot") for p in panels)
    assert client.with_options.return_value.messages.create.call_count == 1


def test_plan_panels_dialogue_two_persons(config):
    payload = _load_fixture("comic_planner_dialogue.json")
    client = _client_returning(payload)
    panels = plan_panels(
        _transcription(20.0),
        _cast("entrevistador", "convidada"),
        speakers=[
            SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=10.0),
            SpeakerSegment(speaker_id="SPEAKER_01", start=10.0, end=20.0),
        ],
        config=config,
        client=client,
    )
    assert any(p.framing == "two_shot" for p in panels)
    closes = [p for p in panels if p.framing == "close"]
    assert len(closes) >= 2


def test_plan_panels_offnarration_multiple_scenes(config):
    payload = _load_fixture("comic_planner_offnarration.json")
    client = _client_returning(payload)
    panels = plan_panels(
        _transcription(18.0),
        _cast("narrator"),
        speakers=[],
        config=config,
        client=client,
    )
    scenes = {p.scene for p in panels}
    assert len(scenes) >= 3


def test_plan_panels_retries_with_correction(config):
    bad = {
        "panels": [
            {"start_time": 0.0, "end_time": 30.0, "participants": ["narrator"], "framing": "wide", "scene": "x", "pose_description": "p"}
        ]
    }
    good = _load_fixture("comic_planner_monologue.json")
    client = _client_returning(bad, good)

    panels = plan_panels(
        _transcription(30.0),
        _cast("narrator"),
        speakers=[],
        config=config,
        client=client,
    )
    assert len(panels) == 8
    assert client.with_options.return_value.messages.create.call_count == 2


def test_plan_panels_raises_after_two_failed_attempts(config):
    bad = {
        "panels": [
            {"start_time": 0.0, "end_time": 30.0, "participants": ["narrator"], "framing": "wide", "scene": "x", "pose_description": "p"}
        ]
    }
    client = _client_returning(bad, bad)

    with pytest.raises(ScriptPlanError, match=r"após 2 tentativas"):
        plan_panels(
            _transcription(30.0),
            _cast("narrator"),
            speakers=[],
            config=config,
            client=client,
        )


def test_plan_panels_raises_on_empty_cast(config):
    client = MagicMock()
    with pytest.raises(ScriptPlanError, match=r"Cast vazio"):
        plan_panels(
            _transcription(10.0),
            cast=[],
            speakers=[],
            config=config,
            client=client,
        )


def test_plan_panels_raises_on_empty_transcription(config):
    client = MagicMock()
    empty = TranscriptionResult(segments=[], language="pt", source_path=Path("v.mp4"))
    with pytest.raises(ScriptPlanError, match=r"Transcrição vazia"):
        plan_panels(empty, _cast("narrator"), speakers=[], config=config, client=client)


def test_plan_panels_propagates_anthropic_error(config):
    import anthropic

    client = MagicMock()
    client.with_options.return_value.messages.create.side_effect = anthropic.APIError(
        message="boom", request=MagicMock(), body=None
    )

    with pytest.raises(ScriptPlanError, match=r"Erro na API do Claude"):
        plan_panels(
            _transcription(10.0),
            _cast("narrator"),
            speakers=[],
            config=config,
            client=client,
        )


def test_plan_panels_in_30s_returns_at_least_6_panels(config):
    payload = _load_fixture("comic_planner_monologue.json")
    client = _client_returning(payload)
    panels = plan_panels(
        _transcription(30.0),
        _cast("narrator"),
        speakers=[],
        config=config,
        client=client,
    )
    assert len(panels) >= 6
    assert all(2.0 <= p.panel_seconds_target <= 5.0 for p in panels)
    total = sum(p.panel_seconds_target for p in panels)
    assert abs(total - 30.0) <= 0.2
