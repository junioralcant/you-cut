import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from youcut.comic.visual_analyzer import (
    _assemble_cast,
    _generic_fallback_cast,
    _map_speakers_to_persons,
    _slugify,
    _truncate_transcript,
    detect_cast,
)
from youcut.models import (
    SpeakerSegment,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)


_FIXTURES_DIR = Path(__file__).parent / "_fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _png_bytes(color: tuple[int, int, int] = (10, 30, 60)) -> bytes:
    img = Image.new("RGB", (320, 240), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_samples(face_counts: list[int]) -> list[dict]:
    out: list[dict] = []
    for i, n in enumerate(face_counts):
        out.append(
            {
                "timestamp": float(i),
                "png_bytes": _png_bytes(),
                "faces": [(10 + j * 50, 20, 80, 80) for j in range(n)],
            }
        )
    return out


@pytest.fixture
def transcription():
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=4.0,
                text="Oi pessoal, hoje eu vou contar uma história.",
                words=[WordTimestamp(word="Oi", start=0.0, end=0.5)],
            ),
            TranscriptionSegment(
                start=4.0,
                end=8.0,
                text="Foi numa terça-feira de manhã.",
                words=[WordTimestamp(word="Foi", start=4.0, end=4.5)],
            ),
        ],
        language="pt",
        source_path=Path("video.mp4"),
    )


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def _tool_response(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="extract_cast", input=payload)
    return SimpleNamespace(content=[block])


def _client_returning(payload: dict) -> MagicMock:
    client = MagicMock()
    response = _tool_response(payload)
    client.with_options.return_value.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_slugify_normalizes_to_lowercase_underscore():
    assert _slugify("Cachorro do Narrador") == "cachorro_do_narrador"
    assert _slugify("Person  --  ONE") == "person_one"
    assert _slugify("") == "x"


def test_truncate_transcript_keeps_full_when_under_limit(transcription):
    text = _truncate_transcript(transcription)
    assert "história" in text


def test_truncate_transcript_truncates_when_over_limit():
    big = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=1.0,
                text="x" * 5000,
                words=[],
            )
        ],
        language="pt",
        source_path=Path("v.mp4"),
    )
    out = _truncate_transcript(big, max_chars=100)
    assert out.endswith("…")
    assert len(out) <= 110


def test_map_speakers_to_persons_two_speakers_two_persons():
    persons = [
        {"spatial_position": "right"},
        {"spatial_position": "left"},
    ]
    speakers = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerSegment(speaker_id="SPEAKER_01", start=2.0, end=4.0),
    ]
    mapping = _map_speakers_to_persons(persons, speakers)
    assert mapping == {1: "SPEAKER_00", 0: "SPEAKER_01"}


def test_map_speakers_returns_empty_when_more_than_two_persons():
    persons = [{"spatial_position": "left"}] * 3
    speakers = [SpeakerSegment(speaker_id="S0", start=0.0, end=1.0)]
    assert _map_speakers_to_persons(persons, speakers) == {}


def test_map_speakers_returns_empty_when_more_than_two_speakers():
    persons = [{"spatial_position": "left"}, {"spatial_position": "right"}]
    speakers = [
        SpeakerSegment(speaker_id=f"S{i}", start=float(i), end=float(i + 1)) for i in range(3)
    ]
    assert _map_speakers_to_persons(persons, speakers) == {}


# ---------------------------------------------------------------------------
# Cast assembly
# ---------------------------------------------------------------------------


def test_assemble_cast_one_person(transcription):
    raw = _load_fixture("comic_vision_one_person.json")["characters"]
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=8.0)]
    cast = _assemble_cast(raw, speakers)

    assert len(cast) == 1
    member = cast[0]
    assert member.kind == "person"
    assert member.gender_apparent == "feminino"
    assert "castanho" in member.hair
    assert member.speaker_id == "SPEAKER_00"
    assert member.character_id == "narradora_principal"
    assert "feminino" in member.text_card


def test_assemble_cast_two_persons_maps_by_spatial_position():
    raw = _load_fixture("comic_vision_two_persons.json")["characters"]
    speakers = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerSegment(speaker_id="SPEAKER_01", start=2.0, end=4.0),
    ]
    cast = _assemble_cast(raw, speakers)

    assert len(cast) == 2
    left_person = next(m for m in cast if m.narrative_role == "entrevistador")
    right_person = next(m for m in cast if m.narrative_role == "convidada")
    assert left_person.speaker_id == "SPEAKER_00"
    assert right_person.speaker_id == "SPEAKER_01"


def test_assemble_cast_animal_does_not_get_speaker():
    raw = _load_fixture("comic_vision_animal_only.json")["characters"]
    speakers = [SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=8.0)]
    cast = _assemble_cast(raw, speakers)

    assert len(cast) == 1
    assert cast[0].kind == "animal"
    assert cast[0].speaker_id is None


def test_assemble_cast_handles_duplicate_roles():
    raw = [
        {"kind": "person", "narrative_role": "amigo"},
        {"kind": "person", "narrative_role": "amigo"},
    ]
    cast = _assemble_cast(raw, [])
    assert {m.character_id for m in cast} == {"amigo", "amigo_2"}


# ---------------------------------------------------------------------------
# detect_cast — public API
# ---------------------------------------------------------------------------


def test_detect_cast_returns_one_person(transcription, config):
    samples = _make_samples([1, 1, 1, 1])
    payload = _load_fixture("comic_vision_one_person.json")
    client = _client_returning(payload)

    cast = detect_cast(
        Path("video.mp4"),
        transcription,
        speakers=[SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=8.0)],
        config=config,
        client=client,
        samples=samples,
    )

    assert len(cast) == 1
    assert cast[0].kind == "person"
    assert cast[0].speaker_id == "SPEAKER_00"
    assert client.with_options.return_value.messages.create.called


def test_detect_cast_two_persons_mapped(transcription, config):
    samples = _make_samples([2, 2, 2])
    payload = _load_fixture("comic_vision_two_persons.json")
    client = _client_returning(payload)

    speakers = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=4.0),
        SpeakerSegment(speaker_id="SPEAKER_01", start=4.0, end=8.0),
    ]
    cast = detect_cast(
        Path("video.mp4"),
        transcription,
        speakers=speakers,
        config=config,
        client=client,
        samples=samples,
    )

    assert len(cast) == 2
    speaker_ids = {m.speaker_id for m in cast}
    assert speaker_ids == {"SPEAKER_00", "SPEAKER_01"}


def test_detect_cast_falls_back_when_no_faces(transcription, config, caplog):
    samples = _make_samples([0, 0, 0])
    client = MagicMock()

    with caplog.at_level(logging.WARNING):
        cast = detect_cast(
            Path("video.mp4"),
            transcription,
            speakers=[SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=8.0)],
            config=config,
            client=client,
            samples=samples,
        )

    assert len(cast) == 1
    assert cast[0].character_id == "narrator"
    assert "história" in cast[0].text_card or "áudio" in cast[0].text_card
    assert any("sem rostos" in record.message for record in caplog.records)
    client.with_options.return_value.messages.create.assert_not_called()


def test_detect_cast_falls_back_when_claude_returns_empty(transcription, config, caplog):
    samples = _make_samples([1, 1])
    client = _client_returning({"characters": []})

    with caplog.at_level(logging.WARNING):
        cast = detect_cast(
            Path("video.mp4"),
            transcription,
            speakers=[],
            config=config,
            client=client,
            samples=samples,
        )

    assert len(cast) == 1
    assert cast[0].character_id == "narrator"


def test_detect_cast_with_animal_and_person(transcription, config):
    samples = _make_samples([1, 1])
    raw = {
        "characters": [
            {
                "kind": "person",
                "narrative_role": "narrador",
                "spatial_position": "center",
            },
            {
                "kind": "animal",
                "narrative_role": "cachorro do narrador",
                "accessories": ["coleira"],
                "spatial_position": "center",
            },
        ]
    }
    client = _client_returning(raw)
    cast = detect_cast(
        Path("v.mp4"),
        transcription,
        speakers=[SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=8.0)],
        config=config,
        client=client,
        samples=samples,
    )
    assert {m.kind for m in cast} == {"person", "animal"}
    person = next(m for m in cast if m.kind == "person")
    animal = next(m for m in cast if m.kind == "animal")
    assert person.speaker_id == "SPEAKER_00"
    assert animal.speaker_id is None


def test_detect_cast_propagates_anthropic_error(transcription, config):
    samples = _make_samples([1])
    import anthropic

    client = MagicMock()
    err = anthropic.APIError(message="boom", request=MagicMock(), body=None)
    client.with_options.return_value.messages.create.side_effect = err
    with pytest.raises(RuntimeError, match=r"Erro na API do Claude"):
        detect_cast(
            Path("v.mp4"),
            transcription,
            speakers=[],
            config=config,
            client=client,
            samples=samples,
        )


def test_generic_fallback_cast_text_card_mentions_audio(transcription):
    cast = _generic_fallback_cast(transcription)
    assert len(cast) == 1
    assert "nenhum rosto" in cast[0].text_card
