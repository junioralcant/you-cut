import pytest

from youcut.config import PipelineConfig
from youcut.models import CropRegion, FaceTrackingResult, SpeakerSegment


@pytest.fixture
def base_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FACE_TRACKING", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("FACE_DETECTION_CONFIDENCE", raising=False)


def test_pipeline_config_face_tracking_defaults(base_env):
    config = PipelineConfig()
    assert config.face_tracking is False
    assert config.huggingface_token is None
    assert config.face_detection_confidence == 0.5


def test_pipeline_config_face_tracking_custom_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("FACE_TRACKING", "true")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "tok")
    monkeypatch.setenv("FACE_DETECTION_CONFIDENCE", "0.7")
    config = PipelineConfig()
    assert config.face_tracking is True
    assert config.huggingface_token == "tok"
    assert config.face_detection_confidence == 0.7


def test_speaker_segment_fields():
    seg = SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=1.5)
    assert seg.speaker_id == "SPEAKER_00"
    assert seg.start == 0.0
    assert seg.end == 1.5


def test_crop_region_fields():
    region = CropRegion(x=0, y=0, w=100, h=100)
    assert region.x == 0
    assert region.y == 0
    assert region.w == 100
    assert region.h == 100


def test_face_tracking_result_empty():
    result = FaceTrackingResult(
        frame_regions=[],
        had_faces=False,
        is_split_screen=[],
        secondary_regions=[],
    )
    assert result.had_faces is False
    assert result.frame_regions == []
    assert result.is_split_screen == []
    assert result.secondary_regions == []


def test_face_tracking_result_with_none_regions():
    result = FaceTrackingResult(
        frame_regions=[CropRegion(x=10, y=20, w=50, h=50), None],
        had_faces=True,
        is_split_screen=[False, False],
        secondary_regions=[None, None],
    )
    assert result.had_faces is True
    assert result.frame_regions[0] == CropRegion(x=10, y=20, w=50, h=50)
    assert result.frame_regions[1] is None
