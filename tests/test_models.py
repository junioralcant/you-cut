from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from youcut.models import (
    ClipRecord,
    SessionData,
    ThumbnailFrameResult,
    TranscriptionResult,
    TranscriptionSegment,
    VideoMetadata,
    ViralClip,
    WordTimestamp,
)
from youcut.uploader.base import UploadResult


def test_viral_clip_creation():
    clip = ViralClip(
        title="Momento Incrível",
        reason="Gancho forte no início com curiosidade",
        viral_score=9.2,
        start_time=30.0,
        end_time=75.0,
        description="Descrição completa do clipe para redes sociais",
        hashtags=["#viral", "#shorts", "#youcut"],
        thumbnail_idea="Frame com expressão de surpresa no segundo 35",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    assert clip.title == "Momento Incrível"
    assert clip.viral_score == 9.2
    assert clip.start_time == 30.0
    assert clip.end_time == 75.0
    assert len(clip.hashtags) == 3


def test_viral_clip_default_thumbnail_text_is_empty():
    clip = ViralClip(
        title="Momento Incrível",
        reason="Gancho forte no início com curiosidade",
        viral_score=9.2,
        start_time=30.0,
        end_time=75.0,
        description="Descrição completa do clipe para redes sociais",
        hashtags=["#viral"],
        thumbnail_idea="Frame com expressão de surpresa no segundo 35",
    )
    assert clip.thumbnail_text == ""
    assert clip.social_hook_title == ""
    assert clip.social_image_prompt == ""
    assert clip.social_visual_style == ""


def test_thumbnail_frame_result_default_methods_are_local(tmp_path):
    result = ThumbnailFrameResult(
        frame_timestamp=1.0,
        frame_score=0.5,
        segmentation_applied=False,
        output_path=tmp_path / "thumb.png",
    )
    assert result.selection_method == "local"
    assert result.generation_method == "local"


def test_thumbnail_frame_result_accepts_ai_methods(tmp_path):
    result = ThumbnailFrameResult(
        frame_timestamp=1.0,
        frame_score=0.5,
        segmentation_applied=False,
        output_path=tmp_path / "thumb.png",
        selection_method="ai",
        generation_method="ai",
    )
    assert result.selection_method == "ai"
    assert result.generation_method == "ai"


def test_thumbnail_frame_result_invalid_method_raises(tmp_path):
    with pytest.raises(ValidationError):
        ThumbnailFrameResult(
            frame_timestamp=1.0,
            frame_score=0.5,
            segmentation_applied=False,
            output_path=tmp_path / "thumb.png",
            selection_method="invalid",
        )


def test_viral_clip_score_too_high_raises():
    with pytest.raises(ValidationError):
        ViralClip(
            title="Clip",
            reason="x",
            viral_score=11.0,
            start_time=0.0,
            end_time=30.0,
            description="desc",
            hashtags=[],
            thumbnail_idea="thumb",
            thumbnail_text="MOMENTO IMPACTANTE",
        )


def test_viral_clip_score_negative_raises():
    with pytest.raises(ValidationError):
        ViralClip(
            title="Clip",
            reason="x",
            viral_score=-1.0,
            start_time=0.0,
            end_time=30.0,
            description="desc",
            hashtags=[],
            thumbnail_idea="thumb",
            thumbnail_text="MOMENTO IMPACTANTE",
        )


def test_viral_clip_score_boundary_values():
    clip_zero = ViralClip(
        title="Clip",
        reason="x",
        viral_score=0.0,
        start_time=0.0,
        end_time=30.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="thumb",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    assert clip_zero.viral_score == 0.0

    clip_ten = ViralClip(
        title="Clip",
        reason="x",
        viral_score=10.0,
        start_time=0.0,
        end_time=30.0,
        description="desc",
        hashtags=[],
        thumbnail_idea="thumb",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    assert clip_ten.viral_score == 10.0


def test_transcription_result_with_segments():
    result = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=5.0,
                text="Olá mundo",
                words=[
                    WordTimestamp(word="Olá", start=0.0, end=0.5),
                    WordTimestamp(word="mundo", start=0.6, end=1.0),
                ],
            ),
            TranscriptionSegment(
                start=5.0,
                end=10.0,
                text="Teste de transcrição",
                words=[
                    WordTimestamp(word="Teste", start=5.0, end=5.4),
                ],
            ),
        ],
        language="pt",
        source_path=Path("video.mp4"),
    )
    assert len(result.segments) == 2
    assert result.language == "pt"
    assert result.source_path == Path("video.mp4")
    assert len(result.segments[0].words) == 2


def test_word_timestamp():
    word = WordTimestamp(word="Python", start=1.2, end=1.8)
    assert word.word == "Python"
    assert word.start == 1.2
    assert word.end == 1.8


def test_video_metadata_creation():
    meta = VideoMetadata(
        title="Minha Live Incrível",
        duration_seconds=3600.0,
        url="https://youtube.com/watch?v=abc123",
    )
    assert meta.title == "Minha Live Incrível"
    assert meta.duration_seconds == 3600.0
    assert meta.url == "https://youtube.com/watch?v=abc123"


def test_clip_record_default_approved():
    record = ClipRecord(
        title="Clip 1",
        start_time=10.0,
        end_time=70.0,
        clip_path=Path("output/clip_01.mp4"),
        thumbnail_path=None,
    )
    assert record.approved is True
    assert record.thumbnail_path is None


def test_clip_record_with_thumbnail():
    record = ClipRecord(
        title="Clip 2",
        start_time=120.0,
        end_time=300.0,
        clip_path=Path("output/clip_02.mp4"),
        thumbnail_path=Path("output/thumbnails/clip_02.png"),
        approved=False,
    )
    assert record.approved is False
    assert record.thumbnail_path == Path("output/thumbnails/clip_02.png")


def test_session_data_creation():
    clip = ClipRecord(
        title="Clip A",
        start_time=0.0,
        end_time=60.0,
        clip_path=Path("output/clip_a.mp4"),
        thumbnail_path=None,
    )
    session = SessionData(
        session_id="sess-001",
        source_url="https://youtube.com/watch?v=xyz",
        cut_mode="youtube",
        transcription_cache_path=Path("~/.youcut/sessions/sess-001_transcription.json"),
        clips=[clip],
        created_at=datetime(2026, 4, 25, 10, 0, 0),
        output_dir=Path("output/sess-001"),
    )
    assert session.session_id == "sess-001"
    assert session.cut_mode == "youtube"
    assert len(session.clips) == 1
    assert session.clips[0].title == "Clip A"


def test_session_data_thumbnail_path_accepts_none():
    clip = ClipRecord(
        title="Clip B",
        start_time=5.0,
        end_time=65.0,
        clip_path=Path("output/clip_b.mp4"),
        thumbnail_path=None,
    )
    assert clip.thumbnail_path is None


def test_viral_clip_accepts_cut_mode():
    clip = ViralClip(
        title="Clip com modo",
        reason="Conteúdo relevante",
        viral_score=8.0,
        start_time=0.0,
        end_time=90.0,
        description="Descrição",
        hashtags=["#youtube"],
        thumbnail_idea="Frame inicial",
        thumbnail_text="MOMENTO IMPACTANTE",
        cut_mode="youtube",
    )
    assert clip.cut_mode == "youtube"


def test_viral_clip_default_cut_mode_is_social():
    clip = ViralClip(
        title="Clip sem modo",
        reason="Conteúdo",
        viral_score=5.0,
        start_time=0.0,
        end_time=60.0,
        description="Desc",
        hashtags=[],
        thumbnail_idea="thumb",
        thumbnail_text="MOMENTO IMPACTANTE",
    )
    assert clip.cut_mode == "social"


def test_cut_mode_rejects_invalid_value():
    with pytest.raises(ValidationError):
        ViralClip(
            title="Clip inválido",
            reason="Razão",
            viral_score=5.0,
            start_time=0.0,
            end_time=60.0,
            description="Desc",
            hashtags=[],
            thumbnail_idea="thumb",
            thumbnail_text="MOMENTO IMPACTANTE",
            cut_mode="facebook",
        )


def test_session_data_rejects_invalid_cut_mode():
    with pytest.raises(ValidationError):
        SessionData(
            session_id="sess-x",
            source_url="https://youtube.com/watch?v=abc",
            cut_mode="facebook",
            transcription_cache_path=Path("~/.youcut/sessions/sess-x.json"),
            clips=[],
            created_at=datetime(2026, 4, 25, 10, 0, 0),
            output_dir=Path("output/sess-x"),
        )


def test_clip_record_new_fields_default_values():
    record = ClipRecord(
        title="Clip Test",
        start_time=0.0,
        end_time=60.0,
        clip_path=Path("output/clip.mp4"),
        thumbnail_path=None,
    )
    assert record.youtube_video_id is None
    assert record.youtube_url is None
    assert record.upload_status == {}


def test_clip_record_new_fields_with_values():
    record = ClipRecord(
        title="Clip YT",
        start_time=0.0,
        end_time=120.0,
        clip_path=Path("output/clip_yt.mp4"),
        thumbnail_path=None,
        youtube_video_id="dQw4w9WgXcQ",
        youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        upload_status={"youtube": "success", "tiktok": "failed"},
    )
    assert record.youtube_video_id == "dQw4w9WgXcQ"
    assert record.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert record.upload_status == {"youtube": "success", "tiktok": "failed"}


def test_clip_record_serialization_with_new_fields():
    record = ClipRecord(
        title="Clip Serialize",
        start_time=5.0,
        end_time=65.0,
        clip_path=Path("output/clip_s.mp4"),
        thumbnail_path=None,
        youtube_video_id="abc123",
        youtube_url="https://www.youtube.com/watch?v=abc123",
        upload_status={"youtube": "success"},
    )
    data = record.model_dump()
    restored = ClipRecord(**data)
    assert restored.youtube_video_id == "abc123"
    assert restored.youtube_url == "https://www.youtube.com/watch?v=abc123"
    assert restored.upload_status == {"youtube": "success"}


def test_clip_record_json_roundtrip_new_fields():
    record = ClipRecord(
        title="Clip JSON",
        start_time=10.0,
        end_time=70.0,
        clip_path=Path("output/clip_j.mp4"),
        thumbnail_path=None,
        youtube_video_id="vid999",
        youtube_url="https://www.youtube.com/watch?v=vid999",
        upload_status={"youtube": "success", "instagram": "failed"},
    )
    json_str = record.model_dump_json()
    restored = ClipRecord.model_validate_json(json_str)
    assert restored.youtube_video_id == "vid999"
    assert restored.youtube_url == "https://www.youtube.com/watch?v=vid999"
    assert restored.upload_status == {"youtube": "success", "instagram": "failed"}


def test_upload_result_video_id_default_none():
    result = UploadResult(
        platform="youtube",
        clip_index=0,
        status="success",
    )
    assert result.video_id is None


def test_upload_result_with_video_id():
    result = UploadResult(
        platform="youtube",
        clip_index=0,
        status="success",
        video_id="dQw4w9WgXcQ",
    )
    assert result.video_id == "dQw4w9WgXcQ"


def test_upload_result_serialization_with_video_id():
    result = UploadResult(
        platform="youtube",
        clip_index=1,
        status="success",
        url="https://www.youtube.com/watch?v=abc",
        video_id="abc",
    )
    data = result.model_dump()
    restored = UploadResult(**data)
    assert restored.video_id == "abc"
    assert restored.platform == "youtube"


def test_upload_result_json_roundtrip_video_id():
    result = UploadResult(
        platform="youtube",
        clip_index=2,
        status="failed",
        error="quota exceeded",
        video_id=None,
    )
    json_str = result.model_dump_json()
    restored = UploadResult.model_validate_json(json_str)
    assert restored.video_id is None
    assert restored.error == "quota exceeded"
