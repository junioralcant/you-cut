from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


class SpeakerSegment(BaseModel):
    speaker_id: str
    start: float
    end: float


class CropRegion(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceTrackingResult(BaseModel):
    frame_regions: list[CropRegion | None]
    had_faces: bool
    is_split_screen: list[bool]
    secondary_regions: list[CropRegion | None]

CutMode = Literal["social", "youtube"]


class VideoMetadata(BaseModel):
    title: str
    duration_seconds: float
    url: str


class ClipRecord(BaseModel):
    title: str
    start_time: float
    end_time: float
    clip_path: Path
    thumbnail_path: Path | None
    approved: bool = True
    description: str = ""
    hashtags: list[str] = []
    youtube_video_id: str | None = None
    youtube_url: str | None = None
    upload_status: dict[str, str] = {}
    captions_applied: bool = True
    caption_warning: str | None = None


class SessionData(BaseModel):
    session_id: str
    source_url: str
    cut_mode: CutMode
    transcription_cache_path: Path
    clips: list[ClipRecord]
    created_at: datetime
    output_dir: Path


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordTimestamp]


class TranscriptionResult(BaseModel):
    segments: list[TranscriptionSegment]
    language: str
    source_path: Path


class CaptionBurnResult(BaseModel):
    output_path: Path
    captions_applied: bool
    warning: str | None = None

    def __fspath__(self) -> str:
        return str(self.output_path)

    def __str__(self) -> str:
        return str(self.output_path)

    def __getattr__(self, name: str):
        return getattr(self.output_path, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self.output_path == other
        return super().__eq__(other)


class ViralClip(BaseModel):
    title: str
    reason: str
    viral_score: float  # 0-10
    start_time: float
    end_time: float
    description: str
    hashtags: list[str]
    thumbnail_idea: str
    thumbnail_text: str = ""
    social_hook_title: str = ""
    social_image_prompt: str = ""
    social_visual_style: str = ""
    cut_mode: CutMode = "social"

    @field_validator("viral_score")
    @classmethod
    def validate_viral_score(cls, v: float) -> float:
        if not 0 <= v <= 10:
            raise ValueError(f"viral_score must be between 0 and 10, got {v}")
        return v


class ThumbnailFrameResult(BaseModel):
    frame_timestamp: float
    frame_score: float
    segmentation_applied: bool
    output_path: Path
    selection_method: Literal["ai", "local"] = "local"
    generation_method: Literal["ai", "local"] = "local"
