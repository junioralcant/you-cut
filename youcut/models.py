from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

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


class ViralClip(BaseModel):
    title: str
    reason: str
    viral_score: float  # 0-10
    start_time: float
    end_time: float
    description: str
    hashtags: list[str]
    thumbnail_idea: str
    cut_mode: CutMode = "social"

    @field_validator("viral_score")
    @classmethod
    def validate_viral_score(cls, v: float) -> float:
        if not 0 <= v <= 10:
            raise ValueError(f"viral_score must be between 0 and 10, got {v}")
        return v
