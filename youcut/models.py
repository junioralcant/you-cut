from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator


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

    @field_validator("viral_score")
    @classmethod
    def validate_viral_score(cls, v: float) -> float:
        if not 0 <= v <= 10:
            raise ValueError(f"viral_score must be between 0 and 10, got {v}")
        return v
