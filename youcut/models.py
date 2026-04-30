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


CastKind = Literal["person", "animal", "object"]
PanelFraming = Literal["close", "medium", "wide", "two_shot"]


class CastMember(BaseModel):
    character_id: str
    kind: CastKind = "person"
    gender_apparent: str = ""
    age_apparent: str = ""
    hair: str = ""
    facial_hair: str = ""
    skin: str = ""
    clothing: str = ""
    accessories: list[str] = []
    narrative_role: str = ""
    speaker_id: str | None = None
    source_frame_path: Path | None = None
    anchor_image_path: Path | None = None
    text_card: str = ""


class Panel(BaseModel):
    index: int
    start_time: float
    end_time: float
    participants: list[str]
    framing: PanelFraming
    scene: str
    pose_description: str
    panel_seconds_target: float

    @field_validator("end_time")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_time")
        if start is not None and v <= start:
            raise ValueError(
                f"end_time ({v}) deve ser maior que start_time ({start})"
            )
        return v

    @field_validator("panel_seconds_target")
    @classmethod
    def validate_seconds_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"panel_seconds_target deve ser positivo, recebido {v}")
        return v


class PanelRenderResult(BaseModel):
    panel_index: int
    base_image_path: Path
    clip_path: Path
    clip_seconds: float
    was_static_fallback: bool = False
    image_attempts: int = 1
    i2v_attempts: int = 0
    cost_usd: float = 0.0


class MotionComicSession(BaseModel):
    session_id: str
    video_path: Path
    created_at: datetime
    transcription_cache_path: Path | None = None
    cast: list[CastMember] = []
    panels: list[Panel] = []
    panel_results: list[PanelRenderResult] = []
    total_cost_usd: float = 0.0
    output_path: Path | None = None
