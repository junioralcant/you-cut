from datetime import datetime
from enum import Enum
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


class MusicTrack(BaseModel):
    """Faixa musical baixada de uma playlist do YouTube e usada na mixagem do clipe social."""

    video_id: str
    name: str
    source_url: str
    local_path: Path
    mood: str
    duration_s: float


class SyncReport(BaseModel):
    """Resumo de uma execução de sincronização da playlist YouTube com o acervo local."""

    new_tracks: int = 0
    cached_tracks: int = 0
    failed_tracks: int = 0
    failed_details: list[tuple[str, str]] = []


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
    music_track: MusicTrack | None = None


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
    narrative_mode: bool = False
    narrative_elements: list[str] = []

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


class PlatformMetadata(BaseModel):
    """Metadados editoriais formatados para uma plataforma específica."""

    platform: str
    title: str
    description: str
    hashtags: list[str] = []


class ComicMetadata(BaseModel):
    """Metadados multi-plataforma gerados ao final do `youcut comic`."""

    summary: str = ""
    tiktok: PlatformMetadata
    instagram_reels: PlatformMetadata
    youtube_shorts: PlatformMetadata


class MouthShape(str, Enum):
    CLOSED = "closed"
    OPEN_MID = "open_mid"
    OPEN_WIDE = "open_wide"
    OPEN_ROUND = "open_round"


class MouthSheet(BaseModel):
    character_id: str
    sheet_path: Path
    cells: dict[MouthShape, tuple[int, int, int, int]]

    @field_validator("cells")
    @classmethod
    def validate_cells_required_shapes(
        cls, v: dict[MouthShape, tuple[int, int, int, int]]
    ) -> dict[MouthShape, tuple[int, int, int, int]]:
        required = {MouthShape.CLOSED, MouthShape.OPEN_MID, MouthShape.OPEN_WIDE}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(
                f"MouthSheet.cells deve conter ao menos {sorted(s.value for s in required)}; "
                f"faltando: {sorted(s.value for s in missing)}"
            )
        for shape, box in v.items():
            if len(box) != 4:
                raise ValueError(
                    f"MouthSheet.cells[{shape.value}] deve ser tupla (x1, y1, x2, y2); recebido {box}"
                )
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"MouthSheet.cells[{shape.value}] deve satisfazer x2>x1 e y2>y1; recebido {box}"
                )
        return v


class MouthEvent(BaseModel):
    character_id: str
    start_sec: float
    end_sec: float
    shape: MouthShape

    @field_validator("end_sec")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_sec")
        if start is not None and v < start:
            raise ValueError(
                f"end_sec ({v}) não pode ser menor que start_sec ({start})"
            )
        return v


class RemotionScene(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    character_ids: list[str]
    speaker_id: str | None = None
    ken_burns: dict = {}
    transition_in: Literal["cut", "crossfade", "wipe"] = "crossfade"
    shakes: list[dict] = []
    lip_sync: list[MouthEvent] = []

    @field_validator("end_sec")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_sec")
        if start is not None and v <= start:
            raise ValueError(
                f"end_sec ({v}) deve ser maior que start_sec ({start})"
            )
        return v


class RemotionInputProps(BaseModel):
    audio_path: str
    duration_sec: float
    fps: int = 30
    width: int = 1080
    height: int = 1920
    characters: dict[str, dict] = {}
    scenes: list[RemotionScene] = []
    background_color: str = "#000000"

    @field_validator("duration_sec")
    @classmethod
    def validate_duration_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"duration_sec deve ser positivo, recebido {v}")
        return v

    @field_validator("fps")
    @classmethod
    def validate_fps_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"fps deve ser positivo, recebido {v}")
        return v
