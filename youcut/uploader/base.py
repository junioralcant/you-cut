from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ClipMetadata(BaseModel):
    title: str
    description: str
    hashtags: list[str]
    caption: str  # description + hashtags formatted per platform


class UploadResult(BaseModel):
    platform: str
    clip_index: int
    status: Literal["success", "skipped", "failed", "pending"]
    url: str | None = None
    error: str | None = None


class UploadReport(BaseModel):
    results: list[UploadResult]
    timestamp: datetime
    output_dir: Path


class Uploader(ABC):
    def __init__(self, token_dir: Path) -> None:
        self.token_dir = token_dir

    @abstractmethod
    def authenticate(self) -> None: ...

    @abstractmethod
    def upload(self, video_path: Path, metadata: ClipMetadata, clip_index: int = 0) -> UploadResult: ...

    @property
    @abstractmethod
    def platform_name(self) -> str: ...
