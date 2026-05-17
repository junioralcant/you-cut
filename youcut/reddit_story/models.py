"""Modelos Pydantic do pipeline reddit-story."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from youcut.models import WordTimestamp


class RedditStorySource(BaseModel):
    """Thread bruta do Reddit, salva pra reprodutibilidade."""

    url: str
    title: str
    author: str
    subreddit: str
    ups: int
    permalink: str
    selftext: str

    @property
    def word_count(self) -> int:
        return len(self.selftext.split())


class ScenePlan(BaseModel):
    """1 beat visual gerado pelo Claude scene planner."""

    beat: str
    prompt: str


class RedditStorySession(BaseModel):
    """Sessão completa persistida em ~/.youcut/sessions/<id>.json."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: RedditStorySource
    work_dir: Path
    script: str | None = None
    narration_path: Path | None = None
    narration_duration_s: float | None = None
    words: list[WordTimestamp] = Field(default_factory=list)
    scenes: list[ScenePlan] = Field(default_factory=list)
    image_paths: list[Path] = Field(default_factory=list)
    final_video_path: Path | None = None
    thumbnail_path: Path | None = None
    metadata_path: Path | None = None
