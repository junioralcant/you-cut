from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    whisper_model: str = "medium"
    claude_model: str = "claude-sonnet-4-6"
    clip_count: int = 5
    subtitle_style: Literal["word", "phrase"] = "word"
    output_dir: Path = Path("output")
    dry_run: bool = False
    blur_background: bool = False
    vertical_fill_mode: Literal["fill_crop", "blur_background"] = "fill_crop"
    title_overlay: bool = False
    upload: bool = False
    platforms: list[str] = ["youtube", "instagram", "tiktok"]
    clips: list[int] | None = None

    @model_validator(mode="after")
    def validate_api_key_present(self) -> "PipelineConfig":
        if not self.anthropic_api_key or not self.anthropic_api_key.strip():
            raise ValueError(
                "ANTHROPIC_API_KEY é obrigatório. "
                "Defina a variável de ambiente ou crie um arquivo .env com ANTHROPIC_API_KEY=sua_chave."
            )
        return self
