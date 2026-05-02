from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from youcut.models import CutMode


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
    cut_mode: CutMode = "social"
    max_clips: int | None = None
    openai_api_key: str | None = None
    session_timeout_minutes: int = 7
    face_tracking: bool = False
    huggingface_token: str | None = None
    face_detection_confidence: float = 0.5
    thumbnail_text: str = ""
    social_layout_mode: Literal["classic", "speaker_bottom_ai_top"] = "classic"
    social_layout_title_enabled: bool = True
    social_layout_image_provider: Literal["openai", "local"] = "openai"
    social_layout_apply_face_tracking: bool = True
    social_layout_top_image_height: int = 860
    social_layout_title_band_height: int = 180
    social_layout_title_color_mode: Literal["engagement_default", "yellow", "orange", "custom"] = "engagement_default"
    social_layout_title_bg_color: str = "#F4C400"
    social_layout_title_text_color: str = "#111111"

    decoupage_enabled: bool = False
    decoupage_noise_db: float = -30.0
    decoupage_min_silence_gap: float = 0.4
    decoupage_keep_padding: float = 0.05
    social_filter_preset: Literal["none", "warm", "cool", "vintage", "punchy"] = "none"

    runway_api_key: str | None = None
    fal_api_key: str | None = None
    replicate_api_token: str | None = None
    comic_max_panels: int = 30
    comic_cost_cap_usd: float = 10.0
    comic_image_provider: Literal["gpt-image-1"] = "gpt-image-1"
    comic_i2v_provider: Literal["runway", "fal", "replicate"] = "runway"
    comic_i2v_fal_model: str = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"
    comic_i2v_replicate_model: str = "kwaivgi/kling-v1.6-pro"
    comic_image_retries: int = 2
    comic_i2v_retries: int = 2
    comic_panel_min_seconds: float = 2.0
    comic_panel_max_seconds: float = 5.0
    comic_i2v_concurrency: int = 1
    comic_i2v_max_poll_seconds: float = 3600.0
    comic_invent_cast: bool = False
    comic_enforce_multi_participant: bool = False
    comic_force_narrative_mode: bool = False
    comic_dialogue_mode: bool = False
    comic_scene_seed: str | None = None
    comic_composition_seed_image: Path | None = None
    comic_output_width: int = 1080
    comic_output_height: int = 1920
    comic_generate_metadata: bool = True

    @model_validator(mode="after")
    def validate_api_key_present(self) -> "PipelineConfig":
        if not self.anthropic_api_key or not self.anthropic_api_key.strip():
            raise ValueError(
                "ANTHROPIC_API_KEY é obrigatório. "
                "Defina a variável de ambiente ou crie um arquivo .env com ANTHROPIC_API_KEY=sua_chave."
            )
        return self
