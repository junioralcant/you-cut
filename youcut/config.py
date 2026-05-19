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
    subtitle_style: Literal[
        "word", "phrase", "phrase_serif_centered", "word_serif_italic"
    ] = "word"
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
    social_layout_mode: Literal[
        "classic",
        "speaker_bottom_ai_top",
        "speaker_top_ai_bottom",
        "youtube_top_ai_bottom",
        "alternating_image",
    ] = "classic"
    social_layout_title_enabled: bool = True
    social_layout_image_provider: Literal["openai", "local"] = "openai"
    social_layout_apply_face_tracking: bool = True
    social_layout_top_image_height: int = 600
    social_layout_title_band_height: int = 140
    social_layout_title_color_mode: Literal["engagement_default", "yellow", "orange", "custom"] = "engagement_default"
    social_layout_title_bg_color: str = "#F4C400"
    social_layout_title_text_color: str = "#111111"
    social_visual_style_enabled: bool = True
    social_image_reference_path: Path | None = None

    decoupage_enabled: bool = True
    decoupage_noise_db: float = -30.0
    decoupage_min_silence_gap: float = 0.4
    decoupage_keep_padding: float = 0.05
    social_filter_preset: Literal[
        "none", "warm", "cool", "vintage", "punchy", "motivacao_lilac"
    ] = "none"

    # ── Preset visual "motivacao" (9:16 social) ────────────────────────────
    # Estilo modelado a partir de Reels motivacionais pt-BR: legenda Lora
    # SemiBold Italic palavra-única central + handle colado abaixo + badge
    # @handle no canto inferior esquerdo + outro fade-to-black 3s.
    # Ver tasks/prd-preset-motivacao/analise-video-referencia.md.
    motivacao_handle: str | None = None  # ex: "tribodavisionaria" (sem @)
    motivacao_overlay: bool = False  # queima badge no canto inferior esquerdo
    motivacao_outro: bool = False  # anexa 3s fade-to-black com badge central

    runway_api_key: str | None = None
    fal_api_key: str | None = None
    replicate_api_token: str | None = None

    # Catálogo local de imagens de jogadores. Usado pelo módulo youcut.players
    # para injetar a foto do jogador como reference frame na geração de
    # thumbnails / imagens sociais quando o nome dele aparece no clipe.
    players_dir: Path = Path.home() / ".youcut" / "players"

    # Catálogo local de imagens de apresentadores / hosts. Usado pelo módulo
    # youcut.presenters: Claude vision identifica qual apresentador aparece
    # nos frames do vídeo (uma vez por source) e a foto é prepended como
    # reference frame nas thumbnails. Independente do players_dir — ambos
    # podem entrar simultaneamente.
    presenters_dir: Path = Path.home() / ".youcut" / "apresentadores"

    # Override manual da detecção de apresentadores. Quando definido, pula
    # a chamada Claude vision e usa os slugs informados (ex.: ["tiago_leifert"]).
    # Setado pela flag CLI ``--presenter SLUG1,SLUG2``.
    presenter_slugs: list[str] | None = None
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
    comic_animation_engine: Literal["prunaai", "panels", "scenes", "remotion"] = "scenes"
    # ── Engine "remotion" — render local programático com lip-sync sílaba-level ──
    comic_remotion_enabled_default: bool = False  # reservado p/ futura promoção a default
    comic_remotion_fps: int = 30  # fps do <Composition>
    comic_remotion_node_bin: str = "node"  # path do binário Node a usar
    comic_remotion_concurrency: int | None = None  # passado p/ renderMedia (CPU threads)
    comic_remotion_studio_port: int = 3000  # porta do Remotion Studio
    comic_remotion_kenburns_default_scale: float = 1.12  # escala alvo do Ken Burns
    comic_remotion_idle_blink_period_sec: float = 4.5  # periodicidade do blink idle
    comic_remotion_pyphen_locale_fallback: str = "pt_BR"  # locale de fallback p/ pyphen
    # ── Engine "scenes" (default) — narrativa multi-cena + word-level lip-sync ──
    comic_scenes_count: int = 4  # número de cenas narrativas
    comic_scenes_crossfade_dur: float = 0.25  # crossfade entre chunks (s)
    comic_scenes_min_chunk_dur: float = 1.05  # mín exigido pelo prunaai (s)
    comic_scenes_gap_absorb_threshold: float = 0.5  # gaps > X são absorvidos pelo chunk anterior
    comic_scenes_smooth_attribution: bool = True  # corrige mis-attributions cercadas
    comic_scenes_inter_call_pause_s: float = 11.0  # pausa entre chamadas prunaai (rate-limit)
    comic_scenes_watermark_text: str | None = None  # ex: "@anima.nos" — None desliga watermark
    comic_scenes_watermark_opacity: float = 0.40
    comic_scenes_watermark_y_from_bottom: int = 280  # px do fundo
    comic_scenes_emit_no_subs_version: bool = True  # gera também versão sem legendas
    comic_scenes_style_ref_image: Path | None = None  # imagem de referência canônica (opcional)

    # ── Trilha Sonora — Playlist Curada do YouTube ────────────────────────────
    youtube_music_playlist_url: str = (
        "https://www.youtube.com/playlist?list=PLrAXtmRdnEQzAHKFQJX4D7QAR0o0eEdQy"
    )

    # ── Pipeline reddit-story (long-form Reddit narrado, 16:9) ────────────────
    # Stack 100% Replicate (Kokoro TTS + Flux Schnell images). Defaults
    # otimizados pra r/MaliciousCompliance / r/ProRevenge baseado em research
    # dos canais virais 2026 — am_adam é o analog do "ElevenLabs Adam" que
    # domina o nicho. Ver memória [[stack-poc-hollowire]] e [[feedback-custo-imagens]].
    reddit_story_voice: str = "am_adam"  # Kokoro 82M voice (authoritative male)
    reddit_story_speed: float = 1.05  # snappier que o default 1.0
    reddit_story_target_words: int = 5000  # ~22-25 min narrados
    reddit_story_scene_count: int = 8  # visual beats Claude planeja
    reddit_story_resolution_w: int = 1920
    reddit_story_resolution_h: int = 1080
    reddit_story_whisper_model: str = "small.en"  # rápido e suficiente p/ narração
    reddit_story_user_agent: str = (
        "youcut-reddit-story/0.1 (https://github.com/youcut)"
    )  # Reddit bloqueia UAs genéricos

    @model_validator(mode="after")
    def validate_api_key_present(self) -> "PipelineConfig":
        if not self.anthropic_api_key or not self.anthropic_api_key.strip():
            raise ValueError(
                "ANTHROPIC_API_KEY é obrigatório. "
                "Defina a variável de ambiente ou crie um arquivo .env com ANTHROPIC_API_KEY=sua_chave."
            )
        return self
