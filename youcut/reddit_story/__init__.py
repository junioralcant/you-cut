"""Pipeline ``youcut reddit-story`` — vídeos long-form 16:9 narrados a partir
de threads do Reddit (r/MaliciousCompliance, r/ProRevenge, r/AmITheAsshole...).

Stack canônica (todas as chamadas pagas via Replicate, ver
[[stack-poc-hollowire]]/[[feedback-custo-imagens]] em memory):

  Reddit JSON → Claude (formatador) → Replicate Kokoro am_adam 1.05x (TTS) →
  faster-whisper (timestamps) → Claude (8 visual beats) → Replicate Flux Schnell
  16:9 (imagens) → ffmpeg Ken Burns + ASS subs (mux) + Flux Schnell + Pillow
  (thumbnail 1280×720).

Custo típico: ~$0.55/vídeo de 24 min. Roda em ~13 min wall time.
"""

from youcut.reddit_story.pipeline import (  # noqa: F401
    RedditStoryPipelineError,
    run_reddit_story_pipeline,
)

__all__ = ["run_reddit_story_pipeline", "RedditStoryPipelineError"]
