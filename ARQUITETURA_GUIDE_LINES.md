# ARQUITETURA_GUIDE_LINES — YouCut

> **Propósito deste documento:** mapa de arquitetura **arquivo a arquivo** do repositório, otimizado para LLMs que precisam navegar, modificar ou raciocinar sobre o código sem precisar abrir cada arquivo. Complementa o `AGENTS.md` (visão geral) e o `README.md` (uso humano).
>
> **Convenções:**
> - Cada entrada lista: **função do arquivo** · **APIs públicas relevantes** · **dependências externas-chave** · **lugar no pipeline**.
> - Sempre que houver classes/funções, os nomes são mantidos **literalmente como aparecem no código** (não traduzidos).
> - "Pipeline" = um dos estágios canônicos: `download · transcrição · análise · corte · face_tracking · legendagem · composição · thumbnail · revisão · exportação · sessão · upload · CLI · suporte`.

---

## 1. Mapa Resumido do Pipeline

```
[SOURCE: URL ou arquivo local]
        │
        ▼
┌───────────────────┐    cli.py  (Typer: run | cuts | auth)
│       CLI         │    config.py  (PipelineConfig via .env)
└───────────────────┘
        │
        ▼
┌───────────────────┐    downloader.py  + yt_dlp_auth.py
│    Download       │    video_metadata.py  + url_utils.py
└───────────────────┘
        │
        ▼
┌───────────────────┐    transcriber.py  (cache MD5)
│    Transcrição    │
└───────────────────┘
        │
        ▼
┌───────────────────┐    analyzer.py  (Claude + tool use)
│   Análise IA      │
└───────────────────┘
        │
        ▼
┌───────────────────┐    clipper.py  (ffmpeg)
│      Corte        │  ─► face_tracker.py  + diarizer.py  (modo social)
└───────────────────┘
        │
        ▼
┌───────────────────┐    captioner.py  (.ass)  ──► caption_burner.py (fallback)
│    Legendagem     │
└───────────────────┘
        │
        ▼
┌───────────────────┐    title_overlay.py     (modo youtube, opcional)
│    Composição     │    social_composer.py   (modo social, layout editorial)
└───────────────────┘
        │
        ▼
┌───────────────────┐    thumbnail_generator.py  (DALL·E 3 + Pillow)
│    Thumbnail      │    preview.py  (preview rápido para CLI)
└───────────────────┘
        │
        ▼
┌───────────────────┐    selector.py  (escolha múltipla)
│    Revisão        │    reviewer.py  (TUI de aprovação)
└───────────────────┘
        │
        ▼
┌───────────────────┐    exporter.py  (clip_NN.txt)
│   Metadados       │    session_store.py  (~/.youcut/sessions/)
└───────────────────┘
        │
        ▼
┌───────────────────┐    uploader/  (YouTube · Instagram · TikTok)
│     Upload        │    + report.py
└───────────────────┘
```

---

## 2. Pacote `youcut/` — Núcleo

### 2.1 Entrada e Configuração

#### `youcut/__init__.py`
- **Função:** raiz do pacote. Apenas exporta `__version__ = "0.1.0"`.
- **Pipeline:** —

#### `youcut/cli.py`
- **Função:** entrypoint Typer. Define os comandos de mais alto nível e orquestra os fluxos.
- **APIs públicas:** `app` (Typer), `run()`, `cuts()`, `auth_login()`, `auth_revoke()`, `auth_status()`; helpers internos: `run_flow_a()`, `run_flow_b()`, `run_flow_c()`, `offer_flow_b()`, `_check_ffmpeg()`.
- **Dependências externas:** `typer`, `rich`, `questionary`; verifica `ffmpeg` no `PATH`.
- **Pipeline:** CLI (orquestra todos os estágios).

#### `youcut/config.py`
- **Função:** configuração tipada via `pydantic-settings`. Lê `.env` e variáveis de ambiente; valida `ANTHROPIC_API_KEY` cedo.
- **APIs públicas:** `PipelineConfig` (campos como `anthropic_api_key`, `openai_api_key`, `whisper_model`, `claude_model`, `cut_mode`, `social_layout_mode`, `face_tracking`, `title_overlay`, etc.).
- **Dependências externas:** `pydantic`, `pydantic-settings`.
- **Pipeline:** suporte (lido em todo lugar).

#### `youcut/models.py`
- **Função:** todos os modelos de dados (Pydantic) compartilhados entre módulos.
- **APIs públicas:** `ViralClip`, `ClipRecord`, `SessionData`, `TranscriptionResult`, `TranscriptionSegment`, `WordTimestamp`, `VideoMetadata`, `CaptionBurnResult`, `FaceTrackingResult`, `CropRegion`, `SpeakerSegment`, `ThumbnailFrameResult`. Type alias: `CutMode = Literal["social", "youtube"]`.
- **Pipeline:** suporte.

---

### 2.2 Download e Metadata

#### `youcut/downloader.py`
- **Função:** baixa vídeos com `yt-dlp` ou resolve caminhos locais.
- **APIs públicas:** `download_video(source, output_dir, auth_config)`; exceção `VideoDownloadError`.
- **Dependências externas:** `yt-dlp`.
- **Pipeline:** download.

#### `youcut/yt_dlp_auth.py`
- **Função:** resolve autenticação para `yt-dlp` (cookies de browser, arquivo de cookies, runtimes JS).
- **APIs públicas:** `YtDlpAuthConfig`, `YtDlpAuthConfigError`, `resolve_yt_dlp_auth_config()`, `apply_yt_dlp_auth()`.
- **Variáveis de ambiente:** `YOUCUT_COOKIES_FROM_BROWSER`, `YOUCUT_COOKIES_FILE`, `YOUCUT_YTDLP_JS_RUNTIMES`.
- **Pipeline:** suporte (download).

#### `youcut/url_utils.py`
- **Função:** normaliza URLs do YouTube (remove parâmetros inúteis, escapa caracteres).
- **APIs públicas:** `normalize_video_url()`.
- **Pipeline:** suporte.

#### `youcut/video_metadata.py`
- **Função:** extrai título e duração via `yt-dlp` **sem baixar** o vídeo.
- **APIs públicas:** `fetch_metadata(url, auth_config)`; exceção `VideoMetadataError`.
- **Dependências externas:** `yt-dlp`.
- **Pipeline:** download (verificação prévia).

---

### 2.3 Transcrição e Análise

#### `youcut/transcriber.py`
- **Função:** transcreve áudio com `faster-whisper` (default) ou `openai-whisper` (fallback). Cache por **MD5 do arquivo** em `<video>_transcript.json`.
- **APIs públicas:** `transcribe(video_path, config)` → `TranscriptionResult`.
- **Dependências externas:** `faster-whisper` (extra `whisper-openai` habilita fallback).
- **Pipeline:** transcrição.

#### `youcut/analyzer.py`
- **Função:** envia transcrição em chunks de 30 min para Claude via tool use; identifica clipes virais com prompts parametrizados por `cut_mode`.
- **APIs públicas:** `analyze(transcription, config)` → `list[ViralClip]`.
- **Tool definida:** `identify_viral_clips`.
- **Constantes:** `YOUTUBE_MIN_DURATION=900s`, `SOCIAL_MIN_DURATION=15s`, `SOCIAL_MAX_DURATION=180s`. Faz remoção de overlap e ordenação por score.
- **Dependências externas:** `anthropic`.
- **Pipeline:** análise.

---

### 2.4 Corte, Diarização e Face Tracking

#### `youcut/clipper.py`
- **Função:** corta vídeo com `ffmpeg`. Modo `youtube` = stream copy 16:9; modo `social` = re-encode 9:16 com `fill_crop` (default) ou `blur_background`.
- **APIs públicas:** `cut_clip(video_path, clip, index, config)`.
- **Dependências externas:** `ffmpeg`, `ffprobe`.
- **Pipeline:** corte.

#### `youcut/diarizer.py`
- **Função:** diariza áudio do clipe com `pyannote.audio` (opcional); fallback para locutor único.
- **APIs públicas:** `diarize(clip_path, config)` → `list[SpeakerSegment]`.
- **Dependências externas:** `pyannote.audio` (extra `face-tracking`), exige `huggingface_token`; usa `ffprobe`.
- **Pipeline:** suporte (face tracking).

#### `youcut/face_tracker.py`
- **Função:** detecta rostos com MediaPipe; gera crops 9:16 com padding superior de 40%; suporta split-screen quando há dois locutores ativos. Suaviza ROIs com EMA.
- **APIs públicas:** `apply_face_tracking(clip_path, config)` → `FaceTrackingResult`.
- **Dependências externas:** `mediapipe`, `opencv-python` (extra `face-tracking`), `ffmpeg`.
- **Pipeline:** corte (modo social).

---

### 2.5 Legendas e Overlays

#### `youcut/captioner.py`
- **Função:** gera arquivo `.ass` (Advanced SubStation) a partir do `TranscriptionResult` e queima via `ffmpeg -vf ass=...`. Estilos: `word` (palavra a palavra, fonte ~80px) ou `phrase` (frase, ~60px).
- **APIs públicas:** `add_captions(clip_path, transcription, clip, config)`.
- **Dependências externas:** `ffmpeg` com `libass`.
- **Pipeline:** legendagem.

#### `youcut/caption_burner.py`
- **Função:** wrapper de fallback que **re-transcreve** o clipe (não a transcrição global) para gerar SRT/ASS — usado quando o caminho normal não pode ser usado (ex.: clipe vindo de outra fonte).
- **APIs públicas:** `CaptionBurner.burn(video_path, style, layout_mode)` → `CaptionBurnResult`.
- **Dependências externas:** `faster-whisper` ou `openai-whisper`; `ffmpeg`.
- **Pipeline:** legendagem (fallback).

#### `youcut/title_overlay.py`
- **Função:** gera card PNG com o título do clipe (Pillow) e queima nos primeiros 5s via `ffmpeg` overlay. Calcula cor dominante do frame e aplica contraste WCAG.
- **APIs públicas:** `add_title_overlay(clip_path, clip, config)`.
- **Dependências externas:** `Pillow`, `ffmpeg`.
- **Pipeline:** composição (modo youtube, opcional).

#### `youcut/social_composer.py`
- **Função:** monta layout editorial 1080×1920 (modo `speaker_bottom_ai_top`): imagem IA no topo, tarja de título/hook no meio, vídeo (com face tracking) embaixo. Pede ao Claude o texto/cores da tarja; gera imagem com OpenAI (DALL·E).
- **APIs públicas:** `compose_social_clip(clip_path, clip, config)`.
- **Dependências externas:** `anthropic`, `openai`, `Pillow`, `ffmpeg`.
- **Pipeline:** composição (modo social).

---

### 2.6 Thumbnail e Preview

#### `youcut/thumbnail_generator.py`
- **Função:** seleciona melhor frame (heurística brilho/contraste/clareza), opcionalmente gera imagem com DALL·E 3, compõe thumbnail final 1280×720 com Pillow. Também gera a imagem de topo do `social_composer`.
- **APIs públicas:** `generate_thumbnail(clip, output_dir, clip_index, clip_path, config)`, `generate_social_top_image(clip, output_dir, clip_path, config)`.
- **Heurísticas de prompt:** texto ≤7% da área; paleta ciano/verde/amarelo/laranja; evita vermelho dominante (regras em `prompt.md`).
- **Dependências externas:** `openai` (DALL·E 3), `Pillow`, `ffmpeg`.
- **Pipeline:** thumbnail.

#### `youcut/preview.py`
- **Função:** gera JPG de preview no midpoint do clipe, escalado 9:16, para a tabela do CLI.
- **APIs públicas:** `generate_clip_preview(video_path, clip, index, config)` → `PreviewArtifact`.
- **Dependências externas:** `ffmpeg`.
- **Pipeline:** suporte (UX).

---

### 2.7 Revisão e Seleção

#### `youcut/selector.py`
- **Função:** seleção múltipla interativa de clipes (Fluxo B) com timeout configurável e fallback automático.
- **APIs públicas:** `prompt_clip_selection(clips, clip_paths, timeout, console)`.
- **Dependências externas:** `questionary`, `rich`.
- **Pipeline:** revisão.

#### `youcut/reviewer.py`
- **Função:** TUI de aprovação por clipe — aprovar / rejeitar / editar título / regenerar thumbnail. Roda antes do upload (a menos que `--skip-review`).
- **APIs públicas:** `review_clips(clips, records, cut_mode, api_key)`.
- **Dependências externas:** `questionary`, `rich`.
- **Pipeline:** revisão.

---

### 2.8 Persistência e Exportação

#### `youcut/exporter.py`
- **Função:** escreve `clip_NN.txt` com título, descrição, hashtags, ideia de thumbnail, score viral e motivo da seleção.
- **APIs públicas:** `export_metadata(clip, index, output_dir)`.
- **Pipeline:** metadados.

#### `youcut/session_store.py`
- **Função:** persiste `SessionData` em `~/.youcut/sessions/<id>.json`. Permite o **Fluxo B** (gerar shorts a partir de cortes longos sem reprocessar o original).
- **APIs públicas:** `save_session()`, `load_session()`, `list_sessions()`.
- **Pipeline:** sessão.

---

## 3. Subpacote `youcut/uploader/`

> Cada plataforma implementa a interface `Uploader`. O orquestrador compartilha autenticação e relatório.

#### `youcut/uploader/__init__.py`
- **Função:** orquestrador. Resolve seleção (`--clips 1,3` ou `all`), autentica todas as plataformas (falhas individuais não derrubam o batch) e faz upload sequencial clipe × plataforma.
- **APIs públicas:** `upload_clips(clips, platforms, token_dir, clips_filter)`.
- **Pipeline:** upload.

#### `youcut/uploader/base.py`
- **Função:** contrato comum.
- **APIs públicas:** `Uploader` (ABC), `ClipMetadata`, `UploadResult`, `UploadReport`.
- **Pipeline:** upload (suporte).

#### `youcut/uploader/auth.py`
- **Função:** lê/grava/revoga tokens em `~/.youcut/credentials/<plataforma>.json` (modo 0600).
- **APIs públicas:** `get_token()`, `save_token()`, `revoke_token()`.
- **Pipeline:** upload (auth).

#### `youcut/uploader/metadata.py`
- **Função:** parse de `clip_NN.txt` e aplicação de **limites por plataforma** (YT: título 100, descrição 5000; IG/TT: caption 2200).
- **APIs públicas:** `parse_clip_metadata()`, `apply_platform_limits()`.
- **Pipeline:** upload (metadados).

#### `youcut/uploader/youtube.py`
- **Função:** OAuth via `client_secrets.json` (`InstalledAppFlow`); `videos.insert` resumable em chunks de 256KB; `thumbnails.set`. Valida thumbnail localmente (`png/jpg/jpeg`, ≤2 MB). Retry em 5xx.
- **APIs públicas:** `YouTubeUploader.authenticate()`, `YouTubeUploader.upload()`.
- **Dependências externas:** `google-api-python-client`, `google-auth-oauthlib`.
- **Pipeline:** upload (YouTube).

#### `youcut/uploader/instagram.py`
- **Função:** Graph API. OAuth com servidor HTTP local de callback; container → publish (Reels/Stories); polling de status.
- **APIs públicas:** `InstagramUploader.authenticate()`, `InstagramUploader.upload()`.
- **Dependências externas:** `httpx`.
- **Pipeline:** upload (Instagram).

#### `youcut/uploader/tiktok.py`
- **Função:** Content Posting API com OAuth PKCE. Upload chunked, `init` → polling até completar. Suporta `TIKTOK_POST_MODE=draft|direct` e `TIKTOK_PRIVACY_LEVEL`.
- **APIs públicas:** `TikTokUploader.authenticate()`, `TikTokUploader.upload()`.
- **Dependências externas:** `httpx`.
- **Pipeline:** upload (TikTok).

#### `youcut/uploader/report.py`
- **Função:** consolida resultados em `UploadReport` (Pydantic), grava `upload_report.json` no diretório do clipe e renderiza tabela no console.
- **APIs públicas:** `generate_report()`.
- **Pipeline:** upload (relatório).

---

## 4. Assets

#### `youcut/assets/Roboto-Regular.ttf`
- Fonte padrão para legendas (`captioner.py`), title overlays (`title_overlay.py`) e tarja editorial (`social_composer.py`).

---

## 5. Configuração & Documentação Externa

#### `pyproject.toml`
- **Build:** `hatchling`. **Entrypoint:** `youcut = "youcut.cli:app"`.
- **Core:** `typer`, `rich`, `pydantic>=2`, `pydantic-settings`, `anthropic>=0.40`, `faster-whisper`, `yt-dlp`, `Pillow`, `google-api-python-client`, `google-auth-oauthlib`, `httpx`, `questionary`, `openai`.
- **Extras:** `whisper-openai` (fallback), `face-tracking` (`mediapipe`, `pyannote.audio`, `opencv-python`), `dev` (`pytest`, `pytest-env`, `respx`).

#### `.env.example`
- Variáveis listadas: `ANTHROPIC_API_KEY` (obrigatório), `WHISPER_MODEL`, `CLAUDE_MODEL`, `YOUTUBE_CLIENT_SECRETS_FILE`, `INSTAGRAM_APP_ID`, `INSTAGRAM_APP_SECRET`, `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_USER_ID`, `TIKTOK_CLIENT_KEY`, `YOUCUT_COOKIES_FROM_BROWSER`, `YOUCUT_COOKIES_FILE`, `YOUCUT_YTDLP_JS_RUNTIMES`.

#### `prompt.md`
- Heurísticas para geração e seleção de thumbnails (lidas em runtime pelo `thumbnail_generator.py` e pela skill local `thumbnail-generator`).
- Princípios: imagem conta a história sozinha; texto ≤7% da área; paleta ciano/verde/amarelo/laranja; evitar vermelho dominante; não repetir o título.

#### `README.md`
- Manual humano de uso (instalação, comandos, exemplos). **Não é referência de arquitetura.**

#### `AGENTS.md`
- Visão geral do repositório para LLMs. Este `Arquitectadline.md` é o **detalhe arquivo a arquivo** que complementa o `AGENTS.md`.

#### `CLAUDE.md`
- Apenas re-exporta `AGENTS.md` (`@AGENTS.md`).

---

## 6. Diretórios de Apoio

#### `tests/` (~46 arquivos)
- Organizados por **prefixo temático**: `test_analyzer*`, `test_caption*`, `test_clipper*`, `test_diarizer*`, `test_face_tracker*`, `test_downloader*`, `test_yt_dlp_auth*`, `test_transcriber*`, `test_uploader_*`, `test_social_*`, `test_pipeline*`, `test_cli`, `test_config`, `test_models`, `test_exporter`, `test_preview`. Inclui suites `*_integration*` com marker `integration` (FFmpeg real). Fixtures compartilhadas em `conftest.py` e `create_fixtures.py`.

#### `tasks/` (PRDs versionados)
- Cada subpasta `prd-<slug>/` é um épico entregue, contendo `prd.md`, `techspec.md`, `tasks.md` e auxiliares. Exemplos: `prd-thumbnail-ai-selection`, `prd-face-tracking-speaker`, `prd-modo-automatico-youtube-social`, `prd-youtube-cuts`.

#### `templates/`
- `prd-template.md`, `task-template.md`, `tasks-template.md`, `techspec-template.md` — usados pelas skills locais para padronizar o ciclo de desenvolvimento.

#### `docs/`
- Site estático (`index.html`), `assets/`, `privacy-policy/`, `terms-of-service/`, arquivos de verificação de domínio TikTok. Necessário para o review oficial das plataformas.

#### `.agents/skills/`
- Skills locais que estruturam o fluxo de dev: `criar-prd`, `criar-techspec.md`, `criar-task`, `executar-task`, `executar-todas-tasks`, `executar-review`, `executar-bugfix`, `executar-qa`, `task-review`, `thumbnail-generator`, `ui-ux-pro-max`, `frontend-design`, `react`, `web-design-guidelines`, `maestro-e2e`, `skill-best-practices`, `ask-questions-if-underspecified`.

#### `output/`
- Saída padrão dos clipes (configurável via `OUTPUT_DIR`).
  ```
  output/
  ├─ downloads/<video>.mp4
  ├─ downloads/<video>_transcript.json
  └─ <video>/
      ├─ clip_NN.mp4
      ├─ clip_NN.txt
      ├─ thumbnails/clip_NN.png
      ├─ social_images/clip_NN.png
      └─ upload_report.json
  ```

#### `~/.youcut/` (estado do usuário, fora do repo)
- `credentials/{youtube,instagram,tiktok}.json` — tokens OAuth (modo 0600).
- `sessions/<id>.json` — `SessionData` para o Fluxo B **ou** `MotionComicSession` (discriminado por presença de `cast`+`panels`).

---

## 5b. Subpacote `youcut/comic/` (motion comic)

Pipeline isolado que converte vídeos curtos (≤120 s) em motion comics 9:16 reusando Whisper, Claude e adicionando `gpt-image-1` (imagens) + Runway `gen4_turbo` (image-to-video).

| Arquivo | Papel |
|---|---|
| `comic/__init__.py` | Re-exporta `run_comic_pipeline`, `PipelineCallbacks`, `ComicPipelineError`. |
| `comic/cli.py` | Subcomando `youcut comic <video>` com flags `--dry-run`, `--session`, `--regenerate-panel`, `--max-panels`, `--cost-cap`, `--yes`, `--no-progress`. UX rich + typer.confirm. |
| `comic/pipeline.py` | Orquestrador `run_comic_pipeline` — valida, transcreve, diariza, detecta cast, planeja painéis, estima custo, aplica cap, renderiza, compõe. |
| `comic/validator.py` | RF-01/RF-02: aceita mp4/mov/mkv/webm, rejeita >120 s. Retorna `VideoSpec`. |
| `comic/visual_analyzer.py` | MediaPipe (até 8 frames com rostos) + Claude vision (`extract_cast` tool) → `list[CastMember]`. Mapeia speaker→pessoa via `spatial_position` quando ≤2/≤2. Fallback genérico se 0 rostos. |
| `comic/cast_builder.py` | Gera ficha textual + imagem-âncora 1024×1024 por personagem (gpt-image-1). Idempotente: reusa arquivo `output/<v>/comic/cast/<id>.png` se existir. |
| `comic/script_planner.py` | Claude texto com tool `plan_panels` → `list[Panel]`. Valida cadência, soma e sobreposição; até 1 retry corretivo com hint da invariante violada. |
| `comic/panel_renderer.py` | Por painel: imagem-base 9:16 (gpt-image-1, refs até 3) + i2v (Runway `gen4_turbo`, ratio `720:1280`) ou fallback estático (`ffmpeg -loop 1 -t`). Paralelismo `asyncio.Semaphore(comic_i2v_concurrency)`. |
| `comic/composer.py` | Extend (`tpad=stop_mode=clone`) ou trim por painel → concat demuxer → mux áudio (`-c:a copy`, hash bit-idêntico) → burn legendas word-by-word. Saída em `output/<v>/motion_comic.mp4`. |
| `comic/cost_estimator.py` | `PriceTable` + `estimate_cost` + `enforce_cap` (pt-BR) + `preflight`. Defaults: anchor $0.04, base $0.04/img, i2v $0.05/s. |
| `comic/session.py` | `save/load/list_motion_comic_session` em `~/.youcut/sessions/<id>.json`. Discrimina de `SessionData` legado por presença de `cast`+`panels`. |
| `comic/run_report.py` | `build_run_report` + `write_run_report` em `output/<v>/comic/run_report.json` (`schema_version=1`). Inclui `total_cost_usd`, `n_panels`, `n_static_fallbacks`, `total_seconds`, `provider_latency_p50/p95`. |
| `comic/providers/images.py` | Protocol `ImageProvider` + `OpenAIImageProvider` (gpt-image-1) com retry exponencial. |
| `comic/providers/i2v.py` | Protocol `ImageToVideoProvider` + `RunwayProvider` (gen4_turbo) com polling até `SUCCEEDED`/`FAILED`/`CANCELED`, retry exponencial e fallback de download por data URL/http. |

---

## 7. Cheatsheet de Contratos Importantes

- **Cache de transcrição** é endereçado por **MD5 do vídeo**. Renomear o arquivo *não* invalida o cache; alterar bytes invalida.
- **Modo `youtube`** = stream copy (`-c copy`), 16:9, sem face tracking nem composição social. Thumbnails via DALL·E.
- **Modo `social`** = re-encode 9:16, opcionalmente com face tracking + diarização + composição editorial (`speaker_bottom_ai_top`).
- **`ANTHROPIC_API_KEY`** é validado no boot — `PipelineConfig` falha cedo se ausente.
- **`ffmpeg 8.1+ com libass`** é dependência crítica em runtime — `cli._check_ffmpeg()` falha cedo se não estiver no `PATH`.
- **Idioma do projeto** = **pt-BR** (mensagens, prompts, docs, commits).
- **Prompts do Claude** estão em `analyzer.py` (cortes) e `social_composer.py` (tarja). Mudanças devem preservar o contrato JSON.
- **Limites de upload:** YouTube título=100 / descrição=5000 · Instagram caption=2200 · TikTok caption=2200.
- **Thumbnail:** texto ≤7% da área · paleta ciano/verde/amarelo/laranja · evita vermelho dominante.
