# AGENTS.md — YouCut

Documento de contexto para LLMs que vão ler, modificar ou operar este repositório. O conteúdo aqui é descritivo do **estado atual do código**, não um manual de uso para humanos (esse fica em `README.md`).

> **📐 Mapa arquivo a arquivo:** veja [`ARQUITETURA_GUIDE_LINES.md`](./ARQUITETURA_GUIDE_LINES.md) para uma descrição **explícita** do que cada arquivo do projeto faz (APIs públicas, dependências externas e papel no pipeline). Use-o como referência detalhada complementar a este documento.

---

## 1. Visão Geral

**YouCut** é uma aplicação Python de **linha de comando** que automatiza a transformação de vídeos longos (lives, podcasts, palestras, vídeos longos do YouTube) em **clipes prontos para publicação**, usando IA em duas frentes principais:

- **transcrição automática** com Whisper (`faster-whisper` ou `openai-whisper`);
- **análise editorial** com Claude (Anthropic) para escolher trechos, gerar títulos, descrições, hashtags, ideias de thumbnail e prompts visuais;
- **geração de thumbnail** com DALL·E 3 (OpenAI) e composição local com Pillow;
- **upload automático** para YouTube, Instagram e TikTok.

**Para que serve:** acelerar o fluxo de um criador de conteúdo que precisa transformar 1 vídeo longo em vários clipes (longos para YouTube, curtos para redes sociais) — desde o download até a publicação, com revisão opcional.

- Linguagem: Python `>=3.11`
- Empacotamento: `hatchling` via `pyproject.toml`
- Entrypoint do CLI: `youcut = "youcut.cli:app"` (Typer)
- Rendering ASS: requer `ffmpeg 8.1+` compilado com `--enable-libass`

---

## 2. Comandos Principais (CLI)

O CLI é construído com `typer` e `rich`. Os subcomandos são:

### `youcut run SOURCE`
Pipeline **legado** focado em clipes virais curtos (9:16). Aceita URL do YouTube ou caminho local. Faz: download → transcrição → análise → corte → legendas → (opcional) overlay de título → metadados → (opcional) upload.

Flags relevantes: `--clips`, `--clip-count/-n`, `--style {word,phrase}`, `--dry-run`, `--title-overlay`, `--upload`, `--platforms`, `--log-level`, `--log-file`.

### `youcut cuts [SOURCE]`
Pipeline **principal** ("Cortes Inteligentes"), interativo. Suporta dois modos selecionados pelo usuário (ou por flag):
- **`youtube`** — clipes longos em paisagem `16:9`, duração ideal 15–25 min (mín. fallback 5 min), com geração de thumbnails via DALL·E 3.
- **`social`** — clipes verticais `9:16` para TikTok / Reels / Shorts, até ~3 min.

Subfluxos:
- **Fluxo A**: cortes longos (modo `youtube`) com revisão e, ao final, oferta automática para gerar shorts a partir dos cortes recém-gerados.
- **Fluxo B**: gera shorts a partir de cortes existentes, **sem reprocessar** o vídeo original (reusa transcrição cacheada e a sessão salva). Acessível também via `youcut cuts --history`.
- **Fluxo C**: modo `social` direto da URL, sem sessão prévia.

Flags relevantes: `--max-clips/-n`, `--skip-review`, `--upload`, `--platforms`, `--thumbnail-text`, `--history/-H`.

### `youcut auth login|revoke|status`
Gerencia tokens OAuth de YouTube / Instagram / TikTok, salvos em `~/.youcut/credentials/<plataforma>.json`.

### `youcut comic <video>`
Pipeline **motion comic**: aceita um vídeo local curto (≤120 s) e gera um MP4 9:16 (1080×1920) com personagens ilustrados, áudio original preservado e legendas queimadas. Suporta 4 engines de animação (via `--engine`):

- **`scenes`** (default, recomendado) — divide a transcrição em N cenas narrativas (Claude scene planner), gera 1 master por cena com **anchor visual canônico** pra consistência de estilo, faz **word-level visual attribution** (Claude vision identifica quem articula a boca em cada palavra), aplica **smoothing conservador** das atribuições, **gap absorption** (laughs/expressões não-fala não viram freeze) e **crossfade** entre chunks. Emite versões com e sem legenda + watermark configurável (`comic_scenes_watermark_text`). Implementação em `youcut/comic/scenes_pipeline.py`.
- **`prunaai`** — gera o vídeo inteiro em 1 chamada à IA (`prunaai/p-video-avatar`). Mais barato (~$0.05/vídeo) e mais rápido (~70s), mas sem controle narrativo nem lip-sync correto em diálogos.
- **`panels`** — modo clássico (Hailuo i2v por painel). Mais caro e demorado mas com máximo controle por beat.
- **`remotion`** — render local programático via Node.js + React/TSX (Remotion 4.x). IA usada apenas para gerar anchors e uma **mouth sheet 4-em-1** por personagem (~$0.04 cada chamada `gpt-image-1`); a animação (lip-sync sílaba-level via `pyphen`, Ken Burns, transições crossfade/cut/wipe, idle blink/breathe, shake) executa em código local sem chamadas pagas adicionais. Custo por vídeo ≤ $1, determinístico, suporta modo preview interativo via Remotion Studio (`--no-preview` força headless; `--yes/-y` também implica `--no-preview`). Implementação em `youcut/comic/remotion_pipeline.py`.

Flags relevantes: `--engine {scenes,prunaai,panels,remotion}`, `--max-panels/-n`, `--cost-cap`, `--dry-run`, `--session <id>`, `--regenerate-panel I[,J,...]`, `--yes/-y`, `--no-progress`, `--no-preview` (engine `remotion`).

Configs específicas do `scenes` (em `PipelineConfig`): `comic_scenes_count` (default 4), `comic_scenes_crossfade_dur` (0.25s), `comic_scenes_gap_absorb_threshold` (0.5s), `comic_scenes_smooth_attribution` (True), `comic_scenes_inter_call_pause_s` (11s para rate-limit Replicate), `comic_scenes_watermark_text`, `comic_scenes_emit_no_subs_version`, `comic_scenes_style_ref_image`.

Configs específicas do `remotion` (em `PipelineConfig`): `comic_remotion_fps` (30), `comic_remotion_node_bin` (`node`), `comic_remotion_concurrency` (None), `comic_remotion_studio_port` (3000), `comic_remotion_kenburns_default_scale` (1.12), `comic_remotion_idle_blink_period_sec` (4.5), `comic_remotion_pyphen_locale_fallback` (`pt_BR`). Reutiliza as chaves de watermark do engine `scenes` (`comic_scenes_watermark_text`/`opacity`/`y_from_bottom`).

Variáveis adicionais: `RUNWAY_API_KEY` (obrigatória pro engine `panels`), `REPLICATE_API_TOKEN` (obrigatória pros engines `prunaai` e `scenes`), `OPENAI_API_KEY` (obrigatória para todos os engines do `comic`). O engine `remotion` requer **Node.js ≥ 20** instalado no PATH (não há chave de API adicional).

---

## 3. Pipeline (Como Funciona)

### 3.1 Pipeline canônico (modo `social`)
1. **Download** (`downloader.py`) — `yt-dlp` para URLs, ou resolve um arquivo local. Suporta cookies de browser/arquivo via `YOUCUT_COOKIES_FROM_BROWSER` / `YOUCUT_COOKIES_FILE` (módulo `yt_dlp_auth.py`).
2. **Transcrição** (`transcriber.py`) — `faster-whisper` (default) com fallback para `openai-whisper`. Faz cache por hash MD5 do vídeo em `<video>_transcript.json` ao lado do arquivo. Retorna `TranscriptionResult` (segments + word-level timestamps).
3. **Análise IA** (`analyzer.py`) — Envia a transcrição em chunks de 30 min para o Claude (`claude-sonnet-4-6` por padrão) com prompts parametrizados pelo `cut_mode`. Retorna lista de `ViralClip` com:
   - `title`, `reason`, `viral_score` (0–10), `start_time`, `end_time`
   - `description`, `hashtags`, `thumbnail_idea`, `thumbnail_text`
   - `social_hook_title`, `social_image_prompt`, `social_visual_style`
4. **Corte** (`clipper.py`) — `ffmpeg`:
   - modo `youtube`: stream copy puro (sem re-encode) usando `-c copy`.
   - modo `social`: re-encode com filtro `scale+crop` para 1080×1920, ou modo "blur background" alternativo.
   - quando `social_layout_mode == "classic"`, queima legendas via `CaptionBurner` no fim do corte.
5. **Face tracking opcional** (`face_tracker.py`) — usa MediaPipe + opcionalmente diarização (`diarizer.py` com pyannote) para detectar speaker ativo e gerar crops 9:16 com padding vertical de 40%, suportando split-screen quando há dois speakers.
6. **Legendas** (`captioner.py` + `caption_burner.py`) — gera arquivo `.ass` (Advanced SubStation) com timestamps por palavra (`word`) ou por segmento (`phrase`) e queima via `ffmpeg -vf ass=...`. Requer `libass`.
7. **Composição social editorial** (`social_composer.py`) — quando `social_layout_mode == "speaker_bottom_ai_top"`, monta um canvas 1080×1920 com imagem gerada por IA no topo, label/hook no meio e o vídeo original (com face tracking) embaixo. Cores e textos da label são gerados pelo Claude (paleta padrão amarelo/laranja).
8. **Title overlay** (`title_overlay.py`) — opcional; queima um card com o título nos primeiros 5s do clipe.
9. **Thumbnail** (`thumbnail_generator.py`) — pipeline de seleção de frame (heurística de brilho/contraste/clareza) e geração via DALL·E 3 com fallback local em Pillow. Texto na thumb limitado a ≤7% da área (heurística do `prompt.md`); paleta favorece ciano/verde/amarelo/laranja, evita vermelho dominante.
10. **Exporter** (`exporter.py`) — escreve `clip_NN.txt` com título / descrição / hashtags / ideia de thumb / nota de viralidade / motivo da seleção.
11. **Reviewer** (`reviewer.py`) — TUI interativa via `questionary` para aprovar / rejeitar / editar título / regenerar thumbnail, antes do upload. Pulável com `--skip-review`.
12. **Upload** (`uploader/`) — opcional; ver §5.
13. **Sessão** (`session_store.py`) — toda execução do `cuts` modo `youtube` persiste `SessionData` em `~/.youcut/sessions/<id>.json`, permitindo o Fluxo B.

### 3.2 Atalhos do modo `youtube`
- Análise pede 15–25 min com fallback de 5 min se não houver trechos longos suficientes.
- Títulos: 5–9 palavras, idealmente ≤30 caracteres.
- Sem face tracking nem composição social — é stream copy direto.

### 3.3 Pipeline `youcut comic` (motion comic)
1. **Validação** (`comic/validator.py`) — aceita mp4/mov/mkv/webm; rejeita >120 s com mensagem em pt-BR.
2. **Transcrição** (`transcriber.py`) — reaproveita o stack existente.
3. **Diarização** (`diarizer.py`) — fallback `SPEAKER_00` quando sem token.
4. **Visual analyzer** (`comic/visual_analyzer.py`) — MediaPipe amostra até 8 frames com rostos; Claude vision extrai cast (gênero, idade, cabelo, barba, roupa, adereços, animais/objetos).
5. **Cast builder** (`comic/cast_builder.py`) — gera ficha textual + imagem-âncora 1024×1024 (gpt-image-1 com `input_fidelity="high"`) **uma única vez** por personagem; idempotente em retomadas.
6. **Script planner** (`comic/script_planner.py`) — Claude texto divide a transcrição em painéis 2–5 s respeitando cadência (≥1/5 s, ≤1/1,5 s) e soma ≈ duração do áudio (±0,2 s); retry corretivo automático.
7. **Cost estimator + cap** (`comic/cost_estimator.py`) — exibe breakdown e enforcement do teto duro (`comic_cost_cap_usd`, default $10) **antes** de qualquer chamada paga.
8. **Panel renderer** (`comic/panel_renderer.py`) — para cada painel: imagem-base 9:16 (gpt-image-1) com fichas-âncora como `reference_images` → mini-clipe 2–5 s (Runway `gen4_turbo`, ratio `720:1280`) → fallback estático via `ffmpeg -loop 1` quando i2v falha. Paralelismo via `asyncio.Semaphore(comic_i2v_concurrency)`.
9. **Composer** (`comic/composer.py`) — extend (`tpad=stop_mode=clone`) ou trim por painel → concat demuxer → mux do áudio original com `-c:a copy` → queima legendas palavra-a-palavra reusando `youcut.captioner.build_ass_for_words`.
10. **Sessão e relatório** (`comic/session.py`, `comic/run_report.py`) — persistência em `~/.youcut/sessions/<id>.json` (discriminada por `cast`+`panels`) e métricas em `output/<video>/comic/run_report.json` (`schema_version=1`).

### 3.4 Pipeline `youcut comic --engine remotion` (render local programático)
1. **Validação + transcrição + diarização + cast** — reusa o stack do pipeline `comic` (passos 1–4 acima).
2. **Cost estimator + cap** (`comic/cost_estimator.py`) — branch dedicado: custo = `n_cast × (anchor_image_usd + mouth_sheet_usd)`, render local = $0. Cap respeitado normalmente (`comic_cost_cap_usd`).
3. **Cast anchors** (`comic/cast_builder.py`) — gera ficha-âncora 1024×1024 por personagem (uma única chamada `gpt-image-1`, idempotente).
4. **Mouth sheets** (`comic/mouth_shapes.py`) — para cada personagem, **uma chamada `gpt-image-1`** gera uma sheet 1024×1024 com 4 mouth shapes em grid 2×2 (closed/open_mid/open_wide/open_round). Validador Pillow + retry com prompt corretivo + fallback (4 chamadas separadas) em caso de falha. Cells nativas 512×512.
5. **Syllable mapper** (`comic/syllable_mapper.py`) — função pura: `WordTimestamp[]` → `MouthEvent[]`. Hifeniza cada palavra via `pyphen` (locale derivado da transcrição), distribui o tempo proporcionalmente entre as sílabas, mapeia a vogal dominante (a/á/â/ã/e/é/ê → `OPEN_WIDE`; i/í/y → `OPEN_MID`; o/ó/ô/õ/u/ú → `OPEN_ROUND`; consoante terminal → `CLOSED`). Smoothing: sílabas < 80ms são fundidas com a vizinha mais curta. Gaps > 120ms entre palavras viram `CLOSED`.
6. **Render Remotion** (`comic/providers/remotion_renderer.py` + `comic/remotion_project/`) — Python serializa `RemotionInputProps` (com cenas, lipsync por scene, mouth sheets) em JSON e invoca `node render.mjs --props ... --out ...` por subprocess. O projeto vendored em TypeScript usa Remotion 4.x (`<Composition>` + `<Sequence>` por cena + `<AbsoluteFill>` + `<Audio>`); cada `Scene.tsx` aplica Ken Burns via `interpolate(frame, [0,N], [scaleFrom,scaleTo])` + transitionIn (`crossfade` opacity, `cut`, `wipe` clipPath); `Character.tsx` lê `lipSync` e exibe a célula correta da mouth sheet via `backgroundImage`+`backgroundPosition`; `Shake.tsx` aplica `transform: translate` por janelas. Progress emitido como JSON-lines em stdout.
7. **Composer single-clip** (`comic/composer.compose_from_single_clip`) — recebe o MP4 do Remotion e emite duas versões: `motion_comic_no_subs.mp4` (stream-copy) e `motion_comic.mp4` (legendas word-by-word + watermark queimados via ffmpeg). Reusa `build_ass_for_words` e o filter `drawtext` do engine `scenes`.
8. **Sessão** — `MotionComicSession` persistida em `~/.youcut/sessions/<id>.json`.

Health-checks específicos: `RemotionRenderer` valida `node --version ≥ 20` e roda `npm install` automático no `remotion_project/` quando `node_modules/` ausente. Em ambiente sem TTY/DISPLAY, o orquestrador (`comic/remotion_pipeline.py`) cai automaticamente para modo headless (RF-18).

---

## 4. Estrutura do Pacote `youcut/`

```
youcut/
  __init__.py             versão 0.1.0
  cli.py                  ponto de entrada Typer (run, cuts, auth)
  config.py               PipelineConfig (pydantic-settings, lê .env)
  models.py               Pydantic: ViralClip, ClipRecord, SessionData,
                          TranscriptionResult, CaptionBurnResult, etc.

  downloader.py           yt-dlp wrapper + VideoDownloadError
  yt_dlp_auth.py          resolve cookies / runtimes JS para yt-dlp
  url_utils.py            normaliza URLs do YouTube
  video_metadata.py       extrai title/duration sem baixar o vídeo

  transcriber.py          faster-whisper / openai-whisper + cache MD5
  analyzer.py             Claude API; prompts por cut_mode
  diarizer.py             pyannote.audio (opcional, extra `face-tracking`)
  face_tracker.py         MediaPipe; ROIs 9:16 com padding e split-screen
  clipper.py              ffmpeg; modos youtube (stream-copy) e social
  caption_burner.py       wrapper alto-nível p/ queimar legenda no clipe
  captioner.py            geração de .ass (palavra-a-palavra ou por frase)
  title_overlay.py        card com título nos primeiros 5s
  social_composer.py      layout editorial 1080x1920 (img IA + tarja + vídeo)
  thumbnail_generator.py  seleção de frame + DALL·E 3 + composição Pillow
  preview.py              gera GIF/preview curto p/ tabela do CLI
  selector.py             seleção interativa de clipes (Fluxo B)
  reviewer.py             TUI de aprovação por clipe
  exporter.py             escreve clip_NN.txt
  session_store.py        persistência em ~/.youcut/sessions/

  comic/                  pipeline motion comic (`youcut comic`)
    __init__.py           re-export `run_comic_pipeline`
    cli.py                subcomando Typer + UX interativa
    pipeline.py           orquestrador (validate→transcribe→…→compose)
    validator.py          RF-01/RF-02 (mp4/mov/mkv/webm, ≤120s, pt-BR)
    visual_analyzer.py    MediaPipe + Claude vision → list[CastMember]
    cast_builder.py       fichas-âncora 1024×1024 (idempotente)
    script_planner.py     Claude texto → list[Panel] (cadência ≥1/5s)
    panel_renderer.py     gpt-image-1 + Runway gen4_turbo + fallback
    composer.py           concat + audio mux + legendas word-by-word
                          + helper compose_from_single_clip (engine remotion)
    cost_estimator.py     PriceTable + enforce_cap pt-BR (branches por engine)
    session.py            save/load/list MotionComicSession
    run_report.py         run_report.json (schema_version=1)
    mouth_shapes.py       gpt-image-1 sheet 4-em-1 + Pillow crop (engine remotion)
    syllable_mapper.py    pyphen + heurística vogal→MouthShape (engine remotion)
    remotion_pipeline.py  orquestrador do engine `--engine remotion`
    providers/
      images.py           OpenAIImageProvider (gpt-image-1)
      i2v.py              RunwayProvider (gen4_turbo)
      remotion_renderer.py wrapper subprocess Node + Studio launcher
    remotion_project/     projeto Remotion vendored (Node + React + TS)
      package.json        deps fixadas em Remotion 4.0.420 + React 19.2
      tsconfig.json       strict, ES2022, react-jsx, Bundler resolution
      remotion.config.ts  codec h264, pixelFormat yuv420p
      render.mjs          entrypoint Node CLI (bundle + selectComposition + renderMedia)
      src/
        index.ts          registerRoot(RemotionRoot)
        Root.tsx          <Composition id="ComicVideo" calculateMetadata=...>
        types.ts          espelho TS de RemotionInputProps Pydantic
        ComicVideo.tsx    AbsoluteFill + Audio + map de Scenes
        components/
          Scene.tsx       Ken Burns + transitionIn + Shake wrapper
          Character.tsx   lip-sync via mouth sheet cell + blink + breathe
          Shake.tsx       translate sinusoidal por janelas de impacto

  uploader/
    __init__.py           orquestra upload_clips() multi-plataforma
    base.py               Uploader (ABC) + ClipMetadata + UploadResult
    auth.py               leitura/gravação de tokens em ~/.youcut/credentials
    metadata.py           parse de clip_NN.txt + limites por plataforma
    youtube.py            Google API (resumable upload + thumbnail.set)
    instagram.py          Graph API (container -> publish)
    tiktok.py             OAuth PKCE + Content Posting API (draft|direct)
    report.py             relatório final consolidado

  assets/
    Roboto-Regular.ttf    fonte default p/ legendas / overlays / labels
```

### Diretórios extras
- `tests/` — pytest, com marker `integration` para testes que geram vídeos sintéticos via FFmpeg. ~40+ arquivos de teste (`test_pipeline*.py`, `test_uploader_*.py`, `test_face_tracker*.py`, etc.).
- `tasks/` — PRDs versionados de cada feature já entregue (cada subpasta = um épico).
- `templates/` — templates de PRD / Tech Spec / Tasks usados pelas skills do `.agents/skills/`.
- `.agents/skills/` — skills locais (`criar-prd`, `criar-techspec.md`, `executar-task`, `task-review`, `executar-qa`, `executar-bugfix`, `thumbnail-generator`, etc.) que estruturam o ciclo de desenvolvimento.
- `docs/` — site estático + termos / privacy policy publicados (necessários para review do TikTok).
- `output/` — destino padrão dos clipes (configurável via `OUTPUT_DIR`).
- `~/.youcut/credentials/` e `~/.youcut/sessions/` — estado persistente do usuário.

---

## 5. Upload (`youcut/uploader/`)

Cada plataforma implementa a interface `Uploader` (`base.py`):

| Plataforma | Auth                                 | Endpoint principal                          | Observações |
|------------|--------------------------------------|---------------------------------------------|-------------|
| YouTube    | OAuth client_secrets (`googleapiclient`) | `videos.insert` (resumable) + `thumbnails.set` | Thumbnail validada localmente: existe, ext em `{png,jpg,jpeg}`, ≤2 MB. Falha de thumb não derruba o vídeo (publicação parcial com aviso). |
| Instagram  | Graph API token                      | container → publish                         | Reels/Stories. |
| TikTok     | OAuth PKCE                           | Content Posting API                         | `TIKTOK_POST_MODE=draft` (default, vai pra inbox) ou `direct` (publica via API). Modo `direct` exige escopo `video.publish` e usa `TIKTOK_PRIVACY_LEVEL` (default `SELF_ONLY`). |

`upload_clips()` em `uploader/__init__.py` é o orquestrador:
1. resolve seleção de clipes (`--clips 1,3` ou `all`);
2. autentica todas as plataformas (falhas individuais não derrubam o batch);
3. faz upload sequencial por clipe × plataforma;
4. gera relatório (`UploadReport`) salvo via `report.py` no diretório do clipe.

---

## 6. Configuração (`PipelineConfig`)

`pydantic-settings` lê `.env` na raiz e variáveis de ambiente. Campos relevantes:

| Campo                              | Default                  | Notas |
|------------------------------------|--------------------------|-------|
| `anthropic_api_key`                | **obrigatório**          | valida no `model_validator`; falha cedo se ausente |
| `whisper_model`                    | `medium`                 | qualquer modelo aceito pelos backends Whisper |
| `claude_model`                     | `claude-sonnet-4-6`      | usado pelo analyzer e pelo social_composer |
| `clip_count`                       | `5`                      | usado pelo `run` legado |
| `subtitle_style`                   | `word`                   | `word` ou `phrase` |
| `output_dir`                       | `output`                 | |
| `cut_mode`                         | `social`                 | `social` ou `youtube` |
| `max_clips`                        | `None`                   | usado pelo `cuts`; `None` deixa a IA decidir |
| `dry_run`                          | `False`                  | só análise, sem render |
| `title_overlay`                    | `False`                  | card de título nos 5s iniciais |
| `upload`, `platforms`, `clips`     | —                        | controle de publicação |
| `vertical_fill_mode`               | `fill_crop`              | alternativa: `blur_background` |
| `face_tracking`                    | `False`                  | habilita pipeline MediaPipe + diarização |
| `huggingface_token`                | `None`                   | necessário p/ `pyannote.audio` no diarizer |
| `face_detection_confidence`        | `0.5`                    | |
| `social_layout_mode`               | `classic`                | `classic` ou `speaker_bottom_ai_top` |
| `social_layout_*`                  | vários                   | controla altura da imagem topo, banda do título, paleta |
| `openai_api_key`                   | `None`                   | obrigatório para gerar thumbnail/imagem social via DALL·E 3 |
| `session_timeout_minutes`          | `7`                      | timeout de inatividade no card de oferta do Fluxo B |
| `comic_animation_engine`           | `scenes`                 | `scenes` (default), `prunaai`, `panels`, `remotion` |
| `comic_remotion_fps`               | `30`                     | fps da composição Remotion |
| `comic_remotion_node_bin`          | `node`                   | path do binário Node usado pelo subprocess |
| `comic_remotion_concurrency`       | `None`                   | threads de CPU para `renderMedia` (Remotion default) |
| `comic_remotion_studio_port`       | `3000`                   | porta do Remotion Studio em modo preview |
| `comic_remotion_kenburns_default_scale` | `1.12`              | escala alvo do Ken Burns aplicado por cena |
| `comic_remotion_idle_blink_period_sec`  | `4.5`               | periodicidade do idle blink do personagem |
| `comic_remotion_pyphen_locale_fallback` | `pt_BR`             | locale `pyphen` quando o idioma da transcrição é desconhecido |

### Variáveis de ambiente extras (não em `PipelineConfig`)
- `YOUTUBE_CLIENT_SECRETS_FILE` — caminho do `client_secrets.json`
- `YOUCUT_COOKIES_FROM_BROWSER` ou `YOUCUT_COOKIES_FILE` — auth do `yt-dlp` (use só uma)
- `YOUCUT_YTDLP_JS_RUNTIMES` — ex: `node`, p/ resolver desafios JS do YouTube
- `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_POST_MODE`, `TIKTOK_PRIVACY_LEVEL`, `TIKTOK_DISABLE_COMMENT|DUET|STITCH`
- `INSTAGRAM_*` — credenciais Graph API

---

## 7. Modelos de Dados-Chave (`youcut/models.py`)

- **`ViralClip`** — saída do `analyzer`. Inclui campos editoriais (`title`, `description`, `hashtags`, `thumbnail_idea`, `thumbnail_text`) e campos sociais (`social_hook_title`, `social_image_prompt`, `social_visual_style`).
- **`ClipRecord`** — estado persistido do clipe gerado: paths, status de aprovação, `upload_status` por plataforma, `youtube_video_id/url`, flags de legenda.
- **`SessionData`** — sessão completa do `cuts` (URL, modo, transcript cache, clips, output dir).
- **`TranscriptionResult`** / `TranscriptionSegment` / `WordTimestamp` — saída do Whisper, com timestamps palavra a palavra.
- **`SpeakerSegment`**, **`CropRegion`**, **`FaceTrackingResult`** — do face tracker.
- **`ThumbnailFrameResult`** — resultado da seleção/geração de thumb (`selection_method` e `generation_method` ∈ {`ai`, `local`}).
- **`CaptionBurnResult`** — wrapper Path-like com `captions_applied` + `warning`.

---

## 8. Heurísticas de Thumbnail (Resumo do `prompt.md`)

O pipeline de thumb segue essas regras (estão codificadas em `thumbnail_generator.py`):
- texto embutido **opcional**, no máximo ~7% da área da imagem;
- paleta preferida: **ciano, verde, amarelo, laranja**; evita vermelho dominante;
- a imagem deve "contar a história sozinha" — não repetir o título literal;
- preferir frames mais brilhantes, expressivos, com rosto/recorte forte quando disponíveis;
- múltiplos rostos em cena reforçam tensão / reação / conversa.

A skill local `.agents/skills/thumbnail-generator/SKILL.md` (e `prd.md`) é referenciada pelo código em runtime para gerar thumbnails consistentes.

---

## 9. Saída no Disco

```
output/
├─ downloads/
│   ├─ <video>.mp4
│   └─ <video>_transcript.json     # cache MD5 da transcrição
└─ <video>/
    ├─ clip_01.mp4
    ├─ clip_01.txt                 # título, descrição, hashtags, motivo, score
    ├─ thumbnails/clip_01.png      # quando aplicável (modo youtube)
    ├─ social_images/clip_01.png   # imagem topo do layout social_composer
    └─ ...

~/.youcut/
├─ credentials/{youtube,instagram,tiktok}.json
└─ sessions/<session_id>.json
```

---

## 10. Testes

- `pytest` (configurado em `pyproject.toml`, `testpaths = ["tests"]`)
- Marker `integration` para testes que invocam FFmpeg de verdade — pulam em CI sem binário compatível.
- Cobertura inclui: pipelines completos (`test_pipeline*.py`), uploaders (`test_uploader_*.py`), face tracking (`test_face_tracker*.py`), composers e geração de thumb. Total: ~45 arquivos de teste.

Comando: `pytest` (instale com `pip install -e .[dev]`).

---

## 11. Dependências (de `pyproject.toml`)

**Core:**
`typer`, `rich`, `pydantic>=2`, `pydantic-settings`, `anthropic>=0.40`, `faster-whisper`, `yt-dlp`, `Pillow`, `google-api-python-client`, `google-auth-oauthlib`, `httpx`, `questionary`, `openai`, `pyphen>=0.14` (hifenização para o engine `remotion`).

**Extras:**
- `whisper-openai`: `openai-whisper` como fallback de transcrição.
- `face-tracking`: `mediapipe>=0.10`, `pyannote.audio>=3.1`, `opencv-python>=4.8`.
- `dev`: `pytest`, `pytest-env`, `respx`.

**Externo (não-Python):**
- `ffmpeg 8.1+` com `--enable-libass` (no Homebrew, pelo tap `homebrew-ffmpeg/ffmpeg`).
- Opcional: `node` no PATH para `YOUCUT_YTDLP_JS_RUNTIMES=node`.
- **`Node.js ≥ 20`** no PATH para o engine `youcut comic --engine remotion` (instalar via `brew install node` ou nvm). O projeto Remotion vendored em `youcut/comic/remotion_project/` instala suas próprias deps via `npm install` na primeira execução (~50 MB de `node_modules`, lockfile commitado para reprodutibilidade).

---

## 12. Convenções para LLMs Operando neste Repositório

- **Antes de modificar qualquer módulo, consulte [`ARQUITETURA_GUIDE_LINES.md`](./ARQUITETURA_GUIDE_LINES.md)** — ele lista, arquivo a arquivo, o papel de cada módulo, suas APIs públicas e onde se encaixam no pipeline. É a fonte canônica para navegar o código sem precisar abrir cada arquivo.
- O idioma padrão é **português (pt-BR)** — mensagens de erro, prompts de IA, docs e commits seguem esse padrão.
- Prompts do Claude estão centralizados em `analyzer.py` (cortes) e `social_composer.py` (label editorial). Mudanças em prompts devem preservar a contratualização do JSON de saída.
- Toda nova feature começa por um PRD em `tasks/prd-<slug>/` (templates em `templates/`). As skills `.agents/skills/criar-prd`, `criar-techspec.md`, `criar-task`, `executar-task` formalizam esse fluxo.
- Não comitar `.env`, `cookies.txt`, tokens em `~/.youcut/credentials/`, nem `youtube-oauth.json`.
- O `ANTHROPIC_API_KEY` é validado no boot — qualquer entrypoint que instancie `PipelineConfig` precisa dele.
- `ffmpeg` é dependência crítica em runtime — `cli._check_ffmpeg()` falha cedo se não estiver no PATH.
- Cache de transcrição é endereçado por **MD5 do arquivo de vídeo**: renomear o arquivo invalida o cache; alterar bytes invalida.
