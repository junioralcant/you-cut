# YouCut

Aplicação de linha de comando que automatiza a transformação de vídeos longos em clipes prontos para publicação **e** vídeos curtos em motion comics animados.

## Comandos

| Comando | Para quê |
|---|---|
| `youcut run <url>` | Pipeline legado: baixa → transcreve → analisa com Claude → corta clipes virais 9:16 com legendas. |
| `youcut cuts <url>` | **Cortes inteligentes** (recomendado): clipes longos 16:9 pra YouTube ou shorts 9:16 pra TikTok/Reels/Shorts. |
| `youcut comic <video>` | **Motion comic** animado a partir de vídeo curto (≤120s) — personagens cartoon ilustrados com lip-sync por palavra, áudio original preservado. |
| `youcut auth …` | Gerencia OAuth de YouTube/Instagram/TikTok pro upload automático. |

O fluxo padrão (`run` / `cuts`):

1. baixa um vídeo do YouTube ou usa um arquivo local;
2. transcreve o áudio com Whisper;
3. envia a transcrição para o Claude identificar os melhores trechos;
4. corta os clipes em formato vertical `1080x1920`;
5. adiciona legendas embutidas;
6. (opcional) queima o título nos primeiros 5 segundos com `--title-overlay`;
7. exporta metadados prontos para publicação.

O comando `youcut comic` segue um pipeline diferente — ver [seção dedicada](#motion-comic--youcut-comic) abaixo.

## Requisitos

- Python `3.11+`
- `ffmpeg 8.1+` compilado com `--enable-libass`, instalado e disponível no `PATH`
- chave da API da Anthropic

## Instalação

### ffmpeg com libass (obrigatório)

O projeto usa `libass` para renderizar legendas ASS/SSA diretamente no vídeo. O `ffmpeg` padrão do Homebrew **não inclui** essa lib — instale via tap específico:

```bash
brew tap homebrew-ffmpeg/ffmpeg
brew install homebrew-ffmpeg/ffmpeg/ffmpeg
```

Verifique se a instalação está correta:

```bash
ffmpeg -buildconf 2>/dev/null | grep libass
# deve retornar: --enable-libass
```

> **macOS:** antes de instalar, certifique-se de que o Xcode Command Line Tools está atualizado (`sudo xcode-select --install`). O tap `homebrew-ffmpeg` compila o ffmpeg do source e exige ferramentas de build atualizadas.

### Dependências Python

Clone o projeto e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Se quiser rodar os testes:

```bash
pip install -e .[dev]
```

### Node.js ≥ 20 (apenas para `youcut comic --engine remotion`)

O engine `remotion` (render local programático para motion comics) exige Node.js no PATH. Instale via Homebrew (macOS), `nvm` ou direto do site oficial:

```bash
# macOS
brew install node

# Linux/macOS via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 20

# Verifique a versão (deve ser ≥ 20)
node --version
```

Os outros engines (`scenes`, `prunaai`, `panels`) **não exigem** Node.js. As dependências npm do projeto Remotion vendored (`youcut/comic/remotion_project/`) são instaladas automaticamente via `npm install` na primeira execução do engine (~50 MB de `node_modules`). O `package-lock.json` é commitado para reprodutibilidade.

## Configuração

Crie um arquivo `.env` na raiz do projeto com base no `.env.example`:

```env
ANTHROPIC_API_KEY=sua_chave_aqui
```

Variáveis opcionais:

```env
# WHISPER_MODEL=medium
# CLAUDE_MODEL=claude-sonnet-4-6
# CLIP_COUNT=5
# SUBTITLE_STYLE=word
# OUTPUT_DIR=output
# DRY_RUN=false

# Necessário para o modo YouTube (geração de thumbnails via gpt-image-1.5) e
# para o `youcut comic` (geração de masters/anchors via gpt-image-1):
# OPENAI_API_KEY=sua_chave_openai

# Necessário para o `youcut comic` engines `scenes` (default) e `prunaai`
# (provider de animação prunaai/p-video-avatar e Hailuo i2v):
# REPLICATE_API_TOKEN=sua_chave_replicate

# Necessário apenas para publicar no YouTube via youcut cuts --upload:
# YOUTUBE_CLIENT_SECRETS_FILE=caminho/para/client_secrets.json

# Opcional: reutiliza a sessão autenticada do navegador para acessar vídeos
# do YouTube via yt-dlp quando o site exigir login/anti-bot.
# Use apenas uma das duas opções abaixo:
# YOUCUT_COOKIES_FROM_BROWSER=chrome
# YOUCUT_COOKIES_FILE=/caminho/para/cookies.txt

# Opcional: habilita runtimes JS extras do yt-dlp para resolver desafios
# modernos do YouTube. Exemplo comum:
# YOUCUT_YTDLP_JS_RUNTIMES=node
```

## Como usar

Depois da instalação, o comando principal é:

```bash
youcut run SOURCE
```

Onde `SOURCE` pode ser:

- uma URL do YouTube
- um caminho local para um arquivo de vídeo

Exemplos:

```bash
youcut run "https://www.youtube.com/watch?v=exemplo"
```

```bash
youcut run "./meu-video.mp4"
```

Gerando 3 clipes:

```bash
youcut run "./meu-video.mp4" --clips 3
```

Gerando 3 clipes com a flag explícita:

```bash
youcut run "./meu-video.mp4" --clip-count 3
```

Usando legenda por frase:

```bash
youcut run "./meu-video.mp4" --style phrase
```

## Acesso autenticado ao YouTube no yt-dlp

Se o YouTube retornar erros como `HTTP 429` ou `Sign in to confirm you're not a bot`, o
`youcut` pode reutilizar sua sessão autenticada do navegador via variáveis de ambiente.

Exemplo no `.env`:

```env
YOUCUT_COOKIES_FROM_BROWSER=chrome
```

Ou com um arquivo exportado de cookies:

```env
YOUCUT_COOKIES_FILE=/caminho/para/cookies.txt
```

Notas:

- defina apenas uma entre `YOUCUT_COOKIES_FROM_BROWSER` e `YOUCUT_COOKIES_FILE`;
- essa autenticação vale para leitura de metadados e download do vídeo;
- se o yt-dlp reclamar de `n challenge solving failed`, configure `YOUCUT_YTDLP_JS_RUNTIMES=node`;
- isso é separado da autenticação OAuth usada para upload no YouTube.

Executando apenas a análise, sem gerar vídeos:

```bash
youcut run "./meu-video.mp4" --dry-run
```

Queimando o título no início de cada clipe:

```bash
youcut run "./meu-video.mp4" --title-overlay
```

Salvando logs em arquivo:

```bash
youcut run "./meu-video.mp4" --log-level DEBUG --log-file output/youcut.log
```

## Opções do comando

```bash
youcut run SOURCE [OPTIONS]
```

Opções disponíveis:

- `--clips`: quantidade de clipes a gerar
- `--clip-count`, `--count`, `-n`: quantidade explícita de clipes a gerar
- `--style`, `-s`: estilo da legenda, `word` ou `phrase`
- `--dry-run`: analisa os trechos sem exportar clipes
- `--title-overlay`: queima o título sugerido nos primeiros 5 segundos de cada clipe
- `--upload`: publica automaticamente os clipes ao final do pipeline
- `--platforms`: define as plataformas de upload (`youtube`, `instagram`, `tiktok` ou `all`)
- `--log-level`: nível de log, como `DEBUG`, `INFO`, `WARNING` ou `ERROR`
- `--log-file`: salva os logs em arquivo

---

## Cortes Inteligentes — `youcut cuts`

O comando `youcut cuts` é o ponto de entrada para gerar cortes otimizados por IA a partir de lives longas, com dois destinos possíveis:

| Modo | Formato | Destino | Duração dos clipes |
|---|---|---|---|
| **YouTube** | Paisagem 16:9 | Canal do YouTube | 15–25 min (definida pela IA) |
| **Redes sociais** | Vertical 9:16 | TikTok, Reels, Shorts | Até ~3 min (definida pela IA) |

O fluxo completo é interativo — basta rodar e responder às perguntas:

```bash
youcut cuts
```

Ou passando a URL direto:

```bash
youcut cuts "https://www.youtube.com/watch?v=exemplo"
```

### Fluxo A — Cortes longos para YouTube

1. Selecione o modo **YouTube** quando solicitado.
2. Confirme os metadados exibidos (título e duração do vídeo).
3. Defina o número máximo de clipes, ou deixe em branco para a IA decidir.
4. Acompanhe o progresso em tempo real: download → transcrição → análise → corte → thumbnails.
5. Revise e aprove os clipes (opcional — é possível publicar direto).
6. Publique no YouTube com `--upload`.
7. Ao final, uma oferta automática aparece para gerar vídeos curtos a partir dos cortes recém-gerados (Fluxo B).

```bash
# Modo YouTube com upload automático
youcut cuts "https://www.youtube.com/watch?v=exemplo" --upload --platforms youtube

# Limitar a 3 clipes e pular revisão
youcut cuts "https://www.youtube.com/watch?v=exemplo" --max-clips 3 --skip-review
```

> **Thumbnails:** para gerar thumbnails via OpenAI `gpt-image-1.5` no modo YouTube, configure `OPENAI_API_KEY` no `.env`.
>
> **Thumbnail com texto por padrão:** no modo YouTube, o pipeline usa por padrão o `thumbnail_text` sugerido pela análise de IA para gerar a thumbnail já com texto embutido.
>
> **Override opcional do texto:** use `--thumbnail-text "SEU TEXTO"` quando quiser sobrescrever esse texto padrão e forçar um texto específico, seguindo as regras de layout do projeto.
>
> **Custo otimizado para cortes sociais:** no modo `social` (Shorts/Reels/TikTok), o pipeline solicita a imagem em `1024x1024` (em vez de `1536x1024`), reduzindo o custo da geração em ~30% sem perda perceptível em mobile. O modo `youtube` e o pipeline `youcut comic` mantêm a qualidade atual sem alteração.

### Fluxo B — Vídeos curtos a partir de cortes existentes

Ao fim do Fluxo A, o YouCut exibe um card de oferta para gerar vídeos curtos para redes sociais **sem reprocessar o vídeo original** — a transcrição e a análise de NLP já realizadas são reutilizadas.

- **Seleção manual:** escolha quais cortes usar como fonte.
- **Timeout automático:** se nenhuma seleção for feita em **7 minutos** (padrão), todos os cortes da sessão são processados automaticamente.

O Fluxo B também pode ser iniciado a qualquer momento pelo histórico de sessões:

```bash
youcut cuts --history
```

Isso lista todas as sessões anteriores e permite selecionar uma para gerar vídeos curtos a partir dos cortes já existentes, sem baixar nem transcrever novamente.

### Fluxo C — Vídeos curtos direto do vídeo original

Selecione o modo **Redes sociais** para gerar clipes verticais (9:16) diretamente da URL, sem sessão prévia:

```bash
youcut cuts "https://www.youtube.com/watch?v=exemplo" --skip-review --upload --platforms all
```

### Opções do comando `cuts`

| Opção | Descrição |
|---|---|
| `SOURCE` | URL do YouTube (argumento posicional, opcional — será solicitado se omitido) |
| `--history`, `-H` | Lista sessões anteriores e permite iniciar o Fluxo B |
| `--max-clips`, `-n` | Número máximo de clipes a gerar (padrão: IA decide) |
| `--skip-review` | Pula a revisão interativa e vai direto para publicação |
| `--upload` | Faz upload dos clipes aprovados ao final do pipeline |
| `--thumbnail-text` | Texto opcional para sobrescrever o texto padrão sugerido pela IA na thumbnail |
| `--platforms` | Plataformas de upload: `youtube`, `instagram`, `tiktok` ou `all` |
| `--log-level` | Nível de log: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | Caminho para salvar o arquivo de log |

### Sessões salvas

Toda execução do Fluxo A salva uma sessão em `~/.youcut/sessions/`. Cada sessão contém:

- URL do vídeo original
- Modo de corte usado
- Lista de clipes gerados (com caminhos e metadados)
- Caminho do cache de transcrição (reutilizado no Fluxo B)

```bash
# Ver sessões disponíveis e iniciar Fluxo B em qualquer uma
youcut cuts --history
```

---

## Motion Comic — `youcut comic`

Transforma um vídeo curto (≤120s) em uma animação cartoon 9:16 (1080×1920) com personagens ilustrados, **lip-sync por palavra**, áudio original preservado e legendas opcionais.

```bash
youcut comic ./meu_video.mp4
```

Saída em `output/<nome_do_video>/`:
- `motion_comic_scenes.mp4` — versão **com legendas** word-by-word
- `motion_comic_scenes_no_subs.mp4` — versão **sem legendas** (publicação em redes que prefiram caption nativa)
- `comic/_scenes/master_*.png` — masters de cada cena (debug)
- `comic/_scenes/_visual_anchor.png` — referência canônica de estilo

### Engines disponíveis (`--engine`)

| Engine | Quando usar | Custo aprox | Tempo aprox |
|---|---|---|---|
| **`scenes`** (default, recomendado) | Diálogos, comédia, narrativa multi-cena. Lip-sync correto via Claude vision word-level. | ~$0.50–1.00 | ~10–15 min |
| `prunaai` | Monólogo simples ou cena única, sem narrativa. | ~$0.10 | ~5 min |
| `panels` | Máximo controle por beat, vídeos longos com muitas trocas. | ~$2.00 | ~20–30 min |
| `remotion` | Custo ≤ $1, **lip-sync sílaba-level**, render local determinístico, modo preview interativo. **Exige Node.js ≥ 20.** | ≤ $1.00 | ~5–10 min |

```bash
# Modo interativo (abre Remotion Studio para preview antes do render)
youcut comic ./meu_video.mp4 --engine remotion

# Modo headless (CI / batch — também acionado por --yes/-y)
youcut comic ./meu_video.mp4 --engine remotion --no-preview

# Dry-run (estima custo e termina)
youcut comic ./meu_video.mp4 --engine remotion --dry-run
```

### Engine `scenes` — como funciona

1. **Scene planner** (Claude texto): divide a transcrição em N cenas narrativas (default 4).
2. **Visual anchor + masters por cena** (gpt-image-1): gera 1 imagem-âncora canônica + 1 master por cena, todas referenciando o anchor pra consistência de estilo, paleta e design dos personagens.
3. **Word-level visual attribution** (Claude vision): extrai 1 frame por palavra (240px) e identifica qual personagem articula a boca em cada palavra.
4. **Smoothing conservador**: corrige mis-attributions onde o texto do chunk é repetição clara do vizinho (ex.: "é viadão" entre dois chunks Eva), preservando interjeições legítimas (ex.: "É mesmo, é?" da Eva entre falas da cobra).
5. **Smart-cuts**: chunks < 1.05s (mínimo do Prunaai) são estendidos absorvendo silêncio adjacente.
6. **Gap absorption**: gaps > 0.5s entre chunks (laughs, "harrá", expressões) são absorvidos pelo chunk anterior — o Prunaai anima essas reações em vez de virar freeze-frame.
7. **Render** (Prunaai per chunk): cada chunk recebe (master da cena pai + áudio do range + prompt direcionado pro speaker correto + outros em silêncio total).
8. **Concat com crossfades** (~0.25s): transições suaves entre chunks, preserva duração via tpad.
9. **Mux + scale + crop + watermark**: scale+crop pra 1080×1920 (sem letterbox), watermark `@username` opcional na safe zone.

### Configurações (`PipelineConfig`)

```env
# .env
COMIC_ANIMATION_ENGINE=scenes               # scenes (default) | prunaai | panels
COMIC_SCENES_COUNT=4                        # nº de cenas narrativas
COMIC_SCENES_CROSSFADE_DUR=0.25             # crossfade entre chunks (s)
COMIC_SCENES_GAP_ABSORB_THRESHOLD=0.5       # gaps > X são absorvidos
COMIC_SCENES_SMOOTH_ATTRIBUTION=True        # corrige mis-attributions cercadas
COMIC_SCENES_INTER_CALL_PAUSE_S=11.0        # pausa entre chamadas Prunaai (rate-limit)
COMIC_SCENES_WATERMARK_TEXT=@anima.nos      # watermark — null/vazio desliga
COMIC_SCENES_WATERMARK_OPACITY=0.40
COMIC_SCENES_WATERMARK_Y_FROM_BOTTOM=280    # px do fundo (safe zone)
COMIC_SCENES_EMIT_NO_SUBS_VERSION=True      # gera versão sem legendas
COMIC_SCENES_STYLE_REF_IMAGE=               # path opcional de imagem de estilo canônico
```

### Variáveis de ambiente obrigatórias

- `OPENAI_API_KEY` — gpt-image-1 (anchors + masters)
- `REPLICATE_API_TOKEN` — Prunaai (`prunaai/p-video-avatar`) e Hailuo i2v
- `ANTHROPIC_API_KEY` — Claude (scene planner + word-level attribution)

> **Atenção:** contas Replicate com saldo < $5 têm rate-limit reduzido (6/min com burst=1). O default `COMIC_SCENES_INTER_CALL_PAUSE_S=11s` respeita esse limite.

### Flags do CLI

```bash
youcut comic <video> \
  --engine scenes \                  # ou prunaai/panels
  --max-panels 4 \                   # nº de cenas (engine scenes) ou painéis (panels)
  --cost-cap 2.0 \                   # teto de custo USD
  --session <id> \                   # retoma sessão anterior (reusa cast/anchors)
  --regenerate-panel 2,5 \           # regen específico (modo panels)
  --invent-cast \                    # inventa cast a partir do áudio (sem rosto real)
  --multi-participant \              # exige ≥2 personagens (modo panels)
  --scene "descrição do cenário" \   # cenário fixo
  --no-metadata \                    # pula geração de metadados editoriais
  --yes -y \                         # auto-aprova cast e custo
  --no-progress                      # sem stages no console
```

### Cache e retomada

Toda execução cacheia:
- transcrição (MD5 do vídeo)
- atribuição visual word-level
- plano de cenas (`scenes.json`)
- visual anchor + masters
- raw chunks renderizados
- pre-rendered intermediates (extended chunks, xfade steps)

Se você matar o processo no meio, basta rodar de novo — só o que faltar é regerado. O `reconcile_cache` detecta automaticamente chunks com áudio fora de sync (ex.: depois de smoothing/gap-absorb mudar timestamps) e os apaga pra regerar.

---

## Upload Automático

Use `--upload` para publicar os clipes gerados sem intervenção manual ao final do pipeline. O upload reutiliza os metadados exportados em cada `clip_N.txt` e suporta seleção de plataformas com `--platforms` e seleção de clipes com `--clips`.

No caso do YouTube, o upload do vídeo e o envio da thumbnail são operações separadas. O CLI agora distingue:

- sucesso completo: vídeo publicado com thumbnail aplicada;
- publicação parcial: vídeo publicado, mas a thumbnail não foi aplicada e o CLI mostra um alerta acionável com a URL do vídeo.

Flags principais:

- `--upload`: ativa a etapa de publicação após a geração dos clipes
- `--platforms`: aceita `youtube`, `instagram`, `tiktok` ou `all`
- `--clips`: com `--upload`, aceita `all` ou uma lista de índices como `1,3`

Exemplos:

```bash
youcut run <url> --upload --platforms all
youcut run <url> --upload --platforms youtube --clips 1,3
```

### Autenticação

Autentique cada plataforma antes do primeiro upload, ou deixe o fluxo pedir login quando necessário:

```bash
youcut auth login --platform youtube
youcut auth revoke --platform instagram
youcut auth login --platform tiktok
youcut auth status
```

#### TikTok

Para autenticar no TikTok, configure primeiro a variável abaixo no `.env`:

```env
TIKTOK_CLIENT_KEY=seu_client_key
TIKTOK_CLIENT_SECRET=seu_client_secret
TIKTOK_POST_MODE=draft
```

Depois execute:

```bash
youcut auth login --platform tiktok
```

O fluxo do TikTok funciona via OAuth PKCE:

1. crie um app no portal de developers do TikTok e obtenha o `client_key`;
2. configure `TIKTOK_CLIENT_KEY` e `TIKTOK_CLIENT_SECRET` no `.env`;
3. rode `youcut auth login --platform tiktok`;
4. o navegador será aberto para o login e autorização do app;
5. ao concluir, o token será salvo em `~/.youcut/credentials/tiktok.json`.

Comandos úteis:

```bash
youcut auth revoke --platform tiktok
youcut auth status
```

Observação importante:

- apps do TikTok não auditados publicam vídeos como `private`; para postagem pública é necessário que o app passe pela auditoria do TikTok.
- deixar a conta do TikTok como privada não troca automaticamente o fluxo de upload para postagem direta. Se `TIKTOK_POST_MODE` continuar como `draft`, o YouCut seguirá usando a inbox do TikTok.
- para publicar direto pela API em vez de enviar para rascunho, use `TIKTOK_POST_MODE=direct` e refaça `youcut auth login --platform tiktok` para obter o escopo `video.publish`.
- no modo `direct`, o YouCut envia `privacy_level` ao TikTok. O padrão seguro é `SELF_ONLY`, mas você pode sobrescrever com `TIKTOK_PRIVACY_LEVEL=PUBLIC_TO_EVERYONE`, `MUTUAL_FOLLOW_FRIENDS`, `FOLLOWER_OF_CREATOR` ou `SELF_ONLY`, desde que essa opção esteja liberada para a conta conectada.
- opcionalmente, você pode controlar interações com `TIKTOK_DISABLE_COMMENT`, `TIKTOK_DISABLE_DUET` e `TIKTOK_DISABLE_STITCH` (`true` ou `false`).

#### YouTube

O uploader do YouTube valida localmente a thumbnail antes de chamar `thumbnails.set`:

- o arquivo precisa existir;
- as extensões aceitas são `.png`, `.jpg` e `.jpeg`;
- o tamanho máximo aceito é 2 MB.

Se a validação local falhar, o vídeo continua sendo publicado e o resultado aparece como publicação parcial com alerta. O mesmo vale quando o YouTube aceita o vídeo, mas rejeita a thumbnail por motivos como permissão insuficiente, política, quota ou imagem inválida.

A elegibilidade final para thumbnail customizada continua dependendo da conta conectada e das regras do próprio YouTube. Quando isso acontecer, o caminho esperado é usar a URL exibida pelo CLI e concluir o ajuste manualmente no YouTube Studio.

## Saída gerada

Por padrão, os arquivos ficam em `output/`.

Estrutura esperada:

```text
output/
  downloads/
    video-baixado.mp4
    video-baixado_transcript.json
  nome-do-video/
    clip_01.mp4
    clip_01.txt
    clip_02.mp4
    clip_02.txt
```

Os arquivos `.txt` incluem:

- título sugerido;
- descrição;
- hashtags;
- ideia de thumbnail;
- nota de viralidade;
- motivo da seleção.

## Observações importantes

- A aplicação falha na inicialização se `ANTHROPIC_API_KEY` não estiver configurada.
- O `ffmpeg` é obrigatório tanto para cortar os clipes quanto para embutir legendas.
- A transcrição é cacheada em um arquivo `*_transcript.json` para evitar reprocessamento do mesmo vídeo.
- O analisador pede ao modelo clipes entre `15` e `60` segundos.
- O estilo `word` mostra legenda palavra por palavra; `phrase` usa blocos por segmento.

## Executando testes

```bash
pytest
```

## Estrutura do projeto

```text
youcut/
  cli.py
  downloader.py
  transcriber.py
  analyzer.py
  clipper.py
  captioner.py
  title_overlay.py
  exporter.py
  config.py
  models.py
  video_metadata.py
  session_store.py
  thumbnail_generator.py
  reviewer.py
  assets/
tests/
~/.youcut/
  credentials/   ← tokens de autenticação das plataformas
  sessions/      ← sessões do youcut cuts (JSON)
```

## Resumo da pipeline

### Comando `run` (clipes virais, legado)

- `downloader.py`: resolve arquivo local ou baixa do YouTube com `yt-dlp`
- `transcriber.py`: transcreve com `faster-whisper` e usa cache
- `analyzer.py`: usa Claude para escolher os melhores trechos
- `clipper.py`: corta e adapta o vídeo para formato vertical ou paisagem
- `captioner.py`: gera e embute legendas
- `title_overlay.py`: queima o título nos primeiros 5s do clipe (ativado com `--title-overlay`)
- `exporter.py`: salva os metadados de publicação

### Comando `cuts` (cortes inteligentes)

- `video_metadata.py`: extrai título e duração do vídeo sem download completo (via `yt-dlp`)
- `analyzer.py`: modo `youtube` (15–25 min, 16:9) ou `social` (até 3 min, 9:16) com prompts parametrizados
- `clipper.py`: stream copy sem re-encoding para modo `youtube`; filtro vertical para modo `social`
- `thumbnail_generator.py`: gera thumbnails via OpenAI `gpt-image-1.5` (modo `youtube` com `OPENAI_API_KEY`); modo `social` usa size reduzido (`1024x1024`) para economizar custo
- `reviewer.py`: revisão interativa via terminal com aprovação, edição de título e regeneração de thumbnail
- `session_store.py`: salva e carrega sessões em `~/.youcut/sessions/` para reaproveitamento no Fluxo B
