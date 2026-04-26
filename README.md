# YouCut

Aplicação de linha de comando para transformar vídeos longos em clipes curtos com potencial viral.

O fluxo atual da aplicação:

1. baixa um vídeo do YouTube ou usa um arquivo local;
2. transcreve o áudio com Whisper;
3. envia a transcrição para o Claude identificar os melhores trechos;
4. corta os clipes em formato vertical `1080x1920`;
5. adiciona legendas embutidas;
6. (opcional) queima o título nos primeiros 5 segundos com `--title-overlay`;
7. exporta metadados prontos para publicação.

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

# Necessário apenas para o modo YouTube (geração de thumbnails via DALL-E 3):
# OPENAI_API_KEY=sua_chave_openai

# Necessário apenas para publicar no YouTube via youcut cuts --upload:
# YOUTUBE_CLIENT_SECRETS_FILE=caminho/para/client_secrets.json
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

- `--clips`: sem `--upload`, mantém o comportamento legado de quantidade de clipes; com `--upload`, seleciona `all` ou índices como `1,3`
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
| **YouTube** | Paisagem 16:9 | Canal do YouTube | 5–20 min (definida pela IA) |
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

> **Thumbnails:** para gerar thumbnails via DALL-E 3 no modo YouTube, configure `OPENAI_API_KEY` no `.env`.

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

## Upload Automático

Use `--upload` para publicar os clipes gerados sem intervenção manual ao final do pipeline. O upload reutiliza os metadados exportados em cada `clip_N.txt` e suporta seleção de plataformas com `--platforms` e seleção de clipes com `--clips`.

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
- `analyzer.py`: modo `youtube` (5–20 min, 16:9) ou `social` (até 3 min, 9:16) com prompts parametrizados
- `clipper.py`: stream copy sem re-encoding para modo `youtube`; filtro vertical para modo `social`
- `thumbnail_generator.py`: gera thumbnails via DALL-E 3 (modo `youtube` com `OPENAI_API_KEY`)
- `reviewer.py`: revisão interativa via terminal com aprovação, edição de título e regeneração de thumbnail
- `session_store.py`: salva e carrega sessões em `~/.youcut/sessions/` para reaproveitamento no Fluxo B
