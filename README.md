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
  assets/
tests/
```

## Resumo da pipeline

- `downloader.py`: resolve arquivo local ou baixa do YouTube com `yt-dlp`
- `transcriber.py`: transcreve com `faster-whisper` e usa cache
- `analyzer.py`: usa Claude para escolher os melhores trechos
- `clipper.py`: corta e adapta o vídeo para formato vertical
- `captioner.py`: gera e embute legendas
- `title_overlay.py`: queima o título nos primeiros 5s do clipe (ativado com `--title-overlay`)
- `exporter.py`: salva os metadados de publicação
