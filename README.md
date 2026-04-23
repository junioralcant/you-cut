# YouCut

Aplicação de linha de comando para transformar vídeos longos em clipes curtos com potencial viral.

O fluxo atual da aplicação:

1. baixa um vídeo do YouTube ou usa um arquivo local;
2. transcreve o áudio com Whisper;
3. envia a transcrição para o Claude identificar os melhores trechos;
4. corta os clipes em formato vertical `1080x1920`;
5. adiciona legendas embutidas;
6. exporta metadados prontos para publicação.

## Requisitos

- Python `3.11+`
- `ffmpeg` instalado e disponível no `PATH`
- chave da API da Anthropic

## Instalação

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

Usando legenda por frase:

```bash
youcut run "./meu-video.mp4" --style phrase
```

Executando apenas a análise, sem gerar vídeos:

```bash
youcut run "./meu-video.mp4" --dry-run
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

- `--clips`, `-n`: quantidade de clipes a gerar
- `--style`, `-s`: estilo da legenda, `word` ou `phrase`
- `--dry-run`: analisa os trechos sem exportar clipes
- `--log-level`: nível de log, como `DEBUG`, `INFO`, `WARNING` ou `ERROR`
- `--log-file`: salva os logs em arquivo

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
  exporter.py
  config.py
tests/
```

## Resumo da pipeline

- `downloader.py`: resolve arquivo local ou baixa do YouTube com `yt-dlp`
- `transcriber.py`: transcreve com `faster-whisper` e usa cache
- `analyzer.py`: usa Claude para escolher os melhores trechos
- `clipper.py`: corta e adapta o vídeo para formato vertical
- `captioner.py`: gera e embute legendas
- `exporter.py`: salva os metadados de publicação
