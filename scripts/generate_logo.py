"""Gera o logo 'Anima Nós' usando gpt-image-1 (mesma stack do pipeline comic)."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "logo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = """Logo 1:1 para canal de redes sociais chamado "Anima Nós" (com
acento agudo na letra "o" da palavra "Nós" — é português brasileiro, NÓS,
não "NOS"), focado em animação e personagens.

REFERÊNCIA DE ESTILO (replicar fielmente): mascote único frontal e
centralizado, cabeça grande circular ocupando a maior parte do quadro,
cabelo curto laranja-queimado/ocre, olhos redondos pretos com pequeno
brilho branco, sorriso simples e amistoso, sem corpo (só a cabeça e
ombros sugeridos). Traço hand-drawn com contorno em tinta marrom-escura
suave, leve textura aquarela e lápis. Visual minimalista, limpo,
poucos detalhes.

FUNDO OBRIGATÓRIO: o quadro inteiro 1024x1024 PREENCHIDO com cor
creme/bege tipo papel envelhecido (aproximadamente #f4ead7 / #ede0c4).
NÃO use fundo branco. NÃO use fundo transparente. O fundo creme deve
ser visível em todas as bordas e cantos do quadro, com leve textura
de papel/aquarela. Esse fundo creme é parte essencial da identidade.

Paleta: tons terrosos quentes — laranja-queimado no cabelo, ocre,
marrom chocolate no contorno, creme no rosto e fundo. NADA de vermelho
saturado, nada de neon, nada de gradiente moderno.

Composição: APENAS UM elemento secundário pequeno e discreto ao lado
do mascote — uma única estrelinha desenhada à mão, no canto superior
direito ou esquerdo. Nada além disso (sem balão de fala, sem claquete,
sem traços de movimento extras).

Texto "Anima Nós" em arco curvo abaixo do mascote, lettering bold
arredondado hand-drawn, cor marrom escuro coordenado com o contorno.
O ACENTO AGUDO em cima do "O" da palavra "Nós" é OBRIGATÓRIO, precisa
estar claramente visível e bem desenhado — escreva exatamente "ANIMA
NÓS" (não "ANIMA NOS"). É português, com acento.

Sem moldura externa, sem borda retangular interna. Logo deve funcionar
como foto de perfil de Instagram, YouTube e TikTok — legível em
miniatura redonda. Sem watermark, sem assinatura de artista, sem texto
extra além de "Anima Nós".
"""


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY ausente", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    print("[1/1] Gerando logo Anima Nos (1024x1024)...")
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=PROMPT,
        size="1024x1024",
        n=1,
    )

    b64 = resp.data[0].b64_json
    if not b64:
        print("ERRO: API retornou sem b64_json", file=sys.stderr)
        return 1

    out = OUT_DIR / "anima_nos_v3_final.png"
    out.write_bytes(base64.b64decode(b64))
    print(f"OK -> {out}")
    print(f"   tamanho: {out.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
