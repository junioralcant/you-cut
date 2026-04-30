"""Edita anima_nos_v1.png mantendo tudo idêntico, só adiciona acento em NÓS."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SOURCE = ROOT / "output" / "logo" / "anima_nos_v1.png"
OUT = ROOT / "output" / "logo" / "anima_nos_v1_acentuado.png"

PROMPT = """Mantenha esta ilustração EXATAMENTE como está — mesmo mascote,
mesmo traço, mesmas cores, mesmo fundo creme, mesma estrelinha, mesma
composição, mesmo lettering. NÃO redesenhe nada. NÃO mude o estilo. NÃO
mexa no rosto, no cabelo, nas cores, na estrela ou no fundo.

A ÚNICA alteração permitida é no texto inferior: trocar "ANIMA NOS" por
"ANIMA NÓS" — ou seja, desenhar um acento agudo (´) sobre a letra "O" da
palavra "NÓS". O acento deve ter o mesmo estilo hand-drawn marrom escuro
do resto do lettering, posicionado naturalmente em cima do O.

Resultado final: imagem visualmente idêntica à original, com o acento
adicionado no O de NÓS. Nada mais muda.
"""


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY ausente", file=sys.stderr)
        return 1
    if not SOURCE.exists():
        print(f"ERRO: arquivo fonte não existe: {SOURCE}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    print(f"[1/1] Editando {SOURCE.name} -> adicionando acento em NÓS...")
    with SOURCE.open("rb") as fh:
        resp = client.images.edit(
            model="gpt-image-1",
            image=fh,
            prompt=PROMPT,
            size="1024x1024",
            input_fidelity="high",
        )

    b64 = resp.data[0].b64_json
    if not b64:
        print("ERRO: API retornou sem b64_json", file=sys.stderr)
        return 1

    OUT.write_bytes(base64.b64decode(b64))
    print(f"OK -> {OUT}")
    print(f"   tamanho: {OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
