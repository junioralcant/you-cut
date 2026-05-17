"""Fetch público de threads do Reddit via JSON endpoint.

Reddit bloqueia user-agents genéricos e o domínio diretamente via WebFetch.
Solução: httpx com UA descritivo (formato recomendado por
https://www.reddit.com/wiki/api).
"""

from __future__ import annotations

import re

import httpx

from youcut.reddit_story.models import RedditStorySource


_URL_RE = re.compile(
    r"^https?://(?:www\.|old\.)?reddit\.com/r/(?P<sub>[^/]+)/comments/(?P<id>[^/]+)"
)


class RedditFetchError(Exception):
    """Falha de rede / parsing ao buscar o thread."""


def _normalize_to_json_url(url: str) -> str:
    """Converte qualquer URL de thread do Reddit para o endpoint JSON."""
    m = _URL_RE.match(url)
    if not m:
        raise RedditFetchError(
            f"URL não parece ser uma thread do Reddit: {url!r}\n"
            "Esperado: https://www.reddit.com/r/<SUB>/comments/<ID>/..."
        )
    # endpoint .json simples (sem comentários extras)
    return f"https://www.reddit.com/r/{m['sub']}/comments/{m['id']}.json"


def extract_thread_id(url: str) -> str:
    """Retorna o ID base36 do thread (ex.: 'q426qi' de uma URL completa)."""
    m = _URL_RE.match(url)
    if not m:
        raise RedditFetchError(f"URL não parece ser thread do Reddit: {url!r}")
    return m["id"]


def fetch_reddit_thread(url: str, *, user_agent: str) -> RedditStorySource:
    json_url = _normalize_to_json_url(url)
    try:
        resp = httpx.get(
            json_url,
            headers={"User-Agent": user_agent},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RedditFetchError(
            f"Falha ao buscar {json_url}: {exc}\n"
            "Reddit bloqueia UAs genéricos. Confirme reddit_story_user_agent no .env."
        ) from exc

    try:
        post = data[0]["data"]["children"][0]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RedditFetchError(f"JSON do Reddit em formato inesperado: {exc}") from exc

    if not post.get("selftext"):
        raise RedditFetchError(
            "Thread não tem corpo de texto (selftext vazio). "
            "É um post de link/imagem — escolha um text post."
        )

    return RedditStorySource(
        url=url,
        title=post["title"],
        author=post.get("author", "[deleted]"),
        subreddit=post["subreddit"],
        ups=post.get("ups", 0),
        permalink=f"https://www.reddit.com{post['permalink']}",
        selftext=post["selftext"],
    )
